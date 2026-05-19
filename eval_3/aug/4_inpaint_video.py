#!/usr/bin/env python3
"""STAGE 4 — composite new celebrity photos into the recorded video.

For each episode, produces N augmented variants. Each variant replaces all
3 printed portraits with NEW photos of the same celebrities (drawn from
the photo bank built by 1_mine_celeb_photos.py), plus picks a separate
held-out reference photo for the TARGET celeb (used as image-as-prompt
at training time).

The "Recommended tier" composite recipe (per STRATEGY.md §3.4):
  Lanczos warp → Gaussian σ=0.8 (camera MTF) → Reinhard Lab transfer
  sampled from a 5-px outer ring → mask erosion → cv2.seamlessClone
  NORMAL_CLONE.

Usage:
    # one variant of one episode
    python 4_inpaint_video.py /path/to/episode_dir --variant 0

    # N variants of every episode under a root
    python 4_inpaint_video.py --root ~/LeMonkey/datasets/eval3_quick --num-variants 5

    # alternative photo bank
    python 4_inpaint_video.py /path/to/ep --photo-bank /custom/path

Output layout:
    <out-root>/<episode_name>__var<NN>/
        videos/observation.images.camera1/chunk-000/file-000.mp4   (augmented)
        data/                                                        (hard-linked from src)
        meta/                                                        (hard-linked from src)
        reference.json                                               (variant sidecar)
        augmentation.json                                            (per-portrait photo manifest)

The action / state parquet is byte-identical to the original — we never
touch it. Only the camera video changes.

See STRATEGY.md §3.4 for design rationale.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util

# Local import — _video_io routes AV1 mp4s through a one-time H.264 transcode
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_video_io", str(Path(__file__).resolve().parent / "_video_io.py"))
_vio = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_vio)
ensure_h264 = _vio.ensure_h264


# ─── Recipe primitives ──────────────────────────────────────────────────────
def reinhard_lab(
    src: np.ndarray, ref: np.ndarray, sample_mask: np.ndarray,
    *,
    std_mult_clamp: tuple[float, float] = (0.3, 2.0),
) -> np.ndarray:
    """Match src's mean/std in Lab to those of ref, sampled where sample_mask > 0.

    Reinhard et al. 2001 "Color Transfer between Images" (IEEE CG&A).

    `std_mult_clamp` defends against the pathological case where the sample
    region is near-uniform (e.g. deep uniform shadow, or unrealistic test
    data) — without a clamp, ref_std/src_std can approach 0 and squash all
    detail in src. Default [0.3, 2.0] preserves Reinhard's dynamic-range
    matching in realistic cases (where ring std ≈ 10-30 in 8-bit space)
    while preventing collapse when ring std is unusually low.
    """
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean = src_lab.mean((0, 1)); src_std = src_lab.std((0, 1)) + 1e-6
    ring_pix = ref_lab[sample_mask > 0]
    if len(ring_pix) < 10:
        # not enough samples; bail (already documented in STRATEGY.md as a
        # corner case worth surfacing — match behaviour of the canonical
        # Reinhard implementations that simply skip when the sample is empty)
        print(f"[WARN] reinhard_lab: only {len(ring_pix)} sample pixels; skipping color transfer", flush=True)
        return src
    ref_mean = ring_pix.mean(0); ref_std = ring_pix.std(0) + 1e-6
    multiplier = np.clip(ref_std / src_std, std_mult_clamp[0], std_mult_clamp[1])
    out_lab = (src_lab - src_mean) * multiplier + ref_mean
    return cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


_FACE_APP_CACHE: list = [None]
def _get_face_app():
    if _FACE_APP_CACHE[0] is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _FACE_APP_CACHE[0] = app
    return _FACE_APP_CACHE[0]


def face_centered_aspect_crop(photo: np.ndarray, target_aspect: float) -> np.ndarray:
    """Crop `photo` to `target_aspect` (= width/height), centered on the
    detected face if found else on the photo centre. This eliminates the
    horizontal/vertical squish that homography warping would otherwise apply
    when the source photo's native aspect differs from the destination
    paper's aspect — important for face content, where a 28-35% horizontal
    squish (measured on real samples) visibly elongates the head.

    Preserves more of the relevant area than warp-stretching because the
    crop is anchored to the face landmark rather than the geometric centre.
    """
    h, w = photo.shape[:2]
    cur_aspect = w / h
    if abs(cur_aspect - target_aspect) < 0.01:
        return photo
    face_app = _get_face_app()
    faces = face_app.get(photo)
    if faces:
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        fx = float((f.bbox[0] + f.bbox[2]) * 0.5)
        fy = float((f.bbox[1] + f.bbox[3]) * 0.5)
    else:
        fx, fy = w * 0.5, h * 0.5
        print(f"[WARN] face_crop_no_face: expected=face_detected_in_source, "
              f"got=no_detection_({w}x{h}), fallback=photo_center_crop", flush=True)
    if cur_aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        x0 = max(0, min(int(round(fx - new_w / 2)), w - new_w))
        return photo[:, x0:x0 + new_w]
    else:
        new_h = int(round(w / target_aspect))
        y0 = max(0, min(int(round(fy - new_h / 2)), h - new_h))
        return photo[y0:y0 + new_h, :]


def replace_portrait(
    src_frame: np.ndarray,
    mask: np.ndarray,
    new_photo: np.ndarray,
    dst_corners: np.ndarray,
    *,
    lum_ratio: np.ndarray | None = None,  # (H,W) float32 — per-pixel shadow multiplier
    mtf_sigma: float = 0.8,
    erode_px: int = 1,
    ring_dilate_px: int = 11,
    apply_unsharp: bool = False,
    apply_reinhard: bool = False,
    blend_mode: str = "alpha_feather",
    feather_sigma: float = 0.8,    # v9.1: was 2.0 — wider feather caused a visible
                                   # halo of the original photo to leak through at
                                   # occluder boundaries (where the visible-paper
                                   # mask transitions 1→0). σ=0.8 ≈ 1.5 px transition
                                   # gives a sharp but anti-aliased edge.
    auto_rotate_to_paper_aspect: bool = True,  # v9.4: rotate the source photo 90°
                                   # if its aspect (w/h) doesn't match the paper's
                                   # long-axis direction. Eliminates horizontal
                                   # stretching when a portrait photo lands on a
                                   # landscape paper rectangle (tradeoff: face ends
                                   # up sideways relative to the viewer).
) -> np.ndarray:
    """Replace the masked region in src_frame with new_photo via the
    Recommended-tier recipe. Returns a new frame; src_frame is not mutated.

    src_frame   : (H,W,3) uint8 BGR
    mask        : (H,W)   uint8 binary {0,1}
    new_photo   : (h,w,3) uint8 BGR  (high-res replacement)
    dst_corners : (4,2)   float32   TL,TR,BR,BL of the original portrait

    Defaults — see eval_3/aug/STRATEGY.md §3.4 + the validation pass dated
    2026-05-10 in eval_3/aug/VALIDATION.md:
      mtf_sigma=0.8     — empirical, in the 0.5-1.0 px range typical for
                          640×480 USB webcam PSFs (Mosleh CVPR 2015,
                          consumer-OLPF lit). EMPIRICAL ONLY — refine via
                          dbg/probe_mtf.py if you can characterise the cam.
      erode_px=1        — OpenCV's seamlessClone already erodes ~3 px
                          internally (issue opencv#17450); stacking another
                          3-px erosion pulls colour too far inward.
      ring_dilate_px=11 — 5-px outer ring sampling for Reinhard local WB.
                          EMPIRICAL ONLY — widen to 19 (9-px ring) if the
                          ring catches too few valid pixels on textured
                          backgrounds.
      apply_unsharp     — optional unsharp-mask pass after MTF blur to
                          recover edges. OFF by default per the canonical
                          recipe; recommended-but-flagged add per
                          VALIDATION.md §8a.
    """
    H, W = src_frame.shape[:2]

    # v9.4: auto-rotate the source photo if its aspect doesn't match the paper's.
    # dst_corners is ordered TL/TR/BR/BL (see _order_tl_tr_br_bl in stage 2).
    # The "top edge" of the paper is corner[0]→corner[1]; the "left edge" is
    # corner[0]→corner[3]. If top > left, the paper is landscape (long side
    # horizontal in viewer-frame). If the source photo's aspect disagrees,
    # rotate the photo 90° clockwise so the long axes align — eliminates
    # horizontal stretching at the cost of an upright→sideways face rotation.
    if auto_rotate_to_paper_aspect:
        top_edge = float(np.linalg.norm(np.array(dst_corners[1]) - np.array(dst_corners[0])))
        left_edge = float(np.linalg.norm(np.array(dst_corners[3]) - np.array(dst_corners[0])))
        paper_landscape = top_edge > left_edge
        ph, pw = new_photo.shape[:2]
        photo_landscape = pw > ph
        if paper_landscape != photo_landscape:
            new_photo = cv2.rotate(new_photo, cv2.ROTATE_90_CLOCKWISE)

    h, w = new_photo.shape[:2]

    # 0. Clip corners to image bounds. cv2.minAreaRect can return corners
    # slightly outside the image when the portrait is partially cropped at
    # the camera edge (we've seen up to ~15 px overshoot on real episodes).
    # cv2.seamlessClone then asserts on the bbox-vs-image ROI math. Clipping
    # here yields a slightly non-rectangular quadrilateral (the homography
    # gracefully handles non-rect quads — getPerspectiveTransform doesn't
    # require axis-alignment).
    dst_corners = dst_corners.astype(np.float32).copy()
    dst_corners[:, 0] = np.clip(dst_corners[:, 0], 0, W - 1)
    dst_corners[:, 1] = np.clip(dst_corners[:, 1], 0, H - 1)

    # 1. Lanczos warp — INTER_LANCZOS4 is the gold-standard high-quality
    # downsampling kernel for photographic content (ImageMagick filters lit).
    src_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped = cv2.warpPerspective(new_photo, M, (W, H), flags=cv2.INTER_LANCZOS4)

    # 2. MTF match — emulate the camera's intrinsic blur (lens + sensor +
    # demosaic) so the replacement doesn't look unnaturally crisp vs the
    # rest of the frame. σ=0.8 ∈ [0.5, 1.0] — the band of measured Gaussian
    # PSF equivalents for consumer webcams (Mosleh et al. CVPR 2015).
    if mtf_sigma > 0:
        warped = cv2.GaussianBlur(warped, (0, 0), sigmaX=mtf_sigma)

    # 2b. Optional unsharp mask — recover edges lost to Lanczos + MTF blur.
    # Cambridge-in-Colour / community-standard "after resize, sharpen".
    # USM σ=1.0, amount=0.5  →  out = 1.5*orig - 0.5*blurred.
    if apply_unsharp:
        usm_blur = cv2.GaussianBlur(warped, (0, 0), sigmaX=1.0)
        warped = cv2.addWeighted(warped, 1.5, usm_blur, -0.5, 0)

    # 2c. SHADOW MODULATION (v7) — modulate the warped new photo by the
    # per-pixel luminance ratio L_i / L_0 so that shadows cast by the gripper /
    # can / hand on the original paper appear naturally on the new photo.
    # Without this, shadow regions either reveal the original (binary alpha
    # excludes them) or paste the new photo at full bright intensity (binary
    # alpha includes them, the shadow is gone). The ratio is computed in stage
    # 4's render_variant against frame_0 (saved by stage 2), restricted to the
    # M_0 paper mask, and clamped to [0.4, 1.4] so the photo isn't pushed to
    # extreme values by ratio outliers.
    if lum_ratio is not None:
        warped = np.clip(warped.astype(np.float32) * lum_ratio[..., None],
                         0, 255).astype(np.uint8)

    # 3. (Optional) Reinhard color transfer (Reinhard et al. 2001) sampled
    # from a ~5-px outer ring of the mask. Disabled by default (apply_reinhard
    # = False) because empirically (Validation Pass 3 on ep01, 2026-05-10)
    # it over-corrects when the ring is dominated by table/wall white —
    # Reinhard pulls the photo's mean toward white and the std-clamp keeps
    # only a fraction of the photo's contrast → bleached output. The pre-fill
    # step (4 below) already provides local DC matching at the Poisson
    # boundary; Poisson's gradient-domain blend handles the interior. So
    # Reinhard is only needed when the ring has rich texture/colour
    # (atypical for our wrist-cam-on-white-table setup).
    ring_dilated = cv2.dilate(mask, np.ones((ring_dilate_px, ring_dilate_px), np.uint8))
    ring = cv2.subtract(ring_dilated, mask)
    if apply_reinhard:
        warped = reinhard_lab(warped, src_frame, sample_mask=ring)

    # 4. Optional 1-px erosion to drop one-pixel JPEG halo artefacts.
    if erode_px > 0:
        mask_for_blend = cv2.erode(mask, np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8))
    else:
        mask_for_blend = mask
    if mask_for_blend.sum() == 0:
        # Erosion vanished the mask. Skip composite for this portrait this
        # frame — but log so flickering artifacts don't go unnoticed (CLAUDE.md §5).
        print(
            f"[WARN] mask_eroded_to_zero: expected=mask_for_blend.sum()>0, "
            f"got=0 (raw_mask_sum={int(mask.sum())}, erode_px={erode_px}), "
            f"fallback=skip_composite_return_src_frame",
            flush=True,
        )
        return src_frame                                 # mask vanished

    # 5. Blend.
    #
    # blend_mode = "alpha_feather" (DEFAULT): Gaussian-feathered alpha paste.
    #   Inside the mask: full new-photo colour. At the boundary: smooth
    #   transition from new photo to surrounding frame.
    #   This is the right choice for "replace one printed photo with another
    #   printed photo at the same pose under the same lighting" because:
    #     a) both photos have similar dynamic range / mean colour
    #     b) the boundary (cardstock edge on table) is a sharp visible line
    #        anyway in the original — feathering matches that look
    #     c) preserves the new photo's identity (ArcFace cosine ≥ 0.4) which
    #        Poisson NORMAL_CLONE bleaches when the ring is dominated by
    #        white table (verified empirically on ep01 frame 0, 2026-05-10:
    #        Poisson output mean = (198, 201, 209), warped photo mean
    #        = (105, 139, 166); ArcFace cosine for Poisson output likely
    #        below threshold).
    #
    # blend_mode = "poisson_normal": cv2.seamlessClone(NORMAL_CLONE) on a
    #   pre-filled destination (mask region replaced with ring-mean before
    #   cloning, per Validation Pass 2 — without the pre-fill, OpenCV's
    #   internal 3-px erosion anchors Poisson to the *original* portrait
    #   colour and reverts the inpaint). Use when the new photo's lighting
    #   genuinely differs from the local environment and you want
    #   gradient-domain DC absorption — but expect bleached output if the
    #   ring is uniformly bright/dark.
    if blend_mode == "alpha_feather":
        # Asymmetric feather: blur the mask for a soft inward transition,
        # then CLAMP to the original mask so the alpha is exactly zero
        # outside it. Without the clamp, the Gaussian extends ~σ × 3 px
        # past the mask boundary and bleeds the new photo onto adjacent
        # pixels — which on our setup means the gripper/can/hand get
        # partly overwritten. Per the user's correct observation: SAM 2's
        # mask already excludes those occluders by construction (SAM
        # tracks the paper *underneath*), so the right behaviour is to
        # respect the mask exactly at its boundary.
        feather = cv2.GaussianBlur(mask_for_blend.astype(np.float32), (0, 0), sigmaX=feather_sigma)
        # Clamp to the (un-eroded) mask: alpha is at most where SAM said paper is.
        feather = np.minimum(feather, mask.astype(np.float32))
        m = feather.max()
        if m > 1e-6:
            feather = feather / m
        feather = feather[:, :, None]                           # H,W,1
        out = warped.astype(np.float32) * feather + src_frame.astype(np.float32) * (1 - feather)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    if blend_mode == "poisson_normal":
        # pre-fill dst's mask region with ring-mean (see VALIDATION.md §8e).
        ring_pix = src_frame[ring > 0]
        if len(ring_pix) >= 10:
            ring_mean_bgr = ring_pix.astype(np.float32).mean(0).astype(np.uint8)
            dst_for_clone = src_frame.copy()
            dst_for_clone[mask > 0] = ring_mean_bgr
        else:
            dst_for_clone = src_frame

        ys, xs = np.where(mask_for_blend > 0)
        center = (int(xs.mean()), int(ys.mean()))
        # Safety: clamp center so seamlessClone's bbox+center math fits.
        bbox_w = xs.max() - xs.min() + 1
        bbox_h = ys.max() - ys.min() + 1
        cx = max(bbox_w // 2 + 1, min(W - bbox_w // 2 - 1, center[0]))
        cy = max(bbox_h // 2 + 1, min(H - bbox_h // 2 - 1, center[1]))
        safe_center = (int(cx), int(cy))
        try:
            out = cv2.seamlessClone(
                warped, dst_for_clone, (mask_for_blend * 255).astype(np.uint8),
                safe_center, cv2.NORMAL_CLONE,
            )
            return out
        except cv2.error as e:
            print(f"[WARN] seamlessClone failed at center={safe_center}, "
                  f"bbox={bbox_w}×{bbox_h}: {e}; falling back to alpha_feather", flush=True)
            # fall through to alpha_feather
            feather = cv2.GaussianBlur(mask_for_blend.astype(np.float32), (0, 0), sigmaX=feather_sigma)
            feather = (feather / max(feather.max(), 1e-6))[:, :, None]
            out = warped.astype(np.float32) * feather + src_frame.astype(np.float32) * (1 - feather)
            return np.clip(out, 0, 255).astype(np.uint8)

    raise ValueError(f"unknown blend_mode {blend_mode!r}; choose 'alpha_feather' or 'poisson_normal'")


# ─── ffmpeg encode (NVENC if available, libx264 fallback) ───────────────────
def encode_video(frames_dir: Path, out_mp4: Path, fps: int) -> None:
    """Encode a directory of zero-padded PNG frames into an mp4."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Try NVENC first
    nvenc_cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "f%06d.png"),
        "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
        "-cq", "23", "-b:v", "0",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    libx264_cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "f%06d.png"),
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    try:
        subprocess.run(nvenc_cmd, check=True, stdin=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.run(libx264_cmd, check=True, stdin=subprocess.DEVNULL)


# ─── Layout decoding ────────────────────────────────────────────────────────
INITIAL_TO_KEY = {"S": "swift", "O": "obama", "L": "lecun"}


def decode_layout(layout: str) -> list[str]:
    """SOL → ['swift','obama','lecun'] (left/middle/right)."""
    if layout == "-" or len(layout) != 3:
        return []
    return [INITIAL_TO_KEY.get(c, c.lower()) for c in layout]


def assign_celebs_to_portraits(
    corners_data: dict,
    layout_celebs: list[str],
    seeds: dict | None = None,
) -> dict[str, str]:
    """Map each portrait_id (the SAM 2 obj_id from frame 0 click order) to a
    celeb key.

    Two strategies:
      1. If `seeds` carries an explicit "celebs" list (saved by the
         interactive clicker when it prompted celeb-by-celeb), trust it
         directly: portrait_id i → seeds["celebs"][i].
      2. Otherwise fall back to mean-x sort (assumes a horizontal layout
         where leftmost portrait corresponds to layout_celebs[0]). This is
         only correct for true left-to-right arrangements; semicircle /
         triangle arrangements need strategy 1.
    """
    if seeds and "celebs" in seeds and len(seeds.get("celebs") or []) == 3:
        celebs = seeds["celebs"]
        return {str(i): celebs[i] for i in range(3)}

    portrait_means: dict[str, float] = {}
    for pid_str, frames in corners_data["portraits"].items():
        # Use the first non-occluded frame's centre x for a stable assignment
        for fi_str, rec in frames.items():
            if rec["corners"] is not None:
                xs = [c[0] for c in rec["corners"]]
                portrait_means[pid_str] = float(np.mean(xs))
                break
        else:
            portrait_means[pid_str] = 0.0

    # Sort portraits by x-coordinate: leftmost gets layout_celebs[0], etc.
    sorted_pids = sorted(portrait_means.keys(), key=lambda k: portrait_means[k])
    return {pid: layout_celebs[idx] for idx, pid in enumerate(sorted_pids) if idx < len(layout_celebs)}


# ─── Photo bank loading ─────────────────────────────────────────────────────
def _is_color_photo(path: Path, min_mean_sat: float = 60.0) -> bool:
    """True when `path` is a colour photo (not B&W / sepia / muted).
    Uses mean saturation in HSV; B&W has near-zero saturation everywhere,
    sepia 10-25, muted 25-55, clearly colour 60+. Threshold empirically
    separates the categories in the eval3 photo bank (see saturation probe
    2026-05-13)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean()) >= min_mean_sat


def load_photo_bank(
    photo_bank_root: Path, *, portrait_only: bool = True, color_only: bool = True,
) -> dict[str, list[Path]]:
    """Load celeb→[photo paths] mapping with two filters:
    - portrait_only=True: drop photos whose width >= height (landscape stretches
      faces horizontally when pasted into typically-portrait paper regions).
    - color_only=True: drop B&W / sepia photos (jarring style mismatch when
      pasted next to color printed portraits — see eval3 dbg_compare F2)."""
    from PIL import Image
    bank: dict[str, list[Path]] = {}
    dropped_portrait: dict[str, int] = {}
    dropped_color: dict[str, int] = {}
    for d in photo_bank_root.iterdir():
        if not (d.is_dir() and not d.name.startswith("_")):
            continue
        kept = []
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"} or p.name.startswith("__"):
                continue
            if portrait_only:
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                except Exception:
                    continue
                if w >= h:                              # landscape OR square → drop
                    dropped_portrait[d.name] = dropped_portrait.get(d.name, 0) + 1
                    continue
            if color_only and not _is_color_photo(p):
                dropped_color[d.name] = dropped_color.get(d.name, 0) + 1
                continue
            kept.append(p)
        if kept:
            bank[d.name] = kept
    if portrait_only and dropped_portrait:
        print(f"  portrait-only filter dropped: {dropped_portrait}", flush=True)
    if color_only and dropped_color:
        print(f"  color-only filter dropped:    {dropped_color}", flush=True)
    return bank


def pick_photos_for_variant(
    pid_to_celeb: dict[str, str],
    bank: dict[str, list[Path]],
    target_celeb: str,
    *,
    rng: random.Random,
) -> tuple[dict[str, Path], Path | None]:
    """Pick 3 distinct workspace replacements + 1 reference photo for the target,
    all of the appropriate celebs, all distinct from each other where applicable.

    Returns (pid → workspace photo path,  reference photo path or None).
    """
    workspace: dict[str, Path] = {}
    used: set[Path] = set()
    for pid, celeb in pid_to_celeb.items():
        pool = bank.get(celeb, [])
        if not pool:
            raise ValueError(f"photo bank has no photos for celeb '{celeb}' (looked in {bank.keys()})")
        choice = rng.choice([p for p in pool if p not in used] or pool)
        workspace[pid] = choice
        used.add(choice)

    # Reference photo: must be different from the workspace photo for that celeb
    target_pool = bank.get(target_celeb, [])
    target_workspace = next((p for pid, p in workspace.items() if pid_to_celeb[pid] == target_celeb), None)
    if target_pool:
        candidates = [p for p in target_pool if p != target_workspace]
        ref = rng.choice(candidates) if candidates else None
    else:
        ref = None
    return workspace, ref


# ─── Variant render ─────────────────────────────────────────────────────────
def _mean_lum_ratio_v9(
    frame_i: np.ndarray, frame_0: np.ndarray, visible_paper: np.ndarray,
) -> float:
    """v9: ONE SCALAR mean luminance ratio over the visible-paper region.

    Replaces v7/v8 per-pixel lum_ratio. The research consensus (LIBERO-Plus
    2025, Pumacay et al., LeRobot/OpenVLA defaults, ROSIE, GenAug, CACTI)
    is that per-pixel shadow modeling is the smallest-impact perturbation
    axis for VLAs (8-11pp impact vs 45-72pp for camera shift). A single
    scalar that crudely matches the inserted photo's brightness to the
    current frame's average is sufficient — ColorJitter at train time
    handles the rest. Clamped to [0.6, 1.2] so a near-fully-occluded
    visible_paper (only a few pixels left, possibly all in shadow) can't
    push the photo to extreme values."""
    if visible_paper.sum() < 50:
        return 1.0
    lab_i = cv2.cvtColor(frame_i, cv2.COLOR_BGR2LAB)
    lab_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2LAB)
    L_i = lab_i[..., 0].astype(np.float32)
    L_0 = lab_0[..., 0].astype(np.float32)
    mask = visible_paper > 0
    L_i_mean = float(L_i[mask].mean())
    L_0_mean = float(L_0[mask].mean())
    ratio = L_i_mean / max(L_0_mean, 5.0)
    return float(np.clip(ratio, 0.6, 1.2))


def render_variant(
    src_video: Path,
    corners_data: dict,
    masks_pkl: Path | None,
    pid_photos: dict[str, np.ndarray],   # pid_str → (h,w,3) uint8 BGR
    *,
    out_video: Path,
    fps: int,
    work_dir: Path,
    frame_0: np.ndarray | None = None,            # v7: shadow-modulation reference
    M_0_per_pid: dict[int, np.ndarray] | None = None,  # v7: paper-rectangle mask per pid
) -> int:
    """Render a single augmented variant. Returns frame count written.

    v7: when `frame_0` and `M_0_per_pid` are provided (saved by stage 2 v7),
    compute a per-pixel luminance ratio L_i/L_0 within the M_0 rectangle and
    pass it to `replace_portrait` so shadows cast on the original paper
    appear on the augmented photo. Falls back gracefully to the v6 binary
    composite when v7 sidecars are absent.

    `masks_pkl` carries per-frame visible-paper masks (M_0 AND NOT object).
    `corners_data` carries the constant 4-corner rectangle per portrait.
    """
    masks_per_frame: dict[int, dict[int, dict]] = {}
    if masks_pkl is not None and masks_pkl.is_file():
        with open(masks_pkl, "rb") as f:
            cache = pickle.load(f)
        masks_per_frame = cache["masks"]
        if M_0_per_pid is None and "M_0_per_pid" in cache:
            M_0_per_pid = cache["M_0_per_pid"]

    cap = cv2.VideoCapture(str(ensure_h264(src_video)))
    n_frames = corners_data["n_frames"]
    work_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        out = frame
        for pid_str, photo in pid_photos.items():
            pid_int = int(pid_str)
            payload = masks_per_frame.get(fi, {}).get(pid_int)
            rec = corners_data["portraits"][pid_str].get(str(fi))
            if rec is None or rec["corners"] is None:
                continue
            if payload is not None:
                mask = mask_util.decode(payload["rle"]).astype(np.uint8)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
            else:
                # No stage-2 per-frame mask available. Fall back to the
                # corners-rectangle which means we LOSE per-frame occlusion
                # handling for this portrait this frame. Log so it surfaces
                # (CLAUDE.md §5) — usually means stage 2 wasn't re-run after
                # a code change.
                print(
                    f"[WARN] per_frame_mask_missing: expected=portrait_masks.pkl["
                    f"frame={fi}][pid={pid_int}], got=None, "
                    f"fallback=fillPoly_from_corners (per-frame occlusion lost)",
                    flush=True,
                )
                H, W = frame.shape[:2]
                mask = np.zeros((H, W), dtype=np.uint8)
                pts = np.asarray(rec["corners"], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1)
            corners = np.asarray(rec["corners"], dtype=np.float32)

            # v9 global brightness match: one scalar = mean(L_i/L_0) over the
            # currently-visible paper. Replaces v7's per-pixel lum_ratio. Per
            # LIBERO-Plus 2025 + Pumacay decomp, this is sufficient — the field
            # skips per-pixel shadow modeling for VLA augmentation.
            lum_ratio = None
            if frame_0 is not None:
                ratio = _mean_lum_ratio_v9(frame, frame_0, mask)
                lum_ratio = np.full(frame.shape[:2], ratio, dtype=np.float32)

            out = replace_portrait(out, mask, photo, corners, lum_ratio=lum_ratio)
        cv2.imwrite(str(work_dir / f"f{fi:06d}.png"), out)
        written += 1
    cap.release()

    encode_video(work_dir, out_video, fps=fps)
    # cleanup PNG frames immediately to bound disk
    for p in work_dir.glob("f*.png"):
        p.unlink()
    return written


# ─── Per-episode driver ─────────────────────────────────────────────────────
def find_video(ep_dir: Path) -> Path:
    cands = list(ep_dir.glob("videos/*/chunk-*/file-*.mp4"))
    if not cands:
        raise FileNotFoundError(f"no video under {ep_dir}/videos/")
    # Prefer the wrist camera (camera1, 640x480). The reference stream is
    # 480x480 and would mismatch the cached portrait masks. Path.glob order
    # is not guaranteed across filesystems (worked on dev box, picked
    # reference first on edna ext4), so don't rely on cands[0].
    cam1 = [p for p in cands if "observation.images.camera1" in str(p)]
    if cam1:
        return cam1[0]
    return cands[0]


def hardlink_meta(src_ep: Path, dst_ep: Path) -> None:
    """Hard-link parquet + meta from original to variant (action labels unchanged)."""
    for sub in ("data", "meta"):
        s = src_ep / sub
        if not s.is_dir():
            continue
        for src_path in s.rglob("*"):
            if not src_path.is_file():
                continue
            rel = src_path.relative_to(src_ep)
            dst_path = dst_ep / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if dst_path.exists():
                # Already linked from a prior --force run; hardlink → same
                # content, so leave it. (Avoids shutil.SameFileError.)
                continue
            try:
                os.link(src_path, dst_path)
            except OSError:
                shutil.copy2(src_path, dst_path)


def process_episode(
    ep_dir: Path,
    out_root: Path,
    bank: dict[str, list[Path]],
    *,
    num_variants: int,
    seed: int,
    fps: int,
    force: bool,
) -> dict:
    corners_json = ep_dir / "portrait_corners.json"
    masks_pkl: Path | None = ep_dir / "portrait_masks.pkl"
    if masks_pkl is not None and not masks_pkl.is_file():
        masks_pkl = None                                  # 2_detect_track.py path — corners-only
    ref_json = ep_dir / "reference.json"

    if not corners_json.is_file():
        return {"ep": ep_dir.name, "error": "portrait_corners.json missing — run 2_detect_track.py (or 2_segment_video.py + 3_extract_corners.py)"}
    if not ref_json.is_file():
        return {"ep": ep_dir.name, "error": "reference.json missing — episode wasn't recorded with our recorder"}

    corners_data = json.loads(corners_json.read_text())
    sidecar = json.loads(ref_json.read_text())
    layout = sidecar.get("layout", "-")
    target_celeb = sidecar["target_celeb"]
    layout_celebs = decode_layout(layout)
    if not layout_celebs:
        return {"ep": ep_dir.name, "error": f"layout '{layout}' is required and must be 3 letters from S/O/L"}

    # Prefer the explicit click-order → celeb mapping written by the
    # interactive clicker (handles semicircle / triangle arrangements).
    seeds_path = ep_dir / "portrait_seeds.json"
    seeds_dict = json.loads(seeds_path.read_text()) if seeds_path.is_file() else None
    pid_to_celeb = assign_celebs_to_portraits(corners_data, layout_celebs, seeds_dict)
    if len(pid_to_celeb) != 3:
        return {"ep": ep_dir.name, "error": f"could not assign 3 portraits — got {pid_to_celeb}"}

    src_video = find_video(ep_dir)
    rng = random.Random(seed + hash(ep_dir.name) % 1_000_000)

    # v7: load frame_0 for shadow-aware luminance modulation. Stage 2 v7
    # writes frame_0.png next to portrait_corners.json. Fall back to None
    # (= v6 binary composite behaviour) when the sidecar is absent.
    frame_0_path = ep_dir / "frame_0.png"
    frame_0_img: np.ndarray | None = None
    if frame_0_path.is_file():
        frame_0_img = cv2.imread(str(frame_0_path), cv2.IMREAD_COLOR)

    work_dir = Path("/tmp") / f"_aug_work_{ep_dir.name}_{os.getpid()}"
    work_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict] = []
    try:
        for var_idx in range(num_variants):
            var_name = f"{ep_dir.name}__var{var_idx:02d}"
            var_out = out_root / var_name
            var_video = var_out / "videos" / "observation.images.camera1" / "chunk-000" / "file-000.mp4"
            if var_video.is_file() and not force:
                rendered.append({"variant": var_idx, "skipped": True})
                continue

            workspace_photos, ref_photo = pick_photos_for_variant(
                pid_to_celeb, bank, target_celeb, rng=rng,
            )
            pid_photos: dict[str, np.ndarray] = {}
            for pid, p in workspace_photos.items():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"cannot read {p}")
                # v9.6: face-centered crop to paper aspect — eliminates the
                # warp's 20-35% face stretch when photo aspect ≠ paper aspect.
                pts = np.asarray(corners_data["portraits"][pid]["0"]["corners"], dtype=np.float32)
                paper_top = float(np.linalg.norm(pts[1] - pts[0]))
                paper_left = float(np.linalg.norm(pts[3] - pts[0]))
                target_aspect = paper_top / max(paper_left, 1e-6)
                img = face_centered_aspect_crop(img, target_aspect)
                pid_photos[pid] = img

            n_written = render_variant(
                src_video, corners_data, masks_pkl, pid_photos,
                out_video=var_video, fps=fps, work_dir=work_dir,
                frame_0=frame_0_img,
            )

            hardlink_meta(ep_dir, var_out)
            new_sidecar = {**sidecar}
            new_sidecar["source"] = "augmented"
            new_sidecar["augmented_from"] = ep_dir.name
            new_sidecar["variant_idx"] = var_idx
            new_sidecar["reference_photo"] = str(ref_photo) if ref_photo else sidecar.get("reference_photo")
            (var_out / "reference.json").write_text(json.dumps(new_sidecar, indent=2))
            (var_out / "augmentation.json").write_text(json.dumps({
                "src_episode": ep_dir.name,
                "variant_idx": var_idx,
                "pid_to_celeb": pid_to_celeb,
                "workspace_photos": {pid: str(p) for pid, p in workspace_photos.items()},
                "reference_photo": str(ref_photo) if ref_photo else None,
                "n_frames": n_written,
            }, indent=2))
            rendered.append({"variant": var_idx, "frames": n_written})
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {"ep": ep_dir.name, "rendered": rendered}


# ─── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("episode_dir", nargs="?", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--out-root", default="/home/lemonkey/LeMonkey/datasets/eval3_aug",
                   help="where augmented variants are written")
    p.add_argument("--photo-bank", default="/home/lemonkey/LeMonkey/datasets/eval3_celebs/web",
                   help="root of the verified photo bank from 1_mine_celeb_photos.py")
    p.add_argument("--num-variants", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if (args.episode_dir is None) == (args.root is None):
        print("[ERROR] specify exactly one of: episode_dir, --root", file=sys.stderr)
        return 2

    bank = load_photo_bank(Path(args.photo_bank))
    if not bank:
        print(f"[ERROR] no photos in bank at {args.photo_bank} — run 1_mine_celeb_photos.py first",
              file=sys.stderr)
        return 1
    print(f"photo bank: {sum(len(v) for v in bank.values())} photos across "
          f"{len(bank)} celebs ({list(bank.keys())})")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.episode_dir:
        ep_dirs = [Path(args.episode_dir)]
    else:
        root = Path(args.root)
        ep_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "reference.json").is_file())

    print(f"will render {args.num_variants} variants × {len(ep_dirs)} episodes = "
          f"{args.num_variants * len(ep_dirs)} videos\n")

    results: list[dict] = []
    for ep_dir in ep_dirs:
        t0 = time.time()
        try:
            r = process_episode(
                ep_dir, out_root, bank,
                num_variants=args.num_variants,
                seed=args.seed, fps=args.fps, force=args.force,
            )
        except Exception as e:
            r = {"ep": ep_dir.name, "error": f"{type(e).__name__}: {e}"}
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        if "rendered" in r:
            n_done = sum(1 for v in r["rendered"] if "frames" in v)
            print(f"  ✓ {r['ep']:50s}  {n_done}/{args.num_variants} variants  ({r['seconds']}s)")
        else:
            print(f"  ✗ {r['ep']:50s}  {r.get('error','?')}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
