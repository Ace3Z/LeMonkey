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


def replace_portrait(
    src_frame: np.ndarray,
    mask: np.ndarray,
    new_photo: np.ndarray,
    dst_corners: np.ndarray,
    *,
    mtf_sigma: float = 0.8,
    erode_px: int = 1,
    ring_dilate_px: int = 11,
    apply_unsharp: bool = False,
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
    h, w = new_photo.shape[:2]

    # 1. Lanczos warp — INTER_LANCZOS4 is the gold-standard high-quality
    # downsampling kernel for photographic content (ImageMagick filters lit).
    src_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_corners, dst_corners.astype(np.float32))
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

    # 3. Reinhard color transfer (Reinhard et al. 2001) sampled from a
    # ~5-px outer ring of the mask — captures local WB / exposure / shadow
    # tint at the portrait's exact location, beats whole-image stats which
    # are dominated by the table colour.
    ring_dilated = cv2.dilate(mask, np.ones((ring_dilate_px, ring_dilate_px), np.uint8))
    ring = cv2.subtract(ring_dilated, mask)
    warped = reinhard_lab(warped, src_frame, sample_mask=ring)

    # 4. Pre-fill the destination's mask region with the local ring-mean
    # colour. This is critical: cv2.seamlessClone uses the destination's
    # boundary pixels (just inside the mask, after OpenCV's ~3-px internal
    # erosion per opencv#17450) as Dirichlet conditions for the Poisson
    # solve. If we leave dst untouched, the boundary lies inside the
    # ORIGINAL portrait region — anchoring Poisson to the original
    # portrait colour and reverting the inpaint. Replacing dst's mask
    # interior with the ring mean makes the boundary lie on a smooth
    # background gray, which is what we actually want.
    # Verified empirically — eval_3/aug/tests/test_replace_portrait.py
    # variant B in the 2026-05-10 A/B test (dist_to_ring=23 vs raw=213).
    ring_pix = src_frame[ring > 0]
    if len(ring_pix) >= 10:
        ring_mean_bgr = ring_pix.astype(np.float32).mean(0).astype(np.uint8)
        dst_for_clone = src_frame.copy()
        dst_for_clone[mask > 0] = ring_mean_bgr
    else:
        dst_for_clone = src_frame

    # 5. Optional 1-px erosion to drop one-pixel JPEG halo artefacts.
    # Kept small because seamlessClone already erodes ~3 px internally
    # (opencv#17450); previously we used 3 px which stacked badly.
    if erode_px > 0:
        mask_for_clone = cv2.erode(mask, np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8))
    else:
        mask_for_clone = mask

    # 6. Poisson NORMAL_CLONE — Pérez et al. SIGGRAPH 2003 Eq. 11
    # ("importing"): copies the *source* gradients, lets boundary
    # continuity absorb DC colour drift. NORMAL because we want the new
    # photo's content fully (MIXED_CLONE would let dst gradients bleed
    # through, which on a portrait-into-portrait swap means the OLD face
    # comes back).
    ys, xs = np.where(mask_for_clone > 0)
    if len(ys) == 0:
        return src_frame                                 # mask vanished, bail
    center = (int(xs.mean()), int(ys.mean()))
    out = cv2.seamlessClone(
        warped, dst_for_clone, (mask_for_clone * 255).astype(np.uint8),
        center, cv2.NORMAL_CLONE,
    )
    return out


# ─── ffmpeg encode (NVENC if available, libx264 fallback) ───────────────────
def encode_video(frames_dir: Path, out_mp4: Path, fps: int) -> None:
    """Encode a directory of zero-padded PNG frames into an mp4."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Try NVENC first
    nvenc_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "f%06d.png"),
        "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
        "-cq", "23", "-b:v", "0",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    libx264_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "f%06d.png"),
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    try:
        subprocess.run(nvenc_cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.run(libx264_cmd, check=True)


# ─── Layout decoding ────────────────────────────────────────────────────────
INITIAL_TO_KEY = {"S": "swift", "O": "obama", "L": "lecun"}


def decode_layout(layout: str) -> list[str]:
    """SOL → ['swift','obama','lecun'] (left/middle/right)."""
    if layout == "-" or len(layout) != 3:
        return []
    return [INITIAL_TO_KEY.get(c, c.lower()) for c in layout]


def assign_celebs_to_portraits(corners_data: dict, layout_celebs: list[str]) -> dict[str, str]:
    """Given 3 portraits with corners, sort by mean-x to identify L/M/R,
    then map each portrait_id → celeb key according to the layout list."""
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
def load_photo_bank(photo_bank_root: Path) -> dict[str, list[Path]]:
    bank: dict[str, list[Path]] = {}
    for d in photo_bank_root.iterdir():
        if d.is_dir() and not d.name.startswith("_"):
            pics = sorted(p for p in d.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith("__"))
            if pics:
                bank[d.name] = pics
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
def render_variant(
    src_video: Path,
    corners_data: dict,
    masks_pkl: Path,
    pid_photos: dict[str, np.ndarray],   # pid_str → (h,w,3) uint8 BGR
    *,
    out_video: Path,
    fps: int,
    work_dir: Path,
) -> int:
    """Render a single augmented variant. Returns frame count written."""
    with open(masks_pkl, "rb") as f:
        cache = pickle.load(f)
    masks_per_frame: dict[int, dict[int, dict]] = cache["masks"]

    cap = cv2.VideoCapture(str(src_video))
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
            # Reconstruct mask from RLE if available; else build from corners as a polygon
            if payload is not None:
                mask = mask_util.decode(payload["rle"]).astype(np.uint8)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
            else:
                # interpolated frame — synthesize a polygon mask from corners
                H, W = frame.shape[:2]
                mask = np.zeros((H, W), dtype=np.uint8)
                pts = np.asarray(rec["corners"], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1)
            corners = np.asarray(rec["corners"], dtype=np.float32)
            out = replace_portrait(out, mask, photo, corners)
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
            try:
                os.link(src_path, dst_path)
            except (OSError, FileExistsError):
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
    masks_pkl = ep_dir / "portrait_masks.pkl"
    ref_json = ep_dir / "reference.json"

    if not corners_json.is_file():
        return {"ep": ep_dir.name, "error": "portrait_corners.json missing — run 3_extract_corners.py"}
    if not masks_pkl.is_file():
        return {"ep": ep_dir.name, "error": "portrait_masks.pkl missing — run 2_segment_video.py"}
    if not ref_json.is_file():
        return {"ep": ep_dir.name, "error": "reference.json missing — episode wasn't recorded with our recorder"}

    corners_data = json.loads(corners_json.read_text())
    sidecar = json.loads(ref_json.read_text())
    layout = sidecar.get("layout", "-")
    target_celeb = sidecar["target_celeb"]
    layout_celebs = decode_layout(layout)
    if not layout_celebs:
        return {"ep": ep_dir.name, "error": f"layout '{layout}' is required and must be 3 letters from S/O/L"}
    pid_to_celeb = assign_celebs_to_portraits(corners_data, layout_celebs)
    if len(pid_to_celeb) != 3:
        return {"ep": ep_dir.name, "error": f"could not assign 3 portraits — got {pid_to_celeb}"}

    src_video = find_video(ep_dir)
    rng = random.Random(seed + hash(ep_dir.name) % 1_000_000)

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
                pid_photos[pid] = img

            n_written = render_variant(
                src_video, corners_data, masks_pkl, pid_photos,
                out_video=var_video, fps=fps, work_dir=work_dir,
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
