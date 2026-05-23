#!/usr/bin/env python3
"""Cotrain augmentation — tuple-driven, 3-celeb-only, full-enumeration capable.

Goal: train the in-distribution SmolVLA cotrain (3-celeb) with maximum coverage
of the (target_celeb, target_photo, layout, distractor_photos) space, using
only Swift / Obama / LeCun. Re-uses the existing per-base-episode caches
(portrait_masks.pkl, portrait_corners.json) — these were computed once when
the v3 4500-variant dataset was generated.

DESIGN
======

Photo bank: 8 photos per IID celeb (5 heldout + 3 scraped for swift/lecun,
4 + 4 for obama). Built by build_cotrain_bank.py.

Combinatorial space:
    3 targets × 8 target_photos × 6 layouts × 8 D1_photos × 8 D2_photos
  = 3 × 8 × 6 × 64
  = 9216 unique (target, target_photo, layout, D1, D2) configs.

Base teleop coverage (178 eps in 9 of 18 (target, layout) cells):
    swift_SOL/OSL/OLS  obama_SOL/SLO/OSL  lecun_SOL/SLO/LSO
Each cell × 4 or 5 heldout photos = 42 (target, target_photo, layout) tuples,
each with one specific (D1, D2) photo config. So base covers ≤42 of the 9216
configurations directly.

Enumeration:
    For each (target, target_photo, layout) tuple:
        - Determine physical-target-slot S (= position of target in layout)
        - Pick a base teleop where the recorded target was at slot S
          (round-robin within the matching group for trajectory diversity).
        - For each of K distractor combos (default 64 = full enumeration):
            - Re-paint: target slot ← target_photo of target;
              other 2 slots ← chosen distractor photos per layout.
            - Reference video ← a DIFFERENT photo of target (face-verification
              setup per STRATEGY.md §3).

CACHING
=======

Per-base-episode cache (already on disk, reused freely):
    portrait_masks.pkl    SAM 2.1 masks per frame per pid
    portrait_corners.json 4-corner quads per frame per pid
    portrait_seeds.json   click prompts / pid→celeb mapping
    reference.json        (target_celeb, layout, ...)

This script does NOT re-compute any of those. It iterates the new tuples,
groups them by base teleop for video-decode locality, and writes one MP4
per variant.

USAGE
=====

    # Full enumeration (9216 aug variants, ~77h single-GPU)
    python generate_aug_cotrain.py \
        --base-root /home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3 \
        --bank-root /home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3_celebs/cotrain_bank \
        --out-root  /home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3_aug_cotrain

    # Smoke-test: 3 tuples × 2 variants each = 6 variants
    python generate_aug_cotrain.py \
        --base-root ... --bank-root ... --out-root /tmp/aug_smoke \
        --limit-tuples 3 --limit-variants-per-tuple 2 --debug

OUTPUT
======

Per variant:
    <out-root>/<base_ep>__t3_<tuple_idx:04d>_v<var_idx:02d>/
        videos/observation.images.camera1/chunk-000/file-000.mp4   (inpainted)
        videos/observation.images.reference/chunk-000/file-000.mp4 (const)
        data/   (hardlinked from base)
        meta/   (hardlinked from base)
        reference.json
        augmentation.json
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import random
import shutil
import sys
import time
import traceback
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np
import pickle
import pycocotools.mask as mask_util

# ─── Local imports — reuse helpers from 4_inpaint_video.py + generate_aug_broad.py
_HERE = Path(__file__).resolve().parent
_spec4 = _ilu.spec_from_file_location("_v4", str(_HERE.parent / "stages" / "inpaint_video.py"))
_v4 = _ilu.module_from_spec(_spec4); _spec4.loader.exec_module(_v4)
_specv3 = _ilu.spec_from_file_location("_v3", str(_HERE / "broad.py"))
_v3 = _ilu.module_from_spec(_specv3); _specv3.loader.exec_module(_v3)
_spec_vio = _ilu.spec_from_file_location("_video_io", str(_HERE.parent / "stages" / "video_io.py"))
_vio = _ilu.module_from_spec(_spec_vio); _spec_vio.loader.exec_module(_vio)

face_centered_aspect_crop = _v4.face_centered_aspect_crop
replace_portrait          = _v4.replace_portrait
encode_video              = _v4.encode_video
_mean_lum_ratio_v9        = _v4._mean_lum_ratio_v9
hardlink_meta             = _v4.hardlink_meta
find_video                = _v4.find_video
load_photo_bank           = _v4.load_photo_bank
decode_layout             = _v4.decode_layout
assign_celebs_to_portraits = _v4.assign_celebs_to_portraits
ensure_h264               = _vio.ensure_h264

write_reference_video = _v3.write_reference_video
slug_to_name          = _v3.slug_to_name
pick_prompt           = _v3.pick_prompt   # reuses 75/15/10 mix

# ─── Constants ───────────────────────────────────────────────────────────
LAYOUTS = ["SOL", "OSL", "OLS", "SLO", "LSO", "LOS"]
SHORT_TO_LETTER = {"swift": "S", "obama": "O", "lecun": "L"}
LETTER_TO_SHORT = {v: k for k, v in SHORT_TO_LETTER.items()}
SHORT_TO_FULL   = {"swift": "taylor_swift",
                    "obama": "barack_obama",
                    "lecun": "yann_lecun"}
FULL_TO_SHORT   = {v: k for k, v in SHORT_TO_FULL.items()}
IID_CELEBS = ("swift", "obama", "lecun")


# ─── Utilities ───────────────────────────────────────────────────────────
# IMPORTANT — filename-layout vs camera-LMR convention.
# The base teleop filenames `quick_<celeb>_<layout>_*` use the OPERATOR'S
# L→M→R perspective (operator faces the table from the side opposite the
# camera). The SAM seeds + portrait_corners.json see camera-LMR. The two
# are mirror-reversed:
#       camera_layout = filename_layout[::-1]
# Verified empirically across all 9 (target,layout) base cells (2026-05-18).
# Internally THIS SCRIPT uses camera-LMR throughout — that is the convention
# that matches the actual paper positions in the wrist-camera frames, which
# is what the inpainter writes into.
def filename_layout_to_camera(filename_layout: str) -> str:
    return filename_layout[::-1]


def physical_slot(layout_camera_lmr: str, celeb_short: str) -> int:
    """Index 0/1/2 = camera-LEFT/MIDDLE/RIGHT — where `celeb_short` sits in a
    layout given in camera-LMR convention."""
    return layout_camera_lmr.index(SHORT_TO_LETTER[celeb_short])


def pid_position_camera_lmr(corners_data: dict) -> dict[str, int]:
    """Map each pid → 0/1/2 (camera-LEFT/MIDDLE/RIGHT) by sorting on mean_x of
    the first decoded corners frame.
    """
    means: dict[str, float] = {}
    for pid, frames in corners_data["portraits"].items():
        for fi, rec in frames.items():
            if rec.get("corners") is not None:
                xs = [c[0] for c in rec["corners"]]
                means[pid] = float(sum(xs)) / float(len(xs))
                break
    sorted_pids = sorted(means, key=lambda k: means[k])
    return {pid: idx for idx, pid in enumerate(sorted_pids)}


def find_orig_target_pid(seeds_dict: dict | None, orig_target_short: str) -> str:
    """Find the pid where the recorded target identity sits (per seeds.celebs
    which was ArcFace-verified at pipeline time)."""
    if seeds_dict and "celebs" in seeds_dict:
        for i, c in enumerate(seeds_dict["celebs"]):
            if c == orig_target_short:
                return str(i)
    raise ValueError(f"cannot find orig target pid: seeds.celebs missing or no match for {orig_target_short!r}")


def pid_to_short_for_camera_layout(corners_data: dict, layout_camera_lmr: str) -> dict[str, str]:
    """Map each pid to the celeb short for the GIVEN camera-LMR layout.
    Driven by physical position only (mean_x sort)."""
    positions = pid_position_camera_lmr(corners_data)
    layout_celebs = [LETTER_TO_SHORT[c] for c in layout_camera_lmr]
    return {pid: layout_celebs[pos] for pid, pos in positions.items()}


def discover_base_teleops(root: Path) -> dict[int, list[Path]]:
    """Group base teleop dirs by camera-physical-target-slot (0=L, 1=M, 2=R).

    Reads reference.json + portrait_seeds.json — uses seeds.celebs (ArcFace-
    verified) to find where the target physically is, NOT the filename layout
    (which is operator-LMR, mirror-reversed from camera-LMR)."""
    groups: dict[int, list[Path]] = {0: [], 1: [], 2: []}
    for ep in sorted(root.iterdir()):
        if not ep.is_dir():
            continue
        ref_path = ep / "reference.json"
        masks_path = ep / "portrait_masks.pkl"
        corners_path = ep / "portrait_corners.json"
        seeds_path = ep / "portrait_seeds.json"
        if not all(p.is_file() for p in (ref_path, masks_path, corners_path, seeds_path)):
            continue
        sidecar = json.loads(ref_path.read_text())
        seeds_dict = json.loads(seeds_path.read_text())
        target_short = sidecar.get("target_celeb", "")
        if target_short not in SHORT_TO_LETTER:
            continue
        try:
            orig_target_pid = find_orig_target_pid(seeds_dict, target_short)
        except ValueError:
            print(f"  [WARN] {ep.name}: no orig target pid in seeds — skipped", flush=True)
            continue
        corners_data = json.loads(corners_path.read_text())
        positions = pid_position_camera_lmr(corners_data)
        slot = positions[orig_target_pid]
        groups[slot].append(ep)
    return groups


def load_cotrain_bank(bank_root: Path) -> dict[str, list[Path]]:
    """Load 3-celeb bank. Returns {full_name: [photo_paths]} sorted by name."""
    bank: dict[str, list[Path]] = {}
    for full in SHORT_TO_FULL.values():
        d = bank_root / full
        if not d.is_dir():
            raise SystemExit(f"[FATAL] missing bank dir: {d}")
        photos = sorted(p for p in d.iterdir()
                          if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not photos:
            raise SystemExit(f"[FATAL] no photos in: {d}")
        bank[full] = photos
    return bank


def enumerate_tuples(
    bank: dict[str, list[Path]],
    base_groups: dict[int, list[Path]],
    *,
    include_base_tuples: bool,
) -> tuple[list[tuple[str, Path, str]], set[tuple[str, str, str]]]:
    """Build the list of (target_short, target_photo_path, layout) tuples.

    `base_covered` is the set of (target_short, photo_filename, layout) tuples
    that are already in base teleops. If include_base_tuples=False, these are
    skipped from the returned list.

    Base coverage assumption: within each (target, layout) recording phase,
    the operator cycled through ALL heldout photos of the target. So every
    heldout_*.png file in the target's bank counts as covered for that
    (target, layout). Scraped photos are NOT in base — only heldout was
    physically printed at recording time.
    """
    # base_covered is keyed by CAMERA-LMR layout (matches the LAYOUTS we
    # enumerate). Convert filename → camera-LMR via [::-1] at load time.
    base_covered: set[tuple[str, str, str]] = set()
    seen_layouts_per_target: dict[str, set[str]] = {c: set() for c in IID_CELEBS}
    for slot in (0, 1, 2):
        for ep in base_groups[slot]:
            sc = json.loads((ep / "reference.json").read_text())
            t = sc["target_celeb"]
            l_camera = filename_layout_to_camera(sc["layout"])
            seen_layouts_per_target.setdefault(t, set()).add(l_camera)
    for t, layouts_seen in seen_layouts_per_target.items():
        full = SHORT_TO_FULL[t]
        for photo in bank[full]:
            if photo.name.startswith("heldout_"):
                for l in layouts_seen:
                    base_covered.add((t, photo.name, l))

    tuples: list[tuple[str, Path, str]] = []
    for short in IID_CELEBS:
        full = SHORT_TO_FULL[short]
        for photo in bank[full]:
            for layout in LAYOUTS:
                key = (short, photo.name, layout)
                if (not include_base_tuples) and key in base_covered:
                    continue
                tuples.append((short, photo, layout))
    return tuples, base_covered


def build_distractor_combos(
    layout: str, target_short: str, bank: dict[str, list[Path]],
    K: int, rng: random.Random,
) -> list[tuple[Path, Path]]:
    """All-or-sampled distractor photo combos for one tuple.

    Returns a list of (D1_photo, D2_photo) tuples where D1 is the photo for
    the position-after-target and D2 for the position-after-D1 (cyclic 0→1→2).
    """
    target_pos = physical_slot(layout, target_short)
    d1_pos = (target_pos + 1) % 3
    d2_pos = (target_pos + 2) % 3
    d1_short = LETTER_TO_SHORT[layout[d1_pos]]
    d2_short = LETTER_TO_SHORT[layout[d2_pos]]
    d1_pool = bank[SHORT_TO_FULL[d1_short]]
    d2_pool = bank[SHORT_TO_FULL[d2_short]]
    full_combos = [(p1, p2) for p1 in d1_pool for p2 in d2_pool]    # 64
    if K >= len(full_combos):
        return full_combos
    return rng.sample(full_combos, K)


# ─── Per-base-episode disk cache ─────────────────────────────────────────
# What we cache PER BASE (persists across all augmentation runs):
#   <base_ep>/portrait_masks.pkl          (already exists from stage 2 — RLE format)
#   <base_ep>/portrait_corners.json       (already exists from stage 3)
#   <base_ep>/portrait_seeds.json         (already exists from stage 2)
#   <base_ep>/frame_0.png                 (already exists)
#   <base_ep>/aug_cache_masks_decoded.npz NEW — pre-decoded uint8 binary masks
#                                         (RLE→binary decode is ~1.6s/variant
#                                         × K variants; this caches it once)
#
# What we cache PER PROCESS, in memory:
#   - portrait_masks.pkl loaded once per base
#   - decoded source frames (~500 MB per base — too big for disk cache)
#   - decoded masks dict (loaded from aug_cache_masks_decoded.npz on first
#     use, kept in memory for the duration of this base's variants)
def load_or_build_decoded_masks(
    base_ep: Path, masks_pkl: Path | None,
) -> dict[int, dict[int, np.ndarray]]:
    """Return {fi: {pid_int: H×W uint8 binary mask}}.

    On first call for a base, decodes from portrait_masks.pkl (RLE) and
    writes a compressed npz cache. Subsequent calls (re-runs) load directly
    from the npz — bypassing the ~1.6s RLE-decode per variant.

    Keyed by string "f{fi}_p{pid}" inside the npz so individual entries can
    be looked up without loading the whole archive.
    """
    if masks_pkl is None or not masks_pkl.is_file():
        return {}
    cache_npz = base_ep / "aug_cache_masks_decoded.npz"
    if cache_npz.is_file():
        try:
            data = np.load(cache_npz, allow_pickle=False)
            decoded: dict[int, dict[int, np.ndarray]] = {}
            for k in data.files:
                # key format: "f{fi:06d}_p{pid:02d}"
                fi = int(k[1:7]); pid = int(k[9:])
                decoded.setdefault(fi, {})[pid] = data[k]
            return decoded
        except Exception as e:
            print(f"  [WARN] failed reading mask cache {cache_npz}: {e}; rebuilding", flush=True)
    # Build cache from RLE
    with open(masks_pkl, "rb") as f:
        cache = pickle.load(f)
    rle_per_frame = cache.get("masks", {})
    decoded: dict[int, dict[int, np.ndarray]] = {}
    to_save: dict[str, np.ndarray] = {}
    for fi, pid_masks in rle_per_frame.items():
        decoded[fi] = {}
        for pid_int, payload in pid_masks.items():
            if "rle" not in payload:
                continue
            m = mask_util.decode(payload["rle"]).astype(np.uint8)
            if m.ndim == 3:
                m = m[:, :, 0]
            decoded[fi][pid_int] = m
            to_save[f"f{int(fi):06d}_p{int(pid_int):02d}"] = m
    if to_save:
        try:
            np.savez_compressed(cache_npz, **to_save)
        except Exception as e:
            print(f"  [WARN] failed writing mask cache {cache_npz}: {e}", flush=True)
    return decoded


# ─── Per-base-episode renderer ───────────────────────────────────────────
def _composite_variant_from_cache(
    cached_frames: list[np.ndarray],
    cached_masks_decoded: dict[int, dict[int, np.ndarray]],
    corners_data: dict,
    pid_photos: dict[str, np.ndarray],
    frame_0: np.ndarray | None,
    out_video: Path,
    fps: int,
    work_dir: Path,
) -> int:
    """Composite ONE variant using pre-decoded frames + pre-decoded masks.
    Inherent per-variant cost: warp + Reinhard + seamlessClone per portrait
    per frame, plus PNG write + ffmpeg encode. Per-base cost is fully
    amortized by the caller."""
    work_dir.mkdir(parents=True, exist_ok=True)
    n_frames = corners_data["n_frames"]
    written = 0
    for fi, src_frame in enumerate(cached_frames):
        if fi >= n_frames:
            break
        out = src_frame
        for pid_str, photo in pid_photos.items():
            pid_int = int(pid_str)
            mask = cached_masks_decoded.get(fi, {}).get(pid_int)
            rec = corners_data["portraits"][pid_str].get(str(fi))
            if rec is None or rec.get("corners") is None:
                continue
            if mask is None:
                print(
                    f"[WARN] per_frame_mask_missing: expected=portrait_masks.pkl["
                    f"frame={fi}][pid={pid_int}], got=None, "
                    f"fallback=fillPoly_from_corners (per-frame occlusion lost)",
                    flush=True,
                )
                H, W = src_frame.shape[:2]
                mask = np.zeros((H, W), dtype=np.uint8)
                pts = np.asarray(rec["corners"], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1)
            corners = np.asarray(rec["corners"], dtype=np.float32)
            lum_ratio = None
            if frame_0 is not None:
                ratio = _mean_lum_ratio_v9(src_frame, frame_0, mask)
                lum_ratio = np.full(src_frame.shape[:2], ratio, dtype=np.float32)
            out = replace_portrait(out, mask, photo, corners, lum_ratio=lum_ratio)
        cv2.imwrite(str(work_dir / f"f{fi:06d}.png"), out)
        written += 1
    encode_video(work_dir, out_video, fps=fps)
    for p in work_dir.glob("f*.png"):
        p.unlink()
    return written


def render_base_ep_variants(
    base_ep: Path,
    variants: list[dict],
    out_root: Path,
    bank: dict[str, list[Path]],
    fps: int,
    seed: int,
    *,
    force: bool = False,
    debug: bool = False,
) -> list[dict]:
    """Render all assigned variants for one base teleop. Per-base work
    (corners load, masks_pkl load, video decode) is done ONCE here and
    reused across all K variants — that's what makes the tuple-driven
    pipeline cheap at K=64 distractor combos per tuple.
    """
    corners_data = json.loads((base_ep / "portrait_corners.json").read_text())
    masks_pkl: Path | None = base_ep / "portrait_masks.pkl"
    if not masks_pkl.is_file():
        masks_pkl = None
    sidecar_orig = json.loads((base_ep / "reference.json").read_text())
    orig_target_short = sidecar_orig["target_celeb"]
    orig_layout_filename = sidecar_orig["layout"]
    orig_layout_camera = filename_layout_to_camera(orig_layout_filename)
    seeds_path = base_ep / "portrait_seeds.json"
    seeds_dict = json.loads(seeds_path.read_text())   # required for orig target pid

    pid_target_orig = find_orig_target_pid(seeds_dict, orig_target_short)
    pid_positions = pid_position_camera_lmr(corners_data)
    orig_target_camera_pos = pid_positions[pid_target_orig]

    src_video = find_video(base_ep)
    frame_0_path = base_ep / "frame_0.png"
    frame_0_img = cv2.imread(str(frame_0_path)) if frame_0_path.is_file() else None

    # ── per-base CACHE — three layers (all loaded/built ONCE here, then
    # reused for every variant of this base):
    #   (a) decoded source frames (in memory)
    #   (b) decoded masks dict (in memory, backed by on-disk
    #       aug_cache_masks_decoded.npz — built first time, reloaded subsequent
    #       runs to bypass ~1.6s RLE decode per variant)
    #   (c) portrait_corners.json + portrait_seeds.json (already in memory above)
    t_cache = time.time()
    n_frames_expected = corners_data["n_frames"]
    cached_masks_decoded = load_or_build_decoded_masks(base_ep, masks_pkl)
    t_masks = time.time() - t_cache
    cached_frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(ensure_h264(src_video)))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cached_frames.append(fr)
        if len(cached_frames) >= n_frames_expected:
            break
    cap.release()
    cache_s = time.time() - t_cache
    print(f"   cache: {len(cached_frames)} frames + {sum(len(v) for v in cached_masks_decoded.values())} masks ready "
          f"in {cache_s:.2f}s (masks={t_masks:.2f}s, decode={cache_s-t_masks:.2f}s); "
          f"reused across {len(variants)} variants this run", flush=True)

    work_dir = Path("/tmp") / f"_aug_t3_{base_ep.name}_{time.time_ns()}"
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + hash(base_ep.name) % 1_000_000)
    rendered: list[dict] = []
    all_slugs_full = [SHORT_TO_FULL[c] for c in IID_CELEBS]

    try:
        for v in variants:
            tuple_idx = v["tuple_idx"]
            var_idx   = v["var_idx"]
            target_short    = v["target_short"]
            target_photo    = v["target_photo"]
            layout          = v["layout"]
            d1_photo        = v["d1_photo"]
            d2_photo        = v["d2_photo"]

            var_name = f"{base_ep.name}__t3_{tuple_idx:04d}_v{var_idx:02d}"
            var_out = out_root / var_name
            var_video = var_out / "videos/observation.images.camera1/chunk-000/file-000.mp4"
            ref_video = var_out / "videos/observation.images.reference/chunk-000/file-000.mp4"
            if var_video.is_file() and ref_video.is_file() and not force:
                rendered.append({"variant": var_name, "skipped": True})
                continue

            # 1. Build pid → photo + pid → celeb_full for this variant.
            #    Constraint: pid_target_orig must end up holding `target_short`,
            #    matching the recorded arm trajectory. `layout` is camera-LMR.
            pid_to_new_short = pid_to_short_for_camera_layout(corners_data, layout)
            if pid_to_new_short[pid_target_orig] != target_short:
                rendered.append({
                    "variant": var_name,
                    "error": f"slot mismatch: pid_to_new[{pid_target_orig}]="
                             f"{pid_to_new_short[pid_target_orig]} != {target_short} "
                             f"(orig_target_camera_pos={orig_target_camera_pos}, "
                             f"layout={layout})",
                })
                continue

            # Distractor pids — match by short name → photo
            tgt_pos = physical_slot(layout, target_short)
            d1_short = LETTER_TO_SHORT[layout[(tgt_pos + 1) % 3]]
            d2_short = LETTER_TO_SHORT[layout[(tgt_pos + 2) % 3]]
            pid_photos: dict[str, np.ndarray] = {}
            pid_to_full: dict[str, str] = {}
            photo_assignment: dict[str, Path] = {}  # for sidecar manifest
            for pid, short in pid_to_new_short.items():
                if pid == pid_target_orig:
                    photo = target_photo
                elif short == d1_short:
                    photo = d1_photo
                elif short == d2_short:
                    photo = d2_photo
                else:
                    raise RuntimeError(f"pid {pid} has unexpected short {short!r}")
                img = cv2.imread(str(photo), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"cannot read photo: {photo}")
                pts = np.asarray(corners_data["portraits"][pid]["0"]["corners"], dtype=np.float32)
                paper_top  = float(np.linalg.norm(pts[1] - pts[0]))
                paper_left = float(np.linalg.norm(pts[3] - pts[0]))
                target_aspect = paper_top / max(paper_left, 1e-6)
                pid_photos[pid] = face_centered_aspect_crop(img, target_aspect)
                pid_to_full[pid] = SHORT_TO_FULL[short]
                photo_assignment[pid] = photo

            # 2. Reference photo: a different photo of target than the one painted
            target_full = SHORT_TO_FULL[target_short]
            ref_candidates = [p for p in bank[target_full] if p != target_photo]
            ref_photo = rng.choice(ref_candidates) if ref_candidates else target_photo

            # 3. Inpaint — uses pre-decoded source frames + pre-decoded masks
            n_written = _composite_variant_from_cache(
                cached_frames, cached_masks_decoded, corners_data, pid_photos,
                frame_0=frame_0_img,
                out_video=var_video, fps=fps, work_dir=work_dir,
            )
            # 4. Reference video
            write_reference_video(ref_photo, n_written, fps, ref_video)

            # 5. Prompt: 75/15/10 mix — reuses generate_aug_broad helper.
            prompt, bucket = pick_prompt(target_full, all_slugs_full, rng)

            # 6. Hardlink data + meta + sidecars
            hardlink_meta(base_ep, var_out)
            new_sidecar = {**sidecar_orig}
            new_sidecar["source"] = "augmented_cotrain"
            new_sidecar["augmented_from"] = base_ep.name
            new_sidecar["tuple_idx"] = tuple_idx
            new_sidecar["variant_idx"] = var_idx
            new_sidecar["target_celeb"] = target_short
            new_sidecar["target_celeb_name"] = slug_to_name(target_full)
            new_sidecar["target_celeb_full"] = target_full
            # `layout` here is camera-LMR (matches what the wrist-camera sees).
            # Filename layouts in base teleops are operator-LMR = camera-LMR
            # reversed; recorded separately for traceability.
            new_sidecar["layout"] = layout
            new_sidecar["layout_camera_lmr"] = layout
            new_sidecar["orig_layout_filename"] = orig_layout_filename
            new_sidecar["orig_layout_camera_lmr"] = orig_layout_camera
            new_sidecar["reference_photo"] = str(ref_photo)
            new_sidecar["prompt"] = prompt
            new_sidecar["prompt_bucket"] = bucket
            (var_out / "reference.json").write_text(json.dumps(new_sidecar, indent=2))

            (var_out / "augmentation.json").write_text(json.dumps({
                "src_episode": base_ep.name,
                "tuple_idx": tuple_idx, "variant_idx": var_idx,
                "strategy_version": "cotrain",
                "pid_to_celeb_full": pid_to_full,
                "orig_target_short": orig_target_short,
                "orig_layout_filename": orig_layout_filename,
                "orig_layout_camera_lmr": orig_layout_camera,
                "new_target_short": target_short,
                "new_layout_camera_lmr": layout,
                "workspace_photos": {pid: str(p) for pid, p in photo_assignment.items()},
                "reference_photo": str(ref_photo),
                "prompt": prompt, "prompt_bucket": bucket,
                "n_frames": n_written,
            }, indent=2))

            if debug:
                _ilu_spec = _ilu.spec_from_file_location(
                    "_dbg_compare", str(_HERE.parent / "dbg" / "compare_gif.py"))
                if _ilu_spec is not None:
                    _dbg = _ilu.module_from_spec(_ilu_spec); _ilu_spec.loader.exec_module(_dbg)
                    try:
                        _dbg.make_compare(var_out)
                    except Exception as e:
                        print(f"  [WARN] debug_gif_failed: {e}", flush=True)

            rendered.append({
                "variant": var_name, "tuple_idx": tuple_idx, "var_idx": var_idx,
                "target": target_short, "layout": layout,
                "target_photo": target_photo.name,
                "d1_photo": d1_photo.name, "d2_photo": d2_photo.name,
                "prompt_bucket": bucket, "n_frames": n_written,
            })
            print(f"  ✓ {var_name}  target={target_short}/{target_photo.name}  "
                  f"layout={layout}  D1={d1_photo.name} D2={d2_photo.name}  "
                  f"bucket={bucket}", flush=True)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return rendered


# ─── Main orchestrator ───────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-root", type=Path,
                   default=Path("/home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3"))
    p.add_argument("--bank-root", type=Path,
                   default=Path("/home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3_celebs/cotrain_bank"))
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--num-distractor-combos", type=int, default=64,
                   help="K distractor-photo combos per (target, photo, layout) tuple. "
                        "Default 64 = full enumeration (8x8). Set lower for sampled mode.")
    p.add_argument("--include-base-tuples", action="store_true", default=True,
                   help="Enumerate all 144 tuples including the 42 already-in-base. "
                        "Default: include (densifies base tuples with new distractor "
                        "combos — see top of file for rationale).")
    p.add_argument("--exclude-base-tuples", dest="include_base_tuples",
                   action="store_false",
                   help="Skip the 42 base-covered (target, photo, layout) tuples — "
                        "generates 102 NEW tuples only.")
    p.add_argument("--limit-tuples", type=int, default=None,
                   help="DEBUG/SMOKE: only render the first N tuples (after enum + filter)")
    p.add_argument("--limit-variants-per-tuple", type=int, default=None,
                   help="DEBUG/SMOKE: cap K to this many distractor combos per tuple")
    p.add_argument("--limit-base-eps", type=int, default=None,
                   help="DEBUG: only process the first N base teleops after grouping")
    p.add_argument("--worker-id", type=int, default=0,
                   help="This worker's id, 0-indexed (use with --num-workers for parallel runs)")
    p.add_argument("--num-workers", type=int, default=1,
                   help="Total number of parallel workers. Each worker processes the "
                        "subset of base teleops where sorted_index %% num_workers == worker_id.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--force", action="store_true")
    p.add_argument("--debug", action="store_true",
                   help="Write dbg_compare.gif per variant (slow)")
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    # 1. Load bank
    bank = load_cotrain_bank(args.bank_root)
    print(f"loaded bank: {sum(len(v) for v in bank.values())} photos "
          f"across {len(bank)} celebs", flush=True)
    for full, photos in bank.items():
        n_h = sum(1 for p in photos if p.name.startswith("heldout_"))
        n_s = sum(1 for p in photos if p.name.startswith("scraped_"))
        print(f"  {full:<15}  heldout={n_h}  scraped={n_s}  total={len(photos)}", flush=True)

    # 2. Discover + group base teleops by physical-target-slot
    base_groups = discover_base_teleops(args.base_root)
    n_total_base = sum(len(g) for g in base_groups.values())
    print(f"\nfound {n_total_base} base teleops "
          f"(L={len(base_groups[0])}, M={len(base_groups[1])}, R={len(base_groups[2])})", flush=True)
    if n_total_base == 0:
        print(f"[FATAL] no base teleops under {args.base_root}", flush=True)
        return 2

    # 3. Enumerate tuples
    tuples, base_covered = enumerate_tuples(
        bank, base_groups, include_base_tuples=args.include_base_tuples,
    )
    K_eff = min(args.num_distractor_combos, 64)
    if args.limit_variants_per_tuple is not None:
        K_eff = min(K_eff, args.limit_variants_per_tuple)
    if args.limit_tuples is not None:
        tuples = tuples[: args.limit_tuples]
    by_target_layout = Counter((t, l) for (t, _p, l) in tuples)
    print(f"\nenumerated {len(tuples)} (target, target_photo, layout) tuples "
          f"(base_covered={len(base_covered)}, include_base={args.include_base_tuples})", flush=True)
    print(f"  K (variants per tuple) = {K_eff}    →  total variants = {len(tuples) * K_eff}", flush=True)

    # 4. Map each (tuple, distractor_combo) to a base teleop
    #    Round-robin within the matching physical-slot group, so each base
    #    teleop gets ≈ uniform load.
    variants_to_render: list[dict] = []
    slot_cursor = {0: 0, 1: 0, 2: 0}
    rng_combos = random.Random(args.seed)
    for tuple_idx, (target_short, target_photo, layout) in enumerate(tuples):
        slot = physical_slot(layout, target_short)
        pool = base_groups[slot]
        if not pool:
            print(f"  [WARN] no base teleops in slot {slot} for tuple "
                  f"({target_short}, {target_photo.name}, {layout}); skipping", flush=True)
            continue
        combos = build_distractor_combos(
            layout, target_short, bank, K_eff,
            random.Random(args.seed + tuple_idx),
        )
        for var_idx, (d1_photo, d2_photo) in enumerate(combos):
            base_ep = pool[slot_cursor[slot] % len(pool)]
            slot_cursor[slot] += 1
            variants_to_render.append({
                "base_ep": base_ep,
                "tuple_idx": tuple_idx,
                "var_idx": var_idx,
                "target_short": target_short,
                "target_photo": target_photo,
                "layout": layout,
                "d1_photo": d1_photo,
                "d2_photo": d2_photo,
            })

    # 5. Group by base teleop for cache reuse
    by_base: dict[Path, list[dict]] = defaultdict(list)
    for v in variants_to_render:
        by_base[v["base_ep"]].append(v)
    ordered_bases = sorted(by_base.keys(), key=lambda p: p.name)
    if args.num_workers > 1:
        if not (0 <= args.worker_id < args.num_workers):
            raise SystemExit(f"--worker-id={args.worker_id} must be in [0, {args.num_workers})")
        ordered_bases = [b for i, b in enumerate(ordered_bases)
                          if i % args.num_workers == args.worker_id]
        print(f"  worker {args.worker_id}/{args.num_workers}: handling "
              f"{len(ordered_bases)} bases (one stripe of the sorted list)", flush=True)
    if args.limit_base_eps is not None:
        ordered_bases = ordered_bases[: args.limit_base_eps]
    print(f"\nplanning {len(variants_to_render)} total variants across {len(ordered_bases)} base teleops"
          f"  (mean {len(variants_to_render) / max(len(ordered_bases), 1):.1f} variants per base)", flush=True)

    # 6. Render
    summary = []
    t_start = time.time()
    for i, base_ep in enumerate(ordered_bases, start=1):
        ep_variants = by_base[base_ep]
        print(f"\n[{i}/{len(ordered_bases)}] {base_ep.name}  ({len(ep_variants)} variants)", flush=True)
        try:
            r = render_base_ep_variants(
                base_ep, ep_variants, args.out_root, bank,
                fps=args.fps, seed=args.seed, force=args.force, debug=args.debug,
            )
            summary.append({"base_ep": base_ep.name, "rendered": r})
        except KeyboardInterrupt:
            print("Interrupted by user.", flush=True)
            break
        except Exception as e:
            traceback.print_exc()
            summary.append({"base_ep": base_ep.name, "error": f"{type(e).__name__}: {e}"})
        elapsed = time.time() - t_start
        avg = elapsed / max(i, 1)
        eta = avg * (len(ordered_bases) - i)
        print(f"   elapsed={elapsed:7.1f}s   avg_per_base={avg:6.1f}s   "
              f"eta={eta/3600:5.2f}h", flush=True)

    (args.out_root / "_run_summary.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_tuples": len(tuples),
        "K_eff": K_eff,
        "n_variants_planned": len(variants_to_render),
        "n_bases_touched": len(by_base),
        "results": summary,
    }, indent=2))
    print(f"\nDone. Summary → {args.out_root / '_run_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
