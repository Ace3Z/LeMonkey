#!/usr/bin/env python3
"""STAGE 2 — SAM 2.1 video segmentation of the 3 printed portraits.

Per episode:
  1. Read frame 0 from the episode's mp4.
  2. Get 3 seed points (interactive OpenCV click OR loaded from a JSON file).
  3. SAM 2.1 hiera-L image predictor → 3 high-quality init masks.
  4. SAM 2.1 hiera-L video predictor → propagate masks across all frames.
  5. Cache: {frame_idx: {portrait_id: {'rle': COCO-RLE, 'score': float}}}
     written to <episode_dir>/portrait_masks.pkl

Usage:
    # interactive: open OpenCV window, click 3 points on frame 0
    python 2_segment_video.py /path/to/quick_swift_SOL_ep01_<ts> --interactive

    # headless: read seeds from <episode_dir>/portrait_seeds.json
    python 2_segment_video.py /path/to/quick_swift_SOL_ep01_<ts>

    # batch: process every episode under a root
    python 2_segment_video.py --root ~/LeMonkey/datasets/eval3_quick --interactive

Seeds JSON schema (one per episode):
    {
      "points": [[cx, cy], [cx, cy], [cx, cy]],
      "labels": [1, 1, 1]                          # 1 = positive prompt
    }

Re-running on an episode that already has portrait_masks.pkl is a no-op
unless --force is given.

See STRATEGY.md §3.2 for design rationale.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Heavy imports deferred for fast --help
def _import_heavy():
    global cv2, torch, build_sam2_image_predictor, build_sam2_video_predictor, mask_util
    import cv2  # type: ignore
    import torch  # type: ignore
    from sam2.build_sam import build_sam2_video_predictor, build_sam2  # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    import pycocotools.mask as mask_util  # type: ignore
    globals().update({
        "cv2": cv2, "torch": torch,
        "build_sam2_video_predictor": build_sam2_video_predictor,
        "build_sam2": build_sam2,
        "SAM2ImagePredictor": SAM2ImagePredictor,
        "mask_util": mask_util,
    })


# ─── Defaults ────────────────────────────────────────────────────────────────
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT_DEFAULT = Path.home() / "checkpoints/sam2.1_hiera_large.pt"


# ─── Frame 0 extraction ──────────────────────────────────────────────────────
def find_episode_video(ep_dir: Path) -> Path:
    """Locate the LeRobot v3 camera1 mp4 in <ep_dir>/videos/.../file-000.mp4."""
    candidates = list(ep_dir.glob("videos/*/chunk-*/file-*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"no .mp4 found under {ep_dir}/videos/")
    return candidates[0]


def read_frame_zero(video_path: Path) -> np.ndarray:
    """Decode and return frame 0 as BGR uint8 ndarray."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame 0 of {video_path}")
    return frame


# ─── Seed acquisition ────────────────────────────────────────────────────────
def load_seeds(ep_dir: Path) -> dict | None:
    seeds_json = ep_dir / "portrait_seeds.json"
    if not seeds_json.is_file():
        return None
    return json.loads(seeds_json.read_text())


def save_seeds(ep_dir: Path, points: list[list[int]], labels: list[int]) -> Path:
    seeds_json = ep_dir / "portrait_seeds.json"
    seeds_json.write_text(json.dumps({"points": points, "labels": labels}, indent=2))
    return seeds_json


def interactive_click(frame_bgr: np.ndarray) -> tuple[list[list[int]], list[int]]:
    """Open an OpenCV window; user left-clicks 3 portrait centers, then any key."""
    points: list[list[int]] = []
    labels: list[int] = []

    def cb(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append([x, y])
            labels.append(1)

    win = "click 3 portrait centers (LMB), then press ENTER (Esc to cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, cb)
    overlay_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    while True:
        disp = frame_bgr.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(disp, (x, y), 8, overlay_colors[i], -1)
            cv2.putText(disp, str(i), (x + 12, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_colors[i], 2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # Esc
            cv2.destroyWindow(win)
            raise KeyboardInterrupt("cancelled")
        if k in (13, 10) and len(points) == 3:  # Enter
            break
    cv2.destroyWindow(win)
    return points, labels


# ─── SAM 2.1 frame 0 → init masks ────────────────────────────────────────────
def sam_image_predict_masks(
    image_predictor: "SAM2ImagePredictor",
    frame_bgr: np.ndarray,
    points: list[list[int]],
    labels: list[int],
) -> list[np.ndarray]:
    """Run SAM 2.1 image predictor once per click → returns 3 binary masks."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_predictor.set_image(rgb)
    masks_out: list[np.ndarray] = []
    for pt, lab in zip(points, labels):
        masks, scores, _ = image_predictor.predict(
            point_coords=np.array([pt], dtype=np.float32),
            point_labels=np.array([lab], dtype=np.int32),
            multimask_output=True,
        )
        # SAM returns 3 candidate masks per prompt; keep the highest-score
        best_idx = int(scores.argmax())
        masks_out.append(masks[best_idx].astype(bool))
    return masks_out


# ─── Video predictor: propagate ──────────────────────────────────────────────
def propagate_video(
    video_predictor,
    video_path: Path,
    init_masks: list[np.ndarray],
) -> dict[int, dict[int, dict]]:
    """Returns {frame_idx: {portrait_id: {'rle': rle, 'score': float}}}."""
    state = video_predictor.init_state(video_path=str(video_path))
    for pid, m in enumerate(init_masks):
        video_predictor.add_new_mask(state, frame_idx=0, obj_id=pid, mask=m)
    out: dict[int, dict[int, dict]] = {}
    for fi, obj_ids, mask_logits in video_predictor.propagate_in_video(state):
        for oid, logit in zip(obj_ids, mask_logits):
            mask_t = (logit > 0).cpu().numpy().astype(np.uint8)
            # logit shape can be (1,H,W) or (H,W) depending on version
            if mask_t.ndim == 3:
                mask_t = mask_t[0]
            score = float(logit.max().item())
            rle = mask_util.encode(np.asfortranarray(mask_t))
            # COCO RLE 'counts' is bytes; pickle handles bytes fine
            out.setdefault(fi, {})[int(oid)] = {"rle": rle, "score": score}
    return out


# ─── Per-episode driver ──────────────────────────────────────────────────────
def process_episode(
    ep_dir: Path,
    image_predictor: "SAM2ImagePredictor",
    video_predictor,
    *,
    interactive: bool,
    force: bool,
) -> dict:
    out_pkl = ep_dir / "portrait_masks.pkl"
    if out_pkl.is_file() and not force:
        return {"ep": ep_dir.name, "skipped": True, "reason": "portrait_masks.pkl exists; use --force to redo"}

    try:
        video = find_episode_video(ep_dir)
    except FileNotFoundError as e:
        return {"ep": ep_dir.name, "error": str(e)}

    print(f"\n  → {ep_dir.name}")
    print(f"    video: {video}")

    # 1. Frame 0
    frame0 = read_frame_zero(video)

    # 2. Seeds
    seeds = load_seeds(ep_dir)
    if seeds is None:
        if not interactive:
            return {"ep": ep_dir.name, "error": "no portrait_seeds.json; pass --interactive to click"}
        print(f"    no seeds — opening interactive clicker")
        try:
            points, labels = interactive_click(frame0)
        except KeyboardInterrupt:
            return {"ep": ep_dir.name, "error": "interactive cancelled"}
        save_seeds(ep_dir, points, labels)
        seeds = {"points": points, "labels": labels}
        print(f"    saved seeds → {ep_dir/'portrait_seeds.json'}")
    points, labels = seeds["points"], seeds["labels"]
    if len(points) != 3:
        return {"ep": ep_dir.name, "error": f"need exactly 3 seed points, got {len(points)}"}

    # 3. Frame-0 init masks
    print(f"    SAM image predictor on frame 0...")
    t0 = time.time()
    init_masks = sam_image_predict_masks(image_predictor, frame0, points, labels)
    print(f"      ({time.time()-t0:.1f}s)")

    # 4. Propagate
    print(f"    SAM video predictor propagating...")
    t0 = time.time()
    masks_per_frame = propagate_video(video_predictor, video, init_masks)
    print(f"      ({time.time()-t0:.1f}s, {len(masks_per_frame)} frames)")

    # 5. Cache
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "video_path": str(video),
            "seeds": seeds,
            "masks": masks_per_frame,
        }, f)
    return {"ep": ep_dir.name, "saved": str(out_pkl), "n_frames": len(masks_per_frame)}


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("episode_dir", nargs="?", default=None,
                   help="Single episode dir to process. Mutually exclusive with --root.")
    p.add_argument("--root", default=None,
                   help="Process all episode subdirs under this root.")
    p.add_argument("--interactive", action="store_true",
                   help="Pop OpenCV window for click-prompts when seeds.json is missing.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if portrait_masks.pkl exists.")
    p.add_argument("--ckpt", default=str(SAM2_CKPT_DEFAULT),
                   help="path to sam2.1_hiera_large.pt")
    p.add_argument("--cfg", default=SAM2_CFG,
                   help="SAM 2.1 config name (sam2 picks up from configs/)")
    args = p.parse_args()

    if (args.episode_dir is None) == (args.root is None):
        print("[ERROR] specify exactly one of: episode_dir, --root", file=sys.stderr)
        return 2

    _import_heavy()
    if not Path(args.ckpt).is_file():
        print(f"[ERROR] SAM 2.1 checkpoint not found: {args.ckpt}\n"
              f"        download with:  wget -P {Path(args.ckpt).parent} "
              f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
              file=sys.stderr)
        return 1

    print(f"loading SAM 2.1 hiera-L from {args.ckpt}...")
    t0 = time.time()
    img_model = build_sam2(args.cfg, args.ckpt, device="cuda")
    image_predictor = SAM2ImagePredictor(img_model)
    video_predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device="cuda")
    print(f"  loaded in {time.time()-t0:.1f}s")

    if args.episode_dir:
        ep_dirs = [Path(args.episode_dir)]
    else:
        root = Path(args.root)
        ep_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    results: list[dict] = []
    for ep_dir in ep_dirs:
        try:
            r = process_episode(
                ep_dir, image_predictor, video_predictor,
                interactive=args.interactive, force=args.force,
            )
        except Exception as e:
            r = {"ep": ep_dir.name, "error": f"{type(e).__name__}: {e}"}
        results.append(r)

    print("\n" + "=" * 60)
    print(" summary")
    print("=" * 60)
    for r in results:
        if "saved" in r:
            print(f"  ✓ {r['ep']:50s}  {r['n_frames']:>4} frames")
        elif "skipped" in r:
            print(f"  - {r['ep']:50s}  (skipped)")
        else:
            print(f"  ✗ {r['ep']:50s}  {r.get('error','?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
