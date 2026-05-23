#!/usr/bin/env python3
"""Debug helper: overlay the 3 portrait masks on frame 0 of each episode.

Eyeball check after stage 2 to confirm SAM is segmenting the right things.
Saves <episode_dir>/dbg_overlay_frame0.png with semi-transparent coloured
polygons drawn over each portrait.

Usage:
    python eval_3/aug/dbg/mask_overlay.py /path/to/episode_dir
    python eval_3/aug/dbg/mask_overlay.py --root ~/LeMonkey/datasets/eval3_quick
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util


COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]   # BGR: green, blue, red


def overlay_one(ep_dir: Path) -> dict:
    """Render a frame-0 PNG with portrait masks overlaid; return
    ``{ep, saved, n_portraits}`` (or ``{ep, error}`` on failure)."""
    masks_pkl = ep_dir / "portrait_masks.pkl"
    if not masks_pkl.is_file():
        return {"ep": ep_dir.name, "error": "portrait_masks.pkl missing"}

    with open(masks_pkl, "rb") as f:
        cache = pickle.load(f)
    # Prefer the local h264 sidecar over the cache's stale absolute path,
    # and over the original mp4 (which is often AV1 and not cv2-decodable).
    local_cands = sorted(ep_dir.glob("videos/observation.images.camera1/chunk-*/file-*.mp4"))
    h264 = [p for p in local_cands if "__h264" in p.name]
    plain = [p for p in local_cands if "__h264" not in p.name]
    if h264:
        video = str(h264[0])
    elif plain:
        video = str(plain[0])
    else:
        video = cache["video_path"]
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return {"ep": ep_dir.name, "error": f"cannot read frame 0 of {video}"}

    overlay = frame.copy()
    f0 = cache["masks"].get(0, {})
    for pid, payload in sorted(f0.items()):
        mask = mask_util.decode(payload["rle"]).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        col = COLORS[pid % 3]
        # tint
        coloured = np.zeros_like(frame); coloured[:] = col
        blended = cv2.addWeighted(coloured, 0.4, overlay, 0.6, 0)
        overlay = np.where(mask[:, :, None] > 0, blended, overlay)
        # contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, col, 2)
        # label
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(overlay, f"id={pid}  s={payload['score']:.1f}",
                        (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    side_by_side = cv2.hconcat([frame, overlay])
    out_path = ep_dir / "dbg_overlay_frame0.png"
    cv2.imwrite(str(out_path), side_by_side)
    return {"ep": ep_dir.name, "saved": str(out_path), "n_portraits": len(f0)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode_dir", nargs="?", default=None,
                   help="Path to a single episode directory to overlay")
    p.add_argument("--root", default=None,
                   help="Root containing many episode directories to overlay")
    args = p.parse_args()

    if (args.episode_dir is None) == (args.root is None):
        print("[ERROR] specify one of: episode_dir, --root", file=sys.stderr)
        return 2
    eps = [Path(args.episode_dir)] if args.episode_dir else \
          sorted(p for p in Path(args.root).iterdir() if p.is_dir())

    for ep in eps:
        r = overlay_one(ep)
        if "error" in r:
            print(f"  ✗ {r['ep']:50s}  {r['error']}")
        else:
            print(f"  ✓ {r['ep']:50s}  {r['n_portraits']} portraits → {r['saved']}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
