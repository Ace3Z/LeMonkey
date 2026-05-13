#!/usr/bin/env python3
"""Debug helper: build a side-by-side MP4 of original vs augmented variant.

Spot-check after stage 4 to confirm the inpaint looks natural.

Usage:
    python dbg_compare_gif.py /path/to/eval3_aug/<variant_dir>
    python dbg_compare_gif.py --root ~/LeMonkey/datasets/eval3_aug --first 5

Output: <variant_dir>/dbg_compare.mp4 (and dbg_compare_3frames.png)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def find_video(d: Path) -> Path | None:
    cands = list(d.glob("videos/*/chunk-*/file-*.mp4"))
    return cands[0] if cands else None


def find_src_video(variant_dir: Path) -> Path | None:
    aug_path = variant_dir / "augmentation.json"
    if not aug_path.is_file():
        return None
    aug = json.loads(aug_path.read_text())
    src = aug["src_episode"]
    candidates: list[Path] = []
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "LeMonkey" and (parent / "datasets").exists():
            candidates += [parent / "datasets/eval3_quick", parent / "datasets/eval3"]
            break
    candidates += [Path.home() / "LeMonkey/datasets/eval3_quick",
                   Path.home() / "LeMonkey/datasets/eval3"]
    for root in candidates:
        if (root / src).is_dir():
            return find_video(root / src)
    return None


def make_compare(variant_dir: Path, *, fps: int | None = None) -> dict:
    """Write a side-by-side MP4 at the SOURCE video's native frame rate
    (every frame, no subsampling). If `fps` is None, read the actual fps
    from the source mp4 so playback duration matches the original clip."""
    aug_video = find_video(variant_dir)
    src_video = find_src_video(variant_dir)
    if aug_video is None or src_video is None:
        return {"variant": variant_dir.name, "error": "couldn't locate src or aug video"}

    cap_src = cv2.VideoCapture(str(src_video))
    cap_aug = cv2.VideoCapture(str(aug_video))
    total = int(min(cap_src.get(cv2.CAP_PROP_FRAME_COUNT),
                    cap_aug.get(cv2.CAP_PROP_FRAME_COUNT)))
    if fps is None:
        src_fps = float(cap_src.get(cv2.CAP_PROP_FPS) or 0)
        fps = int(round(src_fps)) if src_fps > 1 else 30   # default 30 if unreadable

    bgr_frames: list[np.ndarray] = []
    for fi in range(total):
        ok1, fa = cap_src.read()
        ok2, fb = cap_aug.read()
        if not (ok1 and ok2):
            break
        cv2.putText(fa, "ORIGINAL",  (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(fb, "AUGMENTED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        bgr_frames.append(cv2.hconcat([fa, fb]))
    cap_src.release(); cap_aug.release()
    static_3 = [bgr_frames[i] for i in (0, len(bgr_frames)//2, len(bgr_frames)-1)
                if 0 <= i < len(bgr_frames)]

    out_mp4 = variant_dir / "dbg_compare.mp4"
    if bgr_frames:
        # Encode via ffmpeg (libx264) for a small, universally-playable file.
        # cv2.VideoWriter's mp4v fourcc produces files that won't play in
        # browsers/QuickTime; libx264 with -pix_fmt yuv420p works everywhere.
        H, W = bgr_frames[0].shape[:2]
        # Ensure even dimensions (yuv420p requirement).
        if W % 2 or H % 2:
            W2, H2 = W - (W % 2), H - (H % 2)
            bgr_frames = [f[:H2, :W2] for f in bgr_frames]
            H, W = H2, W2
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            for i, f in enumerate(bgr_frames):
                cv2.imwrite(str(tdp / f"f{i:04d}.png"), f)
            subprocess.run([
                "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", str(tdp / "f%04d.png"),
                "-c:v", "libx264", "-crf", "20", "-preset", "medium",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(out_mp4),
            ], check=True, stdin=subprocess.DEVNULL)
    out_png = variant_dir / "dbg_compare_3frames.png"
    if static_3:
        cv2.imwrite(str(out_png), cv2.vconcat(static_3))
    return {"variant": variant_dir.name, "saved_mp4": str(out_mp4), "saved_png": str(out_png)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variant_dir", nargs="?", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--first", type=int, default=None,
                   help="when --root: only do the first N variants")
    p.add_argument("--fps", type=int, default=None,
                   help="output fps (default: read from source video to match original playback speed)")
    args = p.parse_args()

    if (args.variant_dir is None) == (args.root is None):
        print("[ERROR] specify one of: variant_dir, --root", file=sys.stderr)
        return 2

    variants = [Path(args.variant_dir)] if args.variant_dir else \
               sorted(p for p in Path(args.root).iterdir() if p.is_dir())
    if args.first:
        variants = variants[:args.first]

    for v in variants:
        r = make_compare(v, fps=args.fps)
        if "error" in r:
            print(f"  ✗ {r['variant']:50s}  {r['error']}")
        else:
            print(f"  ✓ {r['variant']:50s}  → {r['saved_mp4']}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
