#!/usr/bin/env python3
"""Debug helper: build a side-by-side GIF of original vs augmented variant.

Spot-check after stage 4 to confirm the inpaint looks natural.

Usage:
    python dbg_compare_gif.py /path/to/eval3_aug/<variant_dir>
    python dbg_compare_gif.py --root ~/LeMonkey/datasets/eval3_aug --first 5

Output: <variant_dir>/dbg_compare.gif (and dbg_compare_3frames.png)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def find_video(d: Path) -> Path | None:
    cands = list(d.glob("videos/*/chunk-*/file-*.mp4"))
    return cands[0] if cands else None


def find_src_video(variant_dir: Path) -> Path | None:
    aug_path = variant_dir / "augmentation.json"
    if not aug_path.is_file():
        return None
    aug = json.loads(aug_path.read_text())
    src = aug["src_episode"]
    for root in [Path.home() / "LeMonkey/datasets/eval3_quick",
                 Path.home() / "LeMonkey/datasets/eval3"]:
        if (root / src).is_dir():
            return find_video(root / src)
    return None


def make_compare(variant_dir: Path, *, n_frames: int = 30, fps: int = 15) -> dict:
    aug_video = find_video(variant_dir)
    src_video = find_src_video(variant_dir)
    if aug_video is None or src_video is None:
        return {"variant": variant_dir.name, "error": "couldn't locate src or aug video"}

    cap_src = cv2.VideoCapture(str(src_video))
    cap_aug = cv2.VideoCapture(str(aug_video))
    total = int(min(cap_src.get(cv2.CAP_PROP_FRAME_COUNT),
                    cap_aug.get(cv2.CAP_PROP_FRAME_COUNT)))
    sample_idx = list(np.linspace(0, total - 1, n_frames, dtype=int))

    pil_frames: list[Image.Image] = []
    static_3: list[np.ndarray] = []
    fi = 0
    si = 0
    while si < len(sample_idx):
        ok1, fa = cap_src.read()
        ok2, fb = cap_aug.read()
        if not (ok1 and ok2):
            break
        if fi == sample_idx[si]:
            cv2.putText(fa, "ORIGINAL",  (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(fb, "AUGMENTED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            side = cv2.hconcat([fa, fb])
            pil_frames.append(Image.fromarray(cv2.cvtColor(side, cv2.COLOR_BGR2RGB)))
            if si in (0, len(sample_idx)//2, len(sample_idx) - 1):
                static_3.append(side)
            si += 1
        fi += 1
    cap_src.release(); cap_aug.release()

    out_gif = variant_dir / "dbg_compare.gif"
    if pil_frames:
        pil_frames[0].save(out_gif, save_all=True, append_images=pil_frames[1:],
                           duration=int(1000/fps), loop=0, optimize=True)
    out_png = variant_dir / "dbg_compare_3frames.png"
    if static_3:
        cv2.imwrite(str(out_png), cv2.vconcat(static_3))
    return {"variant": variant_dir.name, "saved_gif": str(out_gif), "saved_png": str(out_png)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variant_dir", nargs="?", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--first", type=int, default=None,
                   help="when --root: only do the first N variants")
    p.add_argument("--n-frames", type=int, default=30)
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()

    if (args.variant_dir is None) == (args.root is None):
        print("[ERROR] specify one of: variant_dir, --root", file=sys.stderr)
        return 2

    variants = [Path(args.variant_dir)] if args.variant_dir else \
               sorted(p for p in Path(args.root).iterdir() if p.is_dir())
    if args.first:
        variants = variants[:args.first]

    for v in variants:
        r = make_compare(v, n_frames=args.n_frames, fps=args.fps)
        if "error" in r:
            print(f"  ✗ {r['variant']:50s}  {r['error']}")
        else:
            print(f"  ✓ {r['variant']:50s}  → {r['saved_gif']}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
