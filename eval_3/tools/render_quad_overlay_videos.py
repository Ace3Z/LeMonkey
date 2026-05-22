#!/usr/bin/env python3
"""Render augmented episode videos with portrait quad overlays.

For each sampled episode, opens its augmented camera1 video and overlays the
3 portrait quads from the base episode's portrait_corners.json (per-frame),
labelled with the celeb names from the VL-pairs manifest. Degenerate quads
(< 4 distinct corners) are drawn in red and flagged.

Originally written to inspect the ~21% degenerate-quad bug in the pre-v3
VL-pairs dataset (the augmented videos themselves were fine — the degenerate
quad was an annotation-only defect). Kept as a general quad-overlay video
renderer. See docs/experiments/2026-05-21_vl_pairs_image_mispairing.md

Usage:
    python eval_3/tools/render_quad_overlay_videos.py --manifest manifest.parquet
    python eval_3/tools/render_quad_overlay_videos.py --manifest m.parquet --only-degenerate --n 4
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def is_degenerate(quad) -> bool:
    pts = np.array([list(p) for p in quad], dtype=float)
    return len({tuple(np.round(p, 5)) for p in pts}) < 4


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="VL-pairs manifest.parquet")
    p.add_argument("--aug-root", default="datasets/eval3_track3_aug",
                   help="per-variant augmented episode dirs")
    p.add_argument("--out", default="eval_3/attention_steering/quad_overlay_videos")
    p.add_argument("--n", type=int, default=4, help="episodes to render")
    p.add_argument("--only-degenerate", action="store_true",
                   help="only sample episodes that have >=1 degenerate quad")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pq.read_table(args.manifest).to_pandas()
    df = df[df.caption_type == "location_explicit"]

    picks, seen = [], set()
    for _, r in df.iterrows():
        if args.only_degenerate and not is_degenerate(r.quad_corners_norm):
            continue
        ep = r.episode
        if ep in seen:
            continue
        vdir = f"{args.aug_root}/{ep}"
        if not os.path.isdir(vdir):
            continue
        seen.add(ep)
        picks.append((ep, vdir))
        if len(picks) >= args.n:
            break

    strips = []
    for ep, vdir in picks:
        sub = df[df.episode == ep].sort_values("pid")
        quads = {int(r.pid): np.array([list(p) for p in r.quad_corners_norm], dtype=float)
                 for _, r in sub.iterrows()}
        labels = {int(r.pid): r.celeb_name for _, r in sub.iterrows()}
        vpath = f"{vdir}/videos/observation.images.camera1/chunk-000/file-000.mp4"
        cap = cv2.VideoCapture(vpath)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        strip_idxs = np.linspace(0, n - 1, 6).astype(int)
        gif_idxs = list(range(0, n, 12))
        want = set(strip_idxs.tolist()) | set(gif_idxs)
        grabbed, fi = {}, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if fi in want:
                grabbed[fi] = fr.copy()
            fi += 1
        cap.release()

        def draw(fr):
            im = fr.copy()
            for pid in (0, 1, 2):
                if pid not in quads:
                    continue
                q = quads[pid].copy()
                q[:, 0] *= W
                q[:, 1] *= H
                poly = q.astype(np.int32)
                deg = is_degenerate(quads[pid])
                col = (0, 0, 255) if deg else (0, 220, 0)
                cv2.polylines(im, [poly], True, col, 2)
                for c in poly:
                    cv2.circle(im, tuple(c), 4, col, -1)
                tx, ty = poly[0]
                txt = f"pid{pid}:{labels.get(pid,'?')}" + (" DEGEN" if deg else "")
                cv2.putText(im, txt, (int(tx), max(14, int(ty) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            return im

        frames = []
        for k in strip_idxs:
            if k in grabbed:
                f = draw(grabbed[k])
                cv2.putText(f, f"frame {k}/{n-1}", (8, H - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                frames.append(f)
        strip = np.hstack(frames)
        hdr = np.zeros((26, strip.shape[1], 3), np.uint8)
        cv2.putText(hdr, ep, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        strips.append(np.vstack([hdr, strip]))

        gif = [Image.fromarray(cv2.cvtColor(draw(grabbed[k]), cv2.COLOR_BGR2RGB))
               for k in gif_idxs if k in grabbed]
        gif[0].save(f"{args.out}/{ep}.gif", save_all=True,
                    append_images=gif[1:], duration=80, loop=0)
        print(f"wrote {args.out}/{ep}.gif  ({len(gif)} frames)")

    if strips:
        maxw = max(s.shape[1] for s in strips)
        strips = [np.pad(s, ((0, 0), (0, maxw - s.shape[1]), (0, 0))) for s in strips]
        cv2.imwrite(f"{args.out}/_montage.jpg", np.vstack(strips))
        print(f"wrote {args.out}/_montage.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
