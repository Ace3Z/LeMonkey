#!/usr/bin/env python3
"""Visually verify the so101_eval3_cotrain_grounding dataset: label <-> bbox <-> face.

Renders verification panels from a VL-pairs manifest: each panel draws the
frame-0 wrist-cam image with all 3 portrait quads (`quad_corners_norm`) and
their `celeb_name` labels overlaid, plus the reference photo. Lets a human
confirm every label matches the face inside its quad and the quads tightly
bound the rotated portraits.

Used 2026-05-21 to validate the v3 VL-pairs dataset after the image<->label
mispairing fix. See 2026-05-21_vl_pairs_image_mispairing.md

Expects the dataset's data.tar.zst already extracted under --data-root, so
that <data-root>/images/ and <data-root>/references/ exist.

Usage:
    python eval_3/tools/verify_vl_pairs.py --manifest manifest.parquet --data-root /tmp/cotrain_vl
"""
from __future__ import annotations

import argparse
import glob
import random

import cv2
import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="VL-pairs manifest.parquet")
    p.add_argument("--data-root", required=True,
                   help="dir with extracted images/ and references/ subdirs")
    p.add_argument("--out", default="eval_3/outputs/vl_pairs_audit")
    p.add_argument("--n-t3", type=int, default=6, help="t3-variant episodes to sample")
    p.add_argument("--n-base", type=int, default=2, help="base-teleop episodes to sample")
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()
    import os
    os.makedirs(args.out, exist_ok=True)

    df = pq.read_table(args.manifest).to_pandas()
    df = df[df.caption_type == "location_explicit"]

    t3 = [e for e in df.episode.unique() if "__t3_" in e]
    base = [e for e in df.episode.unique() if "__t3_" not in e]
    random.seed(args.seed)
    picks = (random.sample(t3, min(args.n_t3, len(t3)))
             + random.sample(base, min(args.n_base, len(base))))

    rows = []
    cols = [(0, 220, 0), (0, 165, 255), (255, 60, 200)]
    for ep in picks:
        sub = df[df.episode == ep].sort_values("pid")
        vlf = glob.glob(f"{args.data_root}/images/chunk-*/{ep}__f0000.jpg")
        if not vlf:
            print(f"[WARN] no image for {ep}; skipping")
            continue
        im = cv2.imread(vlf[0])
        H, W = im.shape[:2]
        for _, r in sub.iterrows():
            q = np.array([list(c) for c in r.quad_corners_norm], dtype=float)
            q[:, 0] *= W
            q[:, 1] *= H
            poly = q.astype(np.int32)
            uniq = len({tuple(c) for c in poly})
            col = cols[int(r.pid) % 3]
            cv2.polylines(im, [poly], True, col, 2)
            for c in poly:
                cv2.circle(im, tuple(c), 4, col, -1)
            tx, ty = poly[0]
            cv2.putText(im, f"pid{r.pid}:{r.celeb_name}" + ("" if uniq == 4 else " DEGEN!"),
                        (int(tx), max(14, int(ty) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 1, cv2.LINE_AA)
        ref = cv2.imread(f"{args.data_root}/references/{ep}__ref.jpg")
        if ref is not None:
            ref = cv2.resize(ref, (H * ref.shape[1] // ref.shape[0], H))
            pair = np.hstack([im, ref])
        else:
            pair = im
        hdr = np.zeros((24, pair.shape[1], 3), np.uint8)
        cv2.putText(hdr, ep, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([hdr, pair]))

    if not rows:
        print("[WARN] no panels rendered")
        return 1
    maxw = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, maxw - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(f"{args.out}/vl_verify.jpg", np.vstack(rows))
    for i, r in enumerate(rows):
        cv2.imwrite(f"{args.out}/vl_row{i+1}.jpg", r)
    print(f"wrote vl_verify.jpg + {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
