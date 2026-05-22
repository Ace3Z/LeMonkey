#!/usr/bin/env python3
"""Visually verify Track-3 robot episodes: prompt <-> target portrait <-> trajectory.

Renders N randomly sampled augmented episodes as annotated videos — the full
trajectory with all 3 portrait quads drawn per-frame, each quad labelled with
its celeb, the prompt + target shown in a banner, the target quad highlighted
green and distractors red. Lets a human confirm the can lands on the labelled
target portrait.

Used 2026-05-21 to verify the Track-3 co-training robot dataset (20/20 correct).
See docs/experiments/2026-05-21_track3_robot_dataset_visual_verify.md

Usage:
    python eval_3/tools/verify_robot_episodes.py
    python eval_3/tools/verify_robot_episodes.py --n 20 --seed 20 --out /tmp/verify
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import cv2
import numpy as np

SHORT_TO_FULL = {"swift": "taylor_swift", "obama": "barack_obama", "lecun": "yann_lecun"}
FULL_TO_DISP = {"taylor_swift": "Taylor Swift", "barack_obama": "Barack Obama",
                "yann_lecun": "Yann LeCun"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aug-root", default="datasets/eval3_track3_aug",
                   help="per-variant augmented episode dirs")
    p.add_argument("--base-root", default="datasets/eval3",
                   help="base teleop dirs (hold portrait_corners.json)")
    p.add_argument("--out", default="eval_3/attention_steering/dataset_verify",
                   help="output dir for annotated mp4s + montage")
    p.add_argument("--n", type=int, default=20, help="episodes to sample")
    p.add_argument("--seed", type=int, default=20)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    random.seed(args.seed)
    variants = sorted(glob.glob(f"{args.aug_root}/*/"))
    picks = []
    for v in random.sample(variants, min(3 * args.n, len(variants))):
        name = os.path.basename(v.rstrip("/"))
        aj = json.load(open(v + "augmentation.json"))
        pc = f"{args.base_root}/{aj['src_episode']}/portrait_corners.json"
        vid = v + "videos/observation.images.camera1/chunk-000/file-000.mp4"
        if os.path.exists(pc) and os.path.exists(vid):
            picks.append((name, v, aj, pc, vid))
        if len(picks) == args.n:
            break
    print(f"selected {len(picks)} episodes")

    montage_rows = []
    for ei, (name, vdir, aj, pc_path, vid) in enumerate(picks):
        corners = json.loads(open(pc_path).read())["portraits"]
        p2c = aj["pid_to_celeb_full"]
        target_full = SHORT_TO_FULL[aj["new_target_short"]]
        target_pid = next((pid for pid, c in p2c.items() if c == target_full), None)
        prompt = aj["prompt"]

        cap = cv2.VideoCapture(vid)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        BAN = 56
        out_path = f"{args.out}/ep_{ei+1:02d}_{name}.mp4"
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H + BAN))
        key_idxs = [int(r * (n - 1)) for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
        keyframes = {}

        fi = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            canvas = np.zeros((H + BAN, W, 3), np.uint8)
            canvas[BAN:] = fr
            cv2.putText(canvas, f"PROMPT: {prompt}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"TARGET: {FULL_TO_DISP[target_full]}  (pid{target_pid}, green)"
                        f"   frame {fi}/{n-1}", (8, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 1, cv2.LINE_AA)
            for pid in ("0", "1", "2"):
                rec = corners.get(pid, {}).get(str(fi)) or corners.get(pid, {}).get("0")
                if not rec or rec.get("corners") is None:
                    continue
                q = np.array(rec["corners"], dtype=np.float32)
                q[:, 1] += BAN
                poly = q.astype(np.int32)
                is_tgt = (pid == target_pid)
                col = (0, 230, 0) if is_tgt else (60, 60, 235)
                cv2.polylines(canvas, [poly], True, col, 3 if is_tgt else 2)
                disp = FULL_TO_DISP[p2c[pid]]
                tag = f"pid{pid}: {disp}" + ("  <-- TARGET" if is_tgt else "")
                tx, ty = poly[0]
                cv2.putText(canvas, tag, (int(tx) - 10, max(BAN + 14, int(ty) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
            vw.write(canvas)
            if fi in key_idxs:
                keyframes[fi] = canvas.copy()
            fi += 1
        cap.release()
        vw.release()

        strip = np.hstack([cv2.resize(keyframes[k], (W // 2, (H + BAN) // 2))
                           for k in key_idxs if k in keyframes])
        montage_rows.append(strip)
        print(f"[{ei+1:02d}/{len(picks)}] {name}  target={FULL_TO_DISP[target_full]} -> {out_path}")

    if montage_rows:
        maxw = max(r.shape[1] for r in montage_rows)
        montage_rows = [np.pad(r, ((0, 0), (0, maxw - r.shape[1]), (0, 0)))
                        for r in montage_rows]
        cv2.imwrite(f"{args.out}/_montage.jpg", np.vstack(montage_rows))
        print(f"\nwrote {args.out}/_montage.jpg  ({len(montage_rows)} rows x 5 keyframes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
