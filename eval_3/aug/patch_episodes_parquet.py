#!/usr/bin/env python3
"""Patch each per-episode meta/episodes/.../file-000.parquet to add
columns describing the reference video stream.

LeRobotDataset's metadata-reader (third_party/lerobot/src/lerobot/datasets/
dataset_reader.py:153) keys into the episodes-meta dataframe via:
    ep[f"videos/{vid_key}/chunk_index"]
    ep[f"videos/{vid_key}/file_index"]
    ep[f"videos/{vid_key}/from_timestamp"]
    ep[f"videos/{vid_key}/to_timestamp"]

The augmentation pipeline wrote the reference.mp4 file + patched info.json
to declare the feature, but the episodes-meta parquet (hardlinked from the
base teleop) still only has these columns for camera1. Without them, the
metadata reader crashes on load. Also adds `stats/observation.images.reference/*`
mirrored from camera1's stat shape (filled with per-channel stats of the
constant reference frame).

For each ep:
  1. Read meta/episodes/chunk-000/file-000.parquet
  2. Read first frame of reference.mp4 → compute per-channel pixel stats
  3. Compute timing (from_timestamp=0, to_timestamp = N_frames / fps)
  4. Add missing columns, write parquet back

Idempotent — no-op if columns already exist.

Usage:
    patch_episodes_parquet.py [--aug-root datasets/eval3_aug_v3]
                              [--base-root datasets/eval3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def reference_stats_from_video(ref_mp4: Path) -> dict:
    """Compute per-channel pixel stats for the (constant) reference video.

    Stats shape mirrors lerobot's per-channel format: [[[r]], [[g]], [[b]]]
    (one outer list per channel, value normalised to [0,1] like camera1's
    stats). For a constant-frame video, std=0 and all quantiles equal mean."""
    cap = cv2.VideoCapture(str(ref_mp4))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read first frame of {ref_mp4}")
    # cv2 reads BGR — flip to RGB to match the camera1 stats convention
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    # Per-channel: shape [3, H, W] after transpose
    arr = frame.transpose(2, 0, 1)
    H, W = arr.shape[1], arr.shape[2]

    def per_ch_nested(values):
        return [[[float(v)]] for v in values]

    means = arr.mean(axis=(1, 2))
    mins = arr.min(axis=(1, 2))
    maxs = arr.max(axis=(1, 2))
    # For constant-frame: std across frames is 0, but we report spatial std
    # which is what camera1's "std" is. Approx behavior.
    stds = arr.std(axis=(1, 2))
    counts = per_ch_nested([H * W] * 3)
    qs = {q: arr.reshape(3, -1).mean(axis=1) for q in
          ("q01", "q10", "q50", "q90", "q99")}
    return {
        "min": per_ch_nested(mins),
        "max": per_ch_nested(maxs),
        "mean": per_ch_nested(means),
        "std": per_ch_nested(stds),
        "count": counts,
        "q01": per_ch_nested(qs["q01"]),
        "q10": per_ch_nested(qs["q10"]),
        "q50": per_ch_nested(qs["q50"]),
        "q90": per_ch_nested(qs["q90"]),
        "q99": per_ch_nested(qs["q99"]),
    }


def patch_episodes_parquet(ep_dir: Path) -> str:
    """Patch one episode's meta/episodes/.../file-000.parquet.
    Returns 'patched', 'already_ok', or 'error: <reason>'."""
    pq_path = ep_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not pq_path.is_file():
        return "error: episodes parquet missing"
    df = pd.read_parquet(pq_path)

    REF = "observation.images.reference"
    chunk_col = f"videos/{REF}/chunk_index"
    if chunk_col in df.columns:
        return "already_ok"

    ref_mp4 = ep_dir / "videos" / REF / "chunk-000" / "file-000.mp4"
    if not ref_mp4.is_file():
        return "error: reference mp4 missing"

    # Read info.json to get fps + total_frames for from/to timestamp
    import json
    info = json.loads((ep_dir / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 30))
    n_frames = int(df.iloc[0]["length"])
    to_ts = n_frames / fps     # exclusive endpoint

    # 1. Video position columns (single-chunk, single-file, one episode per ds)
    df[chunk_col] = 0
    df[f"videos/{REF}/file_index"] = 0
    df[f"videos/{REF}/from_timestamp"] = 0.0
    df[f"videos/{REF}/to_timestamp"] = to_ts

    # 2. Per-channel stats
    try:
        s = reference_stats_from_video(ref_mp4)
    except Exception as e:
        return f"error: stats compute failed: {e}"
    for key, vals in s.items():
        col = f"stats/{REF}/{key}"
        # vals is a list with one entry (per row); we have only one row
        df[col] = [vals]

    df.to_parquet(pq_path, index=False)
    return "patched"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--aug-root", type=Path, default=Path("datasets/eval3_aug_v3"))
    p.add_argument("--base-root", type=Path, default=Path("datasets/eval3"))
    args = p.parse_args()

    targets = []
    if args.aug_root.is_dir():
        targets += sorted(p for p in args.aug_root.iterdir()
                            if p.is_dir() and "__var" in p.name)
    if args.base_root.is_dir():
        targets += sorted(p for p in args.base_root.iterdir()
                            if p.is_dir() and (p / "reference.json").is_file())

    print(f"patching {len(targets)} episode dirs", flush=True)
    counts = {"patched": 0, "already_ok": 0}
    errors: list[tuple[str, str]] = []
    for i, ep in enumerate(targets, start=1):
        status = patch_episodes_parquet(ep)
        if status in counts:
            counts[status] += 1
        else:
            errors.append((ep.name, status))
        if i % 200 == 0:
            print(f"  {i}/{len(targets)}: patched={counts['patched']} "
                  f"already_ok={counts['already_ok']} errors={len(errors)}",
                  flush=True)

    print(f"\nDone. patched={counts['patched']} already_ok={counts['already_ok']} "
          f"errors={len(errors)}")
    if errors:
        for name, e in errors[:10]:
            print(f"  [ERR] {name}: {e}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
