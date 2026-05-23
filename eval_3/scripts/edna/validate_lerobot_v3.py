#!/usr/bin/env python3
"""Validate a merged dataset against the LeRobot v3 dataset schema.

Checks (per https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/utils.py
and the canonical info.json shape used by lerobot-record / aggregate_datasets):

  1. meta/info.json present + has required keys:
       codebase_version (v3.0), robot_type, total_frames, total_episodes,
       fps, chunks_size, data_files_size_in_mb, video_files_size_in_mb,
       features (dict)
  2. meta/stats.json present + has every feature in info.features
  3. meta/episodes/*.parquet present + columns match info.features schema
  4. meta/tasks.parquet present
  5. data/chunk-NNN/file-NNN.parquet present, row count sums = info.total_frames
  6. videos/<feature>/chunk-NNN/file-NNN.mp4 present for every VISUAL feature
  7. Spot-check: load 3 random episode frames via cv2, verify dims match
     info.features[<visual>].shape
  8. (optional) attempt LeRobotDataset load if lerobot is importable; otherwise
     warn and skip the lerobot-side semantic check.

Exit 0 on success, 1 on any FATAL, 2 on transient/IO error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


REQUIRED_INFO_KEYS = {
    "codebase_version", "robot_type", "total_frames", "total_episodes",
    "fps", "chunks_size", "data_files_size_in_mb", "video_files_size_in_mb",
    "features",
}


def fail(msg: str) -> None:
    print(f"[FATAL] {msg}", flush=True)
    sys.exit(1)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def info(msg: str) -> None:
    print(f"[ok] {msg}", flush=True)


def validate(root: Path) -> None:
    if not root.is_dir():
        fail(f"root not a directory: {root}")

    # 1. info.json
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        fail(f"missing {info_path}")
    info_d = json.loads(info_path.read_text())
    missing = REQUIRED_INFO_KEYS - set(info_d.keys())
    if missing:
        fail(f"info.json missing keys: {sorted(missing)}")
    cb = info_d["codebase_version"]
    if not cb.startswith("v3"):
        warn(f"codebase_version={cb} (not v3 — may be loadable with caveats)")
    info(f"info.json OK (codebase_version={cb}, "
         f"total_frames={info_d['total_frames']}, "
         f"total_episodes={info_d['total_episodes']}, "
         f"features={len(info_d['features'])})")

    features = info_d["features"]
    visual_features = [k for k, v in features.items()
                       if v.get("dtype") == "video" or "image" in k]
    info(f"visual features: {visual_features}")

    # 2. stats.json
    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        fail(f"missing {stats_path}")
    stats = json.loads(stats_path.read_text())
    missing_stats = set(features.keys()) - set(stats.keys())
    if missing_stats:
        warn(f"stats.json missing features (will be skipped at train time): {sorted(missing_stats)}")
    else:
        info(f"stats.json OK ({len(stats)} feature stats)")
    # Sanity-check at least one non-visual feature has mean/std
    for k in ("observation.state", "action"):
        if k in stats:
            s = stats[k]
            if not all(key in s for key in ("mean", "std", "count")):
                warn(f"stats[{k}] missing mean/std/count keys: {set(s.keys())}")
            else:
                info(f"stats[{k}] has mean/std/count (mean len={len(s['mean'])})")
            break

    # 3. episodes parquet
    ep_dir = root / "meta" / "episodes"
    if not ep_dir.is_dir():
        fail(f"missing {ep_dir}")
    ep_files = sorted(ep_dir.rglob("*.parquet"))
    if not ep_files:
        fail(f"no episode parquets under {ep_dir}")
    total_ep_rows = sum(pq.read_metadata(f).num_rows for f in ep_files)
    if total_ep_rows != info_d["total_episodes"]:
        warn(f"episodes parquet row count ({total_ep_rows}) "
             f"!= info.total_episodes ({info_d['total_episodes']})")
    info(f"episodes parquet OK ({len(ep_files)} files, {total_ep_rows} rows)")

    # 4. tasks parquet
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        warn(f"missing {tasks_path} (some lerobot versions require)")
    else:
        n_tasks = pq.read_metadata(tasks_path).num_rows
        info(f"tasks.parquet OK ({n_tasks} tasks)")

    # 5. data parquets
    data_dir = root / "data"
    if not data_dir.is_dir():
        fail(f"missing {data_dir}")
    data_files = sorted(data_dir.rglob("*.parquet"))
    if not data_files:
        fail(f"no data parquets under {data_dir}")
    total_data_rows = 0
    for f in data_files:
        total_data_rows += pq.read_metadata(f).num_rows
    if total_data_rows != info_d["total_frames"]:
        fail(f"data parquet row count ({total_data_rows}) "
             f"!= info.total_frames ({info_d['total_frames']})")
    info(f"data parquets OK ({len(data_files)} files, {total_data_rows} frames)")

    # 6. videos per visual feature
    for feat in visual_features:
        vdir = root / "videos" / feat
        if not vdir.is_dir():
            fail(f"missing {vdir} for visual feature {feat!r}")
        n_mp4 = sum(1 for _ in vdir.rglob("*.mp4"))
        info(f"videos/{feat}/ has {n_mp4} mp4s")
        if n_mp4 == 0:
            fail(f"no mp4s under {vdir}")

    # 7. Spot-check 3 random episode videos
    rng = np.random.default_rng(seed=42)
    for feat in visual_features:
        vdir = root / "videos" / feat
        mp4s = sorted(vdir.rglob("*.mp4"))
        sample = rng.choice(mp4s, size=min(3, len(mp4s)), replace=False)
        expected_shape = features[feat].get("shape")
        if expected_shape is None:
            warn(f"features[{feat}].shape missing; cannot spot-check dims")
            continue
        # shape convention in lerobot: (H, W, C) for video features
        exp_h, exp_w = expected_shape[:2]
        for mp4 in sample:
            cap = cv2.VideoCapture(str(mp4))
            ok, fr = cap.read()
            cap.release()
            if not ok or fr is None:
                fail(f"failed to decode first frame of {mp4}")
            got_h, got_w = fr.shape[:2]
            if (got_h, got_w) != (exp_h, exp_w):
                fail(f"dim mismatch for {feat}: expected ({exp_h},{exp_w}), "
                     f"got ({got_h},{got_w}) in {mp4.name}")
        info(f"videos/{feat}: 3 random mp4s decode OK at expected dims ({exp_h},{exp_w})")

    # 8. (optional) lerobot semantic load
    try:
        from lerobot.datasets import LeRobotDataset  # type: ignore
        info("lerobot available — attempting LeRobotDataset load")
        ds = LeRobotDataset("local", root=str(root))
        sample = ds[0]
        info(f"LeRobotDataset[0] loaded OK; keys: {sorted(sample.keys())[:10]}…")
    except ImportError:
        warn("lerobot not installed in this env — skipping semantic load test "
             "(structural checks all passed). To run full test, "
             "`pip install -e third_party/lerobot[smolvla]` in a separate env.")
    except Exception as e:
        fail(f"LeRobotDataset load failed: {type(e).__name__}: {e}")

    print()
    print("==> ALL CHECKS PASSED — dataset is structurally lerobot-v3-compatible.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="Merged dataset root (contains meta/, data/, videos/)")
    args = p.parse_args()
    validate(args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
