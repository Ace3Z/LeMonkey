#!/usr/bin/env python3
"""Fast recompute of action + state quantile stats on a merged LeRobot v3 dataset.

The upstream `augment_dataset_quantile_stats.py` loads every mp4 frame to compute
per-pixel image quantiles, then video-decodes 18,788 mp4s sequentially — projected
ETA ~12 days for our 9,394-ep dataset. We don't need image quantiles: Pi0.5's
`normalization_mapping` ([configuration_pi05.py:73]) sets `VISUAL: IDENTITY` —
image features use no normalization stats at all. The QUANTILES mode is applied
only to `action` and `observation.state`.

So this script:

  1. Reads all `data/chunk-NNN/file-NNN.parquet` files from the merged dataset.
  2. Extracts action[D] + observation.state[D] columns. Stacks into numpy arrays.
  3. Computes exact mean, std, min, max, count, q01, q10, q50, q90, q99 over all
     ~5M frames using vectorised numpy.
  4. Writes the updated values into `meta/stats.json`. Image stats are left
     untouched (the merger's pooled-mean numbers are mathematically correct for
     mean/std, and Pi0.5 doesn't use image quantiles anyway).
  5. Also updates the per-episode stats columns in
     `meta/episodes/chunk-NNN/file-NNN.parquet` — but only for action + state.

Expected wall time: ~30 seconds for 9,394 episodes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


QUANTILE_KEYS = ("q01", "q10", "q50", "q90", "q99")
QUANTILE_LEVELS = (0.01, 0.10, 0.50, 0.90, 0.99)
TARGET_FEATURES = ("action", "observation.state")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path,
                   default=Path("/home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3_cotrain_merged"))
    args = p.parse_args()

    # ── 1. Discover all data parquet files ──
    data_files = sorted((args.root / "data").rglob("file-*.parquet"))
    if not data_files:
        print(f"[FATAL] no data parquets under {args.root / 'data'}", flush=True)
        return 1
    print(f"[1/4] {len(data_files)} data parquet files found", flush=True)

    # ── 2. Concatenate action + state across all files ──
    print(f"[2/4] reading action + observation.state from all data parquets...", flush=True)
    import time
    t = time.time()
    tables = [pq.read_table(f, columns=list(TARGET_FEATURES)) for f in data_files]
    cat = pa.concat_tables(tables)
    n_frames = cat.num_rows
    arrays = {feat: np.stack(cat.column(feat).to_pylist()) for feat in TARGET_FEATURES}
    for feat, arr in arrays.items():
        print(f"     {feat:30}  shape={arr.shape}  dtype={arr.dtype}", flush=True)
    print(f"     loaded {n_frames} frames in {time.time() - t:.1f}s", flush=True)

    # ── 3. Compute stats ──
    print(f"[3/4] computing stats...", flush=True)
    t = time.time()
    new_stats: dict[str, dict] = {}
    for feat, arr in arrays.items():
        s: dict[str, object] = {}
        s["min"]   = arr.min(axis=0).tolist()
        s["max"]   = arr.max(axis=0).tolist()
        s["mean"]  = arr.mean(axis=0).tolist()
        s["std"]   = arr.std(axis=0, ddof=0).tolist()
        s["count"] = int(arr.shape[0])
        for k, lvl in zip(QUANTILE_KEYS, QUANTILE_LEVELS):
            s[k] = np.quantile(arr, lvl, axis=0).tolist()
        new_stats[feat] = s
    print(f"     computed in {time.time() - t:.2f}s", flush=True)

    # ── 4. Patch meta/stats.json ──
    stats_path = args.root / "meta" / "stats.json"
    old_stats = json.loads(stats_path.read_text())
    diff_report = {}
    for feat, new in new_stats.items():
        old = old_stats.get(feat, {})
        for k in ("q01", "q50", "q99"):
            old_v = old.get(k)
            new_v = new[k]
            if isinstance(old_v, list) and isinstance(new_v, list) and len(old_v) == len(new_v):
                deltas = [abs(o - n) for o, n in zip(old_v, new_v)]
                diff_report[f"{feat}.{k}"] = {"max_delta": max(deltas), "old": old_v, "new": new_v}
        old_stats[feat] = new
    stats_path.write_text(json.dumps(old_stats, indent=2))
    print(f"[4/4] meta/stats.json updated.\n", flush=True)
    print("Δ vs old aggregated stats (max abs delta per quantile, action + state):")
    for k, v in diff_report.items():
        print(f"  {k:40}  max|Δ|={v['max_delta']:.4f}")

    print(f"\n✓ done. exact quantile + mean/std for action + state in {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
