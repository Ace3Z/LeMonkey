#!/usr/bin/env python3
"""Consume the ArcFace audit parquet → produce keep_episodes.txt + hardneg_weights.npy.

Two artifacts for the Pi0.5 VL cotrain enhanced training run:

1. `keep_episodes.txt` — episode indices whose mean target_cos >= keep_threshold.
   Used via `--dataset.episodes_file=` in lerobot-train (Enhancement B-2).

2. `hardneg_weights.npy` — per-episode sampling weight; episodes with low mean
   `hardneg_gap` (confusable distractors present) get HARD_WEIGHT (default 2.0),
   others get 1.0.  Used via `--dataset.sample_weights=` (Enhancement B-3).

Run AFTER `arcface_audit_200celeb.py` has produced the audit parquet.

Per CLAUDE.md §5: no silent fallbacks. Per CLAUDE.md §7: triple-source defaults.

Usage:

  python build_keep_list_and_weights.py \
      --audit-parquet audit_200celeb.parquet \
      --keep-threshold 0.50 \
      --hardneg-gap-threshold 0.10 \
      --hard-weight 2.0 \
      --output-dir eval_3/scripts/pi05_vl_cotrain/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Triple-sourced defaults (same rationale as arcface_audit_200celeb.py):
DEFAULT_KEEP_COS = 0.50          # ArcFace LFW FAR=1e-3 = 0.36; inpainted is ~10-15% looser
DEFAULT_HARDNEG_GAP = 0.10       # gap below this = confusable distractor present
DEFAULT_HARD_WEIGHT = 2.0        # standard hard-negative oversampling weight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-parquet", type=Path, required=True,
                        help="Output of arcface_audit_200celeb.py")
    parser.add_argument("--keep-threshold", type=float, default=DEFAULT_KEEP_COS,
                        help="Episode mean target_cos must be >= this to keep")
    parser.add_argument("--hardneg-gap-threshold", type=float,
                        default=DEFAULT_HARDNEG_GAP,
                        help="Episode mean hardneg_gap < this → mark hard")
    parser.add_argument("--hard-weight", type=float, default=DEFAULT_HARD_WEIGHT,
                        help="Sampling weight for hard episodes (easy = 1.0)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write keep_episodes.txt + hardneg_weights.npy")
    parser.add_argument("--min-retention", type=float, default=0.70,
                        help="Hard stop: if keep_threshold drops more than this "
                             "fraction of episodes, abort with [WARN] and require "
                             "explicit threshold adjustment.")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("[ERR] pandas required", file=sys.stderr)
        return 2

    if not args.audit_parquet.is_file():
        print(f"[ERR] audit parquet not found: {args.audit_parquet}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading audit parquet: {args.audit_parquet}")
    df = pd.read_parquet(args.audit_parquet)
    required_cols = {"episode_idx", "target_cos", "hardneg_gap"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERR] audit parquet missing required columns: {missing}",
              file=sys.stderr)
        return 2

    n_rows_total = len(df)
    n_rows_nan = df["target_cos"].isna().sum()
    if n_rows_nan > 0:
        print(f"[WARN] {n_rows_nan} rows have NaN target_cos "
              f"(missing celeb centroid OR embedding failure); "
              f"expected=valid score, got=NaN, fallback=excluded from episode means",
              flush=True)

    # Per-episode aggregation: mean of target_cos and hardneg_gap across frames
    per_episode = df.dropna(subset=["target_cos"]).groupby("episode_idx").agg(
        mean_target_cos=("target_cos", "mean"),
        mean_hardneg_gap=("hardneg_gap", "mean"),
        n_frames=("target_cos", "size"),
    ).reset_index()

    n_episodes_total = len(per_episode)
    print(f"[info] {n_rows_total} frame rows → {n_episodes_total} unique episodes")

    # --- Step 1: keep_episodes.txt ---
    keep_mask = per_episode["mean_target_cos"] >= args.keep_threshold
    n_kept = int(keep_mask.sum())
    retention = n_kept / max(n_episodes_total, 1)
    print(f"[info] keep_threshold={args.keep_threshold:.2f}: "
          f"{n_kept}/{n_episodes_total} episodes kept ({retention*100:.1f}%)")

    if retention < args.min_retention:
        print(f"[WARN] retention {retention*100:.1f}% < min_retention "
              f"{args.min_retention*100:.0f}%: expected>=85% kept, "
              f"got={retention*100:.1f}%, fallback=abort - rerun with lower "
              f"--keep-threshold or investigate data quality",
              flush=True)
        return 3

    kept_ids = per_episode.loc[keep_mask, "episode_idx"].astype(int).tolist()
    kept_ids.sort()
    keep_path = args.output_dir / "keep_episodes.txt"
    keep_path.write_text("\n".join(str(i) for i in kept_ids) + "\n")
    print(f"[done] wrote {len(kept_ids)} episode IDs → {keep_path}")

    # --- Step 2: hardneg_weights.npy ---
    # Build a dense weight array indexed by episode_idx.
    # Episodes not in keep_list get weight 0 (they're filtered anyway).
    # Kept episodes get weight = HARD_WEIGHT if mean_hardneg_gap < threshold else 1.0
    max_episode_id = int(per_episode["episode_idx"].max())
    weights = np.zeros(max_episode_id + 1, dtype=np.float32)

    n_hard = 0
    for _, row in per_episode[keep_mask].iterrows():
        ep_id = int(row["episode_idx"])
        if row["mean_hardneg_gap"] < args.hardneg_gap_threshold:
            weights[ep_id] = args.hard_weight
            n_hard += 1
        else:
            weights[ep_id] = 1.0

    weights_path = args.output_dir / "hardneg_weights.npy"
    np.save(weights_path, weights)
    print(f"[done] wrote weights (shape={weights.shape}) → {weights_path}")
    print(f"  hard episodes ({args.hard_weight:.1f}x weight): "
          f"{n_hard}/{n_kept} ({n_hard/max(n_kept,1)*100:.1f}%)")

    # --- Step 3: summary JSON for downstream traceability ---
    summary = {
        "audit_parquet": str(args.audit_parquet),
        "keep_threshold": args.keep_threshold,
        "hardneg_gap_threshold": args.hardneg_gap_threshold,
        "hard_weight": args.hard_weight,
        "n_episodes_total": n_episodes_total,
        "n_episodes_kept": n_kept,
        "retention": retention,
        "n_hard": n_hard,
        "frac_hard_of_kept": n_hard / max(n_kept, 1),
    }
    summary_path = args.output_dir / "build_keep_list_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] summary → {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
