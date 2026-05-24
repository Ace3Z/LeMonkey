#!/usr/bin/env python3
"""Validate the bbox parquet schema before running the audit.

A parquet of per-frame face bboxes for the 192-celeb dataset.
The exact column names may vary slightly from our assumed schema; this script
checks BEFORE we burn 1 h on the ArcFace audit only to find the columns
weren't what we expected.

Expected schema (per the ObjectVLA spec + arcface_audit_200celeb.py input
contract):

    episode_idx        (int64)   — episode index in HBOrtiz/so101_eval3_aug_v3_200celebs
    frame_idx          (int64)   — frame within the episode
    bbox_x1, bbox_y1, bbox_x2, bbox_y2  (float)  — pixel coords on 480×640 camera1
    target_celeb       (str)     — celeb slug or name (will be resolved via
                                   task_index_to_centroid.json if needed)

Acceptable alternate column naming (we'll auto-adapt):
    episode_idx OR episode_index OR ep_idx
    frame_idx   OR frame_index   OR frame
    bbox_x1/y1/x2/y2 OR x1/y1/x2/y2 OR bbox (list[4])

Per: emit [WARN] with what we expected vs what we got; never
silently rename / drop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Aliases for flexible schema matching (the upstream source may use different conventions).
EPISODE_COL_ALIASES = {"episode_idx", "episode_index", "ep_idx", "ep"}
FRAME_COL_ALIASES = {"frame_idx", "frame_index", "frame"}
BBOX_X1_ALIASES = {"bbox_x1", "x1", "xmin"}
BBOX_Y1_ALIASES = {"bbox_y1", "y1", "ymin"}
BBOX_X2_ALIASES = {"bbox_x2", "x2", "xmax"}
BBOX_Y2_ALIASES = {"bbox_y2", "y2", "ymax"}
BBOX_COMBINED_ALIASES = {"bbox", "bbox_xyxy"}
TARGET_CELEB_ALIASES = {"target_celeb", "target", "celeb", "celeb_slug", "name"}


def find_col(cols, aliases: set[str]) -> str | None:
    """Find the first column in `cols` whose lower-cased name is in `aliases`."""
    for c in cols:
        if c.lower() in aliases:
            return c
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-parquet", type=Path, required=True,
                        help="the per-frame bbox annotations")
    parser.add_argument("--dataset-info", type=Path,
                        default=Path("data/200celebs/meta/info.json"),
                        help="192-celeb info.json for episode range validation")
    parser.add_argument("--write-normalized", type=Path, default=None,
                        help="If set, write a normalized-column version to this path")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("[ERR] pandas required", file=sys.stderr)
        return 2

    if not args.bbox_parquet.is_file():
        print(f"[ERR] bbox parquet not found: {args.bbox_parquet}", file=sys.stderr)
        return 2

    print(f"[info] loading bbox parquet: {args.bbox_parquet}")
    df = pd.read_parquet(args.bbox_parquet)
    print(f"[info] rows: {len(df)}, columns: {list(df.columns)}")

    issues = []

    # Required: episode + frame.
    ep_col = find_col(df.columns, EPISODE_COL_ALIASES)
    if not ep_col:
        issues.append(f"no episode column found in any of {EPISODE_COL_ALIASES}")
    frame_col = find_col(df.columns, FRAME_COL_ALIASES)
    if not frame_col:
        issues.append(f"no frame column found in any of {FRAME_COL_ALIASES}")

    # Required: bbox (either 4 separate cols or 1 combined).
    bbox_combined = find_col(df.columns, BBOX_COMBINED_ALIASES)
    bbox_x1 = find_col(df.columns, BBOX_X1_ALIASES)
    bbox_y1 = find_col(df.columns, BBOX_Y1_ALIASES)
    bbox_x2 = find_col(df.columns, BBOX_X2_ALIASES)
    bbox_y2 = find_col(df.columns, BBOX_Y2_ALIASES)
    if not bbox_combined and not all([bbox_x1, bbox_y1, bbox_x2, bbox_y2]):
        issues.append(f"no bbox columns — need 'bbox' (list[4]) OR "
                      f"all of x1/y1/x2/y2 (or aliases)")

    # Required: target celeb.
    target_col = find_col(df.columns, TARGET_CELEB_ALIASES)
    if not target_col:
        issues.append(f"no target celeb column found in any of {TARGET_CELEB_ALIASES}")

    if issues:
        print("\n[FAIL] schema validation failed:")
        for i in issues:
            print(f"  - {i}")
        print(f"\n[WARN] expected schema documented in arcface_audit_200celeb.py "
              f"docstring. the upstream source may need column renames or extend the audit "
              f"script's column resolution.")
        return 1

    print(f"\n[info] schema resolved:")
    print(f"  episode: {ep_col}")
    print(f"  frame:   {frame_col}")
    if bbox_combined:
        print(f"  bbox:    {bbox_combined} (combined list)")
    else:
        print(f"  bbox:    {bbox_x1}, {bbox_y1}, {bbox_x2}, {bbox_y2}")
    print(f"  target:  {target_col}")

    # Sanity checks.
    n_episodes = df[ep_col].nunique()
    print(f"\n[info] {n_episodes} unique episodes in bbox parquet")

    if args.dataset_info.is_file():
        import json
        info = json.loads(args.dataset_info.read_text())
        expected_eps = info.get("total_episodes")
        if expected_eps and n_episodes != expected_eps:
            print(f"[WARN] episode-count mismatch: bbox parquet has {n_episodes}, "
                  f"dataset has {expected_eps}, fallback=audit will skip episodes "
                  f"without bbox rows (expect target_cos=NaN → filtered out)",
                  flush=True)
    else:
        print(f"[WARN] dataset info.json not at {args.dataset_info}; cannot verify "
              f"episode-count match", flush=True)

    # Per-frame consistency: max frame_idx within an episode should match episode length.
    max_frame_per_ep = df.groupby(ep_col)[frame_col].max()
    print(f"  frames per episode: min={max_frame_per_ep.min()}, "
          f"max={max_frame_per_ep.max()}, mean={max_frame_per_ep.mean():.1f}")
    # Most 192-celeb episodes are 538 frames; bboxes should cover most/all.
    if max_frame_per_ep.max() > 600:
        print(f"[WARN] max frame_idx={max_frame_per_ep.max()}: expected<=538 per "
              f"192-celeb info.json, fallback=audit will silently process them",
              flush=True)

    # Target celeb sanity.
    n_unique_targets = df[target_col].nunique()
    sample_targets = df[target_col].value_counts().head(10)
    print(f"\n[info] {n_unique_targets} unique target celebs; top-10 by frequency:")
    for celeb, count in sample_targets.items():
        print(f"    {celeb}: {count}")

    # Optionally write normalized-schema parquet so the audit script can be called
    # with consistent column names regardless of the choices.
    if args.write_normalized:
        rename_map = {}
        if ep_col != "episode_idx":
            rename_map[ep_col] = "episode_idx"
        if frame_col != "frame_idx":
            rename_map[frame_col] = "frame_idx"
        if target_col != "target_celeb":
            rename_map[target_col] = "target_celeb"
        if bbox_combined and bbox_combined != "bbox":
            rename_map[bbox_combined] = "bbox"
        elif not bbox_combined:
            for alias_col, target in [(bbox_x1, "bbox_x1"), (bbox_y1, "bbox_y1"),
                                       (bbox_x2, "bbox_x2"), (bbox_y2, "bbox_y2")]:
                if alias_col and alias_col != target:
                    rename_map[alias_col] = target
        normalized = df.rename(columns=rename_map)
        # If combined bbox, split into 4 columns.
        if bbox_combined and "bbox" in normalized.columns:
            normalized["bbox_x1"] = normalized["bbox"].apply(lambda b: b[0])
            normalized["bbox_y1"] = normalized["bbox"].apply(lambda b: b[1])
            normalized["bbox_x2"] = normalized["bbox"].apply(lambda b: b[2])
            normalized["bbox_y2"] = normalized["bbox"].apply(lambda b: b[3])
        normalized.to_parquet(args.write_normalized, index=False)
        print(f"\n[done] normalized parquet → {args.write_normalized}")

    print(f"\n[PASS] schema OK — can feed into arcface_audit_200celeb.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
