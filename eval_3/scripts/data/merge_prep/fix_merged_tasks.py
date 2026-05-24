#!/usr/bin/env python3
"""Post-merge fix for the eval3_merged dataset's tasks/prompts.

The augmentation pipeline wrote each variant's actual prompt to
`augmentation.json` (`prompt`, `prompt_bucket`) but never updated the
per-variant `meta/tasks.parquet` — which was a hardlink of the base
teleop's tasks.parquet (so all variants kept the base teleop's prompt
like "Place the coke on Yann LeCun."). After merge, total_tasks=3 and
every episode's task_index points to one of the 3 base prompts.

This script patches the **merged** dataset in place:

  1. Re-derive merge order (sorted base teleops + sorted augmented
     variants), matching eval_3/scripts/data/merge_episodes.py's
     discover_episode_dirs.
  2. For each merged ep, look up its correct prompt:
       - base teleop → reference.json["prompt"]  (already correct)
       - augmented variant → augmentation.json["prompt"]
  3. Build a new tasks.parquet with the union of unique prompts.
  4. Rewrite each chunk's data parquet so `task_index` points to the
     new global task indices.
  5. Rewrite meta/episodes/.../parquet so the `tasks` column carries
     the correct prompt string.
  6. Update meta/info.json's total_tasks count.

Idempotent: if the merged tasks.parquet already has more than 3 rows,
exits early (assumes already-fixed).

Usage:
    fix_merged_tasks.py [--merged datasets/eval3_merged]
                        [--base-root datasets/eval3]
                        [--aug-root  datasets/eval3_aug_v3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def discover_merge_order(base_root: Path, aug_root: Path) -> list[Path]:
    """Replicate eval_3/scripts/data/merge_episodes.discover_episode_dirs to
    recover the same (base sorted, then aug sorted) ordering used at merge time."""
    base = sorted(p for p in base_root.iterdir()
                    if p.is_dir() and (p / "meta" / "info.json").is_file()
                    and (p / "reference.json").is_file()) if base_root.is_dir() else []
    aug = sorted(p for p in aug_root.iterdir()
                   if p.is_dir() and "__var" in p.name
                   and (p / "meta" / "info.json").is_file()) if aug_root.is_dir() else []
    return base + aug


def correct_prompt_for(ep_dir: Path) -> str:
    """Look up the intended prompt for an episode dir.
       - If augmentation.json exists, use its `prompt` field (augmented variant)
       - Otherwise fall back to reference.json `prompt` (base teleop)
    """
    aug_json = ep_dir / "augmentation.json"
    if aug_json.is_file():
        return json.loads(aug_json.read_text())["prompt"]
    ref_json = ep_dir / "reference.json"
    if ref_json.is_file():
        return json.loads(ref_json.read_text())["prompt"]
    # Last resort — read the source tasks.parquet
    return pd.read_parquet(ep_dir / "meta" / "tasks.parquet").index[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--merged", type=Path, default=Path("datasets/eval3_merged"),
                   help="Path to the merged LeRobot v3 dataset whose tasks.parquet will be fixed")
    p.add_argument("--base-root", type=Path, default=Path("datasets/eval3"),
                   help="Root containing the base teleop directories used to source prompts")
    p.add_argument("--aug-root", type=Path, default=Path("datasets/eval3_aug_v3"),
                   help="Root containing the augmented variant directories used to source prompts")
    args = p.parse_args()

    merged = args.merged
    tasks_path = merged / "meta" / "tasks.parquet"
    info_path = merged / "meta" / "info.json"

    # 0. Idempotency check
    current_tasks = pd.read_parquet(tasks_path)
    if len(current_tasks) > 3:
        print(f"  tasks.parquet has {len(current_tasks)} rows already — "
              f"looks fixed, exiting", flush=True)
        return 0
    print(f"  before fix: tasks.parquet has {len(current_tasks)} unique tasks",
          flush=True)

    # 1. Replay merge order
    order = discover_merge_order(args.base_root, args.aug_root)
    info = json.loads(info_path.read_text())
    total_eps = info["total_episodes"]
    if len(order) != total_eps:
        print(f"  [WARN] merge order has {len(order)} eps but merged dataset "
              f"has {total_eps}. Mismatch suggests merge order changed "
              f"since merge time", file=sys.stderr)
        if len(order) < total_eps:
            return 2
    print(f"  re-derived merge order: {len(order)} ep dirs (base + aug)",
          flush=True)

    # 2. Build merged_ep_idx → correct_prompt
    ep_idx_to_prompt: dict[int, str] = {}
    for i, ep_dir in enumerate(order[:total_eps]):
        ep_idx_to_prompt[i] = correct_prompt_for(ep_dir)
        if i % 1000 == 0:
            print(f"    {i}/{total_eps} prompts looked up", flush=True)

    # 3. Compute unique prompts → global task_index assignment
    unique_prompts = sorted(set(ep_idx_to_prompt.values()))
    prompt_to_idx = {p: i for i, p in enumerate(unique_prompts)}
    print(f"  unique prompts after fix: {len(unique_prompts)}", flush=True)

    # 4. Write new tasks.parquet
    new_tasks = pd.DataFrame(
        {"task_index": [prompt_to_idx[p] for p in unique_prompts]},
        index=pd.Index(unique_prompts, name="task"),
    )
    new_tasks.to_parquet(tasks_path)
    print(f"  wrote new tasks.parquet ({len(new_tasks)} rows)", flush=True)

    # 5. Patch every data chunk's task_index column.
    data_dir = merged / "data"
    chunks = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    print(f"  patching {len(chunks)} data parquets...", flush=True)
    for ci, dp in enumerate(chunks):
        df = pd.read_parquet(dp)
        ep_col = df["episode_index"].to_numpy()
        new_idx = [prompt_to_idx[ep_idx_to_prompt[int(e)]] for e in ep_col]
        df["task_index"] = new_idx
        df.to_parquet(dp, index=False)
        print(f"    [{ci+1}/{len(chunks)}] {dp.name}: rows={len(df)} "
              f"unique_new_idx={len(set(new_idx))}", flush=True)

    # 6. Patch the meta/episodes/.../parquet `tasks` column.
    ep_meta = sorted((merged / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    print(f"  patching {len(ep_meta)} meta/episodes parquets...", flush=True)
    for em in ep_meta:
        df = pd.read_parquet(em)
        new_tasks_col = [[ep_idx_to_prompt[int(e)]]
                          for e in df["episode_index"].to_numpy()]
        df["tasks"] = new_tasks_col
        df.to_parquet(em, index=False)
        print(f"    {em.name}: rows={len(df)} patched", flush=True)

    # 7. Update info.json total_tasks
    info["total_tasks"] = len(unique_prompts)
    info_path.write_text(json.dumps(info, indent=4))
    print(f"  updated meta/info.json total_tasks={len(unique_prompts)}",
          flush=True)

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
