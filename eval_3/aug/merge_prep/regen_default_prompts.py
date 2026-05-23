#!/usr/bin/env python3
"""Regenerate every episode's task prompt as a deterministic default-bucket
paraphrase. Replaces the current mix (which still contains ref_only and
counterfactual variants leaked through from the v3 prompt mixture).

Rule: prompt[i] = PARAPHRASES[i % 5].format(name=target_celeb_name)

Where target_celeb_name is read from each source dir's reference.json
("Yann LeCun", "Snoop Dogg", etc.). Both base teleops and augmented
variants have this field.

After this:
  - tasks.parquet has at most 192 celebs × 5 paraphrases = 960 unique tasks
  - data parquets' task_index column points at the right deterministic prompt
  - episodes parquet 'tasks' column and stats/task_index/* are updated
  - info.json total_tasks is updated

Idempotent. Run after merge but before pushing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


# Matches eval_3/aug/generators/broad.py PROMPT_PARAPHRASES exactly.
PARAPHRASES = [
    "Place the coke on {name}.",
    "Put the coke on {name}.",
    "Place the can on the photo of {name}.",
    "Set the coke down on {name}'s picture.",
    "Put the can on {name}'s photo.",
]


def discover_in_merge_order(base_root: Path, aug_root: Path,
                              aug_pattern: str) -> list[Path]:
    """Mirror eval_3/scripts/data/merge_episodes.py:discover_episode_dirs() exactly."""
    base = sorted(p for p in base_root.iterdir()
                    if p.is_dir() and (p / "meta" / "info.json").is_file()
                    and (p / "reference.json").is_file()) if base_root.is_dir() else []
    aug = sorted(p for p in aug_root.iterdir()
                   if p.is_dir() and aug_pattern in p.name
                   and (p / "meta" / "info.json").is_file()) if aug_root.is_dir() else []
    return base + aug


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged-root", type=Path, required=True)
    p.add_argument("--base-root", type=Path,
                   default=Path.home() / "LeMonkey/datasets/eval3")
    p.add_argument("--aug-root", type=Path,
                   default=Path.home() / "LeMonkey/datasets/eval3_aug_v3_200celebs")
    p.add_argument("--aug-pattern", default="__var")
    args = p.parse_args()

    t_start = time.time()

    print("[1/5] discovering sources in merge order…", flush=True)
    src_dirs = discover_in_merge_order(args.base_root, args.aug_root,
                                         args.aug_pattern)
    n_eps = len(src_dirs)
    print(f"      {n_eps} sources", flush=True)

    print("[2/5] reading target_celeb_name + generating default prompts…",
          flush=True)
    prompts: list[str] = []
    missing = 0
    for i, sd in enumerate(src_dirs):
        ref = json.loads((sd / "reference.json").read_text())
        name = ref.get("target_celeb_name")
        if not name:
            missing += 1
            # Fallback: derive from target_celeb_full or target_celeb
            slug = ref.get("target_celeb_full") or ref.get("target_celeb", "")
            name = " ".join(w.capitalize() for w in slug.replace("-", "_").split("_"))
            if not name:
                raise SystemExit(f"no celeb name in {sd / 'reference.json'}")
        prompts.append(PARAPHRASES[i % 5].format(name=name))
        if (i + 1) % 2000 == 0:
            print(f"      {i+1}/{n_eps}", flush=True)
    if missing:
        print(f"      [WARN] {missing} sources missing target_celeb_name, used slug fallback",
              flush=True)

    unique = sorted(set(prompts))
    pidx = {p: i for i, p in enumerate(unique)}
    new_tidx = np.array([pidx[p] for p in prompts], dtype=np.int64)
    print(f"      {len(unique)} unique default prompts across {n_eps} episodes",
          flush=True)
    print(f"      sample: {unique[:5]}", flush=True)

    print("[3/5] rewriting tasks.parquet…", flush=True)
    pdf = pd.DataFrame({"task_index": list(range(len(unique)))},
                         index=pd.Index(unique, name="task"))
    pq.write_table(pa.Table.from_pandas(pdf),
                    args.merged_root / "meta" / "tasks.parquet")

    print("[4/5] rewriting data parquets…", flush=True)
    for df_path in sorted((args.merged_root / "data").rglob("*.parquet")):
        t = pq.read_table(df_path)
        ep_idx = t.column("episode_index").to_numpy()
        rebuilt = new_tidx[ep_idx]
        cols = [t.column(c) if c != "task_index"
                 else pa.array(rebuilt, type=t.schema.field("task_index").type)
                for c in t.column_names]
        pq.write_table(pa.Table.from_arrays(cols, names=t.column_names),
                        df_path, compression="snappy")
        print(f"      {df_path.name}: {len(t):,} rows", flush=True)

    print("[5/5] rewriting episodes parquet + info.json…", flush=True)
    for ep_path in sorted((args.merged_root / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(ep_path)
        ep_idx_col = t.column("episode_index").to_numpy()
        per_row_tasks = [[prompts[i]] for i in ep_idx_col]
        per_row_tidx = new_tidx[ep_idx_col].astype(np.float64)
        new_cols = {}
        for c in t.column_names:
            if c == "tasks":
                new_cols[c] = pa.array(per_row_tasks,
                                          type=t.schema.field("tasks").type)
            elif c.startswith("stats/task_index/"):
                stat_kind = c.split("/")[-1]
                length_arr = t.column("length").to_numpy()
                if stat_kind in ("min", "max", "mean", "q01", "q10", "q50",
                                  "q90", "q99"):
                    val = per_row_tidx
                elif stat_kind == "std":
                    val = np.zeros_like(per_row_tidx)
                elif stat_kind == "count":
                    val = length_arr.astype(np.int64)
                else:
                    val = t.column(c).to_numpy()
                orig_type = t.schema.field(c).type
                if pa.types.is_list(orig_type):
                    new_cols[c] = pa.array([[v] for v in val], type=orig_type)
                else:
                    new_cols[c] = pa.array(val, type=orig_type)
            else:
                new_cols[c] = t.column(c)
        pq.write_table(pa.Table.from_arrays(list(new_cols.values()),
                                              names=list(new_cols.keys())),
                        ep_path, compression="snappy")
        print(f"      {ep_path.name}: {len(t):,} eps", flush=True)

    info_path = args.merged_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    old = info.get("total_tasks", "?")
    info["total_tasks"] = len(unique)
    info_path.write_text(json.dumps(info, indent=2))
    print(f"      info.json total_tasks: {old} -> {len(unique)}", flush=True)

    print(f"\n==> done in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
