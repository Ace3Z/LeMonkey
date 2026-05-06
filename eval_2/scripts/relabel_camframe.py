#!/usr/bin/env python3
"""Relabel Eval 2 prompts from user-frame to camera-frame.

Background: the SO-101's tripod-mounted camera looks at the workspace from
the opposite side of the user, so what the user calls "left" is what the
camera sees on the image's "right". Prompts collected with the user-frame
convention fight SmolVLM2's pretraining prior (which expects image-left ↔
"left"). To align the convention with the image, we swap left ↔ right
tokens in every prompt. Center / colour words are unchanged.

The trajectories themselves are correct in either convention — the demo's
arm motion is unchanged, only the language label needs flipping.

Targets:
  • {dataset_root}/meta/tasks.parquet   — the file that defines the 123
    unique prompts indexed by task_index. Every frame in the data parquets
    refers to one of these rows; rewriting the strings is sufficient.

Usage:
    relabel_camframe.py                         # dry-run, prints diffs
    relabel_camframe.py --apply                 # write the file in place
    relabel_camframe.py --apply --push-to-hub   # also push the changed
                                                  file to HBOrtiz/so101_eval2_all
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

DEFAULT_DS = Path("/home/lemonkey/LeMonkey/datasets/eval2_merged")
HF_REPO = "HBOrtiz/so101_eval2_all"


def relabel(prompt: str) -> str:
    """Swap 'left' ↔ 'right' substrings, case-preserving."""
    def swap(m: re.Match) -> str:
        s = m.group(0)
        target = "right" if s.lower() == "left" else "left"
        return target[0].upper() + target[1:] if s[0].isupper() else target
    return re.sub(r"(?i)left|right", swap, prompt)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DS)
    p.add_argument("--apply", action="store_true",
                   help="Write the relabeled tasks.parquet in place (after backing up).")
    p.add_argument("--push-to-hub", action="store_true",
                   help=f"After --apply, push the new file to {HF_REPO}.")
    args = p.parse_args()

    tasks_p = args.dataset / "meta" / "tasks.parquet"
    if not tasks_p.exists():
        print(f"ERROR: {tasks_p} not found", file=sys.stderr)
        return 1

    df = pd.read_parquet(tasks_p).reset_index()
    if list(df.columns) != ["task", "task_index"]:
        print(f"ERROR: unexpected columns {list(df.columns)}, expected ['task','task_index']", file=sys.stderr)
        return 1

    df["new_task"] = df["task"].apply(relabel)
    n_changed = int((df["task"] != df["new_task"]).sum())
    n_unchanged = len(df) - n_changed

    weird = df[
        (df["task"] != df["new_task"])
        & (~df["task"].str.contains(r"(?i)left|right"))
    ]
    if not weird.empty:
        print(f"ERROR: {len(weird)} rows changed but contain no 'left'/'right' tokens — bug in regex", file=sys.stderr)
        for _, r in weird.iterrows():
            print(f"  - {r['task']!r} → {r['new_task']!r}", file=sys.stderr)
        return 1

    n_distinct_after = df["new_task"].nunique()

    print(f"=== Relabel summary ===")
    print(f"  total prompts    : {len(df)}")
    print(f"  changed          : {n_changed}")
    print(f"  unchanged        : {n_unchanged}")
    print(f"  distinct after   : {n_distinct_after}")
    print(f"  apply            : {args.apply}")
    print(f"  push to hub      : {args.push_to_hub}")
    print()

    if not args.apply:
        print("=== sample diffs (first 10 changed) ===")
        for _, r in df[df["task"] != df["new_task"]].head(10).iterrows():
            print(f"  - {r['task']}")
            print(f"  + {r['new_task']}")
            print()
        print("Pass --apply to write the file. Pass --push-to-hub to also upload.")
        return 0

    # 1. Back up the original
    backup = tasks_p.with_suffix(".parquet.userframe.bak")
    if not backup.exists():
        shutil.copy2(tasks_p, backup)
        print(f"  backed up original → {backup}")
    else:
        print(f"  backup already exists at {backup} (not overwriting)")

    # 2. Rewrite tasks.parquet with new strings, keeping task_index order
    new_df = df[["new_task", "task_index"]].rename(columns={"new_task": "task"})
    new_df = new_df.set_index("task")
    new_df.to_parquet(tasks_p, index=True)

    # 3. Sanity check the round-trip
    rt = pd.read_parquet(tasks_p).reset_index()
    assert list(rt.columns) == ["task", "task_index"]
    assert len(rt) == len(df)
    print(f"  ✓ wrote {tasks_p} ({len(rt)} rows, schema preserved)")

    if args.push_to_hub:
        from huggingface_hub import HfApi
        print(f"  uploading to {HF_REPO} ...")
        HfApi().upload_file(
            path_or_fileobj=str(tasks_p),
            path_in_repo="meta/tasks.parquet",
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message="Relabel prompts: swap left↔right (camera-frame convention)",
        )
        print(f"  ✓ pushed to {HF_REPO}/meta/tasks.parquet")

    return 0


if __name__ == "__main__":
    sys.exit(main())
