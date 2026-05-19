#!/usr/bin/env python
"""
Compute the list of TA-compliant (text-only) episode indices from
HBOrtiz/so101_eval3_all. Output is a JSON file you pass to lerobot-train.

What gets DROPPED: any episode whose task prompt contains "reference" or
"whoever" — those are ref-only prompts that need a reference image at
inference, which the TA banned on 2026-05-18.

Concretely (per our inspection on 2026-05-19):
  4,195 total episodes
   −595 ref-only (3 templates: "the celebrity shown in the reference image",
                  "the person in the reference image",
                  "whoever is in the reference photo")
  = 3,600 text-only episodes kept

Run (anywhere with HF read access to HBOrtiz/so101_eval3_all):
    python eval_3/scripts/filter_eval3_all_episodes.py \
        --out keep_indices_eval3_all_textonly.json

Then on the H100, pass to lerobot-train:
    INDICES=$(cat keep_indices_eval3_all_textonly.json)
    lerobot-train \
        --dataset.repo_id=HBOrtiz/so101_eval3_all \
        --dataset.episodes="$INDICES" \
        ...
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

DATASET_REPO = "HBOrtiz/so101_eval3_all"
# Keywords that mark a prompt as ref-only / TA-disallowed.
# Match is case-insensitive substring on the task text.
REF_KEYWORDS = ("reference", "whoever")


def fetch_episodes_parquet(cache_dir: Path | None) -> Path:
    """Download just the meta/episodes parquet from HF, return the parquet path."""
    print(f"[fetch] downloading meta/episodes/* from {DATASET_REPO}...")
    local = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=["meta/episodes/**"],
        local_dir=str(cache_dir) if cache_dir else None,
    )
    # Find the parquet (chunks_size=1000 → likely meta/episodes/chunk-000/file-000.parquet)
    candidates = list(Path(local).rglob("meta/episodes/**/*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no episodes parquet under {local}/meta/episodes/")
    return candidates[0]


def is_text_only(task: str) -> bool:
    t = task.lower()
    return not any(kw in t for kw in REF_KEYWORDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSON file: KEEP indices (text-only episodes)")
    parser.add_argument("--drop-out", type=Path, default=None,
                        help="Optional: also write the INVERSE (drop indices) to this JSON. "
                             "Useful for `lerobot-edit-dataset delete_episodes`.")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Where to cache the downloaded meta (default: HF cache)")
    args = parser.parse_args()

    parquet_path = fetch_episodes_parquet(args.cache_dir)
    print(f"[load] {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df["task"] = df["tasks"].apply(lambda x: x[0])

    n_total = len(df)
    keep_mask = df["task"].apply(is_text_only)
    keep_indices = df.loc[keep_mask, "episode_index"].astype(int).tolist()
    dropped = df.loc[~keep_mask, ["episode_index", "task"]]

    print()
    print(f"=== Filter summary ===")
    print(f"Total episodes:        {n_total}")
    print(f"Keep (text-only):      {len(keep_indices)}  ({100*len(keep_indices)/n_total:.1f}%)")
    print(f"Drop (ref-only):       {len(dropped)}  ({100*len(dropped)/n_total:.1f}%)")
    print()
    print(f"=== Dropped prompt templates (by frequency) ===")
    for task, n in dropped["task"].value_counts().items():
        print(f"  ({n:>3} eps) {task!r}")

    # Write keep_indices as a single JSON list — lerobot-train --dataset.episodes
    # expects a Python-style list literal, which JSON arrays parse as.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(keep_indices, f)
    size_kb = args.out.stat().st_size / 1024
    print()
    print(f"[write] {len(keep_indices)} keep-indices → {args.out}  ({size_kb:.1f} KB)")

    if args.drop_out:
        drop_indices = dropped["episode_index"].astype(int).tolist()
        args.drop_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.drop_out, "w") as f:
            json.dump(drop_indices, f)
        print(f"[write] {len(drop_indices)} drop-indices → {args.drop_out}")
    print()
    print(f"=== Usage in lerobot-train ===")
    print(f'  INDICES=$(cat {args.out})')
    print(f'  lerobot-train \\')
    print(f'      --dataset.repo_id={DATASET_REPO} \\')
    print(f'      --dataset.episodes="$INDICES" \\')
    print(f'      ...')


if __name__ == "__main__":
    main()
