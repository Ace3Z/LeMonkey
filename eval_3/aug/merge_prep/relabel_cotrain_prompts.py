#!/usr/bin/env python3
"""Relabel cotrain aug variant prompts before merging.

PROBLEM: During augmentation, each variant's `meta/tasks.parquet` and
`meta/episodes/.../file-000.parquet` were hardlinked from the base
teleop — so they all carry the BASE teleop's prompt (e.g. "Place the
coke on Yann LeCun.") regardless of the variant's actual target celeb.
The per-variant correct prompt lives only in `augmentation.json` and
`reference.json`.

Compounding: with the TA's 2026-05-18 text-only ruling, the 15% ref-only
prompts ("Place the coke on the person in the reference image.") and
10% counterfactual prompts ("Place the coke on Yann LeCun." with
target=Swift) are now INVALID — they assumed a reference channel.

THIS SCRIPT, per variant:
  1. Re-derives a default-bucket prompt from the variant's
     `target_celeb_full` (or `new_target_short` → SHORT_TO_FULL[…]).
     Drops the ref_only + counterfactual buckets entirely.
  2. Breaks the hardlinks on `meta/tasks.parquet` +
     `meta/episodes/chunk-000/file-000.parquet` (so we don't corrupt
     the base teleop's meta).
  3. Writes the new prompt into those two parquet files.
  4. Updates `augmentation.json["prompt"]/["prompt_bucket"]` and
     `reference.json["prompt"]/["prompt_bucket"]` for traceability.

Per-base-teleop prompts are left untouched — they already use the
default-bucket format and don't carry the bucket-mixing issue.

After this runs, eval_3/scripts/data/merge_episodes.py can aggregate
cleanly; the global tasks.parquet will deduplicate identical paraphrase
strings.

Usage:
    relabel_cotrain_prompts.py --aug-root datasets/eval3_aug_cotrain
    relabel_cotrain_prompts.py --dry-run                       # report only
    relabel_cotrain_prompts.py --limit 5                       # smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

DEFAULT_PARAPHRASES = [
    "Place the coke on {name}.",
    "Put the coke on {name}.",
    "Place the can on the photo of {name}.",
    "Set the coke down on {name}'s picture.",
    "Put the can on {name}'s photo.",
]
SHORT_TO_FULL_NAME = {
    "swift": "Taylor Swift",
    "obama": "Barack Obama",
    "lecun": "Yann LeCun",
}


def derive_target_name(sidecar_aug: dict) -> str:
    """Read the target celeb from augmentation.json and return the
    human-readable full name."""
    # The cotrain aug pipeline writes `new_target_short` AND `target_celeb_full`;
    # be robust to either being present.
    full = sidecar_aug.get("target_celeb_full") or sidecar_aug.get("target_celeb_name")
    if full and full in ("Taylor Swift", "Barack Obama", "Yann LeCun"):
        return full
    short = sidecar_aug.get("new_target_short") or sidecar_aug.get("target_short") or sidecar_aug.get("target_celeb")
    if short in SHORT_TO_FULL_NAME:
        return SHORT_TO_FULL_NAME[short]
    raise ValueError(f"cannot derive target name from sidecar keys: {list(sidecar_aug.keys())}")


def relabel_one_variant(var_dir: Path, dry_run: bool) -> tuple[bool, str]:
    """Returns (ok, new_prompt). On dry_run, only computes; does not write."""
    aug_json = var_dir / "augmentation.json"
    ref_json = var_dir / "reference.json"
    tasks_pq = var_dir / "meta" / "tasks.parquet"
    eps_pq   = var_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"

    if not aug_json.is_file():
        return False, f"missing augmentation.json"
    if not tasks_pq.is_file():
        return False, f"missing meta/tasks.parquet"
    if not eps_pq.is_file():
        return False, f"missing meta/episodes/.../file-000.parquet"

    aug = json.loads(aug_json.read_text())
    try:
        target_name = derive_target_name(aug)
    except ValueError as e:
        return False, str(e)

    # Deterministic per-variant paraphrase pick.
    rng = random.Random(hash(var_dir.name) % (2**32))
    new_prompt = rng.choice(DEFAULT_PARAPHRASES).format(name=target_name)

    if dry_run:
        return True, new_prompt

    # 1. tasks.parquet — overwrite the single-row table with our prompt.
    tasks_pq.unlink()       # break hardlink to base teleop
    pd.DataFrame({"task_index": [0], "task": [new_prompt]}) \
       .set_index("task") \
       .to_parquet(tasks_pq)

    # 2. episodes/.../file-000.parquet — update the `tasks` column.
    eps_df = pd.read_parquet(eps_pq)
    eps_df["tasks"] = eps_df["tasks"].apply(lambda _: [new_prompt])
    eps_pq.unlink()          # break hardlink
    eps_df.to_parquet(eps_pq)

    # 3. sidecars for traceability
    aug["prompt"] = new_prompt
    aug["prompt_bucket"] = "default"
    aug_json.write_text(json.dumps(aug, indent=2))
    if ref_json.is_file():
        ref = json.loads(ref_json.read_text())
        ref["prompt"] = new_prompt
        ref["prompt_bucket"] = "default"
        ref_json.write_text(json.dumps(ref, indent=2))

    return True, new_prompt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aug-root", type=Path,
                   default=Path("/home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3_aug_cotrain"))
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would change; don't write")
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke: only process the first N variants")
    args = p.parse_args()

    var_dirs = sorted(p for p in args.aug_root.iterdir()
                       if p.is_dir() and "__t3_" in p.name)
    if args.limit:
        var_dirs = var_dirs[:args.limit]
    print(f"found {len(var_dirs)} cotrain aug variants under {args.aug_root}", flush=True)
    if not var_dirs:
        return 1

    ok = 0
    errs = []
    prompt_counter: dict[str, int] = {}
    for i, vd in enumerate(var_dirs):
        success, msg = relabel_one_variant(vd, args.dry_run)
        if success:
            ok += 1
            prompt_counter[msg] = prompt_counter.get(msg, 0) + 1
        else:
            errs.append((vd.name, msg))
        if (i + 1) % 500 == 0 or (i + 1) == len(var_dirs):
            print(f"  [{i+1}/{len(var_dirs)}] ok={ok} errs={len(errs)}", flush=True)

    print(f"\n{'(dry-run) ' if args.dry_run else ''}done. ok={ok}/{len(var_dirs)}  errors={len(errs)}", flush=True)
    print("\nprompt distribution (top 10 by count):")
    for prompt, n in sorted(prompt_counter.items(), key=lambda t: -t[1])[:10]:
        print(f"  {n:>5}  {prompt}")
    if errs:
        print("\nfirst 10 errors:")
        for name, msg in errs[:10]:
            print(f"  {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
