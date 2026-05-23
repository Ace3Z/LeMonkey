#!/usr/bin/env python3
"""Parse the 200-celeb dataset's task strings → map task_index to celeb slug + centroid.

Walks `data/200celebs/meta/tasks.parquet` (960 unique task strings indexed 0..959),
extracts the celeb name from each task string, normalizes to a slug, and looks
up the 512-d centroid from Mahbod's celeb_embeddings.json.

Output: `task_index_to_centroid.json` keyed by task_index:
  {
    "0": {"task": "Place the can on the photo of Adam Sandler.",
          "celeb_name": "Adam Sandler", "celeb_slug": "adam_sandler",
          "centroid_path": "celeb_embeddings.json#celebs.adam_sandler.centroid",
          "centroid_ok": true},
    ...
  }

This decouples the audit script from re-parsing tasks per frame.

Per CLAUDE.md §5: no silent fallbacks — every parse failure or missing-centroid
case emits a [WARN] with context.
Per CLAUDE.md §7: triple-source defaults inline.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Observed task-string patterns in HBOrtiz/so101_eval3_aug_v3_200celebs/meta/tasks.parquet:
#   "Place the can on the photo of Adam Sandler."        (~40% of variants)
#   "Place the coke on Yann LeCun."                       (some IID variants)
#   "Place the can on Andy Jassy."                        (mixed)
#   "Set the coke down on Steve Jobs's picture."          (~60% of variants)
#
# Multiple patterns tried in order; first match wins. The celeb name is always
# in group(1) of the matching pattern.
TASK_PATTERNS = [
    # POSSESSIVE FORMS FIRST — otherwise the bare patterns capture "X's photo" as name.
    # "Place the can on X's photo." / "Place the coke on X's picture."
    re.compile(
        r"^Place the (?:can|coke) on (.+?)'s (?:photo|picture)\.?\s*$",
        re.IGNORECASE,
    ),
    # "Set the coke down on X's picture." / "Set the coke down on X's photo."
    re.compile(
        r"^Set the (?:can|coke) down on (.+?)'s (?:photo|picture)\.?\s*$",
        re.IGNORECASE,
    ),
    # "Put the can on X's photo."  (defensive)
    re.compile(
        r"^Put the (?:can|coke) on (.+?)'s (?:photo|picture)\.?\s*$",
        re.IGNORECASE,
    ),
    # NON-POSSESSIVE FORMS — try AFTER the possessive ones.
    # "Place the can on the photo of X."
    re.compile(
        r"^Place the (?:can|coke) on the photo of (.+?)\.?\s*$",
        re.IGNORECASE,
    ),
    # "Place the can on X." / "Place the coke on X."
    re.compile(
        r"^Place the (?:can|coke) on (.+?)\.?\s*$",
        re.IGNORECASE,
    ),
    # "Put the can on X." (defensive)
    re.compile(
        r"^Put the (?:can|coke) on (?:the photo of )?(.+?)\.?\s*$",
        re.IGNORECASE,
    ),
]


# Manual overrides for celebs whose Title-Case name doesn't slug to their manifest key.
# (e.g., "LeBron James" → "lebron_james" works auto; "BJ Novak" might need this)
# Populate empirically from a first run's [WARN] log if any.
NAME_OVERRIDES: dict[str, str] = {
    # Add as needed when slug normalization fails.
}


def name_to_slug(name: str) -> str:
    """Normalize 'Adam Sandler' → 'adam_sandler', 'Anya Taylor-Joy' → 'anya_taylor-joy'."""
    if name in NAME_OVERRIDES:
        return NAME_OVERRIDES[name]
    # lowercase, strip, split on whitespace → underscore. Preserve hyphens (Taylor-Joy).
    return "_".join(name.strip().lower().split())


def parse_task_string(task: str) -> str | None:
    """Return the celeb name (Title Case) or None if no pattern matches."""
    s = task.strip()
    for pattern in TASK_PATTERNS:
        m = pattern.match(s)
        if m:
            return m.group(1).strip()
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-parquet", type=Path,
                        default=Path("data/200celebs/meta/tasks.parquet"),
                        help="200-celeb dataset's meta/tasks.parquet")
    parser.add_argument("--celeb-manifest", type=Path,
                        default=Path("data/arcface_toolkit/celeb_embeddings.json"),
                        help="Mahbod's celeb_embeddings.json")
    parser.add_argument("--output", type=Path,
                        default=Path("eval_3/scripts/pi05_vl_cotrain/task_index_to_centroid.json"))
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("[ERR] pandas required", file=sys.stderr)
        return 2

    if not args.tasks_parquet.is_file():
        print(f"[ERR] tasks.parquet not found: {args.tasks_parquet}", file=sys.stderr)
        return 2
    if not args.celeb_manifest.is_file():
        print(f"[ERR] celeb manifest not found: {args.celeb_manifest}", file=sys.stderr)
        return 2

    # Load tasks: the parquet has task strings AS THE INDEX, task_index as a column.
    df = pd.read_parquet(args.tasks_parquet)
    if "task_index" not in df.columns:
        print(f"[ERR] tasks.parquet missing task_index column. Got: {list(df.columns)}",
              file=sys.stderr)
        return 2
    df = df.reset_index()  # task string is now a column
    if "task" not in df.columns:
        # The index column name may differ (could be `__index_level_0__` etc.)
        # Take whatever the first non-task_index column is.
        non_task_cols = [c for c in df.columns if c != "task_index"]
        if not non_task_cols:
            print(f"[ERR] cannot find task string column after reset_index", file=sys.stderr)
            return 2
        df = df.rename(columns={non_task_cols[0]: "task"})

    print(f"[info] loaded {len(df)} task strings from {args.tasks_parquet}")

    # Load centroids.
    manifest = json.loads(args.celeb_manifest.read_text())
    available_slugs = set(manifest["celebs"].keys())
    print(f"[info] manifest has {len(available_slugs)} celebs with entries")

    # Parse each task → slug → centroid.
    output: dict[str, dict] = {}
    n_parsed = 0
    n_unparsed = 0
    n_missing_centroid = 0
    n_broken_centroid = 0
    unique_slugs_seen: set[str] = set()

    # Known broken centroid from Mahbod's audit (oier_mees own-photo cosines too low).
    BROKEN_SLUGS = {"oier_mees"}

    for _, row in df.iterrows():
        task_str = row["task"]
        ti = int(row["task_index"])

        celeb_name = parse_task_string(task_str)
        if celeb_name is None:
            print(f"[WARN] task_index={ti}: expected=parseable 'Place the can on X.' pattern, "
                  f"got={task_str!r}, fallback=skip (no centroid mapping)",
                  flush=True)
            output[str(ti)] = {"task": task_str, "celeb_name": None,
                               "celeb_slug": None, "centroid_ok": False,
                               "reason": "unparseable"}
            n_unparsed += 1
            continue

        slug = name_to_slug(celeb_name)
        unique_slugs_seen.add(slug)
        n_parsed += 1

        if slug not in available_slugs:
            print(f"[WARN] task_index={ti}: expected=slug={slug!r} in manifest, "
                  f"got=missing, fallback=skip (celeb_name={celeb_name!r})",
                  flush=True)
            output[str(ti)] = {"task": task_str, "celeb_name": celeb_name,
                               "celeb_slug": slug, "centroid_ok": False,
                               "reason": "slug_not_in_manifest"}
            n_missing_centroid += 1
            continue

        info = manifest["celebs"][slug]
        if info.get("centroid") is None:
            print(f"[WARN] task_index={ti}: slug={slug!r} in manifest but centroid=None, "
                  f"fallback=skip", flush=True)
            output[str(ti)] = {"task": task_str, "celeb_name": celeb_name,
                               "celeb_slug": slug, "centroid_ok": False,
                               "reason": "null_centroid"}
            n_missing_centroid += 1
            continue

        is_broken = slug in BROKEN_SLUGS
        if is_broken:
            n_broken_centroid += 1

        output[str(ti)] = {
            "task": task_str,
            "celeb_name": celeb_name,
            "celeb_slug": slug,
            "centroid_ok": not is_broken,
            "reason": "broken_centroid" if is_broken else "ok",
            "n_photos": info.get("n_photos", 0),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))

    print(f"\n[summary]")
    print(f"  total task_indices:     {len(df)}")
    print(f"  parsed celeb name:      {n_parsed}")
    print(f"  unparseable tasks:      {n_unparsed}")
    print(f"  unique celeb slugs:     {len(unique_slugs_seen)}")
    print(f"  slugs IN manifest:      {len(unique_slugs_seen & available_slugs)}")
    print(f"  slugs MISSING:          {n_missing_centroid}")
    print(f"  slugs broken centroid:  {n_broken_centroid}")
    print(f"  manifest slugs unused:  {len(available_slugs - unique_slugs_seen)}")
    print(f"\n[done] wrote {args.output}")

    if n_unparsed > 0 or n_missing_centroid > 0:
        print(f"\n[WARN] {n_unparsed + n_missing_centroid} task_indices "
              f"({(n_unparsed + n_missing_centroid)/len(df)*100:.1f}%) lack a usable centroid. "
              f"These variants will be filtered out by arcface_audit_200celeb.py "
              f"(target_cos=NaN row → dropped in build_keep_list step).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
