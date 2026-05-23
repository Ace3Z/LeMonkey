#!/usr/bin/env python3
"""Merge all 180 per-episode Eval 2 datasets into one LeRobot v3 dataset.

Wraps `lerobot-edit-dataset --operation.type=merge` for the long input list.
The 180 per-episode dirs each look like:
  ~/LeMonkey/datasets/eval2/ep_NNNN_<arr>_<family>_<ts>/

After merge, the merged dataset lives at:
  ~/LeMonkey/datasets/eval2_merged/

Use that single root when training on Brev — much simpler than 180 inputs.

Usage:
    merge_episodes.py                     # default paths
    merge_episodes.py --src DIR --dst DIR
    merge_episodes.py --dry-run           # print the command, don't run it
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Merge per-episode Eval 2 datasets into a single LeRobot v3 dataset."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path,
                   default=Path(str(Path.home() / "LeMonkey/datasets/eval2")),
                   help="Directory containing the per-episode dataset dirs (ep_*)")
    p.add_argument("--dst", type=Path,
                   default=Path(str(Path.home() / "LeMonkey/datasets/eval2_merged")),
                   help="Output directory for the merged LeRobot v3 dataset")
    p.add_argument("--repo-id", default="local/so101_eval2",
                   help="Made-up local repo_id for the merged dataset")
    p.add_argument("--lerobot-bin",
                   default=str(Path.home() / "miniconda3/envs/lemonkey/bin/lerobot-edit-dataset"),
                   help="Path to the lerobot-edit-dataset CLI binary")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the merge command without executing it")
    args = p.parse_args()

    # 1. Discover episode dirs (skip non-eval dirs like merged outputs)
    ep_dirs = sorted(d for d in args.src.iterdir()
                     if d.is_dir() and d.name.startswith("ep_"))
    if not ep_dirs:
        print(f"ERROR: no ep_* dirs under {args.src}", file=sys.stderr)
        return 1
    print(f"  found {len(ep_dirs)} episode dirs under {args.src}")

    # 2. Verify schema uniformity (cheap pre-flight)
    fingerprints = set()
    for d in ep_dirs:
        info = json.loads((d / "meta" / "info.json").read_text())
        feats = info.get("features", {})
        fp = json.dumps(sorted(feats.keys()))
        fingerprints.add(fp)
    if len(fingerprints) != 1:
        print(f"ERROR: schema mismatch across episode dirs ({len(fingerprints)} fingerprints)",
              file=sys.stderr)
        return 1
    print(f"  ✓ all {len(ep_dirs)} share the same feature schema")

    # 3. Refuse to overwrite an existing merged dst (lerobot will error anyway)
    if args.dst.exists() and not args.dry_run:
        ans = input(f"  ⚠ {args.dst} exists. Delete and re-merge? [y/N]: ").strip().lower()
        if ans != "y":
            print("aborted.")
            return 0
        shutil.rmtree(args.dst)
        print(f"  removed {args.dst}")

    # 4. Build the lerobot-edit-dataset merge invocation.
    # Each input needs a (repo_id, root) pair; pass repo_ids as fake local names.
    repo_ids = [f"local/{d.name}" for d in ep_dirs]
    roots = [str(d) for d in ep_dirs]

    cmd = [
        args.lerobot_bin,
        "--operation.type=merge",
        "--operation.repo_ids=" + json.dumps(repo_ids),
        "--operation.roots="    + json.dumps(roots),
        f"--new_repo_id={args.repo_id}",
        f"--new_root={args.dst}",
        "--push_to_hub=false",
    ]
    print(f"\n  command (truncated): {cmd[0]} {cmd[1]} ...{len(cmd)-2} more args...")
    print(f"  output: {args.dst}")
    print()

    if args.dry_run:
        print(f"  [dry-run] would run merge of {len(ep_dirs)} datasets")
        return 0

    # 5. Run it
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"\n  ✗ lerobot-edit-dataset exited rc={rc}", file=sys.stderr)
        return rc

    # 6. Verify the merged output
    out_info = args.dst / "meta" / "info.json"
    if not out_info.exists():
        print(f"\n  ✗ merge succeeded but {out_info} missing", file=sys.stderr)
        return 1
    info = json.loads(out_info.read_text())
    print(f"\n  ✓ merged dataset:")
    print(f"      total_episodes: {info.get('total_episodes')}")
    print(f"      total_frames  : {info.get('total_frames')}")
    print(f"      fps           : {info.get('fps')}")
    print(f"      output_dir    : {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
