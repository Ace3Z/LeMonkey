#!/usr/bin/env python3
"""Merge all base teleops + augmented variants into one LeRobot v3 dataset.

Discovers all episode dirs under --base-root (any dir with meta/info.json and
reference.json) and --aug-root (any dir containing the --aug-pattern
substring), schema-checks them, and merges into one LeRobot v3 root via
lerobot.datasets.aggregate.aggregate_datasets() (with
video_files_size_in_mb=0.01 to force per-episode video files and avoid the
DTS-monotonicity bug). The --lerobot-bin argparse flag is currently unused
and held for compatibility with an earlier subprocess-based path.

Sources:
  --base-root  datasets/eval3/         (~180 real teleops)
  --aug-root   datasets/eval3_aug_*/   (per-augmentation-run variants)

Total: deployed cotrain has 9,394 episodes (178 base + 9,216 variants);
deployed broad has 9,842 episodes (180 base + ~9,662 variants). Numbers
vary by augmentation run; the merger composes whatever the producers wrote.

Usage:
    merge_episodes.py                         # default paths
    merge_episodes.py --base-root … --aug-root … --dst …
    merge_episodes.py --dry-run               # print stats, don't run
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def discover_episode_dirs(
    base_root: Path,
    aug_root: Path,
    aug_pattern: str = "__var",
) -> list[Path]:
    """Yield (base teleops in alphabetical order) then (augmented variants
    in alphabetical order). The base teleops being first means episode_index
    0..(N_base-1) is base, and N_base..end is augmented — useful for any
    later analysis.

    `aug_pattern` distinguishes variant naming conventions:
        "__var"  matches the broad augmentation output (per-episode 5-variant pipeline)
        "__t3_"  matches the co-train augmentation output (one variant per (target_photo, layout) tuple)
    """
    base = []
    if base_root.is_dir():
        base = sorted(p for p in base_root.iterdir()
                        if p.is_dir() and (p / "meta" / "info.json").is_file()
                        and (p / "reference.json").is_file())
    aug = []
    if aug_root.is_dir():
        aug = sorted(p for p in aug_root.iterdir()
                       if p.is_dir() and aug_pattern in p.name
                       and (p / "meta" / "info.json").is_file())
    return base + aug


def schema_fingerprint(ep: Path) -> str:
    """Return a string fingerprint of an episode's features schema.
    Two episodes with the same fingerprint are merge-compatible."""
    info = json.loads((ep / "meta" / "info.json").read_text())
    feats = info.get("features", {})
    # Order matters for SigLIP-prefix — encode as ordered list of (key, dtype, shape)
    sig = []
    for k, v in feats.items():
        shape = v.get("shape", [])
        sig.append((k, v.get("dtype", ""), tuple(shape)))
    return json.dumps(sig)


def main() -> int:
    """Discover base + augmented episode dirs, schema-check them, and aggregate into one LeRobot v3 dataset at --dst."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-root", type=Path,
                   default=Path("datasets/eval3"),
                   help="Root containing the ~180 real-teleop base episode dirs")
    p.add_argument("--aug-root", type=Path,
                   default=Path("datasets/eval3_aug_broad"),
                   help="Root containing augmented variant dirs from "
                        "aug/generators/{broad,cotrain,broad_missing_celebs}.py")
    p.add_argument("--aug-pattern", default="__var",
                   help="Substring identifying augmented variants. "
                        "'__var' matches the broad augmentation output (per-episode 5-variant pipeline); "
                        "'__t3_' matches the co-train augmentation output (one variant per (target_photo, layout) tuple).")
    p.add_argument("--dst", type=Path,
                   default=Path("datasets/eval3_merged"),
                   help="Output dir for the merged dataset")
    p.add_argument("--repo-id", default="local/so101_eval3_broad",
                   help="Made-up local repo_id for the merged dataset")
    p.add_argument("--lerobot-bin",
                   default="lerobot-edit-dataset",
                   help="Path to lerobot-edit-dataset; falls back to PATH lookup")
    p.add_argument("--dry-run", action="store_true",
                   help="Stats + command preview, no merge")
    args = p.parse_args()

    # 1. Discover all episode dirs
    ep_dirs = discover_episode_dirs(args.base_root, args.aug_root, args.aug_pattern)
    if not ep_dirs:
        print(f"ERROR: no episode dirs found under {args.base_root} or "
              f"{args.aug_root}", file=sys.stderr)
        return 1
    n_base = sum(1 for d in ep_dirs if d.parent == args.base_root.resolve()
                 or d.parent == args.base_root)
    n_aug = len(ep_dirs) - n_base
    print(f"  found {len(ep_dirs)} episode dirs total "
          f"(base={n_base} + augmented={n_aug})")

    # 2. Schema check
    fps = {}     # fingerprint -> [ep names]
    for d in ep_dirs:
        try:
            fp = schema_fingerprint(d)
        except Exception as e:
            print(f"ERROR: schema_fingerprint failed on {d.name}: {e}",
                  file=sys.stderr)
            return 1
        fps.setdefault(fp, []).append(d.name)
    if len(fps) != 1:
        print(f"ERROR: schema mismatch across {len(ep_dirs)} eps "
              f"({len(fps)} distinct fingerprints):", file=sys.stderr)
        for i, (fp, eps) in enumerate(fps.items()):
            print(f"  fingerprint #{i+1}: {len(eps)} eps "
                  f"(e.g. {eps[0]}); features = {fp[:200]}", file=sys.stderr)
        return 1
    fp_dict = json.loads(next(iter(fps)))
    print(f"  ✓ schema uniform across all eps. Features:")
    for k, dtype, shape in fp_dict:
        print(f"      {k:35} dtype={dtype:8}  shape={shape}")

    # 3. Refuse to clobber an existing merged output unless explicit confirm
    if args.dst.exists() and not args.dry_run:
        ans = input(f"\n  ⚠ {args.dst} exists. Delete and re-merge? [y/N]: ").strip().lower()
        if ans != "y":
            print("aborted.")
            return 0
        shutil.rmtree(args.dst)
        print(f"  removed {args.dst}")

    # 4. Invoke the merge via Python API (avoids 600+ KB cmdline / ARG_MAX
    #    issues with subprocess on 4195 paths).
    print(f"  output: {args.dst}")

    if args.dry_run:
        print(f"  [dry-run] would run merge of {len(ep_dirs)} datasets")
        return 0

    # NOTE: merge_datasets() is a thin wrapper that doesn't expose the
    # video_files_size_in_mb parameter. We bypass it and call
    # aggregate_datasets() directly so we can force per-episode video files
    # (no bitstream concat → no DTS-monotonicity bug across heterogenous
    # mp4 encoders). Setting size to 0.01 MB makes every source video its
    # own destination file in the merged dataset.
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print(f"\n  merging {len(ep_dirs)} per-episode datasets via "
          f"aggregate_datasets (video_files_size_in_mb=0.01 → no concat)",
          flush=True)
    repo_ids = [f"local/{d.name}" for d in ep_dirs]
    roots = [d.resolve() for d in ep_dirs]
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=args.repo_id,
        roots=roots,
        aggr_root=args.dst.resolve(),
        video_files_size_in_mb=0.01,
    )
    # Re-open the merged dataset so we can report metadata.
    merged = LeRobotDataset(args.repo_id, root=args.dst.resolve())

    # 5. Verify the merged output
    out_info = args.dst / "meta" / "info.json"
    if not out_info.exists():
        print(f"\n  ✗ merge succeeded but {out_info} missing", file=sys.stderr)
        return 1
    info = json.loads(out_info.read_text())
    print(f"\n  ✓ merged dataset:")
    print(f"      total_episodes: {info.get('total_episodes')}")
    print(f"      total_frames  : {info.get('total_frames')}")
    print(f"      fps           : {info.get('fps')}")
    print(f"      feature keys  : {list(info.get('features', {}).keys())}")
    print(f"      output_dir    : {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
