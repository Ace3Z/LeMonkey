#!/usr/bin/env python3
"""End-to-end orchestrator for the Eval 3 inpainting augmentation pipeline.

Glues stages 2 → 3 → 4 → 5 together. Stage 1 (photo bank) is a one-time
prerequisite; the orchestrator just sanity-checks that the bank exists.

Usage:
    # Full pipeline on the 5 quick-record episodes, 5 augmented variants each
    python pipeline.py --root ~/LeMonkey/datasets/eval3_quick \\
                      --num-variants 5 \\
                      --interactive          # for stage-2 click prompts on first run

    # Skip stages already completed (idempotent)
    python pipeline.py --root ~/LeMonkey/datasets/eval3_quick

    # Run on a single episode end-to-end
    python pipeline.py /path/to/episode_dir --num-variants 3

Stages run in order; failures are collected and reported but the pipeline
keeps going for the remaining episodes (each stage is independent per
episode).

Each stage idempotency:
  - Stage 2 skips episodes that already have portrait_masks.pkl (--force overrides)
  - Stage 3 skips episodes that already have portrait_corners.json
  - Stage 4 skips variants whose final mp4 already exists
  - Stage 5 skips variants that already have verification.json

This means re-running pipeline.py after fixing one episode's seeds is
cheap.

See STRATEGY.md §8 for the build/smoke milestones (M1–M6).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_stage(name: str, cmd: list[str]) -> int:
    print()
    print("─" * 72)
    print(f" {name}")
    print("─" * 72)
    print(f" $ {' '.join(cmd)}")
    print("─" * 72)
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("episode_dir", nargs="?", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--out-root", default="/home/lemonkey/LeMonkey/datasets/eval3_aug",
                   help="where stage 4 writes augmented variants")
    p.add_argument("--photo-bank", default="/home/lemonkey/LeMonkey/datasets/eval3_celebs/web")
    p.add_argument("--num-variants", type=int, default=5)
    p.add_argument("--interactive", action="store_true",
                   help="pass --interactive to stage 2 for click-prompts on first run")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="ArcFace cosine threshold for stage 5 verification")
    p.add_argument("--drop-failed", action="store_true",
                   help="stage 5 deletes variants that fail identity verification")
    p.add_argument("--force", action="store_true",
                   help="passed to all stages — re-do everything")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-photo-bank-check", action="store_true",
                   help="don't error if photo bank is empty (useful when running stages 2-3 first)")
    p.add_argument("--stages", default="2345",
                   help="which stages to run, e.g. '23' to run only segmentation and corners")
    args = p.parse_args()

    if (args.episode_dir is None) == (args.root is None):
        print("[ERROR] specify exactly one of: episode_dir, --root", file=sys.stderr)
        return 2

    # Sanity check the photo bank
    bank_root = Path(args.photo_bank)
    n_celebs = sum(1 for d in bank_root.glob("*/") if d.is_dir() and not d.name.startswith("_")) if bank_root.is_dir() else 0
    if not args.skip_photo_bank_check and "4" in args.stages:
        if n_celebs == 0:
            print(f"[ERROR] photo bank at {bank_root} is empty.\n"
                  f"        Run stage 1 first:\n"
                  f"        python {HERE / '1_mine_celeb_photos.py'} --celebs swift obama lecun --num 30\n"
                  f"        (or pass --skip-photo-bank-check to bypass and run only stages 2-3)",
                  file=sys.stderr)
            return 1

    # Build common args for episode/root
    common = ["--episode_dir", args.episode_dir] if args.episode_dir else ["--root", args.root]
    # argparse flag naming differs between modules — we accept positional or --root
    pos_or_root = [args.episode_dir] if args.episode_dir else ["--root", args.root]

    rc_total = 0

    if "2" in args.stages:
        cmd = [sys.executable, str(HERE / "2_segment_video.py")]
        cmd += pos_or_root
        if args.interactive:
            cmd.append("--interactive")
        if args.force:
            cmd.append("--force")
        rc = run_stage("STAGE 2 — SAM 2.1 video segmentation", cmd)
        rc_total |= rc
        if rc != 0:
            print("[WARN] stage 2 returned non-zero; continuing")

    if "3" in args.stages:
        cmd = [sys.executable, str(HERE / "3_extract_corners.py")]
        cmd += pos_or_root
        if args.force:
            cmd.append("--force")
        rc = run_stage("STAGE 3 — extract 4-corner quads", cmd)
        rc_total |= rc

    if "4" in args.stages:
        cmd = [sys.executable, str(HERE / "4_inpaint_video.py")]
        cmd += pos_or_root
        cmd += ["--out-root", args.out_root,
                "--photo-bank", str(bank_root),
                "--num-variants", str(args.num_variants),
                "--seed", str(args.seed)]
        if args.force:
            cmd.append("--force")
        rc = run_stage("STAGE 4 — composite augmented variants", cmd)
        rc_total |= rc

    if "5" in args.stages:
        cmd = [sys.executable, str(HERE / "5_verify_identity.py"),
               "--root", args.out_root,
               "--threshold", str(args.threshold)]
        if args.force:
            cmd.append("--force")
        if args.drop_failed:
            cmd.append("--drop-failed")
        rc = run_stage("STAGE 5 — ArcFace identity verification", cmd)
        rc_total |= rc

    print()
    print("=" * 72)
    print(f"  pipeline complete (overall rc={rc_total})")
    print("=" * 72)
    return rc_total


if __name__ == "__main__":
    sys.exit(main() or 0)
