#!/usr/bin/env python3
"""User-controlled bowl arrangement, random prompt every rollout.

You place the 3 bowls anywhere you want and tell the script the
left→middle→right colour order. The script then loops, each iteration
picking a NEW random prompt (50/50 trained vs out-of-distribution) and
showing you which bowl is the target. The policy must then put the banana
in that bowl.

Differs from run_rollout_eval2.py:
  - run_rollout_eval2.py   ← script dictates the arrangement; you reshuffle
                             when it tells you to
  - run_rollout_freeplay.py ← YOU control the arrangement; just press 'a'
                             when you want to update it

Every invocation is fully random (no seed, no plan).

Usage:
    run_rollout_freeplay.py                    # default v2/025000, prompt for arrangement
    run_rollout_freeplay.py 020000             # different intermediate ckpt
    run_rollout_freeplay.py --arrangement BRG  # skip the initial prompt
    run_rollout_freeplay.py --ood-prob 0.3     # 30% OOD instead of 50%
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Reuse the random-pick infrastructure from the structured rollout script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rollout_eval2 import GENERATORS, FAMILIES
from record_eval2 import COLOR_NAMES, is_valid_arrangement


def random_pick_for(arr: str, ood_prob: float):
    """Pick a random (source, family, target_idx, prompt) for a fixed arrangement."""
    while True:
        src = "ood" if random.random() < ood_prob else "trained"
        fam = random.choice(FAMILIES)
        ti = random.randint(0, 2)
        out_ti, prompt = GENERATORS[fam](arr, ti, src)
        if prompt is None:
            continue  # invalid combo (e.g. relational_between when target ≠ middle); resample
        return src, fam, out_ti, prompt


def ask_arrangement(prompt_text: str) -> str | None:
    while True:
        try:
            ans = input(prompt_text).strip().upper()
        except (EOFError, KeyboardInterrupt):
            return None
        if is_valid_arrangement(ans):
            return ans
        print(f"  ✗ '{ans}' is not a valid permutation of BRG; try again (or Ctrl+C to quit).")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt_step", nargs="?", default="025000")
    p.add_argument("--arrangement",   default=None,
                   help="Skip the initial prompt and use this arrangement (e.g. BRG)")
    p.add_argument("--ood-prob",      type=float, default=0.5)
    p.add_argument("--episode-time-s", type=float, default=40.0)
    p.add_argument("--reset-time-s",   type=float, default=10.0)
    p.add_argument("--rollout-dir",   default="/home/lemonkey/LeMonkey/eval_2/rollouts")
    p.add_argument("--follower-port", default="/dev/so101-follower")
    p.add_argument("--follower-id",   default="my_follower")
        p.add_argument("--cam-path",      default="/dev/video0")
    p.add_argument("--home-drive-s",  type=float, default=2.0)
    args = p.parse_args()

    random.seed(time.time_ns())  # fully random per invocation

    policy = Path(f"/home/lemonkey/LeMonkey/eval_2/train/smolvla_eval2_v2/checkpoints/{args.ckpt_step}/pretrained_model")
    if not policy.is_dir():
        print(f"ERROR: checkpoint not found: {policy}", file=sys.stderr)
        return 1

    Path(args.rollout_dir).mkdir(parents=True, exist_ok=True)
    auto_home = Path("/home/lemonkey/LeMonkey/eval_1/scripts/auto_home.py")
    pybin = "/home/lemonkey/miniconda3/envs/lerobot/bin/python"
    home_pose = "/tmp/run_rollout_freeplay_home.json"

    print("=" * 72)
    print(f"  Eval 2 free-play rollout (you choose the arrangement, camera-frame v2 model)")
    print(f"  policy      : {policy}")
    print(f"  ood_prob    : {args.ood_prob:.0%}")
    print(f"  rollouts to : {args.rollout_dir}")
    print(f"  CONVENTION  : positions are CAMERA-FRAME — look at the image feed.")
    print("=" * 72)
    print()

    # 1. Initial arrangement
    if args.arrangement:
        if not is_valid_arrangement(args.arrangement.upper()):
            print(f"ERROR: --arrangement must be a permutation of BRG (got {args.arrangement})", file=sys.stderr)
            return 2
        arr = args.arrangement.upper()
    else:
        print("Place the 3 colored bowls in front of the robot wherever you like.")
        print("Then tell me the colour order AS THE CAMERA SEES IT (look at the image):")
        print("  • arr[0] = the bowl on the IMAGE LEFT side")
        print("  • arr[1] = the bowl in the IMAGE MIDDLE")
        print("  • arr[2] = the bowl on the IMAGE RIGHT side")
        print("  e.g. BRG = blue on image-left, red in middle, green on image-right")
        print("       GRB = green on image-left, red in middle, blue on image-right")
        print()
        arr = ask_arrangement("Your arrangement (3-letter perm of BRG, image-frame): ")
        if arr is None:
            print("\nbye.")
            return 0

    print()
    print(f"  → arrangement: {arr}    "
          f"image-left={COLOR_NAMES[arr[0]]}  middle={COLOR_NAMES[arr[1]]}  image-right={COLOR_NAMES[arr[2]]}")
    print()

    # 2. Loop
    i = 1
    while True:
        src, fam, ti, prompt = random_pick_for(arr, args.ood_prob)
        target_color = COLOR_NAMES[arr[ti]]
        target_pos = ["IMAGE-LEFT", "IMAGE-MIDDLE", "IMAGE-RIGHT"][ti]

        print("─" * 72)
        print(f" Rollout #{i}    arrangement={arr}    family={fam}    source={src.upper()}")
        print(f"   prompt      : \"{prompt}\"")
        print(f"   TARGET BOWL : {target_color.upper()} (idx {ti} = {target_pos})")
        print("─" * 72)

        try:
            ans = input("ENTER=record / 's'=skip prompt / 'a'=change arrangement / 'q'=quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0

        if ans == "q":
            return 0
        if ans == "s":
            continue
        if ans == "a":
            new_arr = ask_arrangement("New arrangement (3-letter perm of BRG, image-frame): ")
            if new_arr is None:
                return 0
            arr = new_arr
            print(f"  ✓ arrangement updated → {arr}    "
                  f"image-left={COLOR_NAMES[arr[0]]}  middle={COLOR_NAMES[arr[1]]}  image-right={COLOR_NAMES[arr[2]]}")
            continue

        # Run a rollout
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"freeplay_{i:03d}_{arr}_{src}_{fam}_{ts}"
        run_path = Path(args.rollout_dir) / run_name

        rc = subprocess.call([pybin, str(auto_home), "capture", home_pose])
        if rc != 0:
            print(f"  ⚠ auto_home capture exited rc={rc}; skipping rollout")
            continue

        cmd = [
            "lerobot-record",
            "--robot.type=so101_follower",
            f"--robot.port={args.follower_port}",
            f"--robot.id={args.follower_id}",
            f"--robot.cameras={{ camera1: {{type: opencv, index_or_path: {args.cam_path}, width: 640, height: 480, fps: 30}}}}",
            "--display_data=true",
            f"--dataset.repo_id=local/eval_{run_name}",
            f"--dataset.root={run_path}",
            "--dataset.num_episodes=1",
            f"--dataset.episode_time_s={args.episode_time_s}",
            f"--dataset.reset_time_s={args.reset_time_s}",
            f"--dataset.single_task={prompt}",
            "--dataset.streaming_encoding=true",
            "--dataset.encoder_threads=2",
            "--dataset.push_to_hub=false",
            f"--policy.path={policy}",
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  ⚠ lerobot-record exited rc={rc}")

        subprocess.call([pybin, str(auto_home), "drive", home_pose, str(args.home_drive_s)])

        print(f"  ✓ rollout #{i} → {run_path}")
        i += 1


if __name__ == "__main__":
    sys.exit(main() or 0)
