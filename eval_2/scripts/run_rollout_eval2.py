#!/usr/bin/env python3
"""Random-prompt deployment rollout against the trained Eval 2 SmolVLA.

Every iteration picks NEW random values (no seed, no plan):
  1. Bowl arrangement — one of the 6 permutations (BRG / BGR / RBG / RGB / GBR / GRB)
  2. Source — 50/50 split between TRAINED phrasings (from record_eval2.py's pools)
     and OUT-OF-DISTRIBUTION phrasings (designed not to overlap)
  3. Family — direct / spatial_absolute / spatial_ordinal / relational_lr /
     relational_between / negation
  4. Target bowl — derived from the arrangement and prompt

The script:
  - Tells you which arrangement to set up (banner) and the target bowl
  - On ENTER, captures the arm's current pose, runs lerobot-record (40 s,
    right-arrow ends early), then drives the arm back to that pose
  - Logs to ~/LeMonkey/eval_2/rollouts/run_<i>_<arr>_<src>_<fam>_<ts>/
  - 'q' quits, 's' skips this prompt and resamples

Usage:
    run_rollout_eval2.py                        # default: v2/025000, random
    run_rollout_eval2.py 020000                 # use a different intermediate ckpt
    run_rollout_eval2.py --ood-prob 0.3         # 30% OOD instead of default 50%
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Reuse the TRAINED phrasing pools and per-family generators from the recorder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_eval2 import (
    ARRANGEMENTS, COLOR_NAMES,
    DIRECT_COLOR_PHR, ABS_PHR, ORD_PHR, ORDINAL_WORDS,
    REL_LR_PHR, REL_BETWEEN_PHR, NEG_PHR,
)

# ─── OOD phrasing pools ──────────────────────────────────────────────────────
# These are deliberately disjoint from record_eval2.py's pools so the policy
# has not seen them verbatim. They keep the same compositional structure.

DIRECT_OOD = [
    "I want the banana in the {c} bowl",
    "Could you place the banana in the {c} bowl?",
    "Banana goes in the {c} bowl please",
    "Take the banana and put it in the {c} bowl",
    "Stick the banana in the {c} bowl",
    "Drop the banana inside the {c} bowl",
]

ABS_OOD = {
    "left": [
        "Drop the banana in the bowl on the left side.",
        "Put the banana in the leftward bowl.",
        "Stick the banana in the bowl that's the most to the left.",
    ],
    "middle": [
        "Drop the banana in the bowl right in the middle.",
        "Put the banana in the bowl that's between the other two.",
        "Stick the banana in the central bowl.",
    ],
    "right": [
        "Drop the banana in the bowl on the right side.",
        "Put the banana in the rightward bowl.",
        "Stick the banana in the bowl that's the most to the right.",
    ],
}

ORD_OOD = [
    "Put the banana in the bowl that's number {n} from the {ref}.",
    "Drop the banana into the bowl in position {ord} from the {ref}.",
    "Place the banana in bowl {n} counting from the {ref}.",
]

REL_LR_OOD = [
    "Drop the banana in the bowl that sits to the {side} of the {ref} bowl.",
    "Place the banana in the bowl positioned {side} of the {ref} bowl.",
    "Put the banana in the bowl next to the {ref} bowl on the {side}.",
]

REL_BETWEEN_OOD = [
    "Drop the banana in the bowl with the {a} bowl on one side and the {b} bowl on the other.",
    "Place the banana in the bowl flanked by the {a} and {b} bowls.",
    "Put the banana in the bowl that has the {a} and the {b} bowl as neighbours.",
]

NEG_OOD = [
    "Put the banana in the bowl that is anything but {a} or {b}.",
    "Drop the banana in the bowl whose colour is neither {a} nor {b}.",
    "Place the banana in the bowl excluding the {a} and the {b} ones.",
]


# ─── Generators per (family, source) ─────────────────────────────────────────

def gen_direct(arr, ti, src):
    color = COLOR_NAMES[arr[ti]]
    pool = DIRECT_OOD if src == "ood" else DIRECT_COLOR_PHR
    return ti, random.choice(pool).format(c=color)


def gen_absolute(arr, ti, src):
    pos = ["left", "middle", "right"][ti]
    pool = ABS_OOD[pos] if src == "ood" else ABS_PHR[pos]
    return ti, random.choice(pool)


def gen_ordinal(arr, ti, src):
    side = random.choice(["left", "right"])
    n = ti + 1 if side == "left" else 3 - ti
    if src == "ood":
        word = random.choice(ORDINAL_WORDS[n])
        prompt = random.choice(ORD_OOD).format(n=n, ord=word, ref=side)
    else:
        word = random.choice(ORDINAL_WORDS[n])
        prompt = random.choice(ORD_PHR).format(ord=word, ref=side)
    return ti, prompt


def gen_relational_lr(arr, ti, src):
    options = []
    for side, dx in [("right", -1), ("left", +1)]:
        ref_idx = ti + dx
        if 0 <= ref_idx <= 2:
            options.append((side, COLOR_NAMES[arr[ref_idx]]))
    if not options:
        return None, None
    side, ref = random.choice(options)
    pool = REL_LR_OOD if src == "ood" else REL_LR_PHR
    return ti, random.choice(pool).format(side=side, ref=ref)


def gen_relational_between(arr, ti, src):
    if ti != 1:
        return None, None
    a = COLOR_NAMES[arr[0]]
    b = COLOR_NAMES[arr[2]]
    if random.random() < 0.5:
        a, b = b, a
    pool = REL_BETWEEN_OOD if src == "ood" else REL_BETWEEN_PHR
    return ti, random.choice(pool).format(a=a, b=b)


def gen_negation(arr, ti, src):
    target = COLOR_NAMES[arr[ti]]
    others = [c for c in ["blue", "red", "green"] if c != target]
    if random.random() < 0.5:
        others = others[::-1]
    a, b = others
    pool = NEG_OOD if src == "ood" else NEG_PHR
    return ti, random.choice(pool).format(a=a, b=b)


GENERATORS = {
    "direct":              gen_direct,
    "spatial_absolute":    gen_absolute,
    "spatial_ordinal":     gen_ordinal,
    "relational_lr":       gen_relational_lr,
    "relational_between":  gen_relational_between,
    "negation":            gen_negation,
}
FAMILIES = list(GENERATORS.keys())


def random_pick(ood_prob: float):
    """Pick a fully-random (arrangement, source, family, target_idx, prompt) tuple."""
    while True:
        arr = random.choice(ARRANGEMENTS)
        src = "ood" if random.random() < ood_prob else "trained"
        fam = random.choice(FAMILIES)
        ti = random.randint(0, 2)
        out_ti, prompt = GENERATORS[fam](arr, ti, src)
        if prompt is None:
            continue  # invalid combo (e.g. between when target ≠ middle); resample
        return arr, src, fam, out_ti, prompt


# ─── Main loop ───────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt_step", nargs="?", default="025000",
                   help="Eval 2 v2 checkpoint step under train/smolvla_eval2_v2/checkpoints/ (default 025000, camera-frame retrain)")
    p.add_argument("--ood-prob", type=float, default=0.5,
                   help="Probability of drawing from OOD phrasings (default 0.5)")
    p.add_argument("--episode-time-s", type=float, default=40.0)
    p.add_argument("--reset-time-s",   type=float, default=10.0)
    p.add_argument("--rollout-dir", default="/home/lemonkey/LeMonkey/eval_2/rollouts")
    p.add_argument("--follower-port", default="/dev/so101-follower")
    p.add_argument("--follower-id",   default="my_follower")
    p.add_argument("--cam-path",      default="/dev/video0")
    p.add_argument("--home-drive-s",  type=float, default=2.0)
    args = p.parse_args()

    # No seed: every invocation is fully random.
    random.seed(time.time_ns())

    policy = Path(f"/home/lemonkey/LeMonkey/eval_2/train/smolvla_eval2_v2/checkpoints/{args.ckpt_step}/pretrained_model")
    if not policy.is_dir():
        print(f"ERROR: checkpoint not found: {policy}", file=sys.stderr)
        return 1

    Path(args.rollout_dir).mkdir(parents=True, exist_ok=True)

    # Helper: capture/drive home pose. We reuse eval_1/scripts/auto_home.py.
    auto_home = Path("/home/lemonkey/LeMonkey/eval_1/scripts/auto_home.py")
    pybin = "/home/lemonkey/miniconda3/envs/lemonkey/bin/python"
    home_pose = "/tmp/run_rollout_eval2_home.json"

    print("=" * 72)
    print(f"  Eval 2 random-prompt rollout (camera-frame v2 model)")
    print(f"  policy      : {policy}")
    print(f"  ood_prob    : {args.ood_prob:.0%}  (trained vs out-of-distribution)")
    print(f"  rollouts to : {args.rollout_dir}")
    print(f"  CONVENTION  : positions are CAMERA-FRAME (look at the image feed).")
    print(f"                arr[0] = bowl on the IMAGE LEFT side, arr[2] = IMAGE RIGHT.")
    print(f"  press Ctrl+C any time to quit")
    print("=" * 72)

    last_arr = None
    i = 1
    while True:
        arr, src, fam, ti, prompt = random_pick(args.ood_prob)
        target_color = COLOR_NAMES[arr[ti]]
        target_pos = ["IMAGE-LEFT", "IMAGE-MIDDLE", "IMAGE-RIGHT"][ti]

        print()
        if arr != last_arr:
            print("╔" + "═" * 70 + "╗")
            print(f"║  ARRANGEMENT CHANGE → set bowls (CAMERA view) to: {arr:<17}".ljust(71) + " ║")
            print(f"║  image-left={COLOR_NAMES[arr[0]]:<6}  image-mid={COLOR_NAMES[arr[1]]:<6}  image-right={COLOR_NAMES[arr[2]]:<6}".ljust(71) + " ║")
            print("╚" + "═" * 70 + "╝")
            last_arr = arr

        print("─" * 72)
        print(f" Rollout #{i}    arrangement={arr}    family={fam}    source={src.upper()}")
        print(f"   prompt      : \"{prompt}\"")
        print(f"   TARGET BOWL : {target_color.upper()} (idx {ti} = {target_pos})")
        print("─" * 72)

        try:
            ans = input("ENTER=record / 's'=skip prompt / 'q'=quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0
        if ans == "q":
            return 0
        if ans == "s":
            continue

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{i:03d}_{arr}_{src}_{fam}_{ts}"
        run_path = Path(args.rollout_dir) / run_name

        # 1. Capture starting pose
        rc = subprocess.call([pybin, str(auto_home), "capture", home_pose])
        if rc != 0:
            print(f"  ⚠ auto_home capture exited rc={rc}; skipping rollout")
            continue

        # 2. Run lerobot-record with the policy
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

        # 3. Drive arm back to the starting pose
        subprocess.call([pybin, str(auto_home), "drive", home_pose, str(args.home_drive_s)])

        print(f"  ✓ rollout #{i} → {run_path}")
        i += 1


if __name__ == "__main__":
    sys.exit(main() or 0)
