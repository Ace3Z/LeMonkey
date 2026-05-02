#!/usr/bin/env python3
"""30-rollout structured evaluation with the residual-augmented policy.

Mirrors eval_checkpoint.sh's structure but in one Python process so the
base+residual is loaded just once. Same shuffled mix of in-distribution
and out-of-distribution prompts (10 per color), same CSV schema, same
per-rollout y/n confirmation, output goes to evals/ckpt<step>_residual_<ts>.csv
so compare_evals.py picks it up alongside base-only sessions.

Per CLAUDE.md §5: any fallback path must emit a [WARN] log line.

Usage:
    eval_residual.py --base-path /path/to/base/pretrained_model \\
                     --residual-path /path/to/residual/last \\
                     [--num-episodes 30] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Event

import numpy as np
from pynput import keyboard

# Local
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference_residual import ResidualWrapper

# lerobot
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower


# ─── Args ────────────────────────────────────────────────────────────────────

p = argparse.ArgumentParser()
p.add_argument("--base-path", required=True)
p.add_argument("--residual-path", required=True)
p.add_argument("--ckpt-tag", default="residual",
               help="Tag used in the CSV filename (e.g. 'residual', 'r5k').")
p.add_argument("--num-episodes", type=int, default=30,
               help="Total rollouts. 30 = 10 per color (5 trained + 5 OOD prompts each).")
p.add_argument("--episode-time-s", type=float, default=20)
p.add_argument("--seed", type=int, default=None)
p.add_argument("--follower-port", default="/dev/so101-follower")
p.add_argument("--follower-id",   default="my_follower")
p.add_argument("--cam-path",      default="/dev/video0")
p.add_argument("--cam-width",     type=int, default=640)
p.add_argument("--cam-height",    type=int, default=480)
p.add_argument("--fps",           type=int, default=30)
p.add_argument("--max-relative-target", type=float, default=8.0)
p.add_argument("--device",        default="cuda")
p.add_argument("--home-settle-s", type=float, default=2.0,
               help="Seconds spent driving the follower back to the saved home "
                    "pose between rollouts. 0 disables auto-home.")
args = p.parse_args()


# ─── Keyboard listener: 'n' to end the current rollout immediately ──────────
end_rollout_request = Event()
def _on_press(key):
    try:
        if key.char and key.char.lower() == "n":
            end_rollout_request.set()
    except AttributeError:
        pass
keyboard.Listener(on_press=_on_press, daemon=True).start()

if args.seed is None:
    args.seed = int(time.time()) & 0xffff
print(f"random seed: {args.seed}")
random.seed(args.seed)


# ─── Prompt list (mirrors eval_checkpoint.sh exactly) ───────────────────────

trained = [
    "Put the banana in the {} colored bowl.",
    "Put the banana in the {} bowl",
    "Place the banana in the {} bowl",
    "pick the banana and put it in the {} bowl",
    "Place the banana in the {} colored bowl",
]
untrained = [
    "Move the banana to the {} bowl",
    "Drop the banana in the {} bowl",
    "Take the banana and put it in the {} bowl",
    "Put it into the {} bowl",
    "Banana goes in the {} bowl",
]

prompts = []
for color in ["blue", "red", "green"]:
    block = []
    for t in trained:
        block.append((color, "trained", t.format(color)))
    for t in untrained:
        block.append((color, "untrained", t.format(color)))
    random.shuffle(block)
    prompts.extend(block)

if args.num_episodes != len(prompts):
    print(f"[WARN] --num-episodes={args.num_episodes} != prompt-list length {len(prompts)}; "
          f"truncating/extending to match", flush=True)
    prompts = prompts[:args.num_episodes]


# ─── Output paths ────────────────────────────────────────────────────────────

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
sess = f"ckpt{args.ckpt_tag}_{ts}"
csv_path = Path(f"/home/lemonkey/LeMonkey/eval_1/evals/{sess}.csv")
roll_base = Path(f"/home/lemonkey/LeMonkey/eval_1/rollouts")
csv_path.parent.mkdir(parents=True, exist_ok=True)
roll_base.mkdir(parents=True, exist_ok=True)


# ─── Hardware setup ─────────────────────────────────────────────────────────

print("\nConnecting follower + camera ...")
cam_cfg = OpenCVCameraConfig(
    index_or_path=args.cam_path,
    width=args.cam_width, height=args.cam_height, fps=args.fps,
)
_max_rel = args.max_relative_target if args.max_relative_target > 0 else None
follower_cfg = SOFollowerRobotConfig(
    port=args.follower_port,
    id=args.follower_id,
    cameras={"camera1": cam_cfg},
    max_relative_target=_max_rel,
)
follower = SOFollower(follower_cfg)
follower.connect()
print(f"  follower connected, max_relative_target={_max_rel}\n")


# ─── Clean shutdown ──────────────────────────────────────────────────────────

_done = False
def _shutdown(reason: str = ""):
    global _done
    if _done:
        return
    _done = True
    print(f"\n[shutdown] {reason}")
    try:
        if follower.is_connected:
            follower.bus.disable_torque()
            print("[shutdown] follower torque DISABLED — move arm by hand.")
    except Exception as e:
        print(f"[WARN] follower torque release failed during shutdown: {e}", flush=True)
    try:
        follower.disconnect()
    except Exception as e:
        print(f"[WARN] follower.disconnect() failed: {type(e).__name__}: {e}", flush=True)

def _sigint(_s, _f):
    _shutdown("Ctrl+C")
    sys.exit(130)

signal.signal(signal.SIGINT, _sigint)
signal.signal(signal.SIGTERM, _sigint)


# ─── Load wrapper ────────────────────────────────────────────────────────────

print("=== Loading base + residual ===")
print(f"  base    : {args.base_path}")
print(f"  residual: {args.residual_path}")
wrapper = ResidualWrapper(
    base_policy_path=args.base_path,
    residual_ckpt_path=args.residual_path,
    device=args.device,
)
print(f"  ✓ loaded\n")


# ─── Helpers ─────────────────────────────────────────────────────────────────

ACTION_KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

def get_obs():
    obs = follower.get_observation()
    state_arr = np.array([float(obs[k]) for k in ACTION_KEYS], dtype=np.float32)
    if "camera1" in obs:
        img = obs["camera1"]
    elif "observation.images.camera1" in obs:
        img = obs["observation.images.camera1"]
    else:
        print(f"[WARN] no camera in obs; keys={list(obs.keys())}", flush=True)
        raise KeyError("no camera image in observation")
    return state_arr, img

def array_to_action_dict(arr) -> dict:
    return {k: float(v) for k, v in zip(ACTION_KEYS, arr)}


# ─── Banner ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Residual SmolVLA — structured evaluation")
print(f"  rollouts     : {len(prompts)} (10/color, 5 trained + 5 untrained)")
print(f"  per-ep s     : {args.episode_time_s}")
print(f"  seed         : {args.seed}")
print(f"  csv          : {csv_path}")
print(f"  home settle  : {args.home_settle_s}s after each rollout")
print("  controls     : 'n'   = end the current rollout immediately")
print("                 's' / ENTER 's' at the prompt = skip a rollout")
print("                 'q'   = quit")
print("=" * 60)
print()

# CSV header (mirrors eval_checkpoint.sh)
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "rollout", "color_target", "prompt_type", "prompt", "success",
        "duration_s", "duration_min", "clip_pct", "notes", "run_path",
    ])


# ─── Main rollout loop ───────────────────────────────────────────────────────

dt = 1.0 / args.fps

for i, (color, kind, prompt) in enumerate(prompts, 1):
    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║ ROLLOUT {i:2d} / {len(prompts)}".ljust(60) + "║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║ Target color : {color:<42}║")
    print(f"║ Prompt type  : {kind:<42}║")
    print(f"║ Prompt       :".ljust(60) + "║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print(f"\n    \"{prompt}\"\n")

    try:
        action = input("Position banana + bowls, ENTER to RUN / 's' to skip / 'q' to quit: ")
    except (EOFError, KeyboardInterrupt):
        break
    if action.strip().lower() == "q":
        print("aborted by user.")
        break
    if action.strip().lower() == "s":
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([i, color, kind, prompt, "skipped", "", "", "", "", ""])
        continue

    run_name = f"{sess}_r{i}_{color}"
    run_path = roll_base / run_name

    # Save the pose the user placed the arm at — we'll auto-drive back to it
    # after the rollout so the user doesn't have to manually reset every time.
    home_state, _ = get_obs()
    if i == 1:
        print(f"  📍 saved home pose from rollout 1 (used between rollouts): "
              f"{[round(float(v), 1) for v in home_state]}")

    wrapper.reset()
    end_rollout_request.clear()  # in case it's still set from a prior keypress
    ended_early = False
    t0 = time.time()
    try:
        while time.time() - t0 < args.episode_time_s:
            if end_rollout_request.is_set():
                end_rollout_request.clear()
                ended_early = True
                print(f"\n  ⏭   'n' pressed — ending rollout early.")
                break
            loop_start = time.time()
            state, img = get_obs()
            action_arr = wrapper.select_action(img, state, prompt)
            follower.send_action(array_to_action_dict(action_arr))
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
        rollout_rc = 0
    except Exception as e:
        print(f"[WARN] rollout exception: {type(e).__name__}: {e}", flush=True)
        rollout_rc = 1

    dur_s = int(time.time() - t0)
    dur_min = dur_s / 60.0
    summary = wrapper.episode_summary()
    print(f"\n  ⏱  duration: {dur_s}s ({dur_min:.2f} min)" +
          (" (ended early)" if ended_early else ""))
    print(f"  ⚙  residual clip rate: {summary['clip_pct']:.1f}% of {summary['n_steps']} steps")
    if rollout_rc != 0:
        print(f"  ⚠️  rollout had error (rc={rollout_rc})")

    # Drive arm back to home pose (matches lerobot-record's reset_time_s pattern).
    # The motors interpolate from current pose to home_state; sending the same
    # target each frame for ~home_settle_s gives the arm time to actually arrive.
    if args.home_settle_s > 0:
        print(f"  🏠 driving arm back to home pose for {args.home_settle_s:.1f}s ...")
        home_dict = array_to_action_dict(home_state)
        t_home = time.time()
        while time.time() - t_home < args.home_settle_s:
            try:
                follower.send_action(home_dict)
            except Exception as e:
                print(f"[WARN] send_action during home drive failed: {e}", flush=True)
                break
            time.sleep(dt)

    print(f"\n▶ Was the banana FULLY INSIDE the {color} bowl at the end?")
    while True:
        try:
            s = input("  Success? [y/n]: ")
        except (EOFError, KeyboardInterrupt):
            _shutdown("interrupt during success prompt")
            sys.exit(130)
        sl = s.strip().lower()
        if sl in ("y", "yes"):
            res = 1; break
        if sl in ("n", "no"):
            res = 0; break
        print("  please answer 'y' or 'n'")
    try:
        note = input("  Notes (ENTER to skip): ").strip()
    except (EOFError, KeyboardInterrupt) as e:
        print(f"[WARN] note prompt interrupted ({type(e).__name__}); writing empty note", flush=True)
        note = ""
    if rollout_rc != 0:
        note = f"rc={rollout_rc}; {note}"
    note = note.replace(",", ";")
    prompt_csv = prompt.replace(",", ";")

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([
            i, color, kind, prompt_csv, res,
            dur_s, f"{dur_min:.2f}",
            f"{summary['clip_pct']:.1f}",
            note, str(run_path),
        ])
    print(f"  → recorded: {'SUCCESS' if res else 'FAIL'}")


# ─── Summary ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"  RESULTS — {sess}")
print("=" * 60)
import collections
rows = list(csv.DictReader(open(csv_path)))
done = [r for r in rows if r["success"] not in ("", "skipped")]
n = len(done)
ok = sum(1 for r in done if r["success"] == "1")
print(f"  Total: {ok}/{n}  ({100*ok/n if n else 0:.0f}%)")

by_color = collections.defaultdict(lambda: [0, 0])
by_kind  = collections.defaultdict(lambda: [0, 0])
for r in done:
    by_color[r["color_target"]][0] += int(r["success"])
    by_color[r["color_target"]][1] += 1
    by_kind[r["prompt_type"]][0] += int(r["success"])
    by_kind[r["prompt_type"]][1] += 1
print("\n  By color:")
for c, (s, t) in sorted(by_color.items()):
    print(f"    {c:6s} {s}/{t} ({100*s/t if t else 0:.0f}%)")
print("\n  By prompt type:")
for k, (s, t) in sorted(by_kind.items()):
    print(f"    {k:9s} {s}/{t} ({100*s/t if t else 0:.0f}%)")

# Clip-rate summary
clip_rates = [float(r["clip_pct"]) for r in done if r["clip_pct"]]
if clip_rates:
    print(f"\n  Residual clip rate: avg {sum(clip_rates)/len(clip_rates):.1f}%, max {max(clip_rates):.1f}%")

print(f"\n  CSV: {csv_path}")

_shutdown("normal end of evaluation")
