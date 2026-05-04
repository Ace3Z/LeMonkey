#!/usr/bin/env python3
"""Capture / drive-to-home helper for run_rollout.sh and run_rollout_voice.sh.

Two subcommands:
  capture <out.json>             Read the follower state once and dump to JSON.
  drive <in.json> [seconds=2.0]  Drive the follower toward the saved pose.

The shell scripts call `capture` before lerobot-record and `drive` after,
so pressing right-arrow mid-rollout (lerobot's built-in early-stop)
returns the arm to where it was when the rollout started.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

ACTION_KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

PORT = "/dev/so101-follower"
ID   = "my_follower"


def connect():
    cfg = SOFollowerRobotConfig(port=PORT, id=ID, cameras={})
    f = SOFollower(cfg)
    f.connect()
    return f


def cmd_capture(out_path: str) -> None:
    f = connect()
    try:
        obs = f.get_observation()
        pose = {k: float(obs[k]) for k in ACTION_KEYS}
    finally:
        f.disconnect()
    Path(out_path).write_text(json.dumps(pose))
    print(f"  📍 home pose saved → {out_path}: "
          f"{[round(pose[k], 1) for k in ACTION_KEYS]}")


def cmd_drive(in_path: str, seconds: float) -> None:
    pose = json.loads(Path(in_path).read_text())
    f = connect()
    try:
        print(f"  🏠 driving back to home for {seconds:.1f}s ...")
        dt = 1.0 / 30.0
        t0 = time.time()
        while time.time() - t0 < seconds:
            try:
                f.send_action(pose)
            except Exception as e:
                print(f"[WARN] send_action during home drive failed: {e}", flush=True)
                break
            time.sleep(dt)
    finally:
        f.disconnect()
    print(f"  ✓ at home")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sub = sys.argv[1]
    if sub == "capture":
        cmd_capture(sys.argv[2])
    elif sub == "drive":
        secs = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
        cmd_drive(sys.argv[2], secs)
    else:
        print(f"unknown subcommand: {sub}", file=sys.stderr)
        sys.exit(2)
