#!/usr/bin/env python3
"""Capture / drive-to-home helper for the interactive rollout wrappers.

The rollout wrappers (`run_eval_{1,2,3}.sh` and the per-eval rollout
shells) invoke this helper twice per episode:

* `capture <out.json>` snapshots the follower's current joint pose to a
  JSON file *before* `lerobot-record` starts.
* `drive <in.json> [seconds=2.0]` re-sends that pose to the follower at
  30 Hz for `seconds` *after* the rollout ends.

The pair preserves the operator's chosen home pose across many rollouts,
which is what makes the loop `capture -> record -> drive -> capture ...`
viable: pressing right-arrow mid-rollout (lerobot's built-in early-stop)
returns the arm to where it was when the rollout started, no manual
re-homing required.

JSON schema
-----------
The capture / drive file is a flat object keyed by motor name with float
positions in lerobot's calibrated units::

    {
        "shoulder_pan.pos": 12.4,
        "shoulder_lift.pos": -8.7,
        "elbow_flex.pos":  60.1,
        "wrist_flex.pos":  -3.2,
        "wrist_roll.pos":  0.0,
        "gripper.pos":     20.0
    }

Hardware contract
-----------------
- SO-101 6-DOF follower at ``/dev/so101-follower`` (udev-pinned).
- The follower must be calibrated; if not, ``SOFollower.connect()`` raises.
- No camera is opened (``cameras={}``); the rollout wrappers handle video.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Final

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

#: Canonical motor ordering for the SO-101 follower. Used both to project
#: a raw observation dict down to a 6-tuple and to round-trip the home
#: pose through JSON without losing key order.
ACTION_KEYS: Final[list[str]] = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]

#: udev-pinned serial path for the SO-101 follower arm.
PORT: Final[str] = "/dev/so101-follower"

#: Robot identifier (matches the lerobot calibration registry; do not change).
ID: Final[str] = "my_follower"


def connect() -> SOFollower:
    """Open a serial connection to the SO-101 follower at ``PORT`` / ``ID``.

    Returns:
        A live ``SOFollower`` handle. The caller is responsible for invoking
        ``.disconnect()`` (use a ``try / finally`` block).

    Raises:
        Exception: any low-level serial / calibration error from the
            lerobot ``SOFollower.connect()`` call is propagated unchanged.
    """
    cfg = SOFollowerRobotConfig(port=PORT, id=ID, cameras={})
    f = SOFollower(cfg)
    f.connect()
    return f


def cmd_capture(out_path: str) -> None:
    """Snapshot the follower's current joint pose to JSON.

    Reads exactly one observation from the follower, projects it down to
    the six motor positions in ``ACTION_KEYS``, and writes the result to
    ``out_path`` as a flat ``{motor_name: float}`` JSON object.

    Args:
        out_path: Destination path for the JSON snapshot. The parent
            directory must already exist (the rollout wrappers default to
            ``/tmp/...``, which always does).

    Side effects:
        Connects + disconnects the follower; writes one JSON file at
        ``out_path``; prints a one-line confirmation to stdout.
    """
    f = connect()
    try:
        obs = f.get_observation()
        pose = {k: float(obs[k]) for k in ACTION_KEYS}
    finally:
        f.disconnect()
    Path(out_path).write_text(json.dumps(pose))
    print(f"  home pose saved -> {out_path}: "
          f"{[round(pose[k], 1) for k in ACTION_KEYS]}")


def cmd_drive(in_path: str, seconds: float) -> None:
    """Drive the follower toward the saved pose for ``seconds`` seconds.

    Replays the pose at 30 Hz wall-clock. Loop terminates either when the
    wall-clock budget is exhausted or when ``send_action`` raises (in
    which case the failure is logged and the loop exits cleanly so the
    rollout wrapper can move on to the next episode).

    Args:
        in_path: Path to a JSON snapshot produced by :func:`cmd_capture`.
        seconds: Wall-clock duration of the home drive. Independent of the
            arm's joint travel distance; 2 s is the rollout-wrapper default
            and is enough to converge from any reachable starting pose.

    Side effects:
        Connects + disconnects the follower; sends repeated action
        commands; prints start + end messages to stdout. A failed
        ``send_action`` emits a ``[WARN]`` line and breaks the loop.
    """
    pose = json.loads(Path(in_path).read_text())
    f = connect()
    try:
        print(f"  driving back to home for {seconds:.1f}s ...")
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
    print(f"  at home")


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
