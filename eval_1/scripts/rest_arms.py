#!/usr/bin/env python3
"""Release torque on both arms so they can be moved to rest by hand.

Connects to the SO-101 follower and the SO-101 leader, disables motor
torque on both buses (so they go limp and can be repositioned manually
without fighting motors), and waits for ENTER before re-enabling torque
on the follower at the new pose. The leader stays passive throughout
(its motors are torque-disabled by design, that's why you can backdrive
it during teleop).

Use cases:
    - Before starting a recording session, to make sure both arms are at
      a known home pose.
    - After dagger_record.py exits with the follower stuck mid-trajectory.
    - Whenever the follower is in an awkward pose and you want to home it
      without killing the program first.

Usage:
    rest_arms.py                            # default ports
    rest_arms.py --follower-port /dev/ttyACM2 --leader-port /dev/ttyACM3
    rest_arms.py --hold-after                # re-engage follower torque at new pose
"""
import argparse
import sys

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader


p = argparse.ArgumentParser()
p.add_argument("--follower-port", default="/dev/so101-follower",
               help="Serial device of the SO-101 follower arm")
p.add_argument("--leader-port",   default="/dev/so101-leader",
               help="Serial device of the SO-101 leader arm")
p.add_argument("--follower-id",   default="my_follower",
               help="Calibration id used to look up the follower's calibration JSON")
p.add_argument("--leader-id",     default="my_leader",
               help="Calibration id used to look up the leader's calibration JSON")
p.add_argument("--hold-after",    action="store_true",
               help="After the user presses ENTER, re-engage follower torque "
                    "at the new pose so the arm holds position.")
args = p.parse_args()


print("Connecting follower + leader ...")
follower = SOFollower(SOFollowerRobotConfig(
    port=args.follower_port, id=args.follower_id, cameras={},
))
follower.connect(calibrate=False)

leader = SOLeader(SOLeaderTeleopConfig(
    port=args.leader_port, id=args.leader_id,
))
leader.connect(calibrate=False)

print("Connected. Releasing torque on both arms ...")
follower.bus.disable_torque()
leader.bus.disable_torque()
print("  ✓ both arms are now LIMP — you can move them by hand.\n")

print("Move BOTH arms to your rest / home position.")
input("Press ENTER when both arms are at rest ... ")

if args.hold_after:
    print("Re-engaging follower torque at the current pose ...")
    follower.bus.enable_torque()
    print("  ✓ follower torque ENABLED — arm now holds at this pose.")
else:
    print("Leaving follower torque disabled. Arm is free to move.")

try:
    follower.disconnect()
except Exception:
    pass
try:
    leader.disconnect()
except Exception:
    pass
print("Done.")
