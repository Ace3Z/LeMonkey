#!/usr/bin/env bash
# Pre-flight checks for SO-101 robot hardware + overhead camera.
#
# Source this file from any script that talks to the real robot (record,
# rollout, structured eval). Then call the require_* function(s) you need.
#
# Why a [FATAL] preflight instead of letting lerobot-record explode:
# lerobot-record's failure modes when a USB device is missing are
# inscrutable (timeouts deep inside feetech_sdk, an OSError from cv2 on
# the camera, etc.). A 3-line check up front turns the failure into one
# clear sentence the operator can act on.
#
# Usage:
#   source "$REPO_ROOT/scripts/preflight_robot.sh"
#   require_so101_follower               # any rollout / eval / record script
#   require_so101_leader                 # teleop / record scripts only
#   require_camera                       # default /dev/video0
#   require_camera /dev/video2           # custom path
#
# Each function prints a clear message to stderr and exits 1 on failure.

require_so101_follower() {
    if [ ! -e /dev/so101-follower ]; then
        echo >&2
        echo "[FATAL] SO-101 follower arm not connected." >&2
        echo "        Expected device: /dev/so101-follower" >&2
        echo "        Fix:" >&2
        echo "          1. plug the follower arm in via USB" >&2
        echo "          2. confirm the udev rule is active (see eval_1/README.md hardware section)" >&2
        echo "          3. if the rule was just added: 'sudo udevadm trigger' then re-plug" >&2
        exit 1
    fi
}

require_so101_leader() {
    if [ ! -e /dev/so101-leader ]; then
        echo >&2
        echo "[FATAL] SO-101 leader arm not connected." >&2
        echo "        Expected device: /dev/so101-leader" >&2
        echo "        The leader is the teleoperation arm; it's needed for recording" >&2
        echo "        episodes but not for rollouts. Plug it in via USB and verify" >&2
        echo "        the udev rule (see eval_1/README.md hardware section)." >&2
        exit 1
    fi
}

require_camera() {
    local dev="${1:-/dev/video0}"
    if [ ! -e "$dev" ]; then
        echo >&2
        echo "[FATAL] overhead camera not detected at $dev." >&2
        echo "        Plug in the USB webcam (640x480 at 30 fps, mounted above the" >&2
        echo "        workspace). To list connected cameras:" >&2
        echo "          v4l2-ctl --list-devices" >&2
        echo "        If the device shows up at a different /dev/videoN, set" >&2
        echo "        CAMERA_DEV=/dev/videoN before invoking this script." >&2
        exit 1
    fi
}
