#!/usr/bin/env python3
"""Live camera preview via rerun's viewer.

Useful for composing the workspace before recording — set up your portrait
semicircle, place the can, check that everything's framed and well-lit
through the wrist camera before you press ENTER on `record_quick.py`.

Usage:
    python preview_camera.py                           # /dev/video0, no rotation
    python preview_camera.py --rotate 180              # flip if camera is mounted upside-down
    python preview_camera.py --device-path /dev/video1
    python preview_camera.py --serve                   # run as a rerun web server (browser instead of native window)

Ctrl-C to quit.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import rerun as rr


def main() -> int:
    """Open a V4L2 camera, log frames to rerun (native viewer or web server), and report rolling FPS."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device-path", default="/dev/video0",
                   help="V4L2 video device node to open (default: /dev/video0).")
    p.add_argument("--width", type=int, default=640,
                   help="Requested capture width in pixels (default: 640).")
    p.add_argument("--height", type=int, default=480,
                   help="Requested capture height in pixels (default: 480).")
    p.add_argument("--fps", type=int, default=30,
                   help="Requested capture frame rate (default: 30).")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="Rotate frames CCW by N degrees before logging")
    p.add_argument("--serve", action="store_true",
                   help="Run rerun as a web server (browser-accessible) instead of spawning a native window")
    args = p.parse_args()

    rot_map = {
        0:   None,
        90:  cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }
    rot_code = rot_map[args.rotate]

    rr.init("eval3_camera_preview")
    if args.serve:
        rr.serve_web(open_browser=False)
        print("\n* rerun web server running. Open this in a browser:")
        print("    http://<host-or-localhost>:9090\n")
    else:
        rr.spawn()

    cap = cv2.VideoCapture(args.device_path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.device_path}", file=sys.stderr)
        return 1

    # warm up auto-exposure
    for _ in range(5):
        cap.read()

    print(f"streaming {args.width}x{args.height}@{args.fps} from {args.device_path}  rotate={args.rotate}°")
    print("Ctrl-C to stop.")

    n_frames = 0
    t0 = time.time()
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue
            if rot_code is not None:
                frame_bgr = cv2.rotate(frame_bgr, rot_code)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rr.log("camera1", rr.Image(frame_rgb))
            n_frames += 1
            if n_frames % args.fps == 0:
                fps_now = n_frames / (time.time() - t0)
                # log the running fps as a scalar so you see it in rerun's timeline
                rr.log("stats/fps", rr.Scalars(fps_now))
    except KeyboardInterrupt:
        print(f"\nstopped after {n_frames} frames ({n_frames/(time.time()-t0):.1f} fps avg)")
    finally:
        cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
