#!/usr/bin/env python3
"""Render the full segmentation as a video — all frames with the 3 portrait
masks overlaid in red/green/blue.

Useful sanity check after stage 2: confirms the masks track every frame,
not just the start/end.

Usage:
    python eval_3/aug/dbg/segmentation_video.py /path/to/episode_dir
    python eval_3/aug/dbg/segmentation_video.py --root ~/LeMonkey/datasets/eval3_quick

Output: <episode_dir>/dbg_segmentation.mp4
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mu

# Local _video_io for AV1 → H.264 transcode
_HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "_video_io", str(_HERE.parent / "stages" / "video_io.py"))
_vio = importlib.util.module_from_spec(spec); spec.loader.exec_module(_vio)
ensure_h264 = _vio.ensure_h264


COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]   # BGR: green, blue, red
OCCLUDER_COLOR = (0, 255, 255)                       # BGR: yellow
# Default labels for the 3-celeb in-distribution baseline; otherwise read
# identity from portrait_seeds.json.
CELEB_BY_PID = {0: "swift", 1: "obama", 2: "lecun"}


def render_one(ep_dir: Path, *, fps: int = 30) -> dict:
    """Render dbg_segmentation.mp4 with per-frame portrait masks overlaid;
    return ``{ep, saved, n_frames}`` (or ``{ep, error}`` on failure)."""
    masks_pkl = ep_dir / "portrait_masks.pkl"
    if not masks_pkl.is_file():
        return {"ep": ep_dir.name, "error": "portrait_masks.pkl missing"}

    with open(masks_pkl, "rb") as f:
        cache = pickle.load(f)
    # Prefer the local episode's video over the cache's stale absolute path,
    # which may point at a teleop-time location that no longer exists.
    local_cands = sorted(ep_dir.glob("videos/observation.images.camera1/chunk-*/file-*.mp4"))
    local_cands = [p for p in local_cands if "__h264" not in p.name] + \
                  [p for p in local_cands if "__h264" in p.name]
    if local_cands:
        src_video = local_cands[0]
    else:
        src_video = Path(cache["video_path"])
    h264 = ensure_h264(src_video)
    cap = cv2.VideoCapture(str(h264))

    # Optional celebs from seeds.json for nicer labels
    seeds_path = ep_dir / "portrait_seeds.json"
    celeb_by_pid = dict(CELEB_BY_PID)
    if seeds_path.is_file():
        import json
        s = json.loads(seeds_path.read_text())
        celebs = s.get("celebs") or []
        for i, c in enumerate(celebs):
            celeb_by_pid[i] = c

    # Frame-0 portrait masks (constant across the clip; the inpainter's dst_mask).
    M_0_per_pid = cache.get("M_0_per_pid", {})

    work = Path(tempfile.mkdtemp(prefix="seg_video_"))
    try:
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            overlay = frame.copy()
            fmasks = cache["masks"].get(fi, {})
            # Aggregate the OCCLUDER region across all 3 portraits:
            #   occluder_pid = M_0_pid AND NOT visible_pid
            # i.e. the part of the original paper area that's currently
            # covered by something (gripper / can / hand).
            occluder_total = np.zeros(frame.shape[:2], dtype=bool)
            visible_per_pid: dict[int, np.ndarray] = {}
            for pid in (0, 1, 2):
                if pid not in fmasks:
                    continue
                vis = mu.decode(fmasks[pid]["rle"]).astype(np.uint8)
                if vis.ndim == 3:
                    vis = vis[:, :, 0]
                visible_per_pid[pid] = vis
                m0 = M_0_per_pid.get(pid)
                if m0 is None:
                    continue
                m0_bool = (m0 > 0).astype(bool)
                occluder_total |= m0_bool & (vis == 0)

            # Layer 1: tint each visible portrait in its pid color.
            for pid in (0, 1, 2):
                vis = visible_per_pid.get(pid)
                if vis is None:
                    continue
                payload = fmasks[pid]
                col = COLORS[pid]
                tinted = np.full_like(overlay, col)
                blended = cv2.addWeighted(tinted, 0.30, overlay, 0.70, 0)
                overlay = np.where(vis[:, :, None] > 0, blended, overlay)
                contours, _ = cv2.findContours(vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, col, 2)
                ys, xs = np.where(vis > 0)
                if len(ys) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    label = f"pid{pid} {celeb_by_pid.get(pid,'?')} s={payload['score']:.1f}"
                    cv2.putText(overlay, label, (cx - 70, cy + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                    cv2.putText(overlay, label, (cx - 70, cy + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            # Layer 2: paint the occluder region yellow on top — this is the
            # gripper / can / hand crossing a portrait. Drawn LAST so it sits
            # above the portrait-tint and is always visible.
            if occluder_total.any():
                occ_u8 = occluder_total.astype(np.uint8)
                tinted = np.full_like(overlay, OCCLUDER_COLOR)
                blended = cv2.addWeighted(tinted, 0.55, overlay, 0.45, 0)
                overlay = np.where(occ_u8[:, :, None] > 0, blended, overlay)
                contours, _ = cv2.findContours(occ_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, OCCLUDER_COLOR, 2)
            cv2.putText(overlay, f"frame {fi}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(overlay, f"frame {fi}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imwrite(str(work / f"f{fi:06d}.png"), overlay)
            fi += 1
        cap.release()

        out_mp4 = ep_dir / "dbg_segmentation.mp4"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(work / "f%06d.png"),
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True)
        return {"ep": ep_dir.name, "saved": str(out_mp4), "n_frames": fi}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode_dir", nargs="?", default=None,
                   help="Path to a single episode directory to render")
    p.add_argument("--root", default=None,
                   help="Root containing many episode directories to render")
    args = p.parse_args()
    if (args.episode_dir is None) == (args.root is None):
        print("[ERROR] specify one of: episode_dir, --root", file=sys.stderr)
        return 2
    eps = [Path(args.episode_dir)] if args.episode_dir else \
          sorted(p for p in Path(args.root).iterdir() if p.is_dir())
    for ep in eps:
        try:
            r = render_one(ep)
        except Exception as e:
            r = {"ep": ep.name, "error": f"{type(e).__name__}: {e}"}
        if "saved" in r:
            print(f"  ✓ {r['ep']:50s}  {r['n_frames']:>4} frames → {r['saved']}")
        else:
            print(f"  ✗ {r['ep']:50s}  {r.get('error','?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
