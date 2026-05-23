#!/usr/bin/env python3
"""List Eval 3 episodes on disk — what we have, where the videos live.

By default scans ~/LeMonkey/datasets/eval3_quick/ and ~/LeMonkey/datasets/eval3/
(the smoke-record and main-collection dirs). For each episode it reads the
reference.json sidecar and the video file, prints a table.

Usage:
    list_eval3_episodes.py
    list_eval3_episodes.py --root ~/LeMonkey/datasets/eval3_quick
    list_eval3_episodes.py --paths-only       # just the video paths, one per line
    list_eval3_episodes.py --open             # xdg-open each video (needs a display)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOTS = [
    Path("/home/lemonkey/LeMonkey/datasets/eval3_quick"),
    Path("/home/lemonkey/LeMonkey/datasets/eval3"),
]


def human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def find_video(ep_dir: Path) -> Path | None:
    """LeRobot v3 puts the camera video at videos/observation.images.<key>/chunk-000/file-000.mp4."""
    for p in ep_dir.glob("videos/*/chunk-*/file-*.mp4"):
        return p
    return None


def list_root(root: Path, paths_only: bool, want_open: bool) -> int:
    if not root.is_dir():
        return 0
    eps = sorted(p for p in root.iterdir() if p.is_dir())
    if not eps:
        return 0

    n_total = 0
    if not paths_only:
        print(f"\n=== {root} ({len(eps)} dirs) ===")
    for ep in eps:
        ref_path = ep / "reference.json"
        video = find_video(ep)
        if paths_only:
            if video:
                print(video)
            continue
        meta = {}
        if ref_path.is_file():
            try:
                meta = json.loads(ref_path.read_text())
            except Exception:
                pass
        print()
        print(f"  {ep.name}")
        if meta:
            tgt = meta.get("target_celeb_name", meta.get("target_celeb", "?"))
            layout = meta.get("layout", "?")
            ti = meta.get("target_idx")
            pos = ["LEFT", "MIDDLE", "RIGHT"][ti] if ti is not None else "?"
            print(f"    target  : {tgt}    layout: {layout}    position: {pos}")
            print(f"    prompt  : {meta.get('prompt','?')!r}")
            ref_photo = meta.get("reference_photo")
            print(f"    ref     : {ref_photo or '(none)'}")
        else:
            print("    (no reference.json — recorded outside record_quick.py)")
        if video and video.is_file():
            print(f"    video   : {video}")
            print(f"    size    : {human_size(video.stat().st_size)}")
            if want_open:
                subprocess.Popen(
                    ["xdg-open", str(video)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
        else:
            print("    video   : (no .mp4 found — episode probably failed mid-record)")
        n_total += 1
    return n_total


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", action="append",
                    help="Dataset root to scan (repeatable). Default: eval3_quick + eval3")
    ap.add_argument("--paths-only", action="store_true",
                    help="Only print video paths, one per line (script-friendly)")
    ap.add_argument("--open", dest="want_open", action="store_true",
                    help="xdg-open each video (needs a display)")
    args = ap.parse_args()

    roots = [Path(r).expanduser() for r in args.root] if args.root else DEFAULT_ROOTS
    grand_total = 0
    for r in roots:
        grand_total += list_root(r, args.paths_only, args.want_open)

    if not args.paths_only:
        print()
        if grand_total == 0:
            print("(nothing recorded yet)")
        else:
            print(f"= {grand_total} episode(s) total")
            print()
            print("Open one in your default video viewer:")
            print("  xdg-open <video-path>")
            print("Or all of them at once:")
            print("  python eval_3/scripts/record/list_episodes.py --open")
            print("Or pipe the paths into another tool:")
            print("  python eval_3/scripts/record/list_episodes.py --paths-only | xargs -n1 -I{} echo {}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
