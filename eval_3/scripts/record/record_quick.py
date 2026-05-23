#!/usr/bin/env python3
"""Quick smoke-record for Eval 3 — record N episodes of the full task.

This script collects a handful of episodes (default 5) of the full Eval 3
task — pick up the Coke can, place it on a target celebrity's portrait —
so we can iterate on the inpainting augmentation pipeline.

NOT a balanced-plan recorder — this is single-config, ad-hoc, you specify
the celeb / layout / prompt at the CLI.

Usage:
    # minimal — defaults to swift, layout SOL, default prompt, 5 episodes
    record_quick.py --target swift --layout SOL

    # custom prompt
    record_quick.py --target obama --layout OLS \\
                         --prompt "Put the coke can on Barack Obama."

    # arbitrary celeb (e.g. for the supplementary extra-celeb demos)
    record_quick.py --target federer --layout - \\
                         --prompt "Place the coke on Roger Federer." \\
                         --target-name "Roger Federer" \\
                         --reference-photo /path/to/federer_ref.jpg

    # dry-run (no robot needed, walks the loop)
    record_quick.py --target swift --layout SOL --dry-run

Controls during recording:
    ENTER   record next episode (subprocess: lerobot-record for 20 s)
    d       delete the LAST recorded episode
    q       quit (state auto-saved)

Output schema produces a LeRobot v3 dataset dir per episode plus a sidecar
reference.json so the augmentation pipeline can treat episodes uniformly.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Map of supported short keys ↔ display names. Extra celebs can be added on
# the fly via --target / --target-name.
KNOWN_CELEBS = {
    "swift": "Taylor Swift",
    "obama": "Barack Obama",
    "lecun": "Yann LeCun",
}
KNOWN_INITIALS = {"swift": "S", "obama": "O", "lecun": "L"}
INITIAL_TO_KEY = {v: k for k, v in KNOWN_INITIALS.items()}


def layout_to_human(layout: str) -> str:
    """SOL → 'Swift / Obama / LeCun  (left / middle / right)'."""
    if layout == "-":
        return "(layout not specified — record whatever's on the table)"
    if len(layout) != 3 or any(c not in KNOWN_INITIALS.values() for c in layout):
        return f"(custom layout {layout!r}; not parsed)"
    parts = [KNOWN_CELEBS[INITIAL_TO_KEY[c]] for c in layout]
    return "  /  ".join(parts) + "    (left / middle / right)"


def target_idx_in_layout(layout: str, target_key: str) -> int | None:
    """Where in the layout sits the target? None if unknown."""
    initial = KNOWN_INITIALS.get(target_key)
    if initial is None or layout == "-" or initial not in layout:
        return None
    return layout.index(initial)


def write_reference_sidecar(run_path: Path, payload: dict) -> None:
    """Write the per-episode reference.json sidecar (target celeb, layout, prompt, etc.)."""
    sidecar = run_path / "reference.json"
    sidecar.write_text(json.dumps(payload, indent=2))


def main() -> int:
    """Loop through N quick-recording episodes, invoking lerobot-record per episode and writing the reference sidecar."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target", required=True,
                   help="Short celeb key (swift / obama / lecun) OR an arbitrary key for extra celebs")
    p.add_argument("--target-name", default=None,
                   help="Full display name; defaults to KNOWN_CELEBS[target] if target is a known key")
    p.add_argument("--layout", default="-",
                   help="3-letter layout (e.g. SOL, OLS) describing left/middle/right portraits, "
                        "or '-' to skip. Used only for the sidecar metadata.")
    p.add_argument("--prompt", default=None,
                   help="Override the prompt text (default: 'Place the coke on <name>.')")
    p.add_argument("--reference-photo", default=None,
                   help="Path to a held-out reference photo of the target celeb. Stored in the sidecar "
                        "for the augmentation / training pipelines. Optional — leave blank for now if "
                        "you'll add it later.")
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--episode-time-s", type=float, default=20.0)
    p.add_argument("--reset-time-s", type=float, default=10.0)
    p.add_argument("--root", default=str(Path.home() / "LeMonkey/datasets/eval3_quick"),
                   help="Output dir for episode subdirs. Separate from the main eval3/ dir so the "
                        "smoke-record episodes can't accidentally pollute the main collection.")
    p.add_argument("--leader-port", default="/dev/so101-leader",
                   help="udev path to the SO-101 leader arm serial device (default: /dev/so101-leader).")
    p.add_argument("--leader-id", default="my_leader",
                   help="lerobot-record teleop ID for the leader arm (default: my_leader).")
    p.add_argument("--follower-port", default="/dev/so101-follower",
                   help="udev path to the SO-101 follower arm serial device (default: /dev/so101-follower).")
    p.add_argument("--follower-id", default="my_follower",
                   help="lerobot-record robot ID for the follower arm (default: my_follower).")
    p.add_argument("--cam-path", default="/dev/video0",
                   help="V4L2 device node for the wrist camera (default: /dev/video0).")
    p.add_argument("--dry-run", action="store_true",
                   help="Walk the loop without invoking lerobot-record (no robot needed)")
    args = p.parse_args()

    target_key = args.target.strip().lower()
    target_name = args.target_name or KNOWN_CELEBS.get(target_key, args.target)
    layout = args.layout.upper() if args.layout and args.layout != "-" else "-"
    prompt = args.prompt or f"Place the coke on {target_name}."
    target_idx = target_idx_in_layout(layout, target_key)

    if args.reference_photo:
        ref_path = Path(args.reference_photo).expanduser()
        if not ref_path.is_file():
            print(f"[WARN] reference photo path does not exist: {ref_path}\n"
                  f"       proceeding anyway — you can replace it later before training.")
            args.reference_photo = str(ref_path)  # store the intended path
        else:
            args.reference_photo = str(ref_path.resolve())

    Path(args.root).mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  Eval 3 quick smoke-record")
    print(f"  output root      : {args.root}")
    print(f"  target           : {target_key}    ({target_name})")
    print(f"  layout           : {layout}    {layout_to_human(layout)}")
    if target_idx is not None:
        print(f"  target idx       : {target_idx}  ({['LEFT','MIDDLE','RIGHT'][target_idx]})")
    print(f"  prompt           : {prompt!r}")
    print(f"  reference photo  : {args.reference_photo or '(not specified)'}")
    print(f"  num episodes     : {args.num_episodes}")
    print(f"  episode time     : {args.episode_time_s}s")
    print(f"  dry run          : {args.dry_run}")
    print("=" * 72)
    print()

    last_recorded: list[Path] = []  # stack of recorded run_paths for the 'd' command
    i = 0
    while i < args.num_episodes:
        ep_no = i + 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"quick_{target_key}_{layout}_ep{ep_no:02d}_{ts}"
        run_path = Path(args.root) / run_name

        print("─" * 72)
        print(f" Episode {ep_no} / {args.num_episodes}")
        print(f"   target  : {target_name}")
        print(f"   prompt  : \"{prompt}\"")
        if target_idx is not None:
            print(f"   place the can on the {['LEFT','MIDDLE','RIGHT'][target_idx]} portrait")
        print(f"   saving to: {run_path}")
        print("─" * 72)

        try:
            ans = input("ENTER=record / 'd'=delete last / 'q'=quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0

        if ans == "q":
            return 0
        if ans == "d":
            if not last_recorded:
                print("  (nothing recorded yet to delete)")
                continue
            last = last_recorded.pop()
            if last.exists():
                shutil.rmtree(last)
                print(f"  🗑  removed {last}")
            else:
                print(f"  [WARN] last recorded path missing: {last}")
            i -= 1  # roll back the episode counter
            continue

        # Record
        if args.dry_run:
            print(f"  [dry-run] would record → {run_path}")
            run_path.mkdir(parents=True, exist_ok=True)
        else:
            cmd = [
                "lerobot-record",
                "--robot.type=so101_follower",
                f"--robot.port={args.follower_port}",
                f"--robot.id={args.follower_id}",
                f"--robot.cameras={{ camera1: {{type: opencv, index_or_path: {args.cam_path}, width: 640, height: 480, fps: 30}}}}",
                "--display_data=true",
                "--teleop.type=so101_leader",
                f"--teleop.port={args.leader_port}",
                f"--teleop.id={args.leader_id}",
                f"--dataset.repo_id=local/eval3_quick_{run_name}",
                f"--dataset.root={run_path}",
                "--dataset.num_episodes=1",
                f"--dataset.episode_time_s={args.episode_time_s}",
                f"--dataset.reset_time_s={args.reset_time_s}",
                f"--dataset.single_task={prompt}",
                "--dataset.streaming_encoding=true",
                "--dataset.encoder_threads=2",
                "--dataset.push_to_hub=false",
            ]
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"  ⚠️  lerobot-record exited rc={rc} — episode NOT counted; press 'd' "
                      f"if you want to remove a partial dataset on disk.")
                continue

        # Sidecar — same schema as record_quick.py so downstream tooling is uniform
        write_reference_sidecar(run_path, {
            "episode_idx": ep_no,
            "target_celeb": target_key,
            "target_celeb_name": target_name,
            "layout": layout,
            "target_idx": target_idx,
            "reference_photo": args.reference_photo,
            "prompt": prompt,
            "source": "quick",   # marks origin so the merger can distinguish later
            "timestamp": ts,
        })
        last_recorded.append(run_path)
        print(f"  ✓ saved → {run_path}")
        print(f"  ✓ wrote reference sidecar → {run_path / 'reference.json'}")
        i += 1

    print(f"\n🎉  recorded {args.num_episodes} episodes under {args.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
