#!/usr/bin/env python3
"""Convert DAgger datasets to match the BC dataset schema so they can be
merged with `lerobot-edit-dataset --operation.type=merge`.

Schema delta DAgger → BC:
  - rename feature key `observation.images.camera1` → `observation.images.front`
  - drop `is_intervention` feature (BC has no such column)
  - set `observation.state.names` and `action.names` to BC values

Touches: videos/ dir name, data/*.parquet column drop, meta/info.json,
meta/episodes/*.parquet column rename + drop.

Usage:
    normalize_dagger_to_bc_schema.py --src datasets/eval1_dagger \\
                                     --dst datasets/eval1_dagger_norm
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def convert_color(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"  [WARN] {dst} exists, removing for fresh copy", flush=True)
        shutil.rmtree(dst)
    print(f"  copying {src} → {dst}", flush=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".cache"))

    # 1. videos dir rename
    old_vid = dst / "videos" / "observation.images.camera1"
    new_vid = dst / "videos" / "observation.images.front"
    if old_vid.exists():
        old_vid.rename(new_vid)
    elif new_vid.exists():
        print(f"  [WARN] {old_vid} missing but {new_vid} exists; assuming already renamed", flush=True)
    else:
        raise RuntimeError(f"video dir not found at {old_vid} or {new_vid}")

    # 2. drop is_intervention from data parquets
    data_pqs = list((dst / "data").rglob("*.parquet"))
    for p in data_pqs:
        df = pd.read_parquet(p)
        before = list(df.columns)
        if "is_intervention" in df.columns:
            df = df.drop(columns=["is_intervention"])
        df.to_parquet(p, index=False)
    print(f"    dropped is_intervention from {len(data_pqs)} data parquet(s)", flush=True)

    # 3. rewrite meta/info.json
    info_p = dst / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    feats = info["features"]
    if "observation.images.camera1" in feats:
        feats["observation.images.front"] = feats.pop("observation.images.camera1")
    feats.pop("is_intervention", None)
    feats["observation.state"]["names"] = NAMES
    feats["action"]["names"] = NAMES
    info_p.write_text(json.dumps(info, indent=2))
    print(f"    rewrote {info_p}", flush=True)

    # 4. rewrite meta/episodes/*.parquet
    ep_pqs = list((dst / "meta" / "episodes").rglob("*.parquet"))
    for p in ep_pqs:
        df = pd.read_parquet(p)
        rename = {}
        drop = []
        for c in df.columns:
            if "observation.images.camera1" in c:
                rename[c] = c.replace("observation.images.camera1", "observation.images.front")
            elif c.startswith("stats/is_intervention/"):
                drop.append(c)
        if drop:
            df = df.drop(columns=drop)
        if rename:
            df = df.rename(columns=rename)
        df.to_parquet(p, index=False)
    print(f"    rewrote {len(ep_pqs)} episodes parquet(s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="parent dir containing per-color DAgger subdirs (blue/red/green)")
    ap.add_argument("--dst", type=Path, required=True,
                    help="parent dir to write normalized copies under")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    for color in ["blue", "red", "green"]:
        src = args.src / color
        dst = args.dst / color
        if not src.exists():
            print(f"  [WARN] {src} missing, skipping", flush=True)
            continue
        convert_color(src, dst)
    print("done.", flush=True)
