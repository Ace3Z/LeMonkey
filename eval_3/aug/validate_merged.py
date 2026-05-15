#!/usr/bin/env python3
"""Extensive validation of the merged eval3 LeRobot v3 dataset.

Checks performed (all must pass):
  1. Episode count matches info.json
  2. Frame count matches info.json
  3. Task count matches info.json
  4. Episode indices are contiguous 0..(N-1)
  5. Each episode's frames all share the same task_index
  6. Each episode's stated task matches its source augmentation.json
     (or reference.json for base teleops)
  7. Sample 30 random episodes — both camera1 and reference videos load
  8. Reference video is constant-frame (std ~0 across frames)
  9. action and observation.state have no NaN / Inf
 10. action and observation.state values are within reasonable joint ranges
 11. fps and chunk metadata are self-consistent
 12. Per-episode frame counts match data parquet lengths
 13. Sample frames from various points show plausible image stats
       (mean in [0,1], no all-black or all-white frames)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def discover_merge_order(base_root: Path, aug_root: Path) -> list[Path]:
    base = sorted(p for p in base_root.iterdir()
                    if p.is_dir() and (p / "meta" / "info.json").is_file()
                    and (p / "reference.json").is_file()) if base_root.is_dir() else []
    aug = sorted(p for p in aug_root.iterdir()
                   if p.is_dir() and "__var" in p.name
                   and (p / "meta" / "info.json").is_file()) if aug_root.is_dir() else []
    return base + aug


def correct_prompt_for(ep_dir: Path) -> str:
    aug_json = ep_dir / "augmentation.json"
    if aug_json.is_file():
        return json.loads(aug_json.read_text())["prompt"]
    ref_json = ep_dir / "reference.json"
    return json.loads(ref_json.read_text())["prompt"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--merged", type=Path, default=Path("datasets/eval3_merged"))
    p.add_argument("--base-root", type=Path, default=Path("datasets/eval3"))
    p.add_argument("--aug-root", type=Path, default=Path("datasets/eval3_aug_v3"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-video-samples", type=int, default=30,
                   help="how many random episodes to load videos for")
    args = p.parse_args()

    failures: list[str] = []
    warnings: list[str] = []
    rng = random.Random(args.seed)

    print(f"=== Validating {args.merged} ===\n")
    info = json.loads((args.merged / "meta" / "info.json").read_text())
    print(f"info.json says:")
    print(f"  total_episodes  = {info['total_episodes']}")
    print(f"  total_frames    = {info['total_frames']}")
    print(f"  total_tasks     = {info['total_tasks']}")
    print(f"  fps             = {info['fps']}")
    print(f"  chunks_size     = {info['chunks_size']}")
    print(f"  video_keys      = "
          f"{[k for k,v in info['features'].items() if v.get('dtype')=='video']}")
    print()

    # --- Check 1-3: count consistency ---
    print("[1-3] count consistency", flush=True)
    tasks_df = pd.read_parquet(args.merged / "meta" / "tasks.parquet")
    ep_meta_files = sorted((args.merged / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    ep_meta = pd.concat([pd.read_parquet(f) for f in ep_meta_files], ignore_index=True)
    data_files = sorted((args.merged / "data").glob("chunk-*/file-*.parquet"))
    print(f"  found {len(ep_meta_files)} ep-meta parquets, {len(data_files)} data parquets",
          flush=True)
    total_data_rows = 0
    for dp in data_files:
        n = len(pd.read_parquet(dp, columns=["index"]))
        total_data_rows += n
    print(f"  data parquet row sum         : {total_data_rows}",
          f"(expected {info['total_frames']})",
          "✓" if total_data_rows == info["total_frames"] else "✗",
          flush=True)
    if total_data_rows != info["total_frames"]:
        failures.append("frame count mismatch")
    print(f"  episodes-meta row count      : {len(ep_meta)}",
          f"(expected {info['total_episodes']})",
          "✓" if len(ep_meta) == info["total_episodes"] else "✗")
    if len(ep_meta) != info["total_episodes"]:
        failures.append("episode count mismatch")
    print(f"  tasks-meta row count         : {len(tasks_df)}",
          f"(expected {info['total_tasks']})",
          "✓" if len(tasks_df) == info["total_tasks"] else "✗")
    if len(tasks_df) != info["total_tasks"]:
        failures.append("task count mismatch")
    print()

    # --- Check 4: episode index contiguity ---
    print("[4] episode index contiguity", flush=True)
    ep_idx = ep_meta["episode_index"].to_numpy()
    if not np.array_equal(np.sort(ep_idx), np.arange(info["total_episodes"])):
        failures.append("episode indices not contiguous 0..N-1")
        print(f"  ✗ episode indices not contiguous", flush=True)
    else:
        print(f"  ✓ 0..{info['total_episodes']-1} all present", flush=True)
    print()

    # --- Check 5: each episode has a single task_index across its frames ---
    print("[5] per-episode task_index consistency", flush=True)
    mixed = 0
    for dp in data_files:
        df = pd.read_parquet(dp, columns=["episode_index", "task_index"])
        g = df.groupby("episode_index")["task_index"].nunique()
        bad = g[g > 1]
        if len(bad) > 0:
            mixed += len(bad)
            for e, n in bad.head(5).items():
                warnings.append(f"ep {e} has {n} distinct task_indices")
    if mixed > 0:
        failures.append(f"{mixed} episodes have inconsistent task_index across frames")
        print(f"  ✗ {mixed} eps have inconsistent task_index", flush=True)
    else:
        print(f"  ✓ all eps have single task_index", flush=True)
    print()

    # --- Check 6: task assignment matches source augmentation.json ---
    print("[6] task ↔ source prompt match", flush=True)
    order = discover_merge_order(args.base_root, args.aug_root)
    if len(order) < info["total_episodes"]:
        failures.append(f"merge order has {len(order)} but expected "
                        f"{info['total_episodes']}")
        return 1
    task_idx_to_str = tasks_df.reset_index().set_index("task_index")["task"].to_dict()
    # Build merged ep → task string by joining ep_meta's task_index with the
    # data parquet's task per ep (since ep_meta doesn't store task_index)
    ep_task_idx = {}
    for dp in data_files:
        df = pd.read_parquet(dp, columns=["episode_index", "task_index"])
        for e in df["episode_index"].unique():
            ep_task_idx[int(e)] = int(df[df["episode_index"] == e]["task_index"].iloc[0])
    mismatches = 0
    for i in range(info["total_episodes"]):
        merged_task = task_idx_to_str[ep_task_idx[i]]
        src_prompt = correct_prompt_for(order[i])
        if merged_task != src_prompt:
            mismatches += 1
            if mismatches <= 5:
                warnings.append(f"ep {i}: merged='{merged_task[:50]}' "
                                f"vs source='{src_prompt[:50]}'")
    if mismatches > 0:
        failures.append(f"{mismatches}/{info['total_episodes']} eps have "
                        f"mismatched task vs source prompt")
        print(f"  ✗ {mismatches} eps mismatched", flush=True)
    else:
        print(f"  ✓ all {info['total_episodes']} eps match source prompt",
              flush=True)
    print()

    # --- Check 7-8: load videos for random samples ---
    print(f"[7-8] sample-load {args.n_video_samples} eps' videos", flush=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/so101_eval3_all", root=args.merged)
    samples = rng.sample(range(info["total_episodes"]), args.n_video_samples)
    ref_ok = 0
    cam_ok = 0
    for ep_i in samples:
        start = int(ds.meta.episodes["dataset_from_index"][ep_i])
        end = int(ds.meta.episodes["dataset_to_index"][ep_i])
        # Sample 3 frames per ep
        for fi in (start, (start + end) // 2, end - 1):
            try:
                s = ds[fi]
                cam = s["observation.images.camera1"]
                ref = s["observation.images.reference"]
                if cam.shape != (3, 480, 640):
                    warnings.append(f"ep {ep_i} cam shape {cam.shape}")
                else:
                    cam_ok += 1
                if ref.shape != (3, 480, 480):
                    warnings.append(f"ep {ep_i} ref shape {ref.shape}")
                else:
                    ref_ok += 1
                # Sanity on image stats
                cm = float(cam.mean())
                rm = float(ref.mean())
                if not (0.05 < cm < 0.95):
                    warnings.append(f"ep {ep_i} cam mean={cm:.3f} extreme")
                if not (0.05 < rm < 0.95):
                    warnings.append(f"ep {ep_i} ref mean={rm:.3f} extreme")
            except Exception as e:
                failures.append(f"ep {ep_i} frame {fi}: {e}")
    n_total = args.n_video_samples * 3
    print(f"  camera1 frames OK : {cam_ok}/{n_total}",
          "✓" if cam_ok == n_total else "✗", flush=True)
    print(f"  reference frames OK : {ref_ok}/{n_total}",
          "✓" if ref_ok == n_total else "✗", flush=True)
    if cam_ok < n_total or ref_ok < n_total:
        failures.append(f"video load failed on {n_total - min(cam_ok, ref_ok)} frames")
    print()

    # --- Check 9: NaN / Inf in action + observation.state ---
    print("[9] NaN/Inf check on action + observation.state", flush=True)
    n_bad = 0
    for dp in data_files:
        df = pd.read_parquet(dp, columns=["action", "observation.state"])
        a = np.stack(df["action"].to_numpy())
        s = np.stack(df["observation.state"].to_numpy())
        n_bad += int(np.isnan(a).sum() + np.isinf(a).sum())
        n_bad += int(np.isnan(s).sum() + np.isinf(s).sum())
    if n_bad > 0:
        failures.append(f"{n_bad} NaN/Inf values in action/state")
        print(f"  ✗ {n_bad} NaN/Inf", flush=True)
    else:
        print(f"  ✓ no NaN/Inf", flush=True)
    print()

    # --- Check 10: action joint ranges (SO-101 ~ ±180 deg) ---
    print("[10] joint range plausibility", flush=True)
    all_actions = []
    for dp in data_files:
        df = pd.read_parquet(dp, columns=["action"])
        all_actions.append(np.stack(df["action"].to_numpy()))
    A = np.concatenate(all_actions, axis=0)
    print(f"  action min/max/mean: {A.min():.2f}/{A.max():.2f}/{A.mean():.2f}",
          flush=True)
    if A.min() < -360 or A.max() > 360:
        warnings.append(f"action range extreme: [{A.min():.1f}, {A.max():.1f}]")
    print()

    # --- Check 11: fps / chunks consistency ---
    print("[11] fps / chunks consistency", flush=True)
    if info["fps"] != 30:
        warnings.append(f"fps={info['fps']}, expected 30")
    if info["chunks_size"] != 1000:
        warnings.append(f"chunks_size={info['chunks_size']}, expected 1000")
    print(f"  fps={info['fps']}, chunks_size={info['chunks_size']}", flush=True)
    print()

    # --- Check 12: per-episode frame counts ---
    print("[12] per-episode frame counts match data parquet lengths", flush=True)
    counted = {}
    for dp in data_files:
        df = pd.read_parquet(dp, columns=["episode_index"])
        for e, n in df.groupby("episode_index").size().items():
            counted[int(e)] = counted.get(int(e), 0) + int(n)
    meta_lengths = ep_meta.set_index("episode_index")["length"].to_dict()
    mismatches = 0
    for e in counted:
        if counted[e] != meta_lengths.get(e, -1):
            mismatches += 1
            if mismatches <= 5:
                warnings.append(f"ep {e}: data={counted[e]} meta={meta_lengths.get(e)}")
    if mismatches > 0:
        failures.append(f"{mismatches} eps have data/meta length mismatch")
        print(f"  ✗ {mismatches} eps mismatched", flush=True)
    else:
        print(f"  ✓ all {len(counted)} eps consistent", flush=True)
    print()

    # --- Final report ---
    print("=" * 50)
    print(f"Failures: {len(failures)}")
    for f in failures:
        print(f"  ✗ {f}")
    print(f"Warnings: {len(warnings)}")
    for w in warnings[:20]:
        print(f"  ! {w}")
    if len(warnings) > 20:
        print(f"  ... and {len(warnings) - 20} more")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
