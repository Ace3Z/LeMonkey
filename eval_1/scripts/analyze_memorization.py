#!/usr/bin/env python3
"""Offline memorization risk analysis on training datasets.

Loads the 3 color datasets, computes:
  1. Action diversity per (color, prompt) group — same prompt = how different
     are the demonstrations? Low diversity = teleop was robotic = the policy
     can memorize one trajectory.
  2. Across-color trajectory distance — sanity check that different colors
     produce visibly different motions.
  3. Parameter-to-frame ratio — sanity check on overparameterization.

Outputs a HIGH / MEDIUM / LOW memorization risk verdict with evidence.

Usage:
    analyze_memorization.py
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/home/lemonkey/LeMonkey/datasets/eval1")
COLORS = ["blue", "red", "green"]
TRAINABLE_PARAMS = 360_000_000  # action expert (the only trained part)


def load_color(color):
    info = json.load(open(BASE / color / "meta/info.json"))
    fps = info["fps"]

    eps_files = sorted(glob.glob(str(BASE / color / "meta/episodes/chunk-000/*.parquet")))
    eps_df = pd.concat([pd.read_parquet(p) for p in eps_files])[["episode_index", "tasks", "length"]]

    data_files = sorted(glob.glob(str(BASE / color / "data/chunk-000/*.parquet")))
    data_df = pd.concat([pd.read_parquet(p) for p in data_files])[
        ["episode_index", "task_index", "action", "observation.state"]
    ]

    tasks_df = pd.read_parquet(BASE / color / "meta/tasks.parquet").reset_index()
    idx_to_task = dict(zip(tasks_df["task_index"], tasks_df["task"]))

    return info, fps, eps_df, data_df, idx_to_task


def episode_actions(data_df, ep_idx):
    rows = data_df[data_df["episode_index"] == ep_idx]
    return np.stack(rows["action"].values)


def episode_prompt(data_df, ep_idx, idx_to_task):
    """First frame's task_index → prompt string."""
    rows = data_df[data_df["episode_index"] == ep_idx]
    if len(rows) == 0:
        return None
    return idx_to_task.get(int(rows.iloc[0]["task_index"]), "?")


def resample(traj, n=50):
    if len(traj) < 2:
        return None
    idx_old = np.linspace(0, len(traj) - 1, len(traj))
    idx_new = np.linspace(0, len(traj) - 1, n)
    return np.stack([np.interp(idx_new, idx_old, traj[:, j]) for j in range(traj.shape[1])], axis=1)


def trajectory_distance(a, b):
    """Per-step RMS L2 between two resampled action sequences."""
    a50, b50 = resample(a), resample(b)
    if a50 is None or b50 is None:
        return float("nan")
    return float(np.linalg.norm(a50 - b50) / np.sqrt(50))


def main():
    print("=" * 70)
    print("  Memorization risk analysis")
    print("=" * 70)
    print()

    total_frames = 0
    color_traj_distances = {}
    sample_per_color = {}

    for color in COLORS:
        print(f"--- {color} ---")
        info, fps, eps_df, data_df, idx_to_task = load_color(color)
        n_eps = info["total_episodes"]
        total_frames += info["total_frames"]
        print(f"  {n_eps} episodes, {info['total_frames']} frames @ {fps} fps")

        durs = eps_df["length"] / fps
        print(f"  duration: mean={durs.mean():.1f}s  range=[{durs.min():.1f}, {durs.max():.1f}]s")

        prompt_groups = defaultdict(list)
        for ep_idx in eps_df["episode_index"]:
            p = episode_prompt(data_df, ep_idx, idx_to_task)
            if p:
                prompt_groups[p].append(ep_idx)

        print(f"  prompts seen ({len(prompt_groups)}):")
        for p, eps in prompt_groups.items():
            print(f"    [{len(eps):2d} eps] \"{p}\"")

        # Within-prompt trajectory diversity
        within_dists = []
        details = []
        for prompt, eps in prompt_groups.items():
            if len(eps) < 2:
                continue
            traj_list = [episode_actions(data_df, e) for e in eps]
            ds = []
            for i in range(len(traj_list)):
                for j in range(i + 1, len(traj_list)):
                    d = trajectory_distance(traj_list[i], traj_list[j])
                    if not np.isnan(d):
                        ds.append(d)
            if ds:
                within_dists.append(np.mean(ds))
                details.append((prompt, np.mean(ds), len(eps)))

        print(f"\n  Within-prompt trajectory distance (per-step RMS, lower = more similar)")
        for prompt, dist, n in details:
            print(f"    {dist:6.2f}  [{n:2d} eps]  \"{prompt[:48]}\"")

        avg_within = float(np.mean(within_dists)) if within_dists else 0.0
        color_traj_distances[color] = avg_within
        print(f"  → avg within-prompt distance: {avg_within:.2f}")

        # Save a sample episode action chunk for cross-color comparison
        mid_ep = eps_df.iloc[len(eps_df) // 2]["episode_index"]
        sample_per_color[color] = episode_actions(data_df, mid_ep)
        print()

    # Across-color trajectory distance
    print("--- across colors (representative episodes) ---")
    cross_dists = []
    for i, c1 in enumerate(COLORS):
        for c2 in COLORS[i + 1:]:
            d = trajectory_distance(sample_per_color[c1], sample_per_color[c2])
            cross_dists.append(d)
            print(f"  {c1:6s} <-> {c2:6s}: {d:6.2f}")
    avg_cross = float(np.mean(cross_dists))
    avg_within = float(np.mean(list(color_traj_distances.values())))
    print(f"  avg across-color distance: {avg_cross:.2f}")
    print(f"  avg within-prompt distance (recap): {avg_within:.2f}")
    print(f"  ratio across/within: {avg_cross / avg_within if avg_within else float('inf'):.2f}x")

    # Verdict
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    fpm = total_frames / TRAINABLE_PARAMS * 1e6
    print(f"  Total training frames        : {total_frames:,}")
    print(f"  Trainable params             : {TRAINABLE_PARAMS:,}  (action expert only)")
    print(f"  Frames per million params    : {fpm:.1f}")
    print(f"  Avg within-prompt traj dist  : {avg_within:.2f}  (joint-deg per timestep RMS)")
    print(f"  Avg across-color traj dist   : {avg_cross:.2f}")
    print(f"  Across-color / within-prompt : {avg_cross / avg_within:.2f}x")
    print()

    risk_factors = []
    if fpm < 200:
        risk_factors.append(f"severely overparameterized (only {fpm:.0f} frames/M params, typical is 1000+)")
    if avg_within < 5:
        risk_factors.append(f"teleop within same prompt is nearly identical (mean dist {avg_within:.1f})")
    elif avg_within < 15:
        risk_factors.append(f"teleop within same prompt is similar (mean dist {avg_within:.1f})")
    if avg_cross < avg_within * 2:
        risk_factors.append("across-color motions only slightly more different than within-prompt — model may not need language at all")

    if len(risk_factors) >= 2:
        risk = "HIGH"
    elif len(risk_factors) == 1:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    print(f"  MEMORIZATION RISK: {risk}")
    for f in risk_factors:
        print(f"    • {f}")
    print()
    print("=" * 70)
    print("  HOW TO TEST EMPIRICALLY (the gold standard)")
    print("=" * 70)
    print(
        """
  Run the structured eval (30 rollouts: 5 trained + 5 untrained prompts/color):

      ./scripts/eval_checkpoint.sh

  Then read the trained-vs-untrained breakdown:

      gap < 10pp   → real language generalization (LEARNING)
      gap 10-30pp  → partial generalization, language helps but doesn't fully drive policy
      gap > 30pp   → policy mostly memorizes specific phrasings
      OOD = 0%     → pure memorization, no language understanding
"""
    )


if __name__ == "__main__":
    main()
