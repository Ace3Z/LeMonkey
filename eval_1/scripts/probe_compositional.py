#!/usr/bin/env python3
"""Probe whether v2 has any compositional / spatial language signal already.

Runs the same image through several Eval-2-style prompts and measures pairwise
RMS distance between the predicted action chunks. We don't have ground truth
for "leftmost" on these training images (bowls were in fixed positions), so the
only thing we measure is: does the prompt change the action at all?

Reference scale (from probe_language_conditioning.py on v2 25k):
  • wrong_color (real signal)  ~57
  • paraphrase (overfit signal) ~61
  • empty/nonsense              ~72-86

Reading:
  • A compositional pair distance > 30 means v2 *responds* to the structure.
    SmolVLA's frozen VLM probably has some spatial signal we can lean on.
  • A pair distance < 10 means v2 treats the prompts as identical — full
    retraining needed for that concept.

Usage:
    probe_compositional.py                      # default: v2 / 025000
    probe_compositional.py 020000               # v2 at a different step
    MODEL=v1 probe_compositional.py 020000      # v1
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.control_utils import predict_action

MODEL = os.environ.get("MODEL", "v2")
MODEL_DIR = {"v1": "smolvla_eval1", "v2": "smolvla_eval1"}.get(MODEL)
if MODEL_DIR is None:
    print(f"ERROR: MODEL must be v1 or v2 (got: {MODEL})", file=sys.stderr)
    sys.exit(1)
DEFAULT_STEP = "025000" if MODEL == "v2" else "020000"
CKPT_STEP = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STEP
CKPT = f"/home/lemonkey/LeMonkey/eval_1/train/{MODEL_DIR}/checkpoints/{CKPT_STEP}/pretrained_model"
DATASETS = {
    "blue":  "/home/lemonkey/LeMonkey/datasets/eval1/blue",
    "red":   "/home/lemonkey/LeMonkey/datasets/eval1/red",
    "green": "/home/lemonkey/LeMonkey/datasets/eval1/green",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTION_STEPS = 50

# Eval 2 prompt families, exactly the styles in PROJECT.md §2.
# We don't claim a "correct" answer per image — we just measure pairwise
# distances within each family. If they're zero, v2 can't tell the prompts
# apart, period.
FAMILIES = {
    "spatial_absolute": [
        "Put the banana in the leftmost bowl.",
        "Put the banana in the middle bowl.",
        "Put the banana in the rightmost bowl.",
    ],
    "spatial_ordinal": [
        "Put the banana into the 1st bowl from the left.",
        "Put the banana into the 2nd bowl from the left.",
        "Put the banana into the 3rd bowl from the left.",
    ],
    "relational": [
        "Put the banana in the bowl on the right of the red bowl.",
        "Put the banana in the bowl on the left of the red bowl.",
    ],
    "negation": [
        "Put the banana in the bowl that is not red and not green.",
        "Put the banana in the bowl that is not blue and not green.",
        "Put the banana in the bowl that is not red and not blue.",
    ],
}


def get_first_frame(root: str):
    ds = LeRobotDataset(repo_id="local/probe", root=root)
    return ds[0]


def get_chunk(policy, pre, post, image_np, state_np, task: str):
    policy.reset()
    chunk = []
    obs0 = {
        "observation.images.front": image_np,
        "observation.state": state_np,
    }
    for _ in range(N_ACTION_STEPS):
        a = predict_action(
            obs0, policy, DEVICE, pre, post,
            use_amp=False, task=task, robot_type="so101_follower",
        )
        chunk.append(a.detach().squeeze().cpu().numpy())
    return np.stack(chunk)


def chunk_dist(a, b) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def main():
    print(f"=== Compositional probe — {MODEL} / {CKPT_STEP} ===\n")

    print("Loading policy + preprocessor...")
    policy = SmolVLAPolicy.from_pretrained(CKPT).eval().to(DEVICE)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CKPT,
        preprocessor_overrides={
            "device_processor": {"device": str(DEVICE)},
            "rename_observations_processor": {
                "rename_map": {"observation.images.front": "observation.images.camera1"}
            },
        },
    )
    print("OK\n")

    samples = {}
    for color, root in DATASETS.items():
        print(f"Loading first frame of {color}/episode_0...")
        f = get_first_frame(root)
        img_t = f["observation.images.front"]
        st_t = f["observation.state"]
        if img_t.dtype == torch.float32:
            img_np = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        else:
            img_np = img_t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        samples[color] = {"image": img_np, "state": st_t.cpu().numpy().astype(np.float32)}
    print()

    family_avgs = {}
    print("=" * 78)
    for fam_name, prompts in FAMILIES.items():
        print(f"\n=== Family: {fam_name} ({len(prompts)} prompts) ===")
        for p in prompts:
            print(f"  • \"{p}\"")
        print()

        all_pair_d = []
        for color in ["blue", "red", "green"]:
            image = samples[color]["image"]
            state = samples[color]["state"]
            chunks = [get_chunk(policy, pre, post, image, state, p) for p in prompts]

            print(f"  --- on {color}/episode_0 image — pairwise RMS distance ---")
            for i in range(len(prompts)):
                for j in range(i + 1, len(prompts)):
                    d = chunk_dist(chunks[i], chunks[j])
                    all_pair_d.append(d)
                    print(f"    P{i+1} vs P{j+1}: {d:7.2f}")
        family_avgs[fam_name] = sum(all_pair_d) / max(1, len(all_pair_d))

    print("\n" + "=" * 78)
    print("  Aggregate per-family pairwise distance (avg across 3 images, all pairs)")
    print("=" * 78)
    print()
    for fam, avg in family_avgs.items():
        if avg > 30:
            verdict = "STRONG signal — v2 distinguishes these prompts"
        elif avg > 15:
            verdict = "MODERATE — some signal, but data needed"
        elif avg > 5:
            verdict = "WEAK — barely distinguishes; will need a lot of data"
        else:
            verdict = "NONE — v2 treats these as identical; full retraining needed"
        print(f"  {fam:18s}  avg={avg:6.2f}   {verdict}")
    print()


if __name__ == "__main__":
    main()
