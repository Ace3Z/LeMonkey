#!/usr/bin/env python3
"""Verify the Track D checkpoint loads as a plain SmolVLAPolicy and accepts
the inference contract (camera1 + state + task only) — without needing a
robot connected.

Run on Brev / any GPU box:

    python eval_3/scripts/verify_policy_load.py
    python eval_3/scripts/verify_policy_load.py --revision step-5000
    python eval_3/scripts/verify_policy_load.py --path /local/pretrained_model

What it checks:
1. `SmolVLAPolicy.from_pretrained(revision=...)` succeeds without any M2 imports
2. The loaded class is `SmolVLAPolicy` (not the M2 wrapper)
3. One forward pass works with a dummy batch (the rollout will pass real data
   shaped the same way)
4. `select_action` returns a tensor of the expected action dim
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="HBOrtiz/smolvla_eval3_track_D_m2_mahbod")
    p.add_argument("--revision", default="step-10000")
    p.add_argument("--path", default=None, help="Local pretrained_model dir; overrides --repo/--revision.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"[verify] device={args.device}", flush=True)

    # --- 1. Load policy without importing any M2 code -------------------
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    t0 = time.time()
    if args.path:
        print(f"[verify] loading from local path: {args.path}", flush=True)
        policy = SmolVLAPolicy.from_pretrained(args.path)
    else:
        print(f"[verify] loading {args.repo}@{args.revision}", flush=True)
        policy = SmolVLAPolicy.from_pretrained(args.repo, revision=args.revision)
    policy = policy.to(args.device).eval()
    print(f"[verify] loaded in {time.time()-t0:.1f}s; class={type(policy).__name__}", flush=True)

    assert type(policy).__name__ == "SmolVLAPolicy", (
        f"expected SmolVLAPolicy, got {type(policy).__name__} — the HF artifact may include the M2 wrapper")

    # --- 2. Inspect the config to confirm the inference contract -------
    cfg = policy.config
    print(f"[verify] empty_cameras={cfg.empty_cameras}  "
          f"vlm_model_name={cfg.vlm_model_name}", flush=True)
    print(f"[verify] input_features keys: {sorted(cfg.input_features.keys())}", flush=True)
    print(f"[verify] output_features keys: {sorted(cfg.output_features.keys())}", flush=True)

    # --- 3. Build the inference-time preprocessor ----------------------
    # lerobot-record runs policy_preprocessor.json on every step to:
    #  - normalize the image (IDENTITY for SmolVLA — passthrough)
    #  - tokenize `task` → observation.language.tokens / .attention_mask
    #  - normalize state via MEAN_STD stats
    # select_action then expects the tokenized batch.
    from lerobot.processor.pipeline import DataProcessorPipeline

    if args.path:
        preprocessor = DataProcessorPipeline.from_pretrained(
            args.path, config_filename="policy_preprocessor.json"
        )
    else:
        preprocessor = DataProcessorPipeline.from_pretrained(
            args.repo, config_filename="policy_preprocessor.json",
            revision=args.revision,
        )
    print(f"[verify] loaded preprocessor: {len(preprocessor.steps)} steps", flush=True)

    # --- 4. One forward pass with a dummy batch -------------------------
    bs = 1
    img_key = "observation.images.camera1"
    state_dim = next((v.shape[0] for k, v in cfg.input_features.items()
                      if k.endswith("state")), 6)
    print(f"[verify] running forward pass: bs={bs} state_dim={state_dim}", flush=True)

    batch = {
        img_key: torch.zeros(bs, 3, 480, 640, device=args.device),
        "observation.state": torch.zeros(bs, state_dim, device=args.device),
        "task": "Place the coke on Taylor Swift.",
    }
    batch = preprocessor(batch)

    with torch.inference_mode():
        action = policy.select_action(batch)

    print(f"[verify] select_action returned tensor: shape={tuple(action.shape)} "
          f"dtype={action.dtype}", flush=True)

    assert action.ndim == 2 and action.shape[0] == bs, (
        f"expected (B, action_dim) tensor, got shape {tuple(action.shape)}")
    assert not torch.isnan(action).any(), "action tensor contains NaN"

    print("[verify] OK — checkpoint loads cleanly and select_action runs.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
