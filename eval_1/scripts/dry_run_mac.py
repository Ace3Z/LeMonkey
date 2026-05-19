#!/usr/bin/env python3
"""Headless inference dry-run for the Mac deploy path.

Loads the SmolVLA policy and its pre/postprocessors exactly the way
lerobot-record does, then runs `predict_action` against a synthetic
observation (random image, random state, real prompt). No robot or
camera required.

Confirms:
  * checkpoint loads
  * MPS device works end-to-end
  * preprocessor pipeline (incl. language tokenization) works
  * postprocessor pipeline yields a 6-D action
  * steady-state per-action latency

Usage:
    python dry_run_mac.py                              # default ckpt step 020000
    python dry_run_mac.py 015000                       # different ckpt step
    CKPT=/path/to/pretrained_model python dry_run_mac.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.control_utils import predict_action

DEFAULT_CKPT_STEP = "020000"
DEFAULT_PROMPT = "Put the banana in the blue colored bowl."


def main() -> int:
    ckpt_step = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT_STEP
    ckpt = os.environ.get(
        "CKPT",
        f"/Volumes/T7/LeMonkey/models/smolvla_eval1_v2/checkpoints/{ckpt_step}/pretrained_model",
    )
    if not os.path.isdir(ckpt):
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    print(f"ckpt:   {ckpt}")

    t = time.time()
    policy = SmolVLAPolicy.from_pretrained(ckpt).to(device)
    policy.eval()
    print(f"policy loaded + moved to {device} in {time.time() - t:.1f}s "
          f"({sum(p.numel() for p in policy.parameters()) / 1e6:.0f}M params)")

    # The checkpoint's device_processor step is pinned to 'cuda' (training device);
    # override to our actual device since get_safe_torch_device('cuda') asserts.
    device_override = {"device_processor": {"device": device.type}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt,
        preprocessor_overrides=device_override,
        postprocessor_overrides=device_override,
    )
    print("pre/postprocessor pipelines loaded (device_processor overridden to "
          f"{device.type})")

    # Synthetic observation matching the eval_1 schema (640x480 wrist cam, 6-DOF state)
    rng = np.random.default_rng(0)
    obs = {
        "observation.images.camera1": rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "observation.state": rng.standard_normal(6).astype(np.float32),
    }

    print(f"prompt: {DEFAULT_PROMPT!r}")
    print("warm-up forward pass ...", flush=True)
    t = time.time()
    action = predict_action(
        observation=obs,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=False,
        task=DEFAULT_PROMPT,
        robot_type="so101_follower",
    )
    print(f"  shape={tuple(action.shape)}  dtype={action.dtype}  device={action.device}  "
          f"first-pass={time.time() - t:.2f}s")
    print(f"  sample: {action.cpu().numpy().tolist()}")

    def time_action(reset_queue: bool) -> float:
        if reset_queue:
            policy.reset()  # forces a chunk recompute on next call
        t0 = time.time()
        predict_action(
            observation=obs,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=False,
            task=DEFAULT_PROMPT,
            robot_type="so101_follower",
        )
        return time.time() - t0

    # Two regimes: chunk-pops (cheap, dominate the 30Hz loop) and chunk-recomputes
    # (the model actually runs; happens every ~50 frames).
    n = 5
    print(f"chunk-pop latency ({n} consecutive calls, no reset):", flush=True)
    pop_times = [time_action(reset_queue=False) for _ in range(n)]
    print(f"  min={1000 * min(pop_times):.0f}ms  median={1000 * sorted(pop_times)[n // 2]:.0f}ms  "
          f"max={1000 * max(pop_times):.0f}ms")

    print(f"chunk-recompute latency ({n} calls, reset before each):", flush=True)
    recompute_times = [time_action(reset_queue=True) for _ in range(n)]
    print(f"  min={1000 * min(recompute_times):.0f}ms  median={1000 * sorted(recompute_times)[n // 2]:.0f}ms  "
          f"max={1000 * max(recompute_times):.0f}ms")

    chunk_size = 50  # SmolVLA default
    avg_ms = (chunk_size - 1) * (1000 * sum(pop_times) / len(pop_times)) / chunk_size + \
             (1000 * sum(recompute_times) / len(recompute_times)) / chunk_size
    print(f"effective avg per-frame (1 recompute per {chunk_size} pops): {avg_ms:.1f}ms  "
          f"→ {1000 / avg_ms:.0f} Hz sustainable")
    print("✓ dry run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
