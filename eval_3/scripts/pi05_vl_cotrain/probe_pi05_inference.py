#!/usr/bin/env python3
"""Pi0.5 inference VRAM + latency probe (target: 16 GB GPU, 20 s budget).

Standalone — does NOT need the SO-101 robot connected. Loads
`lerobot/pi05_base` (or whatever --policy-path you pass), constructs a
dummy inference batch matching the demo-day input contract (camera1 +
state + task prompt), runs N warm-up + N timed forwards, and reports:

  - peak VRAM allocated (via torch.cuda.max_memory_allocated)
  - mean + p50 + p95 + p99 latency per forward
  - whether the model OOMs at all

PASS criteria for Pi0.5 VL cotrain / 3 deployment:
  peak_vram_gb  <  14.0       (leaves 2 GB headroom on the 16 GB target)
  p95_latency_s <  20.0       (TA rule for rollout time)

If both pass, green light for the Pi0.5 path.
If VRAM peaks above 14 GB the Pi0.5 path will OOM at inference; use the
SmolVLA checkpoint instead.
If latency > 20 s, demo-day rollouts will be disqualified.

Per: emit [WARN] on any fallback / unexpected condition.

USAGE
=====

    # 1. Make sure HF token is set (for downloading pi05_base if not cached):
    export HF_TOKEN=...   # not strictly needed; pi05_base is public

    # 2. Activate the lemonkey env:
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate lemonkey

    # 3. Run the probe:
    python eval_3/scripts/pi05_vl_cotrain/probe_pi05_inference.py
       # default: lerobot/pi05_base, 5 warm-up + 5 timed forwards

    # Or with a specific checkpoint (e.g. warm-PG):
    python eval_3/scripts/pi05_vl_cotrain/probe_pi05_inference.py \\
        --policy-path HBOrtiz/paligemma_vqa_warm

    # 4. Read the PASS/FAIL summary at the bottom.

    # 5. In a second terminal during the run, you can also watch:
    watch -n 0.5 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv'
"""
from __future__ import annotations

import argparse
import sys
import time

import torch


# PASS thresholds from the ObjectVLA spec.
DEFAULT_VRAM_LIMIT_GB = 14.0     # 2 GB headroom on the 16 GB GPU target
DEFAULT_LATENCY_LIMIT_S = 20.0   # TA rule


def make_dummy_observation(device: torch.device, dtype: torch.dtype = torch.float32) -> dict:
    """Match the demo-day input contract for Pi0.5:
        camera1 (480x640x3 RGB)  +  state (6-d proprio)  +  task (text)
    """
    return {
        "observation.images.right_wrist_0_rgb": torch.rand(
            1, 3, 480, 640, device=device, dtype=dtype
        ),
        "observation.state": torch.zeros(1, 6, device=device, dtype=dtype),
        "task": ["Place the can on the photo of Yann LeCun."],
    }


def probe(policy_path: str, n_warmup: int = 5, n_timed: int = 5,
          vram_limit_gb: float = DEFAULT_VRAM_LIMIT_GB,
          latency_limit_s: float = DEFAULT_LATENCY_LIMIT_S) -> int:
    """Load the policy, run warm-up + timed forwards, and report PASS/FAIL against VRAM and latency limits."""
    if not torch.cuda.is_available():
        print("[ERR] expected=CUDA GPU, got=cpu-only, fallback=abort probe",
              file=sys.stderr)
        return 2

    device = torch.device("cuda")
    print(f"[info] device: {torch.cuda.get_device_name(0)}")
    print(f"[info] total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"[info] loading policy: {policy_path}")
    t0 = time.time()

    # lerobot's Pi05Policy loader. The import path may vary by lerobot version.
    try:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    except ImportError:
        try:
            from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy
        except ImportError as e:
            print(f"[ERR] cannot import PI05Policy from lerobot. "
                  f"Got: {e}. fallback=abort probe", file=sys.stderr)
            return 3

    try:
        policy = PI05Policy.from_pretrained(policy_path).to(device).eval()
    except Exception as e:
        print(f"[ERR] failed to load policy: {e}", file=sys.stderr)
        return 4
    load_time = time.time() - t0
    print(f"[info] policy loaded in {load_time:.1f}s")

    # After-load VRAM (the "static" footprint).
    static_vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[info] static VRAM (weights only): {static_vram_gb:.2f} GB")

    # Construct dummy batch.
    obs = make_dummy_observation(device)
    print(f"[info] dummy batch built (camera1 480x640, state 6-d, 1 task)")

    # Warm-up — first forwards trigger CUDA kernel compilation + memory grow.
    print(f"[info] warm-up forwards ({n_warmup}x) ...")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(n_warmup):
            t = time.time()
            try:
                action = policy.select_action(obs)
            except AttributeError:
                # Some lerobot versions: policy(obs) or policy.forward(obs)
                action = policy(obs)
            torch.cuda.synchronize()
            print(f"  warm-up {i+1}/{n_warmup}: {time.time()-t:.2f}s")

    peak_after_warmup_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[info] peak VRAM after warm-up: {peak_after_warmup_gb:.2f} GB")

    # Timed forwards.
    latencies = []
    torch.cuda.reset_peak_memory_stats()
    print(f"\n[info] timed forwards ({n_timed}x) ...")
    with torch.no_grad():
        for i in range(n_timed):
            torch.cuda.synchronize()
            t = time.time()
            try:
                action = policy.select_action(obs)
            except AttributeError:
                action = policy(obs)
            torch.cuda.synchronize()
            dt = time.time() - t
            latencies.append(dt)
            print(f"  timed {i+1}/{n_timed}: {dt:.2f}s")

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    # Sort for percentiles.
    latencies_sorted = sorted(latencies)
    mean_lat = sum(latencies) / len(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[min(int(0.95 * len(latencies_sorted)), len(latencies_sorted) - 1)]
    p99 = latencies_sorted[min(int(0.99 * len(latencies_sorted)), len(latencies_sorted) - 1)]

    # Summary.
    print("\n" + "="*60)
    print(f"  Pi0.5 VRAM + Latency Probe, {policy_path}")
    print("="*60)
    print(f"  Static VRAM (weights):    {static_vram_gb:6.2f} GB")
    print(f"  Peak VRAM (inference):    {peak_vram_gb:6.2f} GB    "
          f"limit: {vram_limit_gb} GB    "
          f"{'PASS' if peak_vram_gb <= vram_limit_gb else 'FAIL'}")
    print(f"  Mean latency:             {mean_lat:6.2f} s")
    print(f"  P50 latency:              {p50:6.2f} s")
    print(f"  P95 latency:              {p95:6.2f} s     "
          f"limit: {latency_limit_s} s    "
          f"{'PASS' if p95 <= latency_limit_s else 'FAIL'}")
    print(f"  P99 latency:              {p99:6.2f} s")
    print("="*60)

    vram_pass = peak_vram_gb <= vram_limit_gb
    latency_pass = p95 <= latency_limit_s

    if vram_pass and latency_pass:
        print("  PASS, Pi0.5 path is viable for the 16 GB GPU target")
        return 0
    else:
        print("  FAIL, Pi0.5 will not deploy reliably on the 16 GB GPU target")
        if not vram_pass:
            print(f"      VRAM peak {peak_vram_gb:.2f} GB > limit {vram_limit_gb} GB")
            print(f"      If VRAM peaks above 14 GB the Pi0.5 path will OOM at inference; use the SmolVLA checkpoint instead.")
        if not latency_pass:
            print(f"      P95 latency {p95:.2f}s > 20s budget")
            print(f"      TA will disqualify. Reduce prompt length, or pivot to SmolVLA.")
        return 1


def main() -> int:
    """Parse CLI args and run the Pi0.5 inference VRAM + latency probe."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", default="lerobot/pi05_base",
                        help="HF repo id or local path. Default: lerobot/pi05_base")
    parser.add_argument("--n-warmup", type=int, default=5,
                        help="Number of warm-up forwards (not timed). Default: 5.")
    parser.add_argument("--n-timed", type=int, default=5,
                        help="Number of timed forwards used for latency stats. Default: 5.")
    parser.add_argument("--vram-limit-gb", type=float, default=DEFAULT_VRAM_LIMIT_GB,
                        help="Peak VRAM PASS threshold in GB. Default: 14.0 (2 GB headroom on a 16 GB GPU).")
    parser.add_argument("--latency-limit-s", type=float, default=DEFAULT_LATENCY_LIMIT_S,
                        help="P95 per-forward latency PASS threshold in seconds. Default: 20.0 (TA rule).")
    args = parser.parse_args()

    return probe(
        policy_path=args.policy_path,
        n_warmup=args.n_warmup,
        n_timed=args.n_timed,
        vram_limit_gb=args.vram_limit_gb,
        latency_limit_s=args.latency_limit_s,
    )


if __name__ == "__main__":
    sys.exit(main())
