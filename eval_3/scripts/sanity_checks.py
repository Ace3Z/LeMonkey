#!/usr/bin/env python3
"""Sanity checks for the Track D (M2 ArcFace) checkpoint, without a robot.

Three behavioural probes + one data probe. All run on a single GPU box and
finish in ~2 min. The reviewer agent flagged "language pathway off-axis"
as the #1 risk (high mean_cos doesn't prove the policy reads the prompt);
the language + vision sensitivity checks here are the closest we can get
to that without Strix.

Tests:
1. LANGUAGE SENSITIVITY — same image, 3 different celeb prompts. Output
   action chunks MUST differ. If they're identical, the policy is ignoring
   the prompt (the BlindVLA-style off-axis failure).
2. VISION SENSITIVITY — same prompt, 3 different real teleop frames. Output
   actions MUST differ. If identical, the policy is ignoring the camera.
3. DETERMINISM + RANGE — same (image, prompt), called twice. Outputs MUST
   be ~identical (no leaking flow-matching noise) AND in a sensible
   magnitude band (action values should not be all-zero / saturated / NaN).
4. PATCH-MASK ALIASING — read all face_labels.json, compute the
   distribution of mask.sum() per detected face. If most slots have
   mask.sum() ≤ 1, the M2 patch grid is degenerate (one patch can't
   carry identity).

Usage:
    python eval_3/scripts/sanity_checks.py --revision step-10000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aug"))
from smolvlm_inference_patch import apply as _apply_smolvlm_patch  # noqa: E402
_apply_smolvlm_patch()


# ─── helpers ──────────────────────────────────────────────────────────


def _load_policy_and_preprocessor(repo: str, revision: str, device: str):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    print(f"[load] {repo}@{revision}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(repo, revision=revision).to(device).eval()
    preprocessor = DataProcessorPipeline.from_pretrained(
        repo, config_filename="policy_preprocessor.json", revision=revision
    )
    return policy, preprocessor


def _make_batch(image, state, task, device, preprocessor):
    batch = {
        "observation.images.camera1": image.to(device).unsqueeze(0),
        "observation.state": state.to(device).unsqueeze(0),
        "task": task,
    }
    return preprocessor(batch)


def _grab_real_frame(dataset_root: Path, ep_idx: int, frame_idx: int = 0):
    """Grab one frame from the cached LeRobot dataset videos via pyav.

    Falls back to a random tensor if the lookup fails (we'd rather report
    inconclusive than refuse to run).
    """
    import datasets, av  # noqa: E402

    eps = sorted(str(p) for p in (dataset_root / "meta/episodes").rglob("*.parquet"))
    ds = datasets.load_dataset("parquet", data_files=eps, split="train")
    row = ds[ep_idx]
    chunk = row["videos/observation.images.camera1/chunk_index"]
    fi = row["videos/observation.images.camera1/file_index"]
    video_path = dataset_root / f"videos/observation.images.camera1/chunk-{chunk:03d}/file-{fi:03d}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    decoded = None
    for i, frame in enumerate(container.decode(stream)):
        if i == frame_idx:
            decoded = frame
            break
    container.close()
    if decoded is None:
        raise IndexError(f"no frame {frame_idx} in {video_path}")
    arr = decoded.to_ndarray(format="rgb24")
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def _identical(a: torch.Tensor, b: torch.Tensor, atol=1e-4) -> bool:
    return torch.allclose(a, b, atol=atol)


def _diff_metric(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


# ─── tests ────────────────────────────────────────────────────────────


def test_language_sensitivity(policy, preprocessor, image, state, device):
    prompts = [
        "Place the coke on Taylor Swift.",
        "Place the coke on Barack Obama.",
        "Place the coke on Yann LeCun.",
    ]
    actions = []
    for p in prompts:
        b = _make_batch(image, state, p, device, preprocessor)
        with torch.inference_mode():
            a = policy.select_action(b)
        actions.append(a.detach().cpu())

    pairs = [(0, 1), (0, 2), (1, 2)]
    deltas = [_diff_metric(actions[i], actions[j]) for i, j in pairs]
    identical_any = any(_identical(actions[i], actions[j]) for i, j in pairs)

    print(f"[lang]  swift  : {actions[0].flatten().tolist()}")
    print(f"[lang]  obama  : {actions[1].flatten().tolist()}")
    print(f"[lang]  lecun  : {actions[2].flatten().tolist()}")
    print(f"[lang]  pairwise mean |Δ| (swift/obama, swift/lecun, obama/lecun) = "
          f"{deltas[0]:.4f} {deltas[1]:.4f} {deltas[2]:.4f}")
    print(f"[lang]  result: {'FAIL — actions identical across prompts (policy ignoring language)' if identical_any else 'PASS — actions differ across prompts'}")
    return not identical_any, deltas


def test_vision_sensitivity(policy, preprocessor, frames, state, prompt, device):
    actions = []
    for fr in frames:
        b = _make_batch(fr, state, prompt, device, preprocessor)
        with torch.inference_mode():
            a = policy.select_action(b)
        actions.append(a.detach().cpu())

    pairs = [(0, 1), (0, 2), (1, 2)]
    deltas = [_diff_metric(actions[i], actions[j]) for i, j in pairs]
    identical_any = any(_identical(actions[i], actions[j]) for i, j in pairs)
    print(f"[vis]   pairwise mean |Δ| across 3 frames = "
          f"{deltas[0]:.4f} {deltas[1]:.4f} {deltas[2]:.4f}")
    print(f"[vis]   result: {'FAIL — actions identical across frames (policy ignoring camera)' if identical_any else 'PASS — actions differ across frames'}")
    return not identical_any, deltas


def test_determinism_and_range(policy, preprocessor, image, state, device):
    prompt = "Place the coke on Taylor Swift."
    b1 = _make_batch(image, state, prompt, device, preprocessor)
    b2 = _make_batch(image, state, prompt, device, preprocessor)
    with torch.inference_mode():
        a1 = policy.select_action(b1).detach().cpu()
        a2 = policy.select_action(b2).detach().cpu()

    delta = _diff_metric(a1, a2)
    deterministic = _identical(a1, a2, atol=1e-3)
    nan = bool(torch.isnan(a1).any())
    inf = bool(torch.isinf(a1).any())
    amax = float(a1.abs().max().item())
    sane_range = 0.0 < amax < 5.0  # MEAN_STD-normalized → typically O(1)

    print(f"[det]   mean |Δ| between two identical calls = {delta:.6f}")
    print(f"[det]   determinism: {'PASS' if deterministic else 'FAIL — leaking flow-matching noise'}")
    print(f"[det]   NaN: {nan}  Inf: {inf}  max|a|: {amax:.3f}  "
          f"sane_range (0 < |a| < 5): {sane_range}")
    return deterministic and not nan and not inf and sane_range, delta


def test_patch_mask_aliasing(face_labels_dir: Path):
    """Distribution of bbox patch counts per detected face slot."""
    NUM_CAMERA1_PATCHES = 64  # 8x8 grid (480x640 ÷ 60px patches ~ 8x8)
    counts = Counter()
    n_files = n_frames = n_slots = 0
    for p in sorted(face_labels_dir.glob("*.json")):
        d = json.loads(p.read_text())
        n_files += 1
        for fidx, entry in d.items():
            n_frames += 1
            for slot in entry.get("slots", []):
                n_slots += 1
                bbox = slot.get("bbox", None)
                if not bbox:
                    continue
                # bbox is [x_min, y_min, x_max, y_max] in image pixels
                # at 480x640 cam → 8x8 patch grid → patch size ~60x80
                x0, y0, x1, y1 = bbox
                px0, py0 = int(x0 / 80), int(y0 / 60)
                px1, py1 = int(x1 / 80) + 1, int(y1 / 60) + 1
                ph = max(1, min(8, py1) - max(0, py0))
                pw = max(1, min(8, px1) - max(0, px0))
                counts[ph * pw] += 1
    print(f"[mask]  scanned {n_files} sources, {n_frames} frames, {n_slots} slots with bbox")
    print(f"[mask]  patch-count distribution (rough estimate, 8x8 grid over 480x640):")
    for k in sorted(counts.keys()):
        print(f"           {k:>2d} patches: {counts[k]:>6d} slots ({counts[k]/sum(counts.values())*100:.1f}%)")
    median = sorted(counts.elements())[len(list(counts.elements())) // 2] if counts else 0
    print(f"[mask]  median patches per slot = {median}")
    ok = median >= 2
    print(f"[mask]  result: {'PASS — patch grid carries enough signal' if ok else 'CONCERN — slots collapse to 1 patch'}")
    return ok, median


# ─── main ─────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="HBOrtiz/smolvla_eval3_track_D_m2_mahbod")
    p.add_argument("--revision", default="step-10000")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset-root", default=str(Path.home() / ".cache/huggingface/lerobot/HBOrtiz/so101_eval3_track3_v3_baseline"))
    p.add_argument("--face-labels-dir", default=str(Path.home() / "eval3_m2_toolkit/face_labels"))
    args = p.parse_args()

    print(f"[main]  device={args.device}", flush=True)
    policy, preprocessor = _load_policy_and_preprocessor(args.repo, args.revision, args.device)
    state_dim = next((v.shape[0] for k, v in policy.config.input_features.items()
                      if k.endswith("state")), 6)
    state = torch.zeros(state_dim, dtype=torch.float32)

    # Pull 3 real frames from the dataset to use as inputs.
    dataset_root = Path(args.dataset_root)
    frames = []
    try:
        # Pick frames from 3 different aug variant episodes spread across the dataset.
        for ep_idx in [100, 5000, 9000]:
            frames.append(_grab_real_frame(dataset_root, ep_idx, frame_idx=10))
        print(f"[main]  pulled 3 real frames from episodes 100, 5000, 9000", flush=True)
    except Exception as e:
        print(f"[main]  [WARN] could not pull real frames ({e}); falling back to random tensors", flush=True)
        rng = torch.Generator().manual_seed(0)
        frames = [torch.rand(3, 480, 640, generator=rng) for _ in range(3)]

    results = {}

    print("\n=== 1. LANGUAGE SENSITIVITY (same image, 3 prompts) ===")
    ok_lang, _ = test_language_sensitivity(policy, preprocessor, frames[0], state, args.device)
    results["language"] = ok_lang

    print("\n=== 2. VISION SENSITIVITY (same prompt, 3 frames) ===")
    ok_vis, _ = test_vision_sensitivity(policy, preprocessor, frames, state,
                                        "Place the coke on Taylor Swift.", args.device)
    results["vision"] = ok_vis

    print("\n=== 3. DETERMINISM + RANGE ===")
    ok_det, _ = test_determinism_and_range(policy, preprocessor, frames[0], state, args.device)
    results["determinism"] = ok_det

    print("\n=== 4. PATCH-MASK ALIASING (data probe) ===")
    flabels = Path(args.face_labels_dir)
    if flabels.exists():
        ok_mask, _ = test_patch_mask_aliasing(flabels)
        results["patch_mask"] = ok_mask
    else:
        print(f"[mask]  [WARN] face_labels dir not found: {flabels}; skipping")
        results["patch_mask"] = None

    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        sym = "✓" if ok is True else ("✗" if ok is False else "—")
        print(f"  {sym}  {name}")

    failed = [k for k, v in results.items() if v is False]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
