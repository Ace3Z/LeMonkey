#!/usr/bin/env python3
"""Train the residual head on DAgger correction data.

For each frame in the DAgger dataset(s):
    obs (image, state, prompt) → frozen base SmolVLA → base_action_chunk
    target_residual = recorded_action - base_action[0]
    residual = MLP(image_features, state, base_action[0])
    loss = MSE(residual, target_residual)

Frames where the user was driving (is_intervention=1) carry a non-zero
target — that's the correction signal the residual learns. Frames where
the policy was driving (is_intervention=0) carry a near-zero target — the
residual learns to be quiet there.

Per CLAUDE.md §5: every fallback path emits a [WARN] line.

Usage:
    train_residual.py --dataset-root /path/to/eval1_dagger/blue \\
                       --dataset-root /path/to/eval1_dagger/red \\
                       --dataset-root /path/to/eval1_dagger/green \\
                       --policy-path /path/to/smolvla_eval1/.../pretrained_model \\
                       --out /path/to/save \\
                       --steps 5000 --batch-size 32 --lr 3e-4
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

# Local
sys.path.insert(0, str(Path(__file__).resolve().parent))
from residual_head import ResidualHead, IMAGE_DIM

# lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.control_utils import predict_action


# ─── CLI ─────────────────────────────────────────────────────────────────────

p = argparse.ArgumentParser()
p.add_argument("--dataset-root", action="append", required=True,
               help="Path to a LeRobotDataset root. Pass multiple times for several colors.")
p.add_argument("--policy-path", required=True,
               help="Path to the frozen base SmolVLA checkpoint (the pretrained_model dir).")
p.add_argument("--out",         required=True,
               help="Output directory for the trained residual checkpoint.")
p.add_argument("--steps",       type=int, default=5000)
p.add_argument("--batch-size",  type=int, default=64)
p.add_argument("--lr",          type=float, default=3e-4)
p.add_argument("--weight-decay", type=float, default=1e-4)
p.add_argument("--intervention-weight", type=float, default=2.0,
               help="Up-weight loss on intervention frames vs non-intervention. 1.0 = no upweighting.")
p.add_argument("--num-workers", type=int, default=2)
p.add_argument("--device",      default="cuda")
p.add_argument("--seed",        type=int, default=42)
p.add_argument("--log-every",   type=int, default=20)
p.add_argument("--save-every",  type=int, default=500)
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device(args.device)


# ─── Load datasets ───────────────────────────────────────────────────────────

print(f"=== Loading {len(args.dataset_root)} dataset(s) ===")
ds_list = []
for root in args.dataset_root:
    print(f"  {root}")
    ds_list.append(LeRobotDataset(repo_id=f"local/{Path(root).name}", root=root))
dataset = ConcatDataset(ds_list)

# Sanity: confirm the is_intervention feature exists on at least one dataset
sample = ds_list[0][0]
if "is_intervention" not in sample:
    print(f"[WARN] expected 'is_intervention' feature in dataset, got keys: {list(sample.keys())}",
          flush=True)
    print(f"[WARN] all frames will be treated as is_intervention=0; intervention_weight has no effect",
          flush=True)
print(f"  total frames: {sum(len(d) for d in ds_list):,}")
print(f"  sample keys : {list(sample.keys())}")


# ─── Load frozen base policy ─────────────────────────────────────────────────

print(f"\n=== Loading frozen base SmolVLA from {args.policy_path} ===")
base = SmolVLAPolicy.from_pretrained(args.policy_path).eval().to(device)
for p_ in base.parameters():
    p_.requires_grad = False
n_base = sum(p.numel() for p in base.parameters())
print(f"  base params: {n_base/1e6:.1f}M (all frozen)")

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=base.config,
    pretrained_path=args.policy_path,
    preprocessor_overrides={
        "device_processor": {"device": str(device)},
        "rename_observations_processor": {
            "rename_map": {"observation.images.front": "observation.images.camera1"}
        },
    },
)


# ─── Build residual head ─────────────────────────────────────────────────────

residual = ResidualHead().to(device)
print(f"\n=== Residual head ===")
print(f"  trainable params: {residual.num_params():,}")

opt = torch.optim.AdamW(residual.parameters(), lr=args.lr, weight_decay=args.weight_decay)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.03)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _pad_to_512(img_t: torch.Tensor) -> torch.Tensor:
    """Match SmolVLA's resize_with_pad EXACTLY (modeling_smolvla.py:134-153).

    SmolVLA pads on the RIGHT and BOTTOM only (not centered). Mismatching
    this gave the residual head image features in a different spatial layout
    than the base policy ever saw — silent generalization failure.
    """
    # img_t: (B, 3, H, W) float in [0,1]
    B, C, H, W = img_t.shape
    target = 512
    # Match SmolVLA's formula: ratio = max(W/target, H/target); resized = orig / ratio
    ratio = max(W / target, H / target)
    new_h = int(H / ratio)
    new_w = int(W / ratio)
    img_t = F.interpolate(img_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h = max(0, target - new_h)
    pad_w = max(0, target - new_w)
    # F.pad order: (left, right, top, bottom). SmolVLA pads right+bottom only.
    img_t = F.pad(img_t, (0, pad_w, 0, pad_h), value=0.0)
    return img_t  # (B, 3, 512, 512)


def extract_image_features(image_HWC_uint8_batch: torch.Tensor) -> torch.Tensor:
    """Run image batch through frozen base vision encoder, return (B, 960) pooled."""
    # image_HWC_uint8_batch: (B, H, W, 3) uint8 OR (B, 3, H, W) float
    if image_HWC_uint8_batch.dim() == 4 and image_HWC_uint8_batch.shape[-1] == 3:
        # HWC → CHW; uint8 → float [0,1]
        img = image_HWC_uint8_batch.permute(0, 3, 1, 2).float() / 255.0
    else:
        img = image_HWC_uint8_batch.float()
        if img.max() > 1.5:  # likely 0-255 range
            img = img / 255.0
    img = _pad_to_512(img.to(device))
    img = img * 2.0 - 1.0  # SigLIP expects [-1, 1]
    with torch.inference_mode():
        feats = base.model.vlm_with_expert.embed_image(img)  # (B, num_patches, 960)
    return feats.mean(dim=1)  # (B, 960)


def base_action_for_batch(images_uint8_BHWC_np, states_BD_np, task_strs):
    """Run a batch of B frames through the frozen base in one forward pass.

    Replicates the predict_action() pipeline (prep tensors → preprocessor →
    policy → postprocessor) but with a real batch dimension instead of
    looping per-frame. Uses predict_action_chunk and slices the first action
    of each chunk — equivalent to what select_action would return on a fresh
    queue (post base.reset). 4-8× faster than 32 sequential calls.

    Args:
        images_uint8_BHWC_np: (B, H, W, 3) uint8 numpy array.
        states_BD_np: (B, state_dim) float32 numpy array.
        task_strs: list of B task strings.
    Returns:
        (B, action_dim) np.float32 array of base actions.
    """
    base.reset()
    B = len(task_strs)
    img_t = torch.from_numpy(images_uint8_BHWC_np).to(device).float() / 255.0
    img_t = img_t.permute(0, 3, 1, 2).contiguous()  # BHWC → BCHW
    state_t = torch.from_numpy(states_BD_np).to(device)
    obs = {
        "observation.images.front": img_t,
        "observation.state": state_t,
        "task": task_strs,                # tokenizer accepts list[str]
        "robot_type": "so101_follower",
    }
    with torch.inference_mode():
        obs = preprocessor(obs)
        chunks = base.predict_action_chunk(obs)   # (B, n_action_steps, action_dim)
        first = chunks[:, 0, :]                    # (B, action_dim)
        first = postprocessor(first)
    return first.detach().cpu().numpy().astype(np.float32)


# ─── Get task strings per frame via task_index → task lookup ────────────────

import pandas as pd
def get_task_lookup(ds):
    """Return dict task_index → task_string from the dataset's tasks.parquet."""
    tasks_path = Path(ds.root) / "meta/tasks.parquet"
    if not tasks_path.exists():
        print(f"[WARN] tasks.parquet not found at {tasks_path}; falling back to empty task string",
              flush=True)
        return {}
    df = pd.read_parquet(tasks_path).reset_index()
    return dict(zip(df["task_index"], df["task"]))


task_lookups = [get_task_lookup(d) for d in ds_list]


# ─── Custom training loop (no DataLoader complexity for now) ────────────────

print(f"\n=== Training: {args.steps} steps, batch_size={args.batch_size} ===\n")

t_start = time.time()
running_loss = 0.0
running_int = 0.0
running_noint = 0.0
running_int_count = 0
running_noint_count = 0

# Build a flat list of (dataset_idx, frame_idx) for sampling
all_frames = []
for di, d in enumerate(ds_list):
    for fi in range(len(d)):
        all_frames.append((di, fi))
print(f"Sampling pool: {len(all_frames)} frames")


def sample_batch(B):
    """Sample B random frames, gather raw arrays, then run ONE batched base
    inference pass. ~10-20× faster than per-frame calls on H100."""
    idxs = np.random.choice(len(all_frames), size=B, replace=True)
    images_np = []
    states = []
    actions = []
    task_strs = []
    is_int = []
    for di, fi in [all_frames[i] for i in idxs]:
        f = ds_list[di][fi]
        # The image key differs between DAgger datasets ('camera1') and original BC
        # datasets ('front'). Try both, with [WARN] if neither found.
        if "observation.images.camera1" in f:
            img_t = f["observation.images.camera1"]
        elif "observation.images.front" in f:
            img_t = f["observation.images.front"]
        else:
            print(f"[WARN] no image key found in frame; keys={list(f.keys())}", flush=True)
            raise KeyError(f"no image key in frame keys={list(f.keys())}")
        if img_t.dtype == torch.float32:
            img_HWC = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        else:
            img_HWC = img_t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        state_np = f["observation.state"].cpu().numpy().astype(np.float32)
        action_np = f["action"].cpu().numpy().astype(np.float32)
        # Look up task string
        ti = int(f["task_index"].item())
        task_str = task_lookups[di].get(ti, "")
        if task_str == "":
            print(f"[WARN] no task string found for task_index={ti} in dataset[{di}]; using empty",
                  flush=True)
        # is_intervention: present in DAgger datasets, may be missing in BC datasets
        if "is_intervention" in f:
            ii = int(f["is_intervention"].item() if hasattr(f["is_intervention"], "item")
                     else int(f["is_intervention"][0]))
        else:
            ii = 0
        images_np.append(img_HWC)
        states.append(state_np)
        actions.append(action_np)
        task_strs.append(task_str)
        is_int.append(ii)
    images_BHWC = np.stack(images_np)            # (B, H, W, 3) uint8
    states_BD   = np.stack(states)               # (B, state_dim) float32
    actions_BD  = np.stack(actions)              # (B, action_dim) float32
    # ONE batched forward pass through the frozen base
    base_actions_BD = base_action_for_batch(images_BHWC, states_BD, task_strs)
    images_t = torch.from_numpy(images_BHWC)
    states_t = torch.from_numpy(states_BD).to(device)
    actions_t = torch.from_numpy(actions_BD).to(device)
    base_t = torch.from_numpy(base_actions_BD).to(device)
    isint_t = torch.tensor(is_int, dtype=torch.float32, device=device)
    return images_t, states_t, actions_t, base_t, isint_t


pbar = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True, mininterval=0.5)
for step in pbar:
    images, states, actions, base_a, isint = sample_batch(args.batch_size)
    img_feats = extract_image_features(images)            # (B, 960)
    target = actions - base_a                              # (B, 6) — per-step delta

    pred = residual(img_feats, states, base_a)             # (B, 6)
    err  = (pred - target).pow(2).mean(dim=-1)             # (B,)

    # Weighted loss: intervention frames carry more weight
    weight = torch.where(isint > 0.5,
                         torch.full_like(err, args.intervention_weight),
                         torch.ones_like(err))
    loss = (err * weight).mean()

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.0)
    opt.step()
    sched.step()

    n_int_batch = (isint > 0.5).sum().item()
    n_noint_batch = (isint <= 0.5).sum().item()

    running_loss += loss.item()
    running_int += err[isint > 0.5].sum().item()
    running_noint += err[isint <= 0.5].sum().item()
    running_int_count   += n_int_batch
    running_noint_count += n_noint_batch

    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")

    if step % args.log_every == 0:
        elapsed = time.time() - t_start
        avg_loss = running_loss / args.log_every
        # Per-class MSE: divide by per-class counts, not total. (Reviewer fix.)
        int_mse   = running_int   / max(running_int_count,   1)
        noint_mse = running_noint / max(running_noint_count, 1)
        tqdm.write(f"  step {step:>5d}/{args.steps}  loss={avg_loss:.5f}  "
                   f"int_mse={int_mse:.5f} (n={running_int_count}) "
                   f"noint_mse={noint_mse:.5f} (n={running_noint_count})  "
                   f"lr={sched.get_last_lr()[0]:.2e}  "
                   f"elapsed={elapsed:.0f}s")
        running_loss = 0.0
        running_int = 0.0
        running_noint = 0.0
        running_int_count = 0
        running_noint_count = 0

    if step % args.save_every == 0 or step == args.steps:
        ckpt_dir = Path(args.out) / f"step_{step:06d}"
        residual.save(ckpt_dir)
        tqdm.write(f"  ✓ saved checkpoint to {ckpt_dir}")

# Always save the last checkpoint as 'last' too
residual.save(Path(args.out) / "last")
print(f"\n=== Training complete in {time.time() - t_start:.0f}s ===")
print(f"  Checkpoint: {Path(args.out) / 'last'}")
