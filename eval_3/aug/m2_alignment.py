"""M2 ArcFace cosine-distillation loss for SmolVLA — pure math, no SmolVLA dep.

What this implements
--------------------
The contrastive, per-pid-routed variant of the M2 alignment loss documented in
[`docs/report/2026-05-18_m2_arcface_validation.md`](../../docs/report/2026-05-18_m2_arcface_validation.md).

Given:

- `hidden_state` of shape (B, prefix_len, H=960) — captured from SmolVLA's
  mid-LLM via `register_forward_pre_hook` on `text_model.layers[N+1].input_layernorm`.
- `bbox_masks` of shape (B, 3, 64) — binary mask per (batch, slot) marking which
  of the 8×8 camera1 patches overlap the detected face for that slot.
- `bbox_valid` of shape (B, 3) — whether that slot has a usable detection in
  this frame (False ⇒ occluded / off-screen / unmatched, skip in the loss).
- `target_centroids` of shape (B, 3, 512) — the per-slot ArcFace target
  embedding (looked up offline from `celeb_embeddings.json` via
  `augmentation.json[new_layout_camera_lmr]`).

Compute:

1. Extract camera1 patches at positions [0, 64) of the prefix. (For
   `empty_cameras=1`, the prefix layout is `[camera1=0..64) | camera2_zero=64..128) | lang | state]`.)
2. For each (b, slot) with `bbox_valid=True`:
   - Mean-pool the camera1 patches inside `bbox_masks[b, slot]` →
     one (H=960,) vector.
   - Project through a frozen 3-layer MLP → (512,) vector.
   - L2-normalize both the projected vector and `target_centroids[b, slot]`.
   - Cosine similarity in (-1, +1).
3. Loss = `-mean(cos)` over all valid slots (BlindVLA Eq. 9, with k = number
   of valid slots in the batch, *not* number of patches).

Anti-forgetting safeguards (BlindVLA Table 6 + Table 8):
- Projector is **frozen** (`requires_grad=False`). Otherwise gradient flow
  prefers updating the projector instead of the VLA hidden state; the paper
  reports trainable-projector degrades semantic OOD from 0.61 → 0.54.
- λ = 0.2 (caller's responsibility to scale; this module returns the raw
  alignment loss).

Why **not** the literal BlindVLA per-patch sum
----------------------------------------------
BlindVLA's loss `−(1/k)·Σ_j cos(u_j, z_j)` assumes a per-patch teacher (the
teacher emits patch-level features). ArcFace emits **one global identity
vector** per face crop. Applying the per-patch sum with `z_j = z_target ∀j`
pulls background and distractor patches toward the target identity — that's
anti-discriminative. Instead we pool the target's patches → one student vector
→ one cosine. See `2026-05-18_m2_arcface_validation.md` Finding 2.

References
----------
- BlindVLA paper Eq. 9 — arxiv 2510.25616
- BlindVLA code `finetune_align.py:138-152` (frozen MLP arch) and
  `:423-427` (negative-mean cosine).
- Project plan: `docs/experiments/2026-05-19_m2_data_foundation.md`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry: bbox in original 640×480 pixel space → 8×8 patch mask after
# `resize_with_pad(640×480 → 512×512)` + SigLIP patch=16 + connector
# scale_factor=4 (so 32×32 raw patches → 8×8 after pixel-shuffle).
# ---------------------------------------------------------------------------

CAMERA1_PATCH_GRID = 8           # 8×8 patch grid per image after connector
NUM_CAMERA1_PATCHES = 64         # 8 * 8
SMOLLM2_HIDDEN_SIZE = 960        # text_config.hidden_size for SmolVLM2-500M
ARCFACE_EMBED_DIM = 512          # buffalo_l ArcFace output dim


def _resize_with_pad_box(
    bbox_xyxy: tuple[float, float, float, float],
    orig_hw: tuple[int, int] = (480, 640),
    target_hw: tuple[int, int] = (512, 512),
) -> tuple[float, float, float, float]:
    """Map a pixel bbox from the original frame to the resized-with-pad frame.

    `resize_with_pad` (LeRobot's preprocessing) keeps aspect ratio and pads
    the shorter side with zeros centred. For 480×640 → 512×512:
      scale = min(512/640, 512/480) = 0.8
      new_h = 480 * 0.8 = 384, new_w = 640 * 0.8 = 512
      pad_y = (512 − 384) / 2 = 64 on each of top + bottom; pad_x = 0
    """
    x1, y1, x2, y2 = bbox_xyxy
    orig_h, orig_w = orig_hw
    target_h, target_w = target_hw
    scale = min(target_h / orig_h, target_w / orig_w)
    new_w = orig_w * scale
    new_h = orig_h * scale
    pad_x = (target_w - new_w) / 2
    pad_y = (target_h - new_h) / 2
    return (x1 * scale + pad_x, y1 * scale + pad_y,
            x2 * scale + pad_x, y2 * scale + pad_y)


def bbox_to_patch_mask(
    bbox_xyxy: tuple[float, float, float, float],
    orig_hw: tuple[int, int] = (480, 640),
    target_hw: tuple[int, int] = (512, 512),
    patch_grid: int = CAMERA1_PATCH_GRID,
) -> np.ndarray:
    """Return a (patch_grid * patch_grid,) bool mask of patches that overlap bbox.

    A patch is "in" if the bbox overlaps any part of its target-frame footprint.
    Each patch in the 8×8 grid covers a 64×64 region of the 512×512 input
    (512 / 8 = 64). We use ceil/floor inclusively so a bbox that grazes a patch
    boundary still counts.
    """
    rx1, ry1, rx2, ry2 = _resize_with_pad_box(bbox_xyxy, orig_hw, target_hw)
    px = target_hw[1] / patch_grid       # patch width  in target px = 64.0
    py = target_hw[0] / patch_grid       # patch height in target px = 64.0

    col_lo = max(0, int(math.floor(rx1 / px)))
    col_hi = min(patch_grid - 1, int(math.ceil(rx2 / px)) - 1)
    row_lo = max(0, int(math.floor(ry1 / py)))
    row_hi = min(patch_grid - 1, int(math.ceil(ry2 / py)) - 1)

    mask = np.zeros((patch_grid, patch_grid), dtype=bool)
    if col_lo <= col_hi and row_lo <= row_hi:
        mask[row_lo:row_hi + 1, col_lo:col_hi + 1] = True
    return mask.reshape(-1)              # flatten to (patch_grid**2,)


# ---------------------------------------------------------------------------
# Frozen 3-layer MLP projector (BlindVLA Table 6 verbatim, verified against
# https://github.com/CognitiveAISystems/BlindVLA/blob/main/openvla/vla-scripts/finetune_align.py
# lines 138-152). Output dim is the teacher's dim — here 512 (ArcFace).
# ---------------------------------------------------------------------------

class FrozenProjector(nn.Module):
    """LN → 960→2048 → SiLU → Drop(0.1) → 2048→2048 → SiLU → Drop(0.1) → 2048→512.

    Initialised at random, then `requires_grad=False`. The MLP is a fixed
    feature map; the supervision signal flows entirely into the VLA via
    the cosine loss on the projector's output.
    """

    def __init__(self,
                 in_dim: int = SMOLLM2_HIDDEN_SIZE,
                 hidden_dim: int = 2048,
                 out_dim: int = ARCFACE_EMBED_DIM,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim) → (..., out_dim)
        return self.net(x)

    def train(self, mode: bool = True):
        # Always stay in eval mode (so dropout doesn't fire on the frozen MLP).
        super().train(False)
        return self


# ---------------------------------------------------------------------------
# Loss computation.
# ---------------------------------------------------------------------------

@dataclass
class M2LossResult:
    """What `m2_align_loss` returns.

    `loss` is the raw negative-mean cosine (scalar). Caller multiplies by λ.
    `n_valid` is the count of (batch, slot) pairs that contributed. If 0, the
    loss is the zero tensor (with grad) and the caller should `[WARN]`.
    `per_slot_cos` is a (B*3,) tensor of cosines (or NaN for invalid slots);
    useful for monitoring identity-discrimination quality during training.
    """
    loss: torch.Tensor
    n_valid: int
    per_slot_cos: torch.Tensor


def m2_align_loss(
    hidden_state: torch.Tensor,    # (B, prefix_len, H=960)
    bbox_masks: torch.Tensor,      # (B, 3, num_patches=64) bool/float
    bbox_valid: torch.Tensor,      # (B, 3) bool
    target_centroids: torch.Tensor,  # (B, 3, 512) — should be L2-normalized
    projector: FrozenProjector,
    camera1_offset: int = 0,
    num_camera1_patches: int = NUM_CAMERA1_PATCHES,
) -> M2LossResult:
    """Per-pid-routed BlindVLA-style alignment loss.

    Math (one-vs-one cosine per valid slot, mean over all valid slots):

        For each (b, s) with bbox_valid[b, s]:
            patches   = hidden_state[b, camera1_offset : camera1_offset+64, :]
            mask      = bbox_masks[b, s]                       (64,)
            pooled    = (patches * mask).sum() / mask.sum()    (H=960,)
            projected = projector(pooled)                      (512,)
            u         = L2_normalize(projected)
            z         = L2_normalize(target_centroids[b, s])
            cos[b, s] = (u · z)
        loss = − mean(cos[valid])

    Caller multiplies the result by λ (BlindVLA recommends 0.2).
    """
    B, prefix_len, H = hidden_state.shape
    assert bbox_masks.shape == (B, 3, num_camera1_patches), \
        f"bbox_masks shape {bbox_masks.shape} != ({B}, 3, {num_camera1_patches})"
    assert bbox_valid.shape == (B, 3)
    assert target_centroids.shape == (B, 3, ARCFACE_EMBED_DIM)
    assert prefix_len >= camera1_offset + num_camera1_patches, \
        f"prefix_len={prefix_len} too short to contain camera1 patches"

    device = hidden_state.device
    dtype = hidden_state.dtype

    # 1. Extract camera1 patches: (B, 64, 960)
    cam1 = hidden_state[:, camera1_offset : camera1_offset + num_camera1_patches, :]

    # 2. Mean-pool over each (b, s) bbox mask.
    # bbox_masks float in {0, 1}, expand to broadcast over hidden_size.
    mask = bbox_masks.to(dtype=dtype)                       # (B, 3, 64)
    weighted = torch.einsum("bsp,bph->bsh", mask, cam1)     # (B, 3, 960)
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)   # (B, 3, 1)
    pooled = weighted / denom                               # (B, 3, 960)

    # 3. Project to 512-D via the frozen MLP. Apply to (B*3, 960) for stable Linear.
    projected = projector(pooled.reshape(B * 3, H)).reshape(B, 3, ARCFACE_EMBED_DIM)

    # 4. Cosine vs target_centroids (both normalized).
    u = F.normalize(projected, dim=-1, eps=1e-8)
    z = F.normalize(target_centroids.to(dtype=dtype), dim=-1, eps=1e-8)
    cos = (u * z).sum(dim=-1)                                # (B, 3)

    # 5. Mask to valid slots, mean over them.
    valid = bbox_valid.to(device=device)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        # Defensive: return a zero loss with grad attached, log will happen
        # at the caller. (Per project CLAUDE.md §5: no silent fallback; the
        # caller must emit a [WARN] if this branch ever fires in production.)
        loss = (cos.sum() * 0.0)
    else:
        loss = -((cos * valid.to(dtype=dtype)).sum() / n_valid)

    # Pack per-slot cos with NaN where invalid (for monitoring, no grad).
    with torch.no_grad():
        per_slot_cos = cos.detach().masked_fill(~valid, float("nan")).reshape(-1)

    return M2LossResult(loss=loss, n_valid=n_valid, per_slot_cos=per_slot_cos)


# ---------------------------------------------------------------------------
# Dataloader-side helper: build per-batch supervision tensors from face_labels
# + augmentation.json + celeb_embeddings.json.
# ---------------------------------------------------------------------------

# Three-letter layout codes carry the LEFT→MIDDLE→RIGHT celeb identities.
LETTER_TO_SHORT = {"O": "obama", "L": "lecun", "S": "swift"}
SHORT_TO_FULL = {"obama": "barack_obama",
                 "lecun": "yann_lecun",
                 "swift": "taylor_swift"}


def slot_to_celeb(new_layout_camera_lmr: str) -> list[str]:
    """E.g. 'OLS' → ['barack_obama', 'yann_lecun', 'taylor_swift']."""
    return [SHORT_TO_FULL[LETTER_TO_SHORT[c]] for c in new_layout_camera_lmr]


def build_supervision_for_frame(
    face_labels_frame: dict,
    new_layout_camera_lmr: str,
    centroid_lookup: dict[str, np.ndarray],   # celeb → (512,) ArcFace centroid
    orig_hw: tuple[int, int] = (480, 640),
    target_hw: tuple[int, int] = (512, 512),
    patch_grid: int = CAMERA1_PATCH_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For one frame, return:
       bbox_masks: (3, patch_grid**2) bool
       bbox_valid: (3,) bool
       target_centroids: (3, 512) float32 (zeros where invalid)

    Slot order is camera POV left → middle → right (matches face_labels which
    pre-sorts bboxes by x_center, and matches LMR layout-code letters).
    """
    bboxes = face_labels_frame.get("bboxes", [])
    celebs = slot_to_celeb(new_layout_camera_lmr)

    n_patches = patch_grid * patch_grid
    masks = np.zeros((3, n_patches), dtype=bool)
    valid = np.zeros((3,), dtype=bool)
    targets = np.zeros((3, ARCFACE_EMBED_DIM), dtype=np.float32)

    for slot in range(3):
        if slot >= len(bboxes):
            continue
        b = bboxes[slot]
        bbox_xyxy = (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]))
        celeb = celebs[slot]
        if celeb not in centroid_lookup:
            continue
        masks[slot] = bbox_to_patch_mask(bbox_xyxy, orig_hw, target_hw, patch_grid)
        valid[slot] = bool(masks[slot].any())
        targets[slot] = centroid_lookup[celeb]
    return masks, valid, targets
