"""KL Attention Loss (KLAL) — cross-modal grounding fix per WACV 2026
"Direct Visual Grounding by Directing Attention of Visual Tokens"
(arXiv:2511.12738).

Why we need it: M2 distillation shaped face-patch hidden states to align
with ArcFace centroids on SmolVLA, but the language-name token never
attended to those face patches (attention probe found constant argmax
across prompts → `docs/experiments/2026-05-19_attention_probe_step10000`).
KLAL directly supervises the attention distribution from name-tokens to
image-patches with a target distribution built from bounding-box
annotations.

Formulation
-----------
    L_KLAL = (1/L) Σ_l KL( P_target(S) || Q^(l)(S) )

where:
- S = the set of image-patch positions (256 for Pi0.5/PaliGemma).
- Q^(l)(S) = the model's actual attention from name-tokens to image-patches
  at layer l, averaged across heads and across the name-token rows.
  Computed by hooking q_proj / k_proj and recomputing softmax(QK^T/√d).
  Shape (256,), sums to 1.
- P_target(S) = a Gaussian-smoothed distribution over the 16x16 patch
  grid, peaked on the face-bbox center for the prompted celeb. Shape
  (256,), sums to 1.
- Sum is over all monitored layers; we use mid-late VLM layers
  (default [6, 10, 14, 17] for Gemma-2B's 18 layers).

For Pi0.5/PaliGemma's prefix-LM full bidirectional attention, this loss
is well-posed: every text token already has architectural access to every
image patch, so the loss is shaping a working channel rather than trying
to create one from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class KLALConfig:
    capture_layers: tuple = (6, 10, 14, 17)  # which VLM layers to supervise
    target_sigma_patches: float = 1.5         # Gaussian std in patch units
    lam: float = 1.0                          # loss scale per the WACV 2026 paper
    patch_grid: int = 16                      # 16x16 for Pi0.5
    num_image_patches_total: int = 256        # patch_grid ** 2
    eps: float = 1e-8


class KLALHookSet:
    """Wraps multi-layer q_proj / k_proj hooks for a model. Use as a context
    manager so the hooks are removed on exit.
    """

    def __init__(self, text_model, layers, n_heads, n_kv_heads, head_dim):
        self.layers = list(layers)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self._captures: dict[int, dict] = {n: {} for n in self.layers}
        self._handles = []
        for n in self.layers:
            attn = text_model.layers[n].self_attn
            self._handles.append(attn.q_proj.register_forward_hook(self._mk_q(n)))
            self._handles.append(attn.k_proj.register_forward_hook(self._mk_k(n)))

    def _mk_q(self, n):
        return lambda mod, inp, out: self._captures[n].__setitem__("q", out)
    def _mk_k(self, n):
        return lambda mod, inp, out: self._captures[n].__setitem__("k", out)

    def reset(self):
        for n in self.layers:
            self._captures[n].clear()

    def get_attention(self, layer: int, image_patch_slice: slice,
                      name_token_positions: torch.Tensor) -> torch.Tensor | None:
        """Compute attention from name-token rows to image-patch cols at one layer.

        Args:
          layer: layer index (must be in self.layers)
          image_patch_slice: slice(0, NUM_IMAGE_PATCHES) for camera1
          name_token_positions: (B, K) long tensor — K name-token positions
            per batch element in the *prefix* (offsetted by image patches +
            pad left)

        Returns:
          (B, NUM_IMAGE_PATCHES) attention distribution, head-averaged,
          name-token-averaged, or None if the hooks didn't fire.
        """
        cap = self._captures.get(layer, {})
        q = cap.get("q")
        k = cap.get("k")
        if q is None or k is None:
            return None

        B, L, _ = q.shape
        q = q.float().view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.float().view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        # NOTE: we skip RoPE for the KLAL signal. RoPE injects position bias
        # that distorts content-based attention scores between far-apart
        # tokens (image patches at positions [0..255], name tokens at
        # positions [~256+...]). The KLAL paper does the same for the
        # same reason. The downside: ranking patches *within* the image is
        # less precise; the upside: cross-modal content match dominates.
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)  # (B, H, L, L)
        attn_avg_heads = attn.mean(dim=1)     # (B, L, L)

        # Slice out attention from name-token rows to image-patch cols.
        # For each batch element, average over its K name-token rows.
        out = []
        for b in range(B):
            rows = name_token_positions[b]
            rows = rows[rows >= 0]            # filter padding/sentinel -1
            if rows.numel() == 0:
                # No name token (e.g. excluded source) → emit uniform.
                u = torch.full(
                    (image_patch_slice.stop - image_patch_slice.start,),
                    1.0 / (image_patch_slice.stop - image_patch_slice.start),
                    device=attn_avg_heads.device,
                    dtype=attn_avg_heads.dtype,
                )
                out.append(u)
                continue
            sub = attn_avg_heads[b, rows, image_patch_slice].mean(dim=0)  # (P,)
            # Re-normalize (the softmax was over all L cols; selecting a
            # slice doesn't sum to 1).
            sub = sub / (sub.sum() + 1e-12)
            out.append(sub)
        return torch.stack(out, dim=0)  # (B, P)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.remove()


def gaussian_target_from_mask(
    bbox_mask_NN: torch.Tensor, sigma_patches: float, eps: float = 1e-8
) -> torch.Tensor:
    """Convert a (P,) bool patch-mask into a (P,) Gaussian-smoothed
    probability distribution peaked on the mask's centroid.

    bbox_mask_NN: shape (P,) where P = grid_h * grid_w (boolean).
    Returns: shape (P,) float, sums to 1.
    """
    P = bbox_mask_NN.numel()
    grid = int(round(P ** 0.5))
    assert grid * grid == P, f"P={P} not a square grid"

    mask_2d = bbox_mask_NN.view(grid, grid).float()
    if mask_2d.sum() == 0:
        # No bbox → uniform (signals "no supervision for this slot").
        return torch.full((P,), 1.0 / P, device=bbox_mask_NN.device,
                          dtype=torch.float32)

    # Centroid in patch coords.
    idx = torch.arange(grid, device=bbox_mask_NN.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    total = mask_2d.sum()
    cy = (mask_2d * yy).sum() / total
    cx = (mask_2d * xx).sum() / total

    # Isotropic Gaussian centered at (cy, cx).
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    g = torch.exp(-dist2 / (2 * sigma_patches ** 2))
    g = g / (g.sum() + eps)
    return g.flatten()


def klal_loss(
    hookset: KLALHookSet,
    image_patch_slice: slice,
    name_token_positions: torch.Tensor,         # (B, K_max), padded with -1
    target_masks: torch.Tensor,                 # (B, P) bool — bbox-aligned per sample
    cfg: KLALConfig,
) -> torch.Tensor:
    """Compute the KLAL loss = mean over layers of KL(target || predicted).

    `target_masks` is a per-sample bool mask over image patches (1 if the
    patch overlaps the prompted celeb's face bbox). We convert it to a
    Gaussian-smoothed distribution `P_target` peaked on the centroid;
    the model's attention `Q^(l)` is computed from hooks.

    KL(target || model) penalises low attention on the target patches.

    Returns a scalar loss. Samples with no target (all-zero mask) are
    skipped (they contribute 0 — equivalent to no supervision for that
    sample).
    """
    B, P = target_masks.shape
    device = target_masks.device

    # P_target per sample.
    p_targets = torch.stack(
        [gaussian_target_from_mask(target_masks[b].bool(), cfg.target_sigma_patches)
         for b in range(B)],
        dim=0,
    )  # (B, P)

    # Identify "valid" samples (those with non-zero target mass — i.e. a
    # bbox was supplied).
    has_target = target_masks.any(dim=1)  # (B,) bool

    total = []
    for layer in cfg.capture_layers:
        q = hookset.get_attention(layer, image_patch_slice, name_token_positions)
        if q is None:
            continue
        q = q.clamp(min=cfg.eps)
        p = p_targets.clamp(min=cfg.eps)
        # KL(P || Q) = sum P log(P/Q). Per-sample.
        kl_per_sample = (p * (p.log() - q.log())).sum(dim=-1)  # (B,)
        kl = kl_per_sample[has_target].mean() if has_target.any() else torch.tensor(0.0, device=device)
        total.append(kl)

    if not total:
        return torch.tensor(0.0, device=device)
    return cfg.lam * torch.stack(total).mean()
