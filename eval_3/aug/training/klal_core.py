"""KL Attention Loss (KLAL) — cross-modal grounding fix per WACV 2026
"Direct Visual Grounding by Directing Attention of Visual Tokens"
(arXiv:2511.12738).

Why we need it: M2 distillation shaped face-patch hidden states to align
with ArcFace centroids on SmolVLA, but the language-name token never
attended to those face patches (attention probe found constant argmax
across prompts → `2026-05-19_attention_probe_step10000`).
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
  Computed by hooking q_proj / k_proj, applying the model's own RoPE
  (cos/sin captured live from `text_model.rotary_emb`), and recomputing
  softmax(QK^T * scaling). Shape (256,), sums to 1. RoPE is required: the
  real Pi0.5 forward RoPEs q/k before attention, so a no-RoPE recompute
  would supervise a content-only proxy decoupled from the attention the
  policy actually uses.
- P_target(S) = a Gaussian-smoothed distribution over the 16x16 patch
  grid, peaked on the face-bbox center for the prompted celeb. Shape
  (256,), sums to 1. Deviation from the paper: WACV 2026 builds P_target
  from the bbox's *center line* of patches (tuned for elongated RefCOCO
  objects); for a compact face bbox we use a 2-D isotropic Gaussian on the
  bbox centroid instead (see `gaussian_target_from_mask`).
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
from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb


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
        # Per-layer attention softmax scale, read straight from the model so
        # the recompute matches the real forward (Gemma uses head_dim**-0.5).
        self._scaling: dict[int, float] = {}
        # RoPE cos/sin captured live from the model's own rotary_emb.
        self._rope: dict = {}
        self._handles = []
        for n in self.layers:
            attn = text_model.layers[n].self_attn
            self._scaling[n] = attn.scaling
            self._handles.append(attn.q_proj.register_forward_hook(self._mk_q(n)))
            self._handles.append(attn.k_proj.register_forward_hook(self._mk_k(n)))
        # One hook on the shared rotary embedding: captures the exact (cos, sin)
        # the model applies, so KLAL supervises the real RoPE'd attention rather
        # than a content-only proxy. `text_model.rotary_emb` is called once per
        # layer inside `compute_layer_complete` (modeling_pi05.py:257) with
        # identical position_ids, so a single live capture is sufficient.
        self._handles.append(
            text_model.rotary_emb.register_forward_hook(self._mk_rope())
        )

    def _mk_q(self, n):
        return lambda mod, inp, out: self._captures[n].__setitem__("q", out)
    def _mk_k(self, n):
        return lambda mod, inp, out: self._captures[n].__setitem__("k", out)
    def _mk_rope(self):
        def hook(mod, inp, out):
            cos, sin = out
            self._rope["cos"] = cos.detach()
            self._rope["sin"] = sin.detach()
        return hook

    def reset(self):
        for n in self.layers:
            self._captures[n].clear()
        self._rope.clear()

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

        # Apply RoPE before the QK^T product. The real Pi0.5 forward RoPEs
        # query/key states inside compute_layer_complete (modeling_pi05.py:
        # 257-260) *before* attention; skipping it supervises a content-only
        # proxy decoupled from the attention the policy actually uses (RoPE
        # inserts a per-(query,key) relative-position rotation that re-ranks
        # patches). We reuse the model's own captured (cos, sin) so the
        # rotation is exact, and apply it before the GQA key expansion — cos/sin
        # broadcast over heads via unsqueeze_dim=1, matching the real forward.
        cos = self._rope.get("cos")
        sin = self._rope.get("sin")
        if cos is None or sin is None:
            raise RuntimeError(
                "KLAL: rotary_emb hook captured no (cos, sin) — the RoPE hook "
                "did not fire. Aborting rather than silently supervising a "
                "no-RoPE proxy (no silent fallbacks)."
            )
        # cos/sin cover the fused prefix+suffix sequence; q/k here are
        # prefix-only. compute_layer_complete concatenates the prefix FIRST,
        # so rows [:L] are exactly the prefix positions. Assert it explicitly.
        assert cos.shape[1] >= L, (
            f"KLAL: captured cos/sin seq-len {cos.shape[1]} < prefix len {L} — "
            f"the prefix-first layout assumption is violated."
        )
        cos = cos[:, :L].to(dtype=q.dtype)
        sin = sin[:, :L].to(dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scaling[layer]
        # Softmax over all prefix columns. We omit the attention mask: Pi0.5's
        # prefix is fully bidirectional (every prefix att_mask=0), so there is
        # no causal term among prefix tokens. Padded language columns are left
        # unmasked here, unlike the real eager_attention_forward — but the loss
        # slices and RE-NORMALISES over the 256 image-patch columns (never
        # padded), which divides out the shared softmax denominator; the
        # residual effect on the supervised image distribution is second-order
        # (confirmed by 2-agent review, 2026-05-20).
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
