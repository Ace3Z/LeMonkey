#!/usr/bin/env python3
"""KL Attention Loss (KLAL) for SmolVLM2 — the SmolVLA-450M port of the
Pi0.5/PaliGemma KLAL in `eval_3/aug/m2_klal.py`.

Why: the attention-routing diagnosis (docs/experiments/2026-05-19_attention_
probe_step10000) found the name-token attention is sink-locked to a constant
background patch for every prompt — both on Pi0.5 AND SmolVLA. The VQA naming
loss alone never trains the "which of the three printed faces" decision because
naming an isolated face never exercises routing. KLAL supervises the attention
distribution from name-tokens → image-patches with a Gaussian target built from
the prompted celeb's face bounding box. (WACV 2026, arXiv:2511.12738; ObjectVLA
arXiv:2502.19250 §4.1.2 reports bbox grounding lifts OOD 19%→64%.)

What changed from the PaliGemma port (all verified from source, not guessed):
  - SmolVLM2-500M: scale_factor=4, image 512/patch 16 → 1024 SigLIP patches →
    pixel-shuffle ÷16 → **64 image tokens = 8x8 grid** (PaliGemma was 256/16x16).
    Verified: HuggingFaceTB/SmolVLM2-500M-Video-Instruct config.json + the
    pixel_shuffle in transformers/models/smolvlm/modeling_smolvlm.py:437.
  - Text model is **llama** arch (not gemma) → Llama RoPE (rotate_half), head_dim
    64, 15 attn heads / 5 KV heads (GQA, repeat 3x), rope_theta 1e5. Verified:
    same config.json text_config.
  - Image-patch columns are **not a fixed prefix slice**: SmolVLM inserts image
    tokens inline (image_token_id=49190) wrapped by fake-image tokens. We locate
    them per-sample via `input_ids == image_token_id`.
  - SmolVLA truncates text_model.layers to the first 16 (smolvlm_with_expert.py:90)
    → capture_layers must be in [0, 15]. Default (6, 9, 12, 15).
  - SmolVLM text decoder is **causal** (Llama), not bidirectional prefix-LM. Name
    tokens come AFTER the image tokens in the sequence, so every image column is
    causally visible from a name-token row; slicing+renormalising over image
    columns is well-posed (the shared softmax denominator divides out — same
    argument as the PaliGemma port's 2-agent-reviewed note).

Like the reference, KLAL recomputes softmax(QK^T) from hooked q_proj/k_proj
outputs, so it works under SDPA/flash — no `output_attentions=True` / eager
attention required.

Per CLAUDE.md §5: every fallback emits [WARN]. Per CLAUDE.md §8: every constant
above is read from source/config. Per CLAUDE.md §9: this is non-trivial — get a
parallel-agent review + the smoke gates in README before trusting a long run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Llama RoPE (SmolLM2 text backbone) — rotate_half form. Identical math to the
# canonical transformers.models.llama.modeling_llama.apply_rotary_pos_emb, kept
# local so we don't depend on a specific transformers layout. cos/sin are the
# model's OWN captured tensors (see KLALHookSet), so the rotation is exact.
# -----------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin, unsqueeze_dim: int = 1):
    # cos/sin: (B, L, head_dim). Broadcast over the head axis.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


@dataclass
class KLALConfig:
    # Layers to supervise — MUST be within SmolVLA's truncated 0..15 range.
    capture_layers: tuple = (6, 9, 12, 15)
    target_sigma_patches: float = 1.0   # Gaussian std in 8x8-grid patch units
    lam: float = 1.0                    # loss scale (WACV 2026 default)
    eps: float = 1e-8


class KLALHookSet:
    """Hooks q_proj / k_proj on the (truncated) SmolVLM2 text layers and the
    text rotary embedding, then recomputes name-token→image-patch attention.

    Use as a context manager so the hooks are removed on exit.
    """

    def __init__(self, text_model, layers, n_heads, n_kv_heads, head_dim):
        # text_model is the SmolLM2 decoder: `vlm.model.text_model`.
        # Its decoder blocks live at text_model.layers[i] with .self_attn.
        self.layers = list(layers)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self._captures: dict[int, dict] = {n: {} for n in self.layers}
        self._scaling: dict[int, float] = {}
        self._rope: dict = {}
        self._handles = []

        n_total = len(text_model.layers)
        for n in self.layers:
            if n >= n_total:
                raise ValueError(
                    f"KLAL capture layer {n} >= text_model has {n_total} layers. "
                    f"SmolVLA truncates to 16; capture_layers must be in [0,{n_total-1}]."
                )
            attn = text_model.layers[n].self_attn
            # Llama-style attention exposes `scaling` (head_dim**-0.5). If a
            # given transformers version names it differently, fall back loudly.
            scaling = getattr(attn, "scaling", None)
            if scaling is None:
                scaling = head_dim ** -0.5
                print(f"[WARN] KLAL layer {n}: expected attn.scaling attribute, "
                      f"got=missing, fallback=head_dim**-0.5={scaling:.5f}",
                      flush=True)
            self._scaling[n] = scaling
            self._handles.append(attn.q_proj.register_forward_hook(self._mk_q(n)))
            self._handles.append(attn.k_proj.register_forward_hook(self._mk_k(n)))

        # Capture the model's own RoPE (cos, sin). Llama rotary_emb.forward
        # returns (cos, sin). One shared rotary module is reused across layers.
        rotary = getattr(text_model, "rotary_emb", None)
        if rotary is None:
            raise RuntimeError(
                "KLAL: text_model has no `rotary_emb` — cannot capture the exact "
                "RoPE. Refusing to supervise a no-RoPE proxy (CLAUDE.md §5)."
            )
        self._handles.append(rotary.register_forward_hook(self._mk_rope()))

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

    def get_attention(self, layer: int, image_cols: torch.Tensor,
                      name_token_positions: torch.Tensor) -> torch.Tensor | None:
        """Attention from name-token rows to image-patch cols at one layer.

        Args:
          layer: layer index (in self.layers).
          image_cols: (B, P) long tensor of the P image-token column indices per
            sample (located via input_ids == image_token_id). P = 64 for SmolVLM2.
          name_token_positions: (B, K_max) long, padded with -1 — the sequence
            positions of the target-name tokens per sample.

        Returns (B, P) attention distribution, head-averaged + name-row-averaged,
        renormalised over the P image columns. None if hooks didn't fire.
        """
        cap = self._captures.get(layer, {})
        q = cap.get("q")
        k = cap.get("k")
        if q is None or k is None:
            return None

        B, L, _ = q.shape
        q = q.float().view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.float().view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos = self._rope.get("cos")
        sin = self._rope.get("sin")
        if cos is None or sin is None:
            raise RuntimeError(
                "KLAL: rotary_emb hook captured no (cos, sin) — RoPE hook did not "
                "fire. Aborting rather than supervising a no-RoPE proxy (§5)."
            )
        # cos/sin are (B, L, head_dim) for the same sequence as q/k.
        if cos.shape[1] != L:
            # Defensive: some versions return (1, L, hd). Broadcast/clip.
            if cos.shape[1] >= L:
                cos = cos[:, :L]
                sin = sin[:, :L]
            else:
                raise RuntimeError(
                    f"KLAL: captured cos seq-len {cos.shape[1]} < q seq-len {L} — "
                    f"sequence-layout assumption violated."
                )
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        q, k = _apply_rope(q, k, cos, sin, unsqueeze_dim=1)

        # GQA: expand KV heads to match query heads.
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scaling[layer]
        # NOTE on causality: SmolVLM's decoder is causal, but name tokens come
        # AFTER all image tokens, so every image column is in the past of a
        # name-token row (always visible). We softmax over all columns and then
        # slice+renormalise over the image columns only; the shared denominator
        # divides out, so the supervised image distribution is correct. (Same
        # argument as the PaliGemma port, which a 2-agent review signed off.)
        attn = torch.softmax(scores, dim=-1)   # (B, H, L, L)
        attn = attn.mean(dim=1)                # (B, L, L) head-averaged

        out = []
        P = image_cols.shape[1]
        for b in range(B):
            rows = name_token_positions[b]
            rows = rows[rows >= 0]
            cols = image_cols[b]
            if rows.numel() == 0 or cols.numel() == 0:
                out.append(torch.full((P,), 1.0 / P, device=attn.device,
                                      dtype=attn.dtype))
                continue
            sub = attn[b][rows][:, cols].mean(dim=0)   # (P,)
            sub = sub / (sub.sum() + 1e-12)
            out.append(sub)
        return torch.stack(out, dim=0)   # (B, P)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()


# -----------------------------------------------------------------------------
# bbox → patch target distribution
# -----------------------------------------------------------------------------

def bbox_to_patch_mask(bbox_xyxy_norm, grid: int, device) -> torch.Tensor:
    """Normalised [x1,y1,x2,y2] in [0,1] → (grid*grid,) bool patch mask.

    The pixel-shuffle output preserves spatial order (it coarsens the SigLIP
    grid by scale_factor in each axis), so a normalised box maps to the coarse
    grid the same way it would to the fine grid. grid = int(sqrt(image_seq_len))
    = 8 for SmolVLM2-500M.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy_norm]
    # Clamp to [0,1].
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    mask = torch.zeros(grid, grid, dtype=torch.bool, device=device)
    # Zero-area box (e.g. the [0,0,0,0] "no bbox" sentinel) → empty mask, which
    # KLAL treats as "no supervision for this sample". Do NOT force a patch:
    # marking patch (0,0) would train attention toward the top-left corner —
    # the exact sink-lock failure KLAL exists to fix.
    if (x2 - x1) <= 0.0 or (y2 - y1) <= 0.0:
        return mask.flatten()
    # Patch column/row index ranges the box covers (inclusive).
    c1 = int(x1 * grid); c2 = min(grid - 1, int(x2 * grid))
    r1 = int(y1 * grid); r2 = min(grid - 1, int(y2 * grid))
    c2 = max(c1, c2)
    r2 = max(r1, r2)
    mask[r1 : r2 + 1, c1 : c2 + 1] = True
    return mask.flatten()


def gaussian_target_from_mask(mask_P: torch.Tensor, sigma_patches: float,
                              eps: float = 1e-8) -> torch.Tensor:
    """(P,) bool mask → (P,) Gaussian distribution peaked on the mask centroid."""
    P = mask_P.numel()
    grid = int(round(P ** 0.5))
    assert grid * grid == P, f"P={P} not a square grid"
    mask_2d = mask_P.view(grid, grid).float()
    if mask_2d.sum() == 0:
        # No bbox → uniform (signals "no supervision for this sample").
        return torch.full((P,), 1.0 / P, device=mask_P.device, dtype=torch.float32)
    idx = torch.arange(grid, device=mask_P.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    total = mask_2d.sum()
    cy = (mask_2d * yy).sum() / total
    cx = (mask_2d * xx).sum() / total
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    g = torch.exp(-dist2 / (2 * sigma_patches ** 2))
    g = g / (g.sum() + eps)
    return g.flatten()


# -----------------------------------------------------------------------------
# KLAL loss
# -----------------------------------------------------------------------------

def klal_loss(hookset: KLALHookSet, image_cols: torch.Tensor,
              name_token_positions: torch.Tensor, target_masks: torch.Tensor,
              cfg: KLALConfig) -> torch.Tensor:
    """L_KLAL = (1/|layers|) Σ_l KL( P_target || Q^(l) ), averaged over samples
    that have a bbox. Samples with an all-zero mask contribute 0 (no supervision).

    image_cols:            (B, P) long — image-token column indices per sample.
    name_token_positions:  (B, K_max) long, -1 padded — name-token rows.
    target_masks:          (B, P) bool — image-patch mask of the prompted face.
    """
    B, P = target_masks.shape
    device = target_masks.device

    p_targets = torch.stack(
        [gaussian_target_from_mask(target_masks[b].bool(), cfg.target_sigma_patches)
         for b in range(B)],
        dim=0,
    )  # (B, P)
    has_target = target_masks.any(dim=1)   # (B,)

    total = []
    for layer in cfg.capture_layers:
        q = hookset.get_attention(layer, image_cols, name_token_positions)
        if q is None:
            print(f"[WARN] KLAL: layer {layer} hook did not fire, "
                  f"got=q/k missing, fallback=skip this layer this step",
                  flush=True)
            continue
        q = q.clamp(min=cfg.eps)
        p = p_targets.clamp(min=cfg.eps)
        kl_per_sample = (p * (p.log() - q.log())).sum(dim=-1)   # (B,)
        if has_target.any():
            total.append(kl_per_sample[has_target].mean())
        else:
            total.append(torch.tensor(0.0, device=device))

    if not total:
        return torch.tensor(0.0, device=device)
    return cfg.lam * torch.stack(total).mean()
