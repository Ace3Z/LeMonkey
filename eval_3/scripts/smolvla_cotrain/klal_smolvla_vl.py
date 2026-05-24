"""KLAL attention supervision on SmolVLA's VL co-training forward pass.

This module supervises the celebrity-name token to attend to the prompted
celeb's printed-portrait region. It computes that loss on the **VL
batches** (the `so101_eval3_cotrain_grounding` grounding stream), the
companion to `klal_smolvla_action.py`, which handles the robot-action
forward.

Why a separate module, verified facts:

- SmolVLA's VLM (SmolVLM2-500M) text model is a stock transformers
  `LlamaModel`: a shared `rotary_emb` module emits `(cos, sin)`, and
  `LlamaAttention` applies `apply_rotary_pos_emb` to q/k before QK^T.
  GQA: 15 query heads / 5 key-value heads.
- The VL forward is a **causal** LM forward — the attention recompute MUST
  apply a causal mask. (The robot/Pi0.5 KLAL omit it; their prefix is fully
  bidirectional. Do not copy that here.)
- With `do_image_splitting=False` the SmolVLM processor stretch-resizes each
  image to 512x512 (aspect NOT preserved) and emits exactly **64 contiguous
  `<image>` tokens** (id 49190) on a row-major 8x8 patch grid. So a quad
  normalised in original-image coords maps directly: `patch = floor(coord*8)`.
- lerobot truncates the LLM to its first 16 layers — capture layers must be
  in `[0, 16)`.

The loss (`klal_loss`), the target builder (`gaussian_target_from_mask`) and
the config (`KLALConfig`) are model-agnostic and reused from `klal_core.py`.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# SmolVLM2 `<image>` placeholder token id (verified via the processor).
IMAGE_TOKEN_ID = 49190
PATCH_GRID = 8
NUM_IMAGE_PATCHES = PATCH_GRID * PATCH_GRID  # 64


def _celeb_short(slug: str) -> str:
    """Map a VL-dataset `celeb_slug` (e.g. 'barack_obama', 'taylor_swift',
    'yann_lecun') to the short key ('obama' / 'swift' / 'lecun') that
    `build_name_token_ids` / `CELEB_FULL_NAMES` are keyed on. An
    already-short slug passes through unchanged."""
    return slug.rsplit("_", 1)[-1]


class KLALHookSetSmolVLMVL:
    """q/k + ``rotary_emb`` hooks on the SmolVLM2 Llama text model (VL forward).

    Use as a context manager so the hooks are torn down deterministically, or
    call ``remove()`` explicitly. Call ``reset()`` and ``set_attention_mask(...)``
    once per forward pass.

    Args:
        text_model: The SmolVLM2 ``LlamaModel`` text-model module (typically
            ``vlm.model.text_model``).
        layers: Iterable of transformer layer indices to instrument.
        n_heads: Number of query attention heads per layer.
        n_kv_heads: Number of key/value heads (GQA: typically ``< n_heads``).
        head_dim: Per-head hidden size.
        scaling: Softmax pre-scale used by the layer's eager attention; for
            Llama this is ``head_dim ** -0.5``.
    """

    def __init__(self,
                 text_model: "torch.nn.Module",
                 layers: "Iterable[int]",
                 n_heads: int,
                 n_kv_heads: int,
                 head_dim: int,
                 scaling: float) -> None:
        self.layers = list(layers)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.scaling = float(scaling)
        self._cap: dict[int, dict] = {n: {} for n in self.layers}
        self._rope: dict[str, torch.Tensor] = {}
        self._attn_mask: torch.Tensor | None = None
        self._handles = []
        for n in self.layers:
            attn = text_model.layers[n].self_attn
            self._handles.append(attn.q_proj.register_forward_hook(self._mk(n, "q")))
            self._handles.append(attn.k_proj.register_forward_hook(self._mk(n, "k")))
        # The shared rotary_emb runs once per forward and emits the exact
        # (cos, sin) every LlamaAttention layer RoPEs q/k with — capture it so
        # the recompute is the model's real rotation, not a no-RoPE proxy.
        self._handles.append(
            text_model.rotary_emb.register_forward_hook(self._cap_rope))

    def _mk(self, n, which):
        def hook(_mod, _inp, out):
            self._cap[n][which] = out
        return hook

    def _cap_rope(self, _mod, _inp, out):
        # LlamaRotaryEmbedding.forward returns (cos, sin), each (B, L, head_dim).
        cos, sin = out
        self._rope["cos"] = cos
        self._rope["sin"] = sin

    def set_attention_mask(self, attn_mask: torch.Tensor) -> None:
        """Stash the VL batch's (B, L) attention mask so the recompute can
        mask padding columns. Call once per forward, before `klal_loss`."""
        self._attn_mask = attn_mask

    def reset(self):
        for n in self.layers:
            self._cap[n].clear()
        self._rope.clear()
        self._attn_mask = None

    def get_attention(self, layer: int, image_patch_slice: slice,
                      name_token_positions: torch.Tensor) -> torch.Tensor | None:
        """(B, P) name-token -> image-patch attention at `layer`, head- and
        name-averaged and renormalised over the P image columns. None if the
        q/k hooks did not fire for `layer`.
        """
        cap = self._cap.get(layer, {})
        q = cap.get("q")
        k = cap.get("k")
        if q is None or k is None:
            return None
        cos = self._rope.get("cos")
        sin = self._rope.get("sin")
        if cos is None or sin is None:
            raise RuntimeError(
                "KLAL-VL: rotary_emb hook captured no (cos, sin) — aborting "
                "rather than supervising a no-RoPE proxy.")

        B, L, _ = q.shape
        q = q.float().view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.float().view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # RoPE with the model's own captured (cos, sin); default unsqueeze_dim=1
        # broadcasts cos/sin over heads, exactly as LlamaAttention.forward does.
        c = cos[:, :L].to(dtype=q.dtype)
        s = sin[:, :L].to(dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, c, s)
        if self.n_kv_heads != self.n_heads:                       # GQA expand
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # (B,H,L,L)
        # Causal mask — the VL forward is a causal LM (NOT a bidirectional
        # prefix like the robot/Pi0.5 KLAL). Column j > row i is the future.
        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=scores.device,
                       dtype=scores.dtype), diagonal=1)
        scores = scores + causal
        # Padding-column mask — left out by the robot KLAL as second-order, but
        # cheap here (L is small with do_image_splitting=False) so we do it.
        if self._attn_mask is not None:
            pad = (self._attn_mask.to(scores.device)[:, None, None, :] == 0)
            scores = scores.masked_fill(pad, float("-inf"))
        # Head-average the attention probabilities, then (per sample, below)
        # renormalise over the image columns — same ordering as the established
        # klal_core.KLALHookSet. Defensive nan_to_num: under right-padding no row
        # is fully -inf so softmax is NaN-free, but guard so a layout surprise
        # degrades to 0 rather than a silent NaN.
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1).mean(dim=1))

        P = image_patch_slice.stop - image_patch_slice.start
        out = []
        for b in range(B):
            rows = name_token_positions[b]
            rows = rows[rows >= 0]                # drop -1 padding/sentinels
            if rows.numel() == 0:
                # No name token located -> uniform (contributes no signal).
                out.append(torch.full((P,), 1.0 / P, device=attn.device,
                                      dtype=attn.dtype))
                continue
            sub = attn[b, rows, image_patch_slice].mean(dim=0)    # (P,)
            sub = sub / (sub.sum() + 1e-12)       # renormalise over image cols
            out.append(sub)
        return torch.stack(out, dim=0)            # (B, P)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()


def quad_to_patch_mask(quad_corners_norm, grid: int = PATCH_GRID
                       ) -> torch.Tensor:
    """Axis-aligned (grid*grid,) bool patch mask from a normalised portrait
    quad (4 corners, [x, y] in [0, 1]).

    With `do_image_splitting=False` the image is stretch-resized onto the
    patch grid, so `floor(coord*grid)` is the patch index directly. The mask
    is the axis-aligned box of the 4 corners: `gaussian_target_from_mask`
    uses only the mask centroid, and the AABB of a (possibly in-plane-rotated)
    rectangle shares that rectangle's centroid. Row-major flatten matches the
    raster order of the 64 image tokens.
    """
    q = np.asarray(quad_corners_norm, dtype=np.float32).reshape(-1, 2)
    xs = np.clip(q[:, 0], 0.0, 1.0 - 1e-6)
    ys = np.clip(q[:, 1], 0.0, 1.0 - 1e-6)
    px0, px1 = int(np.floor(xs.min() * grid)), int(np.floor(xs.max() * grid))
    py0, py1 = int(np.floor(ys.min() * grid)), int(np.floor(ys.max() * grid))
    mask = np.zeros((grid, grid), dtype=bool)
    mask[py0:py1 + 1, px0:px1 + 1] = True
    return torch.from_numpy(mask.reshape(-1))


def find_image_patch_slice(input_ids_row: torch.Tensor) -> slice | None:
    """Locate the 64 contiguous `<image>` tokens in one VL `input_ids` row.

    Returns None if the row does not hold exactly 64 contiguous image tokens
    (e.g. `do_image_splitting` was left on, or the layout is unexpected).
    """
    pos = (input_ids_row == IMAGE_TOKEN_ID).nonzero().flatten()
    if pos.numel() != NUM_IMAGE_PATCHES:
        return None
    lo, hi = int(pos.min()), int(pos.max())
    if hi - lo + 1 != NUM_IMAGE_PATCHES:
        return None
    return slice(lo, lo + NUM_IMAGE_PATCHES)


def compute_klal_loss_vl(hookset: KLALHookSetSmolVLMVL, cfg, name_ids: dict,
                         batch: dict, device) -> torch.Tensor:
    """KLAL loss for one VL batch.

    Reads the q/k/(cos,sin) the `hookset` captured during the VL forward,
    builds the per-sample portrait-quad target, and returns the scaled loss
    (`cfg.lam` applied inside `klal_loss`). Returns a 0-d tensor (logged
    once) when the image span or name tokens cannot be located.

    Required `batch` keys: `input_ids`, `attention_mask`, `celeb_slug`
    (list[str]), `quad_corners_norm` ((B,4,2)); optional `bbox_refit_ok`
    ((B,) bool) — samples whose quad refit failed are skipped.
    """
    from klal_core import klal_loss
    from klal_smolvla_action import extract_name_token_positions

    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    celeb_slugs = list(batch["celeb_slug"])
    quads = batch["quad_corners_norm"]
    refit_ok = batch.get("bbox_refit_ok")
    B = input_ids.shape[0]

    # The VL batch is right-padded (the collator's prompt-length label masking
    # relies on it), so the image span is identical across rows.
    img_slice = find_image_patch_slice(input_ids[0])
    # `celeb_slug` is the dataset's long form ('barack_obama'); `name_ids` is
    # keyed on the short form ('obama') — normalise before the lookup.
    shorts = [_celeb_short(s) for s in celeb_slugs]
    name_pos = extract_name_token_positions(input_ids, shorts, name_ids)
    if img_slice is None or name_pos is None:
        cause = (f"img_slice_ok={img_slice is not None},"
                 f"name_pos_ok={name_pos is not None}")
        seen = getattr(compute_klal_loss_vl, "_warned", set())
        if cause not in seen:
            print(f"[WARN] KLAL-VL: no supervision this step — expected 64 "
                  f"contiguous <image> tokens + locatable name tokens, got "
                  f"{cause}, fallback=KLAL contributes 0 "
                  f"(logged once per cause)", flush=True)
            seen.add(cause)
            compute_klal_loss_vl._warned = seen
        return torch.zeros((), device=device)

    masks = []
    for b in range(B):
        if refit_ok is not None and not bool(refit_ok[b]):
            # Quad refit failed for this pair — skip it (all-zero mask; the
            # klal_loss `has_target` filter then excludes the sample).
            masks.append(torch.zeros(NUM_IMAGE_PATCHES, dtype=torch.bool))
        else:
            masks.append(quad_to_patch_mask(quads[b]))
    target_masks = torch.stack(masks, dim=0).to(device)

    hookset.set_attention_mask(attn_mask)
    return klal_loss(hookset, image_patch_slice=img_slice,
                     name_token_positions=name_pos.to(device),
                     target_masks=target_masks, cfg=cfg)
