"""KLAL attention-supervision hookset for SmolVLA's policy (action) forward.

`m2_klal.KLALHookSet` is Pi0.5/PaliGemma-specific: it hooks the shared
`text_model.rotary_emb` module and uses Gemma's `apply_rotary_pos_emb`.
SmolVLA has neither — its custom forward (`smolvlm_with_expert.py`) applies
RoPE on the fly via the module-level `apply_rope(x, positions)` function and
never instantiates a rotary_emb module.

This file is the SmolVLA twin, targeting the **policy / robot-action**
forward path (`SmolVLMWithExpertModel.forward`):

- hooks `text_model.layers[n].self_attn.{q,k}_proj` — these fire on the VLM
  prefix stream only (the action expert uses `lm_expert.layers[...]`),
- captures the live `position_ids` by wrapping the module-level `apply_rope`
  (SmolVLA calls `vlm_with_expert.forward` directly, bypassing nn.Module
  `__call__`, so a forward-pre-hook never fires; `apply_rope` is called for
  q/k on every layer with the exact `cumsum(pad_masks)-1` positions — NOT a
  plain arange, which would mis-RoPE on padded tokens),
- recomputes attention with SmolVLA's own `apply_rope` and the eager
  softmax scale `head_dim ** -0.5`, exactly matching `forward_attn_layer`.

The loss (`klal_loss`) and the target builder (`gaussian_target_from_mask`)
are model-agnostic and reused from `m2_klal.py` unchanged.

Why the recompute is faithful (same argument as the Pi0.5 KLAL):
- SmolVLA's prefix (image + language) is fully bidirectional
  (`make_att_2d_masks`, att_mask=0 across the whole prefix), and prefix rows
  attend only to prefix columns (the suffix carries att_mask=1, masked out).
  So the real prefix->prefix attention equals softmax(QK^T * scale) over
  prefix columns — which is exactly what we recompute from the VLM-stream
  q/k capture.
- RoPE MUST be applied: `forward_attn_layer` RoPEs q/k before attention; a
  no-RoPE recompute would supervise a proxy decoupled from the policy's real
  attention — the HIGH-severity bug found in the Pi0.5 KLAL
  (docs/experiments/2026-05-20_track_E_method_validation.md §3).
"""
from __future__ import annotations

import torch

from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope


class KLALHookSetSmolVLA:
    """Multi-layer q/k hooks + position_ids capture for SmolVLA's VLM.

    Use as a context manager so the hooks are removed on exit, or call
    `remove()` explicitly.
    """

    def __init__(self, text_model, vlm_with_expert, layers, n_heads,
                 n_kv_heads, head_dim):
        self.layers = list(layers)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        # Eager softmax scale, matching smolvlm_with_expert.eager_attention_forward.
        self.scaling = head_dim ** -0.5
        self._captures: dict[int, dict] = {n: {} for n in self.layers}
        self._position_ids: torch.Tensor | None = None
        self._handles = []
        for n in self.layers:
            attn = text_model.layers[n].self_attn
            self._handles.append(attn.q_proj.register_forward_hook(self._mk_q(n)))
            self._handles.append(attn.k_proj.register_forward_hook(self._mk_k(n)))
        # position_ids: SmolVLA's modeling code calls `vlm_with_expert.forward(...)`
        # directly (bypassing nn.Module.__call__), so a forward-pre-hook on the
        # module never fires. Instead wrap the module-level `apply_rope` — it is
        # called for q/k on every layer with the exact `positions` the model
        # RoPEs with (`cumsum(pad_masks)-1`, which repeats on padded tokens, so
        # a plain arange would mis-RoPE the recompute). The wrapper is undone in
        # `remove()`.
        import lerobot.policies.smolvla.smolvlm_with_expert as _swe
        self._swe = _swe
        self._orig_apply_rope = _swe.apply_rope
        _orig = self._orig_apply_rope
        _holder = self

        def _apply_rope_capturing(x, positions, max_wavelength=10_000):
            if _holder._position_ids is None:
                _holder._position_ids = positions.detach()
            return _orig(x, positions, max_wavelength)

        _swe.apply_rope = _apply_rope_capturing
        # Image-prefix length is needed to map a language-token position to its
        # prefix row. SmolVLA pads the prefix to a fixed length and the
        # image-stream count depends on `empty_cameras`, so guessing it from
        # config is fragile — measure it at runtime: the connector runs once
        # per image stream and emits (B, patches_per_image, D).
        self._n_image_calls = 0
        self._patches_per_image: int | None = None
        connector = vlm_with_expert.get_vlm_model().connector
        self._handles.append(
            connector.register_forward_hook(self._capture_connector)
        )

    def _mk_q(self, n):
        return lambda mod, inp, out: self._captures[n].__setitem__("q", out)

    def _mk_k(self, n):
        return lambda mod, inp, out: self._captures[n].__setitem__("k", out)

    def _capture_connector(self, mod, inp, out):
        self._n_image_calls += 1
        self._patches_per_image = out.shape[1]

    def patches_per_image(self) -> int | None:
        """Image-patch count per camera stream (8x8=64 for SmolVLA), or None
        if no forward has run since the last reset."""
        return self._patches_per_image

    def image_prefix_len(self) -> int | None:
        """Total image-patch rows before the language tokens in the prefix
        (= n_image_streams * patches_per_image), or None if not yet measured.

        Assumes `add_image_special_tokens=False` (the SmolVLA default and the
        M2-toolkit invariant) — with special tokens on, the image block also
        carries per-image start/end tokens not counted here.
        """
        if self._patches_per_image is None or self._n_image_calls == 0:
            return None
        return self._n_image_calls * self._patches_per_image

    def reset(self):
        for n in self.layers:
            self._captures[n].clear()
        self._position_ids = None
        self._n_image_calls = 0
        self._patches_per_image = None

    def get_attention(self, layer: int, image_patch_slice: slice,
                      name_token_positions: torch.Tensor) -> torch.Tensor | None:
        """Attention from name-token rows to image-patch cols at one layer.

        Returns (B, P) head-averaged, name-token-averaged attention over the
        P image patches in `image_patch_slice`, or None if hooks didn't fire.
        """
        cap = self._captures.get(layer, {})
        q = cap.get("q")
        k = cap.get("k")
        if q is None or k is None:
            return None
        if self._position_ids is None:
            raise RuntimeError(
                "KLAL-SmolVLA: position_ids hook captured nothing — aborting "
                "rather than supervising a wrong-RoPE proxy (CLAUDE.md §5)."
            )

        B, L, _ = q.shape
        q = q.float().view(B, L, self.n_heads, self.head_dim)
        k = k.float().view(B, L, self.n_kv_heads, self.head_dim)

        # SmolVLA's apply_rope expects x as [B, L, H, D] and positions [B, L];
        # it is applied to q/k before attention in forward_attn_layer. The
        # q/k captured here are the VLM (prefix) stream only, so the prefix
        # positions are position_ids[:, :L].
        pos = self._position_ids[:, :L].to(q.device)
        if pos.shape[0] != B:
            pos = pos.expand(B, -1)
        q = apply_rope(q, pos)
        k = apply_rope(k, pos)

        # [B, L, H, D] -> [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # Softmax over all prefix columns. The attention mask is omitted: the
        # SmolVLA prefix is fully bidirectional, so there is no causal term
        # among prefix tokens. Padded language columns are left unmasked here,
        # but the loss slices and RE-NORMALISES over the image-patch columns
        # (never padded), which divides out the shared softmax denominator —
        # a second-order deviation (same argument as the Pi0.5 KLAL).
        attn = torch.softmax(scores, dim=-1)        # (B, H, L, L)
        attn_avg_heads = attn.mean(dim=1)           # (B, L, L)

        P = image_patch_slice.stop - image_patch_slice.start
        out = []
        for b in range(B):
            rows = name_token_positions[b]
            rows = rows[rows >= 0]                  # drop -1 padding/sentinels
            if rows.numel() == 0:
                # No name token located for this sample -> uniform (no signal).
                out.append(torch.full((P,), 1.0 / P,
                                      device=attn_avg_heads.device,
                                      dtype=attn_avg_heads.dtype))
                continue
            sub = attn_avg_heads[b, rows, image_patch_slice].mean(dim=0)  # (P,)
            sub = sub / (sub.sum() + 1e-12)         # renormalize over image cols
            out.append(sub)
        return torch.stack(out, dim=0)              # (B, P)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        # Restore the un-wrapped apply_rope.
        if getattr(self, "_orig_apply_rope", None) is not None:
            self._swe.apply_rope = self._orig_apply_rope
            self._orig_apply_rope = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()


# ---------------------------------------------------------------------------
# Name-token location — find each sample's prompted celeb name inside the
# tokenized language prompt, so KLAL knows which rows to supervise.
# ---------------------------------------------------------------------------

# The track-3 baseline dataset is the 3-celeb TOY bucket; full names as they
# appear verbatim in every training prompt (verified across all 9,394
# episodes by the Track E data audit).
CELEB_FULL_NAMES = {
    "swift": "Taylor Swift",
    "obama": "Barack Obama",
    "lecun": "Yann LeCun",
}


def build_name_token_ids(tokenizer, full_names: dict[str, str] | None = None
                         ) -> dict[str, list[int]]:
    """Tokenize each celeb full name against the SmolVLM2 tokenizer.

    Tries a leading-space (BPE-friendly) variant first, then bare.
    """
    full_names = full_names or CELEB_FULL_NAMES
    ids: dict[str, list[int]] = {}
    for short, name in full_names.items():
        for variant in (" " + name, name):
            t = tokenizer.encode(variant, add_special_tokens=False)
            if t:
                ids[short] = t
                break
    return ids


def extract_name_token_positions(
    lang_tokens: torch.Tensor,                 # (B, L_lang) token ids
    target_shorts: list[str],                  # per-sample celeb short slug
    name_token_ids: dict[str, list[int]],
) -> torch.Tensor | None:
    """Return (B, K_max) positions of each sample's celeb name inside the
    language token sequence, padded with -1. None if no name was found.

    Positions are WITHIN the language sequence; the caller offsets them by
    the image-patch span to get prefix-row indices.
    """
    if not target_shorts:
        return None
    B, L = lang_tokens.shape
    rows: list[list[int]] = []
    max_k = 0
    for b, short in enumerate(target_shorts):
        ids = name_token_ids.get(short)
        if not ids:
            rows.append([])
            continue
        seq = lang_tokens[b].detach().cpu().tolist()
        n = len(ids)
        found: list[int] = []
        for i in range(L - n + 1):
            if seq[i:i + n] == ids:
                found = list(range(i, i + n))
                break
        rows.append(found)
        max_k = max(max_k, len(found))
    if max_k == 0:
        return None
    out = torch.full((B, max_k), -1, dtype=torch.long)
    for b, pos in enumerate(rows):
        for k, p in enumerate(pos):
            out[b, k] = p
    return out
