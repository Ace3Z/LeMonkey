"""Pi0.5 (PaliGemma) hook + partial-freeze for the M2 ArcFace alignment loss.

Pi0.5's text LM lives at:
    policy.model.paligemma_with_expert.paligemma.model.language_model

It has 18 layers (gemma_2b, num_hidden_layers=18, see configuration_pi05.py
+ modeling_pi05.py:322-330). BlindVLA Table 12 found the optimum hook
depth at ~57% of LM depth (16 of 28 on Llama-2-7B); on Gemma-2B's 18
layers, 57% ≈ layer 10 (capture layer 10; hook on layer 11's
input_layernorm).

We attach a `forward_pre_hook` on `layers[11].input_layernorm`. The
existing Pi0.5 forward calls `layernorm_forward(layer.input_layernorm,
hidden_states, adarms_cond[i])` with `adarms_cond[0] = None` for the VLM
stream (see modeling_pi05.py:236 and `use_adarms=[False, True]` at
`PaliGemmaWithExpertModel.__init__`). The pre-hook fires on the VLM side
ONLY because the expert uses a different layernorm module instance.

The captured tensor shape is `(B, prefix_len, 2048)` — fp32, since
Pi0.5 keeps the LN in fp32 (modeling_pi05.py:408-418).

Partial-freeze: freeze `language_model.embed_tokens` and
`language_model.layers[0..N-1]` (where N = capture_layer). Leaves layers
N..end and final norm trainable. Vision tower is already frozen if
`--policy.freeze_vision_encoder=True`. The action expert is independent
and unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# 57% depth on Gemma-2B 18 layers → capture layer 10, hook layer 11.
DEFAULT_PI05_CAPTURE_LAYER = 10
PI05_LM_DEPTH = 18  # gemma_2b


@dataclass
class M2Pi05Hook:
    """Holds the captured tensor + remove() method."""
    captured: torch.Tensor | None = None
    handle: object = None  # torch.utils.hooks.RemovableHandle

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _resolve_text_model(policy) -> nn.Module:
    """policy → paligemma.model.language_model"""
    inner = policy.policy if hasattr(policy, "policy") else policy
    paligemma = inner.model.paligemma_with_expert.paligemma
    text_model = paligemma.model.language_model
    if len(text_model.layers) != PI05_LM_DEPTH:
        # Could be gemma_300m (1024-dim, 18 layers). Same depth, different width.
        # We don't error here — the depth check is informational.
        print(f"[m2_pi05_hook] note: text_model has {len(text_model.layers)} layers "
              f"(expected {PI05_LM_DEPTH}). Proceeding.", flush=True)
    return text_model


def attach_m2_pi05_hook(policy, capture_layer: int = DEFAULT_PI05_CAPTURE_LAYER) -> M2Pi05Hook:
    """Attach the M2 capture hook on `layers[capture_layer + 1].input_layernorm`."""
    text_model = _resolve_text_model(policy)
    if capture_layer < 0 or capture_layer >= len(text_model.layers) - 1:
        raise ValueError(
            f"capture_layer={capture_layer} out of range "
            f"(must be in [0, {len(text_model.layers) - 1}))"
        )

    target = text_model.layers[capture_layer + 1].input_layernorm
    holder = M2Pi05Hook(captured=None, handle=None)

    def pre_hook(module, args, kwargs):
        # Pi0.5 calls layernorm_forward(ln, hidden_states, cond) — both as
        # positional. The first positional arg is `hidden_states`.
        h = args[0] if args else kwargs.get("hidden_states")
        if h is None:
            return None
        # Keep the LIVE autograd tensor — M2's alignment loss must backprop
        # through this capture into the VLM. `.detach()` here silently makes
        # M2 a no-op (loss is still computed + logged, but trains nothing).
        # Matches the working SmolVLA hook (m2_smolvla_hook.py).
        holder.captured = h
        return None

    holder.handle = target.register_forward_pre_hook(pre_hook, with_kwargs=True)
    print(
        f"[m2_pi05_hook] attached pre_hook on "
        f"language_model.layers[{capture_layer + 1}].input_layernorm "
        f"(captures layer-{capture_layer} output)",
        flush=True,
    )
    return holder


def apply_m2_pi05_partial_freeze(
    policy, freeze_below: int = DEFAULT_PI05_CAPTURE_LAYER
) -> tuple[int, int]:
    """Freeze language_model.embed_tokens + layers[0..freeze_below). Leave
    layers[freeze_below..end] and final norm trainable.

    Returns (n_frozen_params, n_trainable_params) for sanity logging.
    """
    text_model = _resolve_text_model(policy)

    # Freeze early-layer params.
    if hasattr(text_model, "embed_tokens") and text_model.embed_tokens is not None:
        for p in text_model.embed_tokens.parameters():
            p.requires_grad = False
    for layer in text_model.layers[:freeze_below]:
        for p in layer.parameters():
            p.requires_grad = False

    # Confirm layers[freeze_below..end] + final norm trainable.
    for layer in text_model.layers[freeze_below:]:
        for p in layer.parameters():
            p.requires_grad = True
    if hasattr(text_model, "norm") and text_model.norm is not None:
        for p in text_model.norm.parameters():
            p.requires_grad = True

    n_frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(
        f"[m2_pi05_hook] partial freeze: layers[0..{freeze_below}) frozen, "
        f"layers[{freeze_below}..{len(text_model.layers)}] + norm trainable. "
        f"frozen={n_frozen/1e6:.1f}M trainable={n_trainable/1e6:.1f}M",
        flush=True,
    )
    return n_frozen, n_trainable
