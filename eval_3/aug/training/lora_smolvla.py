"""Minimal LoRA (Hu et al. 2021 — arXiv:2106.09685) for SmolVLA's VLM attention.

Why this exists
---------------
KLAL (`klal_smolvla_action.py`) supervises the VLM's name-token -> image-patch
attention. For that loss to change anything, the VLM's q/k projections must be
trainable. SmolVLA trained with `train_expert_only=True` freezes the *whole*
VLM, so KLAL would back-prop into frozen weights and learn nothing.

LoRA reopens exactly the needed capacity: a low-rank trainable delta on the
attention projections, base weights frozen. The frozen base preserves the
SmolVLM2 prior (anti-forgetting) -- strictly less destructive than fully
fine-tuning whole layers (the M2 partial-freeze path in `m2_smolvla_hook.py`).

Canonical LoRA update:  h = W0 x + (alpha / r) * B (A x)
with A ~ kaiming-uniform and B = 0, so the adapter is a no-op at init.

Save path
---------
A LoRA-injected module tree has `q_proj.base.weight` / `q_proj.lora_A.weight`
... keys, which a vanilla `SmolVLAPolicy.from_pretrained` cannot load.
`swap_to_merged()` builds plain `nn.Linear`s with `W0 + delta` folded in and
swaps them into the tree for the duration of a checkpoint save;
`swap_to_lora()` puts the LoRA modules back. The base weights are never
mutated, so the round-trip is exact and training resumes bit-identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    """Hyperparameters for LoRA injection into SmolVLA's VLM attention.

    Attributes:
        r: LoRA rank (low-rank-decomposition inner dim). 16 matches the
            published SmolVLA-450M figure.
        alpha: LoRA alpha (scale factor); ``alpha / r`` is multiplied into
            the update path. ``alpha == 2 * r`` is the PEFT-library
            convention (Hu et al. 2021 treat alpha as task-specific and
            do not prescribe a default).
        dropout: Dropout on the LoRA branch. Kept at 0.0 because SmolVLA
            runs the VLM in ``.eval()`` under ``train_expert_only=True``,
            so an active dropout would be a no-op anyway.
        layers: Indices of VLM transformer layers to wrap. When KLAL is
            enabled, the caller is responsible for ensuring this set is a
            superset of the KLAL-supervised layers; otherwise KLAL would
            back-prop into a frozen layer and learn nothing. (Note: the
            default tuple here and ``KLALConfig.capture_layers`` do not
            satisfy this on their own - reconcile both at the call site.)
        target_modules: Attribute names of the linear projections to wrap
            per layer (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``).
    """
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    layers: tuple[int, ...] = (9, 10, 11, 12, 13, 14, 15)
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank delta.

    Forward: ``y = base(x) + (alpha / r) * lora_B(lora_A(dropout(x)))``.

    The wrapped base's ``.weight`` and ``.bias`` are re-exposed as properties
    because SmolVLA's custom forward reads ``q_proj.weight.dtype`` directly
    (``smolvlm_with_expert.py:222``); without these a LoRA-injected ``q_proj``
    would raise ``AttributeError`` on the first forward.

    Args:
        base: The frozen ``nn.Linear`` to adapt. Its parameters get
            ``requires_grad=False`` automatically.
        r: LoRA rank (positive int).
        alpha: LoRA alpha; ``alpha / r`` scales the update.
        dropout: Dropout probability on the LoRA branch input.

    Raises:
        TypeError: if ``base`` is not an ``nn.Linear``.
        ValueError: if ``r <= 0``.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        if r <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.scaling = alpha / r
        w = base.weight
        self.lora_A = nn.Linear(base.in_features, r, bias=False,
                                device=w.device, dtype=w.dtype)
        self.lora_B = nn.Linear(r, base.out_features, bias=False,
                                device=w.device, dtype=w.dtype)
        # Canonical LoRA init: A ~ kaiming-uniform, B = 0 -> adapter is a
        # no-op at step 0 (the policy starts identical to the base VLM).
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # SmolVLA's custom forward reads these off the projection module.
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = self.lora_B(self.lora_A(self.dropout(x)))
        return base_out + self.scaling * delta

    def merged_linear(self) -> nn.Linear:
        """Return a plain `nn.Linear` with the LoRA delta folded into W0.

        The base weights are NOT mutated -- the merge is computed into a fresh
        module, so the exact round-trip swap_to_merged/swap_to_lora is lossless.
        """
        w = self.base.weight
        delta = (self.lora_B.weight.float() @ self.lora_A.weight.float()) * self.scaling
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None,
                           device=w.device, dtype=w.dtype)
        merged.weight.data.copy_(w.data + delta.to(w.dtype))
        if self.base.bias is not None:
            merged.bias.data.copy_(self.base.bias.data)
        return merged


def inject_lora(text_model, cfg: LoRAConfig) -> list[tuple]:
    """Replace the target `nn.Linear` submodules in `text_model.layers[cfg.layers]`
    with `LoRALinear`. Returns a registry list of `(parent_module, attr, lora)`.

    `parent_module` is the `self_attn` module; `attr` is e.g. "q_proj".
    """
    registry: list[tuple] = []
    n_layers = len(text_model.layers)
    for idx in cfg.layers:
        if idx >= n_layers:
            raise ValueError(
                f"LoRA layer index {idx} >= text_model has {n_layers} layers"
            )
        attn = text_model.layers[idx].self_attn
        for name in cfg.target_modules:
            base = getattr(attn, name, None)
            if base is None:
                print(f"[WARN] LoRA inject: layer {idx} self_attn has no "
                      f"'{name}', expected=nn.Linear, got=None, "
                      f"fallback=skip this module", flush=True)
                continue
            if isinstance(base, LoRALinear):
                continue  # idempotent — already injected
            lora = LoRALinear(base, cfg.r, cfg.alpha, cfg.dropout)
            setattr(attn, name, lora)
            registry.append((attn, name, lora))
    return registry


def swap_to_merged(registry: list[tuple]) -> None:
    """Swap every LoRALinear for a plain merged `nn.Linear` (for a clean save)."""
    for parent, attr, lora in registry:
        setattr(parent, attr, lora.merged_linear())


def swap_to_lora(registry: list[tuple]) -> None:
    """Restore the LoRALinear modules after a save (training resumes unchanged)."""
    for parent, attr, lora in registry:
        setattr(parent, attr, lora)


def count_lora_params(registry: list[tuple]) -> int:
    seen: set[int] = set()
    total = 0
    for _parent, _attr, lora in registry:
        for p in (lora.lora_A.weight, lora.lora_B.weight):
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total
