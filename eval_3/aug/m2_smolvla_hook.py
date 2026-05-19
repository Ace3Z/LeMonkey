"""SmolVLA integration for the M2 ArcFace alignment loss.

Three responsibilities:

1) Attach a forward-pre-hook on `text_model.layers[N+1].input_layernorm`
   to capture the OUTPUT of layer N (= INPUT of layer N+1) during the
   custom forward in `SmolVLMWithExpertModel.forward`.
2) Apply the partial-freeze required for M2 supervision to actually move
   parameters: layers 0..N-1 frozen, layers N..15 trainable. Keeps
   LeRobot's upstream re-freeze of layer 15 (DDP-safe; harmless on
   single-GPU).
3) Surface the captured tensor for the training-loop integration.

Choice of N=9 is depth-matched to BlindVLA Table 12's optimum (layer 16 of
28 = 57 % depth in OpenVLA; layer 9 of 16 = 56 % in our truncated stack).
See `docs/experiments/2026-05-19_m2_review_findings.md` for the full
3-reviewer audit that led to this design.

Hook semantics
--------------
SmolVLA's custom forward at `smolvlm_with_expert.py:403-498` does NOT call
`text_model.layers[N].__call__()`. It calls sub-modules manually:
`layer.input_layernorm`, `layer.self_attn.{q,k,v,o}_proj`,
`layer.post_attention_layernorm`, `layer.mlp`. So a forward_hook on
`text_model.layers[N]` would never fire — but a hook on the SUB-MODULE
DOES fire because that sub-module is invoked normally.

Specifically, at the start of layer-iteration N+1, the custom loop runs:
    hidden_states = layer.input_layernorm(inputs_embeds[i])
for `i ∈ {0, 1}` (VLM stream, expert stream). For `i=0`, `layer` is
`text_model.layers[N+1]` and `inputs_embeds[0]` is exactly the OUTPUT of
layer N from the previous iteration. So a `register_forward_pre_hook` on
`text_model.layers[N+1].input_layernorm` captures that input — which is
layer N's output.

For `i=1`, `layer` is `lm_expert.layers[N+1]` — a DIFFERENT module
instance, so our hook on the text-model layer does NOT fire there.

During training (`fill_kv_cache=True`), every layer goes through
`forward_attn_layer` regardless of `self_attn_every_n_layers`, so our hook
fires exactly once per training step at the chosen layer.

What this module does NOT do
----------------------------
- Compile-time wrapping. SmolVLA supports `torch.compile`; hooks + compile
  is fragile. We `assert config.compile_model is False` at attach time
  rather than papering over it.
- DDP coordination. `train_expert_only=False` keeps LeRobot's last-layer
  re-freeze (layer 15) which avoids DDP `find_unused_parameters` errors.
  On single-GPU this is a no-op cost.
"""
from __future__ import annotations

from dataclasses import dataclass


# Layer choice: 9 = 56 % of 16-layer truncated SmolLM2, matches BlindVLA Table 12
# optimum (layer 16 of 28 = 57 % in OpenVLA's LLaMA-7B).
DEFAULT_CAPTURE_LAYER = 9
# Hook is attached on the NEXT layer's input_layernorm because that input
# equals the chosen layer's output (see module docstring).
DEFAULT_HOOK_LAYER = DEFAULT_CAPTURE_LAYER + 1


@dataclass
class M2Hook:
    """Holds the registered hook handle and the most recently captured tensor.

    Usage:
        hook = attach_m2_hook(policy)
        loss, _ = policy.forward(batch)            # hook fires here
        cap = hook.captured                        # (B, prefix_len, 960)
        m2_result = m2_align_loss(cap, ...)
        total = loss + 0.2 * m2_result.loss
        total.backward()
    """
    capture_layer: int
    hook_layer: int
    handle: object                                 # torch.utils.hooks.RemovableHandle
    captured: object = None                        # set by the pre-hook each forward

    def remove(self):
        self.handle.remove()


def _resolve_smolvla(policy_or_model):
    """Walk into a SmolVLA policy / VLAFlowMatching / SmolVLMWithExpertModel
    and return the inner `vlm.model.text_model` and the live config object."""
    # Allow callers to pass either the top-level SmolVLAPolicy, the
    # VLAFlowMatching, or the SmolVLMWithExpertModel directly.
    obj = policy_or_model
    for attr in ("model", "vlm_with_expert"):
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
    # `obj` should now be SmolVLMWithExpertModel
    if not hasattr(obj, "vlm") or not hasattr(obj, "config"):
        raise TypeError(
            f"could not locate SmolVLMWithExpertModel inside {type(policy_or_model).__name__}; "
            "pass the policy, the VLAFlowMatching, or the SmolVLMWithExpertModel."
        )
    text_model = obj.vlm.model.text_model
    return text_model, obj.config


def attach_m2_hook(
    policy_or_model,
    capture_layer: int = DEFAULT_CAPTURE_LAYER,
    config=None,
) -> M2Hook:
    """Register a forward_pre_hook to capture layer N's output.

    Run-time assertions (raise if violated):
    - `add_image_special_tokens=False` — otherwise camera1 patches don't
      start at prefix offset 0.
    - `compile_model=False` — torch.compile + Python-attribute writes from
      hooks cause graph breaks and silent compile-disable.
    - `present_img_keys[0] == 'observation.images.camera1'` is NOT checked
      here because we don't have a batch yet; the dataloader-side helper
      verifies it per-batch.
    - `num_vlm_layers > capture_layer + 1` — otherwise the hook layer
      doesn't exist after truncation.
    """
    text_model, smolvla_config = _resolve_smolvla(policy_or_model)

    # Find the actual SmolVLAConfig if the user passed it explicitly
    if config is None:
        # Walk up to find the SmolVLAConfig — it lives on VLAFlowMatching
        cfg = getattr(policy_or_model, "config", None)
        if cfg is None and hasattr(policy_or_model, "model"):
            cfg = getattr(policy_or_model.model, "config", None)
        config = cfg

    if config is not None:
        if getattr(config, "add_image_special_tokens", False):
            raise AssertionError(
                "M2 hook requires config.add_image_special_tokens=False (the SmolVLA default). "
                "Got True. If you really want this, expand the camera1_offset accordingly."
            )
        if getattr(config, "compile_model", False):
            raise AssertionError(
                "M2 hook is incompatible with config.compile_model=True (graph breaks). "
                "Set compile_model=False for any M2-enabled training run."
            )
        if getattr(config, "num_vlm_layers", 16) <= capture_layer + 1:
            raise AssertionError(
                f"capture_layer={capture_layer} requires num_vlm_layers >= {capture_layer + 2}; "
                f"got {config.num_vlm_layers}."
            )

    hook_layer = capture_layer + 1
    if len(text_model.layers) <= hook_layer:
        raise AssertionError(
            f"text_model.layers has only {len(text_model.layers)} layers; "
            f"need at least {hook_layer + 1} for hook_layer={hook_layer}."
        )

    target = text_model.layers[hook_layer].input_layernorm
    holder = M2Hook(capture_layer=capture_layer, hook_layer=hook_layer, handle=None)

    def _pre_hook(_module, inputs):
        # `inputs` is a single-element tuple containing the LayerNorm's input,
        # which is the OUTPUT of the previous transformer layer (= our capture).
        # SmolVLA's custom forward calls this LN with `inputs_embeds[0]` for
        # the VLM stream and `inputs_embeds[1]` for the action expert; only the
        # VLM stream's `text_model.layers[hook_layer].input_layernorm` hits us
        # (the expert uses `lm_expert.layers[hook_layer].input_layernorm`,
        # a different module).
        holder.captured = inputs[0]
        return None  # don't modify the input

    holder.handle = target.register_forward_pre_hook(_pre_hook)
    return holder


def apply_m2_partial_freeze(
    policy_or_model,
    freeze_below: int = DEFAULT_CAPTURE_LAYER,
):
    """Re-freeze layers [0, freeze_below) of `text_model.layers`.

    Call AFTER `SmolVLMWithExpertModel.set_requires_grad()` has run with
    `train_expert_only=False`. That call leaves all 16 VLM layers trainable
    (except layer 15, re-frozen for DDP).  We add a second freeze pass
    pinning the early layers (0..freeze_below-1) to preserve Hans's
    warm-VLM celeb prior in the early-layer feature geometry while letting
    M2 supervision shape mid-late layers (freeze_below..N-1, plus the
    upstream-preserved layer 15 frozen).

    Returns (n_frozen, n_trainable) counts of VLM-side parameters for sanity.
    """
    text_model, _ = _resolve_smolvla(policy_or_model)
    n_frozen = 0
    n_trainable = 0
    for layer_idx, layer in enumerate(text_model.layers):
        freeze_this = layer_idx < freeze_below
        for p in layer.parameters():
            if freeze_this:
                p.requires_grad = False
            # Do NOT re-enable grads on layer 15; LeRobot's set_requires_grad
            # froze it deliberately for DDP. Stays frozen.
            if p.requires_grad:
                n_trainable += p.numel()
            else:
                n_frozen += p.numel()
    return n_frozen, n_trainable
