"""Probe: confirm the M2 pre-hook captures the right tensor.

Per CLAUDE.md §9 reviewer audit, before any 6 h Brev run we need to verify:

1. **Hook fires.** Registering a pre_hook on `text_model.layers[10].input_layernorm`
   produces a captured tensor when `SmolVLMWithExpertModel.forward` runs.
2. **Right shape.** Captured tensor is (B, prefix_len, 960).
3. **Right semantic position.** The captured tensor is the OUTPUT of layer 9 —
   distinct from layer 0's input (something happened in between) AND
   distinct from layer 15's output (more processing happens after).
4. **VLM-stream only.** The hook fires for the VLM stream (i=0), not for the
   action expert stream (i=1) which uses `lm_expert.layers[10]`, a different
   module instance.
5. **Gradient flow.** A loss computed on the captured tensor produces non-zero
   gradients on parameters that should be trainable (e.g., layer 9 weights),
   and zero gradients on parameters that should be frozen (layer 7 weights
   after applying `apply_m2_partial_freeze(..., freeze_below=9)`).

Loads the real SmolVLM2-500M backbone (~500 MB first time). CPU-only. Synthesizes
a dummy SmolVLA inputs_embeds list and runs one forward pass through the custom
loop. No real images, no real actions — we only care about the hook semantics
and the partial-freeze gradient behavior.

Run from project root:

    python eval_3/aug/dbg/dbg_m2_hook_probe.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import torch

# Make eval_3/aug importable
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "eval_3/aug"))


# Import `SmolVLMWithExpertModel` directly via importlib to bypass the
# lerobot.policies/__init__.py side-effects that pull in pandas/pyarrow/etc.
# The file itself only imports torch + transformers.
def _import_smolvlm_with_expert():
    import importlib.util

    path = REPO_ROOT / "third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py"
    spec = importlib.util.spec_from_file_location("_smolvlm_with_expert_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SmolVLMWithExpertModel


SmolVLMWithExpertModel = _import_smolvlm_with_expert()  # noqa: E402

from m2_smolvla_hook import (  # noqa: E402
    DEFAULT_CAPTURE_LAYER,
    apply_m2_partial_freeze,
    attach_m2_hook,
)


def _build_smolvla_model() -> SmolVLMWithExpertModel:
    """Minimal SmolVLMWithExpertModel matching SmolVLA's defaults
    (num_vlm_layers=16, expert_width_multiplier=0.75, self_attn_every_n_layers=2).

    The constructor at smolvlm_with_expert.py:87 calls
    AutoProcessor.from_pretrained, which in newer transformers pulls in
    SmolVLM's video processor (needs torchvision). Our probe doesn't need
    any processor — we synthesise inputs_embeds directly — so we stub it.
    """
    from transformers import AutoProcessor

    original = AutoProcessor.from_pretrained

    class _StubProcessor:
        class _StubTokenizer:
            fake_image_token_id = 0
            global_image_token_id = 0
        tokenizer = _StubTokenizer()

    AutoProcessor.from_pretrained = staticmethod(lambda *a, **kw: _StubProcessor())
    try:
        return SmolVLMWithExpertModel(
            model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            load_vlm_weights=True,
            train_expert_only=False,            # we need params trainable for grad-flow check
            freeze_vision_encoder=True,
            attention_mode="cross_attn",
            num_expert_layers=-1,
            num_vlm_layers=16,
            self_attn_every_n_layers=2,
            expert_width_multiplier=0.75,
            device="cpu",
        )
    finally:
        AutoProcessor.from_pretrained = original


def _synth_inputs_embeds(model: SmolVLMWithExpertModel, B: int = 2, prefix_len: int = 100):
    """Build a (inputs_embeds, attention_mask, position_ids) trio that exercises
    only the VLM stream.

    SmolVLA's forward expects `inputs_embeds` to be a list of two tensors
    `[vlm_prefix, expert_suffix]`. To keep this probe focused on the VLM stream
    (where our hook lives) and sidestep cross-attn dimension matching for the
    expert, we pass `expert_suffix=None`: the custom forward's
    `if hidden_states is None or layer is None: continue` branch skips the
    expert entirely. Cross-attn is also skipped because `fill_kv_cache=True`
    forces every layer through `forward_attn_layer`.
    """
    H_vlm = model.config.text_config.hidden_size
    vlm_prefix = torch.randn(B, prefix_len, H_vlm, requires_grad=False)
    attn_mask = torch.ones(B, prefix_len, prefix_len, dtype=torch.bool)
    position_ids = torch.arange(prefix_len)[None, :].expand(B, -1)
    return [vlm_prefix, None], attn_mask, position_ids


def test_1_hook_fires_and_shape(model: SmolVLMWithExpertModel) -> bool:
    print("\n[1/5] Hook fires + captured shape")
    hook = attach_m2_hook(model, capture_layer=DEFAULT_CAPTURE_LAYER, config=None)
    inputs_embeds, attn_mask, position_ids = _synth_inputs_embeds(model)
    try:
        with torch.no_grad():
            outputs_embeds, _ = model.forward(
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                fill_kv_cache=True,
            )
        if hook.captured is None:
            print("  FAIL: hook did not capture anything")
            return False
        cap = hook.captured
        expected = (inputs_embeds[0].shape[0], inputs_embeds[0].shape[1], model.config.text_config.hidden_size)
        ok_shape = tuple(cap.shape) == expected
        print(f"  captured shape: {tuple(cap.shape)}  expected: {expected}  {'OK' if ok_shape else 'FAIL'}")
        return ok_shape
    finally:
        hook.remove()


def test_2_distinct_from_layer0_input(model: SmolVLMWithExpertModel) -> bool:
    print("\n[2/5] Captured ≠ layer 0 input (something happened in 9 layers)")
    hook = attach_m2_hook(model, capture_layer=DEFAULT_CAPTURE_LAYER, config=None)
    inputs_embeds, attn_mask, position_ids = _synth_inputs_embeds(model)
    try:
        layer0_input = inputs_embeds[0].clone()
        with torch.no_grad():
            model.forward(
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                fill_kv_cache=True,
            )
        cap = hook.captured
        diff = (cap - layer0_input).abs().mean().item()
        print(f"  mean |captured − layer0_input| = {diff:.4f}  (expect > 0.01)")
        return diff > 0.01
    finally:
        hook.remove()


def test_3_distinct_from_final_output(model: SmolVLMWithExpertModel) -> bool:
    print("\n[3/5] Captured ≠ final output (more processing happens after layer 9)")
    hook = attach_m2_hook(model, capture_layer=DEFAULT_CAPTURE_LAYER, config=None)
    inputs_embeds, attn_mask, position_ids = _synth_inputs_embeds(model)
    try:
        with torch.no_grad():
            outputs_embeds, _ = model.forward(
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                fill_kv_cache=True,
            )
        cap = hook.captured
        final = outputs_embeds[0]
        diff = (cap - final).abs().mean().item()
        print(f"  mean |captured − final_output| = {diff:.4f}  (expect > 0.01)")
        return diff > 0.01
    finally:
        hook.remove()


def test_4_vlm_stream_only(model: SmolVLMWithExpertModel) -> bool:
    print("\n[4/5] Hook is on VLM stream's text_model.layers[10], NOT lm_expert.layers[10]")
    text_layers = model.vlm.model.text_model.layers
    expert_layers = model.lm_expert.layers
    text_ln = text_layers[DEFAULT_CAPTURE_LAYER + 1].input_layernorm
    expert_ln = expert_layers[DEFAULT_CAPTURE_LAYER + 1].input_layernorm
    different = text_ln is not expert_ln
    print(f"  text_model.layers[10].input_layernorm  is  lm_expert.layers[10].input_layernorm = {not different}")
    print(f"  (expect: False — they are different module instances)")
    return different


def test_5_gradient_flow_with_partial_freeze(model: SmolVLMWithExpertModel) -> bool:
    print("\n[5/5] Partial-freeze: layer 7 frozen, layer 9 trainable, grads land in layer 9")
    # Apply our partial freeze (layers 0..8 frozen, layers 9..15 trainable per upstream)
    n_frozen, n_trainable = apply_m2_partial_freeze(model, freeze_below=DEFAULT_CAPTURE_LAYER)
    print(f"  apply_m2_partial_freeze(freeze_below=9): "
          f"n_frozen={n_frozen:,}, n_trainable={n_trainable:,}")

    text_layers = model.vlm.model.text_model.layers
    # NOTE: LeRobot's else-branch tries to re-freeze layer 15 by string-matching
    # "text_model.model.layers.15." in param names. Real SmolVLM2 param names
    # are "text_model.layers.15..." (without the extra .model.), so the
    # upstream re-freeze does NOT fire for SmolVLM2 — all layers ≥ capture_layer
    # remain trainable in practice. Confirmed by reading
    # smolvlm_with_expert.py:155-164.
    for i, layer in enumerate(text_layers):
        any_grad = any(p.requires_grad for p in layer.parameters())
        expected_grad = (i >= DEFAULT_CAPTURE_LAYER)
        if any_grad != expected_grad:
            print(f"  FAIL: layer {i}: any_grad={any_grad}, expected={expected_grad}")
            return False
    print(f"  freeze pattern OK: layers 0-{DEFAULT_CAPTURE_LAYER-1} frozen, "
          f"{DEFAULT_CAPTURE_LAYER}-15 trainable")

    # Now run forward + dummy loss + backward, see where gradients land
    hook = attach_m2_hook(model, capture_layer=DEFAULT_CAPTURE_LAYER, config=None)
    inputs_embeds, attn_mask, position_ids = _synth_inputs_embeds(model)
    # Make the VLM prefix require grad so backprop can reach inputs (not necessary
    # for testing internal grad on layer-9 weights, but good defensive practice).
    inputs_embeds[0] = inputs_embeds[0].detach().clone().requires_grad_(True)
    try:
        outputs_embeds, _ = model.forward(
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            fill_kv_cache=True,
        )
        cap = hook.captured
        # A dummy "M2-like" loss: pull captured toward zero
        dummy_loss = cap.float().pow(2).mean()
        dummy_loss.backward()
        # Check grads on layer 9 (trainable, captured-layer) and layer 7 (frozen)
        # We expect layer 9 to receive grads since cap depends on layers 0..9's output
        # — but layers 0..8 are frozen, so only layer 9 gets parameter grads.
        # Actually: backprop from cap goes through layers 0..9. Of those, only layer 9
        # has requires_grad params, so only layer 9 has parameter grads.
        layer9_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in text_layers[DEFAULT_CAPTURE_LAYER].parameters() if p.requires_grad)
        layer7_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in text_layers[7].parameters())
        layer12_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in text_layers[12].parameters() if p.requires_grad)
        print(f"  layer 9  grads non-zero (trainable, expected): {layer9_has_grad}")
        print(f"  layer 7  grads non-zero (frozen, expected None): {layer7_has_grad}")
        print(f"  layer 12 grads non-zero (after capture layer, no grad path): {layer12_has_grad}")
        # Layer 12 should NOT have grads because the loss is on layer 9's output,
        # not on the final output. cap was captured BEFORE layer 10 runs, so the
        # gradient path from `cap` to layer 12's weights doesn't exist.
        ok = layer9_has_grad and not layer7_has_grad and not layer12_has_grad
        return ok
    finally:
        hook.remove()


def main() -> int:
    print("=" * 70)
    print("M2 hook probe — verifies hook captures the right tensor on real SmolVLM2")
    print("=" * 70)
    print("\nLoading SmolVLM2-500M backbone (first run downloads ~500 MB)...")
    model = _build_smolvla_model()
    print(f"  num_vlm_layers = {len(model.vlm.model.text_model.layers)}")
    print(f"  text_config.hidden_size = {model.config.text_config.hidden_size}")

    tests = [
        test_1_hook_fires_and_shape,
        test_2_distinct_from_layer0_input,
        test_3_distinct_from_final_output,
        test_4_vlm_stream_only,
        test_5_gradient_flow_with_partial_freeze,
    ]
    results = []
    for t in tests:
        try:
            results.append(t(model))
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {e}")
            results.append(False)

    print("\n" + "=" * 70)
    if all(results):
        print(f"ALL {len(results)}/{len(results)} CHECKS PASSED — hook captures what we expect")
        return 0
    print(f"{sum(results)}/{len(results)} CHECKS PASSED — failures: "
          f"{[i + 1 for i, r in enumerate(results) if not r]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
