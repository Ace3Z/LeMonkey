"""Runtime patches for Pi0.5 + transformers 4.55.0 compatibility.

Two patches:

1. `PaliGemmaWithExpertModel.embed_image` (modeling_pi05.py:442) calls
   `self.paligemma.model.get_image_features(image).pooler_output`. In
   transformers 4.55 that returns the pooled tensor directly.

2. `PiGemmaModel.forward` (pi_gemma.py:261) calls
   `create_causal_mask(..., inputs_embeds=...)` but transformers 4.55's
   create_causal_mask uses the kwarg name `input_embeds` (singular).
   Patched via a wrapper that translates the kwarg.

Idempotent — calling `apply()` twice is safe.
Import + call apply() BEFORE constructing any PI05Policy.
"""
from __future__ import annotations

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel
    # Patch 2: create_causal_mask kwarg rename.
    import lerobot.policies.pi_gemma as pi_gemma
    _orig_ccm = pi_gemma.create_causal_mask
    def _patched_ccm(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return _orig_ccm(*args, **kwargs)
    pi_gemma.create_causal_mask = _patched_ccm
    print("[pi05_inference_patch] patched create_causal_mask kwarg "
          "inputs_embeds → input_embeds", flush=True)

    def embed_image(self, image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        # Compat: old transformers returned an obj w/ .pooler_output;
        # new ones return the pooled tensor directly. Subscriptable
        # outputs may use [0].
        if hasattr(image_outputs, "pooler_output"):
            features = image_outputs.pooler_output
        elif isinstance(image_outputs, torch.Tensor):
            features = image_outputs
        else:
            features = image_outputs[0]
        features = features * self.paligemma.config.text_config.hidden_size ** 0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    PaliGemmaWithExpertModel.embed_image = embed_image
    print(
        "[pi05_inference_patch] patched PaliGemmaWithExpertModel.embed_image "
        "for transformers 4.55 compat",
        flush=True,
    )
    _APPLIED = True
