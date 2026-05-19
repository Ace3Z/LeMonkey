"""Runtime patches for Pi0.5 + transformers 4.55.0 compatibility.

Currently one patch:

* `PaliGemmaWithExpertModel.embed_image` (modeling_pi05.py:442) calls
  `self.paligemma.model.get_image_features(image).pooler_output`.
  In transformers 4.55 that method returns the pooled tensor directly
  instead of an object with `pooler_output`, raising
  `AttributeError: 'Tensor' object has no attribute 'pooler_output'`.

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
