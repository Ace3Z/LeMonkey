"""Runtime patches needed for inference with the SmolVLA M2 checkpoint.

Currently one patch:

* `SmolVLMVisionEmbeddings.forward` in transformers==4.55.0 builds the
  `boundaries` tensor on CPU but the surrounding math runs on the
  `pixel_values` device. The resulting `torch.bucketize` call dies with
  "boundaries is on cpu, different from other tensors on cuda:0".
  We replace the method with a version that puts both `boundaries` and
  `position_ids` on `pixel_values.device`.

This is the SAME patch the training launcher already applies; lifted here
so any inference entry point (verify, rollout, deployment) can import and
apply it without depending on the launcher module.

Idempotent — calling `apply()` twice is safe.

Usage:
    from eval_3.aug.smolvlm_inference_patch import apply
    apply()
    # ...then load and run the policy as usual.
"""
from __future__ import annotations

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return

    import torch
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionEmbeddings

    def forward(self, pixel_values, patch_attention_mask):
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        max_nb_patches_h, max_nb_patches_w = (
            max_im_h // self.patch_size,
            max_im_w // self.patch_size,
        )
        boundaries = torch.arange(
            1 / self.num_patches_per_side,
            1.0,
            1 / self.num_patches_per_side,
            device=pixel_values.device,
        )
        position_ids = torch.full(
            (batch_size, max_nb_patches_h * max_nb_patches_w),
            fill_value=0,
            device=pixel_values.device,
        )
        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()
            h_indices = torch.arange(nb_patches_h, device=pixel_values.device, dtype=pixel_values.dtype)
            w_indices = torch.arange(nb_patches_w, device=pixel_values.device, dtype=pixel_values.dtype)
            fractional_coords_h = h_indices / nb_patches_h * (1 - 1e-6)
            fractional_coords_w = w_indices / nb_patches_w * (1 - 1e-6)
            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids
        position_ids = position_ids.to(self.position_embedding.weight.device)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings

    SmolVLMVisionEmbeddings.forward = forward
    print(
        "[smolvlm_inference_patch] patched SmolVLMVisionEmbeddings.forward → "
        "boundaries + position_ids on pixel_values.device",
        flush=True,
    )
    _APPLIED = True
