#!/usr/bin/env python3
"""Launcher: run `lerobot.scripts.lerobot_train` with M2 ArcFace alignment.

Strategy (chosen to need **zero changes** to upstream lerobot_train.py):

1. Read M2-specific paths + hyper-parameters from environment variables.
2. Monkey-patch `lerobot.policies.factory.make_policy` so that the policy
   returned to lerobot's `train()` is already wrapped with `M2WrappedPolicy`
   (hook attached + supervision builder ready + partial freeze applied).
3. Invoke lerobot's `train()` exactly as the upstream script does.

The wrapper's `forward(batch)` injects the M2 loss term transparently;
upstream `update_policy()` sees a normal `(loss, output_dict)` tuple.

Environment variables (all required unless noted):
    M2_FACE_LABELS_DIR     — eval_3/aug/stats/face_labels/
    M2_MANIFEST_PATH       — eval_3/aug/stats/celeb_embeddings.json
    M2_AUG_ROOT            — directory containing the 9,216 aug variant dirs
    M2_EPISODE_MAPPING     — eval_3/aug/stats/episode_mapping.json
    M2_LAMBDA              — float, BlindVLA recommends 0.2 (default 0.2)
    M2_CAPTURE_LAYER       — int, default 9 (depth-matched to BlindVLA)
    M2_LOG_EVERY           — int, default 100 (steps between m2 stat lines)
    M2_DISABLE             — set to "1" to skip wrapping (debug mode; trains
                              the inner policy untouched)

Everything else is passed through to `lerobot_train.train()` via the normal
draccus CLI parser. Usage:

    M2_FACE_LABELS_DIR=eval_3/aug/stats/face_labels \\
    M2_MANIFEST_PATH=eval_3/aug/stats/celeb_embeddings.json \\
    M2_AUG_ROOT=/data/eval3_track3_aug \\
    M2_EPISODE_MAPPING=eval_3/aug/stats/episode_mapping.json \\
    python eval_3/scripts/lerobot_train_with_m2.py \\
      --policy.type=smolvla \\
      --policy.pretrained_path=lerobot/smolvla_base \\
      --policy.vlm_model_name=HansOrtiz/smolvlm2_celeb_warm \\
      --policy.freeze_vision_encoder=True \\
      --policy.train_expert_only=False \\
      --policy.empty_cameras=1 \\
      --policy.optimizer_lr=5e-5 \\
      --policy.compile_model=False \\
      --dataset.repo_id=HBOrtiz/so101_eval3_track3_v3_baseline \\
      --batch_size=64 --steps=30000 \\
      --output_dir=outputs/smolvla_track_D_m2 \\
      --policy.push_to_hub=True \\
      --policy.repo_id=HBOrtiz/smolvla_eval3_track_D_m2_mahbod
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval_3/aug"))


def _env(key: str, default=None, required: bool = False, cast=str):
    v = os.environ.get(key)
    if v is None:
        if required:
            raise SystemExit(f"required env var {key!r} not set")
        return default
    return cast(v)


def _patch_get_safe_version():
    """Bypass lerobot's HF-revision-tag lookup when the dataset is local-only.

    `lerobot.datasets.utils.get_safe_version` queries HF for repo tags and
    rate-limits us at 1000 req/5min. Since we pre-downloaded the dataset
    (and tagged it once locally), there's no need to re-query. Stub it to
    return the input version as-is.
    """
    import lerobot.datasets.utils as ldu
    import lerobot.datasets.dataset_metadata as ldmm
    import lerobot.datasets.lerobot_dataset as lldataset

    def _stub_get_safe_version(repo_id, version):
        v = str(version)
        return v if v.startswith("v") else f"v{v}"

    ldu.get_safe_version = _stub_get_safe_version
    ldmm.get_safe_version = _stub_get_safe_version
    lldataset.get_safe_version = _stub_get_safe_version
    print("[m2 launcher] patched get_safe_version → returns version as-is "
          "(skips HF version-tag query)", flush=True)

    # huggingface_hub.hf_hub_download leaves behind a `.cache/huggingface/`
    # directory inside the dataset root; lerobot's
    # `has_legacy_hub_download_metadata()` detects that marker and forces a
    # re-download of metadata, which then triggers the bulk-video download
    # path (1000-req/5min xet rate limit). Nuke any such marker every launch.
    import shutil
    hf_home = os.environ.get("HF_LEROBOT_HOME") or str(Path.home() / ".cache/huggingface/lerobot")
    repo = os.environ.get("M2_DATASET_REPO_ID")
    if not repo:
        # try to sniff it from CLI argv
        for a in sys.argv:
            if a.startswith("--dataset.repo_id="):
                repo = a.split("=", 1)[1]
                break
    if repo:
        marker = Path(hf_home) / repo / ".cache"
        if marker.exists():
            shutil.rmtree(marker, ignore_errors=True)
            print(f"[m2 launcher] removed stale download marker {marker}", flush=True)


def _patch_smolvlm_vision_embeddings():
    """Move `boundaries` to the input device inside SmolVLMVisionEmbeddings.

    transformers==4.55.0 builds `boundaries` on CPU but the rest of the math
    runs on CUDA, which raises:
        RuntimeError: ... boundaries is on cpu, different from other tensors
        on cuda:0 (...torch.bucketize)
    Fix is a 1-character change upstream; we replace the method here so the
    site-packages tree stays untouched.
    """
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
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
        position_ids = position_ids.to(self.position_embedding.weight.device)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings

    SmolVLMVisionEmbeddings.forward = forward
    print("[m2 launcher] patched SmolVLMVisionEmbeddings.forward → "
          "boundaries on pixel_values.device", flush=True)


def _patch_make_policy():
    """Monkey-patch `lerobot.policies.factory.make_policy` to wrap with M2."""
    _patch_get_safe_version()
    _patch_smolvlm_vision_embeddings()
    if _env("M2_DISABLE", default="0") == "1":
        print("[m2 launcher] M2_DISABLE=1 — skipping M2 wrap (debug mode)", flush=True)
        return

    # Resolve paths
    face_labels_dir = Path(_env("M2_FACE_LABELS_DIR", required=True))
    manifest_path = Path(_env("M2_MANIFEST_PATH", required=True))
    aug_root = Path(_env("M2_AUG_ROOT", required=True))
    episode_mapping = Path(_env("M2_EPISODE_MAPPING", required=True))
    lam = _env("M2_LAMBDA", default=0.2, cast=float)
    capture_layer = _env("M2_CAPTURE_LAYER", default=9, cast=int)
    log_every = _env("M2_LOG_EVERY", default=100, cast=int)

    for p in [face_labels_dir, manifest_path, aug_root, episode_mapping]:
        if not p.exists():
            raise SystemExit(f"[m2 launcher] required path does not exist: {p}")

    from m2_alignment import FrozenProjector
    from m2_dataloader import M2SupervisionBuilder
    from m2_policy_wrapper import M2WrappedPolicy

    print(f"[m2 launcher] building M2SupervisionBuilder", flush=True)
    builder = M2SupervisionBuilder(
        face_labels_dir=face_labels_dir,
        manifest_path=manifest_path,
        aug_root=aug_root,
        episode_mapping_path=episode_mapping,
    )
    n_face_labels = len(builder._face_labels_cache)
    n_celebs = len(builder.centroid_lookup)
    n_episodes = len(builder.episode_mapping) if builder.episode_mapping else 0
    print(f"[m2 launcher] face_labels={n_face_labels} celebs={n_celebs} "
          f"episodes={n_episodes} lam={lam} capture_layer={capture_layer}",
          flush=True)

    projector = FrozenProjector()

    # Now do the actual monkey-patch on lerobot.policies.factory.make_policy.
    factory = importlib.import_module("lerobot.policies.factory")
    original_make_policy = factory.make_policy

    def make_policy_with_m2(*args, **kwargs):
        policy = original_make_policy(*args, **kwargs)
        wrapped = M2WrappedPolicy(
            policy=policy,
            builder=builder,
            projector=projector,
            capture_layer=capture_layer,
            lam=lam,
            log_every=log_every,
        )
        n_frozen, n_trainable = wrapped.apply_partial_freeze()
        print(f"[m2 launcher] wrapped policy: "
              f"{n_frozen/1e6:.1f}M frozen, {n_trainable/1e6:.1f}M trainable",
              flush=True)
        return wrapped

    factory.make_policy = make_policy_with_m2
    # Also patch any module that imported make_policy by reference.
    train_mod = importlib.import_module("lerobot.scripts.lerobot_train")
    if hasattr(train_mod, "make_policy"):
        train_mod.make_policy = make_policy_with_m2


def main() -> int:
    _patch_make_policy()
    # Now invoke lerobot's train via its CLI entrypoint, which will pick up
    # all the --policy.* and --dataset.* flags from sys.argv.
    from lerobot.scripts.lerobot_train import main as lerobot_main
    return lerobot_main() or 0


if __name__ == "__main__":
    sys.exit(main())
