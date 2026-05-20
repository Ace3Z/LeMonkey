#!/usr/bin/env python3
"""SmolVLA + VL co-training, RT-2 §3.2 style.

Mixed-batch trainer: every (vl_ratio+1)-th step is a VL batch (CE loss on
SmolVLM2's LM head), the rest are robot batches (SmolVLA flow-matching
action loss). Both losses share one optimizer; both gradients flow into
the SmolVLM2 body.

This is the SmolVLA-450M sibling of eval_3/scripts/track_2/lerobot_train_with_vl_cotrain.py
(which is Pi0.5-3B). Unlike that one, this is NOT a scaffold — the training loop
is integrated end-to-end so it runs as-is on a single GPU AWS node.

Per RT-2 §3.2: keeping web/VQA data alongside robot data during fine-tuning
prevents catastrophic forgetting of the VLM's web knowledge (celeb-name
recognition, in our case). Diagnosed need: sequential VLM-then-action
fine-tuning produces a positional-shortcut policy that ignores the prompt
celeb name.

Per CLAUDE.md §5: every fallback emits [WARN] with context.
Per CLAUDE.md §7: written but UNSMOKED locally — must be smoke-tested on
the user's AWS node before being trusted for a 24h run.

Usage
=====

    python eval_3/scripts/smolvla_cotrain/cotrain.py \\
        --robot_dataset=HBOrtiz/so101_eval3_track3_v3_baseline \\
        --vl_manifest=HBOrtiz/eval3_objectvla_vl_pairs \\
        --vl_ratio=10 \\
        --output_dir=outputs/smolvla_cotrain_10to1 \\
        --steps=30000 \\
        --batch_size=32 \\
        --vl_batch_size=8 \\
        --lr=5e-5 \\
        --push_to_hub_repo=HBOrtiz/smolvla_eval3_cotrain_10to1

For a smoke test, drop --steps=200 and --batch_size=4.

Optional KLAL + LoRA (celeb-routing enhancement)
------------------------------------------------
`--enable_lora` adapts the VLM via low-rank adapters (base frozen) instead of
full fine-tuning. `--enable_klal` adds the KLAL attention-supervision loss on
robot batches — it teaches the policy's name-token to attend to the prompted
celeb's face patch, against a bbox-derived target. KLAL needs the M2 toolkit's
face-bbox data (bboxes only — the M2 ArcFace loss is NOT added):

    ... --enable_lora --enable_klal \\
        --face_labels_dir=eval_3/aug/stats/face_labels \\
        --celeb_manifest=eval_3/aug/stats/celeb_embeddings.json \\
        --aug_root=/data/eval3_track3_aug \\
        --episode_mapping=eval_3/aug/stats/episode_mapping.json

Checkpoints are saved with the LoRA delta merged into the base weights, so
they load as a vanilla SmolVLAPolicy (eval-day recipe unchanged).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# The m2_* helpers (LoRA, KLAL attention supervision) live in eval_3/aug.
# Add it to the path so the bare `import m2_*` statements resolve regardless
# of the CWD the script is launched from.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "eval_3/aug"))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    p.add_argument("--robot_dataset", required=True,
                   help="LeRobotDataset HF repo id (e.g. HBOrtiz/so101_eval3_track3_v3_baseline)")
    p.add_argument("--vl_manifest", required=True,
                   help="VL pairs HF repo id (e.g. HBOrtiz/eval3_objectvla_vl_pairs) "
                        "OR local parquet path")
    p.add_argument("--vl_image_root", default=None,
                   help="Override path to pre-extracted VL images dir. "
                        "If None, snapshot_download the whole VL repo (~1 GB).")
    # Model
    p.add_argument("--pretrained_path", default="lerobot/smolvla_base",
                   help="SmolVLA starting checkpoint. Use HansOrtiz/smolvlm2_celeb_warm "
                        "for warm-VLM, or lerobot/smolvla_base for cold.")
    p.add_argument("--vlm_model_name", default=None,
                   help="Optional override for the inner VLM checkpoint. "
                        "If set, replaces config.vlm_model_name before loading.")
    # Training
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Robot batch size (per step on robot-batch steps)")
    p.add_argument("--vl_batch_size", type=int, default=8,
                   help="VL batch size (per step on VL-batch steps). Usually smaller.")
    p.add_argument("--vl_ratio", type=int, default=10,
                   help="Number of robot batches per 1 VL batch. RT-2 spec is "
                        "'robot >> web'; ObjectVLA used 10. Test 5/10 in parallel.")
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Optimizer LR. SmolVLA default; RT-2 says 'use VLM-paper hparams'.")
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    # Output
    p.add_argument("--output_dir", required=True)
    p.add_argument("--save_freq", type=int, default=5000,
                   help="Save a checkpoint every N steps")
    p.add_argument("--push_to_hub_repo", default=None,
                   help="If set, push final checkpoint to this HF repo id")
    # Smoke / dev
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--compile_model", action="store_true",
                   help="torch.compile the policy. Recommend OFF for smoke, ON for full run.")
    p.add_argument("--video_backend", default="pyav",
                   help="LeRobotDataset video backend. Default pyav — torchcodec leaks "
                        "~35 GB/worker over long runs (see TORCHCODEC_OOM_REPORT.md).")
    # LoRA — parameter-efficient VLM adaptation. With --enable_lora the VLM
    # base is frozen and adapted only via low-rank adapters (instead of
    # full-fine-tuning the VLM body); both VQA-CE and action gradients flow
    # into the adapters. Off → original cotrain (full VLM fine-tune).
    p.add_argument("--enable_lora", action="store_true",
                   help="Freeze the VLM base, adapt it via LoRA adapters only.")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_layers", default="all",
                   help="Comma-separated VLM layer indices to LoRA, or 'all'.")
    p.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj")
    # KLAL — name-token -> face-patch attention supervision on robot batches.
    p.add_argument("--enable_klal", action="store_true",
                   help="Add the KLAL attention-supervision loss on robot steps.")
    p.add_argument("--klal_lambda", type=float, default=1.0,
                   help="KLAL loss weight (WACV 2026 uses 1.0).")
    p.add_argument("--klal_layers", default="10,12,14",
                   help="VLM layers KLAL supervises. Must be a subset of the "
                        "LoRA layers when --enable_lora.")
    p.add_argument("--klal_sigma", type=float, default=1.0,
                   help="Gaussian target sigma in patch units. SmolVLA's 8x8 "
                        "grid is coarser than Pi0.5's 16x16, so 1.0 (vs the "
                        "Pi0.5 KLAL's 1.5) — empirical default, see the port doc.")
    # KLAL bbox supervision source — the M2 toolkit's data builder. KLAL uses
    # only the face bounding boxes from it; the M2 ArcFace loss is NOT added.
    p.add_argument("--face_labels_dir", default=None,
                   help="M2 toolkit face_labels/ dir (required with --enable_klal).")
    p.add_argument("--celeb_manifest", default=None,
                   help="M2 toolkit celeb_embeddings.json (required with --enable_klal).")
    p.add_argument("--aug_root", default=None,
                   help="Dir of per-variant augmentation.json (required with --enable_klal).")
    p.add_argument("--episode_mapping", default=None,
                   help="M2 toolkit episode_mapping.json (required with --enable_klal).")
    return p.parse_args()


# -----------------------------------------------------------------------------
# VL dataset — reads Roham's `eval3_objectvla_vl_pairs` parquet schema.
# Schema (verified 2026-05-20): image_path, prompt, target, bbox_xyxy_norm,
#                                celeb_name, celeb_slug, caption_type, episode,
#                                frame_idx, pid
# -----------------------------------------------------------------------------

class VLPairsDataset(Dataset):
    """VQA pairs: (image, prompt) → target. Used for the LM-head CE loss."""

    REQUIRED_COLS = {"image_path", "prompt", "target", "celeb_slug", "caption_type"}

    def __init__(self, manifest_path_or_id: str, image_root: Path | None = None):
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError("pandas required for VL dataset") from e

        # Local parquet OR HF repo id.
        if Path(manifest_path_or_id).is_file():
            self.df = pd.read_parquet(manifest_path_or_id)
            dataset_root = Path(manifest_path_or_id).parent
        else:
            from huggingface_hub import hf_hub_download, snapshot_download
            mf_path = hf_hub_download(
                repo_id=manifest_path_or_id, repo_type="dataset",
                filename="manifest.parquet",
                token=os.environ.get("HF_TOKEN"),
            )
            self.df = pd.read_parquet(mf_path)
            dataset_root = Path(mf_path).parent
            if image_root is None:
                if (dataset_root / "images").is_dir():
                    image_root = dataset_root / "images"
                else:
                    print(f"[WARN] vl_dataset: expected pre-extracted images/ under "
                          f"{dataset_root}, got=missing, "
                          f"fallback=snapshot_download(repo_id={manifest_path_or_id})",
                          flush=True)
                    full = snapshot_download(
                        repo_id=manifest_path_or_id, repo_type="dataset",
                        token=os.environ.get("HF_TOKEN"),
                    )
                    image_root = Path(full)
                    # The snapshot may have images.tar.zst still packed; unpack if so.
                    tar_zst = image_root / "images.tar.zst"
                    if tar_zst.is_file() and not (image_root / "images").is_dir():
                        print(f"[vl_dataset] extracting {tar_zst} ...", flush=True)
                        import subprocess
                        subprocess.run(
                            ["tar", "--use-compress-program=unzstd",
                             "-xf", str(tar_zst), "-C", str(image_root)],
                            check=True,
                        )

        missing = self.REQUIRED_COLS - set(self.df.columns)
        if missing:
            raise ValueError(f"VL manifest missing columns: {missing}; "
                             f"got {list(self.df.columns)}")

        self.image_root = image_root
        print(f"[vl_dataset] {len(self.df)} pairs, {self.df['celeb_slug'].nunique()} celebs, "
              f"image_root={image_root}", flush=True)
        print(f"[vl_dataset] caption mix: "
              f"{dict(self.df['caption_type'].value_counts())}", flush=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        row = self.df.iloc[idx]
        rel = Path(str(row["image_path"]))
        # Try several path layouts the manifest may produce.
        candidates = [
            self.image_root / rel,
            self.image_root.parent / rel,
        ]
        if str(rel).startswith("images/") and self.image_root.name == "images":
            candidates.insert(0, self.image_root / rel.relative_to("images"))

        img_path = next((c for c in candidates if c.is_file()), None)
        if img_path is None:
            print(f"[WARN] vl_dataset row {idx}: image not found at "
                  f"{[str(c) for c in candidates]}, fallback=gray 224x224",
                  flush=True)
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        else:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] vl_dataset row {idx}: read failed for "
                      f"{img_path}: {e}, fallback=gray", flush=True)
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        return {
            "image": img,
            "prompt": str(row["prompt"]),
            "target": str(row["target"]),
            "celeb_slug": str(row["celeb_slug"]),
        }


def make_vl_collator(processor):
    """Builds VL batches for SmolVLM2's LM head.

    SmolVLM2 uses an Idefics3-style processor that expands the `<image>`
    placeholder into N_image actual image tokens when `images=` is passed.
    To find the prompt/target boundary for label masking, we must run BOTH
    the prompt-only text AND the prompt+target text through the *same*
    `processor(text=..., images=...)` call so the image expansion happens
    identically in both. We then take the prompt-only sequence length as
    the mask boundary in the full input_ids.

    (The earlier shortcut of using `processor.tokenizer(...)` alone was
    wrong — it does NOT run the image-token expansion, so prompt_lens were
    off by ~80-170 tokens and the model would have trained to predict
    image-placeholder tokens.)
    """
    image_token = "<image>"

    def _ensure_image_placeholder(text: str) -> str:
        return text if image_token in text else f"{image_token}{text}"

    def collate(batch: list[dict]) -> dict:
        prompts_only = [_ensure_image_placeholder(ex["prompt"]) for ex in batch]
        prompts_full = [f"{p} {ex['target']}" for p, ex in zip(prompts_only, batch)]
        # SmolVLM's processor expects `images` as a list-of-lists — one sublist
        # per text sample (`process_vision` does `len(sublist) for sublist in
        # images`). A flat list is read as a single sample's images and fails
        # the n_images_in_text vs n_images_in_images check.
        images = [[ex["image"]] for ex in batch]

        # 1. Process full (prompt + target) text + images. This is what we
        #    feed to the VLM forward.
        full_inputs = processor(
            text=prompts_full,
            images=images,
            return_tensors="pt",
            padding="longest",
            # No truncation: SmolVLM expands each <image> into ~1088 tokens;
            # a small max_length would truncate them and desync the image-token
            # count between text and input_ids. VL pairs are short Q&A.
        )

        # 2. Process prompt-only with the SAME images so image-token
        #    expansion matches identically. We discard the resulting
        #    pixel_values; we only need the input_ids length.
        prompt_inputs = processor(
            text=prompts_only,
            images=images,
            return_tensors="pt",
            padding="longest",
            # No truncation: SmolVLM expands each <image> into ~1088 tokens;
            # a small max_length would truncate them and desync the image-token
            # count between text and input_ids. VL pairs are short Q&A.
        )
        # Per-sample prompt token length (number of non-pad tokens).
        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()

        # 3. Build labels: -100 on prompt portion + pad; target tokens elsewhere.
        input_ids = full_inputs["input_ids"]
        attn = full_inputs["attention_mask"]
        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            # Always leave at least one position to predict — otherwise the
            # CE loss is undefined for that sample.
            cap = max(1, min(int(plen), input_ids.shape[1] - 1))
            labels[i, :cap] = -100
        labels = labels.masked_fill(attn == 0, -100)

        out = dict(full_inputs)
        out["labels"] = labels
        return out

    return collate


# -----------------------------------------------------------------------------
# Loss paths
# -----------------------------------------------------------------------------

def smolvla_vqa_loss(policy, batch: dict, device: torch.device) -> torch.Tensor:
    """CE loss on SmolVLM2's LM head for a VL batch.

    Bypasses the action expert entirely — we call the underlying SmolVLM2
    `AutoModelForImageTextToText` instance, which has its own .lm_head and
    returns CausalLMOutputWithPast(.loss, .logits, ...) when labels= is given.
    """
    vlm = policy.model.vlm_with_expert.vlm
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Match the VLM's parameter dtype for pixel values (vision tower is bf16).
    pv_dtype = next(vlm.model.vision_model.parameters()).dtype
    if pixel_values.dtype != pv_dtype:
        pixel_values = pixel_values.to(pv_dtype)

    out = vlm(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=labels,
    )
    return out.loss


def smolvla_action_loss(policy, batch: dict) -> torch.Tensor:
    """Standard SmolVLA flow-matching loss on a robot batch."""
    loss, _ = policy.forward(batch)
    return loss


# -----------------------------------------------------------------------------
# Policy + dataset construction (reuse lerobot factories)
# -----------------------------------------------------------------------------

def _patch_smolvlm_vision_embeddings() -> None:
    """Move `boundaries` to the input device in SmolVLMVisionEmbeddings.forward.

    transformers==4.55 builds `boundaries` on CPU while the rest of the math
    runs on CUDA, so `torch.bucketize` raises a device-mismatch. This is the
    same fix `eval_3/scripts/lerobot_train_with_m2.py` already applies — copied
    here so the standalone cotrain script needs no external launcher.
    """
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
    print("[cotrain] patched SmolVLMVisionEmbeddings.forward (boundaries→device)",
          flush=True)


def load_policy_and_processor(args, device: torch.device):
    """Constructs SmolVLAPolicy with cotrain-friendly flags."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    # Load config via the ChoiceRegistry base so draccus can dispatch on the
    # "type" field stored in the checkpoint's config.json.  Calling
    # SmolVLAConfig.from_pretrained() directly fails because draccus tries to
    # decode "type" as a dataclass field of the concrete subclass.
    cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    assert isinstance(cfg, SmolVLAConfig), (
        f"Expected SmolVLAConfig from {args.pretrained_path}, got {type(cfg)}"
    )
    # VLM trainability. Without LoRA: full-fine-tune the VLM body (cotrain
    # default — CRITICAL so the VQA-CE loss can move VLM weights). With LoRA:
    # freeze the VLM base; the trainable LoRA adapters (injected below) carry
    # both the VQA-CE and the action gradients into the VLM instead.
    cfg.train_expert_only = bool(args.enable_lora)
    cfg.freeze_vision_encoder = True # Keep SigLIP frozen — RT-2 doesn't tune it either.
    cfg.empty_cameras = max(0, cfg.empty_cameras) if hasattr(cfg, "empty_cameras") else 0
    if args.vlm_model_name is not None:
        cfg.vlm_model_name = args.vlm_model_name
    # The cfg.device read in subordinate constructors should match our device.
    cfg.device = device.type

    policy = SmolVLAPolicy.from_pretrained(args.pretrained_path, config=cfg)
    policy = policy.to(device)

    # Re-apply requires_grad in case the policy load reset them.
    policy.model.vlm_with_expert.set_requires_grad()

    # LoRA: freeze the VLM base, add trainable low-rank adapters on the LM
    # attention projections. Done AFTER set_requires_grad so the adapters
    # (created requires_grad=True) survive. With train_expert_only=True the
    # base + lm_head are frozen; the adapters carry the trainable capacity.
    lora_registry: list = []
    lora_layers: tuple = ()
    if args.enable_lora:
        from m2_lora import LoRAConfig, count_lora_params, inject_lora

        text_model = policy.model.vlm_with_expert.vlm.model.text_model
        n_layers = len(text_model.layers)
        if args.lora_layers.strip().lower() == "all":
            lora_layers = tuple(range(n_layers))
        else:
            lora_layers = tuple(int(x) for x in args.lora_layers.split(","))
        lora_cfg = LoRAConfig(
            r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
            layers=lora_layers,
            target_modules=tuple(m.strip() for m in args.lora_target_modules.split(",")),
        )
        lora_registry = inject_lora(text_model, lora_cfg)
        print(f"[cotrain] LoRA injected: {len(lora_registry)} modules over "
              f"{len(lora_layers)} layers, r={args.lora_r} alpha={args.lora_alpha} "
              f"dropout={args.lora_dropout}, "
              f"{count_lora_params(lora_registry) / 1e6:.2f}M adapter params",
              flush=True)

    # SmolVLM2 processor (Idefics3Processor under the hood) for the VL collator.
    vl_processor = policy.model.vlm_with_expert.processor
    return policy, vl_processor, cfg, lora_registry, lora_layers


def load_robot_dataset(args):
    """Loads the LeRobot dataset.

    We use LeRobotDataset directly (not make_dataset()) to avoid the full
    TrainPipelineConfig surface. The dataset's per-frame items are RAW —
    they need to be put through the policy's preprocessor (see
    `build_robot_preprocessor` below) before `policy.forward()`.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=args.robot_dataset, delta_timestamps=None,
                        video_backend=args.video_backend)
    print(f"[robot_dataset] {len(ds)} frames across {ds.num_episodes} episodes "
          f"(video_backend={args.video_backend})", flush=True)
    return ds


def build_robot_preprocessor(policy_cfg, dataset, pretrained_path: str, device: torch.device):
    """Builds the SmolVLA pre/post-processor pipeline from lerobot's factory.

    The preprocessor is responsible for:
      1. Renaming observation keys to match the policy's expected schema.
      2. Adding a batch dimension (no-op for already-batched DataLoader output).
      3. Adding a trailing newline to `task` strings (SmolVLA tokenizer quirk).
      4. Tokenizing `task` → `observation.language.tokens` / `..._attention_mask`.
      5. Moving everything to `device`.
      6. Normalizing state/action features using dataset stats.

    Without this, `policy.forward(batch)` raises KeyError on the language-token
    keys. Returns (preprocessor, postprocessor) — the postprocessor is
    saved/pushed alongside but not used during training itself.
    """
    from lerobot.policies.factory import make_pre_post_processors

    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy_cfg.input_features, **policy_cfg.output_features},
                "norm_map": policy_cfg.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy_cfg.output_features,
                "norm_map": policy_cfg.normalization_mapping,
            },
        },
    )
    return pre, post


# -----------------------------------------------------------------------------
# KLAL + LoRA helpers
# -----------------------------------------------------------------------------

@contextmanager
def merged_lora_for_save(lora_registry):
    """Within this block LoRA modules are swapped for plain merged Linears so
    `save_pretrained` / `push_to_hub` write a vanilla SmolVLAPolicy checkpoint
    (loadable with no LoRA code — keeps the eval-day recipe unchanged). The
    LoRA modules are restored on exit; base weights are never mutated.
    """
    if lora_registry:
        from m2_lora import swap_to_lora, swap_to_merged

        swap_to_merged(lora_registry)
        try:
            yield
        finally:
            swap_to_lora(lora_registry)
    else:
        yield


def robot_episode_frame_indices(raw_batch: dict) -> tuple[list[int], list[int]]:
    """Per-sample (episode_index, frame_index) from a raw LeRobotDataset batch.

    KLAL supervision is keyed on these so the bbox builder can look up the
    prompted celeb's face box for each frame.
    """
    if "episode_index" not in raw_batch or "frame_index" not in raw_batch:
        raise KeyError(
            "robot batch missing 'episode_index'/'frame_index' — KLAL needs "
            "both to look up face bboxes. Got keys: "
            f"{sorted(k for k in raw_batch if isinstance(k, str))}"
        )
    ep = raw_batch["episode_index"]
    fr = raw_batch["frame_index"]
    if ep.ndim == 2:
        ep = ep[:, -1]
    if fr.ndim == 2:
        fr = fr[:, -1]
    return (ep.detach().cpu().long().tolist(),
            fr.detach().cpu().long().tolist())


def compute_klal_loss(hookset, cfg, builder, name_ids, batch, ep_idx, fr_idx,
                      device) -> torch.Tensor:
    """Build KLAL supervision for the robot batch and return the scaled loss.

    Returns a 0-d tensor; 0.0 (no grad) if no name tokens / bboxes / image
    offset could be resolved for the batch (logged once — CLAUDE.md §5).
    """
    from m2_klal import klal_loss
    from m2_klal_smolvla import extract_name_token_positions

    sup = builder.build_batch_from_episode_indices(
        episode_indices=ep_idx, frame_idxs=fr_idx, device=device)
    bbox_masks = sup["bbox_masks"]                       # (B, 3, P) bool
    B = bbox_masks.shape[0]
    tgt_slot = sup.get("target_slot_idx")
    if tgt_slot is None:
        # No explicit prompted-slot index — union the 3 slots (less precise).
        target_masks = bbox_masks.any(dim=1)             # (B, P)
    else:
        target_masks = torch.stack(
            [bbox_masks[b, int(tgt_slot[b])] for b in range(B)], dim=0)

    lang_tokens = batch.get("observation.language.tokens")
    name_pos = None
    if lang_tokens is not None:
        name_pos = extract_name_token_positions(
            lang_tokens, sup.get("target_celeb_short"), name_ids)

    off = hookset.image_prefix_len()
    patches = hookset.patches_per_image()
    if name_pos is None or off is None or patches is None:
        if not getattr(compute_klal_loss, "_warned", False):
            print(f"[WARN] KLAL: no supervision this step — "
                  f"expected=name tokens + measured image-prefix length, got "
                  f"name_pos={name_pos is not None} off={off} patches={patches}, "
                  f"fallback=KLAL contributes 0 (logged once)", flush=True)
            compute_klal_loss._warned = True
        return torch.zeros((), device=device)

    # name_pos holds positions WITHIN the language sequence; offset by the
    # image-patch span to get prefix-row indices.
    shifted = torch.where(name_pos >= 0, name_pos + off, name_pos).to(device)
    return klal_loss(hookset, image_patch_slice=slice(0, patches),
                     name_token_positions=shifted,
                     target_masks=target_masks.to(device), cfg=cfg)


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # transformers 4.55 SmolVLM device fix — must run before any SmolVLM forward.
    _patch_smolvlm_vision_embeddings()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility.
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cotrain] device={device}, dtype={args.dtype}", flush=True)
    if device.type == "cuda":
        print(f"[cotrain] gpu={torch.cuda.get_device_name(0)}, "
              f"vram={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
              flush=True)

    # 1. Policy + processor.
    print(f"[cotrain] loading SmolVLA policy from {args.pretrained_path} ...", flush=True)
    policy, vl_processor, policy_cfg, lora_registry, lora_layers = \
        load_policy_and_processor(args, device)
    policy.train()
    trainable_n = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in policy.parameters())
    print(f"[cotrain] trainable params: {trainable_n / 1e6:.1f}M / {total_n / 1e6:.1f}M "
          f"({100 * trainable_n / total_n:.1f}%)", flush=True)

    # 1b. KLAL attention supervision (robot steps only). Teaches the VLM's
    # name-token to attend to the prompted celeb's face patch, against a
    # bbox-derived Gaussian target. Bboxes come from the M2 toolkit's data
    # builder — bboxes only, the M2 ArcFace loss is NOT added.
    klal_hookset = None
    klal_cfg = None
    klal_builder = None
    klal_name_ids = None
    if args.enable_klal:
        for flag, val in [("--face_labels_dir", args.face_labels_dir),
                          ("--celeb_manifest", args.celeb_manifest),
                          ("--aug_root", args.aug_root),
                          ("--episode_mapping", args.episode_mapping)]:
            if not val:
                raise SystemExit(f"--enable_klal requires {flag}")
        if getattr(policy_cfg, "add_image_special_tokens", False):
            raise SystemExit(
                "KLAL needs add_image_special_tokens=False — with special "
                "tokens the image-prefix length the loss relies on is wrong.")

        from m2_dataloader import M2SupervisionBuilder
        from m2_klal import KLALConfig
        from m2_klal_smolvla import KLALHookSetSmolVLA, build_name_token_ids

        klal_layers = tuple(int(x) for x in args.klal_layers.split(","))
        if args.enable_lora and not set(klal_layers).issubset(set(lora_layers)):
            raise SystemExit(
                f"--klal_layers {klal_layers} must be a subset of the LoRA "
                f"layers {sorted(lora_layers)} — KLAL can only shape attention "
                f"where the q/k projections are trainable.")

        klal_builder = M2SupervisionBuilder(
            face_labels_dir=Path(args.face_labels_dir),
            manifest_path=Path(args.celeb_manifest),
            aug_root=Path(args.aug_root),
            episode_mapping_path=Path(args.episode_mapping),
        )
        vlm_we = policy.model.vlm_with_expert
        tcfg = vlm_we.config.text_config
        n_heads = tcfg.num_attention_heads
        n_kv = tcfg.num_key_value_heads
        head_dim = getattr(tcfg, "head_dim", None) or (tcfg.hidden_size // n_heads)
        klal_hookset = KLALHookSetSmolVLA(
            vlm_we.vlm.model.text_model, vlm_we, klal_layers,
            n_heads, n_kv, head_dim)
        klal_cfg = KLALConfig(capture_layers=klal_layers,
                              target_sigma_patches=args.klal_sigma,
                              lam=args.klal_lambda,
                              patch_grid=8, num_image_patches_total=64)
        klal_name_ids = build_name_token_ids(vl_processor.tokenizer)
        print(f"[cotrain] KLAL enabled: layers={klal_layers} "
              f"lambda={args.klal_lambda} sigma={args.klal_sigma}, "
              f"celeb name-tokens="
              f"{ {k: len(v) for k, v in klal_name_ids.items()} }", flush=True)

    # 2. Robot dataset + preprocessor + dataloader.
    print(f"[cotrain] loading robot dataset {args.robot_dataset} ...", flush=True)
    robot_ds = load_robot_dataset(args)
    print("[cotrain] building robot preprocessor (tokenizer + normalizer + device) ...", flush=True)
    preprocessor, postprocessor = build_robot_preprocessor(
        policy_cfg=policy_cfg,
        dataset=robot_ds,
        pretrained_path=args.pretrained_path,
        device=device,
    )
    robot_loader = DataLoader(
        robot_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    robot_iter = iter(robot_loader)

    # 3. VL dataset + dataloader.
    print(f"[cotrain] loading VL pairs from {args.vl_manifest} ...", flush=True)
    vl_image_root = Path(args.vl_image_root) if args.vl_image_root else None
    vl_ds = VLPairsDataset(args.vl_manifest, image_root=vl_image_root)
    vl_loader = DataLoader(
        vl_ds,
        batch_size=args.vl_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_vl_collator(vl_processor),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    vl_iter = iter(vl_loader)

    # 4. Optimizer — one AdamW over all trainable params.
    optim = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )

    # 5. Training loop.
    print(f"[cotrain] starting training: steps={args.steps}, "
          f"vl_ratio={args.vl_ratio} (one VL batch per {args.vl_ratio} robot batches), "
          f"lr={args.lr}", flush=True)

    period = args.vl_ratio + 1   # vl_ratio=10 → VL hits at step%11==0
    last_log_time = time.perf_counter()
    last_flow_loss = float("nan")
    last_vqa_loss = float("nan")
    last_klal_loss = float("nan")

    for step in range(args.steps):
        is_vl_step = (step % period == 0)

        if is_vl_step:
            try:
                batch = next(vl_iter)
            except StopIteration:
                vl_iter = iter(vl_loader)
                batch = next(vl_iter)
            with torch.amp.autocast(device_type=device.type,
                                     dtype=torch.bfloat16 if args.dtype == "bfloat16"
                                                          else torch.float32):
                loss = smolvla_vqa_loss(policy, batch, device)
            loss_name = "vqa_loss"
            last_vqa_loss = loss.item()
        else:
            try:
                batch = next(robot_iter)
            except StopIteration:
                robot_iter = iter(robot_loader)
                batch = next(robot_iter)
            # KLAL: read episode/frame indices off the RAW batch and arm the
            # attention hooks before the forward.
            if klal_hookset is not None:
                ep_idx, fr_idx = robot_episode_frame_indices(batch)
                klal_hookset.reset()
            # Apply the SmolVLA preprocessor: tokenizes the language task,
            # normalizes state/action features, moves to device. Without
            # this the policy's forward raises KeyError on language tokens.
            batch = preprocessor(batch)
            with torch.amp.autocast(device_type=device.type,
                                     dtype=torch.bfloat16 if args.dtype == "bfloat16"
                                                          else torch.float32):
                flow_loss = smolvla_action_loss(policy, batch)
            loss = flow_loss
            last_flow_loss = flow_loss.item()
            # KLAL loss — recomputed from the q/k captured during the forward
            # above; its gradient flows back into the VLM q/k (LoRA adapters).
            if klal_hookset is not None:
                klal_v = compute_klal_loss(
                    klal_hookset, klal_cfg, klal_builder, klal_name_ids,
                    batch, ep_idx, fr_idx, device)
                loss = flow_loss + klal_v.to(flow_loss.dtype)
                last_klal_loss = float(klal_v.detach())
            loss_name = "flow_loss"

        if not torch.isfinite(loss):
            print(f"[WARN] step {step}: non-finite loss ({loss_name}={loss.item()}), "
                  f"skipping optimizer step", flush=True)
            optim.zero_grad(set_to_none=True)
            continue

        loss.backward()
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in policy.parameters() if p.requires_grad),
                max_norm=args.grad_clip,
            ).item()
        else:
            grad_norm = float("nan")
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step < 50:
            dt = time.perf_counter() - last_log_time
            last_log_time = time.perf_counter()
            steps_per_sec = args.log_every / dt if dt > 0 else 0
            print(f"step {step:6d}  {loss_name}={loss.item():.4f}  "
                  f"(last flow={last_flow_loss:.4f} vqa={last_vqa_loss:.4f} "
                  f"klal={last_klal_loss:.4f})  "
                  f"grad={grad_norm:.2f}  steps/s={steps_per_sec:.2f}",
                  flush=True)

        if step > 0 and step % args.save_freq == 0:
            ckpt_dir = out_dir / f"step_{step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # merged_lora_for_save folds the LoRA delta into the base for the
            # duration of the save so the checkpoint is a vanilla SmolVLAPolicy.
            with merged_lora_for_save(lora_registry):
                policy.save_pretrained(ckpt_dir)
            # Save preprocessor/postprocessor alongside so eval-day inference
            # can reproduce normalization + tokenization. Without these, the
            # checkpoint cannot be deployed.
            preprocessor.save_pretrained(ckpt_dir)
            postprocessor.save_pretrained(ckpt_dir)
            print(f"[cotrain] checkpoint saved → {ckpt_dir}", flush=True)

    # 6. Final save.
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    with merged_lora_for_save(lora_registry):
        policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)
    print(f"[cotrain] final checkpoint → {final_dir}", flush=True)

    # 7. HF push (policy + processors).
    if args.push_to_hub_repo:
        try:
            with merged_lora_for_save(lora_registry):
                policy.push_to_hub(args.push_to_hub_repo)
            preprocessor.push_to_hub(args.push_to_hub_repo)
            postprocessor.push_to_hub(args.push_to_hub_repo)
            print(f"[cotrain] pushed to https://huggingface.co/{args.push_to_hub_repo}",
                  flush=True)
        except Exception as e:
            print(f"[WARN] HF push failed: expected=push policy+preprocessor+postprocessor "
                  f"to {args.push_to_hub_repo}, got={type(e).__name__}: {e}, "
                  f"fallback=local-only checkpoint at {final_dir}",
                  flush=True)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
