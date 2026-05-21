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
        --vl_manifest=HBOrtiz/eval3_track3_vl_pairs \\
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
the VL batches — it teaches the celeb-name token to attend to the prompted
portrait, against a Gaussian target built from the VL dataset's
`quad_corners_norm` column (no external bbox source needed):

    ... --enable_lora --enable_klal --klal_layers=10,12,14 --klal_lambda=1.0

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
from datetime import timedelta
from pathlib import Path

# HF's Rust `tokenizers` deadlocks if a fast tokenizer is used in the parent
# (it is — policy build + build_name_token_ids) and then again inside a forked
# DataLoader worker. The VL collator runs `processor(...)` in workers, so with
# num_workers>0 the first VL batch fetch hangs silently. Disable the tokenizer
# thread pool before any fork. Must be set before `tokenizers` is imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

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
    # KLAL's attention-supervision target is built from the VL dataset's
    # `quad_corners_norm` column — no external bbox source is needed.
    return p.parse_args()


# -----------------------------------------------------------------------------
# VL dataset — reads the `eval3_*_vl_pairs` parquet schema (e.g.
# eval3_track3_vl_pairs). Required columns: image_path, prompt, target,
# celeb_slug, caption_type (plus bbox_xyxy_norm, celeb_name, frame_idx, … ).
# -----------------------------------------------------------------------------

class VLPairsDataset(Dataset):
    """VQA pairs: (image, prompt) → target. Used for the LM-head CE loss."""

    REQUIRED_COLS = {"image_path", "prompt", "target", "celeb_slug",
                     "caption_type", "quad_corners_norm", "bbox_refit_ok"}

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
                    # The images ship as a packed archive — named data.tar.zst
                    # or images.tar.zst depending on how the dataset was pushed
                    # (eval3_objectvla_vl_pairs uses data.tar.zst). Extract
                    # whichever *.tar.zst exists; it unpacks to images/chunk-*/.
                    if not (image_root / "images").is_dir():
                        archives = sorted(image_root.glob("*.tar.zst"))
                        if archives:
                            import subprocess
                            for arch in archives:
                                print(f"[vl_dataset] extracting {arch.name} ...",
                                      flush=True)
                                subprocess.run(
                                    ["tar", "--use-compress-program=unzstd",
                                     "-xf", str(arch), "-C", str(image_root)],
                                    check=True,
                                )
                        else:
                            print(f"[WARN] vl_dataset: no images/ dir and no "
                                  f"*.tar.zst under {image_root}; expected="
                                  f"packed images, got=neither, fallback=gray "
                                  f"placeholders", flush=True)

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
            # KLAL attention target: the printed-portrait quad (4 corners,
            # normalised [0,1]) and whether its refit was reliable. The parquet
            # stores the quad as an object-array of four (2,)-arrays — np.stack
            # joins them into a clean (4,2); np.asarray(...,float32) would raise.
            "quad_corners_norm": np.stack(
                row["quad_corners_norm"]).astype(np.float32).reshape(-1, 2),
            "bbox_refit_ok": bool(row["bbox_refit_ok"]),
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
            # do_image_splitting=False: one image -> exactly 64 contiguous
            # <image> tokens on a clean 8x8 grid, which KLAL's quad->patch
            # mask relies on. No truncation either (VL pairs are short Q&A).
            do_image_splitting=False,
        )

        # 2. Process prompt-only with the SAME images so image-token
        #    expansion matches identically. We discard the resulting
        #    pixel_values; we only need the input_ids length.
        prompt_inputs = processor(
            text=prompts_only,
            images=images,
            return_tensors="pt",
            padding="longest",
            do_image_splitting=False,
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
        # KLAL (VL-forward attention supervision) inputs — consumed by
        # compute_klal_loss_vl on VL steps; ignored by the VQA loss.
        out["celeb_slug"] = [ex["celeb_slug"] for ex in batch]
        out["quad_corners_norm"] = torch.from_numpy(
            np.stack([ex["quad_corners_norm"] for ex in batch])).float()
        out["bbox_refit_ok"] = torch.tensor(
            [ex["bbox_refit_ok"] for ex in batch], dtype=torch.bool)
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


def load_robot_dataset(args, policy_cfg):
    """Loads the LeRobot dataset.

    We use LeRobotDataset directly (not make_dataset()) to avoid the full
    TrainPipelineConfig surface, but we DO reuse lerobot's
    `resolve_delta_timestamps` so each frame carries the action CHUNK
    (`chunk_size` future steps) SmolVLA's flow-matching head needs — without
    it `policy.forward` raises a prefix/suffix mask-size mismatch. The
    dataset's per-frame items are otherwise RAW and need the policy's
    preprocessor (see `build_robot_preprocessor`) before `policy.forward()`.
    """
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(args.robot_dataset)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    ds = LeRobotDataset(repo_id=args.robot_dataset, delta_timestamps=delta_timestamps,
                        video_backend=args.video_backend)
    dt_summary = ({k: len(v) for k, v in delta_timestamps.items()}
                  if delta_timestamps else None)
    print(f"[robot_dataset] {len(ds)} frames across {ds.num_episodes} episodes "
          f"(video_backend={args.video_backend}, delta_timestamps={dt_summary})",
          flush=True)
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
def main_process_first(is_main: bool, world_size: int):
    """Rank 0 runs the body first (e.g. a dataset download); the other ranks
    wait at a barrier, then run it against the now-warm shared cache.

    Without this, all N ranks construct the LeRobotDataset / VLPairsDataset at
    once and hammer the HF API in parallel — which gets the whole job
    429-rate-limited. With it, exactly one rank downloads.
    """
    if world_size > 1 and not is_main:
        dist.barrier()
    try:
        yield
    finally:
        if world_size > 1 and is_main:
            dist.barrier()


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


# -----------------------------------------------------------------------------
# Multi-GPU helpers
# -----------------------------------------------------------------------------

def _setup_distributed():
    """Initialise torch.distributed when launched under torchrun (WORLD_SIZE>1).

    Returns (rank, world_size, local_rank, is_main). A plain single-process run
    returns (0, 1, 0, True) and never touches the process group.

    Data parallelism is done with a MANUAL gradient all-reduce (see the loop) —
    not DDP — because the VQA step calls the inner `vlm(...)` directly, which
    would bypass a DDP wrapper's forward and break its gradient sync.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        # nccl for real multi-GPU. COTRAIN_DDP_BACKEND=gloo is a fallback that
        # also lets the distributed path be exercised on a single-GPU box.
        backend = os.environ.get("COTRAIN_DDP_BACKEND", "nccl")
        # Generous timeout: rank 0 downloads the full ~15 GB dataset inside a
        # main_process_first() barrier while the other ranks wait — that can
        # exceed the default 10-minute collective timeout.
        dist.init_process_group(backend=backend, timeout=timedelta(hours=2))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
        print(f"[cotrain] distributed: rank {rank}/{world_size} "
              f"(local_rank {local_rank})", flush=True)
    return rank, world_size, local_rank, (rank == 0)


def _save_and_push(policy, preprocessor, postprocessor, lora_registry,
                   ckpt_dir: Path, push_repo: str | None, path_in_repo: str) -> None:
    """Save a merged-LoRA checkpoint locally, then (if push_repo) upload it to
    HF under `path_in_repo`. Rank-0 only. A failed push is logged, not fatal —
    the run keeps going and the local checkpoint is kept.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # merged_lora_for_save folds the LoRA delta into the base for the save, so
    # the checkpoint is a vanilla SmolVLAPolicy.
    with merged_lora_for_save(lora_registry):
        policy.save_pretrained(ckpt_dir)
    # Processors saved alongside so eval-day inference reproduces normalization.
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)
    print(f"[cotrain] checkpoint saved → {ckpt_dir}", flush=True)
    if not push_repo:
        return
    try:
        from huggingface_hub import HfApi, create_repo

        token = os.environ.get("HF_TOKEN")
        create_repo(push_repo, repo_type="model", exist_ok=True, token=token)
        HfApi(token=token).upload_folder(
            folder_path=str(ckpt_dir), repo_id=push_repo,
            path_in_repo=path_in_repo, repo_type="model",
        )
        print(f"[cotrain] pushed → "
              f"https://huggingface.co/{push_repo}/tree/main/{path_in_repo}",
              flush=True)
    except Exception as e:
        print(f"[WARN] HF push failed: expected=upload {ckpt_dir} to "
              f"{push_repo}/{path_in_repo}, got={type(e).__name__}: {e}, "
              f"fallback=local checkpoint kept at {ckpt_dir}", flush=True)


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------

def _assert_ddp_synced(policy, world_size: int, is_main: bool, device, tag: str
                       ) -> None:
    """Verify every rank holds bit-identical trainable weights.

    If the broadcast-sync and the per-step gradient all-reduce are working,
    all ranks apply the identical averaged gradient and stay in lockstep, so
    a checksum of the trainable params is identical across ranks. A mismatch
    means the ranks have diverged into different models — raise rather than
    let a 24h run silently train 8 disagreeing policies. Cheap: one checksum
    + two all-reduces; called post-broadcast and at every checkpoint.
    """
    if world_size <= 1:
        return
    checksum = torch.zeros((), dtype=torch.float64, device=device)
    for p in policy.parameters():
        if p.requires_grad:
            checksum = checksum + p.detach().double().sum()
    lo = checksum.clone()
    hi = checksum.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if not torch.isfinite(checksum) or not torch.isclose(lo, hi, rtol=1e-5,
                                                         atol=1e-3):
        raise RuntimeError(
            f"[cotrain] DDP DESYNC at {tag}: trainable-param checksum differs "
            f"across ranks (min={lo.item():.6f} max={hi.item():.6f}) — the "
            f"ranks have diverged into different models. Aborting.")
    if is_main:
        print(f"[cotrain] DDP sync verified at {tag} "
              f"(trainable-param checksum={lo.item():.6f})", flush=True)


def main() -> int:
    args = parse_args()

    # transformers 4.55 SmolVLM device fix — must run before any SmolVLM forward.
    _patch_smolvlm_vision_embeddings()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Distributed setup (no-op for a single-process run).
    rank, world_size, local_rank, is_main = _setup_distributed()

    # Reproducibility — same seed on every rank so LoRA's random init is
    # identical; ranks are also explicitly broadcast-synced after model build.
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
    print(f"[cotrain] rank {rank}/{world_size}  device={device}  dtype={args.dtype}",
          flush=True)
    if device.type == "cuda":
        print(f"[cotrain] gpu={torch.cuda.get_device_name(device)}, "
              f"vram={torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}GB",
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

    # Sync every rank to rank 0's weights — LoRA's lora_A is randomly inited;
    # broadcast so all ranks start bit-identical (the manual gradient
    # all-reduce in the loop keeps them in sync thereafter).
    if world_size > 1:
        for p in policy.parameters():
            dist.broadcast(p.data, src=0)
        for b in policy.buffers():
            dist.broadcast(b.data, src=0)
        print(f"[cotrain] rank {rank}: broadcast-synced model from rank 0",
              flush=True)
        _assert_ddp_synced(policy, world_size, is_main, device, "post-broadcast")

    # 1b. KLAL attention supervision (VL steps). On each VL batch, after the
    # VQA loss, KLAL supervises the celeb-name token's attention toward the
    # prompted portrait's quad (eval3_track3_vl_pairs.quad_corners_norm),
    # recomputed faithfully from the SmolVLM2 text model's q/k + rotary_emb.
    klal_hookset = None
    klal_cfg = None
    klal_name_ids = None
    if args.enable_klal:
        from m2_klal import KLALConfig
        from m2_klal_smolvla import build_name_token_ids
        from m2_klal_vl import (KLALHookSetSmolVLMVL, compute_klal_loss_vl,
                                NUM_IMAGE_PATCHES, PATCH_GRID)

        klal_layers = tuple(int(x) for x in args.klal_layers.split(","))
        if args.enable_lora and not set(klal_layers).issubset(set(lora_layers)):
            raise SystemExit(
                f"--klal_layers {klal_layers} must be a subset of the LoRA "
                f"layers {sorted(lora_layers)} — KLAL can only shape attention "
                f"where the q/k projections are trainable.")
        text_model = policy.model.vlm_with_expert.vlm.model.text_model
        n_text_layers = len(text_model.layers)
        if max(klal_layers) >= n_text_layers:
            raise SystemExit(
                f"--klal_layers {klal_layers} out of range — the VLM text "
                f"model has only {n_text_layers} layers (lerobot truncates it).")
        tcfg = text_model.config
        attn0 = text_model.layers[0].self_attn
        n_heads = tcfg.num_attention_heads
        n_kv = tcfg.num_key_value_heads
        head_dim = (getattr(attn0, "head_dim", None)
                    or getattr(tcfg, "head_dim", None)
                    or tcfg.hidden_size // n_heads)
        scaling = getattr(attn0, "scaling", None) or (head_dim ** -0.5)
        klal_hookset = KLALHookSetSmolVLMVL(
            text_model, klal_layers, n_heads, n_kv, head_dim, scaling)
        klal_cfg = KLALConfig(capture_layers=klal_layers,
                              target_sigma_patches=args.klal_sigma,
                              lam=args.klal_lambda,
                              patch_grid=PATCH_GRID,
                              num_image_patches_total=NUM_IMAGE_PATCHES)
        klal_name_ids = build_name_token_ids(vl_processor.tokenizer)
        print(f"[cotrain] KLAL (VL-forward) enabled: layers={klal_layers} "
              f"lambda={args.klal_lambda} sigma={args.klal_sigma}, "
              f"heads={n_heads}/{n_kv} head_dim={head_dim}, "
              f"celeb name-tokens="
              f"{ {k: len(v) for k, v in klal_name_ids.items()} }", flush=True)

    # 2. Robot dataset + preprocessor + dataloader.
    print(f"[cotrain] loading robot dataset {args.robot_dataset} ...", flush=True)
    with main_process_first(is_main, world_size):
        robot_ds = load_robot_dataset(args, policy_cfg)
    print("[cotrain] building robot preprocessor (tokenizer + normalizer + device) ...", flush=True)
    preprocessor, postprocessor = build_robot_preprocessor(
        policy_cfg=policy_cfg,
        dataset=robot_ds,
        pretrained_path=args.pretrained_path,
        device=device,
    )
    # The merged dataset's metadata can overcount frames vs the actual data
    # rows (so101_eval3_track3_v3_baseline: meta says 5,053,972, the parquet
    # has 5,053,812). LeRobotDataset.__len__ trusts the metadata, so the
    # sampler would emit out-of-range indices and crash a DataLoader worker.
    # Cap the dataloader to the real row count; robot_ds itself is left intact
    # for the preprocessor above (which needs its LeRobotDataset metadata).
    robot_true_len = len(robot_ds.hf_dataset)
    if len(robot_ds) != robot_true_len:
        print(f"[WARN] robot dataset: metadata frame count {len(robot_ds)} != "
              f"actual data rows {robot_true_len} (off by "
              f"{len(robot_ds) - robot_true_len}); fallback=capping the "
              f"dataloader to {robot_true_len} usable frames", flush=True)
        robot_ds_loader = Subset(robot_ds, range(robot_true_len))
    else:
        robot_ds_loader = robot_ds
    robot_sampler = (DistributedSampler(robot_ds_loader, num_replicas=world_size,
                                        rank=rank, shuffle=True, drop_last=True)
                     if world_size > 1 else None)
    robot_loader = DataLoader(
        robot_ds_loader,
        batch_size=args.batch_size,
        sampler=robot_sampler,
        shuffle=(robot_sampler is None),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    robot_iter = iter(robot_loader)

    # 3. VL dataset + dataloader.
    print(f"[cotrain] loading VL pairs from {args.vl_manifest} ...", flush=True)
    vl_image_root = Path(args.vl_image_root) if args.vl_image_root else None
    with main_process_first(is_main, world_size):
        vl_ds = VLPairsDataset(args.vl_manifest, image_root=vl_image_root)
    vl_sampler = (DistributedSampler(vl_ds, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True)
                  if world_size > 1 else None)
    vl_loader = DataLoader(
        vl_ds,
        batch_size=args.vl_batch_size,
        sampler=vl_sampler,
        shuffle=(vl_sampler is None),
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
    # Fixed, rank-identical list of trainable params for the gradient
    # all-reduce below. policy.parameters() iterates in deterministic
    # registration order, so this list is identical on every rank.
    ddp_params = [p for p in policy.parameters() if p.requires_grad]

    # 5. Training loop.
    print(f"[cotrain] starting training: steps={args.steps}, "
          f"vl_ratio={args.vl_ratio} (one VL batch per {args.vl_ratio} robot batches), "
          f"lr={args.lr}", flush=True)

    period = args.vl_ratio + 1   # vl_ratio=10 → VL hits at step%11==0
    train_start = time.perf_counter()
    last_log_time = train_start
    last_flow_loss = float("nan")
    last_vqa_loss = float("nan")
    last_klal_loss = float("nan")
    robot_epoch = 0
    vl_epoch = 0

    for step in range(args.steps):
        is_vl_step = (step % period == 0)

        if is_vl_step:
            try:
                batch = next(vl_iter)
            except StopIteration:
                vl_epoch += 1
                if vl_sampler is not None:
                    vl_sampler.set_epoch(vl_epoch)
                vl_iter = iter(vl_loader)
                batch = next(vl_iter)
            # KLAL: arm the hooks before the VL forward — q/k + rotary_emb are
            # captured during smolvla_vqa_loss's vlm(...) call.
            if klal_hookset is not None:
                klal_hookset.reset()
            with torch.amp.autocast(device_type=device.type,
                                     dtype=torch.bfloat16 if args.dtype == "bfloat16"
                                                          else torch.float32):
                vqa_loss = smolvla_vqa_loss(policy, batch, device)
            loss = vqa_loss
            last_vqa_loss = float(vqa_loss.detach())
            # KLAL attention-supervision loss, recomputed from the q/k captured
            # during the VL forward; its gradient flows into the VLM q/k LoRA.
            if klal_hookset is not None:
                klal_v = compute_klal_loss_vl(
                    klal_hookset, klal_cfg, klal_name_ids, batch, device)
                loss = vqa_loss + klal_v.to(vqa_loss.dtype)
                last_klal_loss = float(klal_v.detach())
            loss_name = "vqa_loss"
        else:
            try:
                batch = next(robot_iter)
            except StopIteration:
                robot_epoch += 1
                if robot_sampler is not None:
                    robot_sampler.set_epoch(robot_epoch)
                robot_iter = iter(robot_loader)
                batch = next(robot_iter)
            # Apply the SmolVLA preprocessor: tokenizes the language task,
            # normalizes state/action features, moves to device. Without
            # this the policy's forward raises KeyError on language tokens.
            batch = preprocessor(batch)
            with torch.amp.autocast(device_type=device.type,
                                     dtype=torch.bfloat16 if args.dtype == "bfloat16"
                                                          else torch.float32):
                flow_loss = smolvla_action_loss(policy, batch)
            loss = flow_loss
            last_flow_loss = float(flow_loss.detach())
            loss_name = "flow_loss"

        # Skip the step if the loss is non-finite on ANY rank. The finite flag
        # is all-reduced BEFORE backward so every rank decides together — a
        # lone `continue` would deadlock the others at the next collective.
        finite = torch.tensor([1.0 if torch.isfinite(loss) else 0.0], device=device)
        if world_size > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() < 1.0:
            if is_main:
                print(f"[WARN] step {step}: non-finite loss ({loss_name}), "
                      f"expected=finite, got={loss.item()}, fallback=skip step",
                      flush=True)
            optim.zero_grad(set_to_none=True)
            continue

        loss.backward()

        # Distributed data parallelism: average gradients across ranks.
        # All-reduce the FIXED ddp_params list unconditionally — zero-filling
        # any grad autograd left as None this step. Every rank reduces the
        # same tensors in the same order, so the collective is safe by
        # construction (gating on `p.grad is not None` would risk an NCCL
        # hang if the grad-coverage set ever differed across ranks).
        if world_size > 1:
            for p in ddp_params:
                g = p.grad if p.grad is not None else torch.zeros_like(p)
                dist.all_reduce(g, op=dist.ReduceOp.SUM)
                g /= world_size
                p.grad = g

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in policy.parameters() if p.requires_grad),
                max_norm=args.grad_clip,
            ).item()
        else:
            grad_norm = float("nan")
        optim.step()
        optim.zero_grad(set_to_none=True)

        if is_main and (step % args.log_every == 0 or step < 50):
            now = time.perf_counter()
            dt = now - last_log_time
            last_log_time = now
            steps_per_sec = args.log_every / dt if dt > 0 else 0
            elapsed_min = (now - train_start) / 60
            # ETA from the average rate over the whole run so far (smoother
            # than the noisy instantaneous steps/s).
            avg_sps = step / (now - train_start) if now > train_start else 0
            eta_min = (args.steps - step) / avg_sps / 60 if avg_sps > 0 else float("nan")
            print(f"step {step:6d}  {loss_name}={loss.item():.4f}  "
                  f"(last flow={last_flow_loss:.4f} vqa={last_vqa_loss:.4f} "
                  f"klal={last_klal_loss:.4f})  "
                  f"grad={grad_norm:.2f}  steps/s={steps_per_sec:.2f}  "
                  f"vram={torch.cuda.max_memory_reserved(device) / 2**30:.1f}gib  "
                  f"elapsed={elapsed_min:.0f}min  eta={eta_min:.0f}min",
                  flush=True)

        if step > 0 and step % args.save_freq == 0:
            _assert_ddp_synced(policy, world_size, is_main, device,
                               f"step {step}")
            if is_main:
                _save_and_push(policy, preprocessor, postprocessor, lora_registry,
                               out_dir / f"step_{step:06d}",
                               args.push_to_hub_repo, f"step_{step:06d}")
            # Other ranks wait while rank 0 saves — it swaps LoRA modules for
            # the save, so no rank may run a forward against a mutated tree.
            if world_size > 1:
                dist.barrier()

    # 6. Final save + push (rank 0).
    if is_main:
        _save_and_push(policy, preprocessor, postprocessor, lora_registry,
                       out_dir / "final", args.push_to_hub_repo, "final")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
