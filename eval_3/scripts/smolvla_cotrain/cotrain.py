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
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# KLAL lives next to this script; make it importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smolvla_klal import KLALConfig, KLALHookSet, bbox_to_patch_mask, klal_loss  # noqa: E402


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
    # VL caption selection.
    #   qa_grounded      → target is the bare celeb name ("Barack Obama"); use
    #                      this for the KLAL attention-supervision experiment so
    #                      every label position is a name token.
    #   location_explicit→ target encodes the bbox as text ("...is at [0.06,..]");
    #                      this is the CHEAP ObjectVLA bbox-prediction approach —
    #                      trains via VQA CE alone, no KLAL needed.
    #   all              → use both caption types.
    p.add_argument("--caption_filter", default="all",
                   choices=["all", "qa_grounded", "location_explicit"],
                   help="Which caption_type rows to keep in the VL stream.")
    # KLAL attention-supervision loss (L_attn).
    p.add_argument("--use_klal", action="store_true",
                   help="Add the KL attention-supervision loss on VL steps. "
                        "Requires bbox_xyxy_norm in the manifest. Pair with "
                        "--caption_filter=qa_grounded.")
    p.add_argument("--klal_lam", type=float, default=1.0,
                   help="Weight λ on L_attn (total = vqa + λ·klal on VL steps).")
    p.add_argument("--klal_layers", default="6,9,12,15",
                   help="Comma-list of text-layer indices to supervise. MUST be "
                        "in [0,15] (SmolVLA truncates to 16 layers).")
    p.add_argument("--klal_sigma", type=float, default=1.0,
                   help="Gaussian target std in 8x8-grid patch units.")
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

    def __init__(self, manifest_path_or_id: str, image_root: Path | None = None,
                 caption_filter: str = "all", need_bbox: bool = False):
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

        required = set(self.REQUIRED_COLS)
        if need_bbox:
            required.add("bbox_xyxy_norm")
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"VL manifest missing columns: {missing}; "
                             f"got {list(self.df.columns)}")

        # Caption-type filter (qa_grounded for KLAL, location_explicit for the
        # cheap bbox-as-text approach, all for both).
        self.need_bbox = need_bbox
        if caption_filter != "all":
            before = len(self.df)
            self.df = self.df[self.df["caption_type"] == caption_filter].reset_index(drop=True)
            print(f"[vl_dataset] caption_filter={caption_filter}: kept {len(self.df)}/{before} rows",
                  flush=True)
            if len(self.df) == 0:
                raise ValueError(f"caption_filter={caption_filter} left 0 rows; "
                                 f"available types: "
                                 f"{list(self.df['caption_type'].unique())}")

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

        item = {
            "image": img,
            "prompt": str(row["prompt"]),
            "target": str(row["target"]),
            "celeb_slug": str(row["celeb_slug"]),
        }
        if self.need_bbox:
            # bbox_xyxy_norm is a list/array [x1,y1,x2,y2] in [0,1]; may be None
            # for rows without a localised face — emit a sentinel all-zero box
            # (KLAL treats an all-zero target mask as "no supervision").
            bb = row["bbox_xyxy_norm"]
            if bb is None or (hasattr(bb, "__len__") and len(bb) != 4):
                print(f"[WARN] vl_dataset row {idx}: expected 4-elt bbox_xyxy_norm, "
                      f"got={bb}, fallback=zeros (no KLAL supervision this sample)",
                      flush=True)
                bb = [0.0, 0.0, 0.0, 0.0]
            item["bbox_xyxy_norm"] = [float(v) for v in bb]
        return item


def make_vl_collator(processor, max_text_len: int = 256,
                     need_klal: bool = False, image_token_id: int | None = None):
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

    KLAL mode (need_klal=True) additionally returns, per batch:
      - "image_cols"      (B, 64) long: image-token column indices (input_ids
                          == image_token_id). do_image_splitting is pinned False
                          so each image expands to exactly 64 tokens (the 8x8
                          pixel-shuffle grid), matching SmolVLA's deployed
                          single-global-image behaviour.
      - "name_positions"  (B, K_max) long, -1 padded: positions where
                          labels != -100 (the target-name token rows).
      - "bbox"            (B, 4) float: [x1,y1,x2,y2] normalised face boxes.
    """
    image_token = "<image>"

    def _ensure_image_placeholder(text: str) -> str:
        return text if image_token in text else f"{image_token}{text}"

    def collate(batch: list[dict]) -> dict:
        prompts_only = [_ensure_image_placeholder(ex["prompt"]) for ex in batch]
        prompts_full = [f"{p} {ex['target']}" for p, ex in zip(prompts_only, batch)]
        images = [ex["image"] for ex in batch]

        # Pin do_image_splitting=False: one global image → exactly 64 image
        # tokens (8x8 grid). This is what SmolVLA uses at inference and what the
        # KLAL bbox→grid mapping assumes. Without it the processor may tile the
        # image (variable token count, broken grid).
        proc_kwargs = dict(
            return_tensors="pt", padding="longest",
            truncation=True, max_length=max_text_len,
            do_image_splitting=False,
        )

        full_inputs = processor(text=prompts_full, images=images, **proc_kwargs)
        prompt_inputs = processor(text=prompts_only, images=images, **proc_kwargs)
        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()

        input_ids = full_inputs["input_ids"]
        attn = full_inputs["attention_mask"]
        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            cap = max(1, min(int(plen), input_ids.shape[1] - 1))
            labels[i, :cap] = -100
        labels = labels.masked_fill(attn == 0, -100)

        out = dict(full_inputs)
        out["labels"] = labels

        if need_klal:
            if image_token_id is None:
                raise ValueError("need_klal=True requires image_token_id")
            B, L = input_ids.shape
            # Image columns: positions where input_ids == image_token_id.
            # With do_image_splitting=False + 1 image/sample this is exactly 64
            # per sample, so we can stack to (B, 64). If a sample has a
            # different count (shouldn't happen), warn + pad/truncate to 64.
            EXPECT_P = 64
            img_cols_rows = []
            for b in range(B):
                cols = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
                if cols.numel() != EXPECT_P:
                    print(f"[WARN] vl_collator KLAL: sample {b} has {cols.numel()} "
                          f"image tokens, expected={EXPECT_P}, "
                          f"fallback=pad/truncate to {EXPECT_P}", flush=True)
                    if cols.numel() > EXPECT_P:
                        cols = cols[:EXPECT_P]
                    else:
                        pad = torch.full((EXPECT_P - cols.numel(),), cols[-1] if cols.numel() else 0,
                                         dtype=cols.dtype)
                        cols = torch.cat([cols, pad])
                img_cols_rows.append(cols)
            out["image_cols"] = torch.stack(img_cols_rows, dim=0)  # (B, 64)

            # Name-token positions: where labels != -100, padded with -1.
            name_rows = [(labels[b] != -100).nonzero(as_tuple=True)[0] for b in range(B)]
            k_max = max(1, max(r.numel() for r in name_rows))
            name_pos = torch.full((B, k_max), -1, dtype=torch.long)
            for b, r in enumerate(name_rows):
                name_pos[b, : r.numel()] = r
            out["name_positions"] = name_pos

            out["bbox"] = torch.tensor([ex["bbox_xyxy_norm"] for ex in batch],
                                       dtype=torch.float32)
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

def load_policy_and_processor(args, device: torch.device):
    """Constructs SmolVLAPolicy with cotrain-friendly flags."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    # Load config from the pretrained checkpoint; override the cotrain-critical flags.
    cfg = SmolVLAConfig.from_pretrained(args.pretrained_path)
    cfg.train_expert_only = False    # CRITICAL: VLM body must be trainable for VQA CE.
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

    # SmolVLM2 processor (Idefics3Processor under the hood) for the VL collator.
    vl_processor = policy.model.vlm_with_expert.processor
    return policy, vl_processor, cfg


def load_robot_dataset(args):
    """Loads the LeRobot dataset.

    We use LeRobotDataset directly (not make_dataset()) to avoid the full
    TrainPipelineConfig surface. The dataset's per-frame items are RAW —
    they need to be put through the policy's preprocessor (see
    `build_robot_preprocessor` below) before `policy.forward()`.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=args.robot_dataset, delta_timestamps=None)
    print(f"[robot_dataset] {len(ds)} frames across {ds.num_episodes} episodes", flush=True)
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
# Main training loop
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

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
    policy, vl_processor, policy_cfg = load_policy_and_processor(args, device)
    policy.train()
    trainable_n = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in policy.parameters())
    print(f"[cotrain] trainable params: {trainable_n / 1e6:.1f}M / {total_n / 1e6:.1f}M "
          f"({100 * trainable_n / total_n:.1f}%)", flush=True)

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
    vl_ds = VLPairsDataset(args.vl_manifest, image_root=vl_image_root,
                           caption_filter=args.caption_filter,
                           need_bbox=args.use_klal)
    # image_token_id from the VLM config (verified 49190 for SmolVLM2-500M).
    image_token_id = policy.model.vlm_with_expert.vlm.config.image_token_id
    vl_loader = DataLoader(
        vl_ds,
        batch_size=args.vl_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_vl_collator(vl_processor, need_klal=args.use_klal,
                                    image_token_id=image_token_id),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    vl_iter = iter(vl_loader)

    # 3b. KLAL hookset (L_attn) — registers q_proj/k_proj/rotary hooks on the
    # truncated SmolVLM2 text layers. Recomputes name→image-patch attention.
    klal_hooks = None
    klal_cfg = None
    if args.use_klal:
        text_model = policy.model.vlm_with_expert.vlm.model.text_model
        tcfg = policy.model.vlm_with_expert.vlm.config.text_config
        capture_layers = tuple(int(x) for x in args.klal_layers.split(","))
        klal_cfg = KLALConfig(capture_layers=capture_layers,
                              target_sigma_patches=args.klal_sigma,
                              lam=1.0)  # λ applied in the loop, keep cfg.lam=1
        klal_hooks = KLALHookSet(
            text_model=text_model,
            layers=capture_layers,
            n_heads=tcfg.num_attention_heads,
            n_kv_heads=tcfg.num_key_value_heads,
            head_dim=tcfg.head_dim,
        )
        print(f"[cotrain] KLAL enabled: layers={capture_layers}, "
              f"λ={args.klal_lam}, σ={args.klal_sigma}, "
              f"heads={tcfg.num_attention_heads}/{tcfg.num_key_value_heads}, "
              f"head_dim={tcfg.head_dim}", flush=True)

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
            if klal_hooks is not None:
                klal_hooks.reset()
            with torch.amp.autocast(device_type=device.type,
                                     dtype=torch.bfloat16 if args.dtype == "bfloat16"
                                                          else torch.float32):
                vqa = smolvla_vqa_loss(policy, batch, device)
                if klal_hooks is not None:
                    # Build per-sample target masks from the bbox on the 8x8 grid.
                    image_cols = batch["image_cols"].to(device)
                    name_pos = batch["name_positions"].to(device)
                    bbox = batch["bbox"].to(device)
                    grid = int(round(image_cols.shape[1] ** 0.5))
                    masks = torch.stack(
                        [bbox_to_patch_mask(bbox[b].tolist(), grid, device)
                         for b in range(bbox.shape[0])],
                        dim=0,
                    )  # (B, P) bool
                    attn_loss = klal_loss(klal_hooks, image_cols, name_pos,
                                          masks, klal_cfg)
                    loss = vqa + args.klal_lam * attn_loss
                    last_klal_loss = float(attn_loss.item())
                else:
                    loss = vqa
            loss_name = "vqa_loss"
            last_vqa_loss = float(vqa.item())
        else:
            try:
                batch = next(robot_iter)
            except StopIteration:
                robot_iter = iter(robot_loader)
                batch = next(robot_iter)
            # Apply the SmolVLA preprocessor: tokenizes the language task,
            # normalizes state/action features, moves to device. Without
            # this the policy's forward raises KeyError on language tokens.
            batch = preprocessor(batch)
            with torch.amp.autocast(device_type=device.type,
                                     dtype=torch.bfloat16 if args.dtype == "bfloat16"
                                                          else torch.float32):
                loss = smolvla_action_loss(policy, batch)
            loss_name = "flow_loss"
            last_flow_loss = loss.item()

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
            policy.save_pretrained(ckpt_dir)
            # Save preprocessor/postprocessor alongside so eval-day inference
            # can reproduce normalization + tokenization. Without these, the
            # checkpoint cannot be deployed.
            preprocessor.save_pretrained(ckpt_dir)
            postprocessor.save_pretrained(ckpt_dir)
            print(f"[cotrain] checkpoint saved → {ckpt_dir}", flush=True)

    # Remove KLAL hooks before save/push (they hold references to activations).
    if klal_hooks is not None:
        klal_hooks.remove()

    # 6. Final save.
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)
    print(f"[cotrain] final checkpoint → {final_dir}", flush=True)

    # 7. HF push (policy + processors).
    if args.push_to_hub_repo:
        try:
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
