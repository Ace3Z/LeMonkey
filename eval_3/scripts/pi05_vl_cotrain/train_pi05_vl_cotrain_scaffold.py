#!/usr/bin/env python3
"""Pi0.5 + ObjectVLA VL cotrain wrapper around lerobot-train.

SCAFFOLD ONLY — the deployed Pi0.5 checkpoint was trained with
../training_vm/train_pi05.sh; this file documents the intended structure but is
not executed in production.

This wrapper is preserved as a starting point for the enhanced ObjectVLA
recipe; the deployed Pi0.5 reference policy
(HBOrtiz/so101_pi05_eval3) was trained via the vanilla LoRA path in
`../training_vm/train_pi05.sh`, not via this file. The dataloader, mixed-batch
alternation, per-layer LoRA rank, and EMA pieces are written but the
training-loop hook into the installed lerobot-train was not completed.

Recipe (when fully integrated):
  - 10:1 robot:VL batch alternation (ObjectVLA arXiv 2502.11550)
  - per-layer LoRA rank (see precomputed/layer_rank.json)
  - two-phase curriculum (see curriculum_sampler.py)
  - EMA shadow weights

VQA forward path reuses the pattern from
`eval_3/scripts/warmstart/train_paligemma_vqa.py`: PaliGemmaProcessor
with suffix masking and standard HF `model.forward()` loss, with a
manual image-feature splice as a fallback for transformers >= 5.0
dict-typed attention masks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


# -----------------------------------------------------------------------------
# Configuration parsing — extends lerobot's argparse with our flags.
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    """Build the argparser. Extra Pi0.5 VL cotrain flags layer onto lerobot-train's flags.

    lerobot-train uses draccus / pydantic config parsing, NOT raw argparse, so
    most flags pass through to its config system unchanged. We define our extra
    flags as parser known_args and forward the rest.
    """
    p = argparse.ArgumentParser(
        description="Pi0.5 VL cotrain mixed-batch lerobot-train wrapper",
        # parse_known_args to forward unknown flags to lerobot-train.
    )

    # -- VL co-training (NEW) --
    p.add_argument("--vl_dataset.manifest", dest="vl_manifest", required=True,
                   help="HF repo id or local parquet path for VL pairs "
                        "(bbox-grounded face captions, 4 caption forms)")
    p.add_argument("--vl_ratio", dest="vl_ratio", type=int, default=10,
                   help="Number of robot batches per 1 VL batch. ObjectVLA "
                        "published default: 10. Do NOT improvise (see "
                        "the ObjectVLA spec risk).")

    # == Data filtering + per-episode sample weights ==
    p.add_argument("--dataset.episodes_file", dest="episodes_file", default=None,
                   help="Path to keep_episodes.txt (audit-filter). Comma-list "
                        "or newline-list of episode_idx ints to keep.")
    p.add_argument("--dataset.sample_weights", dest="sample_weights", default=None,
                   help="Path to hardneg_weights.npy (per-episode hard-negative weights).")
    p.add_argument("--dataset.curriculum_switch_step", dest="curriculum_switch",
                   type=int, default=0,
                   help="Step at which curriculum sampler flips phase 1 (easy "
                        "only) -> phase 2 (full distribution). 0 = no curriculum.")

    # == Per-layer LoRA rank config ==
    p.add_argument("--peft.layer_rank_config", dest="layer_rank_config",
                   default=None,
                   help="Path to layer_rank.json with per-layer LoRA "
                        "ranks. If unset, lerobot's default --peft.r applies "
                        "uniformly.")

    # == EMA shadow weights ==
    p.add_argument("--train.use_ema", dest="use_ema",
                   type=lambda s: s.lower() in {"true", "1", "yes"},
                   default=False)
    p.add_argument("--train.ema_alpha", dest="ema_alpha", type=float, default=0.999)

    return p


# -----------------------------------------------------------------------------
# VL pairs dataloader (mirrors the collator pattern from train_paligemma_vqa.py)
# -----------------------------------------------------------------------------

class VLPairsDataset(torch.utils.data.Dataset):
    """Loads bbox-grounded face VQA pairs from `HBOrtiz/so101_eval3_broad_grounding`.

    Verified manifest schema (the actual push, 2026-05-20):
        image_path        (str)   relative path inside the HF dataset
                                   e.g. "images/chunk-000/quick_lecun_LSO_ep02_..__f0107.jpg"
        prompt            (str)   "What is in this image?" / "Who is in the printed photo at [...]?"
                                   (NB: no <image> token prepended — collator adds it)
        target            (str)   "The printed photo of Barack Obama is at [0.06,...]" / "Barack Obama"
        bbox_xyxy_norm    (list)  [x1, y1, x2, y2] normalized [0,1] xyxy
        celeb_name        (str)   "Barack Obama"
        celeb_slug        (str)   "barack_obama"
        caption_type      (str)   "location_explicit" | "qa_grounded"
        episode           (str)   source teleop episode name
        frame_idx         (int)   source frame
        pid               (int)   portrait/face index (0=left, 1=middle, 2=right typically)

    Total rows: 176,670 across 29,445 unique images, 192 unique celebs.

    Image source: extracted from `images.tar.zst` inside the HF dataset, OR
    streamed via huggingface_hub's image-resolve API. We accept either an
    extracted directory (preferred — much faster) or auto-extract on first
    construction.
    """

    REQUIRED_COLS = {"image_path", "prompt", "target", "bbox_xyxy_norm",
                     "celeb_slug", "caption_type"}

    def __init__(self, manifest_path_or_id: str, processor,
                 image_root: Path | None = None,
                 max_text_len: int = 384,
                 image_token: str = "<image>"):
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required for VL dataset")

        # Accept either a local parquet OR an HF repo id. For HF, we expect
        # the dataset to contain `manifest.parquet` at the root.
        if Path(manifest_path_or_id).is_file():
            self.df = pd.read_parquet(manifest_path_or_id)
            dataset_root = Path(manifest_path_or_id).parent
        else:
            from huggingface_hub import snapshot_download, hf_hub_download
            import os
            # Pull just the manifest first (fast), defer image extraction.
            mf_path = hf_hub_download(
                repo_id=manifest_path_or_id, repo_type="dataset",
                filename="manifest.parquet",
                token=os.environ.get("HF_TOKEN"),
            )
            self.df = pd.read_parquet(mf_path)
            # The image root is the snapshot's dataset directory — pull on first use.
            # Integration point: pre-extract images.tar.zst before training starts
            # to avoid per-row decode latency. For smoke testing, snapshot_download
            # the whole repo (~1.15 GB) once.
            dataset_root = Path(mf_path).parent
            if image_root is None:
                # Try to find pre-extracted images/.
                if (dataset_root / "images").is_dir():
                    image_root = dataset_root / "images"
                else:
                    print("[WARN] VL dataset: expected pre-extracted images/ dir, "
                          f"got=missing under {dataset_root}, "
                          "fallback=snapshot_download full repo (1.15 GB)",
                          flush=True)
                    full = snapshot_download(
                        repo_id=manifest_path_or_id, repo_type="dataset",
                        token=os.environ.get("HF_TOKEN"),
                    )
                    image_root = Path(full)

        # Schema verification.
        missing = self.REQUIRED_COLS - set(self.df.columns)
        if missing:
            raise ValueError(f"VL manifest missing required columns: {missing}; "
                             f"got {list(self.df.columns)}")

        self.processor = processor
        self.image_root = image_root
        self.max_text_len = max_text_len
        self.image_token = image_token

        print(f"[vl_dataset] loaded {len(self.df)} VL pairs from "
              f"{manifest_path_or_id}", flush=True)
        print(f"[vl_dataset]   unique images: {self.df['image_path'].nunique()}",
              flush=True)
        print(f"[vl_dataset]   unique celebs: {self.df['celeb_slug'].nunique()}",
              flush=True)
        print(f"[vl_dataset]   caption mix:   "
              f"{dict(self.df['caption_type'].value_counts())}", flush=True)
        print(f"[vl_dataset]   image_root:    {self.image_root}", flush=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        row = self.df.iloc[idx]

        # image_path is relative; resolve via image_root.
        rel = Path(str(row["image_path"]))
        # Some snapshots put images under <root>/images/...; others put
        # them at <root>/... directly. Try both.
        candidates = [
            self.image_root / rel,
            self.image_root.parent / rel,  # in case image_root is .../images and rel starts with images/
        ]
        # Also strip a leading "images/" if image_root already points there.
        if str(rel).startswith("images/") and self.image_root.name == "images":
            candidates.insert(0, self.image_root / rel.relative_to("images"))

        img_path = next((c for c in candidates if c.is_file()), None)
        if img_path is None:
            print(f"[WARN] VL row {idx}: expected image at one of "
                  f"{[str(c) for c in candidates]}, got=missing, "
                  f"fallback=blank 224x224 gray", flush=True)
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        else:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] VL row {idx}: expected readable {img_path}, "
                      f"got {e}, fallback=blank", flush=True)
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        prompt = str(row["prompt"])
        # Prepend the PaliGemma <image> token if not already present.
        if self.image_token not in prompt:
            prompt = f"{self.image_token}{prompt}"

        return {
            "image": img,
            "prompt": prompt,
            "target": str(row["target"]),
            "celeb_slug": str(row["celeb_slug"]),
        }


def make_vl_collator(processor, max_text_len: int = 384):
    """Reuses the PaliGemmaProcessor pattern with suffix masking.

    See eval_3/scripts/warmstart/train_paligemma_vqa.py for the canonical
    version. Key: pass `suffix=target` so prompt tokens are masked in labels
    and only the celeb name contributes to the loss.
    """
    def collate(batch):
        """Process the (image, prompt, target) tuples into a PaliGemma-ready batch with suffix-masked labels."""
        images = [b["image"] for b in batch]
        prompts = [b["prompt"] for b in batch]
        targets = [b["target"] for b in batch]
        encoded = processor(
            text=prompts,
            images=images,
            suffix=targets,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_text_len,
        )
        return encoded
    return collate


# -----------------------------------------------------------------------------
# VQA loss path with dict-attention-mask fallback.
# -----------------------------------------------------------------------------

def pi05_vqa_loss(model, batch: dict, _fallback_state: dict | None = None) -> torch.Tensor:
    """Compute VQA CE loss on a VL batch.

    Primary path: HF model.forward() — internal loss via labels.
    Fallback (transformers ≥5.0 dict-mask risk): manual image-feature splice
    + direct language_model() call with tensor mask.

    Integration point: smoke-test gates the fallback. If primary path crashes
    with the dict-mask error, set _fallback_state["use_manual_splice"] = True
    and re-call this function.
    """
    state = _fallback_state if _fallback_state is not None else {}

    if not state.get("use_manual_splice", False):
        try:
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            return outputs.loss
        except (TypeError, AttributeError) as e:
            # The known signature of the dict-mask issue is a TypeError from
            # create_causal_mask(attention_mask=<dict>, ...). Log + fall through.
            if "create_causal_mask" in str(e) or "attention_mask" in str(e):
                print(f"[WARN] pi05_vqa_loss: expected=tensor mask path OK, "
                      f"got={type(e).__name__}: {e}, fallback=manual splice",
                      flush=True)
                state["use_manual_splice"] = True
                # Fall through to manual splice below.
            else:
                raise

    # Manual splice fallback for the dict-attention-mask issue.
    # Integration point: this requires reading the exact lerobot Pi0.5 wrapper
    # to know whether model.model is PaliGemmaWithExpertModel and how to access
    # vision/language submodules. Stub below — fill in once the training-VM env is online.
    raise NotImplementedError(
        "Dict-mask manual splice fallback not yet implemented. "
        "Will be filled in during the training-VM smoke test if the primary path crashes. "
        "Manual splice fallback not implemented in this scaffold."
    )


# -----------------------------------------------------------------------------
# Robot loss path (delegates to lerobot's standard flow-matching).
# -----------------------------------------------------------------------------

def pi05_flow_loss(policy, batch: dict) -> torch.Tensor:
    """Standard lerobot Pi0.5 flow-matching action loss. Unchanged from
    canonical lerobot-train's per-step forward."""
    # Integration point: lerobot's policy.forward returns (loss, output) tuple.
    # Confirm the exact return signature on the training-VM's lerobot version.
    result = policy.forward(batch)
    if isinstance(result, tuple):
        loss = result[0]
    elif hasattr(result, "loss"):
        loss = result.loss
    else:
        loss = result
    return loss


# -----------------------------------------------------------------------------
# == Per-layer LoRA rank config == applied at policy construction.
# -----------------------------------------------------------------------------

def apply_layer_wise_lora(policy, layer_rank_config_path: str):
    """Configure PEFT LoRA with per-layer ranks. Falls back to uniform if
    PEFT version doesn't support per-layer rank.

    Integration point: PEFT's LoraConfig accepts a single `r` value. Per-layer
    rank requires either (a) per-target-module rank dict via `rank_pattern=`,
    or (b) applying LoRA in two passes with different ranks. Implement on the training VM
    once the PEFT version is confirmed.
    """
    if not Path(layer_rank_config_path).is_file():
        print(f"[WARN] layer_rank_config: expected file at {layer_rank_config_path}, "
              f"got=missing, fallback=uniform r (lerobot --peft.r flag applies)",
              flush=True)
        return policy

    cfg = json.loads(Path(layer_rank_config_path).read_text())
    rank_map = cfg.get("layer_rank", {})
    target_modules = cfg.get("target_modules", [])
    print(f"[layer_rank] target_modules={target_modules}, "
          f"distinct ranks={sorted(set(rank_map.values()))}", flush=True)

    # PEFT's rank_pattern uses regex on the parameter NAME.
    # Pattern for Gemma-2B: "...language_model.layers.{i}.self_attn.{proj}_proj"
    # Build a rank_pattern dict: {regex: rank}
    rank_pattern = {}
    for layer_idx_str, r in rank_map.items():
        i = int(layer_idx_str)
        for tm in target_modules:
            # Regex matches the layer-i instance of this target module.
            rank_pattern[rf".*language_model\.layers\.{i}\..*\.{tm}$"] = int(r)

    # Integration point: pass rank_pattern to LoraConfig + reapply. The lerobot
    # train.py instantiates LoraConfig from --peft.* flags; we'd need to
    # monkey-patch the LoraConfig kwargs OR apply LoRA post-construction.
    # Stub for now; will integrate during the training-VM smoke test.
    print(f"[layer_rank] built rank_pattern with {len(rank_pattern)} entries. "
          f"integration needed to feed into LoraConfig.", flush=True)
    return policy


# -----------------------------------------------------------------------------
# == EMA shadow weights ==
# -----------------------------------------------------------------------------

class EMAShadow:
    """α-EMA of trainable parameters. Step updates internal shadow; can swap
    in for inference. Standard SGD stability technique."""

    def __init__(self, model, alpha: float = 0.999):
        self.alpha = alpha
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.detach().clone()
        print(f"[ema] tracking {len(self.shadow)} trainable tensors at α={alpha}",
              flush=True)

    @torch.no_grad()
    def update(self, model) -> None:
        """Update the EMA shadow with the current trainable parameter values via shadow := alpha*shadow + (1-alpha)*param."""
        a = self.alpha
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.data.detach().clone()
                continue
            self.shadow[name].mul_(a).add_(p.data, alpha=1 - a)

    def save(self, path: Path) -> None:
        """Serialize the EMA shadow dict to `path` (parents created)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.shadow, path)
        print(f"[ema] shadow saved to {path}", flush=True)


# -----------------------------------------------------------------------------
# Main entry point — wraps lerobot.scripts.train.train().
# -----------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    """Parse the wrapper's extra flags, then forward the rest to lerobot-train (scaffold: integration points only)."""
    parser = build_argparser()
    pi05_vl_cotrain_args, lerobot_args = parser.parse_known_args(argv)
    print(f"[pi05_vl_cotrain] extra flags: {vars(pi05_vl_cotrain_args)}", flush=True)
    print(f"[pi05_vl_cotrain] forwarding {len(lerobot_args)} flags to lerobot-train", flush=True)

    # Integration point: lerobot's entry point is lerobot.scripts.train.train()
    # via draccus config. We need to:
    #   1. Build the lerobot config from lerobot_args (draccus-parsed).
    #   2. Construct the policy + optimizer + dataloaders.
    #   3. Apply our enhancements (layer-rank LoRA, EMA, curriculum).
    #   4. Build the VL dataloader (VLPairsDataset + make_vl_collator).
    #   5. Replace lerobot's training loop with our modulo-based alternation.
    #
    # This requires importing the lerobot training infrastructure and
    # overriding/wrapping its train() function. The exact integration
    # depends on the lerobot version on the training VM — see the smoke test plan in
    # the ObjectVLA spec.

    print("\n[pi05_vl_cotrain] ---- INTEGRATION POINTS (scaffold) ----")
    print("[pi05_vl_cotrain] This wrapper is a scaffold; the lerobot-train training-loop hook")
    print("[pi05_vl_cotrain] integration is BLOCKED on:")
    print("[pi05_vl_cotrain]   1. the VL manifest schema (locks VLPairsDataset columns)")
    print("[pi05_vl_cotrain]   2. the training VM lerobot+peft+transformers version check")
    print("[pi05_vl_cotrain]   3. 200-step smoke test (Task #13) to verify VQA forward")
    pass
    print("[pi05_vl_cotrain] for the order of integration work.")
    print()
    print("[pi05_vl_cotrain] Components ready for integration:")
    print(f"[pi05_vl_cotrain]   - VLPairsDataset class (collator stub: {make_vl_collator.__name__})")
    print(f"[pi05_vl_cotrain]   - pi05_vqa_loss + dict-mask fallback hook")
    print(f"[pi05_vl_cotrain]   - pi05_flow_loss (delegates to lerobot)")
    print(f"[pi05_vl_cotrain]   - apply_layer_wise_lora (rank_pattern builder)")
    print(f"[pi05_vl_cotrain]   - EMAShadow class")
    print()
    print("[pi05_vl_cotrain] To complete: import lerobot.scripts.train, wrap its train()")
    print("[pi05_vl_cotrain] function with our hooks, run smoke on the training VM.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
