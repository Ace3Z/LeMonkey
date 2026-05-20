#!/usr/bin/env python3
"""Track 2 — mixed-batch lerobot-train with ObjectVLA VL co-training.

Extends lerobot-train with:
  - 10:1 robot:VL batch alternation (ObjectVLA arxiv 2502.11550, +45pp OOD)
  - per-layer LoRA rank (Enhancement B-4)
  - curriculum sampler switch at step N (Enhancement B-5)
  - EMA shadow weights (Enhancement B-7)

VQA forward path reuses Roham's pattern from
`eval_3/scripts/warmstart/train_paligemma_vqa.py` —
PaliGemmaProcessor with suffix masking, standard HF model.forward()
internal loss. Falls back to manual image-feature splicing if
transformers ≥5.0 dict-attention-mask hits (TRACK_B_WARMSTART.md §6).

Per CLAUDE.md §5: every fallback emits [WARN] with context.
Per CLAUDE.md §6: no Claude attribution in any artifact this script produces.

USAGE
=====

Standard invocation goes through eval_3/scripts/brev/run_training_track_2.sh.
For ad-hoc smoke testing:

    python eval_3/scripts/track_2/lerobot_train_with_vl_cotrain.py \\
        --policy.type=pi05 \\
        --policy.pretrained_path=HBOrtiz/pi05_paligemma_celeb_warm \\
        --dataset.repo_id=HBOrtiz/so101_eval3_aug_v3_200celebs \\
        --vl_dataset.manifest=HBOrtiz/eval3_objectvla_vl_pairs \\
        --vl_ratio=10 \\
        --batch_size=8 --steps=200    # smoke

STATUS
======

Scaffolded 2026-05-20. Brev-side integration testing pending.
INTEGRATION POINTS marked with [BREV_INTEGRATE] need:
  - Roham's VL manifest schema (to lock the collator's column reads)
  - lerobot version-specific API (training loop hook points may differ)
  - transformers version on brev_instance2 (dict-mask risk gate)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


# -----------------------------------------------------------------------------
# Configuration parsing — extends lerobot's argparse with our flags.
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    """Build the argparser. Extra Track 2 flags layer onto lerobot-train's flags.

    lerobot-train uses draccus / pydantic config parsing, NOT raw argparse, so
    most flags pass through to its config system unchanged. We define our extra
    flags as parser known_args and forward the rest.
    """
    p = argparse.ArgumentParser(
        description="Track 2 mixed-batch lerobot-train wrapper",
        # parse_known_args to forward unknown flags to lerobot-train.
    )

    # -- VL co-training (NEW) --
    p.add_argument("--vl_dataset.manifest", dest="vl_manifest", required=True,
                   help="HF repo id or local parquet path for VL pairs "
                        "(bbox-grounded face captions, 4 caption forms)")
    p.add_argument("--vl_ratio", dest="vl_ratio", type=int, default=10,
                   help="Number of robot batches per 1 VL batch. ObjectVLA "
                        "published default: 10. Do NOT improvise (see "
                        "TRACK_OBJECTVLA.md §5 risk).")

    # -- Enhancement B-2/B-3 (data prep artifacts) --
    p.add_argument("--dataset.episodes_file", dest="episodes_file", default=None,
                   help="Path to keep_episodes.txt (B-2 filter). Comma-list "
                        "or newline-list of episode_idx ints to keep.")
    p.add_argument("--dataset.sample_weights", dest="sample_weights", default=None,
                   help="Path to hardneg_weights.npy (B-3 per-episode weights).")
    p.add_argument("--dataset.curriculum_switch_step", dest="curriculum_switch",
                   type=int, default=0,
                   help="Step at which curriculum sampler flips phase 1 (easy "
                        "only) → phase 2 (full distribution). 0 = no curriculum.")

    # -- Enhancement B-4 (per-layer LoRA rank) --
    p.add_argument("--peft.layer_rank_config", dest="layer_rank_config",
                   default=None,
                   help="Path to layer_rank_track2.json with per-layer LoRA "
                        "ranks. If unset, lerobot's default --peft.r applies "
                        "uniformly.")

    # -- Enhancement B-7 (EMA) --
    p.add_argument("--train.use_ema", dest="use_ema",
                   type=lambda s: s.lower() in {"true", "1", "yes"},
                   default=False)
    p.add_argument("--train.ema_alpha", dest="ema_alpha", type=float, default=0.999)

    return p


# -----------------------------------------------------------------------------
# VL pairs dataloader (mirrors Roham's collator pattern from train_paligemma_vqa.py)
# -----------------------------------------------------------------------------

class VLPairsDataset(torch.utils.data.Dataset):
    """Loads Darius's bbox-grounded face VQA pairs from a parquet manifest.

    Expected manifest schema (per TRACK_OBJECTVLA.md §2 Darius's deliverable):
        image_path   (str)   path to face crop (relative to dataset root)
        prompt       (str)   e.g. "Who is the person at [0.21,0.32,0.58,0.74]?"
        target       (str)   e.g. "Yann LeCun"
        bbox         (list)  [x1, y1, x2, y2] normalized [0,1] xyxy
        celeb        (str)   slug like "yann_lecun"
        caption_type (str)   "location_explicit" | "qa_grounded" | "qa_open" | "caption"

    [BREV_INTEGRATE]: confirm exact column names with Darius once his
    push lands. Adjust __getitem__ accordingly.
    """

    def __init__(self, manifest_path_or_id: str, processor, image_root: Path | None = None,
                 max_text_len: int = 384):
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pandas required for VL dataset")

        # Accept either local parquet path OR HF repo id.
        if Path(manifest_path_or_id).is_file():
            self.df = pd.read_parquet(manifest_path_or_id)
        else:
            # [BREV_INTEGRATE]: snapshot_download from HF when this is a repo_id.
            from huggingface_hub import snapshot_download
            local = snapshot_download(repo_id=manifest_path_or_id, repo_type="dataset")
            # Locate the parquet in the snapshot.
            parquets = list(Path(local).rglob("*.parquet"))
            if not parquets:
                raise FileNotFoundError(f"No parquet in {local}")
            self.df = pd.read_parquet(parquets[0])

        self.processor = processor
        self.image_root = image_root
        self.max_text_len = max_text_len

        # Sanity: required columns present?
        required = {"image_path", "prompt", "target"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"VL manifest missing columns: {missing}; "
                             f"got {list(self.df.columns)}")
        print(f"[vl_dataset] loaded {len(self.df)} VL pairs", flush=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        if self.image_root is not None and not img_path.is_absolute():
            img_path = self.image_root / img_path
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] VL row {idx}: expected image at {img_path}, "
                  f"got {e}, fallback=skip (returning blank)", flush=True)
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        return {
            "image": img,
            "prompt": str(row["prompt"]),
            "target": str(row["target"]),
        }


def make_vl_collator(processor, max_text_len: int = 384):
    """Reuses Roham's PaliGemmaProcessor pattern with suffix masking.

    See eval_3/scripts/warmstart/train_paligemma_vqa.py for the canonical
    version. Key: pass `suffix=target` so prompt tokens are masked in labels
    and only the celeb name contributes to the loss.
    """
    def collate(batch):
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

    [BREV_INTEGRATE]: smoke-test gates the fallback. If primary path crashes
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

    # Manual splice fallback per TRACK_B_WARMSTART.md §6.
    # [BREV_INTEGRATE]: this requires reading the exact lerobot Pi0.5 wrapper
    # to know whether model.model is PaliGemmaWithExpertModel and how to access
    # vision/language submodules. Stub below — fill in once Brev env is online.
    raise NotImplementedError(
        "Dict-mask manual splice fallback not yet implemented. "
        "Will be filled in during the Brev smoke test if the primary path crashes. "
        "See TRACK_B_WARMSTART.md §6 for the splice pattern."
    )


# -----------------------------------------------------------------------------
# Robot loss path (delegates to lerobot's standard flow-matching).
# -----------------------------------------------------------------------------

def pi05_flow_loss(policy, batch: dict) -> torch.Tensor:
    """Standard lerobot Pi0.5 flow-matching action loss. Unchanged from
    canonical lerobot-train's per-step forward."""
    # [BREV_INTEGRATE]: lerobot's policy.forward returns (loss, output) tuple.
    # Confirm the exact return signature on the Brev env's lerobot version.
    result = policy.forward(batch)
    if isinstance(result, tuple):
        loss = result[0]
    elif hasattr(result, "loss"):
        loss = result.loss
    else:
        loss = result
    return loss


# -----------------------------------------------------------------------------
# Layer-wise LoRA rank (Enhancement B-4) — applied at policy construction.
# -----------------------------------------------------------------------------

def apply_layer_wise_lora(policy, layer_rank_config_path: str):
    """Configure PEFT LoRA with per-layer ranks. Falls back to uniform if
    PEFT version doesn't support per-layer rank.

    [BREV_INTEGRATE]: PEFT's LoraConfig accepts a single `r` value. Per-layer
    rank requires either (a) per-target-module rank dict via `rank_pattern=`,
    or (b) applying LoRA in two passes with different ranks. Implement on Brev
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

    # [BREV_INTEGRATE]: pass rank_pattern to LoraConfig + reapply. The lerobot
    # train.py instantiates LoraConfig from --peft.* flags; we'd need to
    # monkey-patch the LoraConfig kwargs OR apply LoRA post-construction.
    # Stub for now — will integrate during Brev smoke.
    print(f"[layer_rank] built rank_pattern with {len(rank_pattern)} entries. "
          f"[BREV_INTEGRATE] needed to feed into LoraConfig.", flush=True)
    return policy


# -----------------------------------------------------------------------------
# EMA shadow weights (Enhancement B-7).
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
        a = self.alpha
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.data.detach().clone()
                continue
            self.shadow[name].mul_(a).add_(p.data, alpha=1 - a)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.shadow, path)
        print(f"[ema] shadow saved to {path}", flush=True)


# -----------------------------------------------------------------------------
# Main entry point — wraps lerobot.scripts.train.train().
# -----------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    parser = build_argparser()
    track2_args, lerobot_args = parser.parse_known_args(argv)
    print(f"[track_2] extra flags: {vars(track2_args)}", flush=True)
    print(f"[track_2] forwarding {len(lerobot_args)} flags to lerobot-train", flush=True)

    # [BREV_INTEGRATE]: lerobot's entry point is lerobot.scripts.train.train()
    # via draccus config. We need to:
    #   1. Build the lerobot config from lerobot_args (draccus-parsed).
    #   2. Construct the policy + optimizer + dataloaders.
    #   3. Apply our enhancements (layer-rank LoRA, EMA, curriculum).
    #   4. Build the VL dataloader (VLPairsDataset + make_vl_collator).
    #   5. Replace lerobot's training loop with our modulo-based alternation.
    #
    # This requires importing the lerobot training infrastructure and
    # overriding/wrapping its train() function. The exact integration
    # depends on the lerobot version on Brev — see the smoke test plan in
    # TRACK_OBJECTVLA.md §4.

    print("\n[track_2] ---- INTEGRATION POINTS PENDING BREV SMOKE TEST ----")
    print("[track_2] This wrapper is scaffolded but the lerobot-train hook")
    print("[track_2] integration is BLOCKED on:")
    print("[track_2]   1. Darius's VL manifest schema (locks VLPairsDataset columns)")
    print("[track_2]   2. brev_instance2 lerobot+peft+transformers version check")
    print("[track_2]   3. 200-step smoke test (Task #13) to verify VQA forward")
    print("[track_2] See eval_3/tracks/TRACK_OBJECTVLA_ENHANCED.md §D Gantt")
    print("[track_2] for the order of integration work.")
    print()
    print("[track_2] Components ready for integration:")
    print(f"[track_2]   - VLPairsDataset class (collator stub: {make_vl_collator.__name__})")
    print(f"[track_2]   - pi05_vqa_loss + dict-mask fallback hook")
    print(f"[track_2]   - pi05_flow_loss (delegates to lerobot)")
    print(f"[track_2]   - apply_layer_wise_lora (rank_pattern builder)")
    print(f"[track_2]   - EMAShadow class")
    print()
    print("[track_2] To complete: import lerobot.scripts.train, wrap its train()")
    print("[track_2] function with our hooks, run smoke on brev_instance2.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
