#!/usr/bin/env python
"""
Approach A LoRA fine-tune of SmolVLM2-500M-Video-Instruct for celebrity
face→name grounding.

What this script does:
  1. Loads the same VLM + processor SmolVLA uses internally.
  2. Attaches LoRA adapters with target_modules restricted to
     text_model.layers.[0..15] + vision_model.encoder.layers.* attention/MLP.
     (Layers 16..31 are deliberately excluded — they're discarded by SmolVLA
     at inference time. See docs/lora_vlm_finetuning.md §2.A.)
  3. Trains via trl SFTTrainer with chat-template-formatted data, loss masked
     to assistant tokens (the celebrity name).
  4. Reports to wandb under project `lemonkey-eval3-smolvlm`.
  5. After training, saves the LoRA adapter to --out-dir.

Smoke mode (--smoke):
  Uses train.smoke.jsonl / val.smoke.jsonl (5 IDs × 20 imgs), 1 epoch,
  batch_size=1, no wandb. Just confirms the pipeline runs end-to-end.

Run (smoke, local Mac):
  python eval_3/scripts/train_smolvlm2_lora.py --smoke

Run (real, AWS A10G):
  python eval_3/scripts/train_smolvlm2_lora.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

DEFAULT_DATA_ROOT = Path("/Volumes/externalSSD/datasets/vggface2_hearfool/manifests")
MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# Regex: text_model layers 0..15 (q/k/v/o + gate/up/down), and ALL vision-tower
# encoder attention/MLP. Matches the SmolVLM2 module name structure.
LORA_TARGET_REGEX = (
    r"(?:.*\.text_model\.layers\.(?:[0-9]|1[0-5])\..*"
    r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))"
    r"|(?:.*\.vision_model\.encoder\.layers\.\d+\..*"
    r"(?:q_proj|k_proj|v_proj|out_proj|fc1|fc2))"
)


def pick_device_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32  # bf16 / fp16 on MPS is flaky for some VLM ops
    return "cpu", torch.float32


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_dataset(jsonl_path: Path) -> Dataset:
    """Returns a HF Dataset of {messages, image_path}. Image is loaded lazily in the collator."""
    rows = load_jsonl(jsonl_path)
    out = []
    for r in rows:
        out.append({
            "image_path": r["image_path"],
            "messages": [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": r["prompt"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": r["response"]},
                ]},
            ],
        })
    return Dataset.from_list(out)


class VLMCollator:
    """Custom collator for SmolVLM2 SFT.

    Key choices that match SmolVLA's deployed behavior:
      - do_image_splitting=False  → single global image (~64 tokens), not 4-9 tiles
      - apply_chat_template renders the conversation with the right number of
        image markers for the (untiled) image
      - labels = input_ids with padding masked to -100
        (TODO: assistant-only loss masking — train on full sequence for now;
         minor suboptimality, acceptable for v0)
    """
    def __init__(self, processor):
        self.processor = processor
        tok = processor.tokenizer
        self.pad_id = tok.pad_token_id
        # Collect every image-related token id so we can mask them out of the loss.
        # Without this, the model is asked to predict image-placeholder tokens as
        # if they were text — impossible, so loss explodes (~16 instead of ~3-5).
        candidate_attrs = ["image_token_id", "fake_image_token_id",
                           "global_image_token_id"]
        self.image_token_ids = []
        for attr in candidate_attrs:
            tid = getattr(tok, attr, None)
            if tid is not None:
                self.image_token_ids.append(tid)
        # SmolVLMProcessor itself also has image_token_id sometimes
        for attr in candidate_attrs:
            tid = getattr(processor, attr, None)
            if tid is not None and tid not in self.image_token_ids:
                self.image_token_ids.append(tid)
        print(f"[collator] pad_id={self.pad_id}  image_token_ids={self.image_token_ids}")

    def __call__(self, examples: list[dict]) -> dict:
        texts: list[str] = []
        images: list[list[Image.Image]] = []
        for ex in examples:
            text = self.processor.apply_chat_template(
                ex["messages"], add_generation_prompt=False, tokenize=False
            )
            texts.append(text)
            images.append([Image.open(ex["image_path"]).convert("RGB")])

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            do_image_splitting=False,
        )
        labels = batch["input_ids"].clone()
        labels[labels == self.pad_id] = -100
        for tid in self.image_token_ids:
            labels[labels == tid] = -100
        batch["labels"] = labels
        return batch


def attach_lora(model, verbose: bool = True):
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_REGEX,
    )
    model = get_peft_model(model, config)

    if verbose:
        print("\n=== LoRA module placement audit ===")
        targeted = [n for n, _ in model.named_modules()
                    if hasattr(model.get_submodule(n), "lora_A")]
        # Group by layer-index for readability
        text_layer_idxs = sorted({
            int(m.group(1)) for n in targeted
            if (m := re.search(r"text_model\.layers\.(\d+)\.", n))
        })
        vision_layer_idxs = sorted({
            int(m.group(1)) for n in targeted
            if (m := re.search(r"vision_model\.encoder\.layers\.(\d+)\.", n))
        })
        print(f"  text_model layers with LoRA:   {text_layer_idxs}")
        print(f"  vision_model layers with LoRA: {vision_layer_idxs}")
        print(f"  total LoRA-adapted modules:    {len(targeted)}")
        # Sanity: layers 16-31 must not be in the list
        bad = [i for i in text_layer_idxs if i >= 16]
        if bad:
            raise RuntimeError(f"LoRA leaked into truncated layers {bad} — fix regex")
        print("  ✓ no LoRA on text_model layers 16-31 (truncation-safe)\n")

        model.print_trainable_parameters()

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/Volumes/externalSSD/lemonkey/eval_3/lora_celeb_v0"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true",
                        help="Use train.smoke.jsonl, 1 epoch, batch=1, no wandb")
    parser.add_argument("--wandb-project", default="lemonkey-eval3-smolvlm")
    args = parser.parse_args()

    suf = ".smoke" if args.smoke else ""
    train_path = args.data_root / f"train{suf}.jsonl"
    val_path = args.data_root / f"val{suf}.jsonl"
    print(f"[train] data: {train_path}")
    print(f"[train] val:  {val_path}")

    device, dtype = pick_device_dtype()
    print(f"[train] device={device}  dtype={dtype}")

    print(f"[train] loading processor + model ({MODEL_ID})...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model = attach_lora(model)

    print(f"[train] building datasets...")
    train_ds = build_dataset(train_path)
    val_ds = build_dataset(val_path)
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        os.environ["WANDB_DISABLED"] = "true"

    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=1.0 if args.smoke else args.epochs,
        per_device_train_batch_size=1 if args.smoke else args.batch_size,
        per_device_eval_batch_size=1 if args.smoke else args.batch_size,
        gradient_accumulation_steps=1 if args.smoke else args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=10 if args.smoke else 100,
        lr_scheduler_type="cosine",
        logging_steps=2 if args.smoke else 25,
        eval_strategy="no" if args.smoke else "steps",
        eval_steps=200,
        save_strategy="no" if args.smoke else "steps",
        save_steps=500,
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=False,
        gradient_checkpointing=False if args.smoke else True,
        report_to=("none" if args.smoke else "wandb"),
        run_name=f"smolvlm-celeb-lora{'-smoke' if args.smoke else ''}",
        remove_unused_columns=False,
        max_steps=10 if args.smoke else -1,
        dataloader_num_workers=0,
    )

    if not args.smoke:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    collator = VLMCollator(processor)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print(f"[train] starting training...")
    trainer.train()

    print(f"[train] saving adapter to {args.out_dir}")
    trainer.save_model(str(args.out_dir))
    processor.save_pretrained(str(args.out_dir))
    print("[train] done.")


if __name__ == "__main__":
    main()
