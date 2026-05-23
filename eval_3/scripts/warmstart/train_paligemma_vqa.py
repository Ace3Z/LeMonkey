#!/usr/bin/env python3
"""Warm-start PaliGemma inside lerobot/pi05_base via LoRA on VGGFace2 VQA.

Goal: teach the VLM (PaliGemma 2B + SigLIP-So400m + LM head) to do face-identity
naming as a CAUSAL_LM task, so that a subsequent Pi0.5 VLA fine-tune (Pi0.5
re-launch) inherits a celeb-aware prior instead of relying on PaliGemma's
DLP-filtered WebLI pretrain.

ARCHITECTURE — what's trainable
================================

We load lerobot/pi05_base (full Pi0.5: PaliGemma 2B + Gemma-300M action expert).
We then:
  1. Freeze vision_tower, multi_modal_projector, lm_head (small head, LoRA on
     it adds complexity for marginal benefit).
  2. Freeze the whole Gemma-300M action expert — irrelevant for VQA.
  3. Apply LoRA (r=32, alpha=64, dropout=0.05) to PaliGemma's language_model
     on q/k/v/o + gate/up/down projections — same target_modules as Pi0.5
     so adapters align with the architecture used at Pi0.5 re-train.

ARCHITECTURE — what's evaluated
================================

PaliGemmaForConditionalGenerationWithPiGemma (lerobot's subclass) inherits HF's
standard PaliGemma forward signature. Its only override is PiGemmaModel as the
LM decoder, which adds adarms-conditioned RMSNorm + gated residuals. When
adarms_cond is None (no action expert in the loop, which is our case for VQA),
these reduce to standard Gemma behavior. So we can drive it with the standard
HF VQA loss path: pass (pixel_values, input_ids, attention_mask, labels) and
let `model.forward()` compute the loss internally.

KNOWN RISK — verify in smoke test
==================================

`PiGemmaModel.forward` takes `attention_mask: torch.Tensor` (a single tensor).
In transformers ≥5.0, the parent `PaliGemmaModel.forward` builds a dict of
attention masks (one per attention-type tag) and passes it to the LM. If your
transformers version does this, the inner `create_causal_mask(attention_mask=
<dict>, ...)` will raise. The fix is to add a `compute_loss_func` override
that does manual image-feature splicing + calls `language_model(inputs_embeds=
..., attention_mask=<tensor>)` directly. The `--smoke` mode below catches this
in <5 minutes — ALWAYS run smoke before the 10-hour production launch.

DATA — what's expected
=======================

A parquet manifest at MANIFEST_PATH with columns:
    image_path, prompt, target, identity_id, source
produced by `prepare_vggface2_vqa.py`. We load it as a HF Dataset, then a
collator reads images on the fly + invokes PaliGemmaProcessor with `suffix=`
(which masks out prompt tokens in the labels — only the celeb name contributes
to the loss).

OUTPUT
======

At end of training:
  1. Merges LoRA adapters into the PaliGemma submodule (peft .merge_and_unload).
  2. Plugs the merged PaliGemma back into the full Pi0.5 model.
  3. Saves the full Pi0.5 checkpoint to OUT_DIR and pushes to PUSH_REPO.

The pushed checkpoint is then a valid `--policy.pretrained_path` for a Pi0.5
re-launch.

USAGE
=====

    python eval_3/scripts/warmstart/train_paligemma_vqa.py \\
        --manifest /shared/vggface2_vqa_train.parquet \\
        --output-dir outputs/paligemma_celeb_warm \\
        --push-repo HBOrtiz/pi05_paligemma_celeb_warm \\
        --batch-size 8 --grad-accum 4 --epochs 1 --lr 1e-5

"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path

import torch
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True,
                     help="Parquet manifest from prepare_vggface2_vqa.py")
    ap.add_argument("--pretrained-pi05", default="lerobot/pi05_base",
                     help="Pi0.5 base HF repo or local path (default: lerobot/pi05_base)")
    ap.add_argument("--processor-name", default="google/paligemma2-3b-pt-224",
                     help="HF PaliGemma processor (tokenizer + image processor)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--push-repo", default=None,
                     help="HF Hub repo to push the merged warmed Pi0.5 to (e.g. HBOrtiz/pi05_paligemma_celeb_warm)")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", nargs="+",
                     default=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                     help="LoRA target modules — same as Pi0.5 for alignment")
    ap.add_argument("--epochs", type=int, default=1,
                     help="VGGFace2 is large; 1 epoch is usually plenty")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4,
                     help="Effective batch = batch_size * grad_accum")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-text-len", type=int, default=384,
                     help="Truncate prompt+target to this many tokens. PaliGemma 2 "
                          "prepends 256 image tokens + bos; prompt + name adds ~30 more. "
                          "Keep >= 320 to avoid truncating image tokens (which causes "
                          "ValueError: Mismatch in image token count). Default 384.")
    ap.add_argument("--logging-steps", type=int, default=25)
    ap.add_argument("--save-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                     help="Use only 200 manifest rows + skip push, for smoke testing")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load the Pi0.5 base model (full, including action expert) ────────
    print("==> [1/6] loading Pi0.5 base from", args.pretrained_pi05, flush=True)
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(args.pretrained_pi05)
    policy.eval()  # bn/dropout off; LoRA adapters re-enable train mode for themselves

    # Cast to bf16 for memory (vision tower stays fp32 per pi05's mixed-precision plan)
    policy.model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    paligemma = policy.model.paligemma_with_expert.paligemma  # PaliGemmaForConditionalGenerationWithPiGemma

    # ── 2. Freeze everything we don't want to train ─────────────────────────
    print("==> [2/6] freezing vision_tower, projector, lm_head, action expert", flush=True)
    for p in paligemma.model.vision_tower.parameters():
        p.requires_grad = False
    for p in paligemma.model.multi_modal_projector.parameters():
        p.requires_grad = False
    for p in paligemma.lm_head.parameters():
        p.requires_grad = False
    # Action expert + projections are owned by paligemma_with_expert; freeze:
    for p in policy.model.paligemma_with_expert.gemma_expert.parameters():
        p.requires_grad = False

    # ── 3. Apply LoRA to PaliGemma's language_model layers ──────────────────
    print(f"==> [3/6] applying LoRA r={args.lora_r} alpha={args.lora_alpha} "
          f"targets={args.target_modules}", flush=True)
    from peft import LoraConfig, get_peft_model, TaskType
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    # We wrap just the inner paligemma — the action expert / wrapper stays as-is.
    paligemma = get_peft_model(paligemma, peft_config)
    paligemma.print_trainable_parameters()
    # Re-attach into the policy so its forward goes through the wrapped model
    policy.model.paligemma_with_expert.paligemma = paligemma

    # ── 4. Build dataset + collator ─────────────────────────────────────────
    print("==> [4/6] loading manifest + processor", flush=True)
    from datasets import load_dataset
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor_name)

    ds = load_dataset("parquet", data_files=str(args.manifest), split="train")
    if args.smoke:
        ds = ds.shuffle(seed=args.seed).select(range(200))
        print(f"==> SMOKE: trimmed dataset to {len(ds)} rows", flush=True)
    print(f"manifest rows: {len(ds)}", flush=True)

    def collate(batch):
        """Returns the dict PaliGemma's forward expects:
            input_ids, attention_mask, pixel_values, labels
        Uses `suffix=target` so the processor masks prompt tokens (sets label=-100)
        and only the celeb-name tokens contribute to CE loss.

        On image-read failure, skips the row and emits [WARN]. If the whole
        batch fails, falls back to duplicating any one good row from the
        previous batch (caller has to handle the all-failed case — but with
        VGGFace2 + valid manifest this should never actually trigger).
        """
        images = []
        prompts = []
        suffixes = []
        for ex in batch:
            try:
                img = Image.open(ex["image_path"]).convert("RGB")
            except Exception as e:
                print(f"[WARN] image_read_fail: path={ex['image_path']}, "
                      f"err={e}, fallback=skip-row", flush=True)
                continue
            images.append(img)
            prompts.append(ex["prompt"])
            suffixes.append(ex["target"])
        if not images:
            # Whole batch unreadable. Fabricate a 1-row batch from the first
            # manifest row so Trainer keeps moving. Hard-fails if the manifest
            # itself is broken (then we want to crash early anyway).
            print(f"[WARN] collator_whole_batch_failed: expected=>=1 readable image, "
                  f"got=0, fallback=fabricate-from-manifest[0]", flush=True)
            ex0 = ds[0]
            img = Image.open(ex0["image_path"]).convert("RGB")
            images = [img]
            prompts = [ex0["prompt"]]
            suffixes = [ex0["target"]]
        out = processor(
            text=prompts,
            images=images,
            suffix=suffixes,
            return_tensors="pt",
            padding="longest",
            truncation="only_first",
            max_length=args.max_text_len,
        )
        return out

    # ── 5. Train ────────────────────────────────────────────────────────────
    print("==> [5/6] training", flush=True)
    from transformers import Trainer, TrainingArguments

    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,    # critical: keep image_path etc. for the collator
        seed=args.seed,
        dataloader_num_workers=4,
        report_to=[],                    # no wandb; logs go to stdout
        push_to_hub=False,                # we'll push the spliced-back pi05 manually below
    )

    trainer = Trainer(
        model=paligemma,
        args=targs,
        train_dataset=ds,
        data_collator=collate,
    )
    trainer.train()

    # ── 6. Merge adapters + splice back into Pi0.5 + push ───────────────────
    print("==> [6/6] merging adapters + saving full Pi0.5 + pushing", flush=True)
    paligemma_merged = paligemma.merge_and_unload()
    policy.model.paligemma_with_expert.paligemma = paligemma_merged

    save_dir = args.output_dir / "pi05_warmed"
    policy.save_pretrained(str(save_dir))
    print(f"saved merged Pi0.5 with warmed PaliGemma -> {save_dir}", flush=True)

    if args.push_repo and not args.smoke:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.push_repo, repo_type="model", exist_ok=True, private=False)
        policy.push_to_hub(args.push_repo)
        print(f"pushed -> https://huggingface.co/{args.push_repo}", flush=True)
    elif args.smoke:
        print("[SMOKE] skipping push to hub", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
