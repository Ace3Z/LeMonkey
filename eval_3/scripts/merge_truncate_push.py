#!/usr/bin/env python
"""
Merge a LoRA adapter into base SmolVLM2-500M, truncate the text model to the
first 16 layers (matching SmolVLA's expected backbone), and push to HF Hub.

The output model is a drop-in replacement for SmolVLM2 inside SmolVLA — loaded
via `lerobot-train --policy.vlm_model_name=<your-repo>`.

Run (on the H100, after the adapter you want is trained):
    python eval_3/scripts/merge_truncate_push.py \
        --adapter $EVAL3_ROOT/lora_celeb_r256_e10 \
        --hf-repo HansOrtiz/smolvlm2_lora_celebs \
        --local-out $EVAL3_ROOT/smolvlm2_lora_celebs_merged

Use --no-push to skip HF upload (debug locally first).
Use --no-truncate to keep all 32 text layers (debug).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TRUNCATE_TO = 16   # SmolVLA uses first 16 text-model layers; layers 16-31 are discarded


def find_text_layers(model) -> tuple[torch.nn.ModuleList, str]:
    """Return (layers_module_list, dotted_attr_path).

    SmolVLM2's exact attribute path can vary by transformers version. Try the
    common ones in order. Returns the ModuleList plus a human-readable path so
    we can print it for verification.
    """
    candidates = [
        ("model.text_model.layers",           lambda m: m.model.text_model.layers),
        ("model.text_model.model.layers",     lambda m: m.model.text_model.model.layers),
        ("model.model.text_model.layers",     lambda m: m.model.model.text_model.layers),
        ("text_model.layers",                 lambda m: m.text_model.layers),
        ("text_model.model.layers",           lambda m: m.text_model.model.layers),
    ]
    for path, accessor in candidates:
        try:
            layers = accessor(model)
            if isinstance(layers, torch.nn.ModuleList):
                return layers, path
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not locate text_model layers via any known path. "
        "Inspect `model` structure and add the right path to find_text_layers()."
    )


def truncate_text_model(model, keep_n: int) -> None:
    """Delete text-model layers past `keep_n` (in-place) and update config."""
    layers, path = find_text_layers(model)
    n_before = len(layers)
    if keep_n >= n_before:
        print(f"[truncate] already <= {keep_n} layers ({n_before}); no-op")
        return
    # ModuleList supports slice deletion
    del layers[keep_n:]
    n_after = len(layers)
    # Update config so subsequent loads know the new layer count
    if hasattr(model.config, "text_config"):
        model.config.text_config.num_hidden_layers = n_after
    print(f"[truncate] {path}: {n_before} → {n_after} layers")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True,
                        help="Path to LoRA adapter dir (e.g. lora_celeb_r256_e10)")
    parser.add_argument("--hf-repo", type=str, required=True,
                        help="HF repo id to push to (e.g. HansOrtiz/smolvlm2_lora_celebs). "
                             "Created as private; flip via HF web UI later.")
    parser.add_argument("--local-out", type=Path, required=True,
                        help="Local dir to save merged model before push")
    parser.add_argument("--no-truncate", action="store_true",
                        help="Skip layer truncation (keep all 32 text layers)")
    parser.add_argument("--no-push", action="store_true",
                        help="Save locally only; skip HF upload")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Push as private repo (default: True)")
    parser.add_argument("--commit-msg", type=str, default="merge LoRA, truncate to 16 layers")
    args = parser.parse_args()

    print(f"[load] base={MODEL_ID}")
    print(f"[load] adapter={args.adapter}")

    # bf16 if CUDA available — saves memory during the merge, doesn't affect quality
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[load] dtype={dtype}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype)
    peft_model = PeftModel.from_pretrained(base, str(args.adapter))

    # Sanity: confirm LoRA actually attached (catches the silent "0 modules" failure)
    n_lora = sum(1 for n, _ in peft_model.named_modules()
                 if hasattr(peft_model.get_submodule(n), "lora_A"))
    print(f"[load] LoRA-attached modules: {n_lora}")
    if n_lora == 0:
        raise RuntimeError("0 LoRA modules attached — adapter_config.json target_modules "
                           "regex didn't match. Won't merge a no-op adapter.")

    print(f"[merge] merging LoRA delta into base weights...")
    merged = peft_model.merge_and_unload()

    if not args.no_truncate:
        truncate_text_model(merged, TRUNCATE_TO)

    args.local_out.mkdir(parents=True, exist_ok=True)
    print(f"[save] writing merged model → {args.local_out}")
    merged.save_pretrained(str(args.local_out))
    processor.save_pretrained(str(args.local_out))

    # Quick model card so the HF page isn't blank
    card = args.local_out / "README.md"
    card.write_text(f"""---
base_model: {MODEL_ID}
library_name: transformers
tags:
- smolvlm2
- lora-merged
- celebrity-recognition
- robotics
- vla-backbone
---

# smolvlm2_lora_celebs

SmolVLM2-500M with a celebrity-recognition LoRA merged in, then truncated to
the first {TRUNCATE_TO} text-model layers to match SmolVLA's expected backbone.

## Source

- Base: [{MODEL_ID}](https://huggingface.co/{MODEL_ID})
- LoRA training data: ~200-celebrity scraped dataset (eval3_celebs)
- LoRA adapter: `{args.adapter.name}` (rank, epochs encoded in name)

## Intended use

Drop-in VLM backbone for SmolVLA:

```bash
lerobot-train \\
    --policy.path=lerobot/smolvla_base \\
    --policy.vlm_model_name={args.hf_repo} \\
    --dataset.repo_id=<your-robot-dataset> \\
    ...
```

## Notes

- Text model truncated from 32 → {TRUNCATE_TO} layers (layers {TRUNCATE_TO}-31 deleted
  because SmolVLA discards them anyway).
- Vision tower kept in full.
- LoRA targets restricted to the layers SmolVLA actually uses, so no merged
  weight was discarded.
""")

    if args.no_push:
        print(f"[done] saved locally. Skipping push (use without --no-push to upload).")
        return

    print(f"[push] uploading to https://huggingface.co/{args.hf_repo} (private={args.private})")
    merged.push_to_hub(
        args.hf_repo,
        commit_message=args.commit_msg,
        private=args.private,
    )
    processor.push_to_hub(
        args.hf_repo,
        commit_message="add processor",
        private=args.private,
    )
    # Push the README separately via huggingface_hub
    try:
        from huggingface_hub import upload_file
        upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=args.hf_repo,
            commit_message="add model card",
        )
    except Exception as e:
        print(f"[push] model card upload failed (non-fatal): {e}")

    print(f"[done] https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
