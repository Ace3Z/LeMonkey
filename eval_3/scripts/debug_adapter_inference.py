#!/usr/bin/env python
"""
Diagnostic when eval_lora_train_id_accuracy.py reports 0% — verifies whether
the LoRA adapter is actually being applied at inference time.

Loads the base SmolVLM2 + LoRA adapter, then runs inference on one held-out
image TWICE: once with the adapter enabled, once with it explicitly disabled
via PeftModel.disable_adapter_layers(). Same input, same generation params.

If both outputs are identical → LoRA isn't doing anything (likely loading bug
or adapter_config target_modules not matching the model's actual module names).
If outputs differ but neither contains the celebrity name → LoRA is active but
hasn't learned the binding (training problem, not eval problem).

Run:
    python eval_3/scripts/debug_adapter_inference.py \
        --adapter $DATA_ROOT/lora_celeb_v0 \
        --image   $DATA_ROOT/train/n000018/0135_01.jpg \
        --expected "Aaron Schock"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--expected", type=str, default="(unknown)")
    parser.add_argument("--prompt", type=str, default="Who is shown in this photo?")
    args = parser.parse_args()

    print("=== adapter_config.json ===")
    with open(args.adapter / "adapter_config.json") as f:
        cfg = json.load(f)
    # only show the salient fields
    for k in ("base_model_name_or_path", "r", "lora_alpha", "lora_dropout",
              "task_type", "target_modules"):
        v = cfg.get(k)
        if isinstance(v, list) and len(v) > 6:
            v = f"[{len(v)} entries: {v[:3]} ...]"
        print(f"  {k}: {v}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"\n=== loading model on {device}, dtype={dtype} ===")
    proc = AutoProcessor.from_pretrained(str(args.adapter))
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, str(args.adapter)).to(device).eval()

    n_lora = sum(
        1 for n, _ in model.named_modules()
        if hasattr(model.get_submodule(n), "lora_A")
    )
    print(f"  LoRA-attached modules: {n_lora}")
    if n_lora == 0:
        print("  ⚠️  ZERO LoRA modules attached — target_modules regex didn't match the base model")
        print("       Check adapter_config.json target_modules vs base model layer names.")

    # Render and tokenize
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": args.prompt},
        ],
    }]
    rendered = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print(f"\n=== rendered chat template ===")
    print(repr(rendered))

    img = Image.open(args.image).convert("RGB")
    inputs = proc(text=rendered, images=[img],
                  return_tensors="pt", do_image_splitting=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs and inputs["pixel_values"].dtype.is_floating_point:
        inputs["pixel_values"] = inputs["pixel_values"].to(next(model.parameters()).dtype)
    n_input = inputs["input_ids"].shape[-1]

    # Inference WITH adapter
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    new = out[0, n_input:]
    decoded_with = proc.tokenizer.decode(new, skip_special_tokens=False)

    # Inference WITHOUT adapter (peft sets the LoRA delta to zero)
    model.disable_adapter_layers()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    new = out[0, n_input:]
    decoded_without = proc.tokenizer.decode(new, skip_special_tokens=False)
    model.enable_adapter_layers()

    print(f"\n=== expected: {args.expected!r} ===")
    print(f"=== WITH adapter:    {decoded_with!r}")
    print(f"=== WITHOUT adapter: {decoded_without!r}")

    print()
    if decoded_with == decoded_without:
        print("⚠️  IDENTICAL outputs → adapter is loaded but contributes ZERO to the forward pass.")
        print("   Most likely: target_modules regex didn't match → 0 LoRA layers attached.")
        print("   Check the LoRA-attached modules count above.")
    else:
        print("✓ Different outputs → adapter IS being applied.")
        print("   If neither contains the name, the training didn't learn the binding "
              "(check train loss curve / data alignment).")


if __name__ == "__main__":
    main()
