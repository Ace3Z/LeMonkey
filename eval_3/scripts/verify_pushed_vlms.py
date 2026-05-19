#!/usr/bin/env python
"""
Sanity-check the merged+truncated VLMs after pushing to HF.

For each pushed repo (broad + toy), this script:
  1. Loads the model fresh from HF (proves the upload roundtripped intact)
  2. Asserts the text_model has only 16 layers (truncation succeeded)
  3. Runs inference on one image of each toy celeb (Obama / LeCun / Swift)
  4. Reports whether the model names them correctly

Run on the H100 (where the test images live):
    python eval_3/scripts/verify_pushed_vlms.py \
        --repos HBOrtiz/smolvlm2_lora_celebs HBOrtiz/smolvlm2_toy_celebs \
        --data-root $EVAL3_ROOT
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


TOY_CELEBS = [
    ("Barack Obama", "scraped/barack_obama"),
    ("Yann LeCun",  "scraped/yann_lecun"),
    ("Taylor Swift","scraped/taylor_swift"),
]
PROMPT = "Who is shown in this photo?"


def find_text_layer_count(model) -> int:
    """Locate text-model layers and return their count."""
    for path in ("model.text_model.layers", "model.text_model.model.layers",
                 "model.model.text_model.layers"):
        try:
            obj = model
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, torch.nn.ModuleList):
                return len(obj)
        except AttributeError:
            continue
    return -1


def normalize(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def run_repo(repo: str, data_root: Path, device: str, dtype: torch.dtype):
    print(f"\n{'='*70}\n=== {repo}\n{'='*70}")
    print(f"[load] from HF...")
    processor = AutoProcessor.from_pretrained(repo)
    model = AutoModelForImageTextToText.from_pretrained(repo, torch_dtype=dtype).to(device).eval()

    n_layers = find_text_layer_count(model)
    expected = 16
    ok_struct = (n_layers == expected)
    print(f"[check] text_model layers: {n_layers}  (expected {expected})  {'✓' if ok_struct else '✗ MISMATCH'}")

    correct = 0
    for name, subdir in TOY_CELEBS:
        candidates = sorted((data_root / subdir).glob("*.jpg"))
        if not candidates:
            print(f"  [skip] {name}: no images at {data_root / subdir}")
            continue
        img_path = candidates[0]
        img = Image.open(img_path).convert("RGB")

        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, images=[img], return_tensors="pt",
                           do_image_splitting=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype.is_floating_point:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        new_tokens = out[0, inputs["input_ids"].shape[-1]:]
        pred = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().split("\n")[0]

        match = normalize(name) in normalize(pred)
        correct += int(match)
        mark = "✓" if match else "✗"
        print(f"  {mark} expected: {name!r:20s} predicted: {pred!r}")

    print(f"[result] {repo}: {correct}/{len(TOY_CELEBS)} celebs named correctly")
    return ok_struct and correct == len(TOY_CELEBS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", required=True,
                        help="HF repo IDs of pushed VLMs to verify")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root of eval3_celebs (contains scraped/<slug>/*.jpg)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device}  dtype={dtype}")

    results = {}
    for repo in args.repos:
        results[repo] = run_repo(repo, args.data_root, device, dtype)

    print(f"\n{'='*70}\n=== Summary ===")
    for repo, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {repo}")


if __name__ == "__main__":
    main()
