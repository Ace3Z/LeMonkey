#!/usr/bin/env python
"""
Diagnostic: name-recall accuracy of a SmolVLM2 LoRA adapter on **held-out
images of training identities**.

Why this script exists: the val.jsonl built by build_llava_json.py (current
default) holds out *whole identities*, which makes it a zero-shot
identification task — structurally impossible (the model can't predict a
name it never saw labeled). Eval loss going *up* during training measures
overfitting on a meaningless metric.

What we actually want to know: for an identity we DID train on, can the
LoRA-fine-tuned model correctly name a *new* photo of that identity?

This script answers that. For N sampled training identities × K held-out
images per identity, it runs inference with the same prompt format used in
training, normalizes the generated text, and reports accuracy.

Run:
    python eval_3/scripts/eval_lora_train_id_accuracy.py \
        --adapter   $DATA_ROOT/lora_celeb_v0 \
        --data-root $DATA_ROOT \
        --n-identities 50 --n-imgs-per-id 5

Add --no-lora to get the BASE-model baseline (sanity-check that the LoRA
actually moved the needle).

Add --out-jsonl <path> to dump every (image, expected, predicted, correct)
row for inspection.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_PROMPT = "Who is shown in this photo?"


def normalize_for_match(s: str) -> str:
    """lowercase, drop punctuation, collapse whitespace."""
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return " ".join(s.split())


def name_in_prediction(expected: str, predicted: str) -> bool:
    """A prediction counts if the expected name (normalized) appears in it."""
    e = normalize_for_match(expected)
    p = normalize_for_match(predicted)
    if not e:
        return False
    # accept either: predicted starts with expected, OR expected appears as substring
    return p.startswith(e) or (f" {e} " in f" {p} ")


def load_train_image_paths(train_jsonl: Path) -> dict[str, set[str]]:
    """Return {class_id: set(image_path)} for paths used in training."""
    out: dict[str, set[str]] = {}
    with open(train_jsonl) as f:
        for line in f:
            r = json.loads(line)
            out.setdefault(r["class_id"], set()).add(r["image_path"])
    return out


def pick_device_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_model(adapter_path: Path, no_lora: bool, device: str, dtype: torch.dtype):
    # Use the adapter dir's processor (it was saved with the model) when LoRA is on,
    # else use the base model's processor.
    proc_src = MODEL_ID if no_lora else str(adapter_path)
    proc = AutoProcessor.from_pretrained(proc_src)

    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if no_lora:
        model = base
    else:
        model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.to(device).eval()
    return proc, model


def infer_one(proc, model, image_path: str, prompt: str, device: str) -> str:
    img = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]
    text = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = proc(
        text=text, images=[img],
        return_tensors="pt", do_image_splitting=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # cast pixel_values to model dtype if floating
    if "pixel_values" in inputs and inputs["pixel_values"].dtype.is_floating_point:
        inputs["pixel_values"] = inputs["pixel_values"].to(next(model.parameters()).dtype)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
        )
    new_tokens = out[0, inputs["input_ids"].shape[-1]:]
    text = proc.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # cut at first newline (model often continues with another turn)
    return text.split("\n")[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True,
                        help="Path to the LoRA adapter dir (output of train_smolvlm2_lora.py)")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root with manifests/ subdir")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="default: <data-root>/manifests/manifest.parquet")
    parser.add_argument("--train-jsonl", type=Path, default=None,
                        help="default: <data-root>/manifests/train.jsonl  "
                             "(needed to know which images were in training)")
    parser.add_argument("--n-identities", type=int, default=50)
    parser.add_argument("--n-imgs-per-id", type=int, default=5)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--no-lora", action="store_true",
                        help="Skip adapter, eval base model only (BASELINE)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-jsonl", type=Path, default=None,
                        help="Optional: write per-row predictions to this JSONL")
    args = parser.parse_args()

    if args.manifest is None:
        args.manifest = args.data_root / "manifests/manifest.parquet"
    if args.train_jsonl is None:
        args.train_jsonl = args.data_root / "manifests/train.jsonl"

    device, dtype = pick_device_dtype()
    print(f"[eval] device={device} dtype={dtype}")
    print(f"[eval] adapter={args.adapter}{'  (BASELINE — no LoRA)' if args.no_lora else ''}")
    print(f"[eval] manifest:    {args.manifest}")
    print(f"[eval] train jsonl: {args.train_jsonl}")
    print(f"[eval] prompt:      {args.prompt!r}")
    print()

    df = pd.read_parquet(args.manifest)
    train_used = load_train_image_paths(args.train_jsonl)
    print(f"[eval] manifest identities: {len(df)}")
    print(f"[eval] identities seen in training: {len(train_used)}")

    # restrict to identities the LoRA was actually trained on
    df = df[df["class_id"].isin(train_used.keys())].reset_index(drop=True)
    print(f"[eval] eligible identities for held-out test: {len(df)}\n")

    rng = random.Random(args.seed)
    n_pick = min(args.n_identities, len(df))
    sampled = df.sample(n=n_pick, random_state=args.seed).reset_index(drop=True)

    print(f"[eval] loading model...")
    proc, model = load_model(args.adapter, args.no_lora, device, dtype)
    print(f"[eval] model loaded.\n")

    print(f"=== per-identity results ===")
    rows: list[dict] = []
    for _, row in sampled.iterrows():
        cid = row["class_id"]
        name = row["name"].replace("_", " ")
        all_paths = list(row["image_paths"])
        used = train_used.get(cid, set())
        held_out = [p for p in all_paths if p not in used]
        if len(held_out) == 0:
            print(f"  {cid}  {name:30s}  (no held-out imgs available, skipping)")
            continue
        k = min(args.n_imgs_per_id, len(held_out))
        sample = rng.sample(held_out, k=k)
        n_correct = 0
        for p in sample:
            pred = infer_one(proc, model, p, args.prompt, device)
            ok = name_in_prediction(name, pred)
            n_correct += int(ok)
            rows.append({
                "class_id": cid, "expected": name, "predicted": pred,
                "correct": ok, "image_path": p,
            })
        marker = "✓" if n_correct == k else ("·" if n_correct > 0 else "✗")
        print(f"  {marker} {cid}  {name:30s}  {n_correct}/{k}")

    n_total = len(rows)
    n_correct = sum(r["correct"] for r in rows)
    n_ids = len({r["class_id"] for r in rows})
    print()
    print(f"=== summary ===")
    print(f"identities tested: {n_ids}")
    print(f"images tested:     {n_total}")
    if n_total > 0:
        print(f"accuracy:          {n_correct}/{n_total} = {n_correct/n_total*100:.1f}%")
        # also identity-level: at least 1 correct out of K
        per_id_any = {}
        for r in rows:
            per_id_any[r["class_id"]] = per_id_any.get(r["class_id"], False) or r["correct"]
        n_id_any = sum(per_id_any.values())
        print(f"identities w/ ≥1 correct: {n_id_any}/{n_ids} = {n_id_any/n_ids*100:.1f}%")

    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"\n[eval] per-row predictions → {args.out_jsonl}")


if __name__ == "__main__":
    main()
