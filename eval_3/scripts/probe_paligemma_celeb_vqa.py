#!/usr/bin/env python3
"""Gating experiment G2: does PaliGemma know Swift / Obama / LeCun zero-shot?

Loads google/paligemma-3b-pt-224 (the upstream PaliGemma base — same
WebLI-pretrained weights Pi0.5 wraps but with the standard transformers
`generate()` API for VQA) and prompts:

    'answer en\\nWho is in this image?'  on a printed headshot of:
        - Taylor Swift  (heldout_01..08)
        - Barack Obama  (heldout_01..08)
        - Yann LeCun    (heldout_01..08)

Counts top-1 name-match per celeb. If hit rate ≥ ~50% → PaliGemma
already binds these names to faces via WebLI; M2 + KLAL on top of the
action data may be enough. If hit rate ≤ 20% → need to add Stage-3
VQA pretraining (LoRA on a celeb-face VQA dataset, à la M5) BEFORE
the action-data training run.

Inspired by Agent-2's literature: FaceBench, FaceScanPaliGemma, and
VLM4VLA show PaliGemma fine-tunes well on face VQA — but the question
"does it know celeb names without fine-tuning?" has no published
answer for our specific 3 celebs. This probe forces an empirical floor.

Usage:
    python eval_3/scripts/probe_paligemma_celeb_vqa.py \\
        --celeb-bank ~/Downloads/eval3_celebs/track3_bank \\
        --out /tmp/pg_vqa_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

CELEBS = ["taylor_swift", "barack_obama", "yann_lecun"]
ALIASES = {
    "taylor_swift": ["taylor swift", "swift", "taylor"],
    "barack_obama": ["barack obama", "obama", "barack"],
    "yann_lecun":   ["yann lecun", "lecun", "yann"],
}
PROMPT = "answer en\nWho is in this image?"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/paligemma-3b-pt-224")
    p.add_argument("--celeb-bank", default=str(Path.home() / "Downloads/eval3_celebs/track3_bank"),
                   help="dir containing <celeb>/heldout_NN.png folders")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(Path.home() / "pg_vqa_probe.json"))
    p.add_argument("--max-new-tokens", type=int, default=24)
    args = p.parse_args()

    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    print(f"[load] {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    print(f"[load] loaded; params={sum(p.numel() for p in model.parameters())/1e6:.1f}M",
          flush=True)

    bank = Path(args.celeb_bank)
    if not bank.exists():
        print(f"[error] celeb bank not found: {bank}", file=sys.stderr)
        return 2

    results = {}
    for celeb in CELEBS:
        celeb_dir = bank / celeb
        photos = sorted(celeb_dir.glob("heldout_*.png"))
        if not photos:
            print(f"[warn] no photos under {celeb_dir}", flush=True)
            continue
        per_celeb = []
        for ph in photos:
            img = Image.open(ph).convert("RGB")
            inputs = processor(text=PROMPT, images=img, return_tensors="pt").to(args.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            ans_ids = out[0, input_len:]
            ans = processor.decode(ans_ids, skip_special_tokens=True).strip().lower()
            hit = any(alias in ans for alias in ALIASES[celeb])
            per_celeb.append({"photo": ph.name, "answer": ans, "hit": hit})
            print(f"  {celeb:14s} {ph.name:18s} → {ans!r:40s}  hit={hit}", flush=True)
        hit_rate = sum(1 for r in per_celeb if r["hit"]) / max(1, len(per_celeb))
        results[celeb] = {"hit_rate": hit_rate, "n": len(per_celeb), "rows": per_celeb}
        print(f"[summary] {celeb}: {hit_rate:.0%} hit rate ({len(per_celeb)} photos)", flush=True)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results: {out_path}")

    # Gating decision.
    rates = [results[c]["hit_rate"] for c in CELEBS if c in results]
    avg = sum(rates) / max(1, len(rates))
    print("\n=== GATING DECISION ===")
    print(f"Average hit rate across 3 celebs: {avg:.0%}")
    if avg >= 0.5:
        print("PASS — PaliGemma already binds these celeb names to faces.")
        print("Plan: M2 + KLAL on top of the action data alone.")
    elif avg >= 0.2:
        print("MARGINAL — some name binding exists but is weak.")
        print("Recommend: add Stage-3 VQA fine-tune (~10-20k pairs, LoRA r=16) "
              "before action training. Expected lift: ~30-50%.")
    else:
        print("FAIL — PaliGemma does NOT know these celebs zero-shot.")
        print("Mandatory: Stage-3 VQA fine-tune BEFORE action training. "
              "M2 + KLAL alone will not teach name binding from action data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
