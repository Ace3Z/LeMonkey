#!/usr/bin/env python3
"""G2 variant: probe PaliGemma's celeb knowledge on a SCENE (not isolated
headshots).

The eval-day task is "which printout is celeb X" on a scene with 3
printouts visible. This probe asks PaliGemma the same kind of question
on a real teleop scene, to test whether the WebLI prior survives in
both directions: name → identifies face, and position → identifies face.

Two complementary checks per celeb:

1. Open-ended VQA: "Who is on the LEFT of the image?" → expect the
   actual celeb name at that slot (LeCun in our test frame).
2. Verification VQA: "Is Taylor Swift visible in this image? answer yes
   or no." → expect "yes" because Swift IS in the frame.

A passing PaliGemma should ace #2 (Swift IS in the frame). #1 is the
deeper test (does PaliGemma name a face given a position?).

Usage:
    python eval_3/scripts/probe_paligemma_celeb_vqa_scene.py \\
        --image /tmp/probe_input.png \\
        --out /tmp/pg_vqa_scene.json

Pre-requirement: google/paligemma-3b-pt-224 cached in HF (~3 GB).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

CELEBS = {
    "taylor swift": ["swift", "taylor"],
    "barack obama": ["obama", "barack"],
    "yann lecun":   ["lecun", "yann"],
}

# Our test frame (input.png from the SmolVLA attention probe) shows
# LeCun on the LEFT, Obama in the MIDDLE, Swift on the RIGHT.
POSITION_QUESTIONS = {
    "left":   ("Who is on the left of this image? answer en", "yann lecun"),
    "middle": ("Who is in the middle of this image? answer en", "barack obama"),
    "right":  ("Who is on the right of this image? answer en", "taylor swift"),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/paligemma-3b-pt-224")
    p.add_argument("--image", default="/tmp/probe_input.png")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/tmp/pg_vqa_scene.json")
    p.add_argument("--max-new-tokens", type=int, default=24)
    args = p.parse_args()

    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    img = Image.open(args.image).convert("RGB")
    print(f"[load] {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.device).eval()

    def ask(prompt: str) -> str:
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(args.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        return processor.decode(out[0, input_len:], skip_special_tokens=True).strip().lower()

    results = {"open_ended": {}, "verification": {}, "scene_general": None}

    # 0. General "who is in the image".
    a = ask("Who is in this image? answer en")
    results["scene_general"] = a
    print(f"\n[scene_general] 'Who is in this image?' → {a!r}", flush=True)

    # 1. Open-ended positional VQA.
    print("\n[open_ended] Position → celeb")
    for pos, (q, expected) in POSITION_QUESTIONS.items():
        a = ask(q)
        aliases = CELEBS[expected]
        hit = any(al in a for al in aliases)
        results["open_ended"][pos] = {"answer": a, "expected": expected, "hit": hit}
        print(f"  {pos:8s} {q!r:60s} → {a!r:30s}  expected={expected!r}  hit={hit}",
              flush=True)

    # 2. Verification VQA — "Is X visible?".
    print("\n[verification] 'Is X visible?' (expect yes for all 3)")
    for celeb in CELEBS:
        q = f"Is {celeb} visible in this image? answer en"
        a = ask(q)
        # "yes" anywhere = pass.
        hit = "yes" in a or "y" == a.strip()
        results["verification"][celeb] = {"answer": a, "hit": hit}
        print(f"  {celeb:14s} → {a!r:30s}  hit={hit}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {args.out}", flush=True)

    # Gating decision.
    pos_hits = sum(1 for v in results["open_ended"].values() if v["hit"])
    ver_hits = sum(1 for v in results["verification"].values() if v["hit"])
    print("\n=== GATING ===")
    print(f"Open-ended positional VQA: {pos_hits}/3 correct")
    print(f"Verification VQA:          {ver_hits}/3 correct")
    if ver_hits == 3 and pos_hits >= 2:
        print("PASS — PaliGemma knows the celebs AND can localize them.")
        print("       M2 + KLAL on action data should be sufficient.")
    elif ver_hits == 3:
        print("MARGINAL — celeb names known but positional grounding weak.")
        print("           KLAL is essential to teach name→position binding.")
    elif ver_hits >= 1:
        print("WEAK — only some celebs recognized.")
        print("       Recommend adding LoRA VQA pretrain before action training.")
    else:
        print("FAIL — PaliGemma does not recognize these celebs in this scene.")
        print("       Mandatory: LoRA face VQA pretrain BEFORE action training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
