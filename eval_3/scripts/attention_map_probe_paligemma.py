#!/usr/bin/env python3
"""G1 alternative: probe vanilla PaliGemma attention directly via the
standard `transformers` API. Bypasses lerobot's pi05 wrapper entirely so
we avoid the constant transformers-version compat dance.

This answers the SAME gating question: does PaliGemma have any cross-modal
attention that shifts when we change the named celeb in the prompt?

Method
------
1. Load `google/paligemma-3b-pt-224` (the underlying PaliGemma weights
   Pi0.5 wraps).
2. Run `model(...)` with output_attentions=True (forces eager attention).
3. For each celeb prompt, average attention across heads, slice
   [name-token rows, image-patch columns], reshape 16x16, save heatmap.
4. Compare argmax patch positions across prompts.

If vanilla PaliGemma argmax SHIFTS with the celeb name → cross-modal
attention exists; M2 + KLAL training has a channel to supervise.
If it doesn't → reconsider plan.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROMPTS = {
    "swift": "Where is Taylor Swift in this image?",
    "obama": "Where is Barack Obama in this image?",
    "lecun": "Where is Yann LeCun in this image?",
}
NAME = {
    "swift": "Taylor Swift",
    "obama": "Barack Obama",
    "lecun": "Yann LeCun",
}
GRID = 16
NUM_PATCHES = GRID * GRID  # 256
IMG_HW = 224


def _find_name_positions(tokenizer, full_ids: list[int], name_phrase: str):
    for variant in (name_phrase, " " + name_phrase):
        nm = tokenizer.encode(variant, add_special_tokens=False)
        for i in range(len(full_ids) - len(nm) + 1):
            if full_ids[i : i + len(nm)] == nm:
                return i, i + len(nm)
    return None, None


def _overlay(img_chw_01, hm_NN, alpha=0.45):
    img = (img_chw_01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    hm = hm_NN.astype(np.float32)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
    hm_t = torch.from_numpy(hm)[None, None].float()
    hm_up = F.interpolate(hm_t, size=img.shape[:2], mode="bilinear",
                          align_corners=False)[0, 0].numpy()
    color = np.stack([hm_up, hm_up * 0.4, np.zeros_like(hm_up)], axis=-1)
    color = (color * 255).astype(np.uint8)
    return Image.fromarray((img * (1 - alpha) + color * alpha).clip(0, 255).astype(np.uint8))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/paligemma-3b-pt-224")
    p.add_argument("--image", default="/tmp/probe_input.png")
    p.add_argument("--device", default="cuda")
    p.add_argument("--layers", nargs="+", type=int, default=[6, 10, 14, 17])
    p.add_argument("--out", default=str(Path.home() / "paligemma_attn_probe"))
    args = p.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    print(f"[load] {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32,
        attn_implementation="eager",  # required for output_attentions
    ).to(args.device).eval()

    n_layers = model.config.text_config.num_hidden_layers
    print(f"[load] paligemma text layers={n_layers}", flush=True)

    # Test image (raw 480x640 OK; processor will resize).
    img = Image.open(args.image).convert("RGB")
    img_tensor_224 = F.interpolate(
        torch.from_numpy(np.array(img)).permute(2, 0, 1).float()[None] / 255.0,
        size=(IMG_HW, IMG_HW), mode="bilinear", align_corners=False,
    )[0]
    Image.fromarray((img_tensor_224.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(outdir / "input.png")

    tokenizer = processor.tokenizer
    summary = []

    for short, prompt in PROMPTS.items():
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(args.device)
        ids = inputs["input_ids"][0].cpu().tolist()
        # PaliGemma prepends `<image>` tokens (256 of them) + bos. Find where
        # the prompt text begins in the input_ids by skipping image tokens.
        image_token_id = model.config.image_token_index
        # image tokens form a contiguous block at the start; text starts
        # right after the last image token.
        nonimg_start = 0
        for i, t in enumerate(ids):
            if t != image_token_id:
                nonimg_start = i; break
        text_ids = ids[nonimg_start:]
        s_in_text, e_in_text = _find_name_positions(tokenizer, text_ids, NAME[short])
        if s_in_text is None:
            print(f"[warn] couldn't locate {NAME[short]!r} in tokenized prompt", flush=True)
            continue
        name_start = nonimg_start + s_in_text
        name_end = nonimg_start + e_in_text

        with torch.inference_mode():
            out = model(**inputs, output_attentions=True, use_cache=False)
        # out.attentions: tuple of (B, H, L, L) per layer.
        for n in args.layers:
            attn = out.attentions[n][0].float()  # (H, L, L)
            attn_avg = attn.mean(dim=0)          # (L, L)
            # rows = name-token positions; cols = first 256 image tokens.
            sub = attn_avg[name_start:name_end, :NUM_PATCHES].mean(dim=0)
            grid = sub.cpu().numpy().reshape(GRID, GRID)
            ar, ac = divmod(int(sub.argmax().item()), GRID)
            max_a = float(grid.max())
            ent = float(-(sub * sub.clamp(min=1e-12).log()).sum().item())
            print(f"[probe] {short:5s} layer {n:2d}  argmax=({ar:2d},{ac:2d})  "
                  f"max={max_a:.4f}  ent={ent:.3f}", flush=True)
            summary.append((short, n, ar, ac, max_a, ent))

            # Save.
            Image.fromarray(((grid - grid.min()) / (grid.max() - grid.min() + 1e-9) * 255).astype(np.uint8)).save(
                outdir / f"{short}_layer{n:02d}_heatmap.png"
            )
            _overlay(img_tensor_224, grid).save(outdir / f"{short}_layer{n:02d}_overlay.png")

    print("\n=== SUMMARY (vanilla PaliGemma) ===")
    print(f"{'celeb':<6} {'layer':<6} {'argmax(r,c)':<14} {'max':<8} {'ent':<6}")
    for s, n, ar, ac, m, e in summary:
        print(f"{s:<6} {n:<6} ({ar:2d},{ac:2d}){'':<8} {m:<8.4f} {e:<6.3f}")

    by_layer = {}
    for s, n, ar, ac, _, _ in summary:
        by_layer.setdefault(n, set()).add((ar, ac))
    moving = [n for n, pts in by_layer.items() if len(pts) > 1]
    print("\n=== GATING ===")
    print(f"Layers where argmax differs across prompts: {sorted(moving) or 'NONE'}")
    if moving:
        print("PASS — PaliGemma has cross-modal attention that shifts per celeb.")
        print("       M2 + KLAL on action data can supervise this channel.")
    else:
        print("CONCERN — argmax is constant across all 3 prompts at every layer.")
        print("          PaliGemma is sink-locked on this scene. Plan must change:")
        print("          1. VQA pretraining (LoRA on celeb VQA) MAY help unlock")
        print("             the channel by giving the model name-aware visual")
        print("             representations to attend to.")
        print("          2. OR consider a different VLA backbone.")

    print(f"\nSaved: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
