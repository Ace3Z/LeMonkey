#!/usr/bin/env python3
"""Phase-1 gating probe for the Eval 3 plan.

Decides between Path A (image-as-prompt + co-train, robust) and Path B
(text-only Pi0.5 fine-tune, conservative) by measuring whether
google/paligemma-3b-pt-224 can identify named public figures zero-shot
from the workspace-style portraits we will see at demo day.

Decision rule (per eval_3/README.md):
    TOY accuracy ≥ 80 % AND OOD accuracy ≥ 50 %  → Path B viable
    anything below                               → Path A mandatory

Inputs:
  • TOY images extracted from docs/Eval_3_TOY_Celebrity_Images.pdf
    (skips img-006-009 which has "BARACK OBAMA" text overlay → would let the
    VLM OCR-cheat; skips img-008-015, a duplicate of img-008-013).
  • A handful of OOD reference photos pulled from Wikimedia Commons at runtime
    so we don't need to hand-curate. These stand in for the eventual
    TA-published OOD candidate list.

Output: per-image answer, per-celebrity accuracy, and overall verdict.

No Brev compute needed. Runs on local CUDA if available; falls back to CPU.
PaliGemma-3B in fp16 needs ~6 GB; if your GPU is tighter, set --device cpu
(slower: ~30-60 s/image vs <2 s/image on GPU).
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# ─── Ground-truth TOY labels ─────────────────────────────────────────────────
# Visually verified by inspection of the extracted images.
TOY_DIR = Path("/tmp/eval3_toy_pngs")
TOY_LABELS = {
    "img-001-000.png": "Taylor Swift",
    "img-002-001.png": "Taylor Swift",
    "img-002-002.png": "Taylor Swift",
    "img-003-003.png": "Taylor Swift",
    "img-003-004.png": "Taylor Swift",
    "img-004-005.png": "Barack Obama",
    "img-004-006.png": "Barack Obama",
    "img-005-007.png": "Barack Obama",
    "img-005-008.png": "Barack Obama",
    # img-006-009.png deliberately skipped: OCR-leaky (contains "BARACK OBAMA" text)
    "img-006-010.png": "Yann LeCun",
    "img-007-011.png": "Yann LeCun",
    "img-007-012.png": "Yann LeCun",
    "img-008-013.png": "Yann LeCun",
    "img-008-014.png": "Yann LeCun",
    # img-008-015.png deliberately skipped: duplicate of img-008-013
}

# ─── OOD reference photos ────────────────────────────────────────────────────
# Wikipedia's Special:FilePath redirect — the canonical script-friendly way
# to fetch Wikimedia Commons images. Direct upload.wikimedia.org URLs get
# 429-rate-limited or 400'd for non-browser User-Agents.
OOD_LABELS = {
    "Roger Federer":     "https://en.wikipedia.org/wiki/Special:FilePath/Roger%20Federer%202015%20%28cropped%29.jpg?width=600",
    "Angela Merkel":     "https://en.wikipedia.org/wiki/Special:FilePath/Angela%20Merkel%202019%20cropped.jpg?width=600",
    "Elon Musk":         "https://en.wikipedia.org/wiki/Special:FilePath/Elon%20Musk%20Royal%20Society%20%28crop2%29.jpg?width=600",
    "Lionel Messi":      "https://en.wikipedia.org/wiki/Special:FilePath/Lionel%20Messi%2020180626.jpg?width=600",
    "Cristiano Ronaldo": "https://en.wikipedia.org/wiki/Special:FilePath/Cristiano%20Ronaldo%202018.jpg?width=600",
    "Beyoncé":           "https://en.wikipedia.org/wiki/Special:FilePath/Beyonc%C3%A9%20at%20The%20Lion%20King%20European%20Premiere%202019.png?width=600",
}


# ─── Loading + asking ────────────────────────────────────────────────────────
def load_model(model_id: str, device: str, dtype: torch.dtype):
    print(f"  loading {model_id} on {device} ({dtype})...", flush=True)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype
    ).to(device).eval()
    print(f"  ✓ loaded in {time.time() - t0:.0f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f} B params)\n",
          flush=True)
    return proc, model


@torch.inference_mode()
def ask_who(proc, model, image: Image.Image, device: str, dtype: torch.dtype) -> str:
    """Ask PaliGemma 'Who is this person?' for the image. Returns the raw answer."""
    # PaliGemma expects an image token + a short text prompt.
    prompt = "answer en Who is this person?"
    inputs = proc(text=prompt, images=image.convert("RGB"), return_tensors="pt").to(device, dtype)
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    decoded = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


# ─── Scoring ─────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    """Strip case / punctuation / accents for fuzzy matching."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def hits(answer: str, ground_truth: str) -> bool:
    """The answer hits if the celeb's last name appears in it (loose match)."""
    a = normalize(answer)
    last = normalize(ground_truth.split()[-1])
    # Last-name substring match — robust to "Taylor Swift", "Swift", "T. Swift", etc.
    if last in a:
        return True
    # Also accept full name match (covers double-barrelled cases like "Yann LeCun" → "lecun")
    full = normalize(ground_truth)
    return all(p in a for p in full.split())


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="google/paligemma-3b-pt-224")
    ap.add_argument("--device", default=None,
                    help="cuda or cpu; default: cuda if available")
    ap.add_argument("--dtype", default=None,
                    help="float16 / float32; default: float16 on cuda, float32 on cpu")
    ap.add_argument("--skip-ood", action="store_true",
                    help="Skip OOD probe (no internet / Wiki blocked)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype:
        dtype = getattr(torch, args.dtype)
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32

    if not TOY_DIR.is_dir() or not any(TOY_DIR.glob("img-*.png")):
        print(f"ERROR: TOY images not found at {TOY_DIR}", file=sys.stderr)
        print("       Run first:", file=sys.stderr)
        print(f"       pdfimages -p -j /home/lemonkey/LeMonkey/docs/Eval_3_TOY_Celebrity_Images.pdf {TOY_DIR}/img", file=sys.stderr)
        print("       (then convert ppm → png)", file=sys.stderr)
        return 1

    print("=" * 70)
    print("  PaliGemma zero-shot celebrity probe")
    print(f"  model  : {args.model_id}")
    print(f"  device : {device}")
    print(f"  dtype  : {dtype}")
    print("=" * 70)
    print()

    proc, model = load_model(args.model_id, device, dtype)

    # ─── TOY probe ───────────────────────────────────────────────────────────
    print("─" * 70)
    print("  TOY images (in-distribution; from Eval_3_TOY_Celebrity_Images.pdf)")
    print("─" * 70)
    toy_per_celeb = {"Taylor Swift": [0, 0], "Barack Obama": [0, 0], "Yann LeCun": [0, 0]}  # [hits, total]
    for fname, gt in TOY_LABELS.items():
        path = TOY_DIR / fname
        if not path.exists():
            print(f"  ✗ MISSING file {fname} — skipping")
            continue
        img = Image.open(path)
        ans = ask_who(proc, model, img, device, dtype)
        ok = hits(ans, gt)
        toy_per_celeb[gt][1] += 1
        toy_per_celeb[gt][0] += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {fname:25s}  gt={gt:15s}  paligemma=\"{ans}\"")

    toy_total = sum(t for _, t in toy_per_celeb.values())
    toy_hits  = sum(h for h, _ in toy_per_celeb.values())
    toy_acc   = 100 * toy_hits / max(toy_total, 1)

    print()
    print(f"  TOY summary:")
    for celeb, (h, t) in toy_per_celeb.items():
        print(f"    {celeb:15s}  {h}/{t}  ({100*h/max(t,1):.0f}%)")
    print(f"    {'OVERALL':15s}  {toy_hits}/{toy_total}  ({toy_acc:.0f}%)")

    # ─── OOD probe ───────────────────────────────────────────────────────────
    ood_acc = None
    if args.skip_ood:
        print("\n  (OOD probe skipped via --skip-ood)")
    else:
        print()
        print("─" * 70)
        print("  OOD references (popular public figures, Wikimedia Commons)")
        print("─" * 70)
        ood_hits, ood_total = 0, 0
        for name, url in OOD_LABELS.items():
            try:
                resp = requests.get(url, timeout=15, allow_redirects=True,
                                    headers={"User-Agent": "LeMonkey-research/0.1 (mtajdini@student.ethz.ch)"})
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content))
            except Exception as e:
                print(f"  - {name:20s}  fetch failed: {e}")
                continue
            ans = ask_who(proc, model, img, device, dtype)
            ok = hits(ans, name)
            ood_total += 1
            ood_hits += int(ok)
            mark = "✓" if ok else "✗"
            print(f"  {mark} {name:20s}  paligemma=\"{ans}\"")
        ood_acc = 100 * ood_hits / max(ood_total, 1)
        print(f"\n  OOD summary: {ood_hits}/{ood_total}  ({ood_acc:.0f}%)")

    # ─── Verdict ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print(f"  TOY accuracy : {toy_acc:.0f}%   (threshold for Path B viability: ≥ 80%)")
    if ood_acc is not None:
        print(f"  OOD accuracy : {ood_acc:.0f}%   (threshold: ≥ 50%)")

    if toy_acc >= 80 and (ood_acc is None or ood_acc >= 50):
        print("\n  → Path B (text-only Pi0.5 fine-tune) is viable for IID.")
        print("     Path A (image-as-prompt + co-train) still recommended for OOD safety.")
    else:
        print("\n  → Path A (image-as-prompt + co-train) is MANDATORY.")
        print("     PaliGemma's zero-shot recognition is too weak for the eval.")
        print("     Reference photos at inference + VQA co-training is the safe path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
