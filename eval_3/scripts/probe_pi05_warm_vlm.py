#!/usr/bin/env python3
"""Probe whether the PaliGemma VLM inside a Pi0.5 checkpoint recognises celebs.

Loads a Pi0.5 policy (e.g. HBOrtiz/pi05_paligemma_celeb_warm or a local
checkpoint dir), extracts the PaliGemma submodule, and asks it to name
celebrities. Gates Path B: a Pi0.5 policy with the VLM frozen only works as a
face-discriminating policy if the VLM already binds celeb names to faces.

Test 1 -- VQA: "answer en\\nWho is in this image?" on headshots of
  Taylor Swift / Barack Obama / Yann LeCun.  Top-1 name match per celeb.
  Baseline: vanilla google/paligemma-3b-pt-224 scored ~0% on the same celebs
  in gating run G2 (docs/experiments/2026-05-20_pi05_gating).

Test 2 -- attention shift: name-token -> image-patch attention on a 3-up
  composite scene; does the argmax move to the correct third when the
  prompted celeb changes?  (q/k-hook estimate; RoPE omitted -- secondary
  signal only, VQA is the headline.)

Usage:
  python probe_pi05_warm_vlm.py --repo ~/pi05_celeb_warm \\
      --celeb-bank ~/celeb_probe --out ~/warm_vlm_probe
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aug"))
from pi05_inference_patch import apply as _apply_pi05_patch  # noqa: E402

_apply_pi05_patch()

CELEBS = ["yann_lecun", "barack_obama", "taylor_swift"]  # composite order: L, M, R
DIRNAME = {"yann_lecun": "lecun", "barack_obama": "obama", "taylor_swift": "swift"}
NAME = {"yann_lecun": "Yann LeCun", "barack_obama": "Barack Obama",
        "taylor_swift": "Taylor Swift"}
ALIASES = {
    "yann_lecun": ["lecun", "le cun", "yann"],
    "barack_obama": ["obama", "barack"],
    "taylor_swift": ["swift", "taylor"],
}
VQA_PROMPTS = [
    "answer en\nWho is in this image?",
    "answer en\nWho is the person in this photo?",
]
GRID = 16
NUM_PATCHES = GRID * GRID  # 256
PROC_REPO = "google/paligemma-3b-pt-224"
ATTN_LAYERS = [6, 10, 14, 17]


def load_paligemma(repo: str, device: str):
    """Load a Pi0.5 policy and return its PaliGemma submodule (fp32, on device)."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    print(f"[load] PI05Policy.from_pretrained({repo}) ...", flush=True)
    policy = PI05Policy.from_pretrained(repo)
    paligemma = policy.model.paligemma_with_expert.paligemma
    paligemma = paligemma.to(device=device, dtype=torch.float32).eval()
    n = sum(p.numel() for p in paligemma.parameters())
    print(f"[load] PaliGemma submodule: {n / 1e6:.0f}M params, fp32, {device}",
          flush=True)
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return paligemma


def _find_name_span(tokenizer, full_ids: list[int], name_phrase: str):
    """Locate the token span of `name_phrase` inside `full_ids`."""
    for variant in (name_phrase, " " + name_phrase):
        nm = tokenizer.encode(variant, add_special_tokens=False)
        for i in range(len(full_ids) - len(nm) + 1):
            if full_ids[i:i + len(nm)] == nm:
                return i, i + len(nm)
    return None, None


def run_vqa(paligemma, processor, bank: Path, device: str):
    """VQA: ask the VLM to name the celeb in each headshot."""
    print("\n=== TEST 1: VQA (Who is in this image?) ===", flush=True)
    results = {}
    for celeb in CELEBS:
        cdir = bank / DIRNAME[celeb]
        photos = sorted(cdir.glob("*.png"))
        if not photos:
            print(f"[WARN] run_vqa: expected headshots in {cdir}, got none; "
                  f"fallback=skip celeb {celeb}", flush=True)
            continue
        rows = []
        for ph in photos:
            img = Image.open(ph).convert("RGB")
            answers = []
            for prompt in VQA_PROMPTS:
                inputs = processor(text=prompt, images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                ilen = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    out_ids = paligemma.generate(**inputs, max_new_tokens=24,
                                                 do_sample=False)
                ans = processor.decode(out_ids[0, ilen:],
                                       skip_special_tokens=True).strip().lower()
                answers.append(ans)
            hit = any(a in " | ".join(answers) for a in ALIASES[celeb])
            rows.append({"photo": ph.name, "answers": answers, "hit": hit})
            print(f"  {celeb:13s} {ph.name:16s} -> {answers}  hit={hit}",
                  flush=True)
        hr = sum(r["hit"] for r in rows) / max(1, len(rows))
        results[celeb] = {"hit_rate": hr, "n": len(rows), "rows": rows}
        print(f"[vqa] {celeb}: {hr:.0%} hit-rate ({len(rows)} photos)", flush=True)
    return results


def run_attention(paligemma, processor, bank: Path, device: str, out: Path):
    """Attention shift: does name-token->patch argmax track the prompted celeb?"""
    print("\n=== TEST 2: attention shift on 3-up composite (approx, no RoPE) ===",
          flush=True)
    text_model = paligemma.model.language_model
    cfg = text_model.config
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)

    tiles = []
    for celeb in CELEBS:
        ph = sorted((bank / DIRNAME[celeb]).glob("*.png"))[0]
        tiles.append(Image.open(ph).convert("RGB").resize((224, 224)))
    scene = Image.new("RGB", (224 * 3, 224))
    for i, t in enumerate(tiles):
        scene.paste(t, (i * 224, 0))
    scene.save(out / "scene_composite.png")
    print(f"[attn] composite scene (L={CELEBS[0]} M={CELEBS[1]} R={CELEBS[2]}) "
          f"-> {out / 'scene_composite.png'}", flush=True)

    cap: dict[int, dict] = {n: {} for n in ATTN_LAYERS}
    handles = []
    for n in ATTN_LAYERS:
        attn = text_model.layers[n].self_attn
        handles.append(attn.q_proj.register_forward_hook(
            lambda m, i, o, n=n: cap[n].__setitem__("q", o.detach())))
        handles.append(attn.k_proj.register_forward_hook(
            lambda m, i, o, n=n: cap[n].__setitem__("k", o.detach())))

    tok = processor.tokenizer
    summary = []
    for celeb in CELEBS:
        prompt = f"Where is {NAME[celeb]} in this image?"
        inputs = processor(text=prompt, images=scene, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        ids = inputs["input_ids"][0].cpu().tolist()
        ns, ne = _find_name_span(tok, ids, NAME[celeb])
        if ns is None:
            print(f"[WARN] run_attention: name {NAME[celeb]!r} not found in "
                  f"tokenized prompt; fallback=skip", flush=True)
            continue
        for n in ATTN_LAYERS:
            cap[n].clear()
        with torch.inference_mode():
            paligemma(**inputs)
        for n in ATTN_LAYERS:
            q = cap[n].get("q")
            k = cap[n].get("k")
            if q is None or k is None:
                print(f"[WARN] layer {n}: q/k hook did not fire for {celeb}; "
                      f"fallback=skip layer", flush=True)
                continue
            q = q.float()
            k = k.float()
            B, L, _ = q.shape
            q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
            k = k.view(B, L, n_kv, head_dim).transpose(1, 2)
            if n_kv != n_heads:
                k = k.repeat_interleave(n_heads // n_kv, dim=1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = scores.softmax(dim=-1).mean(dim=1)[0]  # (L, L)
            row = attn[ns:ne, :NUM_PATCHES].mean(dim=0)
            grid = row.view(GRID, GRID).cpu().numpy()
            ar, ac = divmod(int(row.argmax().item()), GRID)
            third = "LEFT" if ac < 5 else ("MID" if ac < 11 else "RIGHT")
            expect = {"yann_lecun": "LEFT", "barack_obama": "MID",
                      "taylor_swift": "RIGHT"}[celeb]
            ok = third == expect
            print(f"[attn] {celeb:13s} layer {n:2d}  argmax=({ar:2d},{ac:2d}) "
                  f"third={third:5s} expect={expect:5s} {'OK' if ok else 'x'}",
                  flush=True)
            summary.append((celeb, n, ar, ac, third, expect, ok))
            gn = (grid - grid.min()) / (grid.max() - grid.min() + 1e-9)
            Image.fromarray((gn * 255).astype(np.uint8)).resize(
                (224 * 3, 224)).save(out / f"{DIRNAME[celeb]}_layer{n:02d}.png")
    for h in handles:
        h.remove()

    by_layer: dict[int, set] = {}
    for celeb, n, ar, ac, *_ in summary:
        by_layer.setdefault(n, set()).add((ar, ac))
    moving = sorted(n for n, pts in by_layer.items() if len(pts) > 1)
    correct = sum(1 for *_, ok in summary if ok)
    print(f"\n[attn] layers where argmax shifts across celebs: {moving or 'NONE'}")
    print(f"[attn] argmax in correct third: {correct}/{len(summary)}", flush=True)
    return {"rows": [
        {"celeb": c, "layer": n, "argmax": [ar, ac], "third": th,
         "expect": ex, "ok": ok} for c, n, ar, ac, th, ex, ok in summary],
        "moving_layers": moving, "correct_thirds": correct,
        "total": len(summary)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True,
                   help="Pi0.5 checkpoint: HF repo id or local dir")
    p.add_argument("--revision", default=None)
    p.add_argument("--celeb-bank", default=str(Path.home() / "celeb_probe"),
                   help="dir with <celeb>/*.png headshots (lecun/obama/swift)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(Path.home() / "warm_vlm_probe"))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bank = Path(args.celeb_bank)
    if not bank.exists():
        print(f"[error] celeb bank not found: {bank}", file=sys.stderr)
        return 2

    repo = args.repo
    if args.revision:
        # PI05Policy.from_pretrained takes revision via kwarg; pass through dir.
        pass

    from transformers import AutoProcessor
    print(f"[load] processor {PROC_REPO}", flush=True)
    processor = AutoProcessor.from_pretrained(PROC_REPO)

    if args.revision:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        print(f"[load] PI05Policy.from_pretrained({repo}@{args.revision}) ...",
              flush=True)
        policy = PI05Policy.from_pretrained(repo, revision=args.revision)
        paligemma = policy.model.paligemma_with_expert.paligemma
        paligemma = paligemma.to(device=args.device, dtype=torch.float32).eval()
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        paligemma = load_paligemma(repo, args.device)

    vqa = run_vqa(paligemma, processor, bank, args.device)
    attn = run_attention(paligemma, processor, bank, args.device, out)

    rates = [vqa[c]["hit_rate"] for c in CELEBS if c in vqa]
    avg = sum(rates) / max(1, len(rates))
    verdict = {
        "repo": repo, "revision": args.revision,
        "vqa_avg_hit_rate": avg, "vqa": vqa, "attention": attn,
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 60)
    print(f"VQA average hit-rate across 3 celebs: {avg:.0%}")
    print(f"attention argmax correct-third: {attn['correct_thirds']}/{attn['total']}")
    print("=" * 60)
    if avg >= 0.5:
        print("PASS -- the VLM binds celeb names to faces. Path B is viable:")
        print("       a Pi0.5 policy with this VLM frozen can discriminate celebs.")
    elif avg >= 0.2:
        print("MARGINAL -- weak name binding. Path B risky; more VQA pretrain "
              "or KLAL supervision recommended.")
    else:
        print("FAIL -- the VLM does NOT recognise these celebs. Path B with the "
              "VLM frozen will not produce a face-discriminating policy.")
    print(f"\nSaved: {out / 'verdict.json'}  (+ attention heatmaps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
