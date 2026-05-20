# 2026-05-20 — KLAL (L_attn) ported to SmolVLA co-training

## What was built

Added the KL attention-supervision loss (`L_attn`) to the SmolVLA VL co-train
(`eval_3/scripts/smolvla_cotrain/`), branch `hans-smolvla-cotrain-klal` (off
`feat/cotrain-smolvla-darius`). New file `smolvla_klal.py` + wiring in
`cotrain.py` + `launch.sh`.

The base script (Darius) already co-trains `L_action` (flow-matching on teleop)
+ `L_name` (VQA CE naming) with a shared optimizer. This adds the third loss:
`loss = vqa + klal_lam · klal` on VL steps.

## Why a port rather than new code

KLAL already exists, audited + smoke-tested, on `dev/m2-arcface-toolkit`
(`eval_3/aug/m2_klal.py`) — but for **Pi0.5 / PaliGemma**. This is the SmolVLM2
port. The attention-routing diagnosis (name-token attention sink-locked to a
constant background patch) is corroborated for both Pi0.5 and SmolVLA, so the
same fix applies.

## SmolVLM2 specifics — all verified from source/config (CLAUDE.md §8)

| Item | PaliGemma (reference) | SmolVLM2-500M (this port) | Source |
|---|---|---|---|
| Image tokens | 256 (16×16) | **64 (8×8)** | config `scale_factor=4`, 512/16=32² ÷4²=64; pixel_shuffle modeling_smolvlm.py:437 |
| Text arch | gemma | **llama** → rotate_half RoPE | config text_config.model_type |
| Heads | — | **15 attn / 5 KV (GQA 3×), head_dim 64** | config text_config |
| Image cols | fixed prefix slice | **inline `input_ids==49190`** | config image_token_id |
| Layers | 18 | **truncated to 16** → capture ≤15 | smolvlm_with_expert.py:90 |
| Attention | bidirectional prefix-LM | **causal**; name tokens after image tokens, so image cols always visible; slice+renorm well-posed | — |

KLAL recomputes `softmax(QK^T)` from hooked q_proj/k_proj outputs, so it works
under SDPA — no eager attention / `output_attentions` needed (same as the
reference). Gradients flow into the hooked projections (q/k captured undetached;
RoPE cos/sin detached).

## Bug caught during CPU geometry check

The `[0,0,0,0]` "no bbox" sentinel originally produced a 1-patch mask at corner
(0,0) instead of an empty mask — which would have trained attention toward the
top-left corner, *the exact sink-lock failure KLAL fixes*. Fixed: zero-area
boxes now return an empty mask → KLAL treats them as "no supervision".

CPU unit check (no model needed) confirms left/mid/right bboxes map to grid
columns 1/3/6 with Gaussian peaks at the bbox centroid, and empty bbox → uniform.

## Two experiments to run in parallel (we have the compute)

- **A — cheap ObjectVLA bbox-as-text**: `CAPTION_FILTER=location_explicit`, no
  KLAL. Trains via VQA CE on captions that encode the bbox as text. The PhD's
  "try cheap first."
- **B — KLAL**: `CAPTION_FILTER=qa_grounded USE_KLAL=1 KLAL_LAM=0.1`. Bare-name
  captions + the KL attention loss. The head-on fix.

Launch commands in `eval_3/scripts/smolvla_cotrain/README.md`.

## Update — VLM fine-tune scope ablation (`--lora_scope`)

Added a `--lora_scope {full,wide,qk}` flag for cluster ablations. Grounded in
the KLAL paper (arXiv:2511.12738), which **full-fine-tunes** (LLaVA-1.5-7B,
Qwen2.5-VL-7B) — it uses no LoRA, supervises **all** layers (Eq.4), last-answer-
token attention, **λ=1**. So the paper gives no LoRA-scope guidance; the LoRA
arms are our cost-reduction bets.

Three arms, run order **A → B → (C if spare compute)**:
- **A. full + KLAL** — the paper's exact setup; highest chance of working, sets
  the ceiling, λ=1. Run first: answers "does KLAL fix routing at all?"
- **B. wide LoRA (q/k/v/o+MLP) + KLAL** — "can a cheap adapter match full?";
  most likely LoRA arm to match A, so the informative second run. λ≈0.1.
- **C. q/k-only LoRA + KLAL** — "is reshaping attention alone enough?"; logically
  after B (if wide can't match full, q/k won't). λ≈0.1.

All arms warm-start from `HBOrtiz/smolvlm2_lora_celebs` (celeb prior already in
the weights → KLAL only has to fix the attention routing). full-vs-LoRA does
**not** affect the smallest-model bonus (inference param count is identical).

Two deviations from the paper, both deliberate + documented in code:
- KLAL default `--klal_layers=all` (16 layers) matches the paper; was (6,9,12,15).
- We average over all name-token rows, not just the last answer token (better for
  a multi-token name); and use a centroid Gaussian, not the bbox center-line
  (better for compact face boxes).

## Status / open items

- `smolvla_klal.py` geometry: **CPU-unit-checked**.
- Attention recompute (RoPE capture, GQA expand, causal renorm): **statically
  written, NOT GPU-smoke-tested, no second-reviewer pass.** Per CLAUDE.md §9 this
  needs a parallel-agent review + the 200-step smoke gates before a long run.
- `λ` tuning: start 0.05–0.2; KLAL can dominate VQA CE under low-rank LoRA.
- Capture layers default (6,9,12,15) — mid-late within the 16-layer truncation;
  the Pi0.5 reference used (6,10,14,17) on 18 layers, scaled down here.
