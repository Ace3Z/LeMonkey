# Attention-map probe — step-10000 checkpoint

**Date:** 2026-05-19
**Checkpoint:** `HBOrtiz/smolvla_eval3_track_D_m2_mahbod@step-10000`
**Trigger:** First Strix rollout on this checkpoint failed — the arm did not
move toward the named celeb's printout. This probe was built to
visualize *why*.
**Script:** [`eval_3/scripts/attention_map_probe.py`](../../../eval_3/scripts/attention_map_probe.py)
**Command run:** `python eval_3/scripts/attention_map_probe.py --revision step-10000 --layers 9 11 13 15 --episode 100 --frame 10`

## TL;DR

The name token never selectively attends to the prompted celeb's face.
Argmax patch is **(1, 7) — top-right corner, background, above-right
of Swift's photo — for every prompt at every probed layer**. The
attention is also weaker than uniform (1/64 = 0.016) at layers 9-13.

This mechanistically explains the Strix failure: the policy produces
actions sensitive to the prompt embedding, but with no spatial visual
anchor — it doesn't *know* which printout the named celeb is on.

## What's in this folder

```
input.png                              — the camera1 frame fed to the policy
                                        (LeCun left, Obama center, Swift right,
                                        coke can top-middle)

<celeb>_layer{NN}_heatmap.png          — raw attention heatmap, 8x8 patch
                                        grid upsampled bilinear to 480x640,
                                        brighter = more name->patch attention
<celeb>_layer{NN}_overlay.png          — same heatmap blended (red-orange,
                                        alpha=0.45) onto input.png
```

3 prompts × 4 layers × 2 views (heatmap, overlay) + 1 input = 25 PNGs.

## How to view

These are plain PNGs — open in Preview / any image viewer. Suggested
read order:

1. `input.png` — establish the scene
2. `swift_layer15_overlay.png` — what working grounding *should* look
   like is a bright blob over Swift's face. It doesn't.
3. `obama_layer15_overlay.png`, `lecun_layer15_overlay.png` — same blob
   location, nearly identical to Swift's. The prompt does not steer
   attention.
4. `*_layer09_heatmap.png` — at the M2 hook layer, attention is below
   uniform; the (1,7) sink dominates already.

## Numerical summary

attention from name-token rows to first-64 camera1 patches:

| prompt | layer 9 | layer 11 | layer 13 | layer 15 | argmax |
|---|---|---|---|---|---|
| swift  | max 0.0052, ent 0.42 | 0.0075, 0.46 | 0.0114, 1.04 | 0.0773, 2.06 | (1,7) all |
| obama  | max 0.0240, ent 0.53 | 0.0087, 0.59 | 0.0085, 1.04 | 0.0562, 2.10 | (1,7) all |
| lecun  | max 0.0127, ent 0.34 | 0.0076, 0.48 | 0.0154, 1.15 | 0.0712, 2.13 | (1,7) all |

Uniform-over-64 baseline: 1/64 ≈ 0.0156. Layers 9-13 are below baseline
— the name token is mostly attending to other language tokens, not
to image patches.

## Method caveats

- RoPE skipped. RoPE rotates Q/K by position-dependent angles; for the
  ~130-token gap between image patches (positions 0-127) and the
  celeb-name token (around position 132-137), RoPE contributes a
  position bias that obscures the content signal we want to inspect.
  Without RoPE, the visualized attention is purely content-based —
  which is the relevant signal for "is the name semantically bound to
  this face patch?"
- We probed only self-attention inside the VLM. The action expert
  cross-attends to the prefix from the suffix side; that's a separate
  attention path and was *not* probed here. It could still be
  grounding via the action expert — but the failed Strix rollout
  argues against this too.
- 8x8 patch grid (after SigLIP + SmolVLM2's pixel shuffle 32x32 → 8x8).
  Each patch covers ~60x80 px of the 480x640 frame.

## Diagnosis

M2 ArcFace distillation shaped face-patch hidden states to match
identity centroids (mean_cos ≈ 0.85 in training logs) but it did NOT
force the language name token to *attend* to those face patches. The
VLM self-attention treats the name token as a within-language signal;
the cross-modal binding never formed.

This is the "language pathway off-axis" failure the trajectory
reviewer flagged. The sanity checks (action vectors differ across
prompts) were necessary but not sufficient: prompt-dependent variation
in the action expert's output doesn't prove the variation is *grounded
in visual identity*.

## Implication for the project

SmolVLA's VLM is too small / too generic to bind celebrity names to
visual identity even with M2 supervision. Two candidate fixes:

1. **Pivot the next run to Pi0.5 + M2.** PaliGemma's SigLIP-So400M is
   not face-blurred and its Gemma-2B LM has a stronger celeb prior
   (TriviaQA 53.2 vs SmolLM2's 36.7). Cost: ~12 dev-h + 4-8 Brev-h.
2. **Use Hans's warm VLM when published.** If `HansOrtiz/smolvlm2_celeb_warm`
   already binds celeb names to face features at LoRA-fine-tuning time,
   the action expert can read that binding without M2 shaping. Cost:
   0 dev-h + 6 Brev-h, but blocked on Hans.

Captured in [`TODO.md`](../../../TODO.md) under Day-3 contingencies.
