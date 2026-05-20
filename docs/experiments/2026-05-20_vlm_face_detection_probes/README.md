# VLM face-detection probes — 2026-05-20

Two VLMs tested for whether their name-token attention binds celebrity
names to faces — the property a face-discriminating policy depends on.

1. **`HBOrtiz/pi05_paligemma_celeb_warm`** — a teammate's Pi0.5 checkpoint
   whose PaliGemma VLM was meant to be celeb-warm-started. Tested to gate
   **Path B** (Pi0.5 with the VLM *frozen*, `train_expert_only=True` — see
   [`TODO.md`](../../../TODO.md)).
2. **`HBOrtiz/smolvla_eval3_track_D_m2_mahbod@step-25000`** — our SmolVLA
   Track D run (M2 ArcFace cosine distillation), final 25k checkpoint. A
   re-test of the [step-10000 attention failure](../2026-05-19_attention_probe_step10000/README.md).

Both run on Brev `time2sleep` (A100 80GB).

## TL;DR — neither VLM detects faces

| VLM | name→face attention | verdict |
|---|---|---|
| `pi05_paligemma_celeb_warm` | argmax constant across all 3 celeb prompts at 3 of 4 layers (sink at `(0,1)`/`(15,3)`); layer-14 "movement" is argmax jitter on a diffuse, **prompt-invariant** heatmap | **does not detect faces** |
| SmolVLA Track D step-25000 | argmax `(1,7)` for all 3 celebs at all layers — identical to step-10000 | **does not detect faces; no improvement over step-10000** |

**Path B is not viable with `pi05_paligemma_celeb_warm`.** A frozen VLM that
does not bind celeb names to faces cannot produce a celeb-discriminating
policy. The path that remains supported is **Track E** (Pi0.5 + M2 + KLAL —
*train* the VLM with bbox-supervised attention loss), currently running on
`a-toy-pi05`.

## Method

The trustworthy test for a Pi0.5-embedded VLM is the **attention probe run
through Pi0.5's real prefix-LM forward** (`policy.select_action`), hooking
`q_proj`/`k_proj` on the PaliGemma language-model layers and measuring
name-token → image-patch attention. Script:
[`attention_map_probe_pi05.py`](../../../eval_3/scripts/attention_map_probe_pi05.py)
(extended this session with `--preprocessor-repo`, so the warm checkpoint —
which ships only `config.json` + `model.safetensors` — can borrow the
preprocessor from `lerobot/pi05_base`; the prefix is image+language only, so
the state/action stat mismatch does not affect it).

SmolVLA Track D used the existing
[`attention_map_probe.py`](../../../eval_3/scripts/attention_map_probe.py),
identical invocation to the step-10000 run (episode 100, frame 10, layers
9/11/13/15) — so the two are directly comparable.

### Discarded first attempt — VQA via `generate()`

The first warm-VLM probe ([`probe_pi05_warm_vlm.py`](../../../eval_3/scripts/probe_pi05_warm_vlm.py))
asked the VLM "Who is in this image?" via `PaliGemmaForConditionalGeneration.generate()`.
It returned **non-text gibberish** (`'.┲쫑'`, `',ᅫ'`) on every headshot —
0/14. Gibberish (not wrong names) means the autoregressive text path through
LeRobot's custom `PiGemma` decoder is unreliable, so this result was
**discarded** rather than reported as a face-recognition failure. It is a
weak corroborating negative only.

## Results — `pi05_paligemma_celeb_warm`

Name-token attention argmax (row, col) on the 16×16 PaliGemma patch grid,
3-celeb scene (LeCun left, Obama centre, Swift right):

| layer | swift | obama | lecun | shifts? |
|-------|-------|-------|-------|---------|
| 6  | (0,1)  | (0,1)  | (0,1)  | ✗ constant |
| 10 | (0,1)  | (0,1)  | (0,1)  | ✗ constant |
| 14 | (3,9)  | (3,9)  | (6,1)  | ~ lecun jitters; swift=obama |
| 17 | (15,3) | (15,3) | (15,3) | ✗ constant |

Visual gate (`warm_paligemma/*_layer14_overlay.png`): the layer-14 heatmaps
for swift / obama / lecun are visually indistinguishable — a faint diffuse
wash, no bright lock on any face. The argmax difference at layer 14 is noise
on a flat distribution, not name-conditioned localization.

`attention_map_probe_pi05.py`'s built-in gating line prints "PASS" because
its criterion is the lenient G1 one ("does argmax differ at *any* layer" →
is there a channel for M2+KLAL to *train*). That is **not** the question
here. For Path B the VLM is *frozen*, so it must already localize — and it
does not.

## Results — SmolVLA Track D step-25000

Name-token → camera1-patch attention argmax on the 8×8 grid:

| layer | swift | obama | lecun |
|-------|-------|-------|-------|
| 9  | (5,2) | (1,7) | (1,7) |
| 11 | (1,7) | (1,7) | (1,7) |
| 13 | (1,7) | (1,7) | (1,7) |
| 15 | (1,7) | (1,7) | (1,7) |

11 of 12 cells argmax at `(1,7)` — a fixed background patch top-right of
Swift's photo. `max_attn`/`entropy` values match step-10000 to within noise
(e.g. swift L15: 0.077/2.22 here vs 0.077/2.06 at step-10000). Visual gate
(`smolvla_track_d_25k/*_layer15_overlay.png`): the three celeb overlays are
identical.

**15k more steps of M2 ArcFace training changed nothing for cross-modal
attention.** This is expected and consistent with the step-10000 diagnosis:
M2 shapes face-patch *hidden states* to match identity centroids
(`mean_cos ≈ 0.88` in training logs — M2 succeeds at its own objective) but
does **not** force the name token to *attend* to those patches. That gap is
exactly what KLAL was added to close.

## Implications

- **Path B (Pi0.5, VLM frozen) — drop.** The warm VLM does not localize
  celebs by name; freezing it cannot yield a face-discriminating policy.
- **SmolVLA Track D — confirmed dead end** for face discrimination. M2
  alone is insufficient (representation shaping ≠ cross-modal attention).
  The step-25000 checkpoint is on HF (`@step-25000` + `main`) for the
  record / the `+20` baseline behaviour, but not as a celeb-discriminator.
- **Track E (Pi0.5 + M2 + KLAL) is the remaining supported path.** KLAL
  directly supervises name-token→face-patch attention from bbox labels —
  it does not depend on any pre-existing celeb knowledge in the VLM.
  Running on `a-toy-pi05`; step ~1k/25k at time of writing (~50h ETA).

## Caveats

- The attention recompute omits RoPE (same as the step-10000 probe and the
  `attention_map_probe_pi05.py` design — content-only attention is the
  signal of interest). The visual heatmaps corroborate the argmax, so the
  conclusion does not hinge on RoPE.
- Single scene frame (episode 100, frame 10).
- No same-probe vanilla baseline: G1 probed vanilla PaliGemma with a
  *different* script (`attention_map_probe_paligemma.py`, true
  `output_attentions`, with RoPE), so the warm-vs-vanilla delta is not
  quantified here. Running `attention_map_probe_pi05.py` on
  `lerobot/pi05_base` would close that gap — it would not change the Path B
  verdict (the warm VLM does not localize, regardless of the delta).

## Update 09:47 — Track E step-1000 probe (the live KLAL run)

Probed the running **Track E** checkpoint (`HBOrtiz/pi05_eval3_track_E_m2_mahbod`,
Pi0.5 + M2 + KLAL on `a-toy-pi05`) at its latest pushed checkpoint, `step-1000`.

**Training health at the time of probing** (step ~1.4k/25k):

- KLAL loss: 3.6 → 0.99 → 0.47 → **0.42** — decreasing (the attention-supervision
  loss is engaged), but decelerating.
- M2 `mean_cos`: **flat at ~0** (−0.001 → −0.005 over steps 1050→1200). `M2_LAMBDA=0.2`
  is set and active, so M2 *should* contribute — SmolVLA's M2 reached +0.74 by a
  comparable step. **The Pi0.5 M2 port (`m2_pi05_*`) appears bugged** — the run is
  effectively "Pi0.5 + KLAL", not "+ M2 + KLAL". Root cause not yet found.
- 7.6 s/step, ~49h ETA for the full 25k. The gating doc's discrimination
  milestone is step ~10k (~18h out).

**Attention at step-1000** (16×16 grid, name-token → camera1 patches; LeCun cols
0–5, Obama 5–10, Swift 10–15):

| layer | swift | obama | lecun | shifts? |
|-------|-------|-------|-------|---------|
| 6  | (0,1)   | (6,8)  | (9,6)  | ✓ all 3 differ |
| 10 | (10,8)  | (11,8) | (11,8) | ✓ swift differs |
| 14 | (10,9)  | (10,9) | (10,9) | ✗ constant |
| 17 | (9,4)   | (9,4)  | (8,5)  | ✓ lecun differs |

argmax differs across celebs at 3 of 4 layers — **more movement than the
sink-locked warm VLM** (which differed at only 1/4). But the **visual gate
fails**: the `track_e_step1000/*_layer{06,10}_overlay.png` heatmaps are a diffuse
warm wash, near-identical across the 3 celeb prompts, brightest near the coke can
— **no face lock on any celeb**. `max_attn` 0.001–0.003 is *below* uniform
(1/256 ≈ 0.0039): the name token barely attends to image patches at all. Only
Obama's argmax lands roughly in-column (consistent with G2 — PaliGemma half-knows
Obama).

**Verdict — too early to call, not yet detecting celebs.** Step 1000 is 10% of
the planned step-10k discrimination checkpoint. KLAL is engaged (loss dropping,
argmax starting to move) but nowhere near converged; the attention does not yet
localize celebs by name. This is **expected this early** — it is neither a pass
nor the documented "kill the run" failure. The real go/no-go is **step ~10k**
(gating doc: "If step 10k fails the discrimination test, kill the run").

Two concrete problems independent of the probe: **M2 is not training** (bugged
port), and the run is **slow** (~49h).

Probe-script bug fixed this session: `attention_map_probe_pi05.py` fed a 32-d
state (`max_state_dim`) into the preprocessor, whose quantile normalizer is sized
to the raw 6-d SO-101 state → crash. Fixed to read the dim from
`config.input_features["observation.state"]`.

## Files

- `warm_paligemma/` — 25 PNGs: `input.png` + `<celeb>_layer{06,10,14,17}_{heatmap,overlay}.png`
- `smolvla_track_d_25k/` — 25 PNGs: `input.png` + `<celeb>_layer{09,11,13,15}_{heatmap,overlay}.png`
- `track_e_step1000/` — 25 PNGs: `input.png` + `<celeb>_layer{06,10,14,17}_{heatmap,overlay}.png`
- `warm_paligemma_v2/` — 25 PNGs: `input.png` + `<celeb>_layer{06,10,14,17}_{heatmap,overlay}.png`

## Update 13:00 — pi05_paligemma_celeb_warm_v2 probe

Probed `HBOrtiz/pi05_paligemma_celeb_warm_v2` — the teammate's v2 of the warm
VLM (full Pi0.5, 4.14B). Real weights confirmed loaded (`✓ Loaded state dict`
/ `All keys loaded successfully` — NOT the random-weights failure mode).

Name-token → image-patch argmax (16×16 grid; LeCun cols 0-5, Obama 5-10,
Swift 10-15):

| layer | swift | obama | lecun | shifts? |
|-------|-------|-------|-------|---------|
| 6  | (6,8)  | (6,8)  | (0,1)  | ~ lecun only; swift = obama |
| 10 | (0,1)  | (0,1)  | (0,1)  | ✗ constant |
| 14 | (0,1)  | (0,1)  | (0,1)  | ✗ constant |
| 17 | (15,3) | (15,3) | (15,3) | ✗ constant |

Same sink-locked pattern as v1: argmax constant across all 3 celebs at 3 of
4 layers; the one layer with any movement (6) has swift = obama. `max_attn`
0.002-0.045 — below uniform (1/256). Visual gate (`warm_paligemma_v2/*`):
diffuse warm wash, near-identical across the 3 prompts, no lock on any face.

**Verdict: v2 does NOT detect faces** — no improvement over v1. Path B with
v2's VLM frozen is not viable. The teammate's warm-start (v1 or v2) does not
produce name→face attention; only training the attention directly (Track E
KLAL) addresses it.
