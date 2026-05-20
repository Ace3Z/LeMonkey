# M2 fix + Track E path verification — 2026-05-20

The Pi0.5 Track E run (M2 ArcFace identity loss + KLAL attention supervision)
had M2 silently not training. This documents the bug, the fix, the audit of
the whole training path, and the config changes for the retrain.

## TL;DR

- **Bug 1 (fixed):** the Pi0.5 M2 capture hook stored `h.detach().clone()`,
  cutting the autograd graph — M2's loss was computed + logged but trained
  nothing; `mean_cos` sat flat at ~0.
- **Bug 2 (fixed):** even after bug 1, M2 was too weak at `M2_LAMBDA=0.2`
  (Pi0.5's lr is 5× below SmolVLA's). Raised to 1.0.
- **Smoke test — PASS:** with both fixes, `mean_cos` climbs +0.011 → +0.562
  over 800 steps (was dead flat before). M2 trains; KLAL trains.
- **Audit (3 parallel agents):** KLAL loss math correct, M2 supervision
  targets correct. One real fix: KLAL supervised a *frozen* layer (6) —
  dropped it.
- **Not yet verified:** that 25k steps produces name→face attention. The
  losses train correctly; the outcome needs a ~step-10k checkpoint probe.

## Bug 1 — M2 capture hook detached the gradient

`eval_3/aug/m2_pi05_hook.py` pre-hook stored `h.detach().clone()`. `.detach()`
severs autograd. `m2_align_loss` still computed numerically (so `m2_loss` /
`mean_cos` logged fine) but `backward()` reached no parameter. KLAL was
unaffected (separate hookset, no detach) — which is why KLAL trained while
M2 did not. Fixed to `holder.captured = h`, matching the working SmolVLA
hook (`m2_smolvla_hook.py`). Confirmed by an independent code audit + a
runtime diagnostic: `captured.requires_grad=True`, `m2.loss.grad_fn=SET`,
M2's gradient reaches layer 10.

## Bug 2 — M2_LAMBDA too small for Pi0.5

After bug 1 the gradient flowed but `mean_cos` still barely moved. SmolVLA's
M2 reached `mean_cos 0.88` at `M2_LAMBDA=0.2` with `lr=5e-5`; Pi0.5 trains at
`lr=1e-5` (5× smaller), so M2's effective step was 5× too small. Raised
`M2_LAMBDA` 0.2 → 1.0.

## Smoke test — M2_LAMBDA=1.0 (a-toy-pi05 H100)

| step | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|---|
| `mean_cos` | .011 | .015 | .025 | .043 | .081 | .193 | .405 | .505 | **.562** |
| KLAL | 3.60 | 1.68 | 1.27 | 1.03 | 0.99 | 0.72 | 0.64 | 0.55 | 0.55 |

`mean_cos` rises monotonically and accelerates — heading toward SmolVLA's
0.88. At `M2_LAMBDA=0.2` the same run was flat (−0.038 → −0.036 over 150
steps). **M2 and KLAL both train.**

## Path audit — 3 parallel review agents

| area | verdict |
|---|---|
| KLAL loss math (KL direction, Gaussian target, GQA attention recompute, gradient flow) | correct |
| M2 supervision targets (bbox→patch geometry, slot→celeb mapping, ArcFace centroids) | correct |
| KLAL name-token matching | works — `[WARN] no name_token_positions` never fired; KLAL non-zero + decreasing every logged step |
| KLAL supervised layers | **bug: KLAL@layer6 is dead** — layer 6 is frozen by the partial-freeze, so KLAL@6 trained 0 params and diluted the 4-layer-averaged loss ~25% |

Agent concerns examined and ruled out:
- *Partial-freeze blocks learning* — no: SmolVLA used the identical
  freeze-below-capture pattern and M2 still reached 0.88.
- *KLAL softmax distorted by padding columns* — no: the slice-then-
  renormalize cancels it exactly.
- *`lang_offset` miscounts image streams* — verified correct: Track E has 5
  image streams (camera1 + reference + 3 empties) matching `image_features`.
- *KLAL name-matching broken by the `Task:…Action:` prompt rewrite* — no:
  SentencePiece is context-free for interior substrings and the wrapper
  tries the leading-space variant; empirically the no-match WARN never fired.

Flagged, not changed: KLAL recomputes attention **without RoPE** — a
documented design choice consistent with every probe we've run; worth
checking against the WACV 2026 paper (arXiv:2511.12738) before the final run.

## Config changes for the retrain

In `run_training_track_E_pi05_{3celeb,200celeb}.sh` and
`lerobot_train_with_m2_pi05.py`:
- `M2_LAMBDA` 0.2 → **1.0**
- `KLAL_LAYERS` `6,10,14,17` → **`10,14,17`** (drop the frozen layer)

A step-0 `[m2-diag]` gradient-flow check was added to the wrapper as a
permanent regression guard against bug 1 recurring.

## What is and isn't verified

**Verified:** the detach fix (gradient flows — diagnostic + agent + the
rising `mean_cos`); M2 trains; KLAL trains; KLAL math + M2 targets correct;
the launch path runs 800+ steps with no error.

**NOT verified:** that the trained VLM actually binds celebrity names to
faces. That is the *outcome* of the full run, not something a smoke test can
show. The gating doc set step ~10k as the discrimination milestone.

**Next test:** probe the attention map of a ~step-2k–10k checkpoint with
`attention_map_probe_pi05.py`. The buggy step-1000 probe
([`2026-05-20_vlm_face_detection_probes`](2026-05-20_vlm_face_detection_probes/README.md))
showed flat, prompt-invariant attention; a fixed-run checkpoint should show
name-conditioned movement if KLAL is working. This is the decisive
"detects faces" test and can only be run on a trained checkpoint.
