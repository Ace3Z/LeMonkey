# Eval 3 — Training Strategy (2026-05-17)

**Status:** active. Sits above [`eval_3/aug/STRATEGY_v3.md`](aug/STRATEGY_v3.md) (augmentation-specific) and beneath [`docs/EVAL_3_OPTIONS.md`](../docs/EVAL_3_OPTIONS.md) (full option space + decision tree). This document is the **chosen plan** — what we are actually training, in what order, why, and with what fallbacks.

> Operational checklist for this plan lives in [`/TODO.md`](../TODO.md).

---

## 1. What we are doing

Training **4 parallel tracks** on Brev today. Two are bonus-preserving (SmolVLA-450M), two are capacity-bet (Pi0.5-3.3B). Best-of-4 ships on demo day.

| Track | Recipe | Bonus | Cost | Role |
|---|---|---|---|---|
| **A** | SmolVLA-boost-v2 (refs + print-aug + ArcFace distillation) | +20 | ~5h | **Primary bonus-preserving** |
| **B** | Pi0.5 + ArcFace distillation (hybrid) | +16 | ~30h | Max-effort capacity-bet |
| **C** | Pi0.5 + image-as-prompt (vanilla) | +16 | ~27h | Pure capacity hypothesis test |
| **D** | SmolVLA + 3-celeb only + name-only (baseline) | +20 | ~6h | **Floor / safety net** |

---

## 2. Why this combination

### 2.1 The failure mode we are fixing

SmolVLA-450M's image-as-prompt v1 run failed on Strix 2026-05-17. The failure is **not** a wiring bug (verified: camera2 reaches the model intact), **not** a wrist-cam shift (verified: setup matches training), and **not** simple positional bias on a single celeb (verified: rotating Swift through L/M/R produces the same wrong spot). The remaining diagnosis is two compounding gaps:

- **(G1) Domain gap** — training reference photos are magazine/web style; eval-day workspace is paper printouts. Published face-recognition literature ([arxiv 2404.06559 §4](https://arxiv.org/html/2404.06559v2)) measures this transformation at +5-16% ArcFace FMR shift.
- **(G2) Representation gap** — SmolVLA's vision tower compresses to 64 visual tokens per image via pixel-shuffle ([SmolVLM, arxiv 2504.05299](https://arxiv.org/html/2504.05299v1)); for a face occupying ~30% of the image, that's ~10-20 identity-bearing tokens. SigLIP was trained for image-text alignment, not face identity.

Tracks A and B target both gaps. Track C targets neither — it's the "does more capacity solve it implicitly" experiment. Track D side-steps the OOD case entirely by restricting the training distribution to the 3 known IID celebs.

### 2.2 Why we picked these 4 (and rejected the others)

The full option enumeration is in [`docs/EVAL_3_OPTIONS.md`](../docs/EVAL_3_OPTIONS.md) (15 options). The selection rationale:

- **Option 6 (= Track A)** is the best-evidence bonus-preserving surgical fix. The BlindVLA distillation pattern ([arxiv 2510.25616](https://arxiv.org/html/2510.25616v1)) has +12-24% OOD published gains; Augraphy-inspired print-emulation ([arxiv 2208.14558](https://arxiv.org/abs/2208.14558)) addresses the domain gap directly. Both have triple-source validation in `eval_3/aug/RESEARCH_v3_face_matching_rescue.md` §3, §4.
- **Option 12 (= Track B)** is the maximum-effort path that combines capacity *and* surgical fixes. If neither alone wins, this should.
- **Option 9 (= Track C)** is the cleanest test of the "small model is too weak" hypothesis. The team's prior was that capacity alone fixes it; Track C measures that directly.
- **Track D** is a new addition not in `EVAL_3_OPTIONS.md` — a 3-celeb-only safety baseline. Concedes the 3 OOD runs (16.7 pts max) by design but guarantees IID functionality even if all other tracks fail.

**Rejected (and why):**
- Options 7, 10 (name-only without VQA warm-start) — no evidence at 450M scale; depend on PaliGemma's WebLI prior alone.
- Options 11 (Pi0.5 + VQA + name-only) — needs the LeRobot multi-dataset blocker resolved, 3-5 days plumbing.
- Options 13, 14 — encompassed by our parallel-track strategy already.
- X-VLA, OpenVLA, TinyVLA, FlowerVLA — Brev integration cost, deploy VRAM, or rejected by the user.

---

## 3. The reasoning chain per track

### Track A: SmolVLA-boost-v2

**Loss equation** (from [BlindVLA equation 9, arxiv 2510.25616](https://arxiv.org/html/2510.25616v1)):
```
L = L_flow_matching + λ · L_align
L_align = − (1/k) · Σ_{j=1}^{k} cos( F.normalize(u_j), F.normalize(z_j) )
λ = 0.2
```
where `u_j` are the SigLIP `last_hidden_state` patch features (pre-connector, pre-pixel-shuffle) of camera2 reference image, projected through a small 2-layer MLP (1152 → 2048 → 512); `z_j` are the corresponding cached ArcFace embeddings (`buffalo_l` / `w600k_r50`, 512-dim). Mean-pool the patches inside a pre-computed RetinaFace mask, then compute cosine.

**Why each piece:**
- Cosine over L2: ArcFace lives on a hypersphere by construction ([arxiv 1801.07698](https://arxiv.org/abs/1801.07698)); cosine is the geometrically correct metric. L2 is sub-optimal — verified across BlindVLA Table 8, [Evaluation-Oriented KD CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Evaluation-Oriented_Knowledge_Distillation_for_Deep_Face_Recognition_CVPR_2022_paper.pdf), and [Unified-KD arxiv 2508.11376](https://arxiv.org/html/2508.11376v1).
- Patch-pre-connector injection: SmolVLM's pixel-shuffle merges 2×2 patches into 1 token, losing spatial granularity inside each block. Aligning before the merge preserves face-region resolution.
- λ=0.2: BlindVLA Table 8 reports this as the most stable choice in their sweep.
- Mask-gated: ArcFace produces *one* embedding per detected face, not per patch. Aligning non-face patches against a face embedding would silently corrupt SigLIP's grounding for everything else. Required.
- Camera2-only: camera1 (wrist) sees the workspace, not just faces. Applying the loss there would push SigLIP toward face-discriminative geometry on the workspace view too, degrading manipulation features.

**Plus the print-augmentation pipeline** for camera2 at training time:
1. Lab gamut compression (chroma scale 0.75-0.9, L clip 10-240)
2. Print MTF blur (σ 0.4-0.8 px)
3. Floyd-Steinberg color dither at 300-DPI equivalent
4. Perlin fBm grain (4 octaves, amplitude ±3)
5. Re-image at 224×224 + JPEG q70-90
6. Apply with p=0.7 (30% see clean)

Augmentation operator order matches Augraphy's ink → paper → post pipeline ([arxiv 2208.14558](https://arxiv.org/abs/2208.14558)). Each parameter is triple-sourced — see `eval_3/aug/RESEARCH_v3_face_matching_rescue.md` §3.

### Track B: Pi0.5 + ArcFace distillation

Same loss + injection pattern as Track A, ported to Pi0.5's PaliGemma-3B vision tower (SigLIP-So400m at 400M params, larger than SmolVLM's vision component). The Pi0.5 wrapper is in `third_party/lerobot/src/lerobot/policies/pi05/paligemma_with_expert.py`.

**Why we run both A and B:** they bracket the failure-mode hypothesis. If face-matching capacity is the bottleneck, Track B should outperform Track A. If small + surgical is enough, Track A wins on bonus.

### Track C: Pi0.5 + image-as-prompt vanilla

No surgical fixes. Just a bigger backbone on the same dataset and protocol as the original SmolVLA training. This isolates the **capacity** variable from the **technique** variable. Conclusions:

- If C ≥ A: scaling alone is enough; surgery is unnecessary.
- If C < A: surgery is the load-bearing intervention, and bigger backbones don't automatically solve face-matching.
- If C ≈ A: the failure was probably capacity *and* technique; Track B (which has both) is likely the winner.

### Track D: Stable 3-celeb baseline

Train SmolVLA from `lerobot/smolvla_base` on only the 178 base teleops (Swift/Obama/LeCun) with name-only prompts and `--policy.empty_cameras=2` (no reference stream). This is the same architecture that passed Eval 1 and Eval 2 on similar data scales.

**Worst case:** zero on the 3 OOD runs (we never saw those celebs), full on the 6 IID runs (3 TOY + 3 held-out). Expected 4-6/9 successful = 22-33 pts rollouts + 20 bonus = 42-53 pts total.

This is the **floor**. If A, B, C all collapse, D ships.

---

## 4. Decision criteria for picking the demo-day checkpoint

After training all 4, run a structured dry-run:

| Test | Composition |
|---|---|
| TOY (runs 1-3 analog) | 1 rollout per IID celeb with the exact TOY-PDF print |
| Held-out IID (runs 4-6 analog) | 1 rollout per IID celeb with a different photo of theirs |
| OOD (runs 7-9 analog) | 1 rollout each with 3 OOD celebs (e.g. Federer, Bezos, Beyoncé — pick popular ones in our pool) |

Score per checkpoint:
```
total_pts = (TOY/3 + heldout/3 + OOD/3) × 50 + bonus
```
Pick the highest. Ties broken by lower variance (3-rollout standard deviation).

---

## 5. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| TAs disallow image-as-prompt input at inference | medium | A, B, C invalid for eval | Slack now; Track D is text-only and always valid |
| Pi0.5 doesn't fit Strix 16 GB at 30 Hz | medium | B, C undeployable | Empirical probe before relying on them; SmolVLA fallback (A or D) always fits |
| ArcFace distillation doesn't transfer to printed cutouts | medium | A, B underperform | Pre-filter teacher embeddings by `det_score`; visual gate via `dbg_print_aug_grid.py` before scale-up |
| Print augmentation parameters don't match our specific printer | low/medium | A, B refs look wrong | Calibration prints + ArcFace similarity probe before training |
| All 4 fail | low | Catastrophic | D is the absolute floor; even if its only output is "place coke roughly somewhere," 1-2/9 rollouts + bonus = ~31-37 pts |
| Brev VM crashes or runs out of compute | low | Lost progress | Push checkpoints to HF every 5k steps; track A is resumable from any checkpoint |

---

## 6. Compute budget

| Track | Brev hours | $ @ ~$5/h | Bonus pts | Cost per bonus-pt |
|---|---|---|---|---|
| A | 5 | $25 | +20 | $1.25 |
| B | 32 | $160 | +16 | $10 |
| C | 30 | $150 | +16 | $9.4 |
| D | 7 | $35 | +20 | $1.75 |
| **Total** | **74** | **$370** | best-of-4 | varies |

Remaining budget ~$130 after current spending. We will exceed this if we run all 4 on full retrains. **Mitigation: run A and D first (cheap, bonus-preserving), then evaluate before committing to B + C.**

---

## 7. Hard constraints from the spec

- **VLA-only at inference.** Everything in this plan (ArcFace embeddings, RetinaFace masks, print augmentation, reference recuration) happens **offline at training time only**. Cached asset tables (face crops, embeddings) are not models. Nothing additional runs at inference.
- **20 s per rollout.** SmolVLA fits trivially. Pi0.5 needs an inference-latency probe.
- **16 GB VRAM at Strix deploy.** SmolVLA-450M uses ~2.5 GB. Pi0.5 at bf16 ≈ 7 GB weights + KV cache + ViT activations → borderline but should fit.
- **Smallest-model bonus.** SmolVLA = rank 1 (+20). Pi0.5 = rank 3 (+16). The +4 differential = 0.72 rollouts of slack — Pi0.5 must beat SmolVLA by ≥1 rollout to come out ahead.

---

## 7b. Track A v2 — design after deep paper readings + validation audit (2026-05-17 evening)

After 7 research agents (4 deep-reads + 4 skeptical validators) audited the v1 Track A design, several mechanisms are confirmed missing from v1 that were bundled with the papers we cited. See [`docs/report/EVAL_3_RESEARCH_REPORT.md` §P2](../docs/report/EVAL_3_RESEARCH_REPORT.md) for the per-paper deep-dive and §P2.7b for the validation audit.

### Components added to Track A v2

**A1. Reference photo recuration** (unchanged from v1 design)
Quality filter per NIST FRVT / ISO 19794-5; head+shoulders crop offline; ship as static asset table. ~4h.

**A2. Print-domain augmentation on camera2** (unchanged from v1 design)
Augraphy-inspired Lab gamut → Floyd-Steinberg dither → Perlin grain → JPEG, p=0.7. ~6h eng.

**A3. ArcFace cosine distillation** — refined from V2 validation
- Loss: BlindVLA equation 9, **`F.normalize` then dot product then `mean()` then negate**. λ=0.2 from step 0 (constant).
- **Frozen 3-layer MLP projector** (LayerNorm → Linear(hidden, 2048) → SiLU → Dropout(0.1) → Linear(2048, 2048) → SiLU → Dropout(0.1) → Linear(2048, 512)). Frozen from step 0 — paper Table 6: trainable projector becomes a shortcut.
- **Injection at Backbone2Enc mid-LLM layer** (NOT Enc2Enc / SigLIP output). For SmolVLA's truncated 16-layer SmolLM2, inject at layer 7-8 (mid-network).
- Mask-gated to face patches only (RetinaFace masks precomputed offline). Pooled-student variant for the single-embedding ArcFace teacher.
- Camera2-only (preserve camera1 manipulation grounding).
- Code template: [`finetune_align.py`](https://github.com/CognitiveAISystems/BlindVLA/blob/main/openvla/vla-scripts/finetune_align.py) lines 310-338 + 417-420.
- Caveat: cosine > L2 with p<0.01 holds on semantic + vision axes only; execution axis p=0.05. Track separately on validation.

**A4. Diversify reference photos per celeb** (NEW from V1 agent finding)
- 3-5 photos per celeb in the bank, sampled randomly per training step
- Mirrors Interleave-VLA Table 4 (Mixed 71.0/71.7 > Task-only 67.5/67.1 > Internet-only 59.2/69.1)
- Dataset-side change only. ~3h eng.

**A5. ObjectVLA-style bbox-grounding via prompt relabel** (NEW from V4 agent — strongest single mechanism)
- ObjectVLA verbatim numbers: 100% ID → 19% OOD without bbox grounding; 100% → 64% with. (Verbatim §4.1.1-4.1.2.)
- Their training mix: **10:1 robot:VL** with VL pairs being `(image, "Detecting the bounding box of <object>.", "(x1,y1),(x2,y2)")`. (Verbatim §3.2-3.3.)
- LeRobot 0.5.1 lacks multi-dataset co-training (`MultiLeRobotDataset = NotImplementedError`), so we use a **prompt-relabel proxy**: precompute the face bbox per reference photo offline; inject it as text in the prompt. Example: `"<ref> shows Yann LeCun in bbox (12,15)-(245,230). Set the coke down on his picture."` ~6h eng.
- Optimizer for ObjectVLA was **Adam (NOT AdamW)**, LR 2e-5, bs 128, 50k steps. For Track A v2 we keep SmolVLA's AdamW (different model regime).

**A6. Lower LR to 2.5e-5** (NEW from V1 agent)
- We unfroze both VLM AND SigLIP at LR 5e-5. No source paper unfroze both at that LR.
- Halve to 2.5e-5 to protect pretrained features.

**A7. Tighten color jitter** (NEW from V1 agent)
- Default SmolVLA `image_transforms` includes hue jitter ±0.05 — perturbs skin tones.
- Reduce to ±0.02 or disable.

### Components deferred to Track A-2 (follow-up)

**True Interleave-VLA inline-in-language protocol.** Verified by V3 validation that SmolVLA's prefix is `[images, language, state]` (cannot interleave images between text segments via `add_image_special_tokens` alone). However, the V3 audit also clarifies that **whether the 2nd-camera approach is "invalid" vs the inlined-prompt approach is OUR hypothesis, not paper-tested** — Interleave-VLA never tested the 2nd-camera variant. So we treat true Interleave-VLA as a high-value follow-up to validate the hypothesis, not as an established prerequisite.

Implementing it requires substantial changes:
- `processor_smolvla.py` — insert `<BOI> image_tokens <EOI>` tokens INTO the text token stream during processor
- `modeling_smolvla.py:626-705` `embed_prefix` — change concat order from `[images, language, state]` to `[language_with_inlined_images, state]`
- Multi-day eng work

**KI gradient stop + FAST-token loss.** Verified by V1 — paper Eqs. 5-6 verbatim. Requires forward-pass changes (add `sg(·)` wrapping cross-attention K/V from VLM to action expert; expose LM head for FAST-token cross-entropy). Multi-day eng.

**Web/VQA co-training.** Verified by V1 — Pi0.5 blog 94% → 74% OOD without web data. Blocked by `MultiLeRobotDataset` in our pinned LeRobot.

### Track B v2 (Pi0.5 + ArcFace distillation)

Port A3 (frozen 3-layer MLP at Backbone2Enc) to Pi0.5's PaliGemma vision tower. Same loss equation, same λ=0.2, same teacher.

For Pi0.5 specifically, also adopt at training time:
- `train_expert_only=false` (KI-validated)
- `freeze_vision_encoder=false` (BlindVLA-validated)
- LR 2.5e-5 (Pi0.5 default is 2.5e-5 per `configuration_pi05.py` — keep)
- Gradient checkpointing + bf16 + compile_model
- bs=16-24 (VRAM-bounded on RTX PRO 6000)

### Track C v2 (Pi0.5 vanilla IaP)

No surgical fixes. Pure capacity bet. Isolates the "scale solves it" hypothesis.

### Track D v2 (3-celeb baseline)

Unchanged from v1: SmolVLA from `lerobot/smolvla_base` on 178 base teleops, name-only prompts, `--policy.empty_cameras=2`. ~6h Brev.

### Updated risk register additions

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Track A v2's ObjectVLA prompt-relabel is weaker than true bbox co-train | medium | A v2 underperforms; ObjectVLA's published gain (45pp) may not transfer | Visual gate on the bbox prompts; ablation toggle on the relabel |
| Backbone2Enc injection at layer 7-8 of SmolLM2 is research-grade extrapolation | medium | Wrong layer choice could underperform Enc2Enc | Test 2-3 layers (5, 8, 12) in parallel — `align_layers` is configurable |
| Pooled-student ArcFace adaptation underperforms patch-level | medium | A3 underperforms | A/B against the per-face-patch broadcast variant |
| Print augmentation parameters don't match our specific printer | low-medium | A v2 refs look wrong | Calibration print + ArcFace cosine probe before scaling (mandatory per CLAUDE.md §7) |

---

## 8. Cross-references

- [`/TODO.md`](../TODO.md) — operational checklist
- [`docs/EVAL_3_OPTIONS.md`](../docs/EVAL_3_OPTIONS.md) — full 15-option enumeration with reasoning per option
- [`docs/report/EVAL_3_RESEARCH_REPORT.md`](../docs/report/EVAL_3_RESEARCH_REPORT.md) — definitive research synthesis (7-agent triangulation)
- [`eval_3/aug/RESEARCH_v3_face_matching_rescue.md`](aug/RESEARCH_v3_face_matching_rescue.md) — image-as-prompt branch deep dive
- [`eval_3/aug/STRATEGY_v3.md`](aug/STRATEGY_v3.md) — v3 augmentation pipeline strategy (training data construction)
- [`eval_3/README.md`](README.md) — original Eval 3 plan + architecture decision
- [`docs/VLA_ARCHITECTURES.md`](../docs/VLA_ARCHITECTURES.md) — architecture inventory and knob taxonomy
- [`docs/PROJECT.md`](../docs/PROJECT.md) — official eval rubric + VLA-only constraint
