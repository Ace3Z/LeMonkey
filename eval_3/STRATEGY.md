# Eval 3 — Training Strategy (2026-05-18)

**Status:** active. Sits above [`eval_3/aug/STRATEGY_v3.md`](aug/STRATEGY_v3.md) (augmentation-specific) and beneath [`docs/EVAL_3_OPTIONS.md`](../docs/EVAL_3_OPTIONS.md) (full option space + decision tree). This document is the **chosen plan** — what we are actually training, in what order, why, and with what fallbacks.

> Operational checklist for this plan lives in [`/TODO.md`](../TODO.md).
> **The active locked-in plan is §7c** (Tracks 1, 2, 3 as of 2026-05-18). §7b (Track A v2, 2026-05-17) is the prior iteration kept for history. §1's table reflects the §7b plan; the §7c table at the top of that section reflects the current plan.

---

## 1. What we are doing

**As of 2026-05-18 the plan is 3 tracks, locked in §7c.** Prior text in this section (4 tracks A/B/C/D) is preserved for history.

| Track (current) | Backbone | Mechanisms | Bonus | Brev cost | Role |
|---|---|---|---|---|---|
| **1** | SmolVLA-450M | M1+M2+M3+M6+M7 [+M4-lite] | **+20** | ~6h | Primary surgical bonus-preserving |
| **2** | Pi0.5-3B | M1+M2+M3+M6+M7 [+M4-lite] | +16 | ~32h | Max-effort capacity + surgical |
| **3** | SmolVLA-450M | Current stack + M6 (3-celeb subset) | **+20** | ~6h | **HIGHEST PRIORITY — safety floor** |

**Historical 4-track plan (§7b, 2026-05-17):**

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

## 7c. Locked-in plan — Tracks 1, 2, 3 (2026-05-18, supersedes §7b)

User locked the plan to **3 tracks** (down from 4 in §7b). The mechanism enumeration M1-M8 lives in [`docs/report/EVAL_3_RESEARCH_REPORT.md` §P2](../docs/report/EVAL_3_RESEARCH_REPORT.md); the decisions below are which subset to apply per track.

| Track | Backbone | Mechanisms | Bonus | Brev cost | Role |
|---|---|---|---|---|---|
| **1** | SmolVLA-450M | M1+M2+M3+M6+M7 [+M4-lite] | **+20** | ~6h | Primary surgical bonus-preserving |
| **2** | Pi0.5-3B | M1+M2+M3+M6+M7 [+M4-lite] | +16 | ~32h | Max-effort capacity + surgical |
| **3** | SmolVLA-450M | Current stack + M6 (3-celeb subset) | **+20** | ~6h | **HIGHEST PRIORITY — safety floor for IID runs** |

**Mechanism legend (per the M1-M8 framework):**
- **M1** = Frozen MLP projector (BlindVLA Table 6: trainable projector becomes a shortcut at λ>0)
- **M2** = ArcFace cosine alignment loss at Backbone2Enc mid-LLM layer (BlindVLA §6.2 + Eq. 9; λ=0.2)
- **M3** = Pi0.5-KI stop-gradient `K_b=sg(K_vlm), V_b=sg(V_vlm)` from VLM into action expert (Pi0.5-KI Eqs. 5-6)
- **M4** = FAST discrete-action CE on VLM LM head (Pi0.5-KI §3.3) — **excluded per user ("too heavy"); M4-lite at λ=0.1 recommended as risk mitigation, see §7c.1**
- **M5** = Web/VQA co-training (Pi0.5 blog) — excluded per user; also blocked by LeRobot `MultiLeRobotDataset = NotImplementedError`
- **M6** = Interleave-VLA inline-image-in-language protocol (arxiv 2505.02152 §3.2)
- **M7** = 3-5 reference photos per celeb, sampled uniformly per training step (Interleave-VLA Table 4: mixed > task-only > internet-only)
- **M8** = ObjectVLA bbox-grounding co-train (arxiv 2502.19250 §4.1.2; 45pp OOD gap) — excluded; §7b's prompt-relabel proxy not carried forward into §7c

### 7c.1 Three independent validations (2026-05-18)

CLAUDE.md §7 mandates triple-source validation. Three independent agents (BlindVLA+Pi0.5-KI deep-read, combinatorics check, code inspection) validated the locked-in plan. Findings below; full transcripts at `/tmp/claude-1000/.../tasks/{aedfeb4c3f1329c06,a53749f679951cc94,a1f36a466d2c2c72e}.output`.

**Validation #1 — M2+M3 soundness without M4 (BlindVLA + Pi0.5-KI deep-read):**
- M2 and M3 are mechanically independent (disjoint gradient paths). M3 is a one-way valve on the **action loss** into the VLM; M2's alignment gradient never traverses the K_b/V_b path. **No conflict.** ✓
- M2 alone is published as standalone-OK: BlindVLA §6.2 explicitly says "integrates with standard SFT" — no FAST/CE co-training required. ✓
- **M3 alone (without M4) is NOT validated by Pi0.5-KI:** their full recipe is M3+M4+M5 as a package. Fig. 6(b) shows flow-matching-only (no FAST CE) needs **7.5× more steps** to converge. Fig. 4(b) shows joint training without M3 degrades language following.
- **Risk:** with M2 active at layer 8 and M3 blocking action gradient from the VLM, layers 8-15 of SmolLM2 receive **zero task-relevant signal** and stay at pretrained weights. Upper-layer reasoning (e.g., resolving "leftmost picture" vs. "rightmost picture") gets no learning signal.
- **Recommended mitigation: M4-lite.** Add a lightweight FAST-style cross-entropy loss on the VLM's LM head with small weight λ=0.1 (vs full-strength λ=1.0 in Pi0.5-KI). Cheap relative to full M4; closer to the published recipe. Validate via fast 3k-step ablation comparing λ ∈ {0.0, 0.1, 0.3} before committing the full schedule.

**Validation #2 — 3-celeb baseline combinatorics:**
- User stated "3072 episodes". Arithmetic: 3! × 8³ = 6 × 512 = 3072 is **per-target**. Across 3 targets the total combinatorial space is **9216** (= 3 × 3072). Flag: the original message likely conflated per-target with total.
- We currently have base teleops in only **9 of 18 (target, layout) cells** — `swift_{SOL,OSL,OLS}`, `obama_{SOL,SLO,OSL}`, `lecun_{SOL,SLO,LSO}`. **LOS layout is missing entirely.**
- Max from existing 9 cells = 9 × 512 = **4608 unique (target, layout, photo-tuple) configurations**.
- **LOS-layout options:**
  - **(A) Record ~60 new physical LOS teleops** (~2h recording) — closes the gap; **least bad per CLAUDE.md §7**.
  - (B) Skip LOS (use 9 cells × subsample to 3072 OR full 4608 coverage) — leaves a systematic hole if LOS is sampled at eval.
  - (C) Repaint LOS from existing teleops via inpainting — **VIOLATES CLAUDE.md §7**: re-painting faces without re-recording gripper trajectory breaks (visual scene ↔ motor action) coupling. Forbidden.
- **Recommendation:** Option A. If time-constrained, Option B with **full 4608-variant coverage** of the 9 existing cells (round-robin one photo-tuple per base teleop ≈ 26 variants per teleop).

**Validation #3 — M6 code-feasibility on SmolVLA vs Pi0.5:**
- **SmolVLA** (`third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:626-718`): `embed_prefix()` enforces fixed `[images, language, state]` concatenation (`torch.cat(embs, dim=1)` at line 705). M6 requires processor refactor (parse `<image>` markers), embedding interleaving, attention mask restructuring. **Estimate: 3-4 eng-days** for full Interleave-VLA-style implementation; **~1-2 eng-days** for a minimal split-prompt approach (`[lang_pre, image_embeds, lang_post, state]`).
- **Pi0.5** (`third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py:356-358`): PaliGemma natively registers `image_token_index = 257152` and the HF processor auto-handles `<image>` placeholders in text. M6 = modify the prompt template only. **Estimate: 0.5 eng-days.**
- **M6 on Pi0.5 is ~7-8× cheaper than on SmolVLA.** This shifts launch ordering: Track 2 can launch fastest of the three.

### 7c.2 Engineering implications and launch ordering

Given the three validation findings:

1. **Day 0 (tonight):** Implement M6 on Pi0.5 (~4h, mostly prompt template + processor splice) and start Track 2 dataset prep (quantile stats; M1+M2+M3 port). Begin SmolVLA M6 minimal-split refactor in parallel.
2. **Day 1:** Track 2 ready to launch (~32h Brev). SmolVLA M6 refactor mid-flight. Track 3 dataset prep (3-celeb filter, photo curation, M6 prompt template, LOS layout decision).
3. **Day 2:** SmolVLA M6 refactor done. **Launch Track 3 first (highest priority, ~6h).** Track 1 follows (~6h).
4. **Day 3:** Decision point: did M4-lite help? Run fast 3k-step ablation comparing λ ∈ {0.0, 0.1, 0.3}; commit best to full schedule.

**Critical fallback for the M6 SmolVLA bottleneck:** If the 1-2 eng-day M6 estimate slips past Day 2, fall back to **Track 1-prefix** and **Track 3-prefix** variants (existing `[images, language, state]` prefix, no inlining). This is what v1 ran — it loses the inline mechanism but unblocks training. Decision criterion: by Day 1 evening, if M6 SmolVLA is not on track to land by end-of-Day-2, launch prefix variants overnight.

### 7c.3 Track 1 — SmolVLA + M1+M2+M3+M6+M7 [+M4-lite]

**Goal:** Surgical bonus-preserving fix. Combine face-recognition prior (M2) + VLM protection (M3) + inline-image protocol (M6) + diverse refs (M7) + lightweight task gradient on upper LLM layers (M4-lite). On the merged 4195-episode dataset already on HF (`HBOrtiz/so101_eval3_full_merged`).

**Total loss:**

```
L = L_flow_matching                      [primary action loss]
  + 0.1 · L_FAST_CE                      [M4-lite — see Validation #1]
  + 0.2 · L_align                        [M2 — BlindVLA Eq. 9]
```

with

```
L_align = − (1/k) · Σ_{j=1..k} cos( F.normalize(MLP(h_l)_j), F.normalize(z_j) )
```

where:
- `h_l` = SmolLM2 hidden state at the Backbone2Enc layer `l ∈ {7, 8}` (default `l=8`; ablate `l ∈ {5, 8, 12}` of 16 layers)
- `MLP` = **frozen** 3-layer projector after init (M1): `LN → Linear(hidden, 2048) → SiLU → Dropout(0.1) → Linear(2048, 2048) → SiLU → Dropout(0.1) → Linear(2048, 512)`
- `z` = `buffalo_l` ArcFace embedding for camera2 (precomputed offline, cached), mask-pooled over RetinaFace face region
- `F.normalize` = L2-normalize before dot product (cosine on the hypersphere; BlindVLA Table 5, Evaluation-Oriented KD CVPR 2022, Unified-KD 2508.11376 all confirm cosine > L2 for ArcFace teachers)
- `k` = number of face-region patches sampled per camera2 image (pooled-student variant for single-embedding ArcFace teacher)

**Gradient flow constraints:**
- **M3** (Pi0.5-KI Eqs. 5-6): in action expert cross-attention, `K_b = sg(K_vlm)`, `V_b = sg(V_vlm)`. Action loss never propagates back into VLM weights.
- **M1**: after MLP init, `for p in mlp.parameters(): p.requires_grad = False` (BlindVLA Table 6: trainable projector at λ>0 becomes a shortcut; the alignment loss must steer the VLM, not the projector).

**Reference protocol:**
- **M6** (Interleave-VLA §3.2): prompt template includes inline `<image>` token. Camera2 reference is spliced INTO the language stream at this token position. Requires SmolVLA processor refactor — see §7c.2. Minimal-split variant: `[lang_pre_image, image_embeds, lang_post_image, state]`.
- **M7** (Interleave-VLA Table 4): 3-5 reference photos per celeb in the curated bank (`eval_3/aug/curate_references.py`); sample one uniformly per training step.

**Camera1 (wrist) unchanged.** Distillation, M3, M6 all apply to camera2 only — camera1 carries manipulation grounding which we must not corrupt.

**Citations (primary sources):**
- BlindVLA: arxiv 2510.25616 (§6.2, Eq. 9, Tables 5/6/8; code template [finetune_align.py](https://github.com/CognitiveAISystems/BlindVLA/blob/main/openvla/vla-scripts/finetune_align.py))
- Pi0.5-KI: arxiv 2505.23705 (Eqs. 5-6, Figs. 4b/6b)
- Interleave-VLA: arxiv 2505.02152 (§3.2 protocol, Table 4 mixing)
- ArcFace: arxiv 1801.07698 (`buffalo_l` / `w600k_r50` backbone)
- Cosine-over-L2 for ArcFace teachers: [Evaluation-Oriented KD CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Evaluation-Oriented_Knowledge_Distillation_for_Deep_Face_Recognition_CVPR_2022_paper.pdf); Unified-KD arxiv 2508.11376

**Optimizer:** AdamW, lr=2.5e-5 (halved from default 5e-5 — both VLM and SigLIP are unfrozen, per §7b A6).
**Image transforms:** hue jitter disabled (±0.0) or tightened to ±0.02, per §7b A7.
**Brev cost:** ~6h on RTX PRO 6000 Blackwell, bs=64, ~10-15k steps resume from `HBOrtiz/smolvla_eval3` 30k checkpoint.
**Checkpoint:** push to `HBOrtiz/smolvla_eval3_track1`.

### 7c.4 Track 2 — Pi0.5 + M1+M2+M3+M6+M7 [+M4-lite]

**Goal:** Same mechanism stack on Pi0.5 backbone. Brackets the failure-mode hypothesis: if Track 2 ≫ Track 1, capacity is load-bearing; if Track 2 ≈ Track 1, surgery alone is enough and SmolVLA wins on bonus.

**Differences from Track 1:**
- **Backbone:** `lerobot/pi05_base` (PaliGemma-3B vision + Gemma-300M action expert; ~3.3B total)
- **Injection layer for M2:** PaliGemma has 18 LLM layers; mid-depth = layer 9-12. Default `l=10`; ablation set `l ∈ {6, 10, 14}`.
- **M6 implementation:** **prompt-template change only.** PaliGemma's `image_token_index = 257152` (verified at `modeling_pi05.py:356-358`) and the HF processor auto-substitutes `<image>` placeholders with the image embedding block. No `embed_prefix()` refactor needed. Per Validation #3, ~0.5 eng-days vs ~3-4 days on SmolVLA.
- **Optimizer:** Pi0.5 default (AdamW, lr=2.5e-5 per `configuration_pi05.py` — no change needed)
- **Training infra:** bs=24, grad_checkpoint=True, bf16, `compile_model=True` (VRAM-bounded on RTX PRO 6000 Blackwell)
- **Preprocessing:** Pi0.5 requires **quantile state/action normalization** — run `eval_3/aug/compute_quantile_stats.py` on the HF dataset first (one-time, ~1h compute).

**Loss equation:** identical to Track 1, with PaliGemma-specific `h_l`. M3 stop-gradient applies to PaliGemma → Gemma-action-expert cross-attention.

**Citations:** same primary sources as Track 1. Note that the M3 protocol (Eqs. 5-6) was originally formulated for Pi0.5; Track 2 is the "native" deployment of M3.

**Brev cost:** ~32h on RTX PRO 6000 Blackwell.
**Checkpoint:** push to `HBOrtiz/pi05_eval3_track2`.

### 7c.5 Track 3 — SmolVLA 3-celeb baseline + M6 (HIGHEST PRIORITY)

**User's stated framing (2026-05-18):** *"the MOST important thing we should be testing is SmolVLA 3-celeb baseline. We should have this working this is really important. and we do this with our current training stack + M6."*

This is the **safety floor that must work.** It concedes the 3 OOD runs by design (training distribution restricted to Swift/Obama/LeCun) but maximizes reliability on the 6 IID runs.

**Composition:**
- **Backbone:** SmolVLA-450M from `lerobot/smolvla_base` (clean start, NOT resumed from v1 checkpoint — different dataset).
- **Mechanisms applied:** current training stack (image-as-prompt with camera2 as reference) **+ M6 inline-image-in-language** (NEW vs §7b's Track D, which was prefix-style).
- **Mechanisms NOT applied:** M1, M2, M3, M4-lite, M7-style diverse-photo augmentation distillation. M7-style **diverse photos** *are* used at the dataset level (8 photos per celeb), but no representation-level distillation.
- **Dataset:** restricted to Swift/Obama/LeCun.

**Dataset construction:**

*Photo bank:* 8 photos per IID celeb (5 held-out + 3 from `datasets/eval3_celebs/scraped/`). Per-photo curation via `eval_3/aug/curate_references.py` (NIST FRVT face-quality filter; head+shoulders crop). Inventory verified — Swift: 10 candidates, Obama: 16, LeCun: 29; all ≥ 8 needed.

*Layout cells (from base teleops, 178 total):*
- swift_SOL=20, swift_OSL=20, swift_OLS=20 (60)
- obama_SOL=20, obama_SLO=20, obama_OSL=19 (59)
- lecun_SOL=20, lecun_SLO=20, lecun_LSO=20 (60)
- **LOS layout missing across all 3 targets.**

*Variant generation (per Validation #2):*
- Combinatorial space: 9 cells × 8³ photo-tuples = **4608 unique (target, layout, photo-tuple) configurations** from existing teleops.
- **Recommendation: Option A** — record 60 new LOS teleops (~2h) to close the 18-cell coverage. Then 18 × 512 = 9216 unique configurations possible.
- **Fallback: Option B** — generate full 4608-variant coverage of the existing 9 cells. Round-robin: for each of 9 cells, enumerate all 512 photo-tuples; assign each tuple to one base teleop in that cell. Each base teleop gets ⌈512/~20⌉ ≈ 26 variants. Balanced; exhaustive over photo space.
- **DO NOT** use Option C (face-repaint to fabricate LOS) — violates CLAUDE.md §7 by breaking visual ↔ motor coupling.

*Prompt mix:* per user's 75/15/10 framing:
- **75% default:** `<image> Set the coke can on Taylor Swift's picture.` (M6 inline + name)
- **15% ref-only:** `<image> Set the coke can on her picture.` (M6 inline + pronoun + no name)
- **10% counterfactual:** `<image> Don't put it on Obama. Put it on Taylor Swift.` (negation + target)

The `<image>` token in every prompt is replaced at processor time with the camera2 reference token block (M6 protocol).

**Loss:** L = L_flow_matching (vanilla — no distillation losses).

**Why this is the safety floor:**
- No representation-level surgery (M1/M2/M3 not active) → no risk of mis-steered VLM.
- 3-celeb restriction means the model only ever sees Swift/Obama/LeCun → IID runs are guaranteed in-distribution.
- M6 is the only architectural addition — minimal risk surface beyond the SmolVLA refactor.
- **Worst case:** 0/3 OOD + 6/9 IID + 20 bonus = (6/9) · 50 + 20 ≈ 53 pts.
- **Best case:** 0/3 OOD + 9/9 IID + 20 bonus = (9/9) · (6/9) · 50 + 20 ≈ ... actually scoring is 50 / 9 ≈ 5.6 pts per rollout, so 6/9 = 33, 9/9 = 50. Bonus +20. Best = 70 pts on a 9-rollout floor.

**Citations:**
- Interleave-VLA (M6): arxiv 2505.02152 §3.2

**Brev cost:** ~6h on RTX PRO 6000 Blackwell, bs=64, full fresh train ~30k steps from `lerobot/smolvla_base`.
**Checkpoint:** push to `HBOrtiz/smolvla_eval3_track3_baseline`.

### 7c.6 Updated risk register (additions to §7b)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| M6 SmolVLA refactor takes >3 days | medium | Tracks 1 and 3 blocked from launch on the user's "start today" timeline | Fallback to prefix variant (no inlining). Decision criterion: by Day 1 evening, if not on track for end-of-Day-2 land, launch prefix variants overnight. |
| M4-lite under-mitigates the M3-without-M4 risk (Validation #1) | medium | Tracks 1 and 2 converge slowly; upper VLM layers stay generic | Fast 3k-step ablation comparing M4-lite λ ∈ {0.0, 0.1, 0.3} before committing the full schedule. If λ=0.0 already converges fast enough, the risk was overstated. |
| LOS-layout gap not filled (Option B chosen) | medium | If eval samples LOS, OOD-style failures on a layout the model never saw | Record 60 new LOS teleops (Option A, ~2h) if user has bandwidth; else accept the gap and document in eval-day notes. |
| Track 3 dataset prep takes >1 day | low | Track 3 launch (highest priority) slips | 3-celeb filter is pure relabel — no re-augmentation needed; <1 day eng. Run in parallel with M6 SmolVLA refactor. |
| Pi0.5 quantile-stats computation blocks Track 2 launch | low | Track 2 launch slips by ~1h | Already validated path; ~1h compute on dev box. Schedule before M1+M2+M3 port. |
| User's 3072 figure was per-target, not total (Validation #2) | confirmed | Dataset size differs from user's stated number | Communicate up-front: 4608 total (Option B) vs 9216 total (Option A with new LOS) — both are larger than the stated 3072. |
| Repeated photo-tuples per base teleop (Option B's 26-per-teleop replication) over-fits to trajectories | low | Action expert sees same arm trajectory with different photos | Acceptable — that is exactly the M6+M7 generalization mechanism: same action, different visual context. |

### 7c.7 What §7c supersedes from §7b

- §7b's **Track A v2** (M1+M2+M7+M8-proxy on SmolVLA, +ObjectVLA prompt-relabel, no M3, no M6) → superseded by §7c **Track 1** (M1+M2+M3+M6+M7+M4-lite; ObjectVLA prompt-relabel dropped).
- §7b's **Track B v2** (Pi0.5 + ArcFace distillation, no M3, no M6) → superseded by §7c **Track 2** (same mechanisms as Track 1, ported to Pi0.5).
- §7b's **Track C v2** (Pi0.5 vanilla IaP, no surgical fixes) → **dropped.** The Track 2 vs Track 1 comparison serves the same purpose (capacity vs. surgical-on-small) with cleaner controls.
- §7b's **Track D v2** (3-celeb baseline, name-only prompts, `--policy.empty_cameras=2`) → superseded by §7c **Track 3** (3-celeb + M6 inline + 75/15/10 prompt mix + 8 photos/celeb).

**Bonus-rollout math (after §7c):** Tracks 1+3 both yield +20 (SmolVLA smallest-model bonus); Track 2 yields +16. Maintains the §7b property that two of three tracks preserve maximum bonus.

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
