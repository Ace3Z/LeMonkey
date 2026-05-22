# Track Alpha — research cross-check (fork from Track ArcFace + M2 toolkit)

**Date:** 2026-05-20 · **Author:** Sejohn (auto-collated from 3 parallel research agents) · **Status:** research synthesis, pre-launch

## What this doc is

The user proposed forking [Mahbod's `TRACK_ARCFACE.md`](../../eval_3/tracks/TRACK_ARCFACE.md) into a new **Track alpha** with:
- ArcFace as a **training-only** distillation teacher (VLA-only rule preserved at inference),
- A larger face-ID dataset (VGGFace-class) to pretrain the projector,
- A re-examination of *where* the distillation signal should inject in the VLM/VLA stack.

Per CLAUDE.md §9, three independent research agents audited the idea in parallel. This document is the synthesis. It is the input to [`eval_3/tracks/TRACK_ALPHA.md`](../../eval_3/tracks/TRACK_ALPHA.md).

---

## Agent 1 — best injection point (literature + theory)

**Convergent finding across BlindVLA, VIRAL, ROCKET, REPA, LLaVA-KD, Face-MLLM, FaceLLM, GLaD:** mid-LM hidden-state alignment is where vision–language binding happens and where face-identity supervision lands without poisoning attention.

Ranked candidates:

| Rank | Where | Reason | Confidence |
|---|---|---|---|
| **1** | **LM mid-layer hidden states**, restricted to face-region patches, frozen MLP, cosine on ArcFace sphere | BlindVLA Eq. 9 + VIRAL ablation: single mid-layer beats early, late, and multi-layer | **High** |
| 2 | SigLIP-output / projector layer (Arc2Face / PhotoMaker / InstantID pattern) | Useful **supplementary** per-face token injection point — not a replacement for (1) | Medium |
| 3 | LM final hidden state → action-expert boundary (GLaD-style) | Optional add-on; identity arrives *after* language binding, which is the hard problem | Medium-low |
| 4 | Inside SigLIP ViT | REPA: aligning a frozen upstream encoder gives nothing new at inference | Low |
| 5 | Inside the action expert | No precedent; expert is small and trained from scratch | Very low |

Key cross-checks Agent 1 verified:
- **Cosine, not MSE, on the ArcFace hypersphere.** Magnitudes are uninformative; direction-only matching (ShrinkTeaNet / Angular Distillation) preserves discriminative geometry.
- **Frozen projector beats trainable.** BlindVLA Table 6: trainable projector "collapses to the teacher's output space" and the LM stops being supervised.
- **Multi-face per-region pooling is the novel design move.** BlindVLA/VIRAL align all patches to one teacher per image; identity is per-face, so the loss must be restricted to face-region patches with zero loss elsewhere. InstantID and AnyPhoto confirm per-region identity injection is the right pattern for multi-subject scenes.
- **One layer beats multi-layer.** VIRAL ablation. Default single mid-layer; only add a second loss at the action-expert boundary if motor execution underfits.

Primary recommendation: **layer ~½-depth of Gemma-2B (18 layers)**, per-face region-pooled, cosine-on-sphere with a frozen MLP. Concrete spec at the end.

Sources Agent 1 cited (representative): [BlindVLA 2510.25616](https://arxiv.org/abs/2510.25616), [VIRAL 2509.07979](https://arxiv.org/abs/2509.07979), [GLaD 2512.09619](https://arxiv.org/abs/2512.09619), [ROCKET 2602.17951](https://arxiv.org/abs/2602.17951), [REPA 2410.06940](https://arxiv.org/abs/2410.06940), [Arc2Face 2403.11641](https://arxiv.org/abs/2403.11641), [Face-LLaVA 2504.07198](https://arxiv.org/abs/2504.07198), [FaceLLM 2507.10300](https://arxiv.org/abs/2507.10300), [Face-MLLM 2410.20717](https://arxiv.org/abs/2410.20717), [InstantID 2401.07519](https://arxiv.org/abs/2401.07519), [LLaVA-1.5 2310.03744](https://arxiv.org/abs/2310.03744).

---

## Agent 2 — projector-pretrain dataset

Ranked recommendation:

| Rank | Dataset | Confidence | Notes |
|---|---|---|---|
| **1** | **MS1MV3 via `gaunernst/ms1mv3-wds`** (5.2M images, 93k IDs) | **High** | HF-public, no application, RetinaFace-aligned 112×112. Best access/quality ratio in 2026. |
| 2 | WebFace4M subset | Medium | Requires emailed license — too slow for a 4-day sprint. |
| 3 | VGGFace2 via `ProgramComputer/VGGFace2` | Medium | CC-BY-NC-4.0; Oxford withdrew but HF mirror is widely used. Best pose/age variation. |
| — | Our scraped 192-celeb bank | High | **Use as validation/probe only — NOT pretrain data.** Otherwise overfits the projector to eval IDs. |

Critical cross-checks:
- **Diverse IDs, not eval IDs, for pretrain.** The projector should learn the *general* ArcFace→hidden-state map, not memorize 192 celebs.
- **Apply Track-3 print-style augmentation to ~40% of pretrain crops.** The deployed embedding comes from a wrist-cam-of-A5-printed-photo; the projector must close the clean↔printed domain gap. Track 3's existing Lab/MTF/dither stack is the obvious reuse.
- **License posture.** MS1MV3 is non-commercial research only (Microsoft withdrew MS-Celeb-1M in 2019); ETH course project is squarely within terms. VGGFace2 carries residual consent risk — fine for the course, not for publication.
- **ArcFace teacher checkpoint:** `buffalo_l/w600k_r50` is trained on WebFace600K (not MS1MV3). The mismatch between teacher-training-set and projector-training-set is **fine and probably desired** — it forces the projector to learn the geometry, not memorize the teacher's identity priors.

Sources: [MS1MV3 mirror](https://huggingface.co/datasets/gaunernst/ms1mv3-wds), [VGGFace2 mirror](https://huggingface.co/datasets/ProgramComputer/VGGFace2), [WebFace260M paper 2204.10149](https://arxiv.org/abs/2204.10149), [VGGFace2 paper 1710.08092](https://arxiv.org/abs/1710.08092), [Exposing.ai MS-Celeb withdrawal](https://exposing.ai/msceleb/), [Print-domain FR robustness 2404.06559](https://arxiv.org/abs/2404.06559).

---

## Agent 3 — skeptical audit of `TRACK_ARCFACE.md` before fork

Verdicts on 10 design choices:

| # | Item | Verdict |
|---|---|---|
| 1 | Layer 9-of-18 injection (½-depth extrapolation) | **RISK** — sweep, default upper-middle |
| 2 | Frozen 2-layer MLP projector | **CRITICAL** — restore BlindVLA 3-layer per [`STRATEGY.md` §7b A3](../../eval_3/STRATEGY.md) |
| 3 | λ = 0.2 transferred from BlindVLA | **RISK** — narrow teacher concentrates gradient; halve to 0.1 |
| 4 | Projector pretrain target = vanilla `h_pg^(0)` | **CRITICAL** — collapses identity geometry; either drop pretrain or use VGGFace2/MS1MV3 identity loss |
| 5 | Per-token cosine loss with multi-face frames | **CRITICAL** — Eval 3 has 3–5 prints in frame; design ignores identity assignment |
| 6 | RetinaFace at inference vs VLA-only rule | PASS-with-caveat — force no-bbox variant |
| 7 | Sprint feasibility (Day 3 of 4) | **RISK** — share Mahbod's idle Track-D VM, cap steps |
| 8 | "Novel research contribution" claim | RISK (overstated) — team's own M2 toolkit makes it not novel within this codebase |
| 9 | Compute budget (3rd training run) | **RISK** — ~$180–210 vs $200 budget; tight |
| 10 | Print-augmentation degrading ArcFace | **RISK** — add calibration gate before training |

The two critical findings that load-bear:

**Defect #1 — projector direction inverted.** `TRACK_ARCFACE.md` writes `512 → 2048 → 2048` (project ArcFace up into LM space). The team's own validated spec at `STRATEGY.md` §7b A3 (line 180, line 326) is `LN → Linear(2048) → SiLU → Dropout(0.1) → Linear(2048) → SiLU → Dropout(0.1) → Linear(512)` — student LM hidden state pooled into the ArcFace 512-D sphere. The validated direction is BlindVLA-faithful; the drift in `TRACK_ARCFACE.md` is a silent regression.

**Defect #2 — projector pretrain target.** The proposed pretrain objective `1 − cos(MLP(z_arcface), h_pg^(0))` aligns the projector to the *un-trained* SigLIP mean-pool. But SigLIP is the **identity-blind feature space we're trying to fix**; pretraining the projector to match it preserves zero identity geometry. The right pretrain is either (a) skip it entirely per BlindVLA (random-frozen-init outperforms trainable), or (b) supervise the projector with an MS1MV3 identity-classification head, then discard the head.

**What to preserve from TRACK_ARCFACE.md / STRATEGY.md §7b:**
- Pi0.5 backbone, `train_expert_only=True` + LoRA recipe.
- Frozen ArcFace `buffalo_l/w600k_r50`.
- Cosine alignment on the hypersphere.
- Offline precomputation of bboxes + ArcFace embeddings as dataset columns.
- Use of `HBOrtiz/so101_eval3_track3_v3_baseline` (no new dataset upload).

**What to change in Track alpha:**
- 3-layer projector, output dim 512 (student-down-to-teacher), frozen-from-random-init.
- Drop the defective `h_pg^(0)` pretrain; use MS1MV3 identity-classification pretrain instead.
- Default layer = upper-middle (l ≈ 12 of 18), with cheap probe-set sweep {9, 12, 15}.
- λ default = 0.1 (halved), sweep {0.05, 0.1, 0.2}, anneal to 0.05 over final 20% of training.
- Multi-face mask-gated alignment: target patches → target embedding, distractor patches → distractor embeddings (contrastive). Implements `arcface_audit_200celeb.py`'s target/distractor data model.
- No face detector at inference; pooled-student / no-bbox variant only.
- Calibration gate: print-augmented ArcFace cosine ≥ 0.55 to clean centroid on a 20-crop probe before launch.
- Drop "novel research contribution" framing.

Sources: BlindVLA paper + repo (verified `finetune_align.py` is 3-layer), [PaliGemma 2407.07726](https://arxiv.org/abs/2407.07726), [ArcFace 1801.07698](https://arxiv.org/abs/1801.07698), local: `eval_3/STRATEGY.md` §7b/§7c, `eval_3/scripts/track_2/arcface_audit_200celeb.py`.

---

## Synthesis — Track alpha design (one-screen)

```
                                    TRAINING ONLY
                            ┌────────────────────────────┐
                            │                            │
   wrist/shoulder cam ──┐   │   offline preprocessing    │
                        │   │                            │
                        ├───┼───▶ RetinaFace ──▶ bbox    │
                        │   │              + ArcFace ──▶ z_target  (512-d,
                        │   │           buffalo_l/w600k_r50         L2-norm.)
                        │   │                            │
                        ▼   │                            │
                  PaliGemma SigLIP ─▶ Gemma LM layers ────│──▶ layer L (~12 of 18)
                                                          │             │
                                                          │             ▼
                                                          │   mask-pool patches
                                                          │   inside target bbox
                                                          │             │
                                                          │             ▼
                                                          │   FROZEN 3-layer MLP
                                                          │   2048 → 2048 → 512
                                                          │   (BlindVLA + STRATEGY §7b A3)
                                                          │             │
                                                          │             ▼
                                                          │       u_target (512)
                                                          │             │
                                                          ▼             ▼
                                                Gemma → action-expert
                                                          │
                                                          ▼
              L_total = L_flow_matching + λ_t · L_align
              L_align = − cos(F.normalize(u_target), F.normalize(z_target))
              λ_t = 0.1 (anneal to 0.05 over final 20%)


                                    INFERENCE
                  PaliGemma SigLIP ─▶ Gemma LM ─▶ action expert ─▶ motors
                                            (no face detector, no ArcFace, no extra modules)
```

Stage breakdown:

| Stage | What | Data | Compute | Output |
|---|---|---|---|---|
| **0** Cache | Precompute (target_bbox, distractor_bboxes, target_arcface_z) per frame | `HBOrtiz/so101_eval3_track3_v3_baseline` | ~2h CPU + RTX 5090 | new dataset columns / sidecar parquet |
| **1** Pretrain projector | MS1MV3 identity-classification head on top of MLP(z_arcface); print-style aug on 40% crops | `gaunernst/ms1mv3-wds` | ~12h A100 | `HBOrtiz/eval3_track_alpha_projector` (random orthogonal also valid) |
| **2** Probe sweep | Cheap 3-point layer sweep {9, 12, 15} using mid-layer probe accuracy on 192-celeb bank | scraped bank (validation only) | ~30 min RTX 5090 | chosen `L` |
| **3** Calibration gate | Print-aug 20-crop ArcFace cosine probe ≥ 0.55 mean | — | < 5 min | go/no-go |
| **4** Train Pi0.5 | LoRA + `train_expert_only=True` + L_align at L, λ=0.1 | merged dataset | ~24h Brev (share Track D VM) | `HBOrtiz/pi05_eval3_track_alpha` |
| **5** Strix test | 3-rollout TOY / IID-heldout / OOD protocol | Strix + SO-101 | per Darius's protocol | success log |

Risks ordered by load-bearing:
1. **Compute budget** — share Mahbod's Track D VM, cap at 15k steps.
2. **Projector pretrain quality** — fallback is BlindVLA's frozen-random-init (no pretrain) if MS1MV3 pretrain doesn't converge in time.
3. **Print-domain ArcFace collapse** — calibration gate before launch; asymmetric (clean teacher / printed student) fallback.
4. **Layer choice** — cheap probe sweep ahead of full training.
5. **VLA-only at inference** — explicit no-detector design; verified Camera1+state+prompt input contract per `TODO.md` eval-day reminder.

---

## Open questions for the team

1. **Mahbod's M2 toolkit** (`HBOrtiz/eval3_m2_arcface_toolkit`) likely uses `buffalo_l` off-the-shelf; **confirm** before reusing the embedding cache, since alpha's teacher must match.
2. **Compute** — does sharing Mahbod's standby Track-D VM actually leave headroom for alpha's 24h Pi0.5 run? Roham/Mahbod decision.
3. **PaliGemma tokenization of long-tail celeb names** (e.g. "Bertrand Piccard", "Xherdan Shaqiri") — multi-token; verify the prompt-side embedding target is the *averaged* hidden state over the name's tokens before locking the design.
4. **TA-rule confirmation** that an offline-precomputed bbox column shipped *with the dataset* (not run at inference) is unambiguously training-only.

## Pointers

- Track alpha design doc → [`eval_3/tracks/TRACK_ALPHA.md`](../../eval_3/tracks/TRACK_ALPHA.md)
- Sister track (untouched) → [`eval_3/tracks/TRACK_ARCFACE.md`](../../eval_3/tracks/TRACK_ARCFACE.md)
- Team's validated mechanism set → [`eval_3/STRATEGY.md` §7b A3 / §7c](../../eval_3/STRATEGY.md)
- Sprint schedule → [`TODO.md`](../../TODO.md)

---

## Addendum 2026-05-20 — Architecture pivot from Option A to Option B

After this research synthesis was written and reviewed, the user chose to pivot Track alpha from the originally-scoped Option A (distillation-teacher-only) to **Option B (embedded identity Q-Former, distilled from ArcFace)**.

Changes vs the synthesis above:
- Architecture is now an identity Q-Former (~4-8M params, 4 transformer blocks, K=5 learnable queries) sitting between LM layer 12 hidden states and the action expert. The Q-Former is **trained**, not frozen, and stays in the policy at inference.
- ArcFace is still training-only (the Q-Former is distilled FROM ArcFace via bipartite-matched cosine loss).
- Multi-face handling moves from mask-gated cosine to bipartite-matched K-query output (DETR-style).

**VLA-only rule audit** — full audit in [`TRACK_ALPHA.md` §1.5](../../eval_3/tracks/TRACK_ALPHA.md). Summary:
- At inference: only PaliGemma + Gemma LM + Q-Former + action expert run, all loaded from a single policy HF repo.
- ArcFace and RetinaFace are dropped after offline preprocess.
- Within-team precedent for embedding a face-pretrained sub-module inside the VLA: **Hans's Track A warm-VLM** (`HansOrtiz/smolvlm2_celeb_warm` — SmolVLM2-500M LoRA-fine-tuned on VGGFace2 VQA, used as the frozen VLM inside SmolVLA). The Q-Former is the same pattern at smaller scale (~5M vs ~500M params).
- Verdict: **PASS** (with same TA-confirmation posture as Track A).

**Fallback to Option A** is wired into the plan ([`TRACK_ALPHA.md` §8](../../eval_3/tracks/TRACK_ALPHA.md)) — if TA rules against the Q-Former at Stage 0.1, or if Q-Former pretrain doesn't pass the Stage 1b recall@1 ≥ 0.5 gate, the track downgrades to a BlindVLA-style cosine alignment loss on LM hidden states. ~4 h re-implementation cost, no new inference-time module, unambiguously rule-clean.

**Precedent gap vs the synthesis above.** The synthesis ranked mid-LM-alignment (Option A pattern) as the **highest-confidence** literature-validated choice. Option B (Q-Former-as-trained-state-module distilled from ArcFace) has *no direct published precedent* — closest analogues are InstantID (Q-Former-style identity tokens for diffusion, but with ArcFace at inference) and BLIP-2 (Q-Former pattern, but not for face identity). The user accepted this tradeoff (higher novelty + more capacity + clearer "state slot" for the action expert) in exchange for ~50 % more sprint cost and less precedent.

Status: docs updated. Awaiting Stage 0.1 coordination + TA confirmation before launch.

---

## Addendum 2026-05-20 (2) — FLOWER-VLA sister track added

User requested a second alpha variant on FLOWER-VLA. The Track alpha branch now hosts two parallel design docs:

- [`eval_3/tracks/TRACK_ALPHA_PI05.md`](../../eval_3/tracks/TRACK_ALPHA_PI05.md) + [`TRACK_ALPHA_PI05_PLAN.md`](../../eval_3/tracks/TRACK_ALPHA_PI05_PLAN.md) — Pi0.5 backbone, current primary
- [`eval_3/tracks/TRACK_ALPHA_FLOWER.md`](../../eval_3/tracks/TRACK_ALPHA_FLOWER.md) + [`TRACK_ALPHA_FLOWER_PLAN.md`](../../eval_3/tracks/TRACK_ALPHA_FLOWER_PLAN.md) — FLOWER-VLA backbone, sister track

Same Q-Former-distilled-from-ArcFace mechanism applied to both backbones. FLOWER variant inserts the Q-Former between the half-pruned Florence-2 LLM output and the Flow Transformer (Global-AdaLN conditioning slot).

**Important correction to `VLA_ARCHITECTURES.md`.** That doc (January 2026 cutoff) reported FLOWER as "no fine-tune scripts shipped." This is **outdated**: as of May 2026 there are TWO FLOWER repos, `flower_vla_pret` (pretraining) and **`flower_vla_calvin` (fine-tune scripts, CoRL 2025)**. Pretrained checkpoints are public at [`mbreuss/flower_vla_pret`](https://huggingface.co/mbreuss/flower_vla_pret) and the [FLOWER Collection](https://huggingface.co/collections/mbreuss/flower-vla-67d60e95bf2990699fcef81f). Inference VRAM is <3 GB. The actual blockers in 2026 are:

1. No LeRobot integration — needs a SO-101→CALVIN-format dataset adapter (~6 h Stage F0.5; this is the schedule-critical path)
2. No LoRA/PEFT documented — full fine-tune of 950M params, still feasible on a single H100
3. CALVIN/LIBERO sim-env hooks in the training loop need stripping for our LeRobotDataset path
4. Florence-2's public-figure prior is uncertain — F0.4 zero-shot probe gates this

**VLA-only audit:** identical posture to Pi0.5 alpha; one TA confirmation covers both variants. See [`TRACK_ALPHA_FLOWER.md` §2.5](../../eval_3/tracks/TRACK_ALPHA_FLOWER.md).

**Sprint scheduling:**
- Pi0.5 alpha: ~36 h total, must launch Stage 4 by ~14:00 May 20.
- FLOWER alpha: ~42 h total (extra ~6 h for dataset adapter), must launch Stage F2 by ~16:00 May 20.
- Both share the same MS1MV3 download (one-time, ~1 h) and the same offline ArcFace cache (one-time, ~2 h on the dev box).
- Brev VMs: alpha needs Mahbod's standby VM; alpha-flower needs Hans's VM (after his warm-VLM ships) — coordinate via Roham.

**Force-stop trigger for FLOWER:** if the LeRobot→CALVIN adapter (Stage F0.5) isn't working by 18:00 May 20, FLOWER alpha drops and only Pi0.5 alpha ships. This is enforced in [`TRACK_ALPHA_FLOWER_PLAN.md`](../../eval_3/tracks/TRACK_ALPHA_FLOWER_PLAN.md) decision-points table.
