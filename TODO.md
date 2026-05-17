# TODO.md — active work list for LeMonkey Eval 3

**Last updated:** 2026-05-17
**Status:** 4 parallel training tracks committed. Implementation work in flight.

> **Read this file first** in every new Claude session. It is the operational source of truth for what's being worked on right now. The deeper "why" lives in:
> - [`eval_3/STRATEGY.md`](eval_3/STRATEGY.md) — Eval 3 strategy
> - [`docs/EVAL_3_OPTIONS.md`](docs/EVAL_3_OPTIONS.md) — full option space
> - [`docs/report/EVAL_3_RESEARCH_REPORT.md`](docs/report/EVAL_3_RESEARCH_REPORT.md) — research synthesis
> - [`eval_3/aug/RESEARCH_v3_face_matching_rescue.md`](eval_3/aug/RESEARCH_v3_face_matching_rescue.md) — image-as-prompt branch dive
> - [`eval_3/aug/STRATEGY_v3.md`](eval_3/aug/STRATEGY_v3.md) — augmentation strategy

---

## The 4 tracks committed for training today

| ID | Name | Cost (Brev) | Bonus | Why we're running it |
|---|---|---|---|---|
| **A** | SmolVLA-boost-v2 (refs + print aug + ArcFace distill) | ~5h | **+20** | Primary bonus-preserving path — fixes both the (G1) domain gap and the (G2) representation gap |
| **B** | Pi0.5 + ArcFace distillation (hybrid) | ~30-35h | +16 | Maximum-effort capacity-bet: bigger backbone + explicit face features |
| **C** | Pi0.5 + image-as-prompt (vanilla) | ~27-33h | +16 | Tests the pure capacity hypothesis — does scaling alone fix it? |
| **D** | Stable baseline (3-celeb only, SmolVLA, name-only) | ~5-8h | **+20** | Floor / safety net — guaranteed-functional model for the 6 IID runs even if A/B/C all fail |

**Reasoning chain for each track is in [`eval_3/STRATEGY.md` §3](eval_3/STRATEGY.md).** This file is just the operational checklist.

---

## Track A — SmolVLA-boost-v2 (UPDATED 2026-05-17 evening after deep-read audit)

**Goal:** Resume from the current 30k checkpoint (`HBOrtiz/smolvla_eval3`) for 10-15k more steps with **seven additions** (was three in v1 design). See [`eval_3/STRATEGY.md` §7b](eval_3/STRATEGY.md) for the per-component reasoning chain and [`docs/report/EVAL_3_RESEARCH_REPORT.md` §P2.7b](docs/report/EVAL_3_RESEARCH_REPORT.md) for the validation audit results.

**Subtasks (do in order):**

### A1. Reference photo recuration (~4h eng)
- [ ] `eval_3/aug/curate_references.py` — face-quality filter + head+shoulders crop on the 192-celeb bank

### A2. Print-domain augmentation on camera2 (~6h eng)
- [ ] `eval_3/aug/print_simulate.py` — Augraphy-inspired print-emulation operator (Lab gamut → FS dither → Perlin grain → blur → JPEG)
- [ ] `eval_3/aug/dbg/dbg_print_aug_grid.py` — visual sample generator (4×4 grid: clean / aug / real-print). **User-gate before kicking off training.**
- [ ] Calibration print + ArcFace cosine probe (CLAUDE.md §7 mandatory gate)

### A3. ArcFace cosine distillation (~1 day eng)
- [ ] `eval_3/aug/cache_arcface_embeddings.py` — precompute `buffalo_l` embeddings + RetinaFace masks; store under each variant dir
- [ ] Policy patch in `third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:179` — `embed_image` returns mid-LLM hidden state at layer 7-8 (NOT pre-connector SigLIP output, per V2 Table 5 validation)
- [ ] Policy patch in `third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:355,404,626` — wire mask + cached embedding through `prepare_images`/`embed_prefix`; add `0.2 * align_loss` to `forward`
- [ ] New module `third_party/lerobot/src/lerobot/policies/smolvla/face_align_projector.py` — **3-layer MLP, FROZEN** (LayerNorm → Linear(hidden, 2048) → SiLU → Dropout(0.1) → Linear(2048, 2048) → SiLU → Dropout(0.1) → Linear(2048, 512)). Set `requires_grad=False` after init.
- [ ] **Layer-choice ablation toggle**: try align_layers ∈ {5, 8, 12} of SmolLM2's 16. Layer 8 is the mid-network default.

### A4. Diversify reference photos per celeb (~3h eng) — **NEW from V1 audit**
- [ ] `eval_3/aug/expand_celeb_refs.py` — produce 3-5 face-quality-passing photos per celeb; modify dataset loader to sample one per training step
- [ ] Update `eval_3/aug/cache_arcface_embeddings.py` to cache per-photo embeddings

### A5. ObjectVLA-style bbox-grounding via prompt relabel (~6h eng) — **NEW from V4 audit, strongest single mechanism (45pp gap)**
- [ ] `eval_3/aug/compute_face_bboxes.py` — precompute face bbox per reference photo using RetinaFace
- [ ] `eval_3/aug/relabel_prompts_with_bbox.py` — relabel 50% of training prompts to inject bbox: `"<ref> shows {NAME} in bbox ({x1},{y1})-({x2},{y2}). Set the coke down on his/her picture."`
- [ ] Validate prompt distribution remains balanced

### A6. Lower learning rate to 2.5e-5 (config-only)
- [ ] Update `eval_3/scripts/brev/run_training_boost_v2.sh` with `--policy.optimizer_lr=2.5e-5`

### A7. Tighten color jitter (config-only)
- [ ] Disable hue jitter or reduce to ±0.02 (default ±0.05 perturbs skin tones)

### Launch
- [ ] `eval_3/scripts/brev/run_training_boost_v2.sh` — Brev launch script with all 7 components
- [ ] Visual gates via `dbg_print_aug_grid.py` AND validation set ArcFace cosine probe — user approval before launch
- [ ] Brev launch (~5h, possibly 6-7h with the extra components)
- [ ] Push checkpoint to `HBOrtiz/smolvla_eval3_boost_v2`

**Dependencies:** none — can start now.

**Track A-2 follow-up (deferred):** True Interleave-VLA inline-in-language protocol. Requires processor + embed_prefix changes (multi-day). Activate ONLY if Track A v2 fails to lift face-matching success.

---

## Track B — Pi0.5 + ArcFace distillation

**Goal:** Train Pi0.5 (`lerobot/pi05_base`) on our existing image-as-prompt dataset with the same ArcFace distillation loss as Track A, ported to PaliGemma's vision tower.

**Subtasks:**

- [ ] `eval_3/aug/cache_arcface_embeddings.py` — shared with Track A
- [ ] `eval_3/aug/compute_quantile_stats.py` — Pi0.5 requires quantile state/action norm; run `augment_dataset_quantile_stats.py` on the HF dataset
- [ ] Policy patch on `third_party/lerobot/src/lerobot/policies/pi05/paligemma_with_expert.py` — analogous to the SmolVLA hook
- [ ] Policy patch on `third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py` — add `0.2 * align_loss` in `forward`
- [ ] `eval_3/scripts/brev/run_training_pi05_arcface.sh` — Brev launch script with bs=24 + grad_checkpoint + bf16 + compile_model
- [ ] Brev launch (~30-35h)
- [ ] Push checkpoint to `HBOrtiz/pi05_eval3_arcface`

**Dependencies:** Track A's `cache_arcface_embeddings.py` (shared cache).

---

## Track C — Pi0.5 + image-as-prompt (vanilla)

**Goal:** Train Pi0.5 (`lerobot/pi05_base`) on our existing dataset with the same image-as-prompt protocol we used for SmolVLA. No augmentation changes, no distillation. Pure capacity bet.

**Subtasks:**

- [ ] `eval_3/aug/compute_quantile_stats.py` — shared with Track B
- [ ] `eval_3/scripts/brev/run_training_pi05_vanilla.sh` — Brev launch script with bs=24 + grad_checkpoint + bf16 + compile_model
- [ ] Brev launch (~27-33h)
- [ ] Push checkpoint to `HBOrtiz/pi05_eval3_vanilla`

**Dependencies:** Quantile-stats preprocessing (shared with Track B).

---

## Track D — Stable baseline (3-celeb, SmolVLA, name-only)

**Goal:** A guaranteed-functional fallback. Train SmolVLA on only the 178 base teleops (Swift / Obama / LeCun only — no augmented variants, no other celebrities) with **name-only prompts** matching the eval-day text input format exactly. This loses the OOD 3 runs by design but maximizes reliability on the 6 IID runs (runs 1-6).

**Subtasks:**

- [ ] `eval_3/aug/build_3celeb_dataset.py` — filter the merged dataset to base teleops only; relabel prompts to name-only format
- [ ] Push filtered dataset to `HBOrtiz/so101_eval3_3celeb_baseline`
- [ ] `eval_3/scripts/brev/run_training_baseline_3celeb.sh` — Brev launch script
- [ ] Brev launch (~5-8h)
- [ ] Push checkpoint to `HBOrtiz/smolvla_eval3_baseline_3celeb`

**Dependencies:** none — fully independent, can run first.

---

## Cross-cutting work (shared across tracks)

- [ ] Slack TAs the image-as-prompt-permission question (decides whether A, B, C IaP versions are valid for eval day; D is text-only so always valid)
- [ ] Stand up second Brev VM if running A+D and B+C in parallel
- [ ] Pre-eval dry-run protocol: each checkpoint × 3 rollouts (1 TOY, 1 held-out IID, 1 OOD) — pick best

---

## Priority order for kicking off training tonight

Given Brev compute constraints, recommended order:

1. **Track D** first (~5h) — safety baseline, fastest, fully independent
2. **Track A** in parallel with Track D (~5h) — different code path, no conflict
3. **Track C** (~27h) — once A and D are launched, kick this on either the same or second VM
4. **Track B** (~30h) — last because most expensive and requires the same dataset prep as Track A

---

## How to mark items done

When completing a subtask: change `- [ ]` to `- [x]` and add a short note with the commit hash. Example:
```
- [x] eval_3/aug/curate_references.py  (commit a1b2c3d, 192 → 187 celebs after filter)
```
