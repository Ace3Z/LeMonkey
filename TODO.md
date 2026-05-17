# TODO.md — active work list for LeMonkey Eval 3

**Last updated:** 2026-05-18
**Status:** Plan re-locked to **3 tracks** (Tracks 1, 2, 3 per [`eval_3/STRATEGY.md` §7c](eval_3/STRATEGY.md)). Three independent validations completed 2026-05-18 (see §7c.1). M6 SmolVLA refactor is the critical-path bottleneck.

> **Read this file first** in every new Claude session. It is the operational source of truth for what's being worked on right now. The deeper "why" lives in:
> - [`eval_3/STRATEGY.md` §7c](eval_3/STRATEGY.md) — current locked-in 3-track plan
> - [`docs/report/EVAL_3_RESEARCH_REPORT.md`](docs/report/EVAL_3_RESEARCH_REPORT.md) — 7-agent research synthesis + M1-M8 mechanism enumeration
> - [`eval_3/aug/RESEARCH_v3_face_matching_rescue.md`](eval_3/aug/RESEARCH_v3_face_matching_rescue.md) — image-as-prompt branch dive
> - [`eval_3/aug/STRATEGY_v3.md`](eval_3/aug/STRATEGY_v3.md) — augmentation strategy

---

## The 3 tracks locked in 2026-05-18

| ID | Name | Backbone | Mechanisms | Bonus | Brev cost | Role |
|---|---|---|---|---|---|---|
| **1** | SmolVLA-surgical | SmolVLA-450M | M1+M2+M3+M6+M7 [+M4-lite] | **+20** | ~6h | Primary surgical bonus-preserving |
| **2** | Pi0.5-surgical | Pi0.5-3B | M1+M2+M3+M6+M7 [+M4-lite] | +16 | ~32h | Max-effort capacity + surgical |
| **3** | SmolVLA-3celeb | SmolVLA-450M | Current stack + M6 (3-celeb subset) | **+20** | ~6h | **HIGHEST PRIORITY — safety floor for IID runs** |

**Mechanism legend** (full definitions in [`docs/report/EVAL_3_RESEARCH_REPORT.md` §P2](docs/report/EVAL_3_RESEARCH_REPORT.md)):
- M1 = Frozen MLP projector (BlindVLA Table 6)
- M2 = ArcFace cosine alignment loss at Backbone2Enc mid-LLM layer (BlindVLA Eq. 9, λ=0.2)
- M3 = Pi0.5-KI stop-gradient `K_b=sg(K_vlm), V_b=sg(V_vlm)` (Pi0.5-KI Eqs. 5-6)
- M4 = FAST CE on VLM LM head — **excluded per user**; **M4-lite** at λ=0.1 recommended by Validation #1 as risk mitigation
- M5 = Web/VQA co-train — excluded
- M6 = Interleave-VLA inline-image-in-language (arxiv 2505.02152 §3.2)
- M7 = 3-5 reference photos per celeb, sampled per step (Interleave-VLA Table 4)
- M8 = ObjectVLA bbox-grounding co-train — excluded from §7c

---

## Validation findings (2026-05-18) — implications for the plan

Three independent agents validated the locked-in plan ([STRATEGY §7c.1](eval_3/STRATEGY.md)):

1. **M3-without-M4 is research-unsound:** Pi0.5-KI's own Fig. 6(b) shows flow-matching-only converges 7.5× slower without FAST CE. **Recommended mitigation: M4-lite (λ=0.1 FAST CE on VLM LM head).** Track 1 and Track 2 should ablate `λ ∈ {0.0, 0.1, 0.3}` in a fast 3k-step run before committing the full schedule.
2. **User's 3072 figure is per-target:** total combinatorial space is 9216; existing 9 cells max out at 4608. **LOS layout (Lecun-Obama-Swift slot order) is missing entirely from base teleops** — Option A: record 60 new LOS teleops (~2h, recommended); Option B: full 4608 coverage of existing 9 cells (fallback); Option C: face-repaint forbidden per CLAUDE.md §7.
3. **M6 on SmolVLA ≈ 3-4 eng-days (full Interleave-VLA) or 1-2 days (minimal split); M6 on Pi0.5 ≈ 0.5 eng-days** (PaliGemma native `image_token_index = 257152` support). Track 2 can launch fastest; Tracks 1 and 3 are bottlenecked on the SmolVLA refactor.

---

## Launch ordering (per STRATEGY §7c.2)

| Day | Action |
|---|---|
| 0 (tonight) | Pi0.5 M6 + Track 2 dataset prep (quantile stats, M1+M2+M3 port). SmolVLA M6 minimal-split refactor in parallel. Track 3 dataset prep (3-celeb filter, LOS-layout decision). |
| 1 | Launch Track 2 (~32h Brev). SmolVLA M6 refactor mid-flight. Track 3 photo curation. |
| 2 | SmolVLA M6 lands. **Launch Track 3 (highest priority, ~6h).** Launch Track 1 (~6h). |
| 3 | M4-lite ablation (3k steps × 3 λ values). Commit best λ. Begin pre-eval dry-run protocol. |

**Fallback if M6 SmolVLA slips:** by Day 1 evening, if not on track for end-of-Day-2 land, launch **Track 1-prefix** and **Track 3-prefix** variants (existing `[images, language, state]` prefix, no inlining). Loses the inline mechanism but unblocks training.

---

## Track 1 — SmolVLA + M1+M2+M3+M6+M7 [+M4-lite]

**Subtasks:**

### T1.1 Reference photo recuration (shared with §7b A1, ~4h eng)
- [ ] `eval_3/aug/curate_references.py` — NIST FRVT face-quality filter + head+shoulders crop on the 192-celeb bank. Output: 3-5 photos per celeb.

### T1.2 ArcFace embedding + RetinaFace mask cache (shared with §7b A3a, ~4h eng)
- [ ] `eval_3/aug/cache_arcface_embeddings.py` — precompute `buffalo_l` embeddings + RetinaFace masks for every reference photo in the curated bank. Store under each variant directory.

### T1.3 M1 + M2 — Frozen MLP projector + alignment loss patch (~1 day eng)
- [ ] New module `third_party/lerobot/src/lerobot/policies/smolvla/face_align_projector.py` — 3-layer MLP (LN → Linear(hidden, 2048) → SiLU → Dropout(0.1) → Linear(2048, 2048) → SiLU → Dropout(0.1) → Linear(2048, 512)). Set `requires_grad=False` after init (M1).
- [ ] Policy patch in `third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:179` — `embed_image` returns SmolLM2 hidden state at `align_layer ∈ {7, 8}` (Backbone2Enc, NOT pre-connector SigLIP).
- [ ] Policy patch in `third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:355,404,626` — wire mask + cached embedding through `prepare_images`/`embed_prefix`; add `0.2 · L_align` to `forward` (M2). Cosine + `F.normalize`-then-dot product per BlindVLA Eq. 9.
- [ ] Layer-choice ablation toggle: try `align_layer ∈ {5, 8, 12}` of SmolLM2's 16. Default 8.

### T1.4 M3 — Pi0.5-KI stop-gradient port to SmolVLA (~0.5 day eng)
- [ ] Locate cross-attention from action expert to VLM K/V in `smolvlm_with_expert.py`. Wrap K_b/V_b sources with `.detach()` (Pi0.5-KI Eqs. 5-6).
- [ ] Verify gradient flow with a unit test: action loss gradient through `model.named_parameters()` should not touch any VLM-prefixed parameter.

### T1.5 M4-lite — FAST CE on VLM LM head (~1 day eng)
- [ ] Add FAST tokenizer pass on action chunks. (Pi0.5-KI §3.3 uses [FAST](https://github.com/google-deepmind/dm_aux) tokenizer; we may need to adapt to SmolVLA's action chunk size.)
- [ ] Expose SmolLM2 LM head. Compute CE loss on FAST tokens, add to total loss at `λ=0.1` (M4-lite).
- [ ] **Validation gate:** fast 3k-step ablation comparing `λ_FAST ∈ {0.0, 0.1, 0.3}`. Pick best and commit.

### T1.6 M6 — Interleave-VLA inline-image-in-language protocol (CRITICAL PATH, ~1-2 eng-days minimal, ~3-4 eng-days full)
- [ ] **Minimal-split approach (preferred):** modify `smolvla` processor to parse `<image>` placeholder in prompt text. Split prompt → `lang_pre_image` + `image_embeds` + `lang_post_image`. Update `modeling_smolvla.py:626 embed_prefix` concat order from `[images, language, state]` to `[lang_pre_image, image_embeds, lang_post_image, state]`.
- [ ] Update attention mask logic in `modeling_smolvla.py:707-716` to reflect the new positions.
- [ ] Smoke-test: feed a prompt `"<image> Set the coke on Taylor Swift's picture."` + camera2; verify image embeddings land at the `<image>` token position and the rest of the language is preserved.

### T1.7 M7 — Diverse reference photos per celeb (shared with §7b A4, ~3h eng)
- [ ] `eval_3/aug/expand_celeb_refs.py` — produce 3-5 face-quality-passing photos per celeb; modify dataset loader to sample one per training step.
- [ ] Update T1.2 cache to include per-photo embeddings.

### T1.8 Hyperparameter + launch (config-only)
- [ ] Update `eval_3/scripts/brev/run_training_track1.sh` (new file): bs=64, lr=2.5e-5, 10-15k steps resume from `HBOrtiz/smolvla_eval3` 30k. Hue jitter ±0.0 or ±0.02.
- [ ] Brev launch (~6h)
- [ ] Push checkpoint to `HBOrtiz/smolvla_eval3_track1`

**Dependencies:** T1.1, T1.2 are shared with Track 2. T1.3 → T1.4 → T1.5 are sequential. T1.6 (M6 refactor) is parallelizable with T1.3-T1.5. **All must complete before launch.**

---

## Track 2 — Pi0.5 + M1+M2+M3+M6+M7 [+M4-lite]

**Subtasks:**

### T2.1 Quantile state/action stats (~1h compute, on dev box)
- [ ] `eval_3/aug/compute_quantile_stats.py` (or reuse LeRobot's `augment_dataset_quantile_stats.py`) — run on `HBOrtiz/so101_eval3_full_merged`. Required by Pi0.5's quantile normalization.

### T2.2 M1 + M2 port to PaliGemma (~1 day eng)
- [ ] Mirror T1.3 on `third_party/lerobot/src/lerobot/policies/pi05/paligemma_with_expert.py`. Inject at PaliGemma layer `l ∈ {6, 10, 14}`; default `l=10` (mid of 18).
- [ ] Patch `third_party/lerobot/src/lerobot/policies/pi05/modeling_pi05.py` — add `0.2 · L_align` to `forward` (M2).
- [ ] Reuse `face_align_projector.py` from T1.3 (same MLP, different `hidden` dim for PaliGemma).

### T2.3 M3 — Pi0.5-KI stop-gradient (NATIVE for Pi0.5, ~0.5 day eng)
- [ ] Pi0.5-KI was originally formulated for Pi0.5 — verify if `lerobot/pi05_base` already includes `sg(·)` on K_b/V_b. If yes, just confirm; if no, patch `paligemma_with_expert.py` cross-attention.

### T2.4 M4-lite — FAST CE on PaliGemma LM head (~0.5 day eng)
- [ ] Pi0.5 may already use FAST tokenizer (Pi0.5-KI paper). Verify; if integrated, just expose at `λ=0.1`. If not, port from T1.5.

### T2.5 M6 — Inline-image-in-language (EASY on PaliGemma, ~4h eng)
- [ ] Modify `third_party/lerobot/src/lerobot/policies/pi05/processor_pi05.py:81-82` — change prompt template from `"Task: {cleaned_text}, State: {state_str};\nAction: "` to `"Task: <image> {cleaned_text}, State: {state_str};\nAction: "`.
- [ ] **No `embed_prefix()` refactor needed.** PaliGemma's `image_token_index = 257152` (verified at `modeling_pi05.py:356-358`) means the HF processor auto-substitutes `<image>` placeholders.
- [ ] Smoke-test: same as T1.6 smoke test, with PaliGemma.

### T2.6 M7 — Diverse photos (shared with T1.7)
- [ ] Reuse `expand_celeb_refs.py` from T1.7.

### T2.7 Hyperparameter + launch
- [ ] Create `eval_3/scripts/brev/run_training_track2.sh`: bs=24, grad_checkpoint=True, bf16, compile_model=True, lr=2.5e-5 (Pi0.5 default).
- [ ] Brev launch (~32h on RTX PRO 6000 Blackwell)
- [ ] Push checkpoint to `HBOrtiz/pi05_eval3_track2`

**Dependencies:** T1.1, T1.2, T1.7 shared. T2.1 must precede launch (Pi0.5 normalization). T2.5 is the fastest M6 path of the 3 tracks.

---

## Track 3 — SmolVLA 3-celeb baseline + M6 (HIGHEST PRIORITY)

**Subtasks:**

### T3.1 3-celeb dataset filter + photo curation (~1 day eng)
- [ ] `eval_3/aug/build_3celeb_dataset.py` — filter the merged dataset to **only** Swift/Obama/LeCun episodes.
- [ ] Photo bank: collect 8 photos per IID celeb (5 held-out + 3 from `datasets/eval3_celebs/scraped/`). Confirmed inventory: Swift 10, Obama 16, LeCun 29 — all ≥ 8 ✓.
- [ ] Per-photo curation via `curate_references.py` (T1.1): NIST FRVT face-quality + head+shoulders crop.

### T3.2 LOS layout decision (decision point — required before T3.3)
- [ ] **Option A (recommended):** record 60 new LOS teleops (~2h physical recording). 20 per IID celeb in LOS slot order.
  - swift_LOS (target=Swift, photos in Lecun-Obama-Swift slot order): 20 episodes
  - obama_LOS: 20 episodes
  - lecun_LOS: 20 episodes (we already have lecun_LSO=20; not LOS — verify)
- [ ] **Option B (fallback):** skip LOS, generate 4608 variants from existing 9 cells.
- [ ] **Option C (FORBIDDEN per CLAUDE.md §7):** face-repaint to fabricate LOS — breaks visual ↔ motor coupling.

### T3.3 Variant generation (post-T3.2 decision, ~6h compute)
- [ ] Per cell, enumerate photo-tuples (8³ = 512 per cell). Round-robin assign to base teleops in that cell.
  - **Option A path:** 18 cells × 512 = 9216 variants total.
  - **Option B path:** 9 cells × 512 = 4608 variants total.
- [ ] Relabel prompts per 75/15/10 mix:
  - 75% default: `<image> Set the coke can on {NAME}'s picture.`
  - 15% ref-only: `<image> Set the coke can on {PRONOUN} picture.`
  - 10% counterfactual: `<image> Don't put it on {OTHER}. Put it on {NAME}.`

### T3.4 Dataset push to HF
- [ ] Push filtered + augmented dataset to `HBOrtiz/so101_eval3_3celeb_track3`.

### T3.5 M6 — Inline-image-in-language (SHARED with T1.6)
- [ ] Depends on T1.6 — same SmolVLA processor refactor. **Critical path bottleneck.**

### T3.6 Hyperparameter + launch
- [ ] `eval_3/scripts/brev/run_training_track3.sh` — bs=64, lr=2.5e-5, ~30k steps fresh train from `lerobot/smolvla_base` (NOT resumed from v1).
- [ ] No M1/M2/M3/M4-lite distillation — vanilla loss with M6 prompt protocol.
- [ ] Brev launch (~6h)
- [ ] Push checkpoint to `HBOrtiz/smolvla_eval3_track3_baseline`

**Dependencies:** T1.1 (photo curation), T1.6 (M6 SmolVLA refactor) — both shared with Track 1. T3.2 LOS decision is independent and can be made tonight.

---

## Cross-cutting work (shared across tracks)

- [ ] Slack TAs the image-as-prompt-permission question (decides whether Tracks 1/2/3 IaP variants are valid for eval day — all three use IaP)
- [ ] Stand up second Brev VM if running Track 2 (32h) in parallel with Tracks 1 + 3 (6h each)
- [ ] Pre-eval dry-run protocol: each checkpoint × 3 rollouts (1 TOY, 1 held-out IID, 1 OOD) — pick best
- [ ] Decide LOS-layout option (A vs B) before T3.3

---

## Deferred from §7c

**Track 2 (capacity-only vanilla, formerly §7b Track C):** dropped. The Track 1 vs Track 2 comparison serves the same capacity-vs-surgical hypothesis test with cleaner controls.

**ObjectVLA bbox prompt-relabel (formerly §7b A5):** dropped from §7c. The 45pp OOD gain reported by ObjectVLA may not transfer via prompt-text alone; M2+M3+M6 is a cleaner intervention.

**Print-domain augmentation on camera2 (formerly §7b A2):** **not in §7c.** May reintroduce as a Track 1/3 add-on if early dry-runs show print-domain gap dominates failure. Currently considered lower-priority vs M6.

---

## How to mark items done

When completing a subtask: change `- [ ]` to `- [x]` and add a short note with the commit hash. Example:
```
- [x] eval_3/aug/curate_references.py  (commit a1b2c3d, 192 → 187 celebs after filter)
```
