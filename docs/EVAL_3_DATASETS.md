# Eval 3 — datasets & models on Hugging Face

Single source of truth for what's on `HBOrtiz/*` and which artifacts are
**alive** vs **historical** as of the 2026-05-18 text-only pivot
([`EVAL_3_FINAL_PLAN.html`](report/EVAL_3_FINAL_PLAN.html)).

## TL;DR — what to use today

| Track / role                        | Dataset/model                                          | Status |
|-------------------------------------|--------------------------------------------------------|--------|
| **Training input — all 3 tracks**   | `HBOrtiz/so101_eval3_track3_v3_baseline`               | **★ primary, USE THIS** |
| Hans's warm-VLM for Track A         | `HansOrtiz/smolvlm2_celeb_warm`                        | pending Hans's push (M5 done) |
| Track A / B / C trained checkpoints | `HBOrtiz/smolvla_eval3_track_A` / `pi05_eval3_track_B` / `smolvla_eval3_track_C_baseline` | will exist after Day 2/3 runs |
| **Deprecated, do NOT train on**     | `HBOrtiz/so101_eval3_all` (200-celeb image-as-prompt)  | historical only |
| **Deprecated v1 model**             | `HBOrtiz/smolvla_eval3`                                | historical only |

---

## Datasets

### `HBOrtiz/so101_eval3_track3_v3_baseline` — ★ **PRIMARY, the one we train on**

- Pushed 2026-05-18 23:00 CEST. Public.
- **9,394 episodes** (178 real base teleops + 9,216 augmented variants).
- **5,053,972 total frames** (538 frames/ep × 9,394).
- **15 unique task prompts** = 5 paraphrase templates × 3 celebs (Swift / Obama / LeCun).
- All 6 layout permutations × all 8 photos per celeb × all 64 distractor-photo combos
  enumerated against each base teleop (see [`eval_3/STRATEGY.md` §7c](../eval_3/STRATEGY.md)).
- Text-only prompts (no ref-only, no counterfactual buckets — invalid post-TA-ruling).
- 14.3 GB on disk; uses `xet` content-addressable storage on HF.
- Features:
  - `observation.images.camera1` (wrist, 480×640×3, video) — **the load-bearing input for training**
  - `observation.images.reference` (480×480×3, video) — present in schema but **ignored at training** (use `--policy.empty_cameras=1` for SmolVLA / `=3` for Pi0.5)
  - `observation.state` (float32[6])
  - `action` (float32[6])
- **Build artifact**: also available as `eval3_track3_aug.tar.zst` (13.2 GB) for Drive backup.

### `HBOrtiz/so101_eval3_all` — the 200-celeb v3-aug dataset (DEPRECATED)

- Pushed 2026-05-16. Public. 45 downloads.
- This is the dataset the user remembered: **178 real base teleops + 4,017 augmented variants** drawn from the **full ~200-celeb scraped bank**, M=25 per base.
- Was the training input for v1 SmolVLA training (`HBOrtiz/smolvla_eval3`).
- **Status: deprecated** for Eval 3 training because (a) it carries the
  `observation.images.reference` channel as a load-bearing input (forbidden under
  the TA's 2026-05-18 text-only ruling), and (b) its prompt mix includes 15% ref-only
  and 10% counterfactual prompts that assume the reference image exists.
- **Still useful for**: comparing OOD generalisation breadth (200 identities vs our 3),
  if anyone wants to revisit the image-as-prompt path on a future project.
- **Do NOT train Track A/B/C on this.**

### Other older Eval 3 datasets

- `HBOrtiz/record-test`, `HBOrtiz/eval_my_smolvla_test` — initial pipeline sanity-check
  recordings from April. Not relevant to Eval 3.

---

## Models / checkpoints

### `HBOrtiz/smolvla_eval3` — v1 SmolVLA model (DEPRECATED)

- Public. 59 downloads. The failed image-as-prompt training that prompted the whole pivot.
- Trained on `HBOrtiz/so101_eval3_all` with the camera2 reference channel as input.
- **Status: historical only.** Even if it worked, it's now ineligible for eval (uses
  reference image at inference).
- Keep accessible for diagnostic comparisons; do not use as a starting checkpoint for
  any Track A/B/C run.

### `HansOrtiz/smolvlm2_celeb_warm` — Hans's M5 warm-VLM (PENDING)

- Owner: Hans. Status: not pushed yet as of writing.
- SmolVLM2-500M truncated to 16 layers (matching SmolVLA), LoRA-finetuned on VGGFace2
  (9,131 identities, 3.31M images) with VQA templates ("Who is in this photo?",
  "Is this LeBron James?"). LoRA merged into base before push.
- This IS our M5 mechanism, completed by Hans before we even started the team plan.
- **Track A loads this as `--policy.vlm_model_name=HansOrtiz/smolvlm2_celeb_warm`.**

### `HBOrtiz/smolvla_eval3_track_A` / `_track_B` / `_track_C_baseline` (FUTURE)

- Will be created on Day 2/3 of the sprint when training finishes.
- All read from `HBOrtiz/so101_eval3_track3_v3_baseline` as input.

### Other Eval 1/2 models (out of scope)

- `HBOrtiz/smolvla_eval1`, `_eval1_v2`, `_eval1_residual`, `_eval2`, `_eval2_v2`,
  `_my_smolvla_test` — prior course evals. No bearing on Eval 3.

---

## Importance ranking for the next 4 days

1. **`HBOrtiz/so101_eval3_track3_v3_baseline`** — the one and only training input.
   Every track loads from this. If anything goes wrong with this dataset, Tracks A/B/C
   all fall over.
2. **`HansOrtiz/smolvlm2_celeb_warm`** — load-bearing for Track A. Hans pushes tonight.
3. **`HBOrtiz/so101_eval3_all`** — historical fallback only. Not in the active plan.
4. **`HBOrtiz/smolvla_eval3`** — historical only. Do not load.

---

*Maintained by Roham. Update when a new artifact is pushed.*
