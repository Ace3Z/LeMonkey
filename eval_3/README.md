# eval_3 — SO-101 SmolVLA, Coke can on celebrity image

Runtime artifacts and scripts for **Eval 3** (50 pts + smallest-model bonus
20 pts). Architecture re-locked **2026-05-09** after empirical Phase 1 probe
results — see [Architecture decision](#architecture-decision).

## Task spec (PROJECT.md §2)

- **3 DIN A5 color portraits** placed in a semicircle. Head/shoulder portraits
  only, **no full-body, no white border**. Cut from
  [`docs/Eval_3_TOY_Celebrity_Images.pdf`](../docs/Eval_3_TOY_Celebrity_Images.pdf).
- **Empty 330 ml Coke can** in front of the robot (regular Coke, not Zero).
  May be crumpled at the sides for grip; must still self-stand.
- **Prompt:** `"Place the coke on [celebrity name]"`.
- **In-distribution celebrities:** Taylor Swift, Barack Obama, Yann LeCun.
- **9 rollouts × 5.55 pts each** (50 ÷ 9), **20 s / rollout**, in three tiers:

  | Runs | Tier | Image source |
  |---|---|---|
  | **1–3** known IID | Exact images from the TOY PDF (5 per celeb, 15 total) |
  | **4–6** held-out IID | Different photos of Swift/Obama/LeCun the TAs did NOT hand out |
  | **7–9** OOD | Popular OOD celebs from a TA candidate list (published on Slack) |

**Smallest-model bonus (PROJECT.md §2):** Eval 3 awards 20/18/16/… pts for
1st/2nd/3rd place by **total active inference parameter count** of the policy.
This is **40 % of a perfect rollout score** — heavily incentivises a small
backbone.

**VLA-only at inference (PROJECT.md §3, loosened):**
- At demo day: no YOLO, face-ID, cloud-VLM API calls, or other foundation
  models in the policy itself. Only the deployed VLA runs.
- At training time: external models **are** allowed offline to label or
  synthesise data (face-recognition for clean labels, SDXL for synthetic
  backgrounds, etc.) — only their *outputs* end up in the VLA weights.

This loosening is what makes the data-augmentation strategy below legal.

---

## Architecture decision

**Locked: SmolVLA-450M + Path A (image-as-prompt + co-train) + identity-preserving inpainting augmentation + VGGFace2 VQA co-training.**

### Why SmolVLA, not Pi0.5

The earlier plan (revision A in this repo's git history) chose **Pi0.5
(~3.3 B params)** on the assumption that PaliGemma's WebLI pretraining gave it
celebrity-name knowledge for the OOD tier. We **empirically tested that
assumption on 2026-05-09** by running
[`scripts/probe_paligemma.py`](scripts/probe_paligemma.py) plus a multi-prompt
sweep (results in `~/dl_logs/paligemma_probe.log`):

| Probe | Result |
|---|---|
| `"Who is this person?"` over 14 TOY images | **0/14 names** (got `woman`, `man`, `person`, `artist`, `chef`, `festival`) |
| Same prompt over 6 OOD references (Federer, Merkel, Musk, Messi, Ronaldo, Beyoncé) | **0/6 names** (got `tennis player`, `chancellor`, `man`, `queen`) |
| Gender prompt, all 14 TOY | **13/14 correct** — vision tower is healthy |
| Profession prompt, all 14 TOY | Swift = `artist` 5/5 ✓; Obama = `politician/orator/speaker` 3/4 reasonable; LeCun = `model/chef/artist` 0/5 ✗ |

PaliGemma 3B can **see** but cannot **name**. The premise that justified Pi0.5
(WebLI celebrity recall) does not hold. Once we commit to **image-as-prompt**
(the only viable path — see Path A below), Pi0.5's vision tower (SigLIP-So400m)
no longer beats SmolVLA's vision tower in any meaningful way for our task —
both are SigLIP-derived and produce comparable face-identity embeddings.

### The bonus math

| Pi0.5 success | SmolVLA success | Pi0.5 total | SmolVLA total (with +20 1st-place bonus) | Winner |
|---|---|---|---|---|
| 9/9 | 4/9 | 50.0 | 42.2 + 20 = **62.2** | SmolVLA |
| 8/9 | 4/9 | 44.4 | **62.2** | SmolVLA |
| 7/9 | 5/9 | 38.9 | **47.8** | SmolVLA |
| 7/9 | 3/9 | 38.9 | 36.7 | Pi0.5 (barely) |

Pi0.5 only wins when SmolVLA underperforms it by ≥ 4 rollouts. With both on
the same Path-A pipeline, that gap is unlikely.

### The 3 layers of face-matching the policy needs to learn

| Layer | What | Where it comes from |
|---|---|---|
| **L1 — face-similarity features** | "Photos of same person produce close embeddings" | Pretrained — SmolVLM2 / SigLIP gives this for free |
| **L2 — printed-portrait → portrait robustness** | "Different photo of Obama matches the printed portrait of Obama, despite halftone, glare, paper warp" | **Inpainting augmentation** on our 144 demos (the user's idea — well-precedented per [GenAug](https://genaug.github.io/), [ROSIE](https://diffusion-rosie.github.io/), [RoboEngine](https://roboengine.github.io/)) |
| **L3 — open-set OOD generalisation** | "Recognise a celebrity not in my 3-IID training set when given a held-out reference photo" | Image-as-prompt + VGGFace2 VQA co-training + ~30 supplementary demos with extra celebs |

L1 is free. L3 is image-as-prompt + VQA. **L2 is what the inpainting
augmentation buys us** — the hardest layer because the printed-portrait
domain is OOD vs internet photos.

### Path A — image-as-prompt + co-train

Adopt the [Interleave-VLA pattern (arXiv 2505.02152)](https://arxiv.org/abs/2505.02152)
(2× OOD generalisation gain over text-only, single-VLA at inference):

- Every training and inference step: prompt is interleaved
  `[reference photo of <name>] "Place the coke on <name>"`. Both images
  go through the VLM via `observation.images.{camera1,reference}`. SmolVLA's
  `modeling_smolvla.py` natively iterates over all `image_features` keys
  (`L404–444`).
- The VLA matches the reference photo to one of the 3 prints in the workspace
  — it does **face-matching**, not name-recall. Sidesteps PaliGemma's
  zero-shot blindness entirely.
- **Co-train** with VGGFace2 VQA pairs (~5–10 k) at ≥ 1 VQA per 2 robot demos
  to keep VLM features sharp without catastrophic forgetting (per
  [Pi0.5-KI](https://arxiv.org/abs/2509.13371) and
  [Don't Blind Your VLA](https://arxiv.org/abs/2510.25616)).
- `train_expert_only=False` (let VLM gradients flow so it learns name↔face).
- **Single-VLA-at-inference compliant:** the reference photo is data, not a
  model. Face-ID model used to clean VGGFace2 only runs offline.

---

## The data strategy — augmentation + extra-celeb supplementary demos

### Inpainting augmentation (Phase 2b — Brev-side)

Recorded demos look like: `(camera_video, action, state, prompt, reference_photo_path)`.
The portraits in the workspace are flat printed rectangles, not real faces, so
**diffusion inpainting / face-swap is the wrong tool** (slow, hallucinatory).
Use **homography + Poisson seamless cloning** instead:

```
For each recorded episode:
  1. SAM 2.1 video predictor — click 3 boxes on frame 0 (one per portrait),
     propagate masks across all 600 frames via memory bank.
     Cache RLE masks (~5 MB / demo).
  2. Mine ~30 verified web photos per celeb (Wikimedia + icrawler Bing
     fallback, filtered with InsightFace ArcFace cosine > 0.4 vs reference).
  3. For each (demo, augmentation variant):
        For each frame's 3 portrait masks:
          approxPolyDP → 4 corners
          cv2.findHomography(new_celeb_photo_corners, mask_corners)
          cv2.warpPerspective + cv2.seamlessClone(NORMAL_CLONE)
        Encode H.264 with NVENC (~5 s / video)
  4. Save augmented variant with its own `reference.json` sidecar
     (different reference photo from the workspace photo).
```

**Effective multiplier:** 144 base × 10 augmentation variants ≈ 1440 effective
training episodes. Per Lin et al.,
[ICLR 2025](https://arxiv.org/abs/2410.18647), diversity of (env × object) pairs
matters far more than demos-per-condition; rapid diminishing returns past
~50 demos per pair. **Cap at ~10–15× per demo.**

**Per-demo augmentation cost on Brev RTX Pro 6000:** ~35 s (SAM-2 propagation + homography + encode).

### Supplementary celebrity demos (Phase 2c — robot-side)

Beyond the 144 base demos with Swift/Obama/LeCun, **collect ~30 more demos
with ~5–10 additional public figures** (Bezos, Beyoncé, Federer, Merkel,
Messi, Trump, Cristiano Ronaldo, etc.). These DON'T need to be balanced —
the goal is to teach the policy "the prompt format + face-matching pattern
generalises beyond the 3 training celebs." Critical for the 3 OOD rollouts
(runs 7–9).

**Total effective dataset:** ~144 base × 10 inpainted = 1440 + ~30 × 10 = 300
extra-celeb augmented + ~5 k VGGFace2 VQA pairs ≈ 1700 robot episodes + 5 k VQA.

---

## Phase plan

### Phase 0 — physical prep (you, no compute, today/tomorrow)
- ☐ Print + cut TOY PDF (no white border)
- ☐ Get an empty 330 ml regular Coke can
- ☐ Set up workspace: 3-portrait semicircle, can in front of robot
- ☐ Source ≥ 30 verified web photos per IID celeb (Swift, Obama, LeCun)
- ☐ Source + print ~5–10 extra-celeb portraits at A5 (for supplementary demos)

### Phase 1 — gating probe ✓ done 2026-05-09
- ✓ Built `scripts/probe_paligemma.py` and `scripts/ask_paligemma.py`
- ✓ Ran probe → 0/14 TOY, 0/6 OOD → **Path A mandatory**
- ✓ Multi-prompt sweep confirmed vision-tower healthy / name-recall absent
- ✓ Decision: switch from Pi0.5 to SmolVLA-450M (this README)

### Phase 2a — quick smoke-record (5 episodes, today)
- ☐ Run `scripts/record_eval3_quick.py` to record ~5 episodes of the full task
  (pick up can, place on a target celebrity portrait)
- These 5 episodes are the basis for **iterating on the inpainting augmentation
  pipeline at home** before committing to the 144-ep main collection

### Phase 2b — augmentation pipeline (you at home + me, ~1 day)
- ☐ `scripts/aug/mine_celeb_photos.py` — Wikimedia + icrawler + InsightFace verifier
- ☐ `scripts/aug/segment_portraits.py` — SAM 2 video segmentation
- ☐ `scripts/aug/inpaint_dataset.py` — homography + seamlessClone
- ☐ Smoke-test the full pipeline on the 5 quick-record episodes

### Phase 2c — main recording (~5 h teleop)
- ☐ Update `record_eval3.py` for the 144-episode balanced plan (already exists from rev A)
- ☐ Record 144 demos (3 IID celebs × 6 layouts × 8 reps)
- ☐ Record ~30 supplementary extra-celeb demos
- ☐ Run augmentation pipeline → ~1700 effective training episodes

### Phase 2d — VQA co-train pipeline (parallelizable, ~2 h)
- ☐ `scripts/aug/build_vqa_cotrain.py` — VGGFace2 sample → ~5 k VQA pairs
  formatted as LeRobot v3 episodes
- ☐ Mix into main training dataset

### Phase 2e — merge + push
- ☐ `scripts/merge_eval3_episodes.py` (adapt from `eval_2/scripts/merge_eval2_episodes.py`)
- ☐ Push merged to `HBOrtiz/so101_eval3_all`

### Phase 3 — training on Brev (~10 h compute, $15)
- ☐ Adapt `eval_2/scripts/brev/run_training.sh` for SmolVLA + Eval 3 paths
  (deprecate the existing `eval_3/scripts/brev/setup_pi05.sh` etc. — those
  were for the abandoned Pi0.5 plan)
- ☐ `--policy.path=lerobot/smolvla_base`, `train_expert_only=False`,
  batch 96 (vs 192 for Eval 2 because of dual-image input), 25k steps,
  image augmentation (color + illumination, NO horizontal flip)
- ☐ Push trained checkpoint to `HBOrtiz/smolvla_eval3`

### Phase 4 — pull + smoke (Thor, ~1 h)
- ☐ `scripts/run_rollout_eval3.py` — adapts `eval_2/scripts/run_rollout.sh`,
  loads held-out reference photo at episode start, streams as
  `observation.images.reference`
- ☐ Smoke-test against TOY portraits, then held-out IID, then a sample of OOD

### Phase 5 — demo day
- 9 rollouts: TOY → held-out IID → OOD

---

## Files we have / will have

```
eval_3/
├── README.md                                 ← this file
├── scripts/
│   ├── probe_paligemma.py                    ✓ Phase 1 gating probe
│   ├── ask_paligemma.py                      ✓ interactive PaliGemma helper (file/REPL/sweep/camera)
│   ├── record_eval3_quick.py                 ← Phase 2a (5-episode smoke record)
│   ├── record_eval3.py                       ✓ Phase 2c (144-episode balanced plan; rev A scaffold)
│   ├── aug/                                  ← Phase 2b (build at home / on Brev)
│   │   ├── mine_celeb_photos.py
│   │   ├── segment_portraits.py
│   │   ├── inpaint_dataset.py
│   │   └── build_vqa_cotrain.py
│   ├── merge_eval3_episodes.py               ← Phase 2e
│   ├── run_rollout_eval3.py                  ← Phase 4
│   └── brev/                                 (rev A — scaffolded for Pi0.5; will be replaced
│       │                                       by SmolVLA-targeted scripts adapted from eval_2)
│       ├── setup_pi05.sh
│       ├── run_training.sh
│       ├── start_training.sh
│       ├── follow_training.sh
│       └── training_status.sh
├── state/                                    ← plan.json (gitignored)
├── train/                                    ← model checkpoints (gitignored)
├── rollouts/                                 ← per-rollout dataset dumps (gitignored)
└── evals/                                    ← per-session eval CSVs (gitignored)
```

## Hardware

Same SO-101 setup as Eval 1 / Eval 2. Leader on `/dev/so101-leader`,
follower on `/dev/so101-follower`, camera on `/dev/video0`. **Wrist mount
kept** (matches Eval 1/2 — the previous plan's "shoulder mount" decision is
voided since image-as-prompt + augmentation reduce dependence on always
seeing all 3 portraits).

## Open questions / external blockers

- **OOD celebrity roster** — TAs to publish on Slack. We can prepare for any
  popular celeb with the augmentation + supplementary-celeb training, but
  prints for the 3 OOD rollouts can only be cut after the list lands.
- **Smallest-model bonus tie-breaking** — if multiple teams pick SmolVLA,
  do we tie at 1st (each get 20 pts), or split (e.g., 18-each)? Worth a
  Slack ping to TAs.
- **Demo-day machine specs** — still TBD. Plan assumes Thor or equivalent.
