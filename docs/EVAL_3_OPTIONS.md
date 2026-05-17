# Eval 3 — Complete options for the face-matching failure (2026-05-17)

**Status:** active. Successor doc to [`eval_3/aug/RESEARCH_v3_face_matching_rescue.md`](../eval_3/aug/RESEARCH_v3_face_matching_rescue.md) — that doc only covered the image-as-prompt branch and was too dismissive of VQA. This doc enumerates the complete option space with reasoning chains per option, contingent on the TA-protocol question that's still open.

**Authors:** the LeMonkey claude session synthesizing 7 parallel research agents' findings, three rounds of skeptical pushback from Roham, and the team chat from 2026-05-17.

---

## 1. Setup and the load-bearing assumption

We have:
- A trained **SmolVLA-450M** on `HBOrtiz/smolvla_eval3` with image-as-prompt protocol (camera1 wrist + camera2 reference photo).
- A merged dataset on `HBOrtiz/so101_eval3_all` (4195 episodes = 178 base teleops + 4017 inpainted variants spanning 192 celebrities, 933 unique prompts).
- An empirical failure mode confirmed on Strix 2026-05-17: positional priors dominate over face-matching across rotated-position tests.
- ~24h before training-finish deadline.

We have **one critical open question** that decides which architectures are viable:

> **Will the TAs accept a reference image as input to the policy at eval day, or is the prompt strictly text-only ("Place the coke on [celebrity name]")?**

The official spec says the prompt is text. The image-as-prompt was our team's design choice; it requires a name→photo lookup table to run at inference. **Both interpretations are defensible until a TA confirms.** Slack them.

The option space below covers both scenarios. Options viable in protocol X are tagged `[A]` (image-as-prompt OK), `[B]` (text-only mandatory), or `[AB]` (works either way).

---

## 2. The full option list

### Option 0 — Ship as-is

| Field | Value |
|---|---|
| Tags | `[A]` |
| Brev cost | 0 |
| Eng cost | 0 |
| Bonus | +20 |
| Expected rollouts | ~2/9 (current Strix observation) |
| Total | ~31 pts |
| Risk | very low (we know what we have) |
| Verdict | **Catastrophic baseline.** Use as the floor for comparison. |

### Option 1 — Reference-photo recuration only `[A]`

Re-pick 1 high-quality frontal-pose head+shoulders photo per celeb from the 192-celeb scraped bank using NIST/ISO enrollment-grade quality filter; ship as a static pre-cropped asset table. No training changes.

| Field | Value |
|---|---|
| Brev cost | 0 (just inference-time swap) |
| Eng cost | ~4h |
| Bonus | +20 |
| Expected lift | +0 to +2 rollouts |
| Risk | very low |
| Verdict | Cheap insurance. Helps the (G1) domain gap but doesn't touch (G2). **Recommended as a Day 0 unconditional add-on.** |

### Option 2 — Print-domain forward augmentation in training `[A]`

Apply Augraphy-inspired print-emulation (Lab gamut compression → Floyd-Steinberg dither → Perlin fBm grain → blur → JPEG) to the camera2 reference stream at training time with p=0.7. Resume from the 30k checkpoint for +5-10k steps. **No change to camera1 or actions.**

| Field | Value |
|---|---|
| Brev cost | ~3-4h |
| Eng cost | ~6h |
| Bonus | +20 |
| Expected lift | +1 to +2 rollouts |
| Risk | low/medium (first-mover on photo-print emulation for VLA) |
| Verdict | Direct fix for (G1) domain gap. **Always pair with Option 1.** |

### Option 3 — ArcFace cosine distillation into SigLIP `[A]`

Apply BlindVLA's recipe (arxiv 2510.25616 eq. 9): negative-mean cosine alignment loss with λ=0.2, injected at SigLIP's `last_hidden_state` before the connector, mask-gated to face patches, applied only to camera2 reference stream. ArcFace teacher (`buffalo_l`) embeddings cached offline; rule-compliant (no model runs at inference).

| Field | Value |
|---|---|
| Brev cost | ~1.5-3h |
| Eng cost | ~1 day (80 LOC policy patch + 150 LOC dataset prep) |
| Bonus | +20 |
| Expected lift | +1 to +3 rollouts |
| Risk | medium (first-mover ArcFace→SigLIP combo; published evidence at 7B not 450M) |
| Verdict | Direct fix for (G2) representation gap. **Stack with Options 1+2.** |

### Option 4 — VLM warm-start with face VQA `[AB]`

Before training the full SmolVLA, fine-tune just SmolVLM2-500M (the VLM backbone, in isolation) on a face-VQA dataset built from our 192-celeb bank: `(face_crop, "Who is this person?", "Yann LeCun")`. Use HuggingFace `transformers.Trainer` — completely separate from SmolVLA / LeRobot, no integration friction. Then start the SmolVLA action fine-tune from this warm-started VLM checkpoint instead of the official `lerobot/smolvla_base`.

| Field | Value |
|---|---|
| Brev cost | ~3-6h VLM pretrain + the regular fine-tune |
| Eng cost | ~1 day (Trainer script + face-VQA dataset prep) |
| Bonus | +20 |
| Expected lift | +1 to +3 rollouts |
| Risk | medium (warm-start may shift the VLM enough that the SmolVLA action expert's existing co-adaptation breaks) |
| Verdict | **Best single intervention if name-only is mandatory.** Useful even with image-as-prompt as a side-channel signal. Cheapest VQA path — sidesteps the LeRobot `MultiLeRobotDataset` blocker entirely. |

### Option 5 — VQA caption augmentation in the existing dataset `[AB]`

Replace 10-30% of training prompts with name-explicit captions: instead of `"Set the coke down on Xherdan Shaqiri's picture"`, use `"This person is named Xherdan Shaqiri. Place the coke on his portrait."` Action labels stay byte-identical. No code changes — pure dataset relabel.

| Field | Value |
|---|---|
| Brev cost | ~3-5h (re-fine-tune with new prompts) |
| Eng cost | ~4h (dataset relabel script) |
| Bonus | +20 |
| Expected lift | +0.5 to +1.5 rollouts |
| Risk | low |
| Verdict | Cheapest VQA-equivalent. Pairs cleanly with anything. |

### Option 6 — SmolVLA-boost-v2 (1+2+3 combined) `[A]`

Stack reference recuration + print augmentation + ArcFace distillation in one resumed fine-tune. The primary recommendation in `RESEARCH_v3` for the image-as-prompt branch.

| Field | Value |
|---|---|
| Brev cost | ~5h |
| Eng cost | ~1.5 days |
| Bonus | +20 |
| Expected lift | +2 to +4 rollouts |
| Risk | medium |
| Verdict | **Primary recommendation if TAs allow image-as-prompt.** |

### Option 7 — SmolVLA name-only `[B]`

Re-fine-tune SmolVLA on the 4195 episodes with the **reference stream ignored** (set `--policy.empty_cameras=2` so camera2 + camera3 are zero-padded) and prompts that are just `"Place the coke on Yann LeCun"`. The model has nothing but the prompt name + camera1 view to identify the right celeb.

| Field | Value |
|---|---|
| Brev cost | ~8h (re-fine-tune from scratch; the existing ckpt expected camera2) |
| Eng cost | ~6h dataset relabel |
| Bonus | +20 |
| Expected lift | unknown — depends entirely on the VLM's ability to bind name→face. **Probably catastrophic without Option 4 or 5** as a warm-up. |
| Risk | high (450M SmolVLM2 has weak name-binding capacity without VQA) |
| Verdict | Only viable as **Option 7 + Option 4 stacked.** SmolVLA alone with name-only and no VQA has very weak evidence. |

### Option 8 — SmolVLA + VLM-warm-start + name-only `[B]`

Stack Option 4 (VLM VQA pretrain) + Option 7 (SmolVLA name-only fine-tune).

| Field | Value |
|---|---|
| Brev cost | ~8-10h total |
| Eng cost | ~1.5 days |
| Bonus | +20 |
| Expected lift | +1 to +3 rollouts (most uncertainty here) |
| Risk | medium-high |
| Verdict | **Primary recommendation if TAs require text-only AND we stay on SmolVLA.** |

### Option 9 — Pi0.5 + image-as-prompt `[A]`

Train Pi0.5 (`lerobot/pi05_base`, ~3.3B active params) on our existing image-as-prompt dataset. Same protocol as our current SmolVLA, different backbone. PaliGemma-3B vision tower (SigLIP-So400m at 400M params) is substantially larger than SmolVLM2's vision component.

| Field | Value |
|---|---|
| Brev cost | ~27-33h (bs 16-24, gradient checkpointing + bf16 + compile_model) |
| Eng cost | ~6h (run_training.sh edits + quantile-stats preprocessing) |
| Bonus | -4 (rank 3 expected: +16) |
| Expected lift | +1 to +4 rollouts (capacity-bet; no published face-match benchmark) |
| Risk | medium |
| Verdict | **The "bigger model" hypothesis, fairly tested.** Worth running if Brev budget allows ≥30h. |

### Option 10 — Pi0.5 + name-only (original pre-pivot plan) `[B]`

Train Pi0.5 with name-only prompts directly. Bets on PaliGemma's WebLI pretraining knowing the IID and OOD-popular celebrities well enough that fine-tuning unlocks name→face binding.

| Field | Value |
|---|---|
| Brev cost | ~25-30h |
| Eng cost | ~6h dataset relabel + ~3h training config |
| Bonus | -4 (rank 3: +16) |
| Expected lift | unknown — depends heavily on PaliGemma's celeb-recall (which the 2026-05-09 zero-shot probe failed, but that test may not have been representative) |
| Risk | medium-high |
| Verdict | **Tests whether the original architecture decision was correct.** Most epistemically valuable single experiment. |

### Option 11 — Pi0.5 + VLM-warm-start + name-only `[B]`

Add a VQA warm-start on PaliGemma-3B before the action fine-tune. Same idea as Option 4 but on the bigger backbone.

| Field | Value |
|---|---|
| Brev cost | ~30-35h total |
| Eng cost | ~1.5 days |
| Bonus | -4 (rank 3: +16) |
| Expected lift | +2 to +5 rollouts (highest potential ceiling, highest cost) |
| Risk | medium |
| Verdict | **Most-likely-to-succeed option if name-only is mandatory.** Highest variance. |

### Option 12 — Hybrid Pi0.5 + ArcFace distillation `[A or B]`

Apply the BlindVLA distillation recipe to Pi0.5's PaliGemma vision tower (specifically SigLIP-So400m's `last_hidden_state` before the projector). Should give the best of both: bigger backbone + explicit face-discriminative features.

| Field | Value |
|---|---|
| Brev cost | ~30-35h |
| Eng cost | ~2 days (port the SmolVLA hook to Pi0.5's `paligemma_with_expert.py`) |
| Bonus | -4 (rank 3: +16) |
| Expected lift | +2 to +5 rollouts |
| Risk | medium-high (porting effort; Pi0.5's vision pipeline differs from SmolVLA's) |
| Verdict | **Maximum-effort capacity-bet path.** Only worth it if a Pi0.5 baseline run (Option 9) reveals capacity actually helps. |

### Option 13 — Train Option 6 (SmolVLA-boost) AND Option 9 (Pi0.5) in parallel

Both run on Brev concurrently or sequentially. ~$25-40 of the remaining $200 budget. Pick the better on demo-day dry-run.

| Field | Value |
|---|---|
| Brev cost | ~35h combined |
| Eng cost | ~2 days total |
| Bonus | best-of-two (+20 if SmolVLA wins, +16 if Pi0.5 wins) |
| Expected | best-of-two |
| Risk | low (hedged) |
| Verdict | **Risk-adjusted optimum if TAs allow image-as-prompt and we have ≥35h Brev.** |

### Option 14 — Train Option 8 (SmolVLA name-only) AND Option 11 (Pi0.5 name-only) in parallel

The name-only equivalent of Option 13.

| Field | Value |
|---|---|
| Brev cost | ~35-40h |
| Eng cost | ~2.5 days |
| Bonus | best-of-two |
| Verdict | **Risk-adjusted optimum if TAs require text-only and we have ≥35h Brev.** |

### Option 15 — Collect new HG-location teleops `[AB]`

The spec recommends collecting some training data at the actual eval location (HG building). We haven't. This addresses an environmental shift that may be compounding the face-matching failure.

| Field | Value |
|---|---|
| Brev cost | depends — small additional fine-tune budget |
| Eng cost | several hours of human time on Thor at HG |
| Bonus | any (architecture-agnostic) |
| Expected lift | unclear — addresses lighting/background shift, not face-matching directly |
| Risk | low |
| Verdict | **Worth doing regardless of architecture choice.** Time permitting. |

### Options not pursued

- **OpenVLA-7B** — agent B verified 14GB weights + KV cache exceeds Strix's 16GB at 30Hz. Deploy-blocked.
- **X-VLA-0.9B** — user-rejected.
- **TinyVLA-400M / FlowerVLA / Smol-0-VLA** — no LeRobot integration per `docs/RELATED_WORK.md`; building a new policy class is multi-day work. The spec recommended these but the LeRobot ecosystem doesn't support them out-of-the-box.
- **GR00T-N1.5 / OpenVLA / RoboFlamingo** — too large, no bonus tier benefit.
- **DAgger / interactive correction** — addresses symptoms not the root cause; slow.

---

## 3. The architecture decision tree

```
TA Slack: image-as-prompt at inference?
│
├── YES, allowed
│   ├── Best bet (low risk): Option 6 = SmolVLA-boost-v2
│   ├── Better-evidence-but-higher-cost: Option 13 = Option 6 + Option 9 (Pi0.5) in parallel
│   ├── Add-on regardless: Option 1 (recuration), Option 5 (caption aug)
│   └── Stretch: Option 12 (Pi0.5 + distillation) only if a baseline Pi0.5 run shows capacity helps
│
└── NO, text-only
    ├── Best bet (low risk): Option 8 = SmolVLA + VQA warm-start + name-only
    ├── Better-evidence-but-higher-cost: Option 14 = Option 8 + Option 11 (Pi0.5 name-only + VQA) in parallel
    ├── Bare-minimum baseline: Option 7 = SmolVLA name-only without VQA (probably fails but informs)
    └── Best ceiling: Option 11 alone if we want to maximize the chance of success and accept the bonus loss
```

---

## 4. The reasoning chain per technique

### 4.1 Why reference-photo recuration matters (Option 1)

Three independent biometric standards ([ISO/IEC 19794-5:2011 §7](https://www.iso.org/standard/50867.html), [NIST FRVT Quality](https://pages.nist.gov/frvt/html/frvt_quality.html), [Paravision biometric whitepaper](https://www.paravision.ai/whitepaper-face-recognition-and-biometric-image-quality/)) converge on what "enrollment-quality" means: yaw/pitch/roll ≤ ±5° (relax to ±15°), inter-eye ≥ 60 px, uniform illumination, plain background. Our 192-celeb scraped bank doesn't meet these uniformly — many photos are profiles, full-body, or magazine-filtered. The model sees noise instead of identity.

This is a **purely inference-time** fix (we ship a static curated asset table, no training changes). It's the cheapest possible improvement and is bonus-preserving.

### 4.2 Why print-domain augmentation matters (Option 2)

[Arxiv 2404.06559 §4](https://arxiv.org/html/2404.06559v2) quantifies the magazine-photo → printed-A5-cutout transformation at **+5.64% / +16.00% FMR shift** on ArcFace verification — a real, published, non-trivial domain gap. Effects ordered by published magnitude: halftone dot pattern > color gamut compression (sRGB→CMYK, ~35%→21% of CIE 1931) > paper grain (1/f noise) > print MTF blur > tone compression. [Augraphy (arxiv 2208.14558)](https://arxiv.org/abs/2208.14558) is the canonical library; our recipe adapts its ink→paper→post operator order to photo-as-print.

Training the model on print-domain-augmented references closes the gap so eval-day input matches the training distribution.

### 4.3 Why ArcFace distillation matters (Option 3)

SmolVLA's SigLIP vision tower compresses each image into **64 tokens** via 2×2 pixel-shuffle ([SmolVLM, arxiv 2504.05299 §3.1](https://arxiv.org/html/2504.05299v1)). For a face occupying ~30% of an image, that's ~10-20 identity-bearing tokens. SigLIP was pretrained for image-text alignment, not face identity — the 64-token bottleneck doesn't preserve the angular geometry that distinguishes faces.

ArcFace ([arxiv 1801.07698](https://arxiv.org/abs/1801.07698)) explicitly trains a face encoder on a hypersphere with margin loss; same-person photos are angularly close, different-person photos are far. Pulling SigLIP's face-region patches toward ArcFace's embedding via a per-patch negative-mean cosine loss (BlindVLA equation 9, [arxiv 2510.25616](https://arxiv.org/html/2510.25616v1)) injects identity-discriminative geometry without changing the action expert or LM head.

BlindVLA reports **+24% semantic OOD / +12% vision OOD** on LIBERO with a *general* vision teacher (DINOv2/SigLIP/Theia). A face-specific teacher (ArcFace) on a face-specific task should match-or-exceed that. **No published paper does this exact distillation combination** — we're first-movers on ArcFace→SigLIP; the recipe transfers in principle from BlindVLA's general-teacher pattern.

ArcFace lives offline only (its embeddings are cached); rule-compliant per PROJECT.md §3.

### 4.4 Why VLM-only VQA warm-start matters (Option 4)

A VLA = VLM (vision-language) + action expert. The VLM is what binds prompt-text "Yann LeCun" to a visual pattern. When you fine-tune the VLA on robot demos, the VLM's language-image understanding **drifts** because the robot loss (flow-matching MSE on actions) doesn't reinforce it. This is the "Don't Blind Your VLA" failure ([arxiv 2510.25616](https://arxiv.org/html/2510.25616v1)) — frozen VLM features become misaligned with the evolving action head.

Pre-training the VLM on face-VQA before the SmolVLA fine-tune builds a strong name↔face prior in the VLM. The standard pattern (Pi0.5-KI [arxiv 2505.23705](https://arxiv.org/html/2505.23705v1)) trains the VLM and the action expert in one shot with gradient stops; the **VLM-only-then-action** sequential variant is cheaper, doesn't require multi-task batch interleaving, and avoids LeRobot's `MultiLeRobotDataset = NotImplementedError` blocker entirely.

Concretely: load SmolVLM2-500M from HuggingFace, fine-tune for ~5k steps on face-VQA pairs (face crop, "Who is this?", name) drawn from our 192-celeb bank, save the checkpoint as `lemonkey/smolvlm2_face_warm`, then use `--policy.vlm_model_name=lemonkey/smolvlm2_face_warm` in `lerobot-train`. ~3-6h Brev.

### 4.5 Why Pi0.5 is back on the table (Options 9-12)

Earlier dismissals leaned on the 2026-05-09 PaliGemma probe (0/14 TOY zero-shot naming). That probe tested **zero-shot open-ended naming** — "given a photo, output 'Yann LeCun'." That is *not* our task. Our task is *closed-set selection*: "given the name 'Yann LeCun' and 3 visible portraits, place the can on the matching one." The probe doesn't disprove Pi0.5 can do *that* after fine-tuning.

Additionally, the OOD celebrities in Eval 3 runs 7-9 are explicitly *"very popular celebrities"* — Federer / Bezos / Beyoncé level. WebLI (PaliGemma's pretraining corpus, ~10B image-text pairs) contains many photos of these figures with name captions. The face-name binding may already be in the weights, just not surfaced by the wrong probe question.

**Honest answer:** Pi0.5 + name-only at our project may or may not work. Nobody tested it. The bonus math says it must beat SmolVLA by ≥1 rollout to come out ahead. That's an achievable bar, not an impossible one.

### 4.6 Why VQA isn't a silver bullet (the honest caveat)

VQA fixes name-binding. It does **not** fix:
- The action expert's positional shortcut (if it has one)
- The print-domain gap on camera1 (workspace cam still sees the print directly)
- The 64-token visual bottleneck for fine-grained identity discrimination

Best results come from **stacking** VQA warm-start with reference recuration + print augmentation + (if going image-as-prompt) ArcFace distillation. Single-technique fixes are unlikely to be sufficient.

---

## 5. The honest scorecard

Estimates use the bonus economics from agent D (`RESEARCH_v3` §6) — bonus differential of +4 = 0.72 rollouts.

| Option | If TA = YES (image-as-prompt) | If TA = NO (text-only) | Cost | Bonus |
|---|---|---|---|---|
| 0 — Ship as-is | 31 pts | invalid (model expects camera2) | 0 | +20 |
| 1 — Refs only | 32-38 pts | invalid | low | +20 |
| 2 — Print aug only | 36-42 pts | invalid | low | +20 |
| 3 — ArcFace distill only | 36-42 pts | invalid | low | +20 |
| 4 — VLM-warm-start | 38-44 pts | 36-42 pts | low | +20 |
| 5 — Caption aug | 35-40 pts | 35-40 pts | very low | +20 |
| **6 — SmolVLA-boost-v2** | **42-50 pts** | invalid | low/med | +20 |
| 7 — SmolVLA name-only | invalid | 22-32 pts | medium | +20 |
| 8 — Option 7 + Option 4 | invalid | **38-48 pts** | medium | +20 |
| 9 — Pi0.5 + image-as-prompt | 44-50 pts | invalid | high | +16 |
| 10 — Pi0.5 name-only | invalid | 32-44 pts | high | +16 |
| 11 — Option 10 + VLM warm-start | invalid | **42-52 pts** | high | +16 |
| 12 — Pi0.5 + ArcFace distill | 46-52 pts | invalid | very high | +16 |
| 13 — Parallel SmolVLA-boost + Pi0.5 IaP | **best-of-6,9 = 46-52 pts** | invalid | very high | +16 to +20 |
| 14 — Parallel SmolVLA-no + Pi0.5-no | invalid | **best-of-8,11 = 44-52 pts** | very high | +16 to +20 |

**Bold** marks the recommended path in each protocol scenario.

---

## 6. The recommended plan

### Step 0 (do today, 30 minutes): Resolve the protocol question

Slack the TAs:
> "For Eval 3, may our policy receive a reference image of the named celebrity as a 2nd camera input at inference time (looked up from a pre-built name→photo asset table that ships with the policy)? Or must the policy take only the text prompt as input, with no auxiliary image lookups?"

Decisions below depend on the answer.

### Step 1 (do today regardless of TA answer)

- **Option 1 (reference recuration)** — pure asset prep, runs locally, does not require TA input. Output: 192 high-quality face-cropped portraits + a manifest table. ~4h.
- **Option 4 (VLM-only VQA warm-start)** — pure VLM pretrain, completely independent of the SmolVLA architecture and the TA protocol question. Output: a `lemonkey/smolvlm2_face_warm` HF checkpoint that we can plug into any subsequent VLA training. ~1 day eng + ~3-6h Brev.

These two are unconditional. Run them in parallel today.

### Step 2 (do tonight, conditional on TA answer)

**If TA = YES (image-as-prompt):** launch **Option 6 (SmolVLA-boost-v2 = Refs + Print aug + ArcFace distillation)** as the primary track. If Brev budget allows ≥30h more, also launch **Option 9 (Pi0.5 + image-as-prompt)** as parallel insurance.

**If TA = NO (text-only):** launch **Option 8 (SmolVLA + VQA + name-only)** as the primary track. If Brev budget allows, also launch **Option 11 (Pi0.5 + VQA + name-only)** as parallel insurance.

### Step 3 (tomorrow, pre-eval)

Dry-run all trained checkpoints on the workspace at 3 rollouts each (1 TOY-IID, 1 held-out-IID, 1 OOD). Pick the best.

---

## 7. Hard constraints to remember

- **VLA-only at inference.** PROJECT.md §3 forbids any non-VLA model running at inference. Everything in this doc (ArcFace distillation, RetinaFace masks, VLM warm-start, reference photo cropping) happens **offline at training time only**. Cached asset tables (face crops, embeddings) are not models.
- **20 s per rollout.** Bigger models cost more inference latency. SmolVLA at 30 Hz with action chunk 50 is plenty fast. Pi0.5 needs verification.
- **16 GB VRAM at Strix deploy.** OpenVLA-7B doesn't fit. SmolVLA fits trivially. Pi0.5 is borderline — needs an empirical probe.
- **Brev budget.** ~$50-100 spent on the SmolVLA-eval3 training so far. Remaining ~$100-150. Each option lists its compute cost.

---

## 8. Citations and sources

### Core VLA papers
- **SmolVLA** — [arxiv 2506.01844](https://arxiv.org/abs/2506.01844)
- **Pi0.5** — [arxiv 2504.16054](https://arxiv.org/abs/2504.16054) + [pi.website/blog/pi05](https://www.pi.website/blog/pi05)
- **Pi0.5-KI (knowledge insulation)** — [arxiv 2505.23705](https://arxiv.org/html/2505.23705v1)
- **π0** — [arxiv 2410.24164](https://arxiv.org/abs/2410.24164)
- **Interleave-VLA** — [arxiv 2505.02152](https://arxiv.org/abs/2505.02152)
- **OpenVLA** — [arxiv 2406.09246](https://arxiv.org/abs/2406.09246)
- **TinyVLA** — [arxiv 2409.12514](https://arxiv.org/abs/2409.12514)
- **SmolVLM** — [arxiv 2504.05299](https://arxiv.org/html/2504.05299v1)

### Training techniques
- **Don't Blind Your VLA (alignment loss)** — [arxiv 2510.25616](https://arxiv.org/html/2510.25616v1) + [github.com/CognitiveAISystems/BlindVLA](https://github.com/CognitiveAISystems/BlindVLA)
- **ArcFace** — [arxiv 1801.07698](https://arxiv.org/abs/1801.07698)
- **MagFace (quality-aware face recognition)** — [arxiv 2103.06627](https://arxiv.org/abs/2103.06627)
- **Evaluation-Oriented KD for FR (CVPR 2022)** — [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Evaluation-Oriented_Knowledge_Distillation_for_Deep_Face_Recognition_CVPR_2022_paper.pdf)
- **ICD-Face (ICCV 2023)** — [PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_ICD-Face_Intra-class_Compactness_Distillation_for_Face_Recognition_ICCV_2023_paper.pdf)
- **Unified KD Framework** — [arxiv 2508.11376](https://arxiv.org/html/2508.11376v1)

### Print-domain literature
- **Print-and-Scan morph attacks** — [arxiv 2404.06559](https://arxiv.org/html/2404.06559v2)
- **Augraphy (paper-emulation library)** — [arxiv 2208.14558](https://arxiv.org/abs/2208.14558) + [docs](https://augraphy.readthedocs.io/)
- **Heterogeneous FR** — [arxiv 2404.14247](https://arxiv.org/abs/2404.14247) + [arxiv 2307.07032](https://arxiv.org/abs/2307.07032)

### Biometric standards
- **NIST FRVT Quality** — [pages.nist.gov/frvt/html/frvt_quality](https://pages.nist.gov/frvt/html/frvt_quality.html)
- **ISO/IEC 19794-5:2011** — [iso.org/standard/50867](https://www.iso.org/standard/50867.html)
- **InsightFace model zoo** — [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)

### Face-VQA datasets
- **VGGFace2** — [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) (Academic Torrents mirror)
- **CelebA-Dialog** — [github.com/ziqihuangg/CelebA-Dialog](https://github.com/ziqihuangg/CelebA-Dialog)

### LeRobot pointers (local)
- [`third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`](../third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py)
- [`third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py`](../third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py)
- [`third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py`](../third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py)
- [`third_party/lerobot/src/lerobot/datasets/factory.py`](../third_party/lerobot/src/lerobot/datasets/factory.py) — `MultiLeRobotDataset` NotImplementedError blocker

### Project docs
- [`docs/PROJECT.md`](PROJECT.md) — eval rubric, smallest-model bonus, VLA-only constraint
- [`docs/VLA_ARCHITECTURES.md`](VLA_ARCHITECTURES.md) — architecture inventory and knob taxonomy
- [`docs/RELATED_WORK.md`](RELATED_WORK.md) — prior public work survey
- [`eval_3/README.md`](../eval_3/README.md) — project plan + the 2026-05-09 PaliGemma probe
- [`eval_3/aug/STRATEGY_v3.md`](../eval_3/aug/STRATEGY_v3.md) — augmentation strategy
- [`eval_3/aug/RESEARCH_v3_face_matching_rescue.md`](../eval_3/aug/RESEARCH_v3_face_matching_rescue.md) — predecessor doc, image-as-prompt focused
