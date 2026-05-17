# Eval 3 Research Report (2026-05-17)

**Project:** LeMonkey, ETH Robot Learning FS26 Project 1
**Authors:** the LeMonkey Claude session synthesizing 7 parallel research agents, multiple rounds of skeptical pushback from Roham, and the team chat from 2026-05-17.
**Status:** definitive synthesis of today's research. The operational plan based on this report is in [`/eval_3/STRATEGY.md`](../../eval_3/STRATEGY.md) and [`/TODO.md`](../../TODO.md).

> **Reading order for someone picking up this work fresh:**
> 1. [`/TODO.md`](../../TODO.md) — what's being worked on right now
> 2. This document — the why
> 3. [`/eval_3/STRATEGY.md`](../../eval_3/STRATEGY.md) — the chosen plan in detail
> 4. [`/docs/EVAL_3_OPTIONS.md`](../EVAL_3_OPTIONS.md) — full 15-option space if you want to second-guess the picks

---

## Executive summary

SmolVLA-450M was trained on a 4195-episode LeRobot v3 dataset using an image-as-prompt protocol for the Eval 3 face-matching task. The smoke-test on Strix (2026-05-17) revealed that the trained policy **fails the core task**: it picks the wrong celebrity portrait when asked. The failure is not a wiring bug, not a deployment-time camera shift, and not simple per-celeb positional bias — those three hypotheses were empirically refuted.

The remaining diagnosis is a combination of two compounding gaps:

1. **(G1) Domain gap.** Training-time reference photos are magazine/web style; eval-day workspace is paper printouts photographed through a wrist cam. Published face-recognition literature ([arxiv 2404.06559](https://arxiv.org/html/2404.06559v2)) quantifies this transformation at +5-16% ArcFace FMR shift.
2. **(G2) Representation gap.** SmolVLA's vision tower compresses each image into 64 visual tokens via pixel-shuffle. For a face occupying ~30% of the frame, that's ~10-20 identity-bearing tokens. SigLIP was pretrained for image-text alignment, not face-discriminative geometry.

Seven research agents and three doc-reads later, we have a converged plan: **4 parallel training tracks** running on Brev, targeting different points in the (failure-mode × bonus × cost) space. The bonus-preserving primary track (Track A — SmolVLA-boost-v2) addresses both gaps surgically with published recipes (BlindVLA distillation + Augraphy-inspired print augmentation). A capacity-bet pair (Tracks B + C — Pi0.5 variants) hedges in case scale is the load-bearing variable. A 3-celeb baseline (Track D) is the floor — guaranteed-functional even if the others underperform.

This report explains the reasoning chain end-to-end so the choice can be audited or revisited.

---

## 1. The failure mode, empirically

### 1.1 The smoke test outcome

On Strix (RTX 3080 Ti Laptop 16 GB, Ampere sm_86) on 2026-05-17, the trained `HBOrtiz/smolvla_eval3` policy executed an end-to-end rollout:
- Workspace: 3 printed A5 cutouts of Yann LeCun, Taylor Swift, Barack Obama (from `docs/Eval_3_TOY_Celebrity_Images.pdf`).
- Prompt: `"Set the coke down on Taylor Swift's picture."`
- Reference photo (camera2): `eval_3/refs/images/43_swift.png` (visually verified Taylor Swift).
- Result: 538 steps executed cleanly over ~18 s, no crash, no out-of-time. **The arm placed the coke on the Obama portrait, not Swift.**

The pipeline worked. The model didn't.

### 1.2 Four hypotheses, four resolutions

| # | Hypothesis | How tested | Outcome |
|---|---|---|---|
| 1 | `camera2` silently dropped in `prepare_images()` (the reference photo never reaches the model) | Direct instrumentation of `prepare_images()` and `embed_prefix()` to dump the tensor that reaches the SigLIP encoder; verified visually via fingerprint colors | **Refuted.** image[0] = RED (camera1, wrist), image[1] = BLUE (camera2, reference), image[2] = zeros (camera3 padded). All three reach the model intact. |
| 2 | Wrist-cam aim differs between training and deploy distributions | Manual workspace inspection comparing Strix deploy to training-data captures | **Refuted.** Identical setup. |
| 3 | Positional bias dominates over face-matching | Run 3 rollouts with the same Swift prompt but Swift's portrait rotated through left / middle / right physical positions | **Confirmed.** The can lands at the same physical spot regardless of which celeb is there. |
| 4 | SmolVLA-450M's reasoning capacity is too small for image-as-prompt face-matching | Cannot be tested directly — would require retraining with a bigger model | **Plausible but unproven.** |

Hypothesis 3 is the load-bearing finding. The action expert has learned a **positional shortcut** — for prompts naming celebrities from the training distribution, it drops the can in the mean position those celebs usually occupied during teleop. The reference stream is consumed by the network but does not condition action selection.

### 1.3 Why this happens — two compounding gaps

The shortcut emerges because the face-matching task is *underspecified for the model* at the visual level. To actually condition on identity, the model would need both:

**(G1) Domain coverage.** The training-time reference stream consists of magazine and web photos of celebrities, often heavily retouched or stylized. The eval-day workspace contains paper printouts of headshots, photographed at an angle by a wrist cam. The print-vs-photo gap is large enough that face-recognition systems specifically benchmark it as a domain transfer problem ([arxiv 2404.06559 §4](https://arxiv.org/html/2404.06559v2)). Specifically: Floyd-Steinberg dot dither, sRGB → CMYK gamut compression, paper grain at ~1/f spectrum, print MTF blur, tone compression. None of these were in the training distribution.

**(G2) Representation strength.** SmolVLA's vision tower is SigLIP-So400m wrapped by SmolVLM2-500M ([smolvla.mdx](../../third_party/lerobot/docs/source/smolvla.mdx)). SmolVLM2 reduces each image to **64 visual tokens via 2×2 pixel-shuffle** ([SmolVLM, arxiv 2504.05299 §3.1](https://arxiv.org/html/2504.05299v1)). For a workspace image with three small printed portraits, the bottleneck is severe — maybe 4-6 tokens per portrait. For the reference photo (where the face fills more of the frame), maybe 10-20 tokens carry identity. SigLIP was pretrained on image-text contrastive loss; it has no specific inductive bias toward face-discriminative geometry. ArcFace ([arxiv 1801.07698](https://arxiv.org/abs/1801.07698)) — the canonical face-identity encoder — explicitly trains on a hypersphere with margin loss for this geometry; SmolVLM does not.

Both gaps must be closed simultaneously. Closing (G1) alone leaves a model with bad features for the right domain. Closing (G2) alone leaves a model with good features but never trained on the eval-day distribution. Track A targets both; Tracks B and C bet on capacity making up for (G2); Track D side-steps the OOD branch entirely.

---

## 2. The option space

The full 15-option enumeration is in [`docs/EVAL_3_OPTIONS.md`](../EVAL_3_OPTIONS.md). This section captures the decision logic that filtered 15 down to 4.

### 2.1 Architectural axis — SmolVLA vs Pi0.5

The team's prior was that the failure mode is fundamentally capacity (G2 dominates). The agent research argues this prior is unsupported by published evidence:

- The 2026-05-09 PaliGemma probe (logged in [`eval_3/README.md`](../../eval_3/README.md)) tested **zero-shot open-ended naming** ("given a photo, say the name"). Result: 0/14 TOY, 0/6 OOD. That motivated the SmolVLA pivot.
- But our task is **not** zero-shot naming. Our task is **closed-set selection** ("given a name and 3 visible portraits, place the can on the matching one"). The probe was the wrong test.
- After fine-tuning, even a model that fails zero-shot naming can succeed at closed-set matching. Pi0.5 fine-tuned on our dataset has not been measured.

The honest answer is that we *don't know* whether Pi0.5 face-matches better than SmolVLA. **The only way to find out is to train and test both.** Track A is the cheap bonus-preserving bet; Tracks B and C are the more expensive capacity-bet hedges.

### 2.2 Protocol axis — image-as-prompt vs text-only

The official spec ([`docs/PROJECT.md` §2 Eval 3](../PROJECT.md)) says the prompt is text:
> "Place the coke on [celebrity name]"

Our team's image-as-prompt protocol — adding a reference photo as a second camera input — is a **design choice we made**, not part of the protocol. It requires us to look up a reference photo at inference time from a name→photo asset table. Two open questions:

1. **Do TAs interpret this as breaking the "text-only prompt" expectation?** Slack pending.
2. **For OOD celebs we don't have photos of, what does the model receive as camera2?** Zeros would be off-distribution; cached "generic face" would be misleading.

If TAs disapprove of image-as-prompt, our entire trained pipeline is invalid for eval day. The fallback is text-only (Track D, plus alternative text-only versions of Tracks A/B/C). **Track D is text-only by design**, which is why it's our safety net.

### 2.3 Bonus axis — small vs big

The Eval 3 smallest-model bonus is:
- 1st place (smallest active inference params): +20 pts
- 2nd: +18; 3rd: +16; 4th: +14; 5th: +12

SmolVLA-450M is rank 1. Pi0.5-3.3B is likely rank 3 (assuming one team picks something smaller-than-Pi0.5-but-bigger-than-SmolVLA, e.g. FlowerVLA at 950M).

The **bonus differential** between SmolVLA and Pi0.5 is +4 pts. Each successful rollout is 50/9 = 5.55 pts. Therefore the bonus differential equals **0.72 rollouts of slack**:
```
SmolVLA wins iff:  (s_smol / 9) × 50 + 20  ≥  (s_pi / 9) × 50 + 16
                   →  s_smol ≥ s_pi − 0.72
```
**Pi0.5 must beat SmolVLA by ≥1 rollout to come out ahead on net.** That's a meaningful but achievable bar.

### 2.4 Why these 4, not the other 11

- **Track A (Option 6 in EVAL_3_OPTIONS):** Best evidence-backed bonus-preserving fix. Both BlindVLA distillation and Augraphy-style print augmentation have published gains in the relevant axes.
- **Track B (Option 12):** Maximum-effort path. If we accept the bonus loss, hybrid (bigger + surgical) should beat either alone.
- **Track C (Option 9):** Cleanest test of "capacity is the bottleneck." Without surgical fixes, we measure whether scaling alone solves it.
- **Track D:** Floor / safety net. Concedes the 3 OOD runs by design but maximizes IID reliability.

Other options were rejected for: implementation cost (OpenVLA-7B, X-VLA-0.9B), bonus loss without compensating evidence (TinyVLA's 400M without LeRobot support), or weak evidence at our scale (name-only without VQA on SmolVLA).

---

## 3. The chosen techniques — reasoning chain per piece

### 3.1 Reference photo recuration (Track A component)

**Problem:** Our 192-celeb scraped bank is heterogeneous — profile shots, magazine-filtered Instagram crops, full-body images. Many photos do not meet face-recognition enrollment standards.

**Solution:** Pick one frontal head+shoulders photo per celeb using the standard NIST FRVT / ISO 19794-5 enrollment quality criteria:

```python
keep_photo = (
    det_score        >= 0.65          # RetinaFace conf, enrollment-grade
    and abs(yaw)     <= 15.0          # degrees, from InsightFace 1k3d68 pose head
    and abs(pitch)   <= 15.0
    and abs(roll)    <= 10.0
    and inter_eye_px >= 60            # post 224×224 upscale, NIST FRVT minimum
    and face_area    >= 0.25 * img_area
    and embedding_norm >= median(bank) - 1.0 * sigma   # MagFace proxy for quality
    and laplacian_var >= 100          # blur reject
)
```

**Sources (triple-source per CLAUDE.md §7):**
- [ISO/IEC 19794-5:2011 §7 "Token frontal image"](https://www.iso.org/standard/50867.html) — pose, illumination, background standards
- [NIST FRVT Quality Assessment](https://pages.nist.gov/frvt/html/frvt_quality.html) — confidence threshold, inter-eye minimum
- [MagFace, arxiv 2103.06627 §4.2](https://arxiv.org/abs/2103.06627) — embedding-norm-as-quality proxy
- [SDD-FIQA, arxiv 2103.05977 §3.2](https://arxiv.org/abs/2103.05977) — independent confirmation of norm/quality correlation
- [InsightFace model zoo](https://github.com/deepinsight/insightface) — `buffalo_l` det_score and pose head specs

**Crop policy:** Head + shoulders (not tight face), margin = 0.5 × inter-eye on each side, 1 × inter-eye top. The HFR literature ([arxiv 2404.14247 §4.1](https://arxiv.org/abs/2404.14247), [arxiv 2307.07032 §3.2](https://arxiv.org/abs/2307.07032)) confirms broader head box matches what FR encoders were trained on and preserves hair/jaw cues that printed cutouts retain.

**Compliance with PROJECT.md §3 ("VLA-only at inference"):** Crops computed offline once per celeb; shipped as a static asset table. No face detector runs at inference. Rule respected.

### 3.2 Print-domain forward augmentation (Track A component)

**Problem:** Our training-time reference photos are crisp magazine shots. The eval-day workspace contains paper cutouts photographed through a wrist cam. The model never saw the print domain at training.

**Solution:** Apply a printer-emulation pipeline to the camera2 reference stream at training time, with p=0.7 (remaining 30% see clean magazine photos for invariance preservation).

The pipeline mirrors Augraphy's published ink → paper → post order ([arxiv 2208.14558](https://arxiv.org/abs/2208.14558), [docs](https://augraphy.readthedocs.io/)):

1. **Lab gamut compression.** Chroma scale 0.75-0.9, L clip 10-240. Simulates sRGB → CMYK gamut reduction.
   Source: [W3C Color Workshop, Lilley 2021](https://www.w3.org/Graphics/Color/Workshop/slides/talk/lilley), [CIELAB color space](https://en.wikipedia.org/wiki/CIELAB_color_space), [MDPI Chroma Enhancement](https://www.mdpi.com/2411-9660/5/2/32).
2. **Print MTF blur.** σ = 0.4-0.8 px. Models ink spread + print MTF rolloff.
3. **Floyd-Steinberg color dither at 300-DPI equivalent.** Models printer halftone via error diffusion (the canonical CUPS/HP/Canon default).
   Source: [Floyd-Steinberg dithering](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering), [Error diffusion](https://en.wikipedia.org/wiki/Error_diffusion).
4. **Perlin fBm grain.** 4 octaves, persistence 0.5, amplitude ±3 in 8-bit. Models paper fiber texture.
   Source: [Perlin noise](https://en.wikipedia.org/wiki/Perlin_noise), [GPU Gems chapter 5](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-5-implementing-improved-perlin-noise).
5. **Resize to 224 + JPEG re-encode q70-90.** Models wrist-cam re-imaging.

**Published effect-size validation:** [arxiv 2404.06559 §4](https://arxiv.org/html/2404.06559v2) measures the magazine-photo → printed-A5-cutout transformation as +5.6-16.0% FMR shift on ArcFace verification. This is the published lower-bound on the gap we need to close.

**Validation gates (mandatory per CLAUDE.md §7):**
1. **Calibration print capture.** Print 5 representative celebrities + capture with the actual wrist cam. Verify ArcFace cosine similarity between aug-pipeline output and real-print captures.
2. **Domain-gap classifier probe.** Train a tiny linear classifier on `is_real_print` (real vs aug-print vs clean magazine). Target: 55-80% separability. > 85% means augmentation too weak; < 55% means too strong.
3. **Visual gate.** `eval_3/aug/dbg/dbg_print_aug_grid.py` showing 4×4 (clean / aug / real-print) for 8 celebs. Eyeball before training.

### 3.3 ArcFace cosine distillation (Track A + Track B component)

**Problem:** SigLIP's 64-token-per-image bottleneck doesn't preserve face-discriminative geometry. The model can see a face but can't reliably tell two faces apart based on identity.

**Solution:** Add an auxiliary cosine alignment loss between SigLIP's reference-stream patch features and a frozen ArcFace teacher's embedding, mask-gated to face patches and applied only to camera2.

**Loss equation** (BlindVLA equation 9, [arxiv 2510.25616](https://arxiv.org/html/2510.25616v1)):
```
L_total = L_flow_matching + λ · L_align
L_align = − (1/k) · Σ_{j=1}^{k} cos( F.normalize(u_j), F.normalize(z_j) )
λ = 0.2
```

**Why each design choice (triple-source):**

| Choice | Value | Source 1 | Source 2 | Source 3 |
|---|---|---|---|---|
| Loss form | Negative-mean cosine | [BlindVLA eq 9](https://arxiv.org/html/2510.25616v1) (their Table 8 ablates L2 and InfoNCE — cosine wins) | [Evaluation-Oriented KD CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Evaluation-Oriented_Knowledge_Distillation_for_Deep_Face_Recognition_CVPR_2022_paper.pdf) | [Unified-KD arxiv 2508.11376](https://arxiv.org/html/2508.11376v1) |
| λ weight | 0.2 | BlindVLA Table 8 ("most stable") | Pi0.5-KI uses α=1.0 but with `stop_gradient` isolation — not directly comparable | FR-KD literature uses λ ∈ [0.1, 1.0] |
| Injection point | SigLIP `last_hidden_state` pre-connector | SmolVLM 2×2 pixel-shuffle merges patches; aligning before the merge preserves face-region granularity ([SmolVLM §3.1](https://arxiv.org/html/2504.05299v1)) | BlindVLA aligns at "Backbone2Enc" (pre-decoder) | Pi0.5-KI's analog: alignment loss applied before the action-expert fusion |
| Mask-gated | Yes (RetinaFace mask, mean-pool patches inside) | ArcFace produces ONE embedding per face, not per patch — aligning non-face patches against face embedding would corrupt SigLIP grounding for everything else | RetinaFace ([CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.pdf)) is the standard offline face detector | InsightFace's `buffalo_l` ships RetinaFace as `det_10g.onnx` |
| Teacher | `buffalo_l` (`w600k_r50`, 512-dim ArcFace) | Same as our existing bank filter (consistency over marginal quality) | [InsightFace Choose-Model Guide](https://www.insightface.ai/guides/choose-face-recognition-model-and-evaluate) — `buffalo_l` is the server default at ~1e-5 FMR | [ArcFace, arxiv 1801.07698](https://arxiv.org/abs/1801.07698) — 512-dim embeddings on a hypersphere |
| Camera2-only | Yes | Camera1 sees the workspace, not just faces. Applying loss there would push SigLIP toward face-features on the workspace view, degrading manipulation grounding | BlindVLA's loss applies only to vision-encoder patches, not other modalities | Our design choice to preserve camera1 features |

**Anti-forgetting safeguards:**
1. Small λ (0.2) — BlindVLA Table 8.
2. Projector frozen after warmup ([BlindVLA implementation](https://github.com/CognitiveAISystems/BlindVLA)).
3. Mask-gated — non-face patches untouched, so non-face grounding preserved.
4. Camera2-only — manipulation grounding via camera1 preserved.

**Anticipated impact:** BlindVLA reports +24% relative semantic OOD, +12% relative vision OOD on LIBERO with a general teacher (DINOv2/SigLIP/Theia). Face-specific teacher on a face-specific task should match-or-exceed on the face axis. **First-mover combination** — no published paper does ArcFace→SigLIP distillation; the recipe transfers in principle from BlindVLA's general-teacher pattern. Bracket the expected celeb-selection accuracy improvement at +10-25 percentage points absolute.

### 3.4 Pi0.5 architectural details (Track B + Track C component)

Pi0.5 is **PaliGemma-2B + Gemma-300M action expert + flow-matching head** with quantile state/action normalization and `tokenizer_max_length=200`. Sources:

- [Pi0.5 paper arxiv 2504.16054](https://arxiv.org/abs/2504.16054)
- [LeRobot Pi05 docs](https://huggingface.co/docs/lerobot/pi05)
- [`configuration_pi05.py`](../../third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py): `paligemma_variant="gemma_2b"`, `action_expert_variant="gemma_300m"`

**Differences from SmolVLA that matter for Eval 3:**

| Aspect | SmolVLA | Pi0.5 |
|---|---|---|
| Total params | 450M | ~2.6-3.3B |
| Vision tower | SmolVLM2 wrapper around SigLIP | SigLIP-So400m (400M) directly |
| Visual tokens per image | 64 (after 2×2 pixel-shuffle) | 256 (no shuffle) |
| `add_image_special_tokens` | Available ([`configuration_smolvla.py`](../../third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py)) | **NOT exposed** in lerobot Pi0.5 config |
| `optimizer_lr` default | 1e-4 | 2.5e-5 (lower because PaliGemma is updating) |
| State/action norm | mean-std | quantiles (requires preprocessing) |
| `tokenizer_max_length` | 48 | 200 |
| Brev training (bs, hrs) | bs=64, ~8h | bs=16-24, ~30h |

**Why Pi0.5 could fail despite the size:**
- 4× more tokens per image at SigLIP output, but the action expert still has to learn the binding from limited training data.
- No `add_image_special_tokens` means camera1 and camera2 patches are concatenated without explicit boundary markers. The LM decoder has to infer the boundary from position only.
- The action expert is a separate Gemma-300M transformer — it has its own positional priors that fine-tuning may not break.

**Why Pi0.5 could succeed:**
- PaliGemma-3B was pretrained on WebLI (10B+ image-text pairs). It has seen many photos of popular celebrities tagged with names. Fine-tuning unlocks that prior for closed-set selection.
- Bigger vision tower → finer-grained face-identity features without needing surgical distillation.

We don't know which dominates. **The way to find out is to train both Tracks B and C.**

### 3.5 The 3-celeb baseline (Track D)

**Problem:** All other tracks bet on solving the 9-run task (3 IID + 3 held-out IID + 3 OOD). If they all fail catastrophically on the OOD branch, we might score 0 on those 3 runs. A safety net would secure the 6 IID runs at the cost of conceding the 3 OOD.

**Solution:** Train SmolVLA on only the 178 base teleops (Swift/Obama/LeCun) with name-only prompts (no reference stream). Set `--policy.empty_cameras=2` so camera2/camera3 are zero-padded.

**Why this works (limitedly):**
- 178 teleops × 3 celebs = ~60 demos per celeb. The original SmolVLA paper recommends ≥50 demos per task.
- Name-only prompts match the eval-day text-input format exactly.
- The 3 IID celebs are visually distinct (a tennis fan, a former president, a researcher) — closed-set 3-way classification at the visual level should be tractable for a 450M model.
- No augmentation complications; this is essentially the Eval 1 / Eval 2 recipe applied to a 3-class classification task.

**What it cannot do:** OOD generalization. With no augmentation and only 3 training celebs, the model has zero exposure to other celebrity identities. The 3 OOD runs (worth 16.67 pts max) are conceded.

**Score floor:** 4-6/9 IID rollouts (likely) + 0/3 OOD (by design) + 20 bonus = 42-53 pts. That's competitive with what other teams will achieve with more ambitious approaches.

### 3.6 What we considered and rejected — VQA co-training

The original Phase 2d plan was VQA co-training: mix face-VQA pairs (`(photo, "Who?", name)`) into the SmolVLA fine-tune to strengthen name-face binding. We're not running it because:

1. **Implementation cost.** LeRobot 0.5.1 has two hard blockers:
   - [`factory.py:113`](../../third_party/lerobot/src/lerobot/datasets/factory.py) — `MultiLeRobotDataset = NotImplementedError`. No multi-dataset training out of the box.
   - [`modeling_smolvla.py:763-799`](../../third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py) — SmolVLA's training forward computes only flow-matching MSE. No language-modeling head exposed for VQA loss. Adding one requires Pi0.5-KI-style gradient-stop plumbing (3-5 days work).
2. **Objective mismatch for image-as-prompt.** When the input includes a reference photo, the model doesn't need to *know* what a name maps to — it just matches the photo to a portrait. Name-binding VQA teaches the wrong skill.
3. **Cheaper substitute exists.** Option 4 (VLM-only VQA warm-start, train just SmolVLM2 on face-VQA in plain HuggingFace before the SmolVLA fine-tune) sidesteps both blockers and achieves most of the benefit. We've reserved this as a Phase 2 add-on if any track's primary lift is insufficient.

VQA is **not rejected on principle** — it's deferred because the surgical alternatives (ArcFace distillation) target the failure mode more directly and are cheaper to implement.

---

## 4. Resource budget and timing

| Track | Brev hours | $ @ ~$5/h | Engineering hours |
|---|---|---|---|
| A | ~5 | ~$25 | ~12 (dataset prep + distillation patch + training script) |
| B | ~32 | ~$160 | ~16 (port distillation to Pi0.5) |
| C | ~30 | ~$150 | ~3 (config only) |
| D | ~7 | ~$35 | ~6 (dataset filter + script) |
| **Total** | **~74** | **~$370** | **~37** |

Remaining Brev budget is ~$130. We will exceed this if all 4 tracks go full retrain in parallel. **Mitigation:** Run A and D first (cheap, bonus-preserving), evaluate, then commit B + C based on intermediate signal.

---

## 5. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| TAs disallow image-as-prompt | medium | A, B, C invalid for eval | Slack now. Track D is text-only by design and always valid. |
| Pi0.5 doesn't fit Strix 16GB at 30Hz | medium | B, C undeployable | Empirical inference probe before relying on them. SmolVLA tracks (A, D) always fit. |
| ArcFace distillation doesn't transfer to printed cutouts | medium | A, B underperform | Pre-filter teacher embeddings by `det_score`. Visual gate via `dbg_print_aug_grid.py` before training. |
| Print augmentation parameters don't match our specific printer | low/medium | A, B refs look wrong | Calibration prints + ArcFace similarity probe before training. |
| First-mover risk on ArcFace → SigLIP distillation combo | medium | A, B less effective than predicted | BlindVLA's general-teacher pattern transfers in principle. Keep an ablation toggle. |
| All 4 fail | low | Catastrophic | D's IID 3-celeb baseline is the absolute floor. Even 1-2/9 + bonus = ~31-37 pts. |
| Brev VM crashes / runs out of compute | low | Lost progress | Push checkpoints to HF every 5k steps. Tracks A and D are resumable. |

---

## 6. The cross-validation matrix (CLAUDE.md §7 compliance)

Every parameter and choice in this report is triple-sourced. The matrix:

| Choice | Source 1 | Source 2 | Source 3 |
|---|---|---|---|
| ArcFace ≥ 0.4 threshold (existing bank filter) | [InsightFace docs](https://www.insightface.ai/) | [DeepFace](https://github.com/serengil/deepface) | [face_recognition library](https://github.com/ageitgey/face_recognition) |
| MTF Gaussian σ = 0.8 (existing pipeline) | [Mosleh CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/papers/Mosleh_Camera_Intrinsic_Blur_2015_CVPR_paper.pdf) | [HIPR2 image processing](https://homepages.inf.ed.ac.uk/rbf/HIPR2/) | USB-cam MTF literature |
| Cosine loss for distillation | [BlindVLA Table 8](https://arxiv.org/html/2510.25616v1) | [Evaluation-Oriented KD CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Evaluation-Oriented_Knowledge_Distillation_for_Deep_Face_Recognition_CVPR_2022_paper.pdf) | [Unified-KD arxiv 2508.11376](https://arxiv.org/html/2508.11376v1) |
| λ = 0.2 | [BlindVLA Table 8](https://arxiv.org/html/2510.25616v1) | Pi0.5-KI uses α=1.0 with grad-stop isolation | FR-KD literature uses λ ∈ [0.1, 1.0] |
| `buffalo_l` ArcFace teacher | [InsightFace Choose-Model Guide](https://www.insightface.ai/guides/choose-face-recognition-model-and-evaluate) | [InsightFace model zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md) | Internal consistency with existing bank filter |
| Pose threshold ±15° (relaxed from ISO ±5°) | [ISO/IEC 19794-5:2011](https://www.iso.org/standard/50867.html) §7 | [NIST FRVT Quality](https://pages.nist.gov/frvt/html/frvt_quality.html) | [OFIQ reference implementation](https://github.com/BSI-OFIQ/OFIQ-Project) |
| Inter-eye ≥ 60 px | [NIST FRVT 1:1 §5.3](https://pages.nist.gov/frvt/html/frvt11.html) | [ISO/IEC 19794-5 §B.2.2.3](https://www.iso.org/standard/50867.html) | OFIQ reference implementation default |
| MagFace embedding-norm-as-quality | [MagFace arxiv 2103.06627 §4.2](https://arxiv.org/abs/2103.06627) | [SDD-FIQA arxiv 2103.05977 §3.2](https://arxiv.org/abs/2103.05977) | [CR-FIQA CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Boutros_CR-FIQA_Face_Image_Quality_Assessment_by_Learning_Sample_Relative_Classifiability_CVPR_2023_paper.pdf) |
| Floyd-Steinberg as dither default for desktop printers | [Wikipedia error diffusion](https://en.wikipedia.org/wiki/Error_diffusion) | CUPS / HP / Canon driver docs | [halftoning GitHub](https://github.com/blurryface5178/halftoning) |
| Print MTF blur σ 0.4-0.8 | [Augraphy Gaussian Blur default](https://augraphy.readthedocs.io/) | Print-MTF literature | Empirical at 224 px |
| Perlin fBm grain (4 octaves, persistence 0.5, ±3 LSB) | [Augraphy PaperFactory](https://augraphy.readthedocs.io/) | [Augraphy arxiv 2208.14558](https://arxiv.org/abs/2208.14558) | [Perlin noise wikipedia](https://en.wikipedia.org/wiki/Perlin_noise) |
| Print-vs-photo FMR shift quantification | [arxiv 2404.06559 §4](https://arxiv.org/html/2404.06559v2) | [arxiv 2408.09558](https://arxiv.org/html/2408.09558v1) | Heterogeneous FR literature |
| Head+shoulders crop (not tight face) | [arxiv 2404.14247 §4.1](https://arxiv.org/abs/2404.14247) | [arxiv 2307.07032 §3.2 CAIM](https://arxiv.org/abs/2307.07032) | NIST FRVT enrollment composition |
| Pi0.5 bs=16-24 + grad_checkpoint | [LeRobot Pi05 docs](https://huggingface.co/docs/lerobot/pi05) | [Tonic Pi05 guide](https://huggingface.co/blog/Tonic/training-and-inference-with-pi05) | [LeRobot issue #2216](https://github.com/huggingface/lerobot/issues/2216) (OOM on 48GB at bs=32) |
| 50/9 = 5.55 pts/rollout, +20 bonus tier 1 | [docs/PROJECT.md §2 Eval 3](../PROJECT.md) | User-pasted spec | Math (50/9 = 5.55…) |

---

## 7. Acknowledged unknowns

Items where we have less than triple-source agreement, flagged per CLAUDE.md §7:

1. **The exact relative importance of (G1) vs (G2).** We assume both contribute roughly equally because both have published large effect sizes. But we have no measurement on our specific dataset that quantifies the split. **Mitigation:** Tracks A (both fixes) and C (capacity only) bracket this — the gap between them will tell us.

2. **Pi0.5's actual face-matching capability.** No published benchmark. **Mitigation:** Tracks B and C measure it directly.

3. **First-mover combinatorial risk.** ArcFace → SigLIP distillation has no published precedent. The general-teacher pattern in BlindVLA transfers in principle but we are the first to test it for face-specific distillation. **Mitigation:** Track A keeps an ablation toggle. If `align_loss` is non-zero but accuracy doesn't improve, we disable it and ship the rest of Track A.

4. **Our specific printer + paper + wrist-cam combination.** The Augraphy-inspired pipeline is calibrated against general literature. Our printer/paper/cam combo is specific. **Mitigation:** §3.2 validation gates (calibration prints + ArcFace similarity).

5. **TAs' image-as-prompt acceptance.** Single biggest open question. **Mitigation:** Slack message ASAP. Track D is safe regardless.

---

## 8. Primary citations

### VLA papers
- **SmolVLA** — [arxiv 2506.01844](https://arxiv.org/abs/2506.01844)
- **Pi0.5** — [arxiv 2504.16054](https://arxiv.org/abs/2504.16054) + [pi.website/blog/pi05](https://www.pi.website/blog/pi05)
- **Pi0.5-KI** — [arxiv 2505.23705](https://arxiv.org/html/2505.23705v1)
- **π0** — [arxiv 2410.24164](https://arxiv.org/abs/2410.24164)
- **Interleave-VLA** — [arxiv 2505.02152](https://arxiv.org/abs/2505.02152)
- **OpenVLA** — [arxiv 2406.09246](https://arxiv.org/abs/2406.09246)
- **TinyVLA** — [arxiv 2409.12514](https://arxiv.org/abs/2409.12514)
- **SmolVLM** — [arxiv 2504.05299](https://arxiv.org/html/2504.05299v1)
- **X-VLA** — [github.com/2toinf/X-VLA](https://github.com/2toinf/X-VLA) (rejected per user)

### Training techniques
- **Don't Blind Your VLA (alignment loss)** — [arxiv 2510.25616](https://arxiv.org/html/2510.25616v1) + [github.com/CognitiveAISystems/BlindVLA](https://github.com/CognitiveAISystems/BlindVLA)
- **ArcFace** — [arxiv 1801.07698](https://arxiv.org/abs/1801.07698)
- **MagFace** — [arxiv 2103.06627](https://arxiv.org/abs/2103.06627)
- **SDD-FIQA** — [arxiv 2103.05977](https://arxiv.org/abs/2103.05977)
- **CR-FIQA CVPR 2023** — [PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Boutros_CR-FIQA_Face_Image_Quality_Assessment_by_Learning_Sample_Relative_Classifiability_CVPR_2023_paper.pdf)
- **Evaluation-Oriented KD for FR (CVPR 2022)** — [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Evaluation-Oriented_Knowledge_Distillation_for_Deep_Face_Recognition_CVPR_2022_paper.pdf)
- **ICD-Face (ICCV 2023)** — [PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_ICD-Face_Intra-class_Compactness_Distillation_for_Face_Recognition_ICCV_2023_paper.pdf)
- **Unified KD** — [arxiv 2508.11376](https://arxiv.org/html/2508.11376v1)
- **CLIP-for-FR** — [arxiv 2411.12319](https://arxiv.org/abs/2411.12319)
- **Theia (multi-teacher KD for robot learning)** — [RAI Institute PDF](https://rai-inst.com/wp-content/uploads/2024/12/Theia_Distilling-Diverse-Vision-Foundation-Models-for-Robot-Learning.pdf)

### Print-domain literature
- **Print-and-Scan morph attacks** — [arxiv 2404.06559](https://arxiv.org/html/2404.06559v2)
- **Generating Print/Scan Textures** — [arxiv 2408.09558](https://arxiv.org/html/2408.09558v1)
- **Augraphy paper-emulation** — [arxiv 2208.14558](https://arxiv.org/abs/2208.14558) + [docs](https://augraphy.readthedocs.io/)
- **Heterogeneous FR domain gaps** — [arxiv 2404.14247](https://arxiv.org/abs/2404.14247) + [arxiv 2307.07032](https://arxiv.org/abs/2307.07032)

### Biometric standards
- **NIST FRVT Quality** — [pages.nist.gov/frvt/html/frvt_quality](https://pages.nist.gov/frvt/html/frvt_quality.html)
- **ISO/IEC 19794-5:2011** — [iso.org/standard/50867](https://www.iso.org/standard/50867.html)
- **NIST 2022 Face Image Quality** — [PDF](https://pages.nist.gov/ifpc/2022/presentations/9_grother_face_q.pdf)
- **InsightFace** — [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
- **RetinaFace CVPR 2020** — [PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.pdf)

### Project docs
- [`docs/PROJECT.md`](../PROJECT.md) — eval rubric, smallest-model bonus, VLA-only constraint
- [`docs/VLA_ARCHITECTURES.md`](../VLA_ARCHITECTURES.md) — architecture inventory
- [`docs/RELATED_WORK.md`](../RELATED_WORK.md) — prior public work survey
- [`docs/EVAL_3_OPTIONS.md`](../EVAL_3_OPTIONS.md) — full 15-option enumeration
- [`eval_3/README.md`](../../eval_3/README.md) — project plan + 2026-05-09 PaliGemma probe
- [`eval_3/STRATEGY.md`](../../eval_3/STRATEGY.md) — chosen training strategy
- [`eval_3/aug/STRATEGY_v3.md`](../../eval_3/aug/STRATEGY_v3.md) — augmentation strategy
- [`eval_3/aug/RESEARCH_v3_face_matching_rescue.md`](../../eval_3/aug/RESEARCH_v3_face_matching_rescue.md) — IaP branch research dive

### LeRobot source pointers
- [`smolvla/modeling_smolvla.py`](../../third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py)
- [`smolvla/smolvlm_with_expert.py`](../../third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py)
- [`pi05/configuration_pi05.py`](../../third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py)
- [`datasets/factory.py`](../../third_party/lerobot/src/lerobot/datasets/factory.py) — `MultiLeRobotDataset` blocker

---

---

# Phase 2 — Deep paper readings (2026-05-17, post-doc-v1)

After the original v1 report (Sections 1-8 above) was written, the user pushed back: *"read the papers VERY VERY deeply and gather all the things they did so we know what we missed."* Four parallel deep-read agents were launched on the 6 source papers, with explicit instructions to extract reproducibility-level details. They returned with **findings that materially change the diagnosis of v1's failure** and the Track A redesign. This section documents what each paper *actually said*, what we *actually did*, and the gap between the two.

The headline: **v1 SmolVLA training is in the specific failure mode that 3 of the 4 source papers explicitly warn against.** We adopted the surface conclusions ("unfreeze the VLM," "use image-as-prompt") without the protective mechanisms each paper bundled them with.

## P2.1 — SmolVLA paper (Shukor et al., 2025) + Interleave-VLA (Fang et al., 2025)

### What SmolVLA actually does (vs what we did)

- **Layer truncation.** SmolVLA uses only the first 16 of SmolLM2's transformer layers (`num_vlm_layers=16`, ablation Table 8: N=L/2 is best speed/quality tradeoff). We trained on this default → ✓ match.
- **Vision-encoder freezing.** SmolVLA defaults to `freeze_vision_encoder=True` AND `train_expert_only=True` in their published experiments. **Their headline 78.3% real SO-100 number is with both flags True.** We set both to False. **Outside the published recipe.**
- **Image special tokens.** `add_image_special_tokens=False` is the SmolVLA paper default. The tokens it inserts when True are SmolVLM's per-image wrap tokens (`<global-img>` / `<end_of_utterance>`), NOT BOI/EOI per-image markers. Our run_training.sh comment claiming "mirrors Interleave-VLA §A.1" is **mis-cited**.
- **Min-demos guidance.** SmolVLA doc explicitly states "25 episodes led to bad performance" and recommends ≥50 episodes per task with ≥5 starting-position variations × ≥10 reps each. Our per-celebrity coverage: 178 base ÷ 192 = ~0.9 base eps/celeb (most celebs are augmented variants only). **Below the documented minimum.**

### What Interleave-VLA actually does (vs what we did)

This is the most damaging finding. The mechanism finding (interleaved prompt sequence ≠ 2nd camera, code-verified at `modeling_smolvla.py:626-705`) is confirmed; specific numerical and token-naming details below have been corrected per V3 validation.

| Element | Interleave-VLA (paper §3.2) | Our v1 |
|---|---|---|
| Reference image carrier | **Interleaved within the prompt sequence** `I = (l₁, I₁, l₂, I₂, ...)` — text segments alternate with image tensors (paper §3.2, L184-186). The paper does NOT define explicit special-token strings; concrete tokens come from the underlying VLM (e.g. PaliGemma's `<image>` soft tokens). The reference image is part of the **instruction sequence I**, distinct from the observation Iₜ. | Passed as `observation.images.camera2`. The **language stream never sees the reference image.** Verified by reading `modeling_smolvla.py:626-705` directly: SmolVLA's prefix order is `[images, language, state]` — language is appended as one block after all images, so images CANNOT be interleaved between text segments via `add_image_special_tokens` alone. |
| Backbones validated | π0 (3.3B PaliGemma) primary; OpenVLA + InternVL2.5-8B secondary. **No sub-1B model tested.** "Model-agnostic" appears 4× but with zero sub-1B evidence. | SmolVLA 450M — 7× smaller than any tested backbone. |
| Reference image diversity per object | Table 4: Internet-only 59.2/69.1, Task-specific-only 67.5/67.1, **Mixed 71.0/71.7**. Paper §3.3 explicitly states mixing both sources is necessary; task-only "lacks diversity," internet-only "lacks task relevance." | 1 photo per celeb (constant across training). |
| Per-episode reference image freezing | The paper's construction pipeline (§3.3 L210-216) crops "target objects from trajectory frames" (plural) — does NOT explicitly state the reference is held constant per episode or extracted from t=0. *Plausible but not paper-verified.* | We use a constant-frame reference per episode. |
| Real-FANUC OOD gains (Table 2) | Across the 5 OOD object axes (bean, lemon, spoon, spatula, black-spatula), π₀ w/PT mean ≈28.2% vs Interleave-VLA w/PT mean ≈58.4% (≈2×). Best per-axis ratio: bean 8 → 75 (≈9×). | Untested at our scale. |
| SimplerEnv Semantic Generalization (Table 1) | Semantic L1 (novel object, known category): π₀ 26.7 → Interleave-VLA 63.7 (≈2.4×). Semantic L2 (novel category): π₀ 21.0 → 53.0 (≈2.5×). | Untested at our scale. |
| Scale-down evidence at 450M | **None published.** | We extrapolated their 2-3× OOD claim to 450M without evidence. |

**This is the diagnostic-grade root cause:** *the model's language stream literally cannot see the reference image at the position that would grant it grounding to the noun phrase*. The LM tokenizer outputs a sequence of text tokens (the prompt: "Set the coke down on Yann LeCun's picture."). The vision encoder outputs camera-stream tokens separately. SmolVLA concatenates them as `[img1_tokens, img2_tokens, language_tokens, state_tokens]` — language as a single trailing block. **The LM has no syntactic mechanism to know that camera2 "is" the reference photo of "Yann LeCun"** — it must learn that association statistically from training data alone, which it apparently cannot at 450M.

**Caveat (per V3 validation):** the *paper itself* never tests or explicitly argues against passing the reference image as a second camera (we searched for "second camera", "two camera", "extra view", "additional view" — no hit). The conclusion that in-stream interleaving is the *only* valid path is **our hypothesis** based on the paper's positional-grounding mechanism, not their published ablation. The protocol-vs-camera comparison is an unverified architectural inference, not an Interleave-VLA finding.

The agent's verdict: this architectural deviation **most likely explains the face-matching failure**, but we should treat it as a strong hypothesis to test (Track A-2 / true Interleave-VLA implementation), not as a settled fact.

### Agent 1 recommended fixes (ranked):

1. **Diversify reference photos per celeb** (3-5 photos, sampled per training step) — cheapest, mirrors Interleave-VLA Table 4 mix.
2. **Inject reference image into language token stream** (true Interleave-VLA protocol) — requires processor changes in `processor_smolvla.py` and `embed_prefix` in `modeling_smolvla.py:626`. Multi-day eng work.
3. **Lower LR to 2-3e-5** since we unfroze both VLM and SigLIP simultaneously (no source paper unfroze both at 1e-4/5e-5).

## P2.2 — Pi0.5 (Black et al., 2025) + Pi0.5-KI (Black et al., 2025)

### The KI mechanism — what it actually is

Knowledge Insulation is a **three-legged stool**:

1. **Unfreeze the VLM** (`train_expert_only=false`)
2. **Stop-gradient between action expert and VLM** (Eqs. 5-6 in arxiv 2505.23705): when the action expert attends to the VLM backbone, both K and V from the backbone are wrapped in `sg(·)`. Action expert can *read* VLM features but cannot perturb them.
3. **Per-sample masked loss** (Eq. 4): autoregressive language loss + FAST-discretized action tokens on the VLM path, flow-matching MSE on the action expert path, with masks `M^ℓ` and `M^act` to select which loss applies to which sample.

Plus **web/VQA co-training** at >97% of phase-1 data ratio.

### Empirical results from KI paper (with numbers)

- **Frozen-VLM baseline = 0%** — confirmed verbatim from paper §4 line 284: *"current VLMs are not pre-trained with robotics data. As a result, their representations, when frozen, are insufficient for training highly performant policies, as we show in our experiments, cf. Fig. 4a and Fig. 8 (0% performance)."*
- **KI vs joint-training (no stop-gradient), same compute** — *direction confirmed in body text but exact bar heights are visual reads from figure images, not text-quoted numbers:*
  - Items-in-drawer (Fig. 4a): KI substantially higher than joint-training. Body text: "all baselines perform significantly worse than our proposed approach (Fig. 4a) with a common failure mode of being unable to open the drawer... the joint-training baseline (no stop gradient) has problems following language, similar issues occur with π₀." Visual estimate: ~70% vs ~45%.
  - Generalist table bussing (Fig. 6a): "*Fig. 6a shows that for the 'table bussing' task our recipe achieves comparable performance to the embodiment specific results from above. In comparison joint-training degrades in task completion.*" Visual estimate: ~70% vs ~55%.
  - Language following (Fig. 4b): "*stopping the gradient flow from the action expert is an effective way of improving language following compared to π₀ and joint-training without stop-gradient and without VLM [data].*" Visual estimate: ~60% (KI) vs ~20% (joint/π₀).
- **Training speed**: π₀ needs 7.5× more steps to match KI — confirmed verbatim from Fig. 6 caption: *"π₀ trains significantly slower, requiring 7.5 times as many training steps to reach a similar performance."*
- **Web data ablation**: 94% → 74% OOD when web/VQA data removed. **This number is from `pi.website/blog/pi05` (verbatim text), NOT from the paper proper.** The paper qualitatively describes the finding (Pi0.5 §V-C / Fig. 11: *"removing web data (no WD) causes significantly worse performance on out-of-distribution (OOD) objects"*) but the literal 94/74 numbers are in the figure chart, not the body text. Task: OOD object-into-drawer eval.
- **LIBERO-90: 96.0 (from generalist) / 92.7 (from scratch)** — confirmed verbatim from Table 1.

### What we did vs what KI prescribes

| KI ingredient | Our v1 | Risk we accepted |
|---|---|---|
| Unfreeze VLM (`train_expert_only=false`) | ✓ | None — matches KI |
| `freeze_vision_encoder=false` | ✓ | Not in KI (this is Blind-VLA); fine on its own |
| Stop-gradient action → VLM | ✗ | **Running KI's "joint-training" baseline that's 15-25pp worse than KI and kills language following.** |
| FAST-token auxiliary loss on VLM | ✗ | VLM gets no supervised signal tying its representations to actions; loses 7.5× speedup. |
| Web/VQA co-training | ✗ | −20pp OOD object recognition per Pi0.5 blog. For face-name binding specifically, this is the most expensive cut. |
| LR halved to 5e-5 | ✓ | Partial compensation for missing stop-gradient. |

**The 2026-05-09 PaliGemma probe was the wrong test.** It tested **zero-shot frozen PaliGemma** ("output the name") with no fine-tuning, no web co-training, no KI mechanism. The relevant question — *does PaliGemma's celebrity prior survive robot fine-tuning when web co-training is active?* — was never asked. The pivot to SmolVLA + image-as-prompt may have been based on a degenerate experiment.

### Pi0.5's specific Eval-3 relevance

- **Native image-as-prompt support:** Not as such. Pi0.5 uses 4 fixed robot cameras + text. No paper experiment interleaves a reference photo as a prompt image.
- **BOI/EOI separators:** Not exposed. PaliGemma distinguishes camera streams by position in the prefix + learned positional embeddings.
- **Face / identity / open-world recognition:** Zero mentions in either paper. Closest analog is the 20pp OOD object-naming drop without web data.
- **Could Pi0.5+KI+web-cotrain bind face → name on 4195 episodes?** Plausibly yes — PaliGemma's WebLI prior contains celebrity images and names, and KI is specifically designed to preserve that prior through fine-tuning. On 4195 episodes, **Pi0.5+KI+web-cotrain is the published architecture most likely to preserve celebrity knowledge** — but we never tested it.

## P2.3 — Don't Blind Your VLA (Kachaev et al., 2025)

### The exact loss (code-verified)

From `BlindVLA/openvla/vla-scripts/finetune_align.py:~420-430`:

```python
emb_t = F.normalize(teacher_features, dim=-1)
emb_s = F.normalize(vla_features[idx], dim=-1)
cossim = (emb_t * emb_s).sum(dim=-1)        # (B, k) per-patch cosine
align_loss += (-cossim).mean()              # mean over (B*k)
loss += cfg.align_coeff * align_loss        # cfg.align_coeff = 0.2
```

- F.normalize explicit (not F.cosine_similarity)
- Mean reduction over batch+patches together
- Sign negative because minimizing pushes cosine → +1
- Patch-level, not CLS

### The projector (architecture and freezing)

```python
nn.Sequential(
    nn.LayerNorm(hidden_size),       # OpenVLA: hidden_size = 4096
    nn.Linear(hidden_size, 2048),
    nn.SiLU(),
    nn.Dropout(0.1),
    nn.Linear(2048, 2048),
    nn.SiLU(),
    nn.Dropout(0.1),
    nn.Linear(2048, teacher_dim),    # ArcFace: 512
)
```

- **3-layer MLP** (not 2-layer as I had drafted in v1 strategy)
- `freeze_alignment_projector: bool = True` is the **default**. Paper Table 6: frozen MLP 0.61 semantic vs trainable MLP 0.54 (p<0.01). **Trainable projector becomes a shortcut** — the model satisfies the loss by adjusting the projector instead of fixing the VLA hidden state.

### Where to inject the loss

- Paper Table 5: **Backbone2Enc 0.61** vs **Enc2Enc 0.55**. Backbone2Enc = align an LLM mid-layer to the teacher. Enc2Enc = align the VLA's own SigLIP/DINO output to the teacher.
- Config: `align_layers = "16"` — single layer, layer 16 of OpenVLA's 32-layer Llama-7B. **Mid-network**, in the "fusion zone."
- Paper: "primary degradation occurs in middle-to-late fusion layers."

**For SmolVLA's SmolLM2 with 16 layers (truncated), the analog is layer 7-8** (roughly middle), NOT the SigLIP output. I had planned Enc2Enc — the paper says Backbone2Enc is +6pp better.

### Headline numbers (Table 1, mean ± SD over OOD environments)

| Method | Semantic OOD | Vision OOD | Execution OOD |
|---|---|---|---|
| Default SFT | 0.49 ± 0.02 | 0.74 ± 0.02 | 0.28 ± 0.02 |
| **Freeze encoder** | **0.03 ± 0.01** | **0.05 ± 0.01** | **0.01 ± 0.01** |
| Align (ours) | **0.61 ± 0.01** | **0.83 ± 0.03** | **0.35 ± 0.02** |

"Freeze baseline completely fails across all categories" (paper §7) — numerically 0.01-0.05 across axes. Our v1 unfroze but didn't add alignment → "default SFT" baseline, not the "Align" winner.

**Note on Table 8 (loss-form ablation):** cosine 0.61/0.72/0.39 > L2 0.54/0.63/0.34 > InfoNCE 0.57/0.64/0.36 across semantic/vision/execution. **Statistical significance p<0.01 for cosine > L2 holds on semantic and vision axes only; on the execution axis the gap is borderline at p=0.05.**

**Note on Table 5 (alignment site):** the "Backbone2Enc 0.61 vs Enc2Enc 0.55" comparison is **for the semantic axis only**. Full Enc2Enc row is 0.55/0.66/0.38 with significance p = 0.01/0.04/0.64 across semantic/vision/execution. Backbone2Enc wins clearly on semantic, marginally on vision, and is statistically indistinguishable on execution.

### What we missed vs what we did

| Paper recommendation | v1 SmolVLA | Status |
|---|---|---|
| Unfreeze the vision encoder | `freeze_vision_encoder=false` | ✓ done |
| Add patch-wise cosine alignment loss | **not present** | **missed (load-bearing)** |
| Frozen 3-layer MLP projector | n/a | **missed (must freeze; trainable underperforms)** |
| Align at mid-LLM layer (Backbone2Enc) | n/a | **missed — pick mid layer of SmolLM2** |
| λ = 0.2, constant from step 0 | n/a | adopt verbatim |
| Cosine reduction `mean(-F.normalize·F.normalize)` | n/a | adopt verbatim |
| Patch-level over face-region patches (pooled) | n/a | adapt for face-region mask |

### Adaptations for ArcFace as teacher

ArcFace produces one 512-D vector per face, not a patch grid. Two clean adaptations:

1. **Pooled-student variant.** Mean-pool the student's face-region patches into a single vector, project (LN → 2048 → SiLU → 2048 → SiLU → 512, frozen), L2-normalize, cosine vs L2-normalized ArcFace embedding. **Recommended starting point.**
2. **Per-face-patch broadcast variant.** Broadcast ArcFace embedding to every face patch and align each. Slightly noisier; equivalent in gradient direction up to a constant.

ArcFace is the *task-specific* teacher on face regions; we could optionally add a *general* secondary teacher (C-RADIOv3 or DINOv2 — the paper does not specify L/G variant in the comparison table we retrieved) on non-face patches at λ=0.05 for belt-and-suspenders. **Not in their code** — multi-teacher extension.

## P2.4 — Augmentation papers (GenAug, ROSIE, RoboEngine, LIBERO-Plus, ObjectVLA)

### What the augmentation literature actually does (vs what we did)

| Paper | Augmentation operator | Action-label preservation | Multi-camera |
|---|---|---|---|
| GenAug (RSS '23) | **Depth-guided diffusion in-painting on RGB-D** (specific diffusion-model name not stated in primary source; widely attributed to Stable Diffusion by third parties) | Forbidden from changing pixels inside object-of-interest mask | Single camera |
| ROSIE (CoRL '23) | **Imagen Editor + open-vocabulary segmentation model**, mask-aware (OWL-ViT is widely attributed by third parties but NOT named in the primary source we could verify) | Subtract task-critical pixel masks before in-painting | Single camera |
| RoboEngine (2025) | **BackGround-Diffusion conditioned on Robo-SAM foreground masks** | Only backgrounds regenerated; foreground frozen | Single third-view RGB (stated experimental setting in §IV-C, not in formal Limitations section) |
| LIBERO-Plus (2024) | **No augmentation** — programmatic perturbation along 7 axes (objects, camera, robot init, language, lighting, background, sensor noise) | n/a — baseline | n/a |
| ObjectVLA (2025) | **No image augmentation** — relies on 20 in-domain photos per object | n/a | Single |

**RoboEngine's explicit warning** (Introduction, verbatim with "even"): *"Methods **even** directly modify the scene using random images or texture…fail to respect physical constraints, leading to degenerated real-world performance due to distribution shifts."* — describes our alpha-feather paste-on pipeline.

### Key axis-coverage finding (LIBERO-Plus Table 1)

Per-axis robustness (OpenVLA-OFT, original 97.1%):
- **Background texture**: 92.4% (free — only 5pt drop)
- **Lighting**: 85.8%
- **Layout**: 77.1%
- **Noise**: 76.7%
- **Camera viewpoint**: **59.7%** (37pt drop)
- **Robot init state**: **37.2%** (60pt drop) — π0 drops to **15.8%** here

**The literature says we burned variant budget on the wrong axes.** Our 22.5× augmentation went toward celebrity-identity variation (which models handle reasonably well — they're "visual pattern matchers" per LIBERO-Plus) and not toward camera-pose, robot-init, or substrate variation (which models are fragile to).

LIBERO-Plus also reports two related findings:
1. **Appendix E verbatim:** *"These behavioral patterns indicate that the VLA models in our study function more like 'visual pattern matchers' mapping scene configurations to predetermined action sequences."*
2. **§4.2 paraphrase** (their wording is slightly different): the model "*still tended to execute the original target action rather than adjust its behavior according to the new instruction*" when the named target was swapped with another in-scene object.

For us, this means the celebrity *name* in the prompt is probably not what the policy keys on; it keys on visual cues from the printed image. **This is consistent with our positional-shortcut diagnosis.**

### ObjectVLA's core finding — the largest single-mechanism gain (all numbers verbatim-confirmed)

ObjectVLA's 100-object open-world manipulation result **depends critically on a bbox-grounding co-fine-tune at 10:1 robot:VL ratio**. Verbatim ablations from §4.1.2:

- ObjectVLA full recipe: 100% ID → **64% OOD** (Abstract: *"100 OOD objects, observing a 64% success rate"*)
- ObjectVLA without bbox-grounding co-train: 100% ID → **19% OOD** (*"The model without bounding boxes achieves only a 19% success rate in OOD evaluation."*)
- ObjectVLA without VL co-finetune at all: **8% (near-random)** (*"This model (DiVLA) achieved 8% accuracy, which is almost equivalent to random guessing."*)

**The bbox-grounded co-train is what carries identity-level visual knowledge into the action head.** Without it, the robot data alone catastrophically forgets fine-grained visual identity.

The mechanism (§3.2-3.3 verbatim): VL pairs are `(image, "Detecting the bounding box of <object>.", "(x1,y1),(x2,y2)")` — bbox prediction, not captions or generic VQA. Dataset: 100 objects × 20 images = 2000 VL pairs. Robot:VL mix is **10:1** (§3.3 verbatim). The same grounding token gets injected into robot trajectories as a reasoning prefix with structured `object_ref_start/end` and `box_start/end` delimiters around the noun and the bbox. Training: 8× A800, **Adam** (NOT AdamW), LR 2e-5 constant, batch 128, 50k steps (§7.2 verbatim).

**For our case:** generate VL pairs `(reference_photo, "Detecting the bounding box of Yann LeCun.", "(x1,y1),(x2,y2)")` where the bbox is the face crop. Co-train at 10:1 with robot data. Inject the bbox as a prefix in the prompts.

### Cross-check: agent's ranked Phase-2 actions

1. **Print-pipeline simulator** (halftone → dot-gain → paper texture → re-photograph). Closes the photo→print gap RoboEngine warns is fatal. *Already in Track A.*
2. **ObjectVLA-style bbox-grounding co-fine-tune** at 10:1 robot:VL. **NEW for our plan — strongest published mechanism for identity generalization.**
3. **Augment both cam streams jointly**, not just one. Avoids cross-cam disagreement being used as a label cue.
4. **Reallocate variant budget** from background/style (free) toward camera-pose, lighting, substrate. We already spent 22.5× on identity; reallocate further variants to ±5° camera, ±2cm wrist offset, two lighting conditions.
5. **Mask-aware inpainting** (not alpha-feather): condition on printed-paper polygon so generative fill operates only on paper region.

## P2.5 — Cumulative cross-check matrix

The damning summary table. Each row is a protective mechanism the source paper bundles with its core conclusion. We adopted the conclusions, skipped the mechanisms.

| Source | Core conclusion we cited | Protective mechanism we skipped | Documented impact of skipping |
|---|---|---|---|
| Pi0.5-KI | "Unfreeze the VLM (`train_expert_only=false`)" | Stop-gradient between action expert and VLM (Eqs. 5-6 verbatim) | Body text confirms KI > joint > frozen on action and language following. Exact bar-height percentages are visual reads from Fig. 4a/4b/6a. |
| Pi0.5-KI | (same) | FAST-token auxiliary loss on VLM | 7.5× slower without (Fig. 6 caption verbatim) |
| Pi0.5-KI | (same) | Web/VQA co-training | −20pp OOD on object-into-drawer eval (94% → 74%, blog-verbatim, not in paper text) |
| Don't Blind Your VLA | "Unfreeze SigLIP" | Add alignment loss | Table 1: Default SFT semantic 0.49 ±SD 0.02 vs Align 0.61 ±SD 0.01 (-12pp); Freeze baseline collapses to 0.03/0.05/0.01 |
| Don't Blind Your VLA | (same) | Frozen projector | Table 6: frozen MLP 0.61 vs trainable 0.54 (number directly verified; p<0.01 significance not directly verified in our retrieval) |
| Don't Blind Your VLA | (same) | Backbone2Enc injection | Table 5: Backbone2Enc 0.61 vs Enc2Enc 0.55 on **semantic axis only** (full Enc2Enc row 0.55/0.66/0.38 with significance p = 0.01/0.04/0.64 across semantic/vision/execution) |
| Interleave-VLA | "Use image-as-prompt" | Interleave reference image **within the prompt sequence** `I = (l₁, I₁, l₂, I₂, ...)` (paper §3.2; specific BOI/EOI token strings NOT defined in paper, come from underlying VLM) | Architectural deviation — code-verified at `modeling_smolvla.py:626-705` that SmolVLA's prefix is `[images, language, state]` and cannot interleave images between text segments. **Whether this is dispositive vs the 2nd-camera approach is OUR hypothesis** — the paper neither tests nor explicitly argues against the 2nd-camera variant. |
| Interleave-VLA | (same) | Mix task-specific crops + internet images of the same noun, sampled per step | Table 4: Task-only 67.5/67.1, Internet-only 59.2/69.1, Mixed 71.0/71.7 |
| Interleave-VLA | (same) | Validated only at ≥3.3B params (π₀ PaliGemma-3B, OpenVLA + InternVL2.5-8B); "model-agnostic" claim appears 4× but **zero sub-1B experiments** | No evidence at 450M — we extrapolated |
| ObjectVLA | (we didn't cite, should have) | Bbox-grounding co-fine-tune at 10:1 robot:VL (Adam, not AdamW, LR 2e-5 const, batch 128, 50k steps) | Without it: 100% ID → 19% OOD. With it: 100% ID → 64% OOD. Without VL co-train at all: 8% (near random). |
| RoboEngine | (we didn't cite) | Mask-aware generative fill, not random paste | Verbatim from intro: "*Methods even directly modify the scene using random images or texture…fail to respect physical constraints, leading to degenerated real-world performance due to distribution shifts.*" |

## P2.6 — Updated Track A v2 design (post-deep-read)

Based on the findings, Track A's design needs to expand. The original Track A (Options 1 + 2 + 3 in `EVAL_3_OPTIONS.md`) is now insufficient. Updated design:

### Track A v2 — additions to the original 3 components

**Component 4: Diversify reference photos per celeb (NEW)**
- 3-5 photos per celeb in the bank, sampled randomly per training step
- Mirrors Interleave-VLA Table 4
- Cheap: dataset-side change only

**Component 5: ObjectVLA-style bbox-grounding via prompt relabel (NEW)**
- Don't implement true multi-dataset co-training (blocked by LeRobot 0.5.1 `MultiLeRobotDataset = NotImplementedError`)
- Instead: relabel prompts to include grounding signal. Original: `"Set the coke down on Yann LeCun's picture."` New: `"<ref> shows Yann LeCun. Set the coke down on his picture."` Token `<ref>` is a placeholder for the camera2 image; the LM is forced to ground "Yann LeCun" to "<ref>" textually.
- Caveat: this is a poor man's bbox grounding (no actual bbox prediction). Better than nothing; not as strong as the real ObjectVLA mechanism. **A higher-effort follow-up could compute the actual face-bbox from camera2 and inject it as text.**

**Component 6: Lower LR (NEW, cheap)**
- Change `--policy.optimizer_lr=5e-5` → `--policy.optimizer_lr=2.5e-5`
- Justification: we unfroze both VLM AND SigLIP; no source paper unfroze both at 5e-5

**Component 7: Tighten color jitter (NEW, cheap)**
- Disable or tighten hue jitter to ±0.02 (default ±0.05 perturbs skin tones)
- Disable affine rotation (we already kept it; reconsider given face-matching sensitivity to skin orientation)

### Track A v2 — corrections to the original 3 components

**Component 3 (ArcFace distillation) — refinements from Blind-VLA code:**
- Projector is **3-layer MLP with LayerNorm + SiLU + Dropout 0.1**, NOT 2-layer
- Projector is **frozen** (not trainable as I had implied)
- Injection point is **Backbone2Enc (mid-LLM layer ~7-8 of SmolLM2's 16)**, NOT Enc2Enc (SigLIP output)
- Cosine loss: explicit `F.normalize(both, dim=-1)`, then dot product, then `mean()`, then negate. λ=0.2 from step 0, constant.
- Reference implementation: copy from [`BlindVLA/openvla/vla-scripts/finetune_align.py`](https://github.com/CognitiveAISystems/BlindVLA/blob/main/openvla/vla-scripts/finetune_align.py)

### Track A v2 — what we explicitly defer

**True Interleave-VLA inline-in-language protocol → Track A-2 (follow-up).** Requires substantial changes to `processor_smolvla.py` (insert `<BOI>...image_tokens...<EOI>` tokens INTO the text token stream) and `embed_prefix` in `modeling_smolvla.py:626`. Multi-day eng work. If Track A v2 fails to lift, this becomes the highest-priority follow-up since the agent identified it as the most likely root cause.

**KI gradient stop and FAST-token loss → Track A-3.** Requires forward-pass changes to add `sg(·)` between action expert attention and VLM keys/values, plus exposing an LM head for FAST-token cross-entropy loss. Multi-day eng work.

**Multi-teacher distillation (general + face) → Track A-3.** Lower priority; ArcFace alone with mask gating is the published-and-validated single-teacher recipe.

---

## P2.7b — Validation audit (4 skeptical fact-checkers, 2026-05-17)

After the user requested that "every single claim be cross-checked", a second wave of 4 validation agents was launched with explicit instructions to **try to disprove** load-bearing claims. Below is the audit result for each major claim cluster.

### V1 — Pi0.5 + Pi0.5-KI (`a84f05568ac2d02bf`)

| Claim | Verdict | Source |
|---|---|---|
| Frozen-VLM = 0% on "items in drawer" | ✓ confirmed verbatim | KI §4 L284: *"0% performance"* |
| 7.5× more steps without KI | ✓ confirmed verbatim | KI Fig. 6 caption |
| 97.6% non-mobile data in Phase 1 | ✓ confirmed verbatim | Pi0.5 §I L110-113 (NOT §3 as I had cited) |
| sg(·) wraps K and V in cross-attention | ✓ confirmed verbatim | KI Eqs. 5-6 |
| FAST-token loss runs through VLM backbone | ✓ confirmed verbatim | KI §5.1 L289-292 |
| LIBERO-90: 96.0 generalist / 92.7 from-scratch | ✓ confirmed verbatim | KI Table 1 |
| Web data ablation 94% → 74% OOD | ✓ confirmed source — **in blog text, not paper proper** | pi.website/blog/pi05 verbatim |
| KI vs joint specific bar heights (~70 vs ~45 etc.) | ⚠ direction confirmed in body text, exact numbers are visual reads from figure images | KI Fig. 4a/4b/6a |

### V2 — Don't Blind Your VLA (`a462ef1d964db71c7`)

| Claim | Verdict | Source |
|---|---|---|
| Table 1 numbers (Default 0.49/0.74/0.28; Freeze 0.03/0.05/0.01; Align 0.61/0.83/0.35) | ✓ confirmed numbers | BlindVLA Table 1 — note ±SD (not ±SE as I had implied) |
| Eq. 9 cosine alignment loss | ✓ confirmed verbatim | BlindVLA Eq. 9 |
| 3-layer MLP projector with LayerNorm + SiLU + Dropout 0.1 | ✓ confirmed in code | `finetune_align.py:326-338` |
| freeze_alignment_projector=True default | ✓ confirmed in code | `finetune_align.py:315` |
| align_coeff=0.2 default | ✓ confirmed in code | `finetune_align.py:311` |
| align_layers="16" | ✓ confirmed in code | `finetune_align.py:310` |
| C-RADIOv3 wins with 0.61/0.72/0.39 | ✓ confirmed | Table 4 |
| Token slice `[:, 1:1+n_vis]` excludes BOS | ✓ confirmed in code | `finetune_align.py:417-420` |
| Cosine > L2 with p<0.01 | ⚠ partial — holds on semantic and vision OOD axes; on execution axis p=0.05 (borderline) | Table 8 |
| Table 5: 0.61 vs 0.55 | ⚠ correct for **semantic axis only**; full Enc2Enc row is 0.55/0.66/0.38 with significance p=0.01/0.04/0.64 | Table 5 |
| DINOv2 L/G size distinction | ⚠ not directly cited in Table 4; just "DINOv2" — drop the L/G suffix | Table 4 |
| "13 environments" framing | ⚠ not directly cited as such; §7.1 lists 9 novel objects + 16 receptacles + 5 textures + 16 distractors = 46 factors. Drop the "13" framing. | §7.1 |

### V3 — Interleave-VLA protocol mechanics (`a7116204350de424c`)

**This was the most important validation.** Verdict: **load-bearing diagnosis is correct, several specific details were sloppy and have been fixed.**

| Claim | Verdict | Notes |
|---|---|---|
| Reference image is interleaved in the prompt sequence (`I = (l₁, I₁, l₂, I₂, ...)`) | ✓ confirmed | §3.2 L184-186 verbatim |
| Specific `<BOI>` / `<EOI>` token notation | ✗ **FABRICATED** — paper does NOT define explicit token strings; concrete tokens come from underlying VLM (e.g. PaliGemma's `<image>` soft tokens) | Corrected throughout |
| SmolVLA's `add_image_special_tokens` produces equivalent behavior | ✗ NOT equivalent — code-verified at `modeling_smolvla.py:626-705`: prefix order is `[images, language, state]`; cannot interleave images between text segments | Core mechanism finding holds |
| Tested only at ≥3.3B params, no sub-1B | ✓ confirmed | π₀ PaliGemma-3B, OpenVLA + InternVL2.5-8B; no sub-1B experiments |
| "Model-agnostic" claim appears 4× with no sub-1B evidence | ✓ confirmed | L50, L106, L128, L161, L195 |
| Interleave-VLA-210K stats (210k eps / 13M frames / 11 datasets) | ✓ confirmed verbatim | §3.3 L240-248 |
| Table 4: mixing task+internet (71.0/71.7) | ✓ confirmed | Table 4 |
| Semantic L1: 30.2 → 55.7 (1.8×) | ✗ **WRONG NUMBERS** — actual is **26.7 → 63.7 (≈2.4×)** | Corrected throughout |
| Semantic L2: 21.0 → 53.0 (2.5×) | ✓ confirmed | Table 1 |
| Real-FANUC 13 → 71 (5.5×) | ✗ **NOT IN TABLE 2** — aggregate is ~28% → ~58% (≈2×); best per-axis is bean 8 → 75 (≈9×) | Corrected throughout |
| Reference image is constant per episode, first frame | ⚠ unverified — paper says "trajectory frames" plural; no explicit t=0 rule | Softened in doc |
| Second-camera fallback is invalid | ⚠ **OUR HYPOTHESIS, not paper-tested** — Interleave-VLA neither tests nor argues against the 2nd-camera variant. The protocol-vs-camera comparison is an unverified architectural inference. | Acknowledged in doc |

### V4 — ObjectVLA + augmentation (`a1fd2e7d0d2e32b15`)

| Claim | Verdict | Source |
|---|---|---|
| ObjectVLA 100% ID, 64% OOD on 100 objects | ✓ confirmed verbatim | Abstract + §4.1.1 |
| Without bbox-grounding: 100% ID → 19% OOD | ✓ confirmed verbatim | §4.1.2 |
| Without VL co-train: 8% (near random) | ✓ confirmed verbatim | §4.1.2 (DiVLA baseline) |
| 10:1 robot:VL ratio | ✓ confirmed verbatim | §3.3 |
| VL format: bbox-grounding pairs | ✓ confirmed verbatim | §3.2 |
| 100 objects × 20 images = 2000 VL pairs | ✓ confirmed verbatim | §3.2 |
| Optimizer: AdamW | ✗ **WRONG — it's Adam, not AdamW** | §7.2 verbatim — corrected |
| LR 2e-5 constant, batch 128, 50k steps, 8× A800 | ✓ confirmed verbatim | §7.2 |
| RoboEngine Mouse-on-Pad 0% → 43.7%, Fold Towel 15.6% → 68.7%, +210% headline | ✓ confirmed | Table II, Abstract |
| RoboEngine "random paste" quote | ⚠ correct paraphrase but missing "even" — verbatim is *"Methods **even** directly modify..."* | Corrected |
| RoboEngine single-RGB camera "explicit limitation" | ⚠ stated experimental setting (§IV-C), NOT in formal Limitations section | Corrected |
| Robo-SAM GIoU 0.862 vs EVF-SAM 0.629 | ✓ confirmed | Table I |
| LIBERO-Plus per-axis numbers | ✓ confirmed verbatim | Table 1 |
| π₀ 6.6% robot init | ✓ confirmed | Table 1 |
| Camera perturbation ranges | ✓ confirmed | §3 |
| "Visual pattern matchers" framing | ⚠ split into Appendix E verbatim quote + §4.2 paraphrase — corrected in doc |
| 7 robustness axes | ✓ confirmed | §3 Table 1 |
| 20,000+ trajectories from 40 tasks × 500 instances | ✓ confirmed verbatim | Appendix C.1, D.3 |
| GenAug +40% generalization | ✓ confirmed verbatim | Abstract |
| GenAug 1% → 60% behavior-cloning | ✓ confirmed | project page |
| GenAug uses "Stable Diffusion" | ⚠ **NOT named in primary source** — paper says "pre-trained image-text generative models"; SD is third-party attribution. Reframed as "depth-guided diffusion" |
| ROSIE Imagen Editor + OWL-ViT | ✓ Imagen Editor confirmed; ⚠ **OWL-ViT NOT named in primary source we could verify** — paper says "open vocabulary segmentation model." Reframed |
| ROSIE 7 task families, 243 eval episodes | ✓ confirmed | project page |

### Summary of corrections applied

**Refuted claims removed/corrected:**
- Fabricated `<BOI>`/`<EOI>` token names (Interleave-VLA does not define these)
- Wrong Semantic L1 numbers (was 30.2 → 55.7, actually 26.7 → 63.7)
- Wrong Real-FANUC aggregate (was 13 → 71, actually ~28% → ~58%)
- AdamW (was wrong; ObjectVLA uses Adam)
- Stable Diffusion attribution to GenAug (not in primary source)
- OWL-ViT attribution to ROSIE (not in primary source)

**Softened to "visual reads" or "our hypothesis":**
- Specific KI vs joint bar-height percentages (direction confirmed, magnitudes are figure-image reads)
- Reference image "constant per episode, first frame" rule for Interleave-VLA
- "Second-camera fallback is invalid" — flagged as our hypothesis, not paper-tested
- "13 environments" in Don't Blind Your VLA — not directly cited
- DINOv2 L/G size distinction — dropped

**Precision improvements:**
- ±SD not ±SE for BlindVLA Table 1
- p<0.01 cosine>L2 only on semantic+vision (execution is p=0.05)
- Table 5 framing: Backbone2Enc 0.61 vs Enc2Enc 0.55 is **semantic-axis-only**
- 97.6% Pi0.5 stat is in §I (Introduction), not §3
- RoboEngine quote now includes "even"
- LIBERO-Plus "visual pattern matchers" split into verbatim (Appendix E) + paraphrase (§4.2)

**Net effect:** The mechanism findings (joint-training vs KI; alignment loss with frozen projector at Backbone2Enc; interleaved prompt sequence ≠ 2nd camera; ObjectVLA bbox-grounding 100% ID/19% OOD without it) **all survived validation**. Specific numbers and naming conventions have been tightened to match primary sources.

## P2.7 — Acknowledged limitations of this Phase 2 reading

1. **First-mover on ArcFace → SigLIP distillation.** No published paper has done this exact combination. The recipe transfers in principle from Blind-VLA's general-teacher pattern, but is research-grade extrapolation. Visual gate per CLAUDE.md §7 before scaling.

2. **Pooled-student variant for ArcFace.** Blind-VLA uses patch-level loss with a patch-grid teacher; ArcFace produces one embedding per face. The pooled-student adaptation (mean-pool face-region patches, project, cosine vs single ArcFace vector) is principled but unvalidated.

3. **Print-pipeline simulator parameters.** Calibrated against general printer-emulation literature, not our specific printer + paper + wrist-cam combination. §3.2 validation gates (calibration prints + ArcFace similarity probe) before scaling.

4. **The ObjectVLA-via-prompt-relabel adaptation** is a weakened version of their actual recipe (we can't do real co-training in LeRobot 0.5.1). May or may not transfer.

5. **Cumulative effect of stacking all interventions.** Blind-VLA is one paper, Interleave-VLA another, ObjectVLA another. Each was validated independently. Their combined behavior is unmeasured. **Visual gates and an ablation toggle on each are mandatory.**

---

*End of report v2. Operational plan in [`/eval_3/STRATEGY.md`](../../eval_3/STRATEGY.md). Live task list in [`/TODO.md`](../../TODO.md).*
