# Research v3 - Rescuing SmolVLA-450M's face-matching failure (2026-05-17)

**Status:** active. Supersedes the v2 architecture decision in [`eval_3/README.md` §Architecture decision](../README.md) only with respect to the *post-v1-training rescue plan* - the underlying architecture pick (SmolVLA + image-as-prompt + inpaint augmentation) is unchanged.

**Authors:** Roham + the LeMonkey claude session, 7 parallel research agents (4 first-wave decision agents + 3 second-wave deep-dive agents), cross-checked per CLAUDE.md §7.

**Purpose.** SmolVLA-450M was trained for the Eval 3 image-as-prompt face-matching task (place coke on the celebrity whose photo is in `observation.images.camera2`). Smoke-tested on Strix on 2026-05-17: the pipeline works end-to-end but the policy puts the can on the *wrong* portrait. Four hypotheses were tested:

1. `camera2` silently dropped in `prepare_images()` → **refuted** by direct instrumentation (image[1] = reference photo confirmed reaching the model intact).
2. Wrist-cam aim shift from training distribution → **refuted** by manual workspace inspection.
3. Positional / layout bias dominates over face-matching → **confirmed** by rotating Swift's print across L/M/R positions (the can lands at the same physical spot regardless).
4. SmolVLA-450M's reasoning capacity is too small for image-as-prompt face-matching → **plausible**, but not proven.

The team's first reflex was "scale up - switch to Pi0.5 or OpenVLA-7B." This document presents the research case that **scaling is not the highest-leverage move** and that two surgical, SmolVLA-preserving interventions have stronger published evidence behind them, are cheaper to run, and keep the +20 smallest-model bonus.

---

## 1. The failure mode, precisely

Hypothesis 3 above is the load-bearing finding. Reframed: SmolVLA's action expert appears to have learned a **positional shortcut**: "for any prompt naming a celebrity from the training distribution, drop the can at the mean position of where celebrities like X usually were placed during teleop." The reference-photo stream (camera2) is fed to the model, but the model is *not actually conditioning on it for action selection*.

Why this is the right read:

- **Positional bias confirmed empirically** on the deploy machine (2026-05-17): rotating Swift through L/M/R positions produced the same wrong spot.
- **The visual sub-task is hard** ([`docs/VLA_ARCHITECTURES.md` §3 Eval 3](../../docs/VLA_ARCHITECTURES.md)): "*The bottleneck is celebrity world-knowledge AND visual face-grounding. The policy must compare faces across the two streams (often very different photo styles - a magazine portrait vs a printed cutout from the TOY PDF).*"
- **SmolVLM-500M's vision tower** uses a 64-token-per-image pixel-shuffle bottleneck ([SmolVLM, arxiv 2504.05299, §3.1](https://arxiv.org/html/2504.05299v1)). For an 224×224 image with a face occupying ~30% of pixels, only ~10–20 visual tokens carry face identity. That's a thin channel for identity-discriminative information.
- **No published VLA at the 450M scale demonstrates identity-discriminative reasoning across heterogeneous photos.** Interleave-VLA's 2× OOD gain ([arxiv 2505.02152](https://arxiv.org/abs/2505.02152)) is on π0 (3.3B), and even there the gain is from the *protocol* not the size - the visual reasoning still depends on the backbone's face-discriminative inductive bias.

The fix therefore has to address **two independent gaps**:

| Gap | What's missing | Intervention |
|---|---|---|
| (G1) Domain gap | Training-time reference photos are magazine/web style; eval-day reference is a printed A5 cutout the model never saw during training | Reference-photo curation + print-domain forward augmentation |
| (G2) Representation gap | SigLIP's 64-token-per-image features collapse identity into generic "face token"; face-discriminative geometry is not in the pretrained features | ArcFace cosine distillation into the SigLIP encoder, mask-gated to face patches |

These are **orthogonal**. They compound - fixing the domain gap without fixing the representation gap leaves the model with print-style features it still can't tell apart; fixing the representation gap without the domain gap gives the model strong face-id features that don't transfer to printed cutouts.

---

## 2. Why "just scale up" is not enough

### 2.1 Pi0.5 economic argument

The smallest-model bonus is +20 (rank 1, SmolVLA) vs +16 (rank 3 if 1 OpenVLA team and 1 smaller-than-Pi0.5 team exist) (per [`docs/PROJECT.md` §2 Eval 3](../../docs/PROJECT.md)). That **+4 differential = 0.72 rollouts** of slack on the 5.55-pt-per-rollout scale. Pi0.5 has to beat SmolVLA by **strictly more than 1 rollout** to come out ahead.

Bonus-economics agent verdict (synthesized from [`docs/PROJECT.md`](../../docs/PROJECT.md), [`eval_3/README.md`](../README.md), and 2026 VLA pretraining-budget surveys):

> SmolVLA can be exactly one rollout behind Pi0.5 and still win when its bonus advantage is +4. With only +2 (Pi0.5 is rank 2), the gap must be ≤ 0.36 rollouts → effectively must tie. The bonus is **never** worth more than one rollout's worth of failure.

### 2.2 Pi0.5's face-matching evidence is absent

The Pi0.5 feasibility agent found **no published evidence that Pi0.5 face-matches better than SmolVLA**. Specifically:

- Pi0.5's web-VQA co-training gain is **−20 pp OOD when removed** for general object generalization ([pi.website/blog/pi05](https://www.pi.website/blog/pi05)), but the evaluation domain is household tasks, not cross-image identity reasoning.
- Pi0.5 in LeRobot **does not expose `add_image_special_tokens`** (the BOI/EOI separators between camera streams that SmolVLA uses, [`configuration_pi05.py`](../../third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py)). For image-as-prompt this is a **regression** vs SmolVLA, not a win.
- The 2026-05-09 PaliGemma probe (logged in [`eval_3/README.md` §Architecture decision](../README.md)) found PaliGemma-3B scores **0/14 on naming TOY celebrities** and **0/6 on naming OOD celebrities** zero-shot - the original premise that justified Pi0.5 (web-pretrained celebrity recall) does not hold.

### 2.3 OpenVLA-7B is deploy-blocked

The alt-VLAs agent verified OpenVLA-7B requires **~14 GB bf16 weights + KV cache + ViT activations** for 30 Hz inference, which **does not fit** on the Strix RTX 3080 Ti Laptop's 16 GB ([model card](https://huggingface.co/openvla/openvla-7b), [LeRobot v0.5 release notes](https://huggingface.co/blog/lerobot-release-v050)). It also lacks native LeRobot integration - implementing a new policy class is a multi-day fork.

### 2.4 The X-VLA-0.9B insurance track is real

The one viable scale-up candidate the agent surfaced is **X-VLA-0.9B** ([ICLR 2026 paper](https://github.com/2toinf/X-VLA), [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base)) - Florence-2 vision backbone, native LeRobot integration, multi-image via `num_image_views`, Apache 2.0, **same smallest-model-bonus tier as SmolVLA**, Phase-II adaptation tunes only ~9M params. It is the cheapest "different architecture" parallel insurance bet. No published image-as-prompt benchmark for X-VLA, so it's a bet on backbone quality.

---

## 3. Track 1 - Reference-photo curation + print-domain augmentation (G1 fix)

Cost: ~1 day work, ~3 h Brev. Risk: low. Expected lift: small alone, **compounds with Track 2**.

### 3.1 Why this is necessary

Three independent biometric standards converge on what an enrollment-quality face photo looks like ([ISO/IEC 19794-5:2011 §7](https://www.iso.org/standard/50867.html), [NIST FRVT Quality](https://pages.nist.gov/frvt/html/frvt_quality.html), [Paravision biometric whitepaper](https://www.paravision.ai/whitepaper-face-recognition-and-biometric-image-quality/)):

- Yaw / pitch / roll ≤ ±5° from frontal (relax to ±15° to keep recall on a 192-celeb bank)
- Inter-eye distance ≥ 60 px (NIST FRVT 1:1 §5.3), ≥ 90 px preferred
- Illumination uniformity, no hard shadows, left-right luminance ratio < 1.5
- Plain background separable from skin tone
- No glasses if avoidable, no occlusion

Our 192-celeb scraped bank does **not** meet these standards uniformly. Many photos are profile shots, magazine-filtered portraits, or full-body images where the face is < 25% of the frame. SmolVLA never sees a clean reference; it sees noise.

### 3.2 The print-domain gap (the more important sub-problem)

[Arxiv 2404.06559 §4](https://arxiv.org/html/2404.06559v2) quantifies the magazine-photo → printed-A5-cutout transformation as a **+5.64% / +16.00% FMR shift on ArcFace verification**. Effects ordered by published magnitude:

1. **Halftone dot pattern** - every printer reconstructs continuous-tone images as binary ink dots. Inkjet at 300 DPI on A5 has a ~1.05 mm dot pitch visible at typical viewing distance. ([Wikipedia error diffusion](https://en.wikipedia.org/wiki/Error_diffusion), Floyd-Steinberg is the canonical desktop-printer default per CUPS/HP/Canon driver docs.)
2. **Color gamut compression** - sRGB display covers ~35% of CIE 1931; CMYK print covers ~21%. Saturated colors clip toward gray; blues shift purple. ([W3C Color Workshop, Lilley 2021](https://www.w3.org/Graphics/Color/Workshop/slides/talk/lilley).)
3. **Paper grain / fiber texture** - broadband ~1/f noise, well-modeled by Perlin fractal Brownian motion at 4 octaves. ([Augraphy paper texture, arxiv 2208.14558](https://arxiv.org/abs/2208.14558).)
4. **Print MTF blur** - σ ≈ 0.3–0.6 px at 224 px equivalent.
5. **Tone compression** - printer dynamic range ~6 stops vs screen ~10; specular highlights flatten.

### 3.3 Recipe - bank filter

Use InsightFace `buffalo_l` (`det_10g.onnx` + `1k3d68.onnx` + `w600k_r50.onnx`, same teacher as our existing bank filter for consistency):

```python
keep_photo = (
    det_score        >= 0.65          # RetinaFace conf (InsightFace default 0.5; raise for enrollment)
    and abs(yaw)     <= 15.0          # degrees, from 1k3d68 pose head
    and abs(pitch)   <= 15.0
    and abs(roll)    <= 10.0
    and inter_eye_px >= 60            # post 224×224 upscale
    and face_area    >= 0.25 * img_area
    and embedding_norm >= median(bank) - 1.0 * sigma   # MagFace proxy (arxiv 2103.06627)
    and laplacian_var >= 100          # blur reject (standard FIQA proxy)
)
```

Sources: [InsightFace repo](https://github.com/deepinsight/insightface), [MagFace arxiv 2103.06627 §4.2](https://arxiv.org/abs/2103.06627), [SDD-FIQA arxiv 2103.05977 §3.2](https://arxiv.org/abs/2103.05977), [CR-FIQA CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Boutros_CR-FIQA_Face_Image_Quality_Assessment_by_Learning_Sample_Relative_Classifiability_CVPR_2023_paper.pdf).

Expected: 192 celebs → ~140–170 keep ≥1 enrollment-grade photo. For celebs with 0 surviving photos, fall back to relaxed gates or hand-curate.

### 3.4 Recipe - pre-crop offline

Head + shoulders crop, margin = 0.5 × inter-eye on each side and 1 × inter-eye on top, then resize to 224×224. NOT a tight face crop - HFR literature ([arxiv 2404.14247 §4.1](https://arxiv.org/abs/2404.14247), [arxiv 2307.07032 §3.2](https://arxiv.org/abs/2307.07032)) confirms broader head box matches what FR encoders were trained on and preserves hair/jaw cues that the printed cutouts retain.

**Compliance check (PROJECT.md §3):** the crop is computed *offline once per celeb* and shipped as a static asset table. No face detector runs at inference. Rule respected.

### 3.5 Recipe - print-domain forward augmentation

Apply at training time to the reference (camera2) stream only, with probability 0.7. The remaining 30% see the clean magazine photo to preserve photo↔print invariance.

```python
# Augraphy-inspired ink → paper → post pipeline, adapted for photo-as-print
def print_simulate(bgr, rng):
    big = cv2.resize(bgr, (600, 600), interpolation=cv2.INTER_CUBIC)
    # 1. Lab gamut compression (sRGB → CMYK proxy)
    lab = cv2.cvtColor(big, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = rng.uniform(0.75, 0.90)
    lab[..., 1] = (lab[..., 1] - 128) * chroma + 128
    lab[..., 2] = (lab[..., 2] - 128) * chroma + 128
    lab[..., 0] = np.clip(lab[..., 0], 10, 240)         # tone compression
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    # 2. Print MTF blur
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 0.8))
    # 3. Floyd-Steinberg color dither (300-DPI equivalent)
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    pil = pil.quantize(colors=64, dither=Image.FLOYDSTEINBERG).convert("RGB")
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    # 4. Perlin fBm paper grain (4 octaves, ±3 LSB)
    out = add_perlin_grain(out, rng, amplitude=3.0, octaves=4)
    # 5. Re-image at wrist-cam resolution + JPEG ISP
    small = cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA)
    return jpeg_recode(small, quality=rng.randint(70, 90))
```

The operator order is principled (ink → paper → post per [Augraphy docs](https://augraphy.readthedocs.io/en/latest/doc/source/how_augraphy_works.html)). No published paper validates this exact recipe for *photo-print emulation in a VLA training stream* - we are first-movers here. **Flagged risk.**

### 3.6 Validation gates (CLAUDE.md §7)

Before greenlighting a re-fine-tune on this aug recipe:

1. **Calibration print capture.** Print 5 representative celebrities (1 IID, 4 OOD) on the actual printer + paper that eval-day will use. Capture each with the wrist cam from the workspace pose. This is the ground-truth domain.
2. **ArcFace similarity probe.** Compute cos(`buffalo_l`(aug-photo), `buffalo_l`(real-print-photo)) per celeb. Target: median ≥ 0.55, ≥ 80% pairs above 0.5 (standard ArcFace enrollment threshold).
3. **Domain-gap classifier probe.** Train a tiny linear classifier on `is_real_print` (real vs aug-print vs clean-magazine). If real-vs-aug-print > 85% separability → augmentation is too weak. If < 55% → too strong. Sweet spot is 55–80%.
4. **Visual gate.** Write `eval_3/aug/dbg/dbg_print_aug_grid.py` showing 4×4 (clean / aug / real-print) for 8 celebs. Eyeball before greenlight.

If any gate fails, iterate the augmentation hyperparameters before retraining.

---

## 4. Track 2 - ArcFace cosine distillation into SmolVLA's SigLIP (G2 fix)

Cost: ~0.5 day code + ~150 LOC dataset prep + ~1.5–3 h Brev training. Risk: medium (first-mover on this exact combination). Expected lift: **+10 to +25 pp absolute** on celeb-selection accuracy per the closest published analog.

### 4.1 The mechanism, citation by citation

The technique is a **per-patch auxiliary cosine loss** that pulls SigLIP's reference-stream patch embeddings (`observation.images.camera2` only) toward an ArcFace teacher's identity embedding, gated to face patches.

The canonical published recipe is **"Don't Blind Your VLA"** ([arxiv 2510.25616](https://arxiv.org/html/2510.25616v1), [github.com/CognitiveAISystems/BlindVLA](https://github.com/CognitiveAISystems/BlindVLA)). Their **equation (9)**:

```
L_align = − (1/k) Σ_{j=1}^{k} cos(F.normalize(u_j), F.normalize(z_j))
```

where `u_j` are projected patch features from the VLA's vision encoder, `z_j` are corresponding teacher features, and `k` is the number of aligned patches. Table 8 of the paper ablates L2 and InfoNCE; **cosine wins**. Reported gain: **Semantic OOD +24% relative, Vision OOD +12% relative** on LIBERO with a general vision teacher (DINOv2 / SigLIP / Theia).

ArcFace-family models live on a hypersphere by construction ([ArcFace, arxiv 1801.07698](https://arxiv.org/abs/1801.07698)), so cosine is the *theoretically correct* metric - L2 on un-normalized features under-uses the angular geometry. Face-recognition distillation literature converges: [Evaluation-Oriented KD, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Evaluation-Oriented_Knowledge_Distillation_for_Deep_Face_Recognition_CVPR_2022_paper.pdf), [ICD-Face, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_ICD-Face_Intra-class_Compactness_Distillation_for_Face_Recognition_ICCV_2023_paper.pdf), [Unified-KD, arxiv 2508.11376](https://arxiv.org/html/2508.11376v1) all use cosine/angular distillation between teacher and student face embeddings; L2 underperforms in their ablations.

### 4.2 Differences from BlindVLA we must respect

BlindVLA distills *general* vision teachers (DINOv2, SigLIP, Theia, C-RADIOv3) into a VLA backbone with **no ROI gating** - the teacher produces full-frame patch features, so aligning every student patch is sound. **We can't do that with ArcFace.** ArcFace produces *one embedding per detected face*, not per patch. Aligning non-face patches against a face embedding would silently corrupt the SigLIP features for everything not-face.

**Mitigation:** pre-compute a binary face mask on every camera2 reference image offline via InsightFace RetinaFace (`buffalo_l`'s `det_10g.onnx`); store the mask in the LeRobot v3 dataset alongside the image; at training time, downsample to the SigLIP patch grid (14×14 with SO400m at 224²); mean-pool patches inside the mask; project (`nn.Linear(1152, 2048)` → `GELU` → `nn.Linear(2048, 512)`); L2-normalize; cosine against the L2-normalized cached ArcFace embedding.

This is **first-mover work** - no published paper distills ArcFace specifically into SigLIP. The closest precedent is [CLIP-for-FR (arxiv 2411.12319)](https://arxiv.org/abs/2411.12319), which fine-tunes CLIP with FR labels (not distillation, but related domain transfer); and multi-teacher KD into ViTs ([Theia](https://rai-inst.com/wp-content/uploads/2024/12/Theia_Distilling-Diverse-Vision-Foundation-Models-for-Robot-Learning.pdf), [Unified-KD](https://arxiv.org/html/2508.11376v1)) showing cross-domain teacher transfer works when projector size is matched. The recipe transfers in principle; we keep an ablation toggle and flag this as research-grade rather than engineering-grade.

### 4.3 Injection point - patch-level, pre-connector

SmolVLM uses a 2×2 pixel-shuffle to merge 4 SigLIP patches into 1 visual token (256 → 64 tokens per image; [SmolVLM, arxiv 2504.05299 §3.1](https://arxiv.org/html/2504.05299v1)). The merge throws away spatial granularity *inside* each 2×2 block. For face-region patches that's ~80% of the relevant signal.

**Recommendation:** hook at SigLIP's `last_hidden_state` *before* the connector - the 256-token side. Specific file pointer: [`third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:179`](../../third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py#L179) - `embed_image` method, capture `vision_model(...).last_hidden_state` before `connector(...)`.

Loss combines at [`modeling_smolvla.py:355`](../../third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py#L355) in `forward`:
```python
loss = flow_matching_loss + 0.2 * align_loss
```

### 4.4 Lambda

[BlindVLA Table 8](https://arxiv.org/html/2510.25616v1): **λ = 0.2** gives "most stable improvements." Sensitivity is real but not knife-edge - the λ ∈ [0.1, 0.5] sweep is monotone-ish near 0.2; above 0.5 the action loss starts degrading. Pi0.5-KI uses α = 1.0 because they isolate gradients (`stop_gradient` between action expert and VLM); FR-KD literature uses λ ∈ [0.1, 1.0]. **We pick 0.2 as the default, plan a 0.1 / 0.2 / 0.3 sweep if time allows.**

### 4.5 Teacher

InsightFace `buffalo_l` (`w600k_r50`, 512-dim, ArcFace margin trained, ~1e-5 FMR per [InsightFace Choose-Model Guide](https://www.insightface.ai/guides/choose-face-recognition-model-and-evaluate)). **Same teacher as the existing bank filter** - consistency matters more than the marginal quality gain from heavier `antelopev2`. Embeddings cached offline once per reference photo → zero training-time cost from ArcFace.

### 4.6 Anti-forgetting safeguards

Four concurrent mitigations, all cited:

1. Small λ (0.2) - BlindVLA Table 8.
2. Freeze the projector after warmup ([BlindVLA implementation](https://github.com/CognitiveAISystems/BlindVLA)).
3. Mask-gated loss - patches outside the face are untouched, so non-face grounding (banana, coke, table) is preserved.
4. Camera2-only - camera1 (wrist) never sees alignment pressure, so manipulation grounding is preserved.

### 4.7 Risks

| Risk | Probability | Mitigation |
|---|---|---|
| Teacher mismatch on printed cutouts (printed faces have specular highlights, color shifts; ArcFace may give noisy embeddings) | medium | Quality-filter teacher embeddings; drop frames where ArcFace `det_score` < 0.6 |
| Patch-grid misalignment - small printed face occupies < 10 patches | low for camera2 (whole image is the face) | Skip frames where mask covers < 10 patches |
| Forgetting non-face grounding | low (mitigations in §4.6) | Run a 5-task sanity probe - banana / coke / bowl / table edge / hand - before and after, expect < 5pp regression |
| First-mover: nobody has distilled ArcFace into SigLIP specifically | medium | Keep an ablation toggle; A/B on a 20-celeb held-out validation slice before scaling |

### 4.8 Implementation

Total estimate: **~80 LOC in the policy** + **~150 LOC dataset prep**:

- Dataset prep script (one-shot, ~30 min on dev box): for each variant in `datasets/eval3_aug_v3/`, load `camera2` mp4 (constant-frame), run RetinaFace + ArcFace once, cache `face_mask.png` (binary) and `arcface_emb.npy` (512 fp32) under each variant dir; extend `meta/info.json` `features` with two new keys; extend `meta/episodes/*.parquet` with manifest rows.

- Policy patch in `third_party/lerobot/src/lerobot/policies/smolvla/`:
  - `smolvlm_with_expert.py:179` - modify `embed_image` to optionally return pre-connector `last_hidden_state`
  - `modeling_smolvla.py:404` - `prepare_images` plumbing for the mask + cached embedding
  - `modeling_smolvla.py:626` - `embed_prefix` patch-token capture for camera2
  - `modeling_smolvla.py:355` - `forward` adds `0.2 * align_loss`
  - New module `face_align_projector.py` - 2-layer MLP, ~15 LOC

- Resume training from the existing 30k checkpoint for **5k–15k additional steps** with the new dataset version. Same LR, same recipe except for the new loss term. Brev wall-clock: ~1.5–4 h.

### 4.9 Honest assessment

ArcFace distillation is **likely necessary but probably not sufficient alone**. It directly addresses (G2 - representation gap) but does not address (G1 - domain gap). Combining with Track 1 is what closes both axes simultaneously. The published evidence supporting this exact combination is zero; the evidence supporting each piece independently is strong; the synthesis is research-grade extrapolation.

---

## 5. Why VQA co-training is **deferred**, not rejected

This was the original Phase 2d plan in [`eval_3/README.md`](../README.md), skipped for v1. The deep-dive agent's verdict: **conditional NO for this iteration.**

### 5.1 The hard implementation blockers we didn't know about

Two blockers in our pinned LeRobot tree (0.5.1):

- [`third_party/lerobot/src/lerobot/datasets/factory.py:113`](../../third_party/lerobot/src/lerobot/datasets/factory.py#L113) - `MultiLeRobotDataset` is **explicitly deactivated** (`raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")`). Co-training requires writing a custom interleaving dataloader from scratch.
- [`third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py:763-799`](../../third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py#L763) - SmolVLA's training forward computes **only** `F.mse_loss(u_t, v_t)` against the action expert's velocity field. There is **no language-modeling head in the training graph**, no LM logits surfaced from the prefix tokens. Adding VQA co-training requires:
  1. Adding an LM head on top of `vlm_with_expert`'s prefix output (likely tied to the SmolVLM tokenizer's existing `lm_head`)
  2. A second forward path that skips `embed_suffix` + action tokens entirely for VQA samples
  3. Per-sample loss masking (`M^act`, `M^ℓ`) à la Pi0.5-KI (arxiv 2505.23705 §3)
  4. A custom dataset that interleaves robot frames + VQA tuples with a deterministic ratio

Estimated lift: **3–5 days of focused engineering plus debugging** before the first co-training step runs. That's the entire eval-day budget consumed on plumbing, with no head-to-head evidence the technique helps at our scale.

### 5.2 The objective mismatch

Naive name-binding VQA - `(headshot, "Who is this?", "Yann LeCun")` - trains the LM to bind names. **This is mismatched to our task.** At eval time the model sees a *novel* reference photo and must match it to a workspace portrait - that's reference resolution, not face recognition. The closest published recipe is Interleave-VLA's image-as-prompt format (`(camera2=ref, camera1=scene, "find this person", action)`), which we already trained on (because that's exactly what our 4195-episode dataset is). VQA co-training with name-VQA pairs would be **adding a weaker version of training data we already have**.

### 5.3 The evidence-vs-our-scale gap

Pi0.5's web-data ablation ([pi.website/blog/pi05](https://www.pi.website/blog/pi05)) shows **94% → 74% OOD = −20pp** without web/VQA co-training, but the evaluation is *household object generalization on a 3.3B model*. Three differences from our case:

- They co-train on a **3.3B PaliGemma backbone** with mature web pretraining; SmolVLA-450M lacks that capacity headroom.
- They evaluate **object OOD**, not identity matching.
- They use **Knowledge Insulation** (gradient stop between action expert and VLM, [arxiv 2505.23705 §3](https://arxiv.org/html/2505.23705v1)) which requires the LM-head plumbing we don't have.

The "Don't Blind Your VLA" path (alignment loss, no VQA) reports **+10pp OOD on OpenVLA-7B** but **does not co-train**. Their conclusion: for *visual representation* failures, an alignment loss beats VQA co-training - which is the exact framing of our (G2) gap.

### 5.4 What we keep on the roadmap

VQA co-training is the right intervention for a different failure mode (general OOD object generalization). If Tracks 1 + 2 land us at SmolVLA ≤ 1 rollout behind Pi0.5 and we have wall-clock remaining before demo day, **image-as-prompt VQA synthesized from existing 4195 episodes as an action-masked auxiliary loss** is the right Phase 2 move - exactly the format Interleave-VLA uses, no VGGFace2 dependency.

---

## 6. The recommended plan (24-hour deadline)

Run **three parallel tracks on Brev**. Two are bonus-preserving; one is insurance.

| Track | What | Compute (Brev) | Risk | Bonus |
|---|---|---|---|---|
| **A** | SmolVLA-boost-v2: bank filter + pre-crop + print augmentation (Track 1) + ArcFace distillation (Track 2), resume from 30k for +10–15k steps | ~4–6 h | low/medium | **+20 (preserved)** |
| **B** | X-VLA-0.9B vanilla fine-tune from `lerobot/xvla-base` on our existing dataset, no augmentation/distillation changes | ~10–15 h | medium (no image-as-prompt benchmark for X-VLA) | **+18 or +20** depending on field |
| **C** | (Optional, only if budget allows) Pi0.5 fine-tune from `lerobot/pi05_base` at bs=24 + grad_checkpoint + compile_model, ~30k steps | ~27–33 h | medium | **−4 (third tier)** |

Total Brev cost for A + B: **~15–20 h** = ~$25 of the $200 budget. Affordable.

**The decision tree:**
- If A reaches ≥ 6/9 on the eval-day rollout, ship A. (+20 bonus + 33pt rollout = ~53pt total - beats Pi0.5's expected 7/9 + 16 bonus = 55pt by a hair, or beats clean if A reaches 7/9.)
- If A reaches 4–5/9 and B reaches ≥ 6/9, ship B. (Same tier as A bonus-wise; B has different architecture so it diversifies risk.)
- If both A and B reach < 4/9, ship C (Pi0.5) and eat the bonus loss. This is the worst case but bounds the downside.

**Dry-run protocol** (eval-day morning):
1. 3 rollouts per checkpoint × 3 checkpoints = 9 dry-run rollouts
2. Mix: 1× IID (Swift/Obama/LeCun original photo), 1× held-out IID, 1× OOD (TA-published)
3. Pick highest-success checkpoint for the real eval

---

## 7. Open questions / what's still unknown

1. **First-mover on ArcFace → SigLIP distillation.** No published recipe combines these exact teachers/students. The transfer should work (cited evidence in §4.1) but is research-grade extrapolation.
2. **Print-domain augmentation generalisation to eval-day printer.** Our recipe is calibrated against the Augraphy literature, not our specific printer. The §3.6 calibration probe is mandatory before retraining.
3. **Whether the action expert has actually learned a position prior, vs simply not converged on face-matching.** Track 1+2 fix the face-matching half but might leave the position prior baked in. If retrained policy still fails the position-rotation test, we need a curriculum that randomises celebrity-to-position mapping (we tried - the augmented dataset is balanced - but the *base 178 teleops* are not). Mitigation: re-balance by sampling base teleops with inverse-frequency weights on celebrity-position pairs.
4. **The TA-published OOD list.** Still TBD. If the list is dominated by athletes/musicians (typical magazine-photo profile), our pipeline transfers cleanly. If it's heavy on politicians (often photographed in suits, similar pose) or academics (often photographed at lecterns), the within-class face-feature variance is lower and discrimination is harder.

---

## 8. Synthesis & sources

The 7 research agents triangulated to a stable recommendation:

- (D - bonus economics): Pi0.5 wins only if ≥ +4 rollouts over SmolVLA. No evidence of that.
- (A - Pi0.5 feasibility): Pi0.5 borderline-feasible to deploy on 16 GB, no published face-matching advantage, costs the bonus + 30 h training.
- (B - alt VLAs): OpenVLA-7B is deploy-blocked; X-VLA-0.9B is the one viable scale-up insurance bet (same bonus tier).
- (C - boost SmolVLA): Re-curate refs + halftone aug + ArcFace distillation, ~70% chance to lift, days not weeks, keeps the bonus.
- (E - ArcFace distillation deep dive): BlindVLA recipe transfers, λ = 0.2, patch-pre-connector injection, mask-gated, camera2-only.
- (F - reference photos + print augmentation): NIST/ISO enrollment standards, head+shoulders crop, 10-step Augraphy-inspired pipeline, p = 0.7 mix.
- (G - VQA co-training): defer this iteration - wrong objective for our task, 3–5 days plumbing, evidence only at 3B+.

### Primary citations

- BlindVLA - [arxiv 2510.25616](https://arxiv.org/html/2510.25616v1) + [github.com/CognitiveAISystems/BlindVLA](https://github.com/CognitiveAISystems/BlindVLA)
- Pi0.5 - [arxiv 2504.16054](https://arxiv.org/abs/2504.16054) + [pi.website/blog/pi05](https://www.pi.website/blog/pi05)
- Pi0.5-KI - [arxiv 2505.23705](https://arxiv.org/html/2505.23705v1) + [pi.website/research/knowledge_insulation](https://www.pi.website/research/knowledge_insulation)
- π0 - [arxiv 2410.24164](https://arxiv.org/abs/2410.24164)
- Interleave-VLA - [arxiv 2505.02152](https://arxiv.org/abs/2505.02152)
- SmolVLM - [arxiv 2504.05299](https://arxiv.org/html/2504.05299v1)
- SmolVLA - [arxiv 2506.01844](https://arxiv.org/abs/2506.01844)
- ArcFace - [arxiv 1801.07698](https://arxiv.org/pdf/1801.07698)
- X-VLA - [`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base) + [github.com/2toinf/X-VLA](https://github.com/2toinf/X-VLA)
- MagFace - [arxiv 2103.06627](https://arxiv.org/abs/2103.06627)
- SDD-FIQA - [arxiv 2103.05977](https://arxiv.org/abs/2103.05977)
- Print-and-Scan morph - [arxiv 2404.06559](https://arxiv.org/html/2404.06559v2)
- Heterogeneous FR - [arxiv 2404.14247](https://arxiv.org/abs/2404.14247) + [arxiv 2307.07032](https://arxiv.org/abs/2307.07032)
- Augraphy - [arxiv 2208.14558](https://arxiv.org/abs/2208.14558) + [docs](https://augraphy.readthedocs.io/)
- NIST FRVT Quality - [pages.nist.gov/frvt/html/frvt_quality](https://pages.nist.gov/frvt/html/frvt_quality.html)
- ISO/IEC 19794-5:2011 - [iso.org/standard/50867](https://www.iso.org/standard/50867.html)
- Evaluation-Oriented KD - [CVPR 2022 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Evaluation-Oriented_Knowledge_Distillation_for_Deep_Face_Recognition_CVPR_2022_paper.pdf)
- ICD-Face - [ICCV 2023 PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_ICD-Face_Intra-class_Compactness_Distillation_for_Face_Recognition_ICCV_2023_paper.pdf)
- Unified-KD - [arxiv 2508.11376](https://arxiv.org/html/2508.11376v1)
- CLIP-for-FR - [arxiv 2411.12319](https://arxiv.org/abs/2411.12319)
- InsightFace - [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface) + [model evaluation guide](https://www.insightface.ai/guides/choose-face-recognition-model-and-evaluate)
- LeRobot v3 dataset - [HF docs](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)

### Local source pointers

- [`docs/PROJECT.md`](../../docs/PROJECT.md) - eval rubric, smallest-model bonus
- [`docs/VLA_ARCHITECTURES.md`](../../docs/VLA_ARCHITECTURES.md) - original architecture decision, knob inventory
- [`docs/RELATED_WORK.md`](../../docs/RELATED_WORK.md) - prior public work survey
- [`eval_3/README.md`](../README.md) - Eval 3 project plan + 2026-05-09 PaliGemma probe results
- [`eval_3/aug/STRATEGY_v3.md`](STRATEGY_v3.md) - current augmentation strategy
- [`eval_3/aug/RESEARCH_v2.md`](RESEARCH_v2.md) - detection + temporal-stability research
- [`third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`](../../third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py) - SmolVLA training forward + image plumbing
- [`third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py`](../../third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py) - vision encoder hook point
- [`third_party/lerobot/src/lerobot/datasets/factory.py`](../../third_party/lerobot/src/lerobot/datasets/factory.py) - MultiLeRobotDataset blocker for VQA co-train
