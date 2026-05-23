# VALIDATION - triple-source check on the recipe defaults (2026-05-10)

> Per project CLAUDE.md §7: every numerical default in the inpainting
> pipeline must be cross-checked against ≥ 3 independent sources. This
> document is the audit trail.

## Summary table

| # | Default | Code location | Verdict | Action taken |
|---|---|---|---|---|
| 1 | ArcFace cos-sim ≥ 0.4 | `mining/mine_celeb_photos.py:266`, `_legacy/stage5_verify_identity.py:--threshold` | **CONFIRMED** | Kept; comments + citations added |
| 2 | Reinhard sample 5-px outer ring (`ring_dilate_px=11`) | `stages/inpaint_video.py:replace_portrait` | EMPIRICAL ONLY | Kept; widening guidance documented |
| 3 | MTF Gaussian σ = 0.8 px | `stages/inpaint_video.py:replace_portrait` | EMPIRICAL ONLY | Kept; ESF probe TODO added |
| 4 | Mask erosion = 3 px | `stages/inpaint_video.py:replace_portrait` | **NEEDS CHANGE** | **Reduced to 1 px** (OpenCV erodes ~3 px internally per opencv#17450) |
| 5 | `cv2.seamlessClone` flag = `NORMAL_CLONE` | `stages/inpaint_video.py:replace_portrait` | **CONFIRMED** | Kept; Pérez Eq. 11 cited |
| 6 | SAM 2 occlusion threshold `obj_score < 0` | `_legacy/stage3_extract_corners.py:detect_and_interp_occlusion` | **CONFIRMED** | Kept; sam2_base.py:1115 cited |
| 7 | Rolling window = 15 frames | `_legacy/stage3_extract_corners.py:detect_and_interp_occlusion` | EMPIRICAL ONLY | Kept; bumps proposed for 60 fps |
| + | Recommended ADDS | various | OPTIONAL | `--apply-unsharp` flag added to `replace_portrait` |

## §1 - ArcFace cos-sim threshold = 0.4 - CONFIRMED

- **Source 1:** InsightFace official guide - "Typical 1:1 thresholds for InsightFace recognition packs land in the 0.30–0.45 cosine range at FMR = 1e-4 to 1e-5." (<https://insightface.ai/guides/choose-face-recognition-model-and-evaluate>)
- **Source 2:** DeepFace (serengil) - ArcFace default cos-distance ≤ 0.68 ⇔ cos-sim ≥ 0.32 (derived via C4.5 over labelled pairs). Our 0.4 is **stricter** by ~0.08. (<https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python>)
- **Source 3:** face_recognition (dlib) library - default tolerance 0.6 Euclidean on 128-D ≈ 99.38 % LFW TAR; corroborates that single-global-threshold is conventional. (<https://face-recognition.readthedocs.io>)

**Verdict:** 0.4 is in band, slightly stricter than DeepFace, lighter than the very-strict end of the InsightFace range. Right balance: high recall during photo mining, low false-positives at variant verification.

**Note** (CLAUDE.md §7 "surface unknowns"): if false-positives are observed at stage 5, tighten the verifier to 0.45 while keeping the miner at 0.4 (recall vs precision split).

## §2 - Reinhard sample ring (`dilate 11×11 ⇒ ~5-px ring`) - EMPIRICAL ONLY

- **Source 1:** Reinhard et al. 2001 ("Color Transfer between Images", IEEE CG&A) - defines the Lab mean/std transfer; **does not specify** a sampling region.
- **Source 2:** PyImageSearch & classical impls - typically use whole-image stats; explicitly warn that "large regions with similar pixel intensities can dramatically influence the mean" - exactly why a *local* ring is preferable for inpainting. (<https://pyimagesearch.com/2014/06/30/super-fast-color-transfer-images>)
- **Source 3:** Image-harmonization survey papers (PCT-Net CVPR 2023; "Diverse Image Harmonization" arXiv:2407.15481) - use *background-mask* statistics; no canonical ring width is reported.

**Verdict:** No canonical reference for ring width. 5 px is a defensible local estimator. Code already has a `[WARN]`-style guard (`if len(ring_pix) < 10: return src` in `reinhard_lab`).

**Empirical refinement:** if the augmented portrait shows colour noise on textured backgrounds (e.g., busy table), widen to `ring_dilate_px=19` (≈ 9-px ring).

## §3 - MTF Gaussian σ = 0.8 - EMPIRICAL ONLY

- **Source 1:** Mosleh et al. CVPR 2015 ("Camera Intrinsic Blur Kernel Estimation") - measured PSFs ≈ Gaussian σ ∈ [0.6, 1.2] px for consumer optics.
- **Source 2:** HIPR2 / Wikipedia "PSF" - σ ≈ 1.0 px is the standard "small but visible" anti-alias / OLPF benchmark.
- **Source 3:** OLPF / anti-aliasing literature (PetaPixel, DigitalCameraWorld) - consumer USB cams typically have OLPFs producing 0.5–1.0 px equivalent Gaussian blur.

**Verdict:** 0.8 is squarely in the 0.5-1.0 webcam-PSF range. Defensible as a default.

**Probe TODO (per CLAUDE.md §7):** add `eval_3/aug/dbg/probe_mtf.py` that fits a Gaussian to a real edge-spread function from one of our recorded frames, refines σ. If empirical σ comes out > 1.0, raise.

## §4 - Mask erosion 3 px → NEEDS CHANGE → applied: 1 px

- **Source 1:** Pérez et al. SIGGRAPH 2003 ("Poisson Image Editing") - uses Dirichlet conditions on the *original* mask. **Does not** prescribe erosion.
- **Source 2:** OpenCV issue #17450 - reporters argue that OpenCV's `seamlessClone` already **internally erodes** the mask by ~3 px; stacking another 3-px erosion pulls colour too far inward. (<https://github.com/opencv/opencv/issues/17450>)
- **Source 3:** Community/best-practice (LearnOpenCV, Scaler tutorials) - feather via Gaussian or 1-3 px erosion to suppress halos when the mask edge crosses textured background. The OpenCV `getStructuringElement` default kernel is 3×3 (i.e., 1-pixel erode per iteration).

**Verdict:** Stacking erosions is harmful. **Reduced default to `erode_px = 1`** (one-pixel erosion). Citation added in `stages/inpaint_video.py:replace_portrait`.

## §5 - `seamlessClone(NORMAL_CLONE)` - CONFIRMED

- **Source 1:** OpenCV 4.x docs: "*NORMAL_CLONE … ideal for inserting objects with complex outlines into a new background. It preserves the original appearance and lighting of the inserted object.*" "*MIXED_CLONE … combines structure from the source and texture from the destination … effective when masks may yield halos.*" (<https://docs.opencv.org/4.x/df/da0/group__photo__clone.html>)
- **Source 2:** Pérez et al. 2003 - Eq. 11 ("importing") = NORMAL pure source-gradient guidance; Eq. 12 ("transparent" mixed) = MIXED max-magnitude gradient mixing.
- **Source 3:** LearnOpenCV - "*Normal Cloning the texture of the source image is preserved in the cloned region. Mixed Cloning … picks the dominant texture between the source and destination.*" Picks MIXED for "lazy masks"; NORMAL otherwise.

**Verdict:** NORMAL is correct for our case (replacing one flat photo with another at the same pose; we want the new photo's content fully). Citation now in code.

## §6 - SAM 2 `obj_score < 0` for occlusion - CONFIRMED

- **Source 1:** facebookresearch/sam2 source - `sam2/modeling/sam2_base.py:1115`: `is_obj_appearing = object_score_logits > 0`. The model emits a *logit*; sigmoid > 0.5 ⇔ logit > 0. (<https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py>)
- **Source 2:** SAM 2 paper (Ravi et al., arXiv 2408.00714) - adds an "occlusion prediction head" emitting a presence score.
- **Source 3:** Ultralytics SAM-2 docs / Roboflow & Encord tutorials - confirm the score is a presence-logit; threshold 0 is the canonical decision boundary.

**Verdict:** Our `score_threshold=0.0` (with `<` for occluded) matches the canonical implementation exactly. Citation added.

## §7 - Rolling window = 15 frames - EMPIRICAL ONLY

- **Source 1:** SAM 2 default tracking memory window = 30 frames (sam2 README).
- **Source 2:** Multi-object tracking lit - 5-30 frame sliding windows for area/IoU smoothing; no canonical value.
- **Source 3:** Frigate / surveillance - "minimum of 3 frames to determine object type, median score must exceed threshold". 15 frames ≈ 0.5 s at 30 fps - comfortably above the 3-frame minimum.

**Verdict:** Reasonable midpoint between SAM2's 30-frame default and the literature's 5-frame minimum. If we ever capture at 60 fps, double to 30.

## §8 - Recommended additions

### 8a. Optional unsharp mask after warp + MTF blur - IMPLEMENTED behind flag

Cambridge-in-Colour / community-standard "after resize, sharpen". Currently we *only* blur (MTF), never recover edges. Added an `apply_unsharp: bool = False` parameter to `replace_portrait` - USM σ=1.0, amount=0.5. **Off by default** (canonical recipe is blur-only); enabling it is a documented knob.

### 8b. Laplacian-pyramid blend as an alternative to Poisson - DEFERRED

Pérez/Poisson alone can introduce subtle DC colour shifts ("Laplacian-Membrane Modulation" S0097849316300176). A 3-level Laplacian-pyramid blend with a 3-px-feathered mask is a published alternative. **Defer:** add behind `--blend=poisson|pyramid|hybrid` only if the visual gates from `dbg/dbg_compare_gif.py` show seam issues that Poisson alone can't fix.

### 8c. Per-frame ArcFace gate during inpainting - DEFERRED

CLAUDE.md §7 calls out "frame-by-frame ArcFace verification" as the right path. Currently stage 5 samples 5 frames per variant and checks min cosine. A per-frame fast gate that aborts a variant the moment one frame drops below threshold would be more rigorous. **Defer:** evaluate after running the current pipeline on the 5 quick episodes; if the 5-sample QA misses bad frames, upgrade to per-frame.

### 8d. Don't double-erode - DONE (see §4 above)

### 8e. Pre-fill dst's mask region with ring-mean before seamlessClone - DISCOVERED + FIXED

**Critical bug found by Validation Pass 2** (synthetic smoke test, 2026-05-10).

Synthetic test `test_basic_replacement` failed with output mean ≈ original
portrait colour (`[44, 53, 201]` vs original `[50, 50, 200]`), not the
replacement photo's mean (`[133, 113, 113]`). A/B test of 8 inpaint
variants on the same input revealed **the issue is OpenCV's internal
~3-px erosion of the mask** in `cv2.seamlessClone` (per opencv#17450):
the Dirichlet boundary for the Poisson solve lands ~3 px **inside** the
original portrait region, anchoring the solution to the original
portrait colour rather than the surrounding background.

The integrated source gradients can't override the boundary condition
when the source has reduced contrast (which it has, post-Reinhard's
std-matching).

Without the fix, `dist_to_ring=212` (we want low). After the fix,
`dist_to_ring=23` - exactly what we want for "this photo lives on this
table under this lighting."

**Fix:** before calling `seamlessClone`, replace the destination's mask
region with the ring-mean BGR colour:

```python
ring_pix = src_frame[ring > 0]
if len(ring_pix) >= 10:
    ring_mean_bgr = ring_pix.astype(np.float32).mean(0).astype(np.uint8)
    dst_for_clone = src_frame.copy()
    dst_for_clone[mask > 0] = ring_mean_bgr
else:
    dst_for_clone = src_frame
```

This makes the Poisson boundary land on a smooth ring-mean colour, so
the solution = ring_mean + integrated src gradients ≈ Reinhard'd new
photo at correct local DC.

**A/B numbers** (from `tests/test_replace_portrait.py` 8-variant probe,
synthetic noisy-gray scene with red-ish portrait + blue/yellow photo):

| Variant | dist_to_ring | dist_to_orig_portrait | dist_to_photo | Verdict |
|---|---|---|---|---|
| A. NORMAL_CLONE raw frame (the OLD broken recipe) | 212.8 | 7.1 | 139.0 | catastrophic revert |
| **B. NORMAL_CLONE pre-filled dst (THIS FIX)** | **22.8** | **228.7** | 161.1 | **chosen** |
| C. MIXED_CLONE raw frame | 211.6 | 5.8 | 139.1 | also broken |
| D. NAIVE PASTE (no Poisson) | 61.0 | 261.2 | 198.6 | seam visible |
| E. ALPHA FEATHER paste | 57.1 | 256.5 | 194.6 | softer seam |
| G. NORMAL_CLONE with mask DILATED+2 | 108.9 | 103.4 | 96.7 | partial improvement |

**Without this fix, all augmented videos would look identical to the
originals** - the policy would learn nothing from the augmentation pass.
This is a non-negotiable change.

Code: `stages/inpaint_video.py:replace_portrait` step 4 (new pre-fill block).
Test: `tests/test_replace_portrait.py` (5/5 pass).

## Sign-off

All defaults either **CONFIRMED** with three independent sources, **modified** in response to source evidence (erode 3 → 1 px), or marked **EMPIRICAL ONLY** with reasoning + a refinement plan. Code now carries inline citations to specific source URLs / line numbers so future-me can re-validate quickly.

Audit performed: 2026-05-10. Next re-audit trigger: any time we change one of these defaults or add a new pipeline stage.
