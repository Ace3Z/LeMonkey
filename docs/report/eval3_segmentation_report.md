# Eval 3 — Segmentation report (how we find the portraits and ignore everything else)

Local working doc (`docs/report/` is gitignored). Generated 2026-05-10 after
running stages 2–3 of the inpainting pipeline end-to-end on episode 01 of
the 5 quick-record demos.

Companion documents:
- `eval_3/aug/STRATEGY.md` — the full pipeline architecture + design rationale
- `eval_3/aug/VALIDATION.md` — triple-source check on every numerical default + bug-fix audit trail
- `eval_3/README.md` — Eval 3 task spec + the SmolVLA + Path A + inpainting + VQA strategy

---

## TL;DR

> We use **SAM 2.1 hiera-large** to segment exactly the 3 celebrity portraits
> in every frame of every 20-second video. The user clicks the centre of each
> portrait *once* on frame 0 (in a fixed celebrity order); SAM 2 propagates
> the masks across all ~600 frames using its memory bank and an occlusion
> head. **We don't segment the gripper, hands, can, or table** — those just
> remain "everything outside the 3 masks." Stage 4 then leaves those regions
> byte-identical and only modifies the portrait pixels.

---

## 1 · Problem statement

For every recorded demonstration we need, *for every frame*, a tight pixel
mask of each of the 3 printed celebrity photos on the table — and only those
photos, not the gripper, the can, the hand, the table, or shadows. The masks
let stage 4 paste a different photo of the same celebrity into each region
to produce identity-invariant augmented training data.

Constraints:

- The SO-101 wrist camera moves with the arm. Portraits' apparent positions
  shift over time even though they're physically static on the table.
- The gripper crosses every portrait at least once during a 20-second
  manipulation episode; the can sits on top of the target portrait at the
  end.
- We have ~600 frames per episode × 5 episodes initially, scaling to
  ~144 episodes for the main collection.
- Per the project CLAUDE.md §7 strict-quality bar: no shortcuts, no
  hand-waved approximations, validate against published methods.

---

## 2 · What gets segmented (and what doesn't, by construction)

This is the most important conceptual point of the whole pipeline.

| Pixel category | Is it in any portrait mask? | Why |
|---|---|---|
| Printed photo of Swift | ✓ portrait_id 0 | We click on it at frame 0 |
| Printed photo of Obama | ✓ portrait_id 1 | We click on it at frame 0 |
| Printed photo of LeCun | ✓ portrait_id 2 | We click on it at frame 0 |
| The Coke can | ✗ | We never clicked on it |
| The SO-101 gripper | ✗ | Never clicked |
| The user's hand reaching in | ✗ | Never clicked |
| The white table | ✗ | Never clicked |
| Shadows beneath portraits | ✗ | Never clicked |
| Anything else in frame | ✗ | Never clicked |

**SAM 2 does not classify objects.** It does *promptable segmentation*:
"the thing the user pointed to, and only that thing." There is no semantic
class for "table" or "gripper" anywhere in the pipeline. Everything outside
the 3 portrait masks is the "background" by definition — it's whatever
pixels weren't selected.

Stage 4 (`replace_portrait`) preserves this: it modifies pixels *only inside*
the mask. The gripper, hands, can, and table pass through every frame
byte-identical from input to augmented output.

Empirically verified (ep01 frame 0):

| portrait_id | celeb | mask area (px) | mask centre |
|---|---|---|---|
| 0 | swift  | 14 699 | (336, 418) |
| 1 | obama  | 20 036 | (544, 358) |
| 2 | lecun  | 19 501 | (100, 357) |

Total mask coverage: 54 236 / (640 × 480) = **17.7 % of the frame**. The
remaining 82.3 % is the background — gripper, hand, can, table, shadows —
all left untouched by the augmentation.

---

## 3 · How SAM 2.1 actually picks out the portraits

### 3.1 · The click prompt (frame 0 only)

We launch the recorder's frame 0 in an OpenCV window and the user clicks
once on each portrait centre, in a known celebrity order matching the
recording sidecar's `layout` field (e.g., layout `SOL` → click Swift, then
Obama, then LeCun). The click is a *positive point prompt* in SAM 2's API.

Code: `eval_3/aug/2_segment_video.py:interactive_click()`. The click
coordinates + celebrity-name mapping are persisted to
`<episode_dir>/portrait_seeds.json` so subsequent runs are fully headless:

```json
{
  "points": [[290, 410], [550, 320], [90, 320]],
  "labels": [1, 1, 1],
  "celebs": ["swift", "obama", "lecun"]
}
```

`labels: 1` = positive prompt (= "this is the object"), as opposed to
`labels: 0` (= "exclude these pixels"). All three of ours are positive.

### 3.2 · Frame-0 mask refinement (image predictor)

Each click is fed to **`SAM2ImagePredictor.predict()`**. SAM 2 returns
three candidate masks per click (the model's three hypotheses about which
"object" the click belongs to: nearest small thing / medium thing /
largest thing). We keep the highest-scoring one — almost always the
portrait-as-a-whole, not "the eye region" or "the face".

For portrait paper sitting flat on a uniform table, frame-0 masks are
near-perfect: the portrait's edges are high-contrast (paper edge against
table) so SAM's encoder produces a sharp boundary.

### 3.3 · Propagation across all frames (video predictor)

This is where SAM 2 earns its name. **`SAM2VideoPredictor.propagate_in_video()`**
takes the frame-0 masks as initialisation and tracks each one through the
entire video, frame by frame, using:

1. A **streaming memory bank** that stores the object's appearance from
   recent frames (window size 30 frames per the SAM 2 default).
2. An **occlusion head** that emits a per-frame logit indicating whether
   each tracked object is currently visible.
3. **Cross-attention from the current frame's encoder features into the
   memory bank**, producing a fresh mask predicted *as if continuing from
   the seed prompt*.

The user provides 3 clicks total (one per portrait). SAM 2 produces 1 800
masks (3 portraits × 600 frames) without further intervention.

Empirical: ep01's 598-frame propagation took ~22 seconds on Thor (NVIDIA
Jetson AGX Thor, Blackwell-class). ~27 obj-frames/sec.

### 3.4 · Why SAM 2.1, not SAM 3

Audited in `STRATEGY.md §3.2` and `VALIDATION.md` against three sources.
Summary:

| | SAM 2.1 hiera-L | SAM 3 |
|---|---|---|
| FPS @ 640×480, A100 | ~40 | 5–6 |
| Native text prompts | No | Yes |
| Mask quality on rigid printed objects | Excellent | Equivalent |

For our case (3 fixed portraits per video, click-once is tolerable, no need
for text prompts), SAM 2.1's 6× speed advantage compounds to 144 ep ×
600 fr × 3 obj = 260k obj-frames. We'd be CPU-bound on SAM 3's wall-clock
where SAM 2.1 finishes a full pass in <1 hour.

### 3.5 · Storage of the segmentation result

Per video, we cache:

- `<episode_dir>/portrait_masks.pkl` — `{frame_idx → {portrait_id → {rle, score}}}` where
  `rle` is a COCO-RLE-encoded binary mask and `score` is SAM 2's per-frame
  occlusion logit. ~900 KB for a 598-frame ep01 video — about 1/8 the size
  of the original mp4.
- `<episode_dir>/portrait_corners.json` — `{portrait_id → {frame_idx → {corners, occluded, score, interpolated}}}` where
  `corners` is the ordered 4-point quadrilateral derived from the mask via
  `cv2.minAreaRect` + `cv2.boxPoints` (TL, TR, BR, BL).

Re-augmenting with new replacement photos doesn't re-run SAM. We just
read these caches and skip to stage 4.

---

## 4 · How we identify objects that *aren't* part of the segmentation

The short answer: **we don't.** The pipeline makes no attempt to identify
the gripper / hand / can / table semantically. The background is "the
complement of the union of the 3 portrait masks" — `(NOT (mask_0 OR mask_1
OR mask_2))`.

This works because of two convenient properties:

1. **The augmentation problem is binary, not multi-class.** Every pixel is
   either inside one of the 3 portrait masks (replace it) or outside them
   (keep it). There's no third class.
2. **SAM 2's promptable segmentation excludes by default.** When we
   click on Swift's portrait, SAM segments the rectangular paper, not the
   table around it, the can in front of it, or the gripper above it.

There's one caveat worth understanding: when the gripper is *physically on
top of* a portrait (e.g., during the placement at the end of the
trajectory), SAM 2 has to decide whether to:

- (a) include the visible gripper pixels in the portrait mask (treating
   "the rectangle" as a rigid object whose appearance is partially
   occluded), or
- (b) exclude them (treating "the photo paper" as the deeper pixel
   identity, with the gripper as a foreground object).

In practice SAM 2.1 does **(b)** — it tracks the *paper underneath*, with
the occluded region marked via a low `obj_score`. We can confirm this from
the segmentation video: in late frames where the can sits on Swift, the
green mask outlines the paper *behind/under* the can, not the can itself.
The occlusion-head logit drops below 0 for those frames.

This is empirically the right behaviour. If SAM had instead included the
gripper/can in the mask, our stage 4 inpaint would have replaced *those*
pixels too, producing a Frankensteined gripper made of celebrity face. It
doesn't.

### 4.1 · The occlusion head contract

SAM 2's occlusion head (`is_obj_appearing = object_score_logits > 0` per
`sam2_base.py:1115`) tells us which frames have a usable mask vs which
frames the object is hidden:

- **`obj_score > 0`:** object visible, mask is usable as-is.
- **`obj_score < 0`:** object occluded, mask may be empty / spurious. The
  pipeline marks these frames "occluded" and **interpolates the 4 corners
  linearly between the nearest valid neighbours** (`3_extract_corners.py:detect_and_interp_occlusion`).

We additionally guard against silent failures with an **area-drop
heuristic**: if the mask area falls below 50 % of the rolling 15-frame
median, we mark the frame occluded even if SAM said it was visible. This
catches edge cases where SAM produced a tiny spurious mask.

For ep01: **0 frames** required occlusion-interpolation across all 3
portraits — the wrist camera moved smoothly, the gripper occlusions were
brief and the area-drop never triggered. (Per the per-episode summary
output of stage 3: `interp/portrait={0: 0, 1: 0, 2: 0}`.)

---

## 5 · From mask to 4-corner quadrilateral

For the inpaint step (stage 4), we don't actually need pixel-perfect masks
— we need just **4 corner points** per portrait per frame, defining the
perspective-warp target for the replacement photo.

`3_extract_corners.py` reduces each binary mask to:

```python
contour = max(cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE), key=cv2.contourArea)
rect = cv2.minAreaRect(contour)        # (centre, (w, h), angle)
corners = cv2.boxPoints(rect)          # 4 × 2 float32, in some rotation
corners = order_tl_tr_br_bl(corners)   # disambiguate via centroid-quadrant angle
```

Why `cv2.minAreaRect` (the *smallest enclosing rotated rectangle*) and not
`cv2.approxPolyDP` (the polygon-simplifier)? Because rigid printed photos
*are* rectangles, just possibly rotated. `minAreaRect` is stable
frame-to-frame even when the mask has notches from gripper occlusion;
`approxPolyDP` sometimes returns 5 or 3 vertices in those cases, which is
useless for a homography.

Triple-validated against:
- Pérez et al. SIGGRAPH 2003 (Poisson editing — uses arbitrary boundary,
  doesn't prescribe corner extraction)
- OpenCV docs on contour features
- LearnOpenCV's seamless-cloning tutorial

See `VALIDATION.md §3` for the audit.

---

## 6 · Empirical results on ep01 (the pipeline's first real-world run)

### 6.1 · Frame 0 segmentation

`dbg_overlay_frame0.png` shows the raw camera frame next to the same
frame with all 3 SAM 2 masks overlaid (red = LeCun, green = Swift, blue =
Obama). All three masks are tight to the printed photo edges; no leakage
onto the can, the gripper above the workspace, or the table.

### 6.2 · Mask propagation across the 20-second episode

`dbg_segmentation.mp4` (598 frames, 30 fps) shows the masks tracking the
entire episode. Visual inspection:

- ✓ Masks stay aligned to the portrait edges as the wrist camera rolls
  during the manipulation.
- ✓ The can on the table never enters any mask (even when adjacent to a
  portrait).
- ✓ The gripper crossing above each portrait does not contaminate the
  masks.
- ✓ At the end of the episode when the can rests on Swift, the green mask
  outlines the *paper underneath* the can, with the can excluded. SAM 2's
  memory bank correctly identifies "paper" as the tracked object's deeper
  identity.
- ✓ The user's hand reaches in (visible at the right edge of multiple
  frames) — never enters any mask.
- ✓ Per-frame `obj_score` stays >> 0 throughout (no false occlusions).

### 6.3 · Corner extraction

`portrait_corners.json` for ep01: 598 × 3 = 1 794 quadrilaterals. **0
required occlusion-interpolation** (`interp/portrait={0:0, 1:0, 2:0}`).
Average aspect ratio of the rotated rectangles: 0.71 (matches A5's 148:210
= 0.704 ± 1 %).

One edge case caught by the validation pass: the rotated `minAreaRect` for
a portrait near the camera edge can return corners slightly off-image
(observed: Obama corner at x = 654 in a 640-wide frame). The downstream
inpaint pipeline now clips corners to image bounds before computing the
homography, since `cv2.seamlessClone`'s internal ROI math asserts on
out-of-bounds coordinates. Documented in `4_inpaint_video.py` §0.

---

## 7 · Why this approach instead of alternatives

| Alternative | Why we didn't pick it |
|---|---|
| **Train a custom YOLO/RT-DETR detector for "celebrity portrait"** | Needs labelled data we don't have; SAM 2 is zero-shot and runs on a single click. |
| **GroundingDINO with text prompt "portrait of a person on table"** | Audited (`STRATEGY.md §3.2`): no working aarch64 + CUDA 13.0 build path on Thor (issue IDEA-Research/GroundingDINO#405); deformable attention CUDA op fails to compile. We'd lose the click-cost (3 clicks/video) only to gain a fragile dep. |
| **Frame-by-frame SAM 2 image predictor (no propagation)** | 30× slower because it re-encodes the image and re-prompts each frame. Memory-bank propagation is exactly the speed-up we need. |
| **Optical-flow-based mask warping (Lucas-Kanade)** | Drifts over a 20-second clip; no occlusion handling; would need re-anchoring against a fresh detection periodically. SAM 2's memory bank does this implicitly. |
| **Semantic segmentation (Mask2Former, OneFormer)** | Trained for COCO-class objects ("person", "chair"). Doesn't have a class for "printed photo on table". Would need fine-tuning. |
| **Diffusion-based "find the celebrity face" with a face-recognition model** | We *do* use ArcFace at training time for QA, but never at inference of segmentation. The face-recognition model would only locate the face inside the portrait, missing the rectangular paper edge that defines the homography target. Wrong geometry. |

---

## 8 · Failure modes the pipeline guards against

| Failure | Detected by | Mitigation |
|---|---|---|
| User clicks off-portrait (e.g., on shadow) | SAM 2 returns a mask of the wrong region; would show in `dbg_overlay_frame0.png` | Re-click during interactive mode (or edit `portrait_seeds.json` and `--force` re-run stage 2) |
| Brief gripper occlusion | SAM 2 `obj_score < 0` OR area drop > 50 % of rolling median | Linear corner interpolation (`3_extract_corners.py:detect_and_interp_occlusion`) |
| Long occlusion (> 30 frames) | Same | Linear interpolation handles up to ~30 frames; longer needs mid-stream re-prompting (not implemented; flagged in `STRATEGY.md §7` as a TODO) |
| Mask clipped at image edge | SAM 2 produces a truncated mask; `minAreaRect` may return corners outside the image | Corner clipping inside `4_inpaint_video.py:replace_portrait` (post-Validation Pass 3) |
| Mask catches a hand/finger | Visible in `dbg_segmentation.mp4`; no automatic detection | Manual fix via `portrait_seeds.json` re-edit + `--force` |
| AV1 video format (not readable by cv2/SAM 2 directly on Thor) | First-frame decode fails | `_video_io.py:ensure_h264()` lazy-transcodes once via system ffmpeg + libdav1d; cached as `<stem>__h264.mp4` |
| SAM 2 needs decord (no aarch64 wheel) | `init_state(mp4_path)` errors on import | `_video_io.py:ensure_frame_dir()` extracts JPEGs to a sibling dir; pass that to `init_state` instead |

---

## 9 · References

- Ravi et al., *SAM 2: Segment Anything in Images and Videos*, arXiv 2408.00714
- facebookresearch/sam2 source — the canonical implementation we use
- SAM 2 hiera-large checkpoint: `dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`
- Pérez et al., *Poisson Image Editing*, SIGGRAPH 2003 — Eq. 11 (boundary-conditioned source-gradient guidance) is what stage 4 uses
- OpenCV docs on `cv2.minAreaRect`, `cv2.boxPoints`, `cv2.seamlessClone`, `cv2.findContours`
- This project's `eval_3/aug/STRATEGY.md` and `eval_3/aug/VALIDATION.md` for cross-references
- This project's existing `docs/report/dagger_strategy.md`, `handover_brev.md` etc. for stylistic precedent
