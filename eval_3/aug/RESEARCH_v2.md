# Research v2 — fixing detection + temporal stability (2026-05-10)

After running stages 2–4 end-to-end on ep01 and observing **jittery / blinking
masks** that occasionally vanish during gripper crossings, we re-evaluated:

1. how to automatically detect the 3 portraits without manual clicking (replace `--interactive` click prompts)
2. how to produce **temporally stable** masks (no flicker, no blinking)

Two independent research threads converged on the same answer. This document
records the decision and the evidence.

---

## 1. Detection — automate the frame-0 seeding

**Decision: HuggingFace transformers `GroundingDinoForObjectDetection` with `disable_custom_kernels=True`.**

### Why not the original IDEA-Research GroundingDINO repo

The upstream repo's CUDA op `ms_deform_attn_cuda.cu` uses deprecated
`.type()` / `DeprecatedTypeProperties` APIs that PyTorch 2.6+ removed.
GroundingDINO issues #405 / #397 / PR #409 confirm this; PR #409 is still
open, has only been tested on torch 2.5/2.6 + CUDA 12.4, and has **zero
Thor + CUDA 13 confirmation**. Building from source on Thor will fail.

### Why HF transformers' port works

HuggingFace re-implemented GroundingDINO inside the `transformers` library
with a config flag **`disable_custom_kernels=True`** that swaps the broken
CUDA op for a pure-PyTorch reference path. Same weights
(`IDEA-Research/grounding-dino-tiny` / `-base`), same accuracy, **no nvcc
build required**:

```python
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny",
    disable_custom_kernels=True,                  # the key flag
).to("cuda")
```

Source: <https://huggingface.co/docs/transformers/model_doc/grounding-dino>

### Concrete usage for our case

```python
text_labels = [["a printed photo", "a portrait on the table"]]
inputs = processor(images=frame, text=text_labels, return_tensors="pt").to("cuda")
outputs = model(**inputs)
results = processor.post_process_grounded_object_detection(
    outputs, threshold=0.3, target_sizes=[(H, W)],
)
# results[0]["boxes"]: list of (x0, y0, x1, y1) bboxes; feed each to SAM 2 as a box prompt
```

Pipeline becomes: GroundingDINO finds 3 boxes → SAM 2 image predictor refines
each box → 3 clean masks → we extract corners and track them across frames.
No click prompts; fully automatic.

### Speed budget on Thor

`grounding-dino-tiny` is Swin-T + ~172 M params. ~50–100 ms / frame fp16 on
A100; expect 10–20 fps on Blackwell-class Thor without TensorRT export. We
only run it on **frame 0** of each video, so this is one-shot cost per
video, not per-frame.

### Fallback ordering

If GroundingDINO misses a portrait at frame 0 (low-confidence threshold,
extreme lighting, etc.):
1. Re-run with lowered threshold (0.3 → 0.2)
2. Fall back to manual click prompts (`--interactive`, the current path)

### Alternatives evaluated and rejected

| Model | Rejection reason |
|---|---|
| **YOLO-World** (Ultralytics) | Backup option (Ultralytics has confirmed Thor + JetPack 7.0 support). No mask output, would still need SAM 2 chain. Slightly faster than GroundingDINO but adds another dep. |
| **OWLv2** (HF transformers) | ViT-L/14 backbone, 2–5 fps at 640×480 on Thor — too slow. |
| **Florence-2** (Microsoft) | Generative decode adds latency (3–8 fps). Versatile but heavy. |
| **SAM 3 native text prompts** | See §2 below — SAM 3 is rejected for stability + speed reasons. |

---

## 2. Temporal stability — replace per-frame SAM with track-once-then-interpolate

**Decision: SAM 2.1 once on frame 0 → Lucas-Kanade optical-flow track the 4 corners across all remaining frames. Re-anchor via a fresh detection if LK confidence drops.**

### Why the current pipeline jitters

SAM 2's memory bank propagation **re-derives the mask each frame** from
learned features attending into the recent-frame memory. Even for a
physically static object, the re-derivation is noisy — the contours shift
by 1–3 px per frame, masks occasionally shrink during partial gripper
occlusion, and recover differently. The visible result: portraits that
"blink" or "shrink and grow" when the inpaint pastes a photo at the
shifting mask boundary.

### Why static + planar = optical flow is provably better

The user's setup has three properties that classical tracking exploits:

1. **The portraits are physically static** — they sit on the table and do not move.
2. **The portraits are planar** — printed paper is approximately flat.
3. **The camera motion is smooth** — robot arm at 30 fps produces smooth inter-frame motion.

For this regime, the **homography between any two frames is a single 3×3
matrix** that maps every point on a portrait from frame i to frame j.
The 4 corners specifically obey this homography exactly. So:

- Estimating the homography between frame 0 and frame N gives **exact**
  4-corner positions in frame N (modulo measurement noise on the
  feature matches).
- This is **deterministic** — no neural-net per-frame variance.
- Per-frame cost is dominated by feature matching: ~1 ms with sparse ORB,
  ~5–10 ms with SuperPoint + LightGlue.

### Why SAM 3 doesn't help here

| Aspect | Evidence |
|---|---|
| FPS | SAM 3 is 5–6 fps vs SAM 2.1's 40 fps at 640×480 on A100 ([SAM 3 issue #155](https://github.com/facebookresearch/sam3/issues/155)). On Thor + raw PyTorch, expect 1–2 fps. |
| Install | Officially requires torch 2.7 + CUDA 12.6 ([sam3 README](https://github.com/facebookresearch/sam3)); we have torch 2.10 + CUDA 13. No published Thor success. |
| Stability on static objects | A Nov 2025 medical-imaging benchmark found SAM 2 propagation outperformed SAM 3 by 76.9 % vs 7.3 % on compact static structures ([arXiv 2511.21926](https://arxiv.org/html/2511.21926v1)). SAM 3 catastrophically loses track despite a good frame-0 mask. |

SAM 3 is *worse* for our case, not better.

### Why RADIO doesn't help here

RADIO ([github.com/nvlabs/RADIO](https://github.com/NVlabs/RADIO)) is a
**vision foundation backbone** that distills CLIP+DINOv2+SAM into one
encoder. It outputs spatial features, not masks. Using it would require
bolting on a SAM-style decoder, which is strictly more code than what we
already have. Not a SAM replacement.

### Concrete tracking recipe

```python
import cv2
import numpy as np

# Frame 0: get the 4 corners per portrait (from SAM 2.1 image predictor +
# minAreaRect — same as current stage 3, just done ONCE per video).
corners_0: dict[int, np.ndarray] = {0: (4, 2), 1: (4, 2), 2: (4, 2)}

# Build an LK pyramid tracker on the corner points.
lk_params = dict(
    winSize=(31, 31), maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
prev_gray = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)

# Stack all 12 points (4 per portrait × 3 portraits) for one LK call
pts_prev = np.vstack([corners_0[pid] for pid in (0, 1, 2)]).astype(np.float32).reshape(-1, 1, 2)
corners_per_frame = {0: corners_0}

for fi in range(1, n_frames):
    gray = cv2.cvtColor(frames[fi], cv2.COLOR_BGR2GRAY)
    pts_next, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, pts_prev, None, **lk_params
    )
    # Reshape into per-portrait corners
    pts_next_r = pts_next.reshape(3, 4, 2)
    status_r = status.reshape(3, 4)
    confidence = float(status_r.sum() / 12)   # fraction of points kept by LK

    if confidence < 0.5:
        # Gripper occluded too many corners — re-anchor.
        # Cheap option: SuperPoint+LightGlue homography from frame 0 to current.
        # Cheaper option: hold last known corners until LK recovers.
        ...
    corners_per_frame[fi] = pts_next_r
    pts_prev, prev_gray = pts_next, gray
```

Sub-ms per frame (LK on 12 points is essentially free).

### Failure modes + mitigations

| Failure | Mitigation |
|---|---|
| Gripper covers 1–2 corners | LK status flags those corners as lost; the remaining corners suffice to estimate the rigid update (we only need 3 to recover a homography). Hold the lost corners at their previous position. |
| Gripper covers all 4 corners of one portrait | Hold the portrait's last known corner positions until LK recovers. ≤30-frame holds are within homography tolerance for ~150 × 200 px portraits. |
| Long camera move + LK drift over time | Periodic re-anchor via planar homography from frame 0's reference patch (ORB or SuperPoint feature matching). Trigger when LK confidence drops below 0.5 OR every 60 frames. |
| Portrait actually moves (sliding on table) | Doesn't happen in our setup — but if it did, LK would smoothly track the slide. |

### What replaces what in the pipeline

| Current pipeline (per-frame SAM 2) | New pipeline (track-once-then-flow) |
|---|---|
| `2_segment_video.py` → SAM 2 video predictor produces 600 masks/portrait | `2_detect_track.py` → GroundingDINO + SAM 2 image predictor on frame 0; LK tracks the 4 corners across all 599 subsequent frames |
| `3_extract_corners.py` → mask → 4 corners per frame, with occlusion interp | folded into stage 2 (corners are directly produced by the tracker) |
| `portrait_masks.pkl` cache (~900 KB) | replaced by `portrait_track.json` (~50 KB) — just 4 corners per frame |
| `portrait_corners.json` (~840 KB) | same role; produced directly by stage 2 |

The `4_inpaint_video.py` and `5_verify_identity.py` stages are unchanged
— they consume `portrait_corners.json`.

### Expected improvement

- **No jitter** — corners come from a deterministic LK tracker, not a stochastic neural net
- **No blinking** — even during gripper occlusion, corners are held at their last known position rather than vanishing
- **Faster** — LK is ~1 ms / frame vs SAM 2's ~37 ms / frame propagation. Whole 600-frame video processes in ~1 second instead of ~22 seconds
- **Smaller cache** — corners are 4×2 floats vs a 480×640 RLE mask
- **No decord dep** — we no longer need SAM 2's video predictor (only the image predictor on frame 0)

---

## 3. Implementation plan

1. Install HF transformers GroundingDINO support: `pip install "transformers>=4.40"` (already at 5.3.0)
2. Verify GroundingDINO works on Thor: load `IDEA-Research/grounding-dino-tiny` with `disable_custom_kernels=True`, test on ep02 frame 0
3. Write `eval_3/aug/2_detect_track.py` — new combined detect-and-track stage
4. Keep `eval_3/aug/2_segment_video.py` as legacy for one release; add deprecation comment
5. Update `pipeline.py` to use `2_detect_track.py` by default
6. Re-run on ep02-ep05 (skip ep01 — was the "test" episode per user instruction)
7. Generate new dbg_segmentation_videos for visual verification of stability
8. If stability looks good, also re-render augmented variants and compare

---

## 4. References

### Detection
- HF GroundingDINO docs incl. `disable_custom_kernels`: <https://huggingface.co/docs/transformers/model_doc/grounding-dino>
- GroundingDINO upstream issues #405 / #397, PR #409 (won't build on torch 2.10 / CUDA 13): <https://github.com/IDEA-Research/GroundingDINO/issues/405>
- Ultralytics Jetson Thor support: <https://docs.ultralytics.com/guides/nvidia-jetson/>

### Temporal tracking
- OpenCV Lucas-Kanade tutorial: <https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html>
- SuperPoint + LightGlue (heavier-but-robust feature matching for re-anchoring): <https://github.com/cvg/LightGlue>
- SAM 3 vs SAM 2 stability benchmark (arXiv 2511.21926): <https://arxiv.org/html/2511.21926v1>
- SAM 3 speed on Jetson (Medium, Apr 2026): <https://medium.com/@priyadarshinichavan/i-built-a-9x-faster-sam-3-inference-engine-heres-how-i-migrated-meta-s-segment-anything-model-to-0709c600494d>

---

## 5. What this document supersedes

This document supersedes the previous "use SAM 2 video predictor for everything" decision in `STRATEGY.md §3.2`. The strategy doc should be updated to reflect:

- Detection: GroundingDINO via HF transformers
- Tracking: LK optical flow on the 4 corners
- SAM 2 used only on frame 0 (image predictor mode)
- decord dep removed
- portrait_masks.pkl no longer cached per-frame
