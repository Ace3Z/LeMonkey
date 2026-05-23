#!/usr/bin/env python3
"""STAGE 2 v6 — static-camera pipeline (the simplest correct design).

REPLACES the v5 SAM-2-video-propagator + Kalman+RTS pipeline. Premise:
the wrist camera does NOT move during a 20-second teleop episode, and the
printed portraits do NOT move on the table. The only things that change
between frames are the gripper, the Coke can, and the user's hand — all
of which are *occluders* on top of static paper.

Given that, the right pipeline is:

  Once on frame 0:
    1. GroundingDINO → 3 portrait bounding boxes
    2. SAM 2.1 image predictor → 3 tight masks M_0
    3. cv2.minAreaRect → 4 corners C_0 per portrait (CONSTANT for all frames)
    4. ArcFace → identify which celeb is in each portrait

  For each subsequent frame i:
    occluder_mask_i = (|frame_i − frame_0| > T) AND M_0
    visible_paper_i = M_0 AND NOT occluder_mask_i

  Stage 4 then uses C_0 (fixed) to warp the new photo, and visible_paper_i
  as the alpha for compositing. The gripper / can / hand pixels stay
  byte-identical to the original because they're outside visible_paper_i.

Why this is correct + fast:
  - Corners are CONSTANT across all frames — no tracking, no Kalman, no
    RTS, no jitter. The augmented photo is pixel-locked to the static paper.
  - Change-detection per frame is ~3 ms (absdiff + morphology) — total
    stage-2 cost ~5 s per episode (vs ~628 s for v5 SAM-2-video).
  - Shadows show up as moderate-difference pixels and (by design) get
    excluded from the paste, so the original shadow is preserved through
    the composite where the paper itself shows through.

CLI:
    python 2_detect_static.py /path/to/episode_dir
    python 2_detect_static.py --root ~/LeMonkey/datasets/eval3_quick
    python 2_detect_static.py /path/to/ep --diff-threshold 25 --pre-blur-sigma 1.0

Defaults — see VALIDATION.md v6 §:
    --diff-threshold 25   per-channel BGR diff above which a pixel is
                          declared occluded (typical camera noise is 5-10;
                          slight shadows ~20-40; objects 50+).
    --pre-blur-sigma 1.0  Gaussian blur σ applied to both frames before
                          differencing — suppresses sub-pixel jitter from
                          any negligible camera shake.
    --morph-radius 2      morphological open + close radius on the
                          occluder mask, removes 1-2 px speckle noise.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np


def _import_heavy():
    global cv2, torch, AutoProcessor, AutoModelForZeroShotObjectDetection, build_sam2, build_sam2_video_predictor, SAM2ImagePredictor, ensure_h264, ensure_frame_dir, read_frame_zero, FaceAnalysis, mask_util
    import cv2  # type: ignore
    import torch  # type: ignore
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from insightface.app import FaceAnalysis as _FaceAnalysis
    import pycocotools.mask as mask_util
    spec = importlib.util.spec_from_file_location("_video_io", str(Path(__file__).resolve().parent / "video_io.py"))
    _vio = importlib.util.module_from_spec(spec); spec.loader.exec_module(_vio)
    ensure_h264 = _vio.ensure_h264
    ensure_frame_dir = _vio.ensure_frame_dir
    read_frame_zero = _vio.read_frame_zero
    globals().update({
        "cv2": cv2, "torch": torch,
        "AutoProcessor": AutoProcessor,
        "AutoModelForZeroShotObjectDetection": AutoModelForZeroShotObjectDetection,
        "build_sam2": build_sam2,
        "build_sam2_video_predictor": build_sam2_video_predictor,
        "SAM2ImagePredictor": SAM2ImagePredictor,
        "FaceAnalysis": _FaceAnalysis,
        "mask_util": mask_util,
        "ensure_h264": ensure_h264, "ensure_frame_dir": ensure_frame_dir,
        "read_frame_zero": read_frame_zero,
    })


SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT_DEFAULT = Path.home() / "checkpoints/sam2.1_hiera_large.pt"
GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"
PHOTO_BANK_DEFAULT = Path("/home/lemonkey/LeMonkey/datasets/eval3_celebs/web")

# ─── v8: occluder text prompts (Grounded-SAM-2) ──────────────────────────────
# Period-separated multi-class prompt is the canonical Grounding DINO format
# for multi-object detection. The three classes are everything that can
# physically occlude the paper portraits in our scene.
#
# IMPORTANT: each prompt must be a single noun phrase. "coke can" sometimes
# fires on red regions of the original portraits. Mitigation: we cap the
# per-class detection count + require the box to be MOSTLY OUTSIDE the
# paper masks M_0 (occluders enter the scene from above; if a "can" box
# centers inside a paper mask at frame 0 before the can has been placed,
# it's a false positive).
OCCLUDER_TEXT_PROMPTS = ["robot gripper", "coca cola can"]
OCCLUDER_BOX_SCORE_MIN = 0.40    # v10b: raised from 0.20 → 0.40. The 0.20
                                  # default let through 0.27-confidence "human"
                                  # detections that GroundingDINO was firing
                                  # diffusely on celebrity photos in frame.
                                  # 0.40 matches the IDEA-Research/GroundingDINO
                                  # official BOX_THRESHOLD default (0.35) with
                                  # a small margin. Legit gripper/can detections
                                  # score 0.70–0.80 so they're well above.
                                  # Verified visually 2026-05-13 on swift_OLS_ep01:
                                  # the FP "human" at 0.27 on Swift's portrait
                                  # is now dropped at the score gate.
OCCLUDER_DILATE_PX = 0            # v9.1: was 3 — created a halo ring of original
                                  # photo visible around occluders. The classifier
                                  # (chroma + dark-on-bright) catches anything SAM
                                  # 2 leaks at thin edges, so dilation is no longer
                                  # needed and was actively hurting edge sharpness.

# Aspect / area filter for GroundingDINO box outputs (filter out the noisy
# whole-frame box + any spurious tiny detections).
ASPECT_MIN, ASPECT_MAX = 0.40, 1.30
AREA_MIN, AREA_MAX = 5_000, 60_000
BOX_SCORE_MIN = 0.20


# ─── GroundingDINO detection ─────────────────────────────────────────────────
def filter_portrait_boxes(boxes: np.ndarray, scores: np.ndarray, shape) -> list[int]:
    H, W = shape[:2]
    cands = []
    for i, (b, s) in enumerate(zip(boxes, scores)):
        x0, y0, x1, y1 = [float(v) for v in b]
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0: continue
        if w > 0.85 * W and h > 0.85 * H: continue       # whole-frame box
        ar = w / h
        if not (ASPECT_MIN <= ar <= ASPECT_MAX): continue
        area = w * h
        if not (AREA_MIN <= area <= AREA_MAX): continue
        if s < BOX_SCORE_MIN: continue
        cands.append((i, float(s), np.array([x0, y0, x1, y1])))
    cands.sort(key=lambda x: -x[1])
    chosen, chosen_boxes = [], []

    def _iou(a, b):
        xa = max(a[0], b[0]); ya = max(a[1], b[1])
        xb = min(a[2], b[2]); yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / u if u > 0 else 0

    for idx, s, b in cands:
        if any(_iou(b, cb) > 0.3 for cb in chosen_boxes): continue
        chosen.append(idx); chosen_boxes.append(b)
        if len(chosen) == 3: break
    return chosen


def detect_portraits_grounding_dino(processor, model, frame_bgr: np.ndarray, threshold: float = 0.25):
    from PIL import Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    text_labels = [["a printed portrait photo on a table", "a paper photograph"]]
    inputs = processor(images=Image.fromarray(rgb), text=text_labels, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        out, threshold=threshold, target_sizes=[(frame_bgr.shape[0], frame_bgr.shape[1])],
    )
    boxes = res[0]["boxes"].cpu().numpy()
    scores = res[0]["scores"].cpu().numpy()
    keep = filter_portrait_boxes(boxes, scores, frame_bgr.shape)
    return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], float(scores[i])) for i in keep]


# ─── v8: Occluder detection via Grounding DINO ───────────────────────────────
def detect_occluders_grounding_dino(
    processor, model, frame_bgr: np.ndarray, *, threshold: float = 0.20,
) -> list[dict]:
    """Detect gripper / can / hand boxes on a single frame. Returns a list of
    dicts: {label, box, score}. Multi-class Grounding DINO uses period-
    separated noun phrases (Grounded-SAM-2 canonical format)."""
    from PIL import Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    text_labels = [OCCLUDER_TEXT_PROMPTS]
    inputs = processor(images=Image.fromarray(rgb), text=text_labels,
                       return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        out, threshold=threshold,
        target_sizes=[(frame_bgr.shape[0], frame_bgr.shape[1])],
    )
    boxes = res[0]["boxes"].cpu().numpy()
    scores = res[0]["scores"].cpu().numpy()
    # text_labels per detection — the transformers post-processor returns
    # the original noun phrase under res[0]["text_labels"] in recent versions.
    labels = res[0].get("text_labels") or res[0].get("labels")
    if labels is None:
        labels = ["?"] * len(boxes)
    elif hasattr(labels, "tolist"):
        labels = labels.tolist()
    out_list = []
    for box, score, label in zip(boxes, scores, labels):
        if score < OCCLUDER_BOX_SCORE_MIN: continue
        out_list.append({
            "label": str(label),
            "box": [float(v) for v in box[:4]],
            "score": float(score),
        })
    return out_list


def filter_occluders_against_paper(
    occluder_dets: list[dict], paper_masks: list[np.ndarray],
    *, max_in_paper_fraction: float = 0.60,
    drop_if_center_in_paper: bool = False,
) -> list[dict]:
    """Drop occluder boxes whose AREA is mostly contained in a paper mask M_0.
    These are FALSE POSITIVES: Grounding DINO firing on photo content (e.g.
    Yann LeCun's hand-in-portrait adjusting his glasses, or a red region of
    a portrait matching "coca cola").

    The fraction-based test is robust to both cases we care about:
      - Hand-in-photo (entirely inside paper): fraction ≈ 1.0 → drop ✓
      - Real hand reaching in over the paper (most of box outside paper):
        fraction ≈ 0.1-0.3 → keep ✓
      - Operator's hand placing the can ON the paper (box mostly inside):
        fraction may be ≥ 0.6 too — caveat: we'd drop it. The compromise
        is acceptable because the gripper+can are seeded at frame 0 and
        SAM 2 propagates them; we don't typically need an additional hand
        seed mid-clip to track the placement act.

    v9.2: was center-inside-paper; that mis-handled the corner case where
    a real hand entered the workspace late in the episode with its center
    happening to land inside a paper. Fraction is more robust."""
    keep = []
    for d in occluder_dets:
        x0, y0, x1, y1 = [int(v) for v in d["box"]]
        if not paper_masks:
            keep.append(d); continue
        H, W = paper_masks[0].shape[:2]
        x0c = max(0, x0); y0c = max(0, y0)
        x1c = min(W, x1 + 1); y1c = min(H, y1 + 1)
        if x1c <= x0c or y1c <= y0c:
            keep.append(d); continue
        box_area = (x1c - x0c) * (y1c - y0c)
        max_frac = 0.0
        for M in paper_masks:
            in_paper = M[y0c:y1c, x0c:x1c] > 0
            frac = float(in_paper.sum()) / max(box_area, 1)
            max_frac = max(max_frac, frac)
        # v10b: also reject when the box CENTER lies inside any portrait.
        # Catches the "giant bbox that covers Swift's paper plus a lot of
        # surrounding table" case where in_paper_fraction is small
        # (because box is huge) but the box is centered on a portrait so
        # SAM 2 propagation latches onto the portrait content. Applied at
        # frame 0 only (caller controls).
        cx = int(round(0.5 * (x0c + x1c - 1)))
        cy = int(round(0.5 * (y0c + y1c - 1)))
        center_in_paper = False
        if drop_if_center_in_paper and 0 <= cx < W and 0 <= cy < H:
            for M in paper_masks:
                if M[cy, cx] > 0:
                    center_in_paper = True
                    break

        if max_frac < max_in_paper_fraction and not center_in_paper:
            print(f"  [occluder_kept] label={d['label']!r} score={d['score']:.2f} "
                  f"box=[{x0},{y0},{x1},{y1}] in_paper_frac={max_frac:.2f} "
                  f"center=({cx},{cy})", flush=True)
            keep.append(d)
        else:
            reason = (
                f"in_paper_frac={max_frac:.2f}"
                + (f" AND center_in_paper@({cx},{cy})" if center_in_paper else "")
            )
            print(f"[WARN] occluder_fp_filtered: expected=box_in_paper_fraction<"
                  f"{max_in_paper_fraction:.2f}"
                  + (" AND box_center_outside_papers" if drop_if_center_in_paper else "")
                  + f", got=label={d['label']!r}_score={d['score']:.2f}_{reason}, "
                  f"fallback=drop", flush=True)
    return keep


# ─── v8: SAM 2 video propagation for occluders ───────────────────────────────
def track_occluders_through_video(
    video_predictor, frame_dir: Path, n_frames: int, H_img: int, W_img: int,
    occluder_dets_per_frame: dict[int, list[dict]],
):
    """Seed SAM 2 video predictor with one box per detected occluder at the
    frame it was detected on, then propagate masks through the entire video.

    Returns: per_frame_masks[frame_idx][obj_id] -> (H, W) bool mask
    """
    # SAM 2 expects bfloat16 inference (mixed precision) for video — wrap
    # init_state / add_new_points_or_box / propagate in autocast, otherwise
    # box prompts come in as fp32 and the cross-attention matmul fails with
    # "mat1 and mat2 must have the same dtype" (SAM 2 issue #239).
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = video_predictor.init_state(video_path=str(frame_dir))
        obj_id_to_label: dict[int, str] = {}
        next_oid = 1
        for frame_idx, dets in sorted(occluder_dets_per_frame.items()):
            for d in dets:
                box = np.array(d["box"], dtype=np.float32)
                video_predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=frame_idx, obj_id=next_oid,
                    box=box,
                )
                obj_id_to_label[next_oid] = d["label"]
                next_oid += 1
        if next_oid == 1:
            return {}, {}                                 # nothing to track

        per_frame: dict[int, dict[int, np.ndarray]] = {}
        for fi, obj_ids, mask_logits in video_predictor.propagate_in_video(state):
            for i, oid in enumerate(obj_ids):
                m = (mask_logits[i] > 0.0).cpu().numpy().squeeze().astype(bool)
                if m.shape != (H_img, W_img):
                    m_u8 = cv2.resize(m.astype(np.uint8), (W_img, H_img),
                                      interpolation=cv2.INTER_NEAREST)
                    m = m_u8.astype(bool)
                per_frame.setdefault(fi, {})[int(oid)] = m
    return per_frame, obj_id_to_label


# ─── Paper rectangle detection (the v7 fix) ──────────────────────────────────
#
# Two-stage with safe fallbacks. We need M_0 to be the FULL RECTANGULAR PAPER,
# not the person on the paper (the bug v6 had: SAM picked the most salient
# object inside the GDINO box, which for portrait photos is the *person*, so
# the augmented photo only filled a person-shaped region instead of the
# rectangular paper).
#
# Strategy (in order of preference):
#   1) SAM 2.1 with `multimask_output=True`. SAM returns 3 candidates at
#      whole/part/subpart granularities; we pick the LARGEST area that fits
#      inside the GDINO box. If the largest covers ≥SAM_BOX_FILL_MIN of the
#      box area, accept it.
#   2) Canny + contour fallback: detect the paper's edge on the table
#      (high contrast, the table is light) and approximate to a convex
#      quadrilateral. Robust when SAM keeps locking onto the person.
#   3) Last resort: use the GDINO box itself as an axis-aligned rectangle.
#      This always succeeds (it's the input) but doesn't capture rotation.
#
# In all cases we return BOTH the contour mask (for diagnostics) AND the
# FILLED `minAreaRect` (rectangular by construction) which is what M_0
# ultimately becomes.

SAM_BOX_FILL_MIN = 0.35    # SAM mask must cover ≥35% of the GDINO box area.
                           # A rotated paper rectangle inscribed in its
                           # axis-aligned bbox has fill = cos²(angle), so a
                           # 45° rotated paper gives 0.50; we accept down to
                           # ~55° rotation to be safe.
SAM_IN_BOX_MIN = 0.95      # ≥95% of SAM mask pixels must lie inside the GDINO
                           # box — i.e., spillover must be ≤5%. Filters polluted
                           # "whole" candidates that grab paper+table or paper+
                           # gripper.
SAM_MULTIMASK = True       # request all 3 SAM candidates so we can rank them.

def sam_box_to_mask(image_predictor, frame_bgr: np.ndarray, box) -> tuple[np.ndarray, float, int]:
    """Return (boolean mask, box_fill_fraction, idx_of_chosen).

    v10: keeps SAM's 3-candidate "argmax(area)" ranking (which empirically
    picks the paper-as-whole when the GDINO box is set to the paper —
    SAM's candidates are nested whole/part/sub-part, so the largest valid
    candidate IS the paper). Adds a new spillover gate (in_box_frac ≥
    SAM_IN_BOX_MIN) to reject "whole" candidates that bleed into the
    surrounding table — these were the F1 failure mode (defect #1 in the
    2026-05-13 bottleneck diagnosis).

    Why area-argmax and not iou_pred-argmax:
      - SAM 2's pred_iou score rates mask *quality* (boundary sharpness,
        coherence), NOT which candidate is "paper" vs "face within photo".
        Empirically on swift_OLS_ep01 (smoke test 2026-05-13), iou-pred
        argmax picked the face-content candidate over the paper
        (verification dropped from cos=0.399 to cos=-0.058).
      - SAM 1's AMG uses pred_iou_thresh=0.88 as a *filter*, then a
        custom NMS — it doesn't rank candidates by score for instance
        selection. We use the spillover gate analogously: filter, don't
        re-rank.
      - First principles: for a paper inscribed in a near-tight GDINO box,
        paper_area ≈ box_area · cos²(rotation). The "face content"
        candidate is intrinsically smaller. Largest-passing-spillover =
        paper, never face-content.

    Sources for the spillover gate:
      - facebookresearch/segment-anything automatic_mask_generator.py
        (stability_score_thresh=0.95 — analogous quality filter)
      - IDEA-Research/Grounded-SAM-2 grounded_sam2_local_demo.py
        (uses multimask_output=False to skip selection entirely)
      - First-principles: in_box_frac = mask∩box / mask_total quantifies
        how much SAM "leaked" outside the box; ≥0.95 = ≤5% leak.
    """
    image_predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    masks, scores, _ = image_predictor.predict(
        box=np.array(box[:4], dtype=np.float32),
        multimask_output=SAM_MULTIMASK,
    )
    if masks.ndim == 2:
        masks = masks[None, ...]
    scores = np.asarray(scores).ravel()
    x0, y0, x1, y1 = [int(v) for v in box[:4]]
    box_area = max(1, (x1 - x0) * (y1 - y0))
    H_img, W_img = frame_bgr.shape[:2]
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W_img, x1), min(H_img, y1)

    in_box_counts, fills, in_box_fracs = [], [], []
    for m in masks:
        in_box = int(m[y0c:y1c, x0c:x1c].sum())
        total = int(m.sum())
        in_box_counts.append(in_box)
        fills.append(in_box / box_area)
        in_box_fracs.append(in_box / max(total, 1))
    in_box_arr = np.asarray(in_box_fracs, dtype=np.float32)
    fills_arr = np.asarray(fills, dtype=np.float32)
    counts_arr = np.asarray(in_box_counts, dtype=np.int64)

    valid = (fills_arr >= SAM_BOX_FILL_MIN) & (in_box_arr >= SAM_IN_BOX_MIN)
    if valid.any():
        cand = np.where(valid)[0]
        idx = int(cand[np.argmax(counts_arr[cand])])  # largest among non-spilled
    else:
        idx = int(np.argmax(counts_arr))  # largest overall (legacy behaviour)
        print(
            f"[WARN] sam_no_valid_candidate: expected=fill>={SAM_BOX_FILL_MIN} "
            f"AND in_box_frac>={SAM_IN_BOX_MIN}, "
            f"got=fills={fills_arr.round(2).tolist()} "
            f"in_box={in_box_arr.round(2).tolist()} "
            f"scores={scores.round(3).tolist()}, "
            f"fallback=argmax_area(idx={idx})",
            flush=True,
        )
    return masks[idx].astype(bool), float(fills_arr[idx]), idx


def find_paper_quadrilateral_canny(frame_bgr: np.ndarray, box, pad: int = 12) -> np.ndarray | None:
    """Edge-based paper detection inside the GDINO box. Returns a 4×2 contour
    (in full-frame coords) or None. We:
      - crop the box region with a small padding,
      - Canny + dilate to close gaps,
      - find the largest external contour,
      - approxPolyDP with growing epsilon until 4 vertices, OR fall back to
        the contour's minAreaRect.
    """
    H_img, W_img = frame_bgr.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box[:4]]
    x0p = max(0, x0 - pad); y0p = max(0, y0 - pad)
    x1p = min(W_img, x1 + pad); y1p = min(H_img, y1 + pad)
    if x1p - x0p < 20 or y1p - y0p < 20:
        return None
    crop = frame_bgr[y0p:y1p, x0p:x1p]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    crop_area = (x1p - x0p) * (y1p - y0p)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[:5]:
        area = cv2.contourArea(c)
        if area < 0.15 * crop_area:
            continue
        peri = cv2.arcLength(c, True)
        # Try several epsilon values until we get a 4-vertex approximation.
        for eps_frac in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10):
            approx = cv2.approxPolyDP(c, eps_frac * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(-1, 2).astype(np.float32)
                pts += np.array([x0p, y0p], dtype=np.float32)
                return pts
        # No 4-vertex match — accept the contour's minAreaRect.
        rect = cv2.minAreaRect(c)
        pts = cv2.boxPoints(rect).astype(np.float32)
        pts += np.array([x0p, y0p], dtype=np.float32)
        return pts
    return None


def box_to_axis_aligned_corners(box) -> np.ndarray:
    """Final fallback: GDINO box as an axis-aligned quadrilateral. Always
    returns 4 corners; never captures rotation."""
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


# ─── Mask → ordered 4-corner quadrilateral ───────────────────────────────────
def _reorder_corners_face_aware(box4x2: np.ndarray, face_center) -> np.ndarray:
    """Re-anchor adjacency-preserving 4-corner output so TL→TR is the SHORT
    edge nearer the face. Robust at all paper tilts (the geometric
    midpoint-y heuristic in `_order_tl_tr_br_bl` flips at θ ≈ atan(H/W)
    for non-square papers, putting the LONG edge into TL→TR and triggering
    auto-rotate to spin the photo 90° → sideways face).

    Inputs:
        box4x2      : (4,2) adjacency-preserving corners (e.g. from
                      _order_tl_tr_br_bl or directly cv2.boxPoints)
        face_center : (fx, fy) in the same image-frame coords as box4x2
    """
    pts = np.asarray(box4x2, dtype=np.float32).copy()
    if pts.shape != (4, 2):
        raise ValueError(f"_reorder_corners_face_aware: expected (4,2), got {pts.shape}")
    fc = np.asarray(face_center, dtype=np.float32).reshape(2)
    nxt = np.roll(pts, -1, axis=0)
    edge_lens = np.linalg.norm(nxt - pts, axis=1)
    edge_mids = (pts + nxt) * 0.5
    short_len = float(edge_lens.min())
    short_mask = edge_lens <= short_len * 1.01
    short_idx = np.where(short_mask)[0]
    dists = np.linalg.norm(edge_mids[short_idx] - fc, axis=1)
    top_edge_idx = int(short_idx[int(np.argmin(dists))])
    e0, e1 = pts[top_edge_idx], nxt[top_edge_idx]
    tl_idx = top_edge_idx if e0[0] <= e1[0] else (top_edge_idx + 1) % 4
    rolled = np.roll(pts, -tl_idx, axis=0)
    x, y = rolled[:, 0], rolled[:, 1]
    if float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) < 0:
        rolled = rolled[[0, 3, 2, 1]]
    return rolled.astype(np.float32)


def _order_tl_tr_br_bl(box4x2: np.ndarray) -> np.ndarray:
    """Reorder 4 corners (adjacency-preserving input, e.g. cv2.boxPoints)
    into TL, TR, BR, BL traversal in image coords (y grows down).

    Old impl used ideal angles ±45°/±135° from centroid; for papers tilted
    near 45° the four corners sit at the cardinals (0°/±90°/180°), and the
    greedy nearest-ideal match put diagonally-opposite corners into
    adjacent slots — producing a self-intersecting (bowtie) polygon. Warp
    into a bowtie quadrilateral renders as two solid triangles meeting at
    the center (Obama-as-black-diamond bug, 2026-05-11).

    Fix: pick the actual "top" edge as the one with smallest midpoint y,
    then TL is its left endpoint. Adjacency comes for free from
    cv2.boxPoints; we just rotate to start at TL and flip if traversal is
    CCW in image coords."""
    pts = np.asarray(box4x2, dtype=np.float32).copy()
    if pts.shape != (4, 2):
        raise ValueError(f"_order_tl_tr_br_bl: expected (4,2), got {pts.shape}")
    nxt = np.roll(pts, -1, axis=0)
    edge_mid_y = (pts[:, 1] + nxt[:, 1]) * 0.5
    top_edge_idx = int(np.argmin(edge_mid_y))
    e0, e1 = pts[top_edge_idx], nxt[top_edge_idx]
    tl_idx = top_edge_idx if e0[0] <= e1[0] else (top_edge_idx + 1) % 4
    rolled = np.roll(pts, -tl_idx, axis=0)
    x, y = rolled[:, 0], rolled[:, 1]
    signed_area_2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if signed_area_2 < 0:
        rolled = rolled[[0, 3, 2, 1]]
    return rolled.astype(np.float32)


def mask_to_corners_and_filled_rect(
    mask: np.ndarray, frame_shape: tuple
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Returns (corners 4×2 ordered TL/TR/BR/BL, filled_rect_mask uint8) where
    filled_rect_mask is the minAreaRect of the input mask FILLED — i.e.
    rectangular by construction. The filled-rect mask is what M_0 should be:
    we want the augmented photo to fill the FULL paper rectangle, not just
    whatever SAM happened to segment.

    NOTE (2026-05-13 bottleneck diagnosis, defect #3): minAreaRect inherits
    SAM-mask noise. A morph-open here was tried (v10 first attempt) but did
    not help on the dominant failure case (swift_OLS_ep01) because that
    case's oversizing is upstream of SAM — a too-generous GroundingDINO
    box that SAM faithfully fills. The right fix is a classical sub-pixel
    rectangle refit constrained to the paper boundary (planned next).
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 200: return None, None
    corners = _order_tl_tr_br_bl(cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32))
    filled = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(filled, corners.astype(np.int32), 1)
    return corners, filled


def corners_to_filled_rect(corners: np.ndarray, frame_shape: tuple) -> np.ndarray:
    """Fill a 4-corner quadrilateral to produce a rectangular M_0."""
    filled = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(filled, corners.astype(np.int32), 1)
    return filled


# v10: classical sub-pixel rectangle refinement (Hough+contrast+subpix corner).
# Imported lazily so existing imports still work if the module is missing.
try:
    from refine_paper_quad import refine_paper_quad_to_edges
    _REFINE_AVAILABLE = True
except ImportError:
    try:
        from .refine_paper_quad import refine_paper_quad_to_edges
        _REFINE_AVAILABLE = True
    except (ImportError, ValueError):
        # Fallback: load by file path (we're often run as a script, not a module)
        import importlib.util as _ilu_refine
        _spec_r = _ilu_refine.spec_from_file_location(
            "refine_paper_quad",
            str(Path(__file__).resolve().parent / "refine_paper_quad.py"),
        )
        if _spec_r is not None:
            _mod_r = _ilu_refine.module_from_spec(_spec_r)
            _spec_r.loader.exec_module(_mod_r)
            refine_paper_quad_to_edges = _mod_r.refine_paper_quad_to_edges
            _REFINE_AVAILABLE = True
        else:
            _REFINE_AVAILABLE = False
            print("[WARN] refine_paper_quad_import_failed: expected=eval_3/aug/"
                  "refine_paper_quad.py, got=not_loadable, fallback=skip_refinement",
                  flush=True)

REFINE_PAPER_EDGES = True  # toggle classical sub-pixel refit on the paper quad


def find_paper_mask_for_box(image_predictor, frame_bgr: np.ndarray, box,
                              refit_debug_dir = None):
    """Apply the v7 strategy: SAM multimask → Canny → box-axis-aligned,
    then a v10 classical sub-pixel rectangle refit on the result.

    Returns: (corners 4×2, filled_rect_mask uint8, source: str)
    where `source` ∈ {"sam_multimask", "canny_edges", "gdino_box"} plus an
    optional "+edge_refined" suffix when the v10 refit succeeded.
    The filled_rect_mask is what should be stored as M_0.

    If `refit_debug_dir` is set, the refit module saves per-step diagnostic
    PNGs there (01_band_and_coarse_quad.png, 02_canny_edges_in_band.png,
    03_hough_lines.png, 04_oriented_lines_and_selected_sides.png,
    05_final_refined_vs_coarse.png)."""
    H_img, W_img = frame_bgr.shape[:2]

    # Tier 1: SAM multimask.
    sam_mask, fill, idx = sam_box_to_mask(image_predictor, frame_bgr, box)
    coarse_corners: np.ndarray | None = None
    coarse_filled: np.ndarray | None = None
    source: str = ""
    coarse_mask_for_refine: np.ndarray | None = None
    if fill >= SAM_BOX_FILL_MIN:
        corners, filled = mask_to_corners_and_filled_rect(sam_mask, frame_bgr.shape)
        if corners is not None:
            coarse_corners, coarse_filled = corners, filled
            coarse_mask_for_refine = sam_mask.astype(np.uint8)
            source = f"sam_multimask[idx={idx},fill={fill:.2f}]"
    # Tier 2: Canny.
    if coarse_corners is None:
        quad = find_paper_quadrilateral_canny(frame_bgr, box)
        if quad is not None:
            ordered = _order_tl_tr_br_bl(quad)
            coarse_corners = ordered
            coarse_filled = corners_to_filled_rect(ordered, frame_bgr.shape)
            print(f"[WARN] paper_mask_fallback: expected=sam_fill>={SAM_BOX_FILL_MIN}, "
                  f"got=sam_fill={fill:.2f}, fallback=canny_edges", flush=True)
            source = f"canny_edges (sam_fill={fill:.2f})"
    # Tier 3: axis-aligned GDINO box.
    if coarse_corners is None:
        coarse_corners = box_to_axis_aligned_corners(box)
        coarse_filled = corners_to_filled_rect(coarse_corners, frame_bgr.shape)
        print(f"[WARN] paper_mask_fallback: expected=sam_or_canny_paper_quad, "
              f"got=neither_succeeded, fallback=gdino_box_axis_aligned "
              f"(sam_fill={fill:.2f})", flush=True)
        source = f"gdino_box_axis_aligned (sam_fill={fill:.2f})"

    # v10 sub-pixel rectangle refit (snap each side to the true paper edge).
    # Skips silently when the module isn't loadable; logs [WARN] when the
    # refit fails any sanity gate.
    if REFINE_PAPER_EDGES and _REFINE_AVAILABLE:
        refined = refine_paper_quad_to_edges(
            frame_bgr, coarse_corners,
            sam_mask=coarse_mask_for_refine,
            verbose=True,
            debug_dir=refit_debug_dir,
        )
        if refined is not None:
            refined_filled = corners_to_filled_rect(refined, frame_bgr.shape)
            return refined, refined_filled, f"{source}+edge_refined"

    return coarse_corners, coarse_filled, source


# ─── ArcFace identification ──────────────────────────────────────────────────
def build_celeb_prototypes(face_app, photo_bank_root: Path, celebs: list[str]) -> dict[str, np.ndarray]:
    protos = {}
    for celeb in celebs:
        d = photo_bank_root / celeb
        if not d.is_dir(): continue
        embs = []
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"} or p.name.startswith("__"):
                continue
            img = cv2.imread(str(p))
            if img is None: continue
            faces = face_app.get(img)
            if not faces: continue
            faces.sort(key=lambda f: -((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
            embs.append(faces[0].normed_embedding)
        if embs:
            m = np.stack(embs).mean(0)
            m /= max(np.linalg.norm(m), 1e-9)
            protos[celeb] = m
    return protos


def identify_portraits(face_app, frame_bgr: np.ndarray, corners_per_pid: dict[int, np.ndarray],
                       protos: dict[str, np.ndarray]) -> list[dict]:
    """For each portrait_id, find the face whose centre lies inside its corners
    polygon and return a dict per pid with:
        {best: str, score: float, all_scores: {celeb: cosine}, found: bool}
    Searches at 2× upscale because faces in the portrait region are typically
    only ~80 px wide at native scale (below ArcFace's 112 px training size)."""
    UP = 2
    big = cv2.resize(frame_bgr, (frame_bgr.shape[1]*UP, frame_bgr.shape[0]*UP),
                     interpolation=cv2.INTER_LANCZOS4)
    faces = face_app.get(big)
    centres = [(float((f.bbox[0]+f.bbox[2])/2/UP), float((f.bbox[1]+f.bbox[3])/2/UP),
                f.normed_embedding) for f in faces]
    out = []
    for pid in (0, 1, 2):
        corners = corners_per_pid[pid].astype(np.float32)
        found = False; best = "?"; best_s = 0.0; all_scores = {}
        face_center = None
        for cx, cy, emb in centres:
            if cv2.pointPolygonTest(corners, (cx, cy), False) >= 0:
                all_scores = {c: float(emb @ p) for c, p in protos.items()}
                best = max(all_scores, key=all_scores.get)
                best_s = all_scores[best]
                found = True
                face_center = (cx, cy)
                break
        out.append({"best": best, "score": best_s, "all_scores": all_scores,
                    "found": found, "face_center": face_center})
    return out


# ─── Per-frame OBJECT detection — LAB classifier (v7.1) ──────────────────────
# Two binary criteria, OR'd together; everything else is paper-with-shadow.
#
#   (a) chroma_diff > CHROMA_T
#       Frame i differs from frame 0 in hue/saturation. Catches colored
#       objects: red cola label, skin-tone hand, any object whose pigment
#       differs from the paper-or-print underneath.
#
#   (b) (L_i < L_DARK_T)  AND  (L_0 > L_LIGHT_T)
#       Current pixel is DARK and the reference pixel was BRIGHT. Catches
#       solid dark objects on a lighter surface — the gripper on white
#       paper, dark wires casting across a portrait edge. Crucially this
#       does NOT trigger for:
#         - shadows on bright paper (L_i stays > L_DARK_T even under deep
#           shadow; paper at L_0=200 only drops to ~130 in shadow),
#         - dark photo content (Swift's hair, suit lapels — L_0 already low,
#           so the second clause `L_0 > L_LIGHT_T` is false).
#
# Why v7's prior `|L_i - L_0| > LUM_OBJECT_T` criterion was wrong:
#   Strong shadows on bright paper (the cola can casts a sharp shadow on
#   the table-side of each portrait) routinely produce |ΔL| ≈ 70-100. With
#   LUM_OBJECT_T=55 these shadow pixels were mis-classified as "object" →
#   alpha=0 → the original photo showed through everywhere a shadow fell.
#   The user reported this as: "the augmented image gets transparent and
#   the real photo beneath starts to appear". The new criterion lets all
#   shadow pixels through; stage 4's L_i/L_0 luminance modulation then
#   darkens the new photo to match the shadow visually.
#
# Thresholds (OpenCV LAB: L,a,b each in [0,255]):
#   CHROMA_T   = 18   ≈ ΔE76 7 in standard [-128,127] LAB; "just noticeable"
#                     hue/sat shift on natural images (CIE76 recommendation).
#   L_DARK_T   = 70   ≈ 27 in standard L*; everything darker is "very dark"
#                     (Adobe RGB ~5% lightness). Gripper black ≈ L=30-50.
#   L_LIGHT_T  = 130  ≈ 51 in standard L*; paper at L=200 satisfies it; dark
#                     photo regions at L=30-50 don't.
CHROMA_T = 18
L_DARK_T = 70
L_LIGHT_T = 130

# v8.1: Cucchiara 2003 shadow detection (per-channel BGR ratio invariant).
# A pixel is SHADOW iff: r_min > SHADOW_ALPHA AND r_max < SHADOW_BETA AND ratio_spread < SHADOW_TAU.
# This is the *rescue* test — we use it to UN-EXCLUDE pixels that SAM 2 or the
# classifier wrongly marked as object but which are actually shadows on paper.
# Refs (cross-checked):
#   - Cucchiara, Grana, Piccardi, Prati 2003 (TPAMI 25:10) "Detecting Moving
#     Objects, Ghosts, and Shadows in Video Streams" §III — HSV: α ≤ V_i/V_0 ≤ β
#     with α≈0.4, β∈[0.6, 0.95]. The BGR per-channel form is equivalent
#     under scalar attenuation.
#   - Salvador, Cavallaro, Ebrahimi 2004 (CVIU 95) "Cast Shadow Segmentation"
#     — channels co-scale ⇒ ratio_spread small (recommended 0.10-0.15).
#   - Sanin, Sanderson, Lovell 2012 survey (arXiv 1304.1233) confirms the
#     scalar-attenuation criterion is the canonical physical model.
SHADOW_ALPHA = 0.40
SHADOW_BETA = 0.95
SHADOW_TAU = 0.12          # max-min of (B/B0, G/G0, R/R0)
SHADOW_CHROMA_MAX = 12     # chroma_diff must also be small (preserves hue)


def detect_object(
    frame_i: np.ndarray, frame_0: np.ndarray, M_0: np.ndarray,
    *,
    chroma_t: int = CHROMA_T,
    l_dark_t: int = L_DARK_T,
    l_light_t: int = L_LIGHT_T,
    pre_blur_sigma: float = 1.0,
    morph_radius: int = 2,
) -> np.ndarray:
    """v8.1 classifier — restored to v7.1 criteria (chroma + dark_on_bright).
    Bright/specular and asymmetric-channel cases are now caught by SAM 2 video
    tracking; the over-aggressive v7.2 criteria were causing shadow pixels in
    textured photo regions to be falsely classified as objects."""
    if pre_blur_sigma > 0:
        a = cv2.GaussianBlur(frame_i, (0, 0), sigmaX=pre_blur_sigma)
        b = cv2.GaussianBlur(frame_0, (0, 0), sigmaX=pre_blur_sigma)
    else:
        a, b = frame_i, frame_0
    lab_i = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    lab_0 = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
    L_i = lab_i[..., 0].astype(np.int16); a_i = lab_i[..., 1].astype(np.int16); b_i = lab_i[..., 2].astype(np.int16)
    L_0 = lab_0[..., 0].astype(np.int16); a_0 = lab_0[..., 1].astype(np.int16); b_0 = lab_0[..., 2].astype(np.int16)
    chroma_diff = np.sqrt((a_i - a_0) ** 2 + (b_i - b_0) ** 2).astype(np.int16)
    is_dark_object = (L_i < l_dark_t) & (L_0 > l_light_t)
    is_object = (chroma_diff > chroma_t) | is_dark_object
    is_object &= (M_0 > 0)
    if morph_radius > 0:
        k = np.ones((morph_radius * 2 + 1, morph_radius * 2 + 1), np.uint8)
        u8 = is_object.astype(np.uint8)
        u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, k)
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k)
        return u8.astype(bool)
    return is_object


def detect_shadow_cucchiara(
    frame_i: np.ndarray, frame_0: np.ndarray, M_0: np.ndarray,
    *,
    alpha: float = SHADOW_ALPHA,
    beta: float = SHADOW_BETA,
    tau: float = SHADOW_TAU,
    chroma_max: int = SHADOW_CHROMA_MAX,
    pre_blur_sigma: float = 1.0,
) -> np.ndarray:
    """v8.1: Cucchiara 2003 per-channel ratio shadow test.

    A pixel is SHADOW iff all four hold (within M_0):
      (1) chroma_diff < SHADOW_CHROMA_MAX  — hue preserved
      (2) min(B_i/B_0, G_i/G_0, R_i/R_0) > SHADOW_ALPHA  — not too dark
      (3) max(B_i/B_0, G_i/G_0, R_i/R_0) < SHADOW_BETA   — darker than reference
      (4) ratio_spread < SHADOW_TAU                       — channels co-scale

    Returns a boolean mask of shadow pixels. These should be RESCUED from the
    union of (SAM 2 occluder, classifier object) — i.e. kept as paper-with-
    shadow so the augmented photo gets pasted with luminance modulation."""
    if pre_blur_sigma > 0:
        a = cv2.GaussianBlur(frame_i, (0, 0), sigmaX=pre_blur_sigma)
        b = cv2.GaussianBlur(frame_0, (0, 0), sigmaX=pre_blur_sigma)
    else:
        a, b = frame_i, frame_0
    lab_i = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    lab_0 = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
    a_i = lab_i[..., 1].astype(np.int16); b_i = lab_i[..., 2].astype(np.int16)
    a_0 = lab_0[..., 1].astype(np.int16); b_0 = lab_0[..., 2].astype(np.int16)
    chroma_diff = np.sqrt((a_i - a_0) ** 2 + (b_i - b_0) ** 2)

    bi = a.astype(np.float32) + 1.0
    b0 = b.astype(np.float32) + 1.0
    ratios = bi / b0
    r_min = ratios.min(axis=2); r_max = ratios.max(axis=2)
    ratio_spread = r_max - r_min

    is_shadow = (
        (chroma_diff < chroma_max) &
        (r_min > alpha) &
        (r_max < beta) &
        (ratio_spread < tau) &
        (M_0 > 0)
    )
    return is_shadow


# ─── Per-episode driver ──────────────────────────────────────────────────────
def process_episode(
    ep_dir: Path,
    gd_proc: "AutoProcessor",
    gd_model: "AutoModelForZeroShotObjectDetection",
    image_predictor: "SAM2ImagePredictor",
    video_predictor,  # sam2.sam2_video_predictor.SAM2VideoPredictor (lazy-imported)
    face_app: "FaceAnalysis",
    celeb_protos: dict[str, np.ndarray],
    *,
    pre_blur_sigma: float = 1.0,
    morph_radius: int = 2,
    force: bool = False,
) -> dict:
    """Detect the three portrait quads in one teleop episode and persist them.

    Runs the static-camera v6 detection pipeline on the episode's first
    frame:

    1. GroundingDINO open-vocabulary detection finds candidate portrait
       boxes.
    2. SAM 2 (image predictor) tightens each candidate box into a binary
       mask.
    3. ``refine_paper_quad`` snaps the SAM-coarse mask to the sub-pixel
       paper edge via Canny + Hough + ``cornerSubPix``.
    4. ArcFace embedding for each refined quad is compared against
       ``celeb_protos`` to assign the celebrity identity.
    5. The frame-0 corners are persisted; per-frame masks are produced by
       lightweight change-detection against frame 0 (camera is static).

    Args:
        ep_dir: Path to the LeRobot v3 episode directory. Must contain
            ``videos/observation.images.camera1/chunk-*/file-*.mp4`` and
            ``reference.json`` with the layout sidecar.
        gd_proc: GroundingDINO processor
            (``transformers.AutoProcessor``-loaded).
        gd_model: GroundingDINO model
            (``AutoModelForZeroShotObjectDetection``).
        image_predictor: SAM 2 ``SAM2ImagePredictor`` for the frame-0
            mask refinement.
        video_predictor: SAM 2 ``SAM2VideoPredictor``. Despite the
            "static" name on this pipeline, the video predictor IS
            stepped across the full clip - by ``track_occluders_through_video``
            to seed SAM 2 with frame-0 (gripper / can) and frame-N-1
            (hand) boxes and propagate occluder masks. Only the *portrait
            paper quads* are camera-static; moving foreground objects
            still need per-frame masks.
        face_app: InsightFace ``FaceAnalysis`` instance used for ArcFace
            embedding of each refined portrait crop.
        celeb_protos: Mapping ``{celeb_slug: centroid_embedding}`` used
            to assign identity to each portrait by max cosine similarity.
        pre_blur_sigma: Gaussian sigma applied to frame 0 before the
            change-detection diff. Default 1.0 px.
        morph_radius: Half-side of the morphological-close kernel applied
            to the per-frame occluder mask. Default 2 px.
        force: If True, re-run even when the output JSON/pickle pair is
            already on disk. If False, skip and return ``{"skipped": True}``.

    Returns:
        A status dict. On success::

            {
                "ep":              <ep_dir.name>,
                "saved":           <str>,            # path to portrait_corners.json
                "n_frames":        int,
                "celebs":          list[str],        # celeb slug per portrait (pid 0/1/2)
                "id_cosines":      list[float],      # ArcFace cos per portrait, rounded to 3 dp
                "arcface_trusted": bool,             # min(id_cosines) >= ARCFACE_MIN_COS
            }

        On skipped re-run: ``{"ep": ..., "skipped": True}``.
        On any pre-flight failure (missing video, missing reference.json,
        SAM mask too small, etc.) the dict carries ``"error": <reason>``.

    Side effects:
        Writes three sibling files under ``ep_dir``:

        * ``portrait_corners.json`` - frame-0 (camera-static) quads.
        * ``portrait_masks.pkl``    - per-frame occluder masks (RLE).
        * ``portrait_seeds.json``   - GD/SAM seeds + identity assignment.

        Plus diagnostic images when the corresponding debug branches
        fire: ``dbg_v7_midframe.png`` (always) and
        ``dbg_refit/pid*/...`` (when the sub-pixel paper-quad refit runs).
    """
    out_corners_json = ep_dir / "portrait_corners.json"
    out_masks_pkl = ep_dir / "portrait_masks.pkl"
    out_seeds_json = ep_dir / "portrait_seeds.json"
    if not force and out_corners_json.is_file() and out_masks_pkl.is_file():
        return {"ep": ep_dir.name, "skipped": True}

    cands = list(ep_dir.glob("videos/*/chunk-*/file-*.mp4"))
    if not cands:
        return {"ep": ep_dir.name, "error": "no video"}
    video = cands[0]
    print(f"\n  → {ep_dir.name}")

    # Frame 0 (also the only frame SAM 2 will see)
    frame_0 = read_frame_zero(video)
    H, W = frame_0.shape[:2]

    # Read layout sidecar
    layout_celebs = None
    ref = json.loads((ep_dir / "reference.json").read_text())
    layout = ref.get("layout", "-")
    i2k = {"S": "swift", "O": "obama", "L": "lecun"}
    if len(layout) == 3 and all(c in i2k for c in layout):
        layout_celebs = [i2k[c] for c in layout]

    # 1. GroundingDINO → boxes; then per-box paper-mask finder (multi-strategy)
    print("    GroundingDINO + paper-rectangle detection on frame 0...", end=" ")
    t0 = time.time()
    boxes = detect_portraits_grounding_dino(gd_proc, gd_model, frame_0)
    if len(boxes) < 3:
        return {"ep": ep_dir.name, "error": f"only {len(boxes)} portraits detected"}
    box_centres = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]
    pid_order = sorted(range(3), key=lambda i: box_centres[i][0])

    # 2. Per-pid corners (4×2) and M_0 (FILLED rotated rectangle).
    #    M_0 is always rectangular by construction (filled minAreaRect), so the
    #    augmented photo fills the full paper region regardless of whether SAM
    #    happened to grab only the person.
    initial_corners: dict[int, np.ndarray] = {}
    M0_per_pid: dict[int, np.ndarray] = {}
    mask_sources: dict[int, str] = {}
    for pid in range(3):
        original_idx = pid_order[pid]
        refit_dbg = ep_dir / "dbg_refit" / f"pid{pid}"
        corners, filled, source = find_paper_mask_for_box(
            image_predictor, frame_0, boxes[original_idx],
            refit_debug_dir=refit_dbg,
        )
        initial_corners[pid] = corners
        M0_per_pid[pid] = filled
        mask_sources[pid] = source
    print(f"({time.time()-t0:.1f}s)")
    print(f"    mask sources: pid0={mask_sources[0]}, pid1={mask_sources[1]}, pid2={mask_sources[2]}")

    # 3. ArcFace identification.
    # Identification policy (validated empirically on ep04):
    #   - ArcFace at 2× upscale on the full frame is RELIABLE here: when all 3
    #     faces are detected inside their portrait polygons and each best-match
    #     cosine ≥ ARCFACE_MIN_COS (we use 0.40 — the buffalo_l "same-identity"
    #     threshold from the InsightFace docs and the ArcFace paper §4.3), use
    #     ArcFace. Each cosine independently exceeding 0.40 against three
    #     orthogonal prototypes is a strong joint signal.
    #   - Otherwise, fall back to the layout sidecar — the operator-recorded
    #     order. But the sidecar can be wrong (ep04: operator typed "SOL" but
    #     placed L-S-O), so we log a [WARN] and surface the disagreement.
    ARCFACE_MIN_COS = 0.40
    ARCFACE_MIN_GAP = 0.15        # v9.3: best - 2nd-best ≥ this is "confident enough"
                                  # even if absolute cosine < ARCFACE_MIN_COS.
                                  # Justification: on ep01, swift was correctly
                                  # identified at cos=0.366 but rejected by the
                                  # absolute threshold; the 2nd-best (obama=0.151)
                                  # was 0.215 below — an unambiguous win. The
                                  # gap test catches these confident-but-low
                                  # matches without admitting noise.
    print("    ArcFace identify each portrait...", end=" ")
    t0 = time.time()
    id_results = identify_portraits(face_app, frame_0, initial_corners, celeb_protos)
    arcface_celebs = [r["best"] for r in id_results]
    arcface_cosines = [r["score"] for r in id_results]
    all_found = all(r["found"] for r in id_results)
    def _conf(r):
        scores_sorted = sorted(r["all_scores"].values(), reverse=True)
        gap = scores_sorted[0] - scores_sorted[1] if len(scores_sorted) > 1 else scores_sorted[0]
        return r["score"] >= ARCFACE_MIN_COS or gap >= ARCFACE_MIN_GAP
    all_above = all(_conf(r) for r in id_results)
    arcface_unique = len(set(arcface_celebs)) == 3 and "?" not in arcface_celebs
    arcface_trusted = all_found and all_above and arcface_unique
    print(f"({time.time()-t0:.1f}s, trust={arcface_trusted}, "
          f"order={list(zip(arcface_celebs, [round(c,3) for c in arcface_cosines]))})")

    if arcface_trusted:
        pid_to_celeb: dict[str, str] = {str(i): arcface_celebs[i] for i in range(3)}
        if layout_celebs is not None and arcface_celebs != layout_celebs:
            print(f"[WARN] arcface_vs_layout: expected={layout_celebs}, "
                  f"got={arcface_celebs}, fallback=trust_arcface "
                  f"(cosines all >= {ARCFACE_MIN_COS} → ArcFace is reliable)", flush=True)
    elif layout_celebs is not None:
        pid_to_celeb = {str(i): layout_celebs[i] for i in range(3)}
        print(f"[WARN] arcface_unreliable: expected_all_cos>={ARCFACE_MIN_COS}_and_unique, "
              f"got={list(zip(arcface_celebs, [round(c,3) for c in arcface_cosines]))}, "
              f"fallback=trust_layout_sidecar={layout_celebs}", flush=True)
    else:
        return {"ep": ep_dir.name,
                "error": f"ArcFace unreliable ({list(zip(arcface_celebs, arcface_cosines))}) "
                         f"and no layout sidecar"}

    # 3b. Face-aware corner re-anchoring. The detected face's centre is the
    # only ground-truth signal for "which short edge of the paper is the
    # top of the photo". Without it, the geometric heuristic flips above
    # ~atan(H/W) ≈ 56° of tilt, which sends auto-rotate into a 90° photo
    # spin and the augmented face ends up sideways.
    n_reanchored = 0
    for pid in range(3):
        fc = id_results[pid].get("face_center")
        if fc is None:
            print(f"[WARN] face_center_missing pid={pid}: expected=face_inside_polygon, "
                  f"got=None, fallback=keep_geometric_TL", flush=True)
            continue
        initial_corners[pid] = _reorder_corners_face_aware(initial_corners[pid], fc)
        n_reanchored += 1
    print(f"    face-aware corner reorder applied to {n_reanchored}/3 portraits.")

    # 4. Persist seeds
    seeds_data = {
        "points": [[int(box_centres[pid_order[pid]][0]), int(box_centres[pid_order[pid]][1])]
                   for pid in range(3)],
        "labels": [1, 1, 1],
        "detector": "GroundingDINO-tiny@disable_custom_kernels",
        "boxes_xyxy": [[float(v) for v in boxes[pid_order[pid]][:4]] for pid in range(3)],
        "box_scores": [float(boxes[pid_order[pid]][4]) for pid in range(3)],
        "celebs": [pid_to_celeb[str(i)] for i in range(3)],
        "arcface_best":    [id_results[i]["best"]      for i in range(3)],
        "arcface_cosines": [id_results[i]["score"]     for i in range(3)],
        "arcface_full":    [id_results[i]["all_scores"] for i in range(3)],
        "arcface_trusted": arcface_trusted,
        "layout_celebs":   layout_celebs,
        "pipeline_version": "v8.1_grounded_sam2_shadow_rescue",
        "mask_sources": mask_sources,
        "params": {
            "chroma_t": CHROMA_T,
            "l_dark_t": L_DARK_T,
            "l_light_t": L_LIGHT_T,
            "pre_blur_sigma": pre_blur_sigma,
            "morph_radius": morph_radius,
        },
    }
    out_seeds_json.write_text(json.dumps(seeds_data, indent=2))

    # Save frame_0 for stage 4's shadow-aware luminance modulation.
    cv2.imwrite(str(ep_dir / "frame_0.png"), frame_0)

    # 4.5. v8: Grounded-SAM-2 occluder tracking through the video.
    #   - Grounding DINO on frame 0 + last frame → bounding boxes for
    #     {gripper, can, hand}. Frame 0 catches gripper+can (always
    #     present from t=0); last frame catches the hand (typically only
    #     appears later when the operator interacts).
    #   - Filter out false positives whose box centers fall inside a paper
    #     mask (Grounding DINO sometimes fires on red regions of portraits).
    #   - SAM 2.1 video predictor seeds boxes at the appropriate frame_idx
    #     and propagates forward+backward through all frames.
    #   - Per-frame occluder mask = union of all tracked obj_id masks,
    #     dilated by OCCLUDER_DILATE_PX to seal thin gripper fingertip
    #     leakage (SAM 2 paper §6 known issue with thin/articulated).
    print("    Grounding DINO occluder detection on frame 0 + last frame...", end=" ")
    t0 = time.time()
    # Read last frame
    cap_last = cv2.VideoCapture(str(ensure_h264(video)))
    n_video_frames = int(cap_last.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_last.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_video_frames - 1))
    ok_last, frame_last = cap_last.read()
    cap_last.release()
    occluder_dets_per_frame: dict[int, list[dict]] = {}
    dets_frame0_raw = detect_occluders_grounding_dino(gd_proc, gd_model, frame_0)
    dets_frame0 = dets_frame0_raw
    # v10: at frame 0 the recording convention is "workspace clear of
    # occluders" — gripper above, no can placed yet. ANY GDINO detection
    # of {gripper, can, hand} that lands ≥20% inside a portrait paper at
    # frame 0 is photo content (Swift's hand in her photo, LeCun adjusting
    # glasses, red dress matching "coca cola can"). Drop aggressively.
    # Verified visually 2026-05-13 — at 0.60 the filter was letting Swift
    # photos through as "human hand" and SAM 2 propagated them across
    # the whole clip, covering Swift's face with a yellow occluder mask.
    dets_frame0 = filter_occluders_against_paper(
        dets_frame0, list(M0_per_pid.values()), max_in_paper_fraction=0.20,
        drop_if_center_in_paper=True,
    )
    if dets_frame0:
        occluder_dets_per_frame[0] = dets_frame0
    dets_last_raw: list[dict] = []
    if ok_last and n_video_frames > 1:
        dets_last_raw = detect_occluders_grounding_dino(gd_proc, gd_model, frame_last)
        # Last frame: keep 0.60 (lenient) because a real can sitting on the
        # target paper IS mostly inside that paper. We accept the chance of
        # photo-content FPs slipping through at this frame because the
        # frame-0 propagation already covers most of the trajectory.
        dets_last = filter_occluders_against_paper(
            dets_last_raw, list(M0_per_pid.values()), max_in_paper_fraction=0.60
        )
        already = {d["label"].lower() for d in dets_frame0}
        new_last = [d for d in dets_last if d["label"].lower() not in already]
        if new_last:
            occluder_dets_per_frame[n_video_frames - 1] = new_last
    n_seeds = sum(len(v) for v in occluder_dets_per_frame.values())
    print(f"({time.time()-t0:.1f}s, {n_seeds} occluder seeds: "
          f"{[(fi, [d['label'] for d in v]) for fi, v in sorted(occluder_dets_per_frame.items())]})")

    # v10: save debug PNGs of GroundingDINO occluder detections (KEPT in green,
    # DROPPED in red) on frame 0 and last frame, with portrait outlines for
    # context. So we can SEE every bbox that's being considered.
    def _draw_occluder_dbg(img, raw_dets, kept_dets, m0_per_pid, label_title):
        out = img.copy()
        kept_keys = {(tuple(round(v, 1) for v in d["box"]), d["label"]) for d in kept_dets}
        # portrait outlines
        for pid_i, M in m0_per_pid.items():
            cnts, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)
            ys, xs = np.where(M > 0)
            if xs.size > 0:
                cv2.putText(out, f"pid{pid_i}", (int(xs.min()) + 4, int(ys.min()) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        for d in raw_dets:
            x0, y0, x1, y1 = [int(v) for v in d["box"]]
            box_key = (tuple(round(v, 1) for v in d["box"]), d["label"])
            kept = box_key in kept_keys
            color = (0, 255, 0) if kept else (0, 0, 255)
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
            cx = (x0 + x1) // 2; cy = (y0 + y1) // 2
            cv2.circle(out, (cx, cy), 4, color, -1)
            tag = ("[KEPT] " if kept else "[DROP] ") + f"{d['label']} s={d['score']:.2f}"
            cv2.rectangle(out, (x0, max(0, y0 - 18)),
                          (x0 + 9 * len(tag) + 4, y0), color, -1)
            cv2.putText(out, tag, (x0 + 2, max(13, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(out, label_title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out

    dbg_frame0 = _draw_occluder_dbg(
        frame_0, dets_frame0_raw, dets_frame0, M0_per_pid,
        f"GroundingDINO occluders @ frame 0  GREEN=kept  RED=dropped  "
        f"(raw n={len(dets_frame0_raw)}, kept n={len(dets_frame0)})",
    )
    cv2.imwrite(str(ep_dir / "dbg_occluder_detections_frame0.png"), dbg_frame0)
    if ok_last and n_video_frames > 1:
        dbg_last = _draw_occluder_dbg(
            frame_last, dets_last_raw,
            occluder_dets_per_frame.get(n_video_frames - 1, []),
            M0_per_pid,
            f"GroundingDINO occluders @ frame {n_video_frames - 1}  GREEN=kept (as new)  "
            f"RED=dropped  (raw n={len(dets_last_raw)})",
        )
        cv2.imwrite(str(ep_dir / "dbg_occluder_detections_last_frame.png"), dbg_last)

    occluder_masks_per_frame: dict[int, dict[int, np.ndarray]] = {}
    obj_id_to_label: dict[int, str] = {}
    if n_seeds > 0:
        print("    SAM 2.1 video predictor → propagating occluder masks...", end=" ")
        t0 = time.time()
        frame_dir = ensure_frame_dir(video)
        occluder_masks_per_frame, obj_id_to_label = track_occluders_through_video(
            video_predictor, frame_dir, n_video_frames, H, W, occluder_dets_per_frame,
        )
        print(f"({time.time()-t0:.1f}s, "
              f"{sum(len(m) for m in occluder_masks_per_frame.values())} per-frame mask records)")
    else:
        print(f"[WARN] no_occluders_detected: expected=gripper_or_can_or_hand, "
              f"got=nothing_above_score_threshold_{OCCLUDER_BOX_SCORE_MIN}, "
              f"fallback=classifier_only", flush=True)

    # 5. Per-frame visible-paper masks (shadow-tolerant LAB classifier UNION
    # SAM 2 occluder mask).
    print("    per-frame combined occluder + LAB classifier → visible-paper masks...", end=" ")
    t0 = time.time()
    cap = cv2.VideoCapture(str(ensure_h264(video)))
    masks_per_frame: dict[int, dict[int, dict]] = {}
    n_frames = 0
    dilate_k = np.ones((OCCLUDER_DILATE_PX * 2 + 1, OCCLUDER_DILATE_PX * 2 + 1), np.uint8)
    while True:
        ok, frame_i = cap.read()
        if not ok: break
        # Union all SAM 2 occluder masks at this frame, dilate to seal edges
        sam2_occluder = np.zeros((H, W), dtype=np.uint8)
        for m in occluder_masks_per_frame.get(n_frames, {}).values():
            sam2_occluder |= m.astype(np.uint8)
        if OCCLUDER_DILATE_PX > 0 and sam2_occluder.any():
            sam2_occluder = cv2.dilate(sam2_occluder, dilate_k, iterations=1)
        sam2_occluder_bool = sam2_occluder > 0
        for pid in range(3):
            cls_obj = detect_object(
                frame_i, frame_0, M0_per_pid[pid],
                chroma_t=CHROMA_T,
                l_dark_t=L_DARK_T,
                l_light_t=L_LIGHT_T,
                pre_blur_sigma=pre_blur_sigma,
                morph_radius=morph_radius,
            )
            # v9: simple union (drop the Cucchiara shadow rescue).
            # Per the LIBERO-Plus / Pumacay decomposition, lighting/shadow is
            # the smallest-impact perturbation axis for VLAs (8-11 pp vs 45-72
            # pp for camera/state). The entire VLA augmentation field (ROSIE,
            # GenAug, CACTI, RoboEngine, OpenVLA, LeRobot defaults) skips
            # shadow modeling. Stage 4 applies a single scalar lum match
            # instead of per-pixel — see render_variant().
            combined_object = cls_obj | sam2_occluder_bool
            visible_paper = (M0_per_pid[pid] > 0) & (~combined_object)
            visible_u8 = visible_paper.astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(visible_u8))
            score = float(visible_paper.sum()) / max(float((M0_per_pid[pid] > 0).sum()), 1)
            masks_per_frame.setdefault(n_frames, {})[pid] = {"rle": rle, "score": score}
        n_frames += 1
    cap.release()
    print(f"({time.time()-t0:.1f}s, {n_frames} frames)")

    # 6. Save corners (constant across frames) + per-frame visible-paper masks
    corners_data = {
        "video_shape": [H, W],
        "n_frames": n_frames,
        "pipeline_version": "v9.5_face_aware_corner_anchor",
        "face_centers": {
            str(pid): list(id_results[pid]["face_center"])
                       if id_results[pid].get("face_center") is not None else None
            for pid in range(3)
        },
        "portraits": {
            str(pid): {
                str(fi): {
                    "corners": initial_corners[pid].tolist(),
                    "occluded": masks_per_frame[fi][pid]["score"] < 0.5,
                    "score": masks_per_frame[fi][pid]["score"],
                    "interpolated": False,
                }
                for fi in range(n_frames)
            }
            for pid in range(3)
        },
    }
    out_corners_json.write_text(json.dumps(corners_data, indent=2))

    with open(out_masks_pkl, "wb") as f:
        pickle.dump({
            "video_path": str(video),
            "seeds": seeds_data,
            "masks": masks_per_frame,
            "M_0_per_pid": {pid: M0_per_pid[pid] for pid in range(3)},
            "pipeline_version": "v8.1_grounded_sam2_shadow_rescue",
        }, f)

    # Save a single-frame debug overlay showing M_0 + a mid-clip occluder demo
    cap = cv2.VideoCapture(str(ensure_h264(video)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)
    ok, mid_frame = cap.read()
    cap.release()
    if ok:
        dbg = mid_frame.copy()
        COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        for pid in range(3):
            payload = masks_per_frame[n_frames // 2][pid]
            visible = mask_util.decode(payload["rle"]).astype(np.uint8)
            if visible.ndim == 3: visible = visible[:, :, 0]
            tint = np.full_like(dbg, COLORS[pid])
            dbg[visible > 0] = (0.5 * dbg[visible > 0] + 0.5 * tint[visible > 0]).astype(np.uint8)
            occ = (M0_per_pid[pid] > 0) & (visible == 0)
            dbg[occ] = (0, 255, 255)                          # yellow = object
            # Draw the (constant) rectangular M_0 boundary
            pts = initial_corners[pid].astype(np.int32)
            cv2.polylines(dbg, [pts], True, COLORS[pid], 2)
        cv2.imwrite(str(ep_dir / "dbg_v7_midframe.png"), dbg)

    return {
        "ep": ep_dir.name, "saved": str(out_corners_json),
        "n_frames": n_frames,
        "celebs": [pid_to_celeb[str(i)] for i in range(3)],
        "id_cosines": [round(id_results[i]["score"], 3) for i in range(3)],
        "arcface_trusted": arcface_trusted,
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode_dir", nargs="?", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--pre-blur-sigma", type=float, default=1.0)
    p.add_argument("--morph-radius", type=int, default=2)
    p.add_argument("--ckpt", default=str(SAM2_CKPT_DEFAULT))
    p.add_argument("--cfg", default=SAM2_CFG)
    p.add_argument("--photo-bank", default=str(PHOTO_BANK_DEFAULT))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if (args.episode_dir is None) == (args.root is None):
        print("[ERROR] specify exactly one of: episode_dir, --root", file=sys.stderr); return 2

    _import_heavy()

    print("loading SAM 2.1 hiera-L (image predictor)...")
    t0 = time.time()
    sam = build_sam2(args.cfg, args.ckpt, device="cuda")
    image_predictor = SAM2ImagePredictor(sam)
    print(f"  ({time.time()-t0:.1f}s)")

    print("loading SAM 2.1 hiera-L (video predictor for occluders)...")
    t0 = time.time()
    video_predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device="cuda")
    print(f"  ({time.time()-t0:.1f}s)")

    print("loading GroundingDINO-tiny (disable_custom_kernels=True)...")
    t0 = time.time()
    gd_proc = AutoProcessor.from_pretrained(GDINO_MODEL)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_MODEL, disable_custom_kernels=True
    ).to("cuda").eval()
    print(f"  ({time.time()-t0:.1f}s)")

    print("loading InsightFace buffalo_l + building celeb prototypes...")
    t0 = time.time()
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    protos = build_celeb_prototypes(face_app, Path(args.photo_bank), ["swift", "obama", "lecun"])
    print(f"  ({time.time()-t0:.1f}s, prototypes for {sorted(protos.keys())})")

    eps = [Path(args.episode_dir)] if args.episode_dir else \
          sorted(p for p in Path(args.root).iterdir() if p.is_dir() and (p / "reference.json").is_file())

    results = []
    for ep_dir in eps:
        try:
            r = process_episode(
                ep_dir, gd_proc, gd_model, image_predictor, video_predictor, face_app, protos,
                pre_blur_sigma=args.pre_blur_sigma,
                morph_radius=args.morph_radius,
                force=args.force,
            )
        except Exception as e:
            r = {"ep": ep_dir.name, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        if "saved" in r:
            print(f"  ✓ {r['ep']:50s}  {r['n_frames']:>4} frames  ids={r['celebs']} cos={r['id_cosines']}")
        elif r.get("skipped"):
            print(f"  - {r['ep']:50s}  (skipped)")
        else:
            print(f"  ✗ {r['ep']:50s}  {r.get('error','?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
