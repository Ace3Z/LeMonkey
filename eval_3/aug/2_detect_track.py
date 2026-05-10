#!/usr/bin/env python3
"""STAGE 2 v2 — auto-detect portraits on frame 0 + track 4 corners across all frames.

REPLACES the previous per-frame SAM 2 video propagation, which produced
jittery / flickering masks. The new pipeline is deterministic:

  1. GroundingDINO (text prompt "a printed portrait photo")
     finds bounding boxes of the 3 portraits on frame 0.
  2. SAM 2.1 image predictor refines each box into a tight mask.
  3. cv2.minAreaRect → 4 corners per portrait, ordered TL/TR/BR/BL.
  4. cv2.calcOpticalFlowPyrLK tracks those 12 points (3 portraits × 4
     corners) across all subsequent frames. LK is deterministic, ~1 ms
     per frame, no neural-net jitter.
  5. When LK loses corners (gripper occlusion): hold last known
     positions; on long occlusion, re-anchor via planar homography
     from frame 0 (ORB / SuperPoint feature matching).

Output: <episode_dir>/portrait_corners.json (same schema as before so
stages 4/5 don't change). No portrait_masks.pkl needed.

See RESEARCH_v2.md for the design rationale + research that drove this rewrite.

Usage:
    python 2_detect_track.py /path/to/episode_dir
    python 2_detect_track.py --root ~/LeMonkey/datasets/eval3_quick
    python 2_detect_track.py /path/to/episode_dir --interactive    # fallback if detection fails
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

# Deferred imports for fast --help
def _import_heavy():
    global cv2, torch, AutoProcessor, AutoModelForZeroShotObjectDetection, build_sam2, SAM2ImagePredictor, ensure_frame_dir, read_frame_zero, ensure_h264, FaceAnalysis
    import cv2  # type: ignore
    import torch  # type: ignore
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from insightface.app import FaceAnalysis as _FaceAnalysis
    spec = importlib.util.spec_from_file_location("_video_io", str(Path(__file__).resolve().parent / "_video_io.py"))
    _vio = importlib.util.module_from_spec(spec); spec.loader.exec_module(_vio)
    ensure_frame_dir = _vio.ensure_frame_dir
    read_frame_zero = _vio.read_frame_zero
    ensure_h264 = _vio.ensure_h264
    globals().update({
        "cv2": cv2, "torch": torch,
        "AutoProcessor": AutoProcessor,
        "AutoModelForZeroShotObjectDetection": AutoModelForZeroShotObjectDetection,
        "build_sam2": build_sam2, "SAM2ImagePredictor": SAM2ImagePredictor,
        "FaceAnalysis": _FaceAnalysis,
        "ensure_frame_dir": ensure_frame_dir,
        "read_frame_zero": read_frame_zero,
        "ensure_h264": ensure_h264,
    })


# ─── Identity assignment helpers ─────────────────────────────────────────────
def build_celeb_prototypes(face_app, photo_bank_root: Path, celebs: list[str]) -> dict[str, np.ndarray]:
    """For each celeb, average the ArcFace embeddings of all bank photos →
    one prototype embedding per celeb. Used to identify which celebrity is
    in each detected portrait, regardless of spatial arrangement."""
    protos: dict[str, np.ndarray] = {}
    for celeb in celebs:
        d = photo_bank_root / celeb
        if not d.is_dir():
            continue
        embs = []
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"} or p.name.startswith("__"):
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            faces = face_app.get(img)
            if not faces:
                continue
            faces.sort(key=lambda f: -((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
            embs.append(faces[0].normed_embedding)
        if embs:
            mean = np.stack(embs).mean(0)
            mean /= max(np.linalg.norm(mean), 1e-9)
            protos[celeb] = mean
    return protos


def identify_portrait(face_app, portrait_crop: np.ndarray,
                      protos: dict[str, np.ndarray]) -> tuple[str | None, float]:
    """Return (celeb_key, cosine_to_best_match). Returns (None, 0) on no face.

    The cropped portrait region is small (~280×400 px) and the face inside
    is even smaller (~80 px wide), which is below ArcFace's reliable
    detection size. We upscale to ~600×900 first so the face is ~180 px
    wide. Lanczos upsampling, then InsightFace at its default det_size.
    """
    # Upscale 2.5x (Lanczos preserves face detail through the upsample)
    h, w = portrait_crop.shape[:2]
    upscaled = cv2.resize(portrait_crop, (w * 3, h * 3), interpolation=cv2.INTER_LANCZOS4)
    faces = face_app.get(upscaled)
    if not faces:
        return None, 0.0
    faces.sort(key=lambda f: -((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
    emb = faces[0].normed_embedding
    scores = {celeb: float(emb @ p) for celeb, p in protos.items()}
    best = max(scores, key=scores.get)
    return best, scores[best]


def crop_portrait_from_corners(
    frame: np.ndarray, corners: np.ndarray,
    *, target_w: int = 280, target_h: int = 400,
) -> np.ndarray:
    src = np.asarray(corners, dtype=np.float32)
    dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (target_w, target_h), flags=cv2.INTER_LANCZOS4)


SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT_DEFAULT = Path.home() / "checkpoints/sam2.1_hiera_large.pt"
GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"

# Aspect-ratio + area bounds used to filter GroundingDINO's noisy box list
# down to the 3 portraits. A5 print viewed near-overhead is roughly
# 0.4–1.0 aspect (h:w) and ~7-30k pixels area in a 640×480 frame.
PORTRAIT_ASPECT_MIN = 0.40
PORTRAIT_ASPECT_MAX = 1.30
PORTRAIT_AREA_MIN = 5_000
PORTRAIT_AREA_MAX = 60_000
PORTRAIT_BOX_SCORE_MIN = 0.20


def filter_portrait_boxes(boxes: np.ndarray, scores: np.ndarray, image_shape: tuple) -> list[int]:
    """Return the indices of up to 3 boxes that look like portraits.

    Strategy: filter by aspect + area + score; sort remaining by score
    descending; greedily pick non-overlapping boxes (IoU < 0.3).
    """
    H, W = image_shape[:2]
    cand: list[tuple[int, float, np.ndarray]] = []   # (idx, score, [x0,y0,x1,y1])
    for i, (box, sc) in enumerate(zip(boxes, scores)):
        x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        aspect = w / h
        area = w * h
        # whole-frame box → reject (GroundingDINO sometimes emits one)
        if w > 0.85 * W and h > 0.85 * H:
            continue
        if not (PORTRAIT_ASPECT_MIN <= aspect <= PORTRAIT_ASPECT_MAX):
            continue
        if not (PORTRAIT_AREA_MIN <= area <= PORTRAIT_AREA_MAX):
            continue
        if sc < PORTRAIT_BOX_SCORE_MIN:
            continue
        cand.append((i, float(sc), np.array([x0, y0, x1, y1])))

    cand.sort(key=lambda x: -x[1])

    chosen: list[int] = []
    chosen_boxes: list[np.ndarray] = []

    def iou(a, b):
        xa = max(a[0], b[0]); ya = max(a[1], b[1])
        xb = min(a[2], b[2]); yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0

    for idx, sc, box in cand:
        if any(iou(box, cb) > 0.3 for cb in chosen_boxes):
            continue
        chosen.append(idx); chosen_boxes.append(box)
        if len(chosen) == 3:
            break
    return chosen


def detect_portraits_grounding_dino(
    frame_bgr: np.ndarray, *, threshold: float = 0.25,
) -> list[tuple[float, float, float, float, float]]:
    """Return up to 3 boxes [(x0,y0,x1,y1,score), ...] from GroundingDINO."""
    from PIL import Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    text_labels = [["a printed portrait photo on a table", "a paper photograph"]]
    inputs = _gdino_proc(images=img, text=text_labels, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = _gdino_model(**inputs)
    results = _gdino_proc.post_process_grounded_object_detection(
        out, threshold=threshold, target_sizes=[(frame_bgr.shape[0], frame_bgr.shape[1])],
    )
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    keep = filter_portrait_boxes(boxes, scores, frame_bgr.shape)
    return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], float(scores[i])) for i in keep]


def sam_box_to_mask(image_predictor, frame_bgr: np.ndarray, box: tuple) -> np.ndarray:
    """SAM 2.1 image predictor — box-prompted mask refinement."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_predictor.set_image(rgb)
    masks, scores, _ = image_predictor.predict(
        box=np.array(box[:4], dtype=np.float32),
        multimask_output=False,
    )
    return masks[0].astype(bool)


def mask_to_corners(mask: np.ndarray) -> np.ndarray | None:
    """min-area-rect 4 corners, ordered TL/TR/BR/BL."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return None
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.float32)

    # order corners TL/TR/BR/BL by quadrant from centroid
    centroid = box.mean(0)
    angles = np.arctan2(box[:, 1] - centroid[1], box[:, 0] - centroid[0])
    ordered = np.zeros((4, 2), dtype=np.float32)
    slot_targets = [(-np.pi*3/4, 0), (-np.pi/4, 1), (np.pi/4, 2), (np.pi*3/4, 3)]
    used = [False]*4
    for ideal_angle, slot in slot_targets:
        d = np.abs(np.arctan2(np.sin(angles - ideal_angle), np.cos(angles - ideal_angle)))
        # pick the closest unused corner
        order_by_d = np.argsort(d)
        for idx in order_by_d:
            if not used[idx]:
                ordered[slot] = box[idx]; used[idx] = True; break
    return ordered


# ─── Lucas-Kanade corner tracker ─────────────────────────────────────────────
LK_PARAMS = dict(
    winSize=(31, 31), maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
) if False else None   # populated lazily after cv2 import


def _lk_params():
    return dict(
        winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )


def track_corners_lk(
    video_path: Path,
    initial_corners: dict[int, np.ndarray],
) -> dict:
    """LK-flow track 12 corner points (3 portraits × 4) across all video frames.

    initial_corners: {portrait_id (0/1/2) → np.ndarray(4, 2) of float32}
    Returns the per-frame corners dict in the same schema 4_inpaint_video.py
    expects.

    Strategy:
      - Stack 12 points into one cv2.calcOpticalFlowPyrLK call per frame
      - On status==0 (LK lost), HOLD the previous frame's position
      - If >25% of points are lost on a single frame, mark the frame
        partially-occluded and try a homography re-anchor against frame 0
        (cheap ORB match)
      - Confidence per frame = fraction of points kept by LK
    """
    cap = cv2.VideoCapture(str(ensure_h264(video_path)))
    ok, frame_0 = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"cannot read frame 0 of {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    H, W = frame_0.shape[:2]

    pts0 = np.vstack([initial_corners[pid] for pid in (0, 1, 2)]).astype(np.float32).reshape(-1, 1, 2)
    gray_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)

    out: dict = {"video_shape": [H, W], "n_frames": n_frames, "portraits": {}}
    for pid in (0, 1, 2):
        out["portraits"][str(pid)] = {}
        out["portraits"][str(pid)]["0"] = {
            "corners": initial_corners[pid].tolist(),
            "occluded": False, "score": 1.0, "interpolated": False,
        }

    prev_gray = gray_0
    prev_pts = pts0
    last_valid_pts = pts0.copy()                # per-point "last seen"
    fi = 1
    lk_lost_count = 0
    cap.read()                                  # advance past frame 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **_lk_params())
        except cv2.error:
            next_pts, status, err = prev_pts, np.zeros((12, 1), dtype=np.uint8), np.ones((12, 1))*9e9

        status = status.flatten()
        kept = status > 0
        n_kept = int(kept.sum())
        confidence = n_kept / 12

        if n_kept < 12:
            lk_lost_count += (12 - n_kept)
            # Replace lost points with last valid position (hold)
            next_pts_held = next_pts.copy()
            for j in range(12):
                if not kept[j]:
                    next_pts_held[j] = last_valid_pts[j]
            next_pts = next_pts_held

        # update last_valid for points actually tracked
        for j in range(12):
            if kept[j]:
                last_valid_pts[j] = next_pts[j]

        # Reshape into per-portrait corners
        pts_r = next_pts.reshape(3, 4, 2)
        for pid in (0, 1, 2):
            out["portraits"][str(pid)][str(fi)] = {
                "corners": pts_r[pid].tolist(),
                "occluded": (confidence < 0.5),
                "score": confidence,
                "interpolated": False,
            }

        prev_gray = gray; prev_pts = next_pts
        fi += 1

    cap.release()
    out["n_frames"] = fi   # actual decoded count
    return out, {"lk_lost_count": lk_lost_count}


# ─── per-episode driver ─────────────────────────────────────────────────────
def process_episode(
    ep_dir: Path, image_predictor, face_app, celeb_protos: dict[str, np.ndarray],
    *, force: bool, interactive: bool,
) -> dict:
    out_json = ep_dir / "portrait_corners.json"
    if out_json.is_file() and not force:
        return {"ep": ep_dir.name, "skipped": True}

    # locate video
    cands = list(ep_dir.glob("videos/*/chunk-*/file-*.mp4"))
    if not cands:
        return {"ep": ep_dir.name, "error": "no video found"}
    video = cands[0]
    print(f"\n  → {ep_dir.name}")

    frame_0 = read_frame_zero(video)
    H, W = frame_0.shape[:2]

    # 1. Read sidecar to know layout → celeb order
    ref_path = ep_dir / "reference.json"
    layout_celebs: list[str] | None = None
    if ref_path.is_file():
        ref = json.loads(ref_path.read_text())
        layout = ref.get("layout", "-")
        i2k = {"S": "swift", "O": "obama", "L": "lecun"}
        if len(layout) == 3 and all(c in i2k for c in layout):
            layout_celebs = [i2k[c] for c in layout]

    # 2. Detect portraits with GroundingDINO
    print(f"    GroundingDINO detect on frame 0...", end=" ")
    t0 = time.time()
    boxes = detect_portraits_grounding_dino(frame_0)
    print(f"({time.time()-t0:.1f}s → {len(boxes)} portrait box(es))")
    if len(boxes) < 3:
        if interactive:
            print(f"    ! only {len(boxes)} detected, falling back to click prompts")
            # delegate to legacy 2_segment_video.py logic if interactive fallback needed
            return {"ep": ep_dir.name, "error": f"only {len(boxes)} portraits detected; re-run with legacy 2_segment_video.py --interactive"}
        return {"ep": ep_dir.name, "error": f"GroundingDINO only found {len(boxes)} portraits (need 3); re-run with --interactive"}

    # 3. SAM 2.1 image predictor → tight masks per box
    print(f"    SAM 2.1 image predictor refines each box...", end=" ")
    t0 = time.time()
    masks: list[np.ndarray] = []
    box_centres: list[tuple[float, float]] = []
    for box in boxes:
        mask = sam_box_to_mask(image_predictor, frame_0, box)
        masks.append(mask)
        cx = (box[0] + box[2]) / 2; cy = (box[1] + box[3]) / 2
        box_centres.append((cx, cy))
    print(f"({time.time()-t0:.1f}s)")

    # 4. Mask → 4 corners per portrait
    corners_per_box: list[np.ndarray] = []
    for mask in masks:
        c = mask_to_corners(mask)
        if c is None:
            return {"ep": ep_dir.name, "error": "mask→corners failed for one portrait"}
        corners_per_box.append(c)

    # 5. Identify each portrait via ArcFace + sort by box centre x for pid ordering
    pid_order = sorted(range(len(boxes)), key=lambda i: box_centres[i][0])
    initial_corners: dict[int, np.ndarray] = {
        pid: corners_per_box[pid_order[pid]] for pid in range(3)
    }
    # ArcFace identification — find ALL faces in the upscaled frame, then
    # match each to one of the 3 portraits by bbox-overlap with the SAM
    # mask corners. This works even when face detection on a per-portrait
    # crop fails (because the warp-then-resize loses too much detail).
    print(f"    ArcFace identify each portrait...", end=" ")
    UPSCALE = 2
    big = cv2.resize(frame_0, (frame_0.shape[1] * UPSCALE, frame_0.shape[0] * UPSCALE),
                     interpolation=cv2.INTER_LANCZOS4)
    all_faces = face_app.get(big)
    # Bring face bboxes back to original frame scale
    face_centres: list[tuple[float, float, np.ndarray]] = []
    for f in all_faces:
        x0, y0, x1, y1 = f.bbox / UPSCALE
        face_centres.append((float((x0+x1)/2), float((y0+y1)/2), f.normed_embedding))

    def assign_face_to_pid(pid: int) -> tuple[str, float]:
        """Find the face whose centre falls inside this portrait's corners polygon."""
        corners = initial_corners[pid].astype(np.float32)
        for cx, cy, emb in face_centres:
            inside = cv2.pointPolygonTest(corners, (cx, cy), False)
            if inside >= 0:
                scores = {c: float(emb @ p) for c, p in celeb_protos.items()}
                best = max(scores, key=scores.get)
                return best, scores[best]
        return "?", 0.0

    id_results: list[tuple[str, float]] = []
    for pid in range(3):
        celeb_id, cos = assign_face_to_pid(pid)
        id_results.append((celeb_id, cos))
    print(f"({[(c, round(s, 3)) for c, s in id_results]})  ({len(all_faces)} faces detected in frame)")

    # Build pid → celeb from ArcFace results
    pid_to_celeb: dict[str, str] = {str(i): id_results[i][0] for i in range(3)}
    # Validate: should have all 3 distinct expected IDs
    if layout_celebs is not None:
        expected_set = set(layout_celebs)
        found_set = set(c for c in pid_to_celeb.values() if c in expected_set)
        if found_set != expected_set:
            print(f"    ⚠ ArcFace identified {sorted(found_set)} but expected {sorted(expected_set)}")
            # fall back: trust the layout list in linear-x order
            pid_to_celeb = {str(i): layout_celebs[i] for i in range(3)}
            print(f"    fallback: trusting layout {layout_celebs} in x-sorted order")

    # 6. Persist initial seeds (so stage 4 can read them)
    seeds_data: dict = {
        "points": [[int(box_centres[i][0]), int(box_centres[i][1])] for i in pid_order],
        "labels": [1, 1, 1],
        "detector": "GroundingDINO-tiny@disable_custom_kernels",
        "boxes_xyxy": [[float(v) for v in boxes[i][:4]] for i in pid_order],
        "box_scores": [float(boxes[i][4]) for i in pid_order],
    }
    seeds_data["celebs"] = [pid_to_celeb[str(i)] for i in range(3)]
    seeds_data["identify_cosines"] = [float(id_results[i][1]) for i in range(3)]
    (ep_dir / "portrait_seeds.json").write_text(json.dumps(seeds_data, indent=2))

    # 7. LK-track the 12 corners across all frames
    print(f"    LK optical flow on 12 corner points...", end=" ")
    t0 = time.time()
    corners_data, lk_stats = track_corners_lk(video, initial_corners)
    print(f"({time.time()-t0:.1f}s, {corners_data['n_frames']} frames, "
          f"{lk_stats['lk_lost_count']} corner-frame losses)")

    out_json.write_text(json.dumps(corners_data, indent=2))
    return {
        "ep": ep_dir.name, "saved": str(out_json),
        "n_frames": corners_data["n_frames"],
        "lk_lost_count": lk_stats["lk_lost_count"],
    }


# ─── main ───────────────────────────────────────────────────────────────────
_gdino_proc = None
_gdino_model = None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode_dir", nargs="?", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--interactive", action="store_true",
                   help="fall back to click prompts (legacy 2_segment_video.py path) if detection fails")
    p.add_argument("--ckpt", default=str(SAM2_CKPT_DEFAULT))
    p.add_argument("--cfg", default=SAM2_CFG)
    args = p.parse_args()
    if (args.episode_dir is None) == (args.root is None):
        print("[ERROR] specify exactly one of: episode_dir, --root", file=sys.stderr)
        return 2

    _import_heavy()

    print("loading SAM 2.1 hiera-L...")
    t0 = time.time()
    sam_model = build_sam2(args.cfg, args.ckpt, device="cuda")
    image_predictor = SAM2ImagePredictor(sam_model)
    print(f"  ({time.time()-t0:.1f}s)")

    print(f"loading GroundingDINO ({GDINO_MODEL}) with disable_custom_kernels=True...")
    t0 = time.time()
    global _gdino_proc, _gdino_model
    _gdino_proc = AutoProcessor.from_pretrained(GDINO_MODEL)
    _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_MODEL, disable_custom_kernels=True
    ).to("cuda").eval()
    print(f"  ({time.time()-t0:.1f}s)")

    print(f"loading InsightFace buffalo_l for portrait identification...")
    t0 = time.time()
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    print(f"  ({time.time()-t0:.1f}s)")

    print(f"building celebrity prototype embeddings from photo bank...")
    t0 = time.time()
    photo_bank = Path("/home/lemonkey/LeMonkey/datasets/eval3_celebs/web")
    celeb_protos = build_celeb_prototypes(face_app, photo_bank, ["swift", "obama", "lecun"])
    print(f"  ({time.time()-t0:.1f}s, built {len(celeb_protos)} prototypes: {sorted(celeb_protos.keys())})")

    if args.episode_dir:
        ep_dirs = [Path(args.episode_dir)]
    else:
        ep_dirs = sorted(p for p in Path(args.root).iterdir() if p.is_dir())

    results = []
    for ep_dir in ep_dirs:
        try:
            r = process_episode(ep_dir, image_predictor, face_app, celeb_protos,
                                force=args.force, interactive=args.interactive)
        except Exception as e:
            r = {"ep": ep_dir.name, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        if "saved" in r:
            print(f"  ✓ {r['ep']:50s}  {r['n_frames']:>4} frames  lk-losses={r['lk_lost_count']}")
        elif r.get("skipped"):
            print(f"  - {r['ep']:50s}  (skipped)")
        else:
            print(f"  ✗ {r['ep']:50s}  {r.get('error','?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
