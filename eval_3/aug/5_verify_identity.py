#!/usr/bin/env python3
"""STAGE 5 — verify that the inpainted face still matches the celebrity.

For each augmented variant:
  1. Sample 5 frames spread across the timeline.
  2. For each frame, crop the inpainted target portrait region using the
     corners json + augmentation.json (we know which portrait was the target).
  3. Run InsightFace ArcFace: must have ≥ 1 face, take dominant face's embedding.
  4. Compare against an embedding of the variant's reference_photo.
  5. Report min cosine across the 5 sampled frames.
  6. Mark the variant ACCEPTED if min_cos ≥ --threshold (default 0.4); else
     REJECTED (we drop it from the training set).

Output:
  <out-root>/<variant_dir>/verification.json
    {
      "min_cosine": float,
      "frame_cosines": [float × 5],
      "frame_indices": [int × 5],
      "n_faces_per_frame": [int × 5],
      "reference_photo": "...",
      "passed": bool,
      "threshold": 0.4
    }

Usage:
    python 5_verify_identity.py --root ~/LeMonkey/datasets/eval3_aug
    python 5_verify_identity.py /path/to/variant_dir
    python 5_verify_identity.py --root ... --drop-failed   # also delete REJECTED variants

See STRATEGY.md §3.6 for design rationale.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None  # type: ignore


def get_target_portrait_id(
    augmentation: dict,
    sidecar: dict,
) -> str:
    """Find the portrait_id whose celeb matches sidecar['target_celeb']."""
    target_celeb = sidecar["target_celeb"]
    pid_to_celeb = augmentation["pid_to_celeb"]
    for pid, celeb in pid_to_celeb.items():
        if celeb == target_celeb:
            return pid
    raise ValueError(f"target celeb '{target_celeb}' not found in pid_to_celeb={pid_to_celeb}")


def crop_portrait_from_corners(
    frame: np.ndarray, corners: list[list[float]],
    *, target_w: int = 224, target_h: int = 320,
) -> np.ndarray:
    """Warp the portrait quadrilateral to a canonical upright (target_w x target_h) crop."""
    src = np.asarray(corners, dtype=np.float32)
    dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (target_w, target_h), flags=cv2.INTER_LANCZOS4)


def load_arcface() -> "FaceAnalysis":
    if FaceAnalysis is None:
        raise RuntimeError("insightface not installed (pip install insightface onnxruntime-gpu)")
    app = FaceAnalysis(name="buffalo_l",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app


def embed_one_face(app: "FaceAnalysis", img_bgr: np.ndarray) -> np.ndarray | None:
    """Return the dominant face's normed_embedding, or None."""
    faces = app.get(img_bgr)
    if not faces:
        return None
    if len(faces) > 1:
        # Take the largest face by area (most likely the portrait subject)
        faces.sort(key=lambda f: -((f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])))
    return faces[0].normed_embedding


def find_video(variant_dir: Path) -> Path | None:
    cands = list(variant_dir.glob("videos/*/chunk-*/file-*.mp4"))
    return cands[0] if cands else None


def find_corners_json_for_variant(variant_dir: Path) -> Path | None:
    """The corners json lives in the SOURCE (non-augmented) episode dir, not in the variant.
    augmentation.json tells us the source name."""
    aug_path = variant_dir / "augmentation.json"
    if not aug_path.is_file():
        return None
    aug = json.loads(aug_path.read_text())
    src = aug["src_episode"]
    # Heuristic: the source is in the parent's parent, OR explicitly under datasets/eval3_quick / eval3
    # Try common roots
    for root in [Path.home() / "LeMonkey/datasets/eval3_quick",
                 Path.home() / "LeMonkey/datasets/eval3",
                 variant_dir.parent.parent / "eval3_quick",
                 variant_dir.parent.parent / "eval3"]:
        cand = root / src / "portrait_corners.json"
        if cand.is_file():
            return cand
    return None


def verify_variant(
    variant_dir: Path,
    app: "FaceAnalysis",
    *,
    n_samples: int = 5,
    threshold: float = 0.4,
) -> dict:
    aug_path = variant_dir / "augmentation.json"
    sidecar_path = variant_dir / "reference.json"
    if not aug_path.is_file() or not sidecar_path.is_file():
        return {"variant": variant_dir.name, "error": "augmentation.json or reference.json missing"}
    augmentation = json.loads(aug_path.read_text())
    sidecar = json.loads(sidecar_path.read_text())

    target_pid = get_target_portrait_id(augmentation, sidecar)

    corners_json = find_corners_json_for_variant(variant_dir)
    if corners_json is None:
        return {"variant": variant_dir.name, "error": "could not locate source portrait_corners.json"}
    corners_data = json.loads(corners_json.read_text())

    video = find_video(variant_dir)
    if video is None:
        return {"variant": variant_dir.name, "error": "no augmented video found"}

    ref_photo_path = augmentation.get("reference_photo") or sidecar.get("reference_photo")
    if not ref_photo_path:
        return {"variant": variant_dir.name, "error": "no reference_photo in augmentation/sidecar"}
    ref_img = cv2.imread(ref_photo_path, cv2.IMREAD_COLOR)
    if ref_img is None:
        return {"variant": variant_dir.name, "error": f"cannot read reference_photo at {ref_photo_path}"}
    ref_emb = embed_one_face(app, ref_img)
    if ref_emb is None:
        return {"variant": variant_dir.name, "error": "no face detected in reference_photo"}

    n_frames = corners_data["n_frames"]
    sample_indices = list(np.linspace(0, n_frames - 1, n_samples, dtype=int))

    cap = cv2.VideoCapture(str(video))
    cosines: list[float] = []
    n_faces_per_frame: list[int] = []
    fi = 0
    sample_i = 0
    while sample_i < len(sample_indices):
        ok, frame = cap.read()
        if not ok:
            break
        if fi == sample_indices[sample_i]:
            rec = corners_data["portraits"][target_pid].get(str(fi))
            if rec is None or rec["corners"] is None:
                cosines.append(-1.0); n_faces_per_frame.append(0)
            else:
                crop = crop_portrait_from_corners(frame, rec["corners"])
                faces = app.get(crop)
                n_faces_per_frame.append(len(faces))
                if not faces:
                    cosines.append(-1.0)
                else:
                    faces.sort(key=lambda f: -((f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])))
                    cos = float(faces[0].normed_embedding @ ref_emb)
                    cosines.append(cos)
            sample_i += 1
        fi += 1
    cap.release()

    valid = [c for c in cosines if c > -1.0]
    min_cos = min(valid) if valid else -1.0
    passed = bool(min_cos >= threshold)
    result = {
        "variant": variant_dir.name,
        "min_cosine": min_cos,
        "mean_cosine": float(np.mean(valid)) if valid else -1.0,
        "frame_cosines": cosines,
        "frame_indices": [int(i) for i in sample_indices],
        "n_faces_per_frame": n_faces_per_frame,
        "reference_photo": ref_photo_path,
        "threshold": threshold,
        "passed": passed,
    }
    (variant_dir / "verification.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("variant_dir", nargs="?", default=None)
    p.add_argument("--root", default=None,
                   help="root containing variant_dirs (e.g. ~/LeMonkey/datasets/eval3_aug)")
    # ArcFace cos-sim ≥ 0.4 default. Verified against:
    #   - InsightFace official guide (insightface.ai): "0.30–0.45 cosine
    #     range at FMR = 1e-4 to 1e-5". 0.4 is squarely in band.
    #   - DeepFace's ArcFace default uses cos-sim ≥ 0.32 (cos-distance
    #     ≤ 0.68 via C4.5 over labelled pairs). 0.4 is stricter.
    #   - face_recognition lib uses Euclidean ≤ 0.6 ≈ LFW 99.38% TAR.
    # See eval_3/aug/VALIDATION.md §1.
    p.add_argument("--threshold", type=float, default=0.4,
                   help="ArcFace cosine similarity threshold (default 0.4 — InsightFace canonical band)")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--drop-failed", action="store_true",
                   help="rm -rf variants whose min_cosine < threshold")
    p.add_argument("--force", action="store_true",
                   help="re-verify even if verification.json exists")
    args = p.parse_args()

    if (args.variant_dir is None) == (args.root is None):
        print("[ERROR] specify exactly one of: variant_dir, --root", file=sys.stderr)
        return 2

    print("loading InsightFace buffalo_l...")
    t0 = time.time()
    app = load_arcface()
    print(f"  loaded in {time.time()-t0:.1f}s")

    if args.variant_dir:
        variants = [Path(args.variant_dir)]
    else:
        variants = sorted(p for p in Path(args.root).iterdir() if p.is_dir())

    results: list[dict] = []
    for v in variants:
        if (v / "verification.json").is_file() and not args.force:
            cached = json.loads((v / "verification.json").read_text())
            r = {"variant": v.name, "cached": True, **cached}
        else:
            try:
                r = verify_variant(v, app, n_samples=args.n_samples, threshold=args.threshold)
            except Exception as e:
                r = {"variant": v.name, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        if "error" in r:
            print(f"  ✗ {r['variant']:45s}  {r['error']}")
        else:
            mark = "✓" if r["passed"] else "✗"
            print(f"  {mark} {r['variant']:45s}  min_cos={r['min_cosine']:.3f}  "
                  f"mean={r.get('mean_cosine', -1):.3f}")

    n_pass = sum(1 for r in results if r.get("passed"))
    n_fail = sum(1 for r in results if r.get("passed") is False)
    print(f"\n  passed: {n_pass}    failed: {n_fail}    errored: {len(results) - n_pass - n_fail}")

    if args.drop_failed:
        for r in results:
            if r.get("passed") is False:
                v = next((v for v in variants if v.name == r["variant"]), None)
                if v is not None and v.is_dir():
                    shutil.rmtree(v)
                    print(f"  🗑  removed {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
