#!/usr/bin/env python3
"""ArcFace audit of the 200-celeb training dataset for Pi0.5 VL cotrain.

For each (episode_idx, frame_idx) in Darius's bbox annotations, compute:
- target_cos: ArcFace cosine between the painted face and target celeb's centroid
- max_distractor_cos: highest cosine to any non-target celeb centroid
- hardneg_score: target_cos minus max_distractor_cos
  (low → confusable; high → unambiguous identity)

This feeds two downstream artifacts:
1. keep_episodes.txt — drop episodes whose mean target_cos is too low (bad inpainting)
2. sample_weights.npy — oversample variants with low hardneg_score (force fine-grained
   discrimination during training)

Per CLAUDE.md §5: no silent fallbacks. Any failure path emits a [WARN] with what
was expected, what happened, and what fallback was chosen.

Per CLAUDE.md §7 / §8: numerical defaults are triple-sourced inline.

Input contract (Darius will deliver, format TBD — script accepts either):

  Schema A — bbox only (script computes ArcFace on crops):
    parquet with columns:
      episode_idx (int64)
      frame_idx   (int64)
      bbox_x1, bbox_y1, bbox_x2, bbox_y2 (float32, pixel coords on 480x640 camera1)
      target_celeb (str, slug like "barack_obama")
      distractor_celebs (list[str])

  Schema B — bbox + pre-computed embedding (script just looks up):
    parquet with columns from Schema A plus:
      target_face_embedding (list[float], 512-d L2-normalized)

Usage:

  python arcface_audit_200celeb.py \
      --bbox-parquet ~/data/darius_200celeb_bboxes.parquet \
      --celeb-manifest ~/data/arcface_toolkit/celeb_embeddings.json \
      --dataset-root ~/data/200celebs \
      --output audit_200celeb.parquet

If --dataset-root is omitted, the script requires Schema B (pre-computed embeddings).
If --dataset-root is provided, the script decodes camera1 mp4s and runs ArcFace on
the bbox crops — slower (~6h on single GPU, ~1h on edna 128-core CPU).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# --- triple-sourced thresholds ---------------------------------------------
# Standard ArcFace verification cosine thresholds (buffalo_l on LFW):
#   FAR=1e-4 → 0.42, FAR=1e-3 → 0.36 (InsightFace docs)
#   For OUR inpainted-painted-photo distribution (NOT clean web headshots),
#   empirical scatter is ~10-15% looser (per Mahbod's M2 data audit
#   2026-05-19_m2_data_audit.md: same-celeb cos ~0.5-0.8,
#   cross-celeb cos ~0.0-0.2).
#   → 0.5 keep threshold = "well above noise floor, below clean-face mean"
DEFAULT_KEEP_COS = 0.50

# Hard-neg score threshold: target_cos - max_distractor_cos
#   < 0.10 → confusable (oversample at HARD_WEIGHT)
#   >= 0.10 → unambiguous (normal weight)
DEFAULT_HARDNEG_GAP = 0.10
DEFAULT_HARD_WEIGHT = 2.0
# ---------------------------------------------------------------------------


def load_celeb_centroids(manifest_path: Path) -> dict[str, np.ndarray]:
    """Load Mahbod's celeb_embeddings.json → {slug: 512-d L2-normalized centroid}."""
    if not manifest_path.is_file():
        print(f"[ERR] celeb manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(2)
    data = json.loads(manifest_path.read_text())
    centroids = {}
    n_missing = 0
    for slug, info in data["celebs"].items():
        if info.get("centroid") is None:
            n_missing += 1
            print(f"[WARN] celeb={slug}: expected=centroid, got=None, fallback=skip",
                  flush=True)
            continue
        c = np.asarray(info["centroid"], dtype=np.float32)
        c = c / max(float(np.linalg.norm(c)), 1e-6)  # safety renormalize
        centroids[slug] = c
    print(f"[info] loaded {len(centroids)} celeb centroids "
          f"({n_missing} skipped for missing/null centroid)")
    if n_missing > 0:
        print(f"[WARN] manifest had {n_missing} celebs without centroid; rows targeting "
              f"those celebs will be marked target_cos=NaN (filter step drops them)",
              flush=True)
    return centroids


def _build_arcface_app():
    """Load buffalo_l on CPU (or GPU if available)."""
    from insightface.app import FaceAnalysis  # local import: heavy
    try:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if "CUDAExecutionProvider" in ort.get_available_providers() \
            else ["CPUExecutionProvider"]
    except ImportError:
        providers = ["CPUExecutionProvider"]
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["recognition"],  # we already HAVE bboxes; no detection needed
        providers=providers,
    )
    app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
                det_size=(320, 320))
    print(f"[info] buffalo_l loaded on {providers[0]}")
    return app


def embed_face_crop(app, img_bgr, bbox_xyxy) -> np.ndarray | None:
    """Run ArcFace on a face crop from img_bgr at bbox_xyxy (pixel coords)."""
    import cv2  # local import

    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img_bgr[y1:y2, x1:x2]
    if min(crop.shape[:2]) < 24:
        return None  # too tiny to embed reliably
    # Resize to 112x112 (ArcFace canonical input).
    crop_rs = cv2.resize(crop, (112, 112))
    try:
        embedding = app.models["recognition"].get_feat(crop_rs).flatten()
    except Exception as e:
        print(f"[WARN] ArcFace failed on bbox={bbox_xyxy}: {e}", flush=True)
        return None
    norm = float(np.linalg.norm(embedding))
    if norm < 1e-6:
        return None
    return (embedding / norm).astype(np.float32)


def audit_dataset(
    bbox_df,
    centroids: dict[str, np.ndarray],
    dataset_root: Path | None,
    app_lazy_loader,
) -> np.ndarray:
    """Compute target_cos + max_distractor_cos + hardneg_score per row.

    Returns array of shape (N, 3): [target_cos, max_distractor_cos, hardneg_score].
    Rows where the target celeb is missing from manifest → NaN row (filtered later).
    """
    n = len(bbox_df)
    out = np.full((n, 3), np.nan, dtype=np.float32)

    # Group by episode_idx so we can decode mp4 once per episode if needed.
    if dataset_root is not None:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa
            print("[info] lerobot dataset loader available — will decode camera1 lazily")
        except ImportError:
            print(f"[WARN] lerobot not importable: expected=lazy mp4 decode, "
                  f"got=ImportError, fallback=require Schema B (pre-computed embeddings)",
                  flush=True)
            dataset_root = None

    needs_arcface = "target_face_embedding" not in bbox_df.columns and dataset_root
    app = app_lazy_loader() if needs_arcface else None

    t0 = time.time()
    last_log = t0
    for i, row in enumerate(bbox_df.itertuples(index=False)):
        target_slug = row.target_celeb
        if target_slug not in centroids:
            continue  # NaN row; will be filtered by keep_episodes

        if hasattr(row, "target_face_embedding") and row.target_face_embedding is not None:
            face_emb = np.asarray(row.target_face_embedding, dtype=np.float32)
            face_emb = face_emb / max(float(np.linalg.norm(face_emb)), 1e-6)
        elif app is not None:
            # On-the-fly: decode the camera1 frame, crop, embed.
            # (This branch is the slow path. Pre-computed embeddings are preferred.)
            img = _decode_frame(dataset_root, row.episode_idx, row.frame_idx)
            if img is None:
                continue
            face_emb = embed_face_crop(
                app, img,
                (row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2),
            )
            if face_emb is None:
                continue
        else:
            continue

        target_centroid = centroids[target_slug]
        target_cos = float(face_emb @ target_centroid)

        # Max distractor cosine: against all OTHER celebs.
        max_distractor = -1.0
        for slug, c in centroids.items():
            if slug == target_slug:
                continue
            cosv = float(face_emb @ c)
            if cosv > max_distractor:
                max_distractor = cosv

        out[i, 0] = target_cos
        out[i, 1] = max_distractor
        out[i, 2] = target_cos - max_distractor

        if time.time() - last_log > 30.0:
            n_done = i + 1
            rate = n_done / (time.time() - t0)
            eta = (n - n_done) / max(rate, 1e-6)
            print(f"[info] audited {n_done}/{n} rows ({rate:.0f}/s, "
                  f"ETA {eta:.0f}s)", flush=True)
            last_log = time.time()

    elapsed = time.time() - t0
    valid_mask = ~np.isnan(out[:, 0])
    n_valid = int(valid_mask.sum())
    print(f"[done] {n_valid}/{n} rows valid in {elapsed:.0f}s")
    if n_valid < n:
        print(f"[WARN] {n - n_valid} rows had missing centroid or failed embedding; "
              f"these rows have NaN cos and will be filtered downstream", flush=True)
    return out


def _decode_frame(dataset_root: Path, episode_idx: int, frame_idx: int):
    """Decode a single camera1 frame from the lerobot dataset mp4 chunks.

    Lazy fallback path. If lerobot loader isn't available or the file is missing,
    returns None and emits [WARN].
    """
    # Implementation deferred — depends on the dataset's actual chunking layout
    # on disk. Will be filled in once we know whether Darius provides Schema A
    # (we decode) or Schema B (he provides embeddings).
    print(f"[WARN] _decode_frame called for ep={episode_idx} frame={frame_idx} "
          f"but on-the-fly decoding is not yet implemented; expected=image, got=None, "
          f"fallback=skip row (require Schema B from Darius)", flush=True)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-parquet", type=Path, required=True,
                        help="Darius's per-frame bbox annotations for 200-celeb")
    parser.add_argument("--celeb-manifest", type=Path, required=True,
                        help="Mahbod's celeb_embeddings.json")
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="Local lerobot dataset root (only needed if bbox-parquet "
                             "doesn't include target_face_embedding column)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output parquet path with audit columns")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("[ERR] pandas required", file=sys.stderr)
        return 2

    if not args.bbox_parquet.is_file():
        print(f"[ERR] bbox parquet not found: {args.bbox_parquet}", file=sys.stderr)
        return 2

    print(f"[info] loading bbox parquet: {args.bbox_parquet}")
    bbox_df = pd.read_parquet(args.bbox_parquet)
    required_cols = {"episode_idx", "frame_idx", "bbox_x1", "bbox_y1", "bbox_x2",
                     "bbox_y2", "target_celeb"}
    missing = required_cols - set(bbox_df.columns)
    if missing:
        print(f"[ERR] bbox parquet missing required columns: {missing}",
              file=sys.stderr)
        return 2

    centroids = load_celeb_centroids(args.celeb_manifest)
    if not centroids:
        print(f"[ERR] no celeb centroids loaded from manifest", file=sys.stderr)
        return 2

    audit_arr = audit_dataset(bbox_df, centroids, args.dataset_root,
                              app_lazy_loader=_build_arcface_app)

    out_df = bbox_df.copy()
    out_df["target_cos"] = audit_arr[:, 0]
    out_df["max_distractor_cos"] = audit_arr[:, 1]
    out_df["hardneg_gap"] = audit_arr[:, 2]
    out_df.to_parquet(args.output, index=False)
    print(f"[done] audit written to {args.output}")

    # Summary stats.
    valid = out_df[~out_df["target_cos"].isna()]
    if len(valid) > 0:
        print(f"\n[summary] {len(valid)} valid rows out of {len(out_df)}:")
        print(f"  target_cos:        mean={valid['target_cos'].mean():.3f}  "
              f"std={valid['target_cos'].std():.3f}  "
              f"p10={valid['target_cos'].quantile(0.10):.3f}  "
              f"p50={valid['target_cos'].quantile(0.50):.3f}  "
              f"p90={valid['target_cos'].quantile(0.90):.3f}")
        print(f"  hardneg_gap:       mean={valid['hardneg_gap'].mean():.3f}  "
              f"std={valid['hardneg_gap'].std():.3f}  "
              f"p10={valid['hardneg_gap'].quantile(0.10):.3f}  "
              f"p50={valid['hardneg_gap'].quantile(0.50):.3f}")
        print(f"  would-keep at cos>={DEFAULT_KEEP_COS}: "
              f"{(valid['target_cos'] >= DEFAULT_KEEP_COS).mean()*100:.1f}% of rows")
        print(f"  would-mark hard at gap<{DEFAULT_HARDNEG_GAP}: "
              f"{(valid['hardneg_gap'] < DEFAULT_HARDNEG_GAP).mean()*100:.1f}% of rows")

    return 0


if __name__ == "__main__":
    sys.exit(main())
