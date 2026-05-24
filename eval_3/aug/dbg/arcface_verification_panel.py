#!/usr/bin/env python3
"""Render the Stage 5 ArcFace identity-verification gate on a real aug variant.

For one augmented variant on disk, this script:

  1. Reads the variant's `augmentation.json` to learn which celebrity is
     painted at each of the 3 portrait slots.
  2. Decodes frame 0 of the variant's inpainted video.
  3. Crops the 3 portraits using the source teleop's saved corners.
  4. Runs InsightFace ArcFace (buffalo_l) on each portrait crop.
  5. Loads a few reference photos per celebrity, averages their ArcFace
     embeddings to form the per-celebrity centroid.
  6. For every (portrait, centroid) pair, computes cosine similarity.
  7. Renders a single panel that shows the 3 portrait crops side by side
     and a 3x3 cosine matrix below them, with PASS (>= 0.40) cells
     coloured green and FAIL cells red. The diagonal of the matrix is
     the verification check that decides whether the variant is kept
     in the dataset.

Usage:
    python eval_3/aug/dbg/arcface_verification_panel.py \\
        --variant datasets/eval3_track3_aug/<var_dir> \\
        --base-root datasets/eval3 \\
        --bank-root datasets/eval3_celebs/scraped \\
        --out media/figures/aug/stage5_arcface_verification.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


THRESHOLD = 0.40
PORTRAIT_W = 220        # rendered width of each portrait crop
HEADER_H = 30           # title bar above each portrait
CELL_W = 220            # cosine-matrix cell width
CELL_H = 84             # cosine-matrix cell height
LABEL_W = 220           # left-side row-label width


def crop_portrait(frame: np.ndarray, corners: np.ndarray,
                    pad: int = 16) -> np.ndarray:
    """Tight crop around the portrait quad with `pad` px margin."""
    H, W = frame.shape[:2]
    x0 = max(0, int(corners[:, 0].min()) - pad)
    y0 = max(0, int(corners[:, 1].min()) - pad)
    x1 = min(W, int(corners[:, 0].max()) + pad)
    y1 = min(H, int(corners[:, 1].max()) + pad)
    return frame[y0:y1, x0:x1].copy()


def get_embedding(face_app, img: np.ndarray) -> np.ndarray | None:
    """Run InsightFace on a BGR image; return the L2-normalized embedding
    of the largest detected face, or None if no face is found."""
    faces = face_app.get(img)
    if not faces:
        return None
    # largest face by bbox area
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    emb = f.normed_embedding
    return emb.astype(np.float32)


def celeb_centroid(face_app, bank_root: Path, celeb_slug: str,
                     max_photos: int = 5) -> np.ndarray | None:
    """Average L2-normalized ArcFace embedding over up to `max_photos`
    scraped reference photos of one celebrity. Returns None if no usable
    photos are found."""
    celeb_dir = bank_root / celeb_slug
    if not celeb_dir.is_dir():
        return None
    photos = sorted(list(celeb_dir.glob("*.jpg")) +
                      list(celeb_dir.glob("*.png")))[:max_photos]
    embs = []
    for p in photos:
        img = cv2.imread(str(p))
        if img is None:
            continue
        emb = get_embedding(face_app, img)
        if emb is not None:
            embs.append(emb)
    if not embs:
        return None
    mean = np.mean(embs, axis=0)
    return (mean / (np.linalg.norm(mean) + 1e-8)).astype(np.float32)


def short_name(slug: str) -> str:
    """taylor_swift -> Taylor Swift."""
    parts = slug.replace("-", "_").split("_")
    return " ".join(w.capitalize() for w in parts)


def render_portrait_strip(crops: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Lay the 3 portrait crops in a horizontal strip with per-crop headers."""
    panels = []
    target_h = None
    for caption, img in crops:
        h, w = img.shape[:2]
        scale = PORTRAIT_W / w
        img = cv2.resize(img, (PORTRAIT_W, int(round(h * scale))),
                            interpolation=cv2.INTER_LANCZOS4)
        bar = np.full((HEADER_H, PORTRAIT_W, 3), 28, dtype=np.uint8)
        cv2.putText(bar, caption, (8, HEADER_H - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1,
                     cv2.LINE_AA)
        panel = np.vstack([bar, img])
        panels.append(panel)
        target_h = panel.shape[0] if target_h is None else max(target_h, panel.shape[0])
    # Equalise heights by padding the bottom of shorter ones
    normalized = []
    for p in panels:
        if p.shape[0] < target_h:
            extra = np.full((target_h - p.shape[0], p.shape[1], 3), 28, dtype=np.uint8)
            p = np.vstack([p, extra])
        normalized.append(p)
    # Hstack with thin separators
    out = [normalized[0]]
    sep = np.full((target_h, 6, 3), 12, dtype=np.uint8)
    for p in normalized[1:]:
        out.extend([sep, p])
    # Add a row label on the left so this strip aligns visually with the
    # cosine matrix below it.
    label_col = np.full((target_h, LABEL_W, 3), 12, dtype=np.uint8)
    cv2.putText(label_col, "Inpainted portrait", (10, target_h // 2 - 8),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250, 250, 250), 1,
                 cv2.LINE_AA)
    cv2.putText(label_col, "(crop from variant)", (10, target_h // 2 + 16),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
                 cv2.LINE_AA)
    return np.hstack([label_col, np.hstack(out)])


def render_cosine_matrix(row_labels: list[str], col_labels: list[str],
                            cos_matrix: np.ndarray, *,
                            threshold: float = THRESHOLD) -> np.ndarray:
    """Render the M x N cosine matrix as colored cells with the value
    written in each cell. Row labels go on the left, column labels on top.

    Diagonal cells correspond to the verification check
    (portrait[i] vs target-celeb-centroid[i]) and should pass.
    """
    n_rows = len(row_labels)
    n_cols = len(col_labels)

    # Row body width: LABEL_W + n_cols*CELL_W + (n_cols-1)*6 separators
    body_w = LABEL_W + n_cols * CELL_W + max(0, n_cols - 1) * 6

    # Top column-header strip
    header_h = 36
    cols_strip = np.full((header_h, body_w, 3), 12, dtype=np.uint8)
    cv2.putText(cols_strip, "vs reference centroid:", (10, header_h - 12),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    for j, lab in enumerate(col_labels):
        x0 = LABEL_W + j * (CELL_W + 6) + 8
        cv2.putText(cols_strip, lab, (x0, header_h - 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
                     cv2.LINE_AA)

    rows_imgs = [cols_strip]
    for i, rlab in enumerate(row_labels):
        # Left row label
        left = np.full((CELL_H, LABEL_W, 3), 12, dtype=np.uint8)
        cv2.putText(left, rlab, (10, CELL_H // 2 + 6),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1,
                     cv2.LINE_AA)
        row_cells = [left]
        for j in range(n_cols):
            v = float(cos_matrix[i, j])
            # Colour: green if PASS, red if FAIL. Use a saturated background
            # for the diagonal (the actual verification check) and a muted
            # one for off-diagonal (separation-evidence cells).
            is_diag = (i == j)
            if v >= threshold:
                color = (40, 110, 40) if is_diag else (38, 70, 38)   # BGR greens
            else:
                color = (40, 40, 130) if is_diag else (38, 38, 70)   # BGR reds
            cell = np.full((CELL_H, CELL_W, 3), color, dtype=np.uint8)
            # Value
            cv2.putText(cell, f"{v:+.2f}", (10, 36),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2,
                         cv2.LINE_AA)
            # PASS / FAIL tag
            tag = "PASS" if v >= threshold else "FAIL"
            cv2.putText(cell, f"{tag}  (>=0.40)" if is_diag else tag,
                         (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                         (235, 235, 235), 1, cv2.LINE_AA)
            sep = np.full((CELL_H, 6, 3), 12, dtype=np.uint8)
            if j == 0:
                row_cells.append(cell)
            else:
                row_cells.extend([sep, cell])
        rows_imgs.append(np.hstack(row_cells))
        rows_imgs.append(np.full((6, rows_imgs[-1].shape[1], 3), 12, dtype=np.uint8))
    return np.vstack(rows_imgs[:-1])


def main() -> int:
    """CLI entry: render the Stage 5 ArcFace verification panel."""
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, type=Path,
                    help="Augmented variant dir (with augmentation.json + videos/)")
    p.add_argument("--base-root", required=True, type=Path,
                    help="Root of base teleops (we read portrait_corners.json "
                         "from the variant's src_episode under this root)")
    p.add_argument("--bank-root", required=True, type=Path,
                    help="Scraped photo bank (we read up to 5 photos per "
                         "celebrity to compute its ArcFace centroid)")
    p.add_argument("--out", required=True, type=Path,
                    help="Output composite PNG path")
    args = p.parse_args()

    # 1. Load variant metadata
    aug = json.loads((args.variant / "augmentation.json").read_text())
    pid_to_celeb = aug["pid_to_celeb_full"]   # {"0": "barack_obama", ...}
    src_ep_name = aug["src_episode"]
    src_ep = args.base_root / src_ep_name
    if not src_ep.is_dir():
        raise SystemExit(f"source teleop not found: {src_ep}")

    # 2. Frame 0 of the inpainted variant video
    cam_dir = args.variant / "videos" / "observation.images.camera1" / "chunk-000"
    mp4 = sorted(cam_dir.glob("*.mp4"))
    if not mp4:
        raise SystemExit(f"no mp4 under {cam_dir}")
    cap = cv2.VideoCapture(str(mp4[0]))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"failed to read frame 0 of {mp4[0]}")

    # 3. Source episode's corners (variants share geometry)
    corners_json = json.loads((src_ep / "portrait_corners.json").read_text())

    # 4. Crop the 3 portraits
    pids = ["0", "1", "2"]
    crops: list[tuple[str, str, np.ndarray]] = []   # (pid, painted_slug, crop)
    for pid in pids:
        slug = pid_to_celeb[pid]
        corners = np.asarray(corners_json["portraits"][pid]["0"]["corners"],
                              dtype=np.float32)
        crops.append((pid, slug, crop_portrait(frame, corners, pad=12)))

    # 5. Init InsightFace
    from insightface.app import FaceAnalysis  # type: ignore
    print("[info] initialising InsightFace buffalo_l (CPU OK; first call downloads ~280 MB)...",
          flush=True)
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 480))

    # 6. Embeddings for the inpainted crops + centroids for the 3 celebs
    crop_embeddings: list[np.ndarray | None] = []
    for _, _, c in crops:
        crop_embeddings.append(get_embedding(face_app, c))

    centroids: dict[str, np.ndarray | None] = {}
    for _, slug, _ in crops:
        if slug not in centroids:
            print(f"[info] centroid for {slug}...", flush=True)
            centroids[slug] = celeb_centroid(face_app, args.bank_root, slug)

    # 7. Cosine matrix: rows = inpainted portraits, cols = reference centroids.
    #    Both are L2-normalised so dot product == cosine similarity.
    n = len(crops)
    cos_matrix = np.zeros((n, n), dtype=np.float32)
    for i, emb_i in enumerate(crop_embeddings):
        for j, (_, slug_j, _) in enumerate(crops):
            cj = centroids.get(slug_j)
            if emb_i is None or cj is None:
                cos_matrix[i, j] = -1.0
            else:
                cos_matrix[i, j] = float(np.dot(emb_i, cj))

    # 8. Render
    strip_captions = [
        f"pid {pid}: painted with {short_name(slug)}"
        for (pid, slug, _) in crops
    ]
    strip = render_portrait_strip([(cap, crop) for cap, (_, _, crop) in zip(strip_captions, crops)])

    row_labels = [f"pid {pid}" for (pid, _, _) in crops]
    col_labels = [short_name(slug) for (_, slug, _) in crops]
    matrix = render_cosine_matrix(row_labels, col_labels, cos_matrix)

    # Stitch strip on top of matrix, padding widths to match
    target_w = max(strip.shape[1], matrix.shape[1])
    def pad_w(img):
        if img.shape[1] < target_w:
            extra = np.full((img.shape[0], target_w - img.shape[1], 3),
                              12, dtype=np.uint8)
            img = np.hstack([img, extra])
        return img
    sep = np.full((10, target_w, 3), 12, dtype=np.uint8)
    composite = np.vstack([pad_w(strip), sep, pad_w(matrix)])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), composite)
    print(f"wrote {args.out} ({composite.shape[1]}x{composite.shape[0]})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
