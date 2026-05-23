#!/usr/bin/env python3
"""Face-crop landscape+color photos to portrait+color around the largest face.

Used to recover the 11 missing celebs whose photos passed the saturation
gate (HSV.sat.mean() >= 60) but failed the portrait gate (h > w) in
load_photo_bank(). We detect the face, pad to a 3:4 portrait box around
it, and save as a new file alongside the original.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


CELEBS = [
    "andrej_karpathy", "clement_delangue", "drake",
    "lebron_james", "marc_pollefeys", "marco_hutter",
    "oier_mees", "roland_siegwart", "sergey_levine",
    "stan_wawrinka", "yann_lecun",
]


def is_color_photo(img_bgr: np.ndarray, min_mean_sat: float = 60.0) -> bool:
    if img_bgr is None or img_bgr.ndim != 3:
        return False
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean()) >= min_mean_sat


def face_crop_to_portrait(img_bgr: np.ndarray, faces, aspect: float = 3 / 4,
                          pad_ratio: float = 1.6) -> np.ndarray | None:
    """Crop the largest face to a portrait (W:H = aspect, e.g. 3:4) box.

    pad_ratio controls how much head+shoulders context to include
    (1.6 = ~head + shoulders)."""
    if not faces:
        return None
    # Pick the largest face by bbox area
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return (x2 - x1) * (y2 - y1)
    f = max(faces, key=area)
    x1, y1, x2, y2 = f.bbox
    fx, fy = (x1 + x2) / 2, (y1 + y2) / 2
    fw, fh = x2 - x1, y2 - y1
    H, W = img_bgr.shape[:2]
    target_h = fh * pad_ratio * 1.5     # face is ~67% of crop height
    target_w = target_h * aspect
    if target_w > W:
        target_w = W
        target_h = target_w / aspect
    if target_h > H:
        target_h = H
        target_w = target_h * aspect

    cx, cy = fx, fy - fh * 0.1   # nudge up slightly so chin doesn't hit bottom
    x1c = int(round(cx - target_w / 2))
    y1c = int(round(cy - target_h / 2))
    x2c = int(round(x1c + target_w))
    y2c = int(round(y1c + target_h))
    # Clamp
    if x1c < 0:
        x2c -= x1c
        x1c = 0
    if y1c < 0:
        y2c -= y1c
        y1c = 0
    if x2c > W:
        x1c -= (x2c - W)
        x2c = W
    if y2c > H:
        y1c -= (y2c - H)
        y2c = H
    x1c, y1c = max(0, x1c), max(0, y1c)
    crop = img_bgr[y1c:y2c, x1c:x2c]
    return crop


def main() -> int:
    from insightface.app import FaceAnalysis

    root = Path("datasets/eval3_celebs/scraped")
    if not root.is_dir():
        print(f"[FATAL] missing {root}", file=sys.stderr)
        return 2

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    total_kept = 0
    for celeb in CELEBS:
        cdir = root / celeb
        if not cdir.is_dir():
            print(f"  {celeb:30}  MISSING DIR — skipping", flush=True)
            continue
        sources = []
        for p in sorted(cdir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if p.name.startswith("face_crop_"):
                continue   # don't recurse on our own outputs
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            if w < h:
                continue          # already portrait — bank loader keeps these
            if not is_color_photo(img):
                continue          # not color enough
            sources.append((p, img))

        kept = 0
        for src_path, img in sources:
            faces = app.get(img)
            if not faces:
                print(f"    {celeb:25}  {src_path.name}: no face detected — skip",
                      flush=True)
                continue
            crop = face_crop_to_portrait(img, faces)
            if crop is None or crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            if cw >= ch:
                # Safety fallback: force portrait by central-cropping width
                ratio = 3 / 4
                new_w = int(round(ch * ratio))
                x1c = (cw - new_w) // 2
                crop = crop[:, x1c:x1c + new_w]
                ch, cw = crop.shape[:2]
            # Verify crop still passes color filter
            if not is_color_photo(crop):
                print(f"    {celeb:25}  {src_path.name}: crop sat<60 — skip",
                      flush=True)
                continue
            out_path = cdir / f"face_crop_{src_path.stem}.jpg"
            cv2.imwrite(str(out_path), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            kept += 1
        print(f"  {celeb:30}  {kept:2} new portrait crops from "
              f"{len(sources):2} landscape sources", flush=True)
        total_kept += kept

    print(f"\nDONE. {total_kept} new portrait crops written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
