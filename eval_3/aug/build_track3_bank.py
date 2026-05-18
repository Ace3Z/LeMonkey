#!/usr/bin/env python3
"""Build the Track 3 photo bank: 8 photos per IID celeb.

Source layout:
    datasets/eval3_celebs/heldout/<short>/       (the printed photos used at recording)
        swift/   {5 PNGs}      → swift_01..05.png
        obama/   {4 PNGs}      → obama_01..04.png
        lecun/   {5 PNGs}      → lecun_01..05.png
    datasets/eval3_celebs/scraped/<full>/        (the larger scrape bank)
        taylor_swift/  {5 JPGs}
        barack_obama/  {12 JPGs}
        yann_lecun/    {24 JPGs}

Output layout (per Track 3 spec: 8 photos/celeb):
    datasets/eval3_celebs/track3_bank/<full>/
        taylor_swift/   heldout_01..05.png  +  scraped_01..03.jpg   = 8
        barack_obama/   heldout_01..04.png  +  scraped_01..04.jpg   = 8
        yann_lecun/     heldout_01..05.png  +  scraped_01..03.jpg   = 8

Selection of the scraped picks is by face-quality score:
    score = det_score · (1 if face_frac >= 5% of image else 0.5) ·
            (1 if exactly 1 face else 0.4)

Re-running is idempotent: overwrites only the scraped picks.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# Local — reuse the cached buffalo_l face app from 4_inpaint_video.py
import importlib.util as _ilu
_HERE = Path(__file__).resolve().parent
_spec = _ilu.spec_from_file_location("_v4", str(_HERE / "4_inpaint_video.py"))
_v4 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_v4)
_get_face_app = _v4._get_face_app
_is_color_photo = _v4._is_color_photo

CELEBS = [
    # (short_name_in_heldout, full_name_in_scraped, n_heldout_expected, n_scraped_needed)
    ("swift", "taylor_swift", 5, 3),
    ("obama", "barack_obama", 4, 4),
    ("lecun", "yann_lecun",   5, 3),
]


def score_scraped_photo(path: Path, face_app) -> tuple[float, dict]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return -1.0, {"err": "cannot_decode"}
    h, w = img.shape[:2]
    faces = face_app.get(img)
    if not faces:
        return -1.0, {"err": "no_face"}
    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    fa = (largest.bbox[2]-largest.bbox[0]) * (largest.bbox[3]-largest.bbox[1])
    face_frac = float(fa) / float(h * w)
    det = float(largest.det_score)
    n_faces = len(faces)
    color_ok = _is_color_photo(path)
    score = det \
            * (1.0 if face_frac >= 0.05 else 0.5) \
            * (1.0 if n_faces == 1 else 0.4) \
            * (1.0 if color_ok else 0.6)
    meta = {
        "det_score": det,
        "face_frac": face_frac,
        "n_faces": n_faces,
        "color_ok": color_ok,
        "wxh": [w, h],
    }
    return score, meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--heldout-root", type=Path,
                   default=Path("/home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3_celebs/heldout"))
    p.add_argument("--scraped-root", type=Path,
                   default=Path("/home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3_celebs/scraped"))
    p.add_argument("--out-root", type=Path,
                   default=Path("/home/rohamzn/ETH_Uni/LeMonkey/datasets/eval3_celebs/track3_bank"))
    args = p.parse_args()

    face_app = _get_face_app()
    args.out_root.mkdir(parents=True, exist_ok=True)

    grand = []
    for short, full, n_heldout, n_scraped in CELEBS:
        print(f"\n=== {full} ({short}) ===", flush=True)
        out_dir = args.out_root / full
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy heldout photos verbatim
        heldout_dir = args.heldout_root / short
        heldout_photos = sorted(heldout_dir.glob("*.png"))
        if len(heldout_photos) != n_heldout:
            print(f"  [WARN] expected_heldout={n_heldout}, got={len(heldout_photos)} "
                  f"under {heldout_dir} — using what's there", flush=True)
        for i, src in enumerate(heldout_photos, start=1):
            dst = out_dir / f"heldout_{i:02d}.png"
            shutil.copy2(src, dst)
            print(f"  heldout_{i:02d}.png  ← {src.name}")

        # 2. Score + pick top-K scraped photos by face quality
        scraped_dir = args.scraped_root / full
        scraped_candidates = sorted(
            c for c in (
                list(scraped_dir.glob("*.jpg"))
                + list(scraped_dir.glob("*.jpeg"))
                + list(scraped_dir.glob("*.png"))
            )
            # face_crop_* are face-only crops of other scraped files —
            # prefer the original whole-portrait versions so the photo bank
            # matches the eval-day printed-photo style.
            if not c.name.startswith("face_crop_")
        )
        print(f"  scoring {len(scraped_candidates)} scraped candidates...")
        scored = []
        for c in scraped_candidates:
            score, meta = score_scraped_photo(c, face_app)
            scored.append((c, score, meta))
            print(f"    {c.name:<40} score={score:6.3f}  meta={meta}")
        scored.sort(key=lambda t: -t[1])

        picked = scored[:n_scraped]
        if len(picked) < n_scraped:
            print(f"  [WARN] need {n_scraped} scraped photos but only "
                  f"{len(picked)} candidates available — bank will be short", flush=True)

        # Wipe any prior scraped_* in out_dir so re-running is clean
        for old in out_dir.glob("scraped_*"):
            old.unlink()

        for i, (src, score, meta) in enumerate(picked, start=1):
            ext = src.suffix.lower()
            dst = out_dir / f"scraped_{i:02d}{ext}"
            shutil.copy2(src, dst)
            print(f"  scraped_{i:02d}{ext}  ← {src.name}  (score={score:.3f})")

        grand.append({
            "celeb": full,
            "n_heldout": len(heldout_photos),
            "n_scraped_picked": len(picked),
            "scraped_picks": [str(s[0]) for s in picked],
            "scraped_rejected": [str(s[0]) for s in scored[n_scraped:]],
        })

    # Print summary
    print("\n=== summary ===")
    total = 0
    for g in grand:
        n_total = g["n_heldout"] + g["n_scraped_picked"]
        total += n_total
        print(f"  {g['celeb']:<15}  heldout={g['n_heldout']}  "
              f"scraped={g['n_scraped_picked']}  total={n_total}")
    print(f"  bank total: {total} photos across 3 celebs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
