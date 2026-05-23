#!/usr/bin/env python3
"""Generate Pi0.5 VL cotrain's bbox-grounded face VQA pairs from the 193-celeb scraped bank.

This is the deliverable per the ObjectVLA spec , but written so a teammate
can run it on a CPU host directly when Darius is unavailable.

For each photo in `eval3_celebs/scraped/<slug>/<photo>`:
  1. Run InsightFace RetinaFace → bbox of largest face.
  2. Normalize bbox to [x1, y1, x2, y2] in [0,1] image coords.
  3. Emit ~10 caption variants in the 50/30/10/10 mix:
       50% — location-explicit (ObjectVLA-style):
                "The face of {Name} is at [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]."
       30% — Q&A grounded:
                prompt "Who is the person at [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]?"
                target "{Name}"
       10% — Q&A open:
                prompt "Who is the person in this image?"
                target "{Name}"
       10% — Caption form, no bbox:
                "{Name} is in this image."

Output parquet has columns:
    image_path   (str)
    prompt       (str)  — what goes into PaliGemma processor as text
    target       (str)  — what goes as suffix= (the answer/CE target)
    bbox         (list[float], 4 entries)  — normalized xyxy
    celeb        (str)  — slug like "barack_obama"
    caption_type (str)  — "location_explicit" | "qa_grounded" | "qa_open" | "caption"

Per the ObjectVLA spec: ObjectVLA's +45pp OOD lift depends on the BBOX
GROUNDING being present in the supervision. The 50/30/10/10 mix is from the
canonical spec — do not improvise.

Per: photos without a detected face are skipped with [WARN].
Per: no Claude attribution anywhere in output.
Per the triple-source-defaults rule: numerical defaults inline-cited.

USAGE
=====

On edna (or anywhere with `lemonkey-arcface` conda env / InsightFace + buffalo_l):

    python eval_3/scripts/pi05_vl_cotrain/generate_vl_pairs.py \\
        --scraped-root ~/LeMonkey/datasets/eval3_celebs/scraped \\
        --out manifests/so101_eval3_broad_grounding.parquet \\
        --captions-per-photo 10 \\
        --push-repo HBOrtiz/so101_eval3_broad_grounding

Expected output: ~193 celebs × ~7-10 photos × 10 captions = ~15,000 rows.
~5-10 min on edna's 128-core CPU box (RetinaFace is the bottleneck).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


# Caption mix per canonical the ObjectVLA spec.
# 10 captions per photo: 5 location-explicit, 3 Q&A grounded, 1 Q&A open, 1 caption.
DEFAULT_CAPTIONS_PER_PHOTO = 10
DEFAULT_MIX = {
    "location_explicit": 5,
    "qa_grounded":       3,
    "qa_open":           1,
    "caption":           1,
}

# Multiple phrasing variants per caption type for diversity.
# Each format string uses {name} and (where bbox is present) {bbox_str}.
LOCATION_TEMPLATES = [
    "The face of {name} is at {bbox_str}.",
    "{name}'s face is located at {bbox_str}.",
    "At {bbox_str} there is {name}.",
    "{name} appears at {bbox_str} in this image.",
    "The bounding box of {name}'s face is {bbox_str}.",
]
QA_GROUNDED_PROMPTS = [
    "Who is the person at {bbox_str}?",
    "Whose face is at {bbox_str}?",
    "Identify the person located at {bbox_str}.",
]
QA_OPEN_PROMPTS = [
    "Who is the person in this image?",
    "Who is shown in this photo?",
    "Identify the person in the picture.",
]
CAPTION_TEMPLATES = [
    "{name} is in this image.",
    "This is a photo of {name}.",
    "The image shows {name}.",
]


def slug_to_name(slug: str) -> str:
    """barack_obama → Barack Obama; anya_taylor-joy → Anya Taylor-Joy."""
    parts = slug.split("_")
    out = []
    for part in parts:
        # Preserve hyphens (Taylor-Joy) while title-casing each segment.
        sub_parts = part.split("-")
        sub_parts = [s[:1].upper() + s[1:] if s else s for s in sub_parts]
        out.append("-".join(sub_parts))
    return " ".join(out)


def _build_arcface_app(det_size: int = 320):
    """Load buffalo_l (RetinaFace + ArcFace) — we only use detection here."""
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
        allowed_modules=["detection"],  # bbox only — recognition not needed here
        providers=providers,
    )
    app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
                det_size=(det_size, det_size))
    print(f"[info] buffalo_l RetinaFace loaded on {providers[0]}", flush=True)
    return app


def _largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def detect_bbox_normalized(app, img_bgr) -> tuple[float, float, float, float] | None:
    """RetinaFace on a BGR image; return (x1, y1, x2, y2) normalized to [0,1]."""
    faces = app.get(img_bgr)
    face = _largest_face(faces)
    if face is None:
        return None
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = face.bbox.tolist()
    return (max(0.0, x1 / w), max(0.0, y1 / h),
            min(1.0, x2 / w), min(1.0, y2 / h))


def fmt_bbox(bbox: tuple[float, float, float, float]) -> str:
    """Format bbox as [x1,y1,x2,y2] with 2 decimal places for prompt embedding."""
    return f"[{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}]"


def emit_captions(name: str, bbox: tuple[float, float, float, float],
                  celeb_slug: str, image_path: str, rng: random.Random,
                  mix: dict[str, int]) -> list[dict]:
    """Produce N caption rows for one photo, sampled from the templates."""
    bbox_str = fmt_bbox(bbox)
    rows: list[dict] = []

    for _ in range(mix["location_explicit"]):
        tpl = rng.choice(LOCATION_TEMPLATES)
        rows.append({
            "image_path": image_path,
            "prompt": "<image>Describe this image.\n",
            "target": tpl.format(name=name, bbox_str=bbox_str),
            "bbox": list(bbox),
            "celeb": celeb_slug,
            "caption_type": "location_explicit",
        })

    for _ in range(mix["qa_grounded"]):
        q = rng.choice(QA_GROUNDED_PROMPTS).format(bbox_str=bbox_str)
        rows.append({
            "image_path": image_path,
            "prompt": f"<image>{q}\n",
            "target": name,
            "bbox": list(bbox),
            "celeb": celeb_slug,
            "caption_type": "qa_grounded",
        })

    for _ in range(mix["qa_open"]):
        q = rng.choice(QA_OPEN_PROMPTS)
        rows.append({
            "image_path": image_path,
            "prompt": f"<image>{q}\n",
            "target": name,
            "bbox": list(bbox),  # bbox present in row metadata but NOT in prompt
            "celeb": celeb_slug,
            "caption_type": "qa_open",
        })

    for _ in range(mix["caption"]):
        tpl = rng.choice(CAPTION_TEMPLATES)
        rows.append({
            "image_path": image_path,
            "prompt": "<image>Describe this image.\n",
            "target": tpl.format(name=name),
            "bbox": list(bbox),
            "celeb": celeb_slug,
            "caption_type": "caption",
        })

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scraped-root", type=Path, required=True,
                        help="Directory containing <celeb_slug>/<photo>.{jpg,png}")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output parquet path")
    parser.add_argument("--captions-per-photo", type=int,
                        default=DEFAULT_CAPTIONS_PER_PHOTO,
                        help="N captions emitted per photo (mix per CANONICAL_MIX)")
    parser.add_argument("--include-iid", action="store_true",
                        help="Also walk heldout/ subdir (the 3 IID celebs). "
                             "Off by default — 200-celeb training set's "
                             "augmentation pipeline already handles IID.")
    parser.add_argument("--push-repo", default=None,
                        help="If set, push to this HF dataset repo after writing")
    parser.add_argument("--push-token", default=None,
                        help="HF token (or set HF_TOKEN env)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument("--max-photos-per-celeb", type=int, default=None,
                        help="If set, cap photos per celeb (for smoke runs)")
    args = parser.parse_args()

    try:
        import cv2
        import pandas as pd
    except ImportError as e:
        print(f"[ERR] missing dependency: {e}", file=sys.stderr)
        return 2

    if not args.scraped_root.is_dir():
        print(f"[ERR] --scraped-root not found: {args.scraped_root}", file=sys.stderr)
        return 2

    if args.captions_per_photo != DEFAULT_CAPTIONS_PER_PHOTO:
        # Scale the mix proportionally if user wants more/fewer captions.
        scale = args.captions_per_photo / DEFAULT_CAPTIONS_PER_PHOTO
        mix = {k: max(1, round(v * scale)) if v > 0 else 0
               for k, v in DEFAULT_MIX.items()}
        # Ensure we hit exactly args.captions_per_photo by adjusting the
        # dominant bucket.
        total = sum(mix.values())
        diff = args.captions_per_photo - total
        if diff != 0:
            mix["location_explicit"] = max(0, mix["location_explicit"] + diff)
        print(f"[info] scaled mix: {mix}")
    else:
        mix = DEFAULT_MIX.copy()

    rng = random.Random(args.seed)

    # Discover celeb dirs.
    celeb_dirs: list[Path] = []
    if args.include_iid and (args.scraped_root.parent / "heldout").is_dir():
        for d in sorted((args.scraped_root.parent / "heldout").iterdir()):
            if d.is_dir():
                celeb_dirs.append(d)
    for d in sorted(args.scraped_root.iterdir()):
        if d.is_dir():
            celeb_dirs.append(d)
    print(f"[info] discovered {len(celeb_dirs)} celeb directories")

    app = _build_arcface_app(det_size=args.det_size)

    rows: list[dict] = []
    n_no_face = 0
    n_photos_kept = 0
    t0 = time.time()
    last_log = t0

    for ci, celeb_dir in enumerate(celeb_dirs):
        slug = celeb_dir.name
        name = slug_to_name(slug)
        photos = sorted(p for p in celeb_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if args.max_photos_per_celeb:
            photos = photos[:args.max_photos_per_celeb]
        if not photos:
            print(f"[WARN] {slug}: expected>=1 photos, got=0, fallback=skip celeb",
                  flush=True)
            continue

        for photo in photos:
            img = cv2.imread(str(photo))
            if img is None:
                print(f"[WARN] {photo}: expected readable image, got=cv2 decode "
                      f"failure, fallback=skip", flush=True)
                continue
            bbox = detect_bbox_normalized(app, img)
            if bbox is None:
                n_no_face += 1
                print(f"[WARN] {photo}: expected detected face, got=none, "
                      f"fallback=skip photo", flush=True)
                continue

            photo_rows = emit_captions(
                name=name, bbox=bbox,
                celeb_slug=slug,
                image_path=str(photo.resolve()),
                rng=rng, mix=mix,
            )
            rows.extend(photo_rows)
            n_photos_kept += 1

        if time.time() - last_log > 30.0:
            rate = (ci + 1) / (time.time() - t0)
            eta = (len(celeb_dirs) - (ci + 1)) / max(rate, 1e-6)
            print(f"[info] celeb {ci+1}/{len(celeb_dirs)} ({slug}) "
                  f"rows_so_far={len(rows)} ETA={eta:.0f}s", flush=True)
            last_log = time.time()

    elapsed = time.time() - t0
    print(f"\n[done] {len(rows)} VL-pair rows from {n_photos_kept} photos "
          f"across {len(celeb_dirs)} celebs in {elapsed:.0f}s")
    if n_no_face > 0:
        print(f"[WARN] {n_no_face} photos skipped (no face detected). "
              f"Output parquet excludes these.", flush=True)

    # Caption-type distribution sanity check.
    df = __import__("pandas").DataFrame(rows)
    print(f"\n[summary] caption-type distribution:")
    for ct, count in df["caption_type"].value_counts().items():
        print(f"  {ct:18s}: {count:6d} ({count/len(df)*100:.1f}%)")
    print(f"  total: {len(df)} rows ({len(df)/n_photos_kept:.1f} per photo)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\n[done] parquet → {args.out}")

    if args.push_repo:
        push_to_hf(args.out, args.push_repo,
                   token=args.push_token or os.environ.get("HF_TOKEN"))

    return 0


def push_to_hf(parquet_path: Path, repo_id: str, token: str | None) -> None:
    """Push the manifest parquet to an HF dataset repo."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERR] huggingface_hub not installed; skip push", file=sys.stderr)
        return
    if not token:
        print("[WARN] no HF token; expected --push-token or HF_TOKEN env var, "
              "got=None, fallback=skip push", flush=True)
        return

    api = HfApi(token=token)
    # Create repo if it doesn't exist.
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    # Upload single parquet file as the canonical artifact.
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo="vl_pairs.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Pi0.5 VL cotrain VL pairs (bbox-grounded face VQA, 4 caption forms)",
    )
    print(f"[done] pushed to https://huggingface.co/datasets/{repo_id}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
