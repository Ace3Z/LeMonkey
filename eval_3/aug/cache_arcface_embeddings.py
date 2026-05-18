#!/usr/bin/env python3
"""Cache buffalo_l ArcFace embeddings for the eval_3 celeb bank.

Walks both `heldout/` and `scraped/` under the celeb bank, runs RetinaFace
(largest face per image) → ArcFace `buffalo_l` (512-D), L2-normalises,
writes a sibling `<photo>.arcface.npy` per image, and a single manifest
`<output_dir>/celeb_embeddings.json` with the per-photo paths plus a
per-celeb centroid embedding.

Per project CLAUDE.md §8 + §9: rectified-face ArcFace embeddings are the
established InsightFace pattern, this script reuses the buffalo_l
checkpoint that the existing `5_verify_identity.py` already validated
against the 3 IID celebs. No fallbacks — if buffalo_l fails to detect a
face the photo is skipped and reported in the manifest so the failure is
visible (no silent skip).

Usage (Mac or Linux, CPU is enough — ~5-10 min for ~700 photos):

    python eval_3/aug/cache_arcface_embeddings.py \
        --bank-root ~/Downloads/eval3_celebs \
        --output-dir eval_3/aug/stats

The script is idempotent: re-running with the same args is a no-op for
photos that already have a sibling `.arcface.npy`. Pass `--force` to
recompute.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# Heldout dir uses short slugs; scraped dir uses full slugs.
# Maps short → full so the manifest is unified.
SHORT_TO_FULL = {
    "swift": "taylor_swift",
    "obama": "barack_obama",
    "lecun": "yann_lecun",
}


def _build_arcface_app(det_size: int = 320):
    """Load buffalo_l on CPU.

    Returns an InsightFace FaceAnalysis object configured for detection +
    recognition only (skip landmarks/genderage/etc — we don't need them).
    """
    from insightface.app import FaceAnalysis  # local import: heavy

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def _largest_face(faces):
    """Return the largest detected face by bbox area, or None."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _photo_celeb_slug(photo_path: Path, root_short_or_full: str) -> str:
    """Map a photo path to a canonical full-slug celeb id."""
    if root_short_or_full == "heldout":
        # heldout/obama/obama_01.png → barack_obama
        short = photo_path.parent.name
        return SHORT_TO_FULL.get(short, short)
    # scraped/barack_obama/web_obama_007_cos717.jpg → barack_obama
    return photo_path.parent.name


def _embed_one(app, img_bgr) -> np.ndarray | None:
    """Run RetinaFace+ArcFace; return the 512-D L2-normalised embedding or None."""
    faces = app.get(img_bgr)
    face = _largest_face(faces)
    if face is None or not hasattr(face, "normed_embedding"):
        return None
    emb = np.asarray(face.normed_embedding, dtype=np.float32)
    # buffalo_l already L2-normalises `normed_embedding`, but renormalise
    # for safety (cheap, removes float-cast drift).
    norm = float(np.linalg.norm(emb))
    if norm < 1e-6:
        return None
    return emb / norm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", type=Path, required=True,
                        help="Directory containing heldout/ and scraped/ subdirs")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where celeb_embeddings.json gets written")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if .arcface.npy already exists")
    parser.add_argument("--det-size", type=int, default=320,
                        help="RetinaFace det_size (square); buffalo_l recommended >=320")
    args = parser.parse_args()

    if not args.bank_root.is_dir():
        print(f"[ERR] bank-root not found: {args.bank_root}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading buffalo_l (det_size={args.det_size}x{args.det_size}) ...")
    app = _build_arcface_app(det_size=args.det_size)

    import cv2

    manifest: dict[str, dict] = {}  # celeb_slug → {photos: {...}, centroid: [...]}
    failed: list[str] = []
    cached_hits = 0
    computed = 0
    t0 = time.time()

    photo_dirs = []
    for set_name in ("heldout", "scraped"):
        set_dir = args.bank_root / set_name
        if not set_dir.is_dir():
            print(f"[WARN] {set_name} dir missing under bank-root: expected={set_dir}, "
                  f"got=missing, fallback=skip", flush=True)
            continue
        for celeb_dir in sorted(set_dir.iterdir()):
            if not celeb_dir.is_dir():
                continue
            photo_dirs.append((set_name, celeb_dir))

    n_celeb_dirs = len(photo_dirs)
    print(f"[info] found {n_celeb_dirs} celeb directories across heldout + scraped")

    for set_name, celeb_dir in photo_dirs:
        celeb_slug = _photo_celeb_slug(celeb_dir / "_dummy.png", set_name)
        photos = sorted(p for p in celeb_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not photos:
            print(f"[WARN] {celeb_slug} ({set_name}): expected>=1 photos, got=0, fallback=skip",
                  flush=True)
            continue

        if celeb_slug not in manifest:
            manifest[celeb_slug] = {"photos": {}, "centroid": None,
                                    "n_photos": 0, "n_failed": 0}

        cs = manifest[celeb_slug]
        for photo in photos:
            cache_path = photo.with_suffix(photo.suffix + ".arcface.npy")
            rel_photo = str(photo.relative_to(args.bank_root))

            if cache_path.exists() and not args.force:
                emb = np.load(cache_path)
                cached_hits += 1
            else:
                img = cv2.imread(str(photo))
                if img is None:
                    failed.append(f"{rel_photo} (cv2 decode failed)")
                    cs["n_failed"] += 1
                    continue
                emb = _embed_one(app, img)
                if emb is None:
                    failed.append(f"{rel_photo} (no face detected)")
                    cs["n_failed"] += 1
                    continue
                np.save(cache_path, emb)
                computed += 1

            cs["photos"][rel_photo] = str(cache_path.relative_to(args.bank_root))
            cs["n_photos"] += 1

        if cs["n_photos"] > 0:
            embs = np.stack([
                np.load(args.bank_root / rel)
                for rel in cs["photos"].values()
            ])
            centroid = embs.mean(axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-6)
            cs["centroid"] = centroid.tolist()

    manifest_path = args.output_dir / "celeb_embeddings.json"
    manifest_path.write_text(json.dumps({
        "bank_root": str(args.bank_root),
        "buffalo_l_det_size": args.det_size,
        "schema_version": 1,
        "celebs": manifest,
        "failed": failed,
    }, indent=2))

    elapsed = time.time() - t0
    n_celebs = len(manifest)
    n_total = sum(c["n_photos"] for c in manifest.values())
    print(f"[done] {n_celebs} celebs, {n_total} embeddings cached "
          f"({computed} computed, {cached_hits} from cache) in {elapsed:.1f}s")
    if failed:
        print(f"[warn] {len(failed)} photos failed (see manifest['failed'])")
    print(f"[done] manifest: {manifest_path}")

    # Sanity check: print cosine same-vs-different for the 3 IID celebs.
    print("\n[sanity] same-vs-different cosines (IID celebs):")
    iid = ["taylor_swift", "barack_obama", "yann_lecun"]
    embs_by_celeb = {}
    for c in iid:
        if c not in manifest or not manifest[c]["photos"]:
            print(f"  {c}: MISSING — bank-root has no usable photos")
            continue
        rels = list(manifest[c]["photos"].values())
        embs_by_celeb[c] = np.stack([np.load(args.bank_root / r) for r in rels])

    if len(embs_by_celeb) >= 2:
        for c, embs in embs_by_celeb.items():
            if embs.shape[0] >= 2:
                same = (embs @ embs.T)[np.triu_indices(embs.shape[0], k=1)].mean()
                print(f"  same({c:14s}): cos_mean = {same:+.3f}  (n_pairs={embs.shape[0]*(embs.shape[0]-1)//2})")
        keys = list(embs_by_celeb.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = embs_by_celeb[keys[i]], embs_by_celeb[keys[j]]
                diff = (a @ b.T).mean()
                print(f"  diff({keys[i]:14s} vs {keys[j]:14s}): cos_mean = {diff:+.3f}")
        print("\n[interpretation]")
        print("  same > 0.5 and diff < 0.2 → ArcFace is reliable on this bank, proceed to face-labelling.")
        print("  same < 0.4 or diff > 0.3 → printed photos too low-quality; fall back to Path C+ (warm-VLM extension).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
