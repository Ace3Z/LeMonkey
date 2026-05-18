#!/usr/bin/env python
"""
Build a manifest from the eval3_celebs/ scraped dataset (the ~200-celebrity
collection at /Users/hansbaumannortiz/Downloads/eval3_celebs/ locally or
/home/ubuntu/datasets/eval3_celebs/ on AWS).

Differs from build_celeb_dataset.py (VGGFace2):
  - No identity_meta CSV — slug IS the identity; "Barack Obama" derived from
    snake_case "barack_obama" via .title().
  - Two source dirs: scraped/ (most data) and heldout/ (3 small dirs).
    heldout/{lecun,obama,swift} are MERGED into scraped/{yann_lecun,
    barack_obama,taylor_swift} per user spec ("toy identities always in train").
  - Marks those 3 as is_toy=True so build_llava_json.py --toy can filter to them.
  - Strips .DS_Store and other dotfiles.

Inputs (override with --data-root):
    <data-root>/scraped/<slug>/*.jpg
    <data-root>/heldout/{lecun,obama,swift}/*.png

Output:
    <data-root>/manifests/manifest.parquet

Manifest columns (compatible with build_llava_json.py):
    class_id      str   "barack_obama"
    name          str   "Barack Obama"
    n_images      int   12
    gender        str   "?"   (unknown for scraped data)
    source_split  str   "train"   (single pool; splits happen in build_llava_json)
    image_paths   list[str]   absolute paths
    is_toy        bool  True for {yann_lecun, barack_obama, taylor_swift}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATA_ROOT = Path("/Users/hansbaumannortiz/Downloads/eval3_celebs")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# heldout/<src> images get appended to scraped/<target> identity
HELDOUT_TO_SCRAPED = {
    "lecun": "yann_lecun",
    "obama": "barack_obama",
    "swift": "taylor_swift",
}
TOY_SLUGS = set(HELDOUT_TO_SCRAPED.values())


def slug_to_name(slug: str) -> str:
    """barack_obama -> Barack Obama, anya_taylor-joy -> Anya Taylor-Joy."""
    return slug.replace("_", " ").title()


def list_images(d: Path) -> list[str]:
    if not d.is_dir():
        return []
    return sorted(
        str(p) for p in d.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in IMAGE_EXTS
    )


def build_manifest(data_root: Path, min_images: int) -> pd.DataFrame:
    scraped_root = data_root / "scraped"
    heldout_root = data_root / "heldout"

    print(f"[build] data root: {data_root}")
    print(f"[build] min imgs/id: {min_images}")

    # 1) scan scraped/
    ids: dict[str, list[str]] = {}
    if not scraped_root.is_dir():
        raise FileNotFoundError(f"scraped/ not found at {scraped_root}")
    for d in sorted(scraped_root.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name.startswith("_"):
            continue
        imgs = list_images(d)
        if imgs:
            ids[d.name] = imgs
    print(f"[build] scraped/: {len(ids)} identities, "
          f"{sum(len(v) for v in ids.values())} images")

    # 2) merge heldout/
    n_heldout_added = 0
    if heldout_root.is_dir():
        for d in sorted(heldout_root.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            target_slug = HELDOUT_TO_SCRAPED.get(d.name, d.name)
            imgs = list_images(d)
            if not imgs:
                continue
            ids.setdefault(target_slug, []).extend(imgs)
            n_heldout_added += len(imgs)
            print(f"[build]   heldout/{d.name} → {target_slug}: +{len(imgs)} images")
    print(f"[build] heldout/: merged {n_heldout_added} images")

    # 3) build rows
    rows = []
    skipped_low = 0
    for slug, paths in sorted(ids.items()):
        if len(paths) < min_images:
            skipped_low += 1
            continue
        rows.append({
            "class_id": slug,
            "name": slug_to_name(slug),
            "n_images": len(paths),
            "gender": "?",
            "source_split": "train",
            "image_paths": paths,
            "is_toy": slug in TOY_SLUGS,
        })

    df = pd.DataFrame(rows)
    print(f"[build] kept: {len(df)} identities, {df['n_images'].sum()} images")
    print(f"[build] skipped (low image count <{min_images}): {skipped_low}")
    print(f"[build] toy identities: {sorted(df[df['is_toy']]['class_id'].tolist())}")

    if not df.empty:
        print("\n[build] sample rows:")
        print(df[["class_id", "name", "n_images", "is_toy"]].head(10).to_string(index=False))
        print(f"\n[build] image-count distribution:")
        print(df["n_images"].describe().to_string())

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <data-root>/manifests/manifest.parquet")
    parser.add_argument("--min-images", type=int, default=1,
                        help="Drop identities with fewer than this many images")
    args = parser.parse_args()

    if args.out is None:
        args.out = args.data_root / "manifests/manifest.parquet"

    df = build_manifest(args.data_root, args.min_images)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\n[build] wrote {args.out}  ({args.out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
