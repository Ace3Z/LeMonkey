#!/usr/bin/env python3
"""Build an A5 50-portrait sample PDF for Eval 3 eval-day reference.

Selection:
  - 3 mandatory celebs from datasets/eval3_celebs/heldout/ (the IID
    celebs' OOD photos — yann_lecun, barack_obama, taylor_swift)
  - 47 random celebs from datasets/eval3_celebs/scraped/ (excluding the
    3 above to avoid duplicates)
  - 1 portrait+color photo per celeb (h > w AND HSV.saturation.mean ≥ 60
    — same gate the augmentation bank loader uses)

Each PDF page is A5 portrait (148 × 210 mm) with the photo filling most
of the page and the celebrity's display name printed underneath. 50
pages total, alphabetized by display name for easy lookup.

Usage:
    make_celeb_sample_pdf.py [--seed 42] [--out Eval_3_Sample_Celebrity_Images.pdf]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# A5 in inches (148 mm × 210 mm)
A5_W = 148 / 25.4
A5_H = 210 / 25.4

HELDOUT_MAP = {
    "lecun": "Yann LeCun",
    "obama": "Barack Obama",
    "swift": "Taylor Swift",
}


def slug_to_name(slug: str) -> str:
    """yann_lecun -> Yann LeCun (preserves CamelCase capitalizations
    inside tokens like 'mcavoy' -> 'McAvoy' if user explicitly wrote it
    that way; otherwise titlecase per word)."""
    parts = slug.split("_")
    out = []
    for p in parts:
        if p.isupper() or any(c.isupper() for c in p[1:]):
            out.append(p)
        else:
            out.append(p.capitalize())
    return " ".join(out).replace("-", "-")


def is_portrait_and_color(p: Path, min_mean_sat: float = 60.0) -> bool:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    if w >= h:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean()) >= min_mean_sat


def pick_photo(d: Path, rng: random.Random, *, strict: bool = True) -> Path | None:
    """Pick a portrait+color photo from `d`. If strict and none pass,
    fall back to any image (lets us still include the heldout celebs
    whose photos may not pass the strict gate)."""
    cands = sorted(p for p in d.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not cands:
        return None
    good = [p for p in cands if is_portrait_and_color(p)]
    pool = good if good else (cands if not strict else [])
    if not pool:
        return None
    return rng.choice(pool)


def crop_to_aspect(img, target_aspect: float):
    """Center-crop `img` (H, W, 3) to exactly `target_aspect` = W/H.
    No padding, no stretching — loses small slivers from top/bottom or
    left/right depending on which dimension is over-long."""
    h, w = img.shape[:2]
    cur = w / h
    if abs(cur - target_aspect) < 1e-4:
        return img
    if cur > target_aspect:
        # too wide → crop sides
        new_w = int(round(h * target_aspect))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    else:
        # too tall → crop top/bottom (bias upward slightly to keep face)
        new_h = int(round(w / target_aspect))
        # Bias the crop so the upper third is kept (face usually upper third)
        y0 = max(0, (h - new_h) // 3)
        return img[y0:y0 + new_h]


def draw_page(pdf: PdfPages, photo: Path) -> None:
    """A5 full-bleed: image fills the entire page exactly, no border, no
    text. Matches the TOY PDF format ("no white border")."""
    img = cv2.cvtColor(cv2.imread(str(photo), cv2.IMREAD_COLOR),
                       cv2.COLOR_BGR2RGB)
    img = crop_to_aspect(img, target_aspect=A5_W / A5_H)
    fig = plt.figure(figsize=(A5_W, A5_H))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])    # full bleed
    ax.imshow(img, aspect="auto", interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    pdf.savefig(fig, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scraped-root", type=Path,
                   default=Path("datasets/eval3_celebs/scraped"))
    p.add_argument("--heldout-root", type=Path,
                   default=Path("datasets/eval3_celebs/heldout"))
    p.add_argument("--out", type=Path,
                   default=Path("Eval_3_Sample_Celebrity_Images.pdf"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-random", type=int, default=47)
    args = p.parse_args()

    rng = random.Random(args.seed)

    # 1. Heldout celebs first
    iid_picks: list[tuple[str, Path]] = []
    for slug, name in HELDOUT_MAP.items():
        d = args.heldout_root / slug
        if not d.is_dir():
            print(f"[FATAL] missing heldout dir: {d}", file=sys.stderr)
            return 2
        photo = pick_photo(d, rng, strict=False)
        if photo is None:
            print(f"[FATAL] no photos in {d}", file=sys.stderr)
            return 2
        iid_picks.append((name, photo))
        print(f"  IID    {name:25}  {photo.name}")

    # 2. 47 random scraped celebs, excluding the 3 IID slugs in scraped form
    IID_SCRAPED_SLUGS = {"yann_lecun", "barack_obama", "taylor_swift"}
    all_scraped = sorted(
        d for d in args.scraped_root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and d.name not in IID_SCRAPED_SLUGS
    )
    if len(all_scraped) < args.n_random:
        print(f"[FATAL] only {len(all_scraped)} scraped celebs, need {args.n_random}",
              file=sys.stderr)
        return 2

    chosen = rng.sample(all_scraped, args.n_random)
    ood_picks: list[tuple[str, Path]] = []
    for d in chosen:
        photo = pick_photo(d, rng, strict=True)
        if photo is None:
            # Fall back to any image if strict gate fails (rare — we
            # confirmed the bank has portrait+color photos for all 192).
            photo = pick_photo(d, rng, strict=False)
        if photo is None:
            print(f"  [WARN] no usable photo in {d}, skipping")
            continue
        ood_picks.append((slug_to_name(d.name), photo))

    # 3. Combine, alphabetize, write PDF
    all_picks = sorted(iid_picks + ood_picks, key=lambda x: x[0].lower())
    print(f"\nbuilding PDF with {len(all_picks)} portraits → {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        for i, (name, photo) in enumerate(all_picks, start=1):
            draw_page(pdf, photo)
            if i % 10 == 0:
                print(f"  rendered {i}/{len(all_picks)}")
    print(f"\nDONE — {args.out} ({args.out.stat().st_size/1024/1024:.1f} MB)")

    # Also write an index TXT next to the PDF for printable lookup
    index_path = args.out.with_suffix(".index.txt")
    with index_path.open("w") as f:
        f.write(f"# {args.out.name} — celeb index (alphabetical, 1-based page number)\n")
        f.write(f"# Seed: {args.seed}\n\n")
        for i, (name, photo) in enumerate(all_picks, start=1):
            f.write(f"  page {i:2d}: {name:30}  ({photo.name})\n")
    print(f"index    → {index_path}")

    # Also pack a zip of the original source images + manifest CSV
    # (so a downstream consumer can recrop / relabel without re-running
    # selection logic). Image filenames in the zip are `NN_<slug>.<ext>`
    # where NN is the alphabetized 1-based page number (matches the PDF).
    import csv, shutil, zipfile
    zip_path = args.out.with_suffix(".zip")
    name_to_slug = {n: s for s, n in HELDOUT_MAP.items()}
    for d in chosen:
        name_to_slug[slug_to_name(d.name)] = d.name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. manifest.csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["page", "display_name", "slug", "image_in_zip",
                    "source_path"])
        for i, (name, photo) in enumerate(all_picks, start=1):
            slug = name_to_slug.get(name, name.lower().replace(" ", "_"))
            ext = photo.suffix.lower()
            zname = f"images/{i:02d}_{slug}{ext}"
            w.writerow([i, name, slug, zname, str(photo)])
        zf.writestr("manifest.csv", buf.getvalue())
        # 2. images
        for i, (name, photo) in enumerate(all_picks, start=1):
            slug = name_to_slug.get(name, name.lower().replace(" ", "_"))
            ext = photo.suffix.lower()
            zf.write(photo, arcname=f"images/{i:02d}_{slug}{ext}")
        # 3. lightweight README
        zf.writestr("README.txt",
            f"Eval 3 sample celebrities — 50 portraits used in "
            f"{args.out.name}.\n\n"
            f"images/  : original source photos, numbered to match the "
            f"alphabetical\n"
            f"           page order in the PDF.\n"
            f"manifest.csv : page → display name → slug → in-zip filename "
            f"→ original source path on the dev box.\n\n"
            f"Seed = {args.seed}. Selection is reproducible by re-running "
            f"eval_3/scripts/celebs/make_celeb_sample_pdf.py with the same seed.\n")
    print(f"zip      → {zip_path} ({zip_path.stat().st_size/1024/1024:.1f} MB)")
    print("\nIndex (alphabetical):")
    for i, (name, _) in enumerate(all_picks, start=1):
        print(f"  page {i:2d}: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
