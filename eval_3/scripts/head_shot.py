"""
Scrape clean headshot images per celebrity from Google Images.

Pipeline:
  1. Download a candidate pool from Google (more than we need).
  2. Run OpenCV face detection on each image.
  3. Keep only images with exactly ONE face that fills >=20% of the frame.
  4. Save the top N per celebrity, named <celeb>_1.jpg, <celeb>_2.jpg, ...

Install:
    pip install icrawler opencv-python

Run:
    python celeb_headshots.py
"""

import os
import shutil

import cv2
from icrawler.builtin import BingImageCrawler

CELEBS = [
    # Movie celebs (top 20-ish Hollywood)
    "Tom Hanks",
    "Scarlett Johansson",
    "Denzel Washington",
    "Leonardo DiCaprio",
    "Brad Pitt",
    "Angelina Jolie",
    "Robert Downey Jr",
    "Chris Hemsworth",
    "Chris Evans",
    "Robert De Niro",
    "Al Pacino",
    "Meryl Streep",
    "Jennifer Lawrence",
    "Will Smith",
    "Morgan Freeman",
    "Tom Cruise",
    "Johnny Depp",
    "Natalie Portman",
    "Anne Hathaway",
    "Keanu Reeves",
    "Ryan Reynolds",
    "Emma Stone",
    "Margot Robbie",

    # Tech / AI
    "Elon Musk",
    "Sam Altman",
    "Andrej Karpathy",
    "Oier Mees",
    "Marc Pollefeys",
    "Bill Gates",
    "Steve Jobs",
    "Jensen Huang",
    "Mark Zuckerberg",
    "Jeff Bezos",
    "Tim Cook",
    "Sundar Pichai",
    "Satya Nadella",
    "Demis Hassabis",
    "Yann LeCun",
    "Geoffrey Hinton",
    "Ilya Sutskever",
    "Dario Amodei",

    # Politics
    "Donald Trump",

    # Switzerland
    "Roger Federer",
    "Stan Wawrinka",
    "Bertrand Piccard",
    "Ursula Andress",
    "Granit Xhaka",
    "Xherdan Shaqiri",
]

IMAGES_PER_CELEB = 10
CANDIDATE_POOL = 40           # download this many, then filter down
MIN_FACE_AREA_RATIO = 0.20    # face bbox must cover >=20% of the image
OUTPUT_DIR = "celeb_headshots"

# Haar cascade ships with opencv-python — no extra download needed.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def is_clean_headshot(path: str) -> bool:
    """Return True if image has exactly one large, centered-ish face."""
    img = cv2.imread(path)
    if img is None:
        return False

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) != 1:
        return False

    (_, _, fw, fh) = faces[0]
    face_area = fw * fh
    img_area = w * h
    return (face_area / img_area) >= MIN_FACE_AREA_RATIO


def download_candidates(name: str, folder: str, count: int) -> None:
    crawler = BingImageCrawler(
        storage={"root_dir": folder},
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=2,
    )
    crawler.crawl(
        keyword=f"{name} headshot portrait face closeup",
        max_num=count,
        filters={"type": "photo"},
        file_idx_offset=0,
    )


def filter_and_rename(folder: str, celeb_slug: str, keep: int) -> int:
    """Keep up to `keep` clean headshots; delete the rest. Returns count kept."""
    candidates = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    )

    kept = []
    for path in candidates:
        if len(kept) >= keep:
            os.remove(path)
            continue
        if is_clean_headshot(path):
            kept.append(path)
        else:
            os.remove(path)

    for i, path in enumerate(kept, start=1):
        ext = os.path.splitext(path)[1] or ".jpg"
        new_path = os.path.join(folder, f"{celeb_slug}_{i}{ext}")
        if path != new_path:
            shutil.move(path, new_path)

    return len(kept)


def scrape_celeb(name: str) -> None:
    slug = slugify(name)
    folder = os.path.join(OUTPUT_DIR, slug)

    # Reset folder so reruns don't mix old/new files
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

    print(f"\n=== {name} ===")
    print(f"  downloading {CANDIDATE_POOL} candidates...")
    download_candidates(name, folder, CANDIDATE_POOL)

    print(f"  filtering for clean headshots...")
    kept = filter_and_rename(folder, slug, IMAGES_PER_CELEB)
    print(f"  kept {kept}/{IMAGES_PER_CELEB}")

    if kept < IMAGES_PER_CELEB:
        print(f"  warning: only {kept} clean headshots found — "
              f"try raising CANDIDATE_POOL or loosening MIN_FACE_AREA_RATIO")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for celeb in CELEBS:
        scrape_celeb(celeb)
    print(f"\nDone. Images saved under ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
