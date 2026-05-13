"""
Scrape clean headshot images per celebrity from Bing Images.

Why: Eval 3 (LeMonkey, ETH RC FS26 Project 1) will be tested on photos of
"top 50 famous people" per the TAs. We need a broad OOD photo bank so the
SmolVLA policy + inpainting augmentation pipeline cover whichever celebs
get printed at demo time.

Pipeline:
  1. Download a candidate pool from Bing (more than we need).
  2. Run OpenCV face detection on each image.
  3. Keep only images with exactly ONE face that fills >=20% of the frame.
  4. Save the top N per celebrity, named <celeb>_1.jpg, <celeb>_2.jpg, ...

The CELEBS dict-of-lists below was assembled from four cross-checked sources
(see commit message for triple-source bibliography):
  - Forbes Celebrity 100 (Wikipedia) + Time 100 + Instagram most-followed
  - IMDB "100 MOST POPULAR CELEBRITIES IN THE WORLD" (list ls052283250)
  - Kaggle "Celebrity Face Image Dataset" (vishesh1412, 17 actors)
  - Time 100 AI 2025 + Fortune Most Powerful in Business 2025

Install:
    pip install icrawler opencv-python

Run:
    python head_shot.py                       # scrape all categories
    python head_shot.py --only tech actors    # scrape selected categories
"""

import argparse
import os
import shutil
import unicodedata

import cv2
from icrawler.builtin import BingImageCrawler

# Master list organized by category so we can scrape a subset on demand.
# Within each category, names are roughly ordered by fame / relevance.
CELEBS: dict[str, list[str]] = {
    # ── Tech / AI / Robotics (50) ───────────────────────────────────────
    # Big Tech CEOs / Founders, AI lab leaders, AI academics, robotics
    # pioneers, HuggingFace / open-source. Triple-sourced from Time 100 AI
    # 2025, Fortune Most Powerful in Business 2025, ACM Turing, Nobel 2024.
    "tech_ceos": [
        "Elon Musk",
        "Bill Gates",
        "Steve Jobs",
        "Mark Zuckerberg",
        "Jeff Bezos",
        "Tim Cook",
        "Sundar Pichai",
        "Satya Nadella",
        "Jensen Huang",
        "Andy Jassy",
        "Lisa Su",
        "Masayoshi Son",
    ],
    "ai_labs": [
        "Sam Altman",
        "Demis Hassabis",
        "Dario Amodei",
        "Daniela Amodei",
        "Ilya Sutskever",
        "Greg Brockman",
        "Mira Murati",
        "Mustafa Suleyman",
        "Liang Wenfeng",
        "Aravind Srinivas",
        "John Jumper",
        "Chris Olah",
        "Jared Kaplan",
        "Shane Legg",
    ],
    "ai_academia": [
        "Andrej Karpathy",
        "Yann LeCun",
        "Geoffrey Hinton",
        "Yoshua Bengio",
        "Andrew Ng",
        "Fei-Fei Li",
        "Stuart Russell",
        "Richard Sutton",
        "Andrew Barto",
    ],
    "robotics": [
        "Marc Raibert",
        "Robert Playter",
        "Brett Adcock",
        "Bernt Børnich",
        "Marco Hutter",
        "Marc Pollefeys",
        "Roland Siegwart",
        "Davide Scaramuzza",
        "Sergey Levine",
        "Pieter Abbeel",
        "Chelsea Finn",
        "Karol Hausman",
        "Oier Mees",
    ],
    "open_source": [
        "Clément Delangue",
        "Thomas Wolf",
        "Rémi Cadène",
    ],

    # ── Top 50 globally famous in 2025/2026 ─────────────────────────────
    # Sources: Instagram most-followed, Time 100 of 2026, Forbes Celebrity
    # 100 (Wikipedia top-10s 2005-2020). Categorised below.
    "musicians": [
        "Taylor Swift",
        "Beyoncé",
        "Rihanna",
        "Selena Gomez",
        "Ariana Grande",
        "Drake",
        "The Weeknd",
        "Bad Bunny",
        "Dua Lipa",
        "Billie Eilish",
        "Sabrina Carpenter",
        "Charli XCX",
        "Chappell Roan",
        "Olivia Rodrigo",
        "Justin Bieber",
        "Lady Gaga",
        "Katy Perry",
        "Miley Cyrus",
        "Jennifer Lopez",
        "Demi Lovato",
        "Cardi B",
        "Chris Brown",
        "Shakira",
        "Snoop Dogg",
        "Lisa BLACKPINK",
        "Ed Sheeran",
        "Kanye West",
    ],
    "actors_modern": [
        # Younger / gen-Z / current-Hollywood (Time 100 2026, social-media
        # top followed). Includes everyone in the merged top-50 famous list.
        "Zendaya",
        "Jenna Ortega",
        "Timothée Chalamet",
        "Sydney Sweeney",
        "Pedro Pascal",
        "Jeremy Allen White",
        "Jacob Elordi",
        "Cillian Murphy",
        "Anya Taylor-Joy",
        "Tom Holland",
        "Margot Robbie",
        "Emma Stone",
        "Daisy Ridley",
        "Chris Pratt",
        "Jennifer Lawrence",
    ],
    "actors_classic": [
        # IMDB ls052283250 "100 MOST POPULAR CELEBRITIES IN THE WORLD"
        # + Kaggle Celebrity Faces Dataset (17 names) — all merged and
        # deduplicated against actors_modern above.
        "Johnny Depp",
        "Leonardo DiCaprio",
        "Tom Cruise",
        "Robert Downey Jr",
        "Brad Pitt",
        "Tom Hanks",
        "Hugh Jackman",
        "Matt Damon",
        "Will Smith",
        "Morgan Freeman",
        "Angelina Jolie",
        "Scarlett Johansson",
        "Anne Hathaway",
        "Natalie Portman",
        "Ryan Reynolds",
        "Keanu Reeves",
        "Denzel Washington",
        "Chris Hemsworth",
        "Chris Evans",
        "Robert De Niro",
        "Al Pacino",
        "Meryl Streep",
        "George Clooney",
        "Harrison Ford",
        "Arnold Schwarzenegger",
        "Jim Carrey",
        "Emma Watson",
        "Daniel Radcliffe",
        "Russell Crowe",
        "Liam Neeson",
        "Kate Winslet",
        "Sean Connery",
        "Mark Wahlberg",
        "Pierce Brosnan",
        "Orlando Bloom",
        "Dwayne Johnson",
        "Jackie Chan",
        "Adam Sandler",
        "Heath Ledger",
        "Daniel Craig",
        "Jessica Alba",
        "Edward Norton",
        "Keira Knightley",
        "Bradley Cooper",
        "Will Ferrell",
        "Julia Roberts",
        "Nicolas Cage",
        "Ian McKellen",
        "Halle Berry",
        "Bruce Willis",
        "Samuel L. Jackson",
        "Ben Stiller",
        "Tommy Lee Jones",
        "Jack Black",
        "Antonio Banderas",
        "Steve Carell",
        "Shia LaBeouf",
        "Megan Fox",
        "James Franco",
        "Mel Gibson",
        "Vin Diesel",
        "Tim Allen",
        "Robin Williams",
        "Jean-Claude Van Damme",
        "Owen Wilson",
        "Christian Bale",
        "Sandra Bullock",
        "Bruce Lee",
        "Drew Barrymore",
        "Jack Nicholson",
        "Bill Murray",
        "Sigourney Weaver",
        "Jake Gyllenhaal",
        "Jason Statham",
        "Jet Li",
        "Kate Beckinsale",
        "Rowan Atkinson",
        "Marlon Brando",
        "John Travolta",
        "Ben Affleck",
        "Jennifer Aniston",
        "James McAvoy",
        "Brendan Fraser",
        "Rachel McAdams",
        "Tom Hiddleston",
        "Cameron Diaz",
        "Sylvester Stallone",
        "Clint Eastwood",
        "Nicole Kidman",
    ],
    "directors": [
        "Steven Spielberg",
        "Christopher Nolan",
        "Peter Jackson",
        "James Cameron",
    ],
    "athletes": [
        "Cristiano Ronaldo",
        "Lionel Messi",
        "LeBron James",
        "Virat Kohli",
        "Roger Federer",
        "Tiger Woods",
    ],
    "creators_and_media": [
        "MrBeast",
        "Oprah Winfrey",
        "Kim Kardashian",
        "Kylie Jenner",
        "Kendall Jenner",
        "Khloé Kardashian",
    ],
    "politics": [
        "Donald Trump",
        "Barack Obama",
    ],
    # Swiss locals — kept from prior scraper. Useful because Eval 3 is at
    # ETH Zurich and there's a non-zero chance Swiss public figures appear.
    "swiss": [
        "Stan Wawrinka",
        "Bertrand Piccard",
        "Ursula Andress",
        "Granit Xhaka",
        "Xherdan Shaqiri",
    ],
}

IMAGES_PER_CELEB = 10
CANDIDATE_POOL = 40           # download this many, then filter down
MIN_FACE_AREA_RATIO = 0.20    # face bbox must cover >=20% of the image
OUTPUT_DIR = "celeb_headshots"

# Haar cascade ships with opencv-python — no extra download needed.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def slugify(name: str) -> str:
    """ASCII-only slug. Strips accents (Beyoncé → beyonce) AND replaces
    single-letter non-ASCII chars (Bernt Børnich → bernt_bornich) so paths
    are portable and shell-safe across Linux / macOS / Windows."""
    nfkd = unicodedata.normalize("NFKD", name)
    no_combining = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Handle single-letter non-ASCII (Ø, Æ, ß, etc.) that NFKD doesn't decompose.
    manual = {"Ø": "O", "ø": "o", "Æ": "Ae", "æ": "ae", "ß": "ss",
              "Đ": "D", "đ": "d", "Ł": "L", "ł": "l"}
    for k, v in manual.items():
        no_combining = no_combining.replace(k, v)
    ascii_name = no_combining.encode("ascii", errors="ignore").decode("ascii")
    return ascii_name.lower().replace(" ", "_").replace(".", "")


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


def scrape_celeb(name: str, force: bool = False) -> None:
    slug = slugify(name)
    folder = os.path.join(OUTPUT_DIR, slug)

    if os.path.exists(folder) and not force:
        n_existing = sum(
            1 for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        )
        if n_existing >= IMAGES_PER_CELEB:
            print(f"\n=== {name} ===")
            print(f"  already has {n_existing}/{IMAGES_PER_CELEB} headshots — skipping (pass --force to re-scrape)")
            return

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
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", default=None,
                   help=f"Restrict to these categories. Available: "
                        f"{', '.join(CELEBS.keys())}")
    p.add_argument("--force", action="store_true",
                   help="Re-scrape even if a celeb folder already has enough images")
    args = p.parse_args()

    cats = args.only if args.only else list(CELEBS.keys())
    bad = [c for c in cats if c not in CELEBS]
    if bad:
        raise SystemExit(f"unknown categories: {bad} (available: {list(CELEBS.keys())})")

    names = [n for cat in cats for n in CELEBS[cat]]
    # dedupe while preserving first-seen order across categories
    seen = set()
    names_dedup = []
    for n in names:
        if n not in seen:
            seen.add(n)
            names_dedup.append(n)

    print(f"Total celebs to scrape: {len(names_dedup)} "
          f"({len(names) - len(names_dedup)} duplicates removed)")
    print(f"Categories: {cats}")
    print(f"Target: {IMAGES_PER_CELEB} headshots × "
          f"~{CANDIDATE_POOL} candidates each = ~{CANDIDATE_POOL * len(names_dedup)} downloads")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for celeb in names_dedup:
        scrape_celeb(celeb, force=args.force)
    print(f"\nDone. Images saved under ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
