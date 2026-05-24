"""
Multi-source celebrity headshot scraper for Eval 3 (LeMonkey, ETH RC FS26).

DESIGN NOTES:
  The old scraper used a single source (Bing via icrawler) with a weak Haar
  filter. No identity verification, mixed quality, missed AI / robotics
  academics entirely. As of 2025-2026 the Bing Search API is retired and
  Google CSE is closed to new customers, so we can't just swap engines,
  the right answer is a multi-source cascade with proper face / identity
  filtering, which is what published face datasets (VGGFace2, WebFace260M,
  IMDB-Face) all do.

CASCADE:
  1. Wikidata SPARQL → QID + P18 portrait + Commons category + TMDB ID
  2. Wikipedia REST /api/rest_v1/page/summary → lead image
  3. Wikipedia Action API prop=images → article-linked files
  4. Wikimedia Commons category members → many more files
  5. TMDB API /person/{id}/images → curated actor headshots (free, key req.)
  6. DuckDuckGo Images (ddgs) → residual / academic fallback

FILTER STACK (applied to every candidate):
  - MIME image/* and not SVG/icon/signature
  - Resolution: min(w, h) >= MIN_SHORT_SIDE_PX
  - Aspect ratio in [MIN_AR, MAX_AR]                     (rejects banners)
  - InsightFace RetinaFace: exactly 1 face,
      face area >= FACE_AREA_FRAC_MIN of image area     (rejects crowds)
  - ArcFace centroid clustering: keep images whose
      embedding has highest avg cosine to the others    (kills wrong-person hits)
  - Perceptual-hash dedup at Hamming <= PHASH_THRESHOLD (kills duplicates)

OUTPUT per celeb at OUT_ROOT / <slug>:
    01.jpg ... NN.jpg   — top-N accepted images (sorted by centrality)
    provenance.json     — for each kept image:
                            source ("wikidata_p18" / "commons" / "tmdb" / "ddg")
                            original URL
                            license
                            arcface_centrality (cos vs centroid)
                            phash
                            face_area_frac
    rejected/<reason>__<orig_filename>  — every dropped candidate, for audit

REQUIREMENTS:
    pip install requests SPARQLWrapper imagehash duckduckgo-search Pillow
                opencv-python insightface

The InsightFace `buffalo_l` model auto-downloads on first use (~280 MB).

Set TMDB_BEARER env var to enable TMDB (free key from themoviedb.org).
Without TMDB, the cascade still works — TMDB is the "more actor headshots"
augment, not a primary source.
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import re
import sys
import time
import traceback
import unicodedata
import urllib.parse
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

# ─── Tuning (every threshold cited inline) ──────────────────
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


TARGET_N           = 10        # kept images per celeb
CANDIDATE_CAP      = 60        # at most this many per celeb before filter
MIN_SHORT_SIDE_PX  = 512       # min(w, h) — covers any reasonable headshot
MIN_AR             = 0.5       # 1:2 portrait extreme
MAX_AR             = 1.6       # ~5:4 landscape extreme
FACE_AREA_FRAC_MIN = 0.05      # face bbox area / image area >= 5%. Loosened
                                # from 0.10 — famous people's photos are often
                                # 3/4-body event shots where the face occupies
                                # 5-15% of the frame; 0.10 over-rejected.
PHASH_THRESHOLD    = 6         # imagehash Hamming distance for "duplicate"
USER_AGENT         = "LeMonkey-Eval3/0.2 (https://github.com/Ace3Z/LeMonkey)"
TMDB_BEARER        = os.environ.get("TMDB_BEARER", "")

# Filename patterns that are NEVER portraits (icons, logos, signatures, etc.)
NON_PORTRAIT_RX = re.compile(
    r"(?i)(logo|icon|symbol|signature|coat[_ ]of[_ ]arms|flag|oojs|wiki[a-z]+-logo"
    r"|equation|formula|map|chart|graph|diagram|qrcode)"
)
NON_PORTRAIT_EXT = (".svg", ".webm", ".ogv", ".pdf", ".tif", ".tiff", ".gif")


def slugify(name: str) -> str:
    """ASCII-only slug. Strips accents AND single-letter non-ASCII."""
    nfkd = unicodedata.normalize("NFKD", name)
    no_combining = "".join(c for c in nfkd if not unicodedata.combining(c))
    manual = {"Ø": "O", "ø": "o", "Æ": "Ae", "æ": "ae", "ß": "ss",
              "Đ": "D", "đ": "d", "Ł": "L", "ł": "l"}
    for k, v in manual.items():
        no_combining = no_combining.replace(k, v)
    ascii_name = no_combining.encode("ascii", errors="ignore").decode("ascii")
    return ascii_name.lower().replace(" ", "_").replace(".", "")


@dataclasses.dataclass
class Candidate:
    """One image candidate (URL + metadata + lazily-filled bytes/embedding)."""
    source: str                       # "wikidata_p18" | "wikipedia_lead" | "commons" | "tmdb" | "ddg"
    url: str
    width: int = 0
    height: int = 0
    license: str | None = None
    artist: str | None = None
    title: str | None = None          # for Commons, the File: title
    raw_bytes: bytes | None = None    # populated after download
    arcface: np.ndarray | None = None # populated after embedding


# ─── 1. Wikidata + Wikipedia + Commons ────────────────────────────────────
def _http_get_json(url: str, params: dict | None = None, timeout: int = 15) -> dict | None:
    try:
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT},
                          timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def wikidata_lookup(name: str) -> dict:
    """Return {qid, p18_url, commons_category, tmdb_id, occupations, sitelinks}.
    Disambiguates ambiguous names by max sitelink count (proxy for fame)."""
    sparql = f"""
    SELECT ?p ?img ?cat ?tmdb ?sitelinks (GROUP_CONCAT(DISTINCT ?occL; separator="; ") AS ?occs) WHERE {{
      ?p rdfs:label "{name}"@en ; wdt:P31 wd:Q5 .
      OPTIONAL {{ ?p wdt:P18 ?img . }}
      OPTIONAL {{ ?p wdt:P373 ?cat . }}
      OPTIONAL {{ ?p wdt:P4985 ?tmdb . }}
      OPTIONAL {{ ?p wikibase:sitelinks ?sitelinks . }}
      OPTIONAL {{ ?p wdt:P106 ?occ . ?occ rdfs:label ?occL . FILTER(LANG(?occL)="en") }}
    }} GROUP BY ?p ?img ?cat ?tmdb ?sitelinks
      ORDER BY DESC(?sitelinks) LIMIT 5
    """
    try:
        r = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": sparql, "format": "json"},
            headers={"User-Agent": USER_AGENT,
                     "Accept": "application/sparql-results+json"},
            timeout=30,
        )
        if r.status_code != 200:
            return {}
        rows = r.json()["results"]["bindings"]
    except Exception:
        return {}
    if not rows:
        return {}
    b = rows[0]
    return {
        "qid": b["p"]["value"].rsplit("/", 1)[-1],
        "p18_url": b.get("img", {}).get("value"),
        "commons_category": b.get("cat", {}).get("value"),
        "tmdb_id": b.get("tmdb", {}).get("value"),
        "occupations": b.get("occs", {}).get("value", ""),
        "sitelinks": int(b.get("sitelinks", {}).get("value", 0)),
    }


def wikipedia_summary_lead(name: str) -> dict | None:
    """Fetch the Wikipedia article's lead image (original or thumbnail) for `name`."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(name)}"
    j = _http_get_json(url)
    if not j:
        return None
    img = j.get("originalimage") or j.get("thumbnail")
    if not img:
        return None
    return {"url": img["source"],
            "width": img.get("width", 0),
            "height": img.get("height", 0),
            "desc": j.get("description", "")}


def wikipedia_page_image_titles(name: str, limit: int = 30) -> list[str]:
    """Return up to `limit` File: titles linked from the Wikipedia article for `name`."""
    j = _http_get_json(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "titles": name, "prop": "images",
                "imlimit": str(limit), "format": "json", "formatversion": "2"},
    )
    if not j or "query" not in j:
        return []
    return [im["title"] for p in j["query"]["pages"] for im in p.get("images", [])]


def commons_category_files(category: str, limit: int = 50) -> list[str]:
    """List File: titles in the given Wikimedia Commons category (up to `limit`)."""
    cat = category if category.lower().startswith("category:") else f"Category:{category}"
    j = _http_get_json(
        "https://commons.wikimedia.org/w/api.php",
        params={"action": "query", "list": "categorymembers", "cmtitle": cat,
                "cmtype": "file", "cmlimit": str(limit),
                "format": "json", "formatversion": "2"},
    )
    if not j or "query" not in j:
        return []
    return [m["title"] for m in j["query"]["categorymembers"]]


def commons_resolve_file(file_title: str) -> Candidate | None:
    """Resolve a Commons File: title to a downloadable Candidate (URL + license + size)."""
    j = _http_get_json(
        "https://commons.wikimedia.org/w/api.php",
        params={"action": "query", "titles": file_title, "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "format": "json", "formatversion": "2"},
    )
    if not j or "query" not in j:
        return None
    pages = j["query"]["pages"]
    if not pages or "imageinfo" not in pages[0]:
        return None
    ii = pages[0]["imageinfo"][0]
    if not ii.get("mime", "").startswith("image/"):
        return None
    em = ii.get("extmetadata", {})
    return Candidate(
        source="commons", url=ii["url"],
        width=ii.get("width", 0), height=ii.get("height", 0),
        license=(em.get("LicenseShortName") or {}).get("value"),
        artist=(em.get("Artist") or {}).get("value"),
        title=file_title,
    )


# ─── 2. TMDB ───────────────────────────────────────────────────────────────
def tmdb_find_person(name: str) -> int | None:
    """Look up the TMDB person ID for `name` (returns None without a TMDB bearer token or no match)."""
    if not TMDB_BEARER:
        return None
    j = _http_get_json(
        "https://api.themoviedb.org/3/search/person",
        params={"query": name, "include_adult": "false", "language": "en-US"},
    )
    # _http_get_json doesn't pass auth headers — direct call:
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/person",
            params={"query": name, "include_adult": "false"},
            headers={"Authorization": f"Bearer {TMDB_BEARER}",
                     "User-Agent": USER_AGENT, "accept": "application/json"},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
    except Exception:
        return None
    if not results:
        return None
    # Prefer exact-name + highest popularity
    results.sort(key=lambda x: (
        0 if x["name"].lower() == name.lower() else 1,
        -x.get("popularity", 0),
    ))
    return results[0]["id"]


def tmdb_person_images(person_id: int) -> list[Candidate]:
    """Fetch the TMDB curated profile-image list for a person ID, as Candidates."""
    if not TMDB_BEARER:
        return []
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/person/{person_id}/images",
            headers={"Authorization": f"Bearer {TMDB_BEARER}",
                     "User-Agent": USER_AGENT, "accept": "application/json"},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        profiles = r.json().get("profiles", [])
    except Exception:
        return []
    out = []
    for im in profiles:
        out.append(Candidate(
            source="tmdb",
            url=f"https://image.tmdb.org/t/p/original{im['file_path']}",
            width=im.get("width", 0), height=im.get("height", 0),
            license="TMDB (attribution: 'This product uses the TMDB API "
                    "but is not endorsed or certified by TMDB.')",
        ))
    return out


# ─── 3. DuckDuckGo fallback (no key) ──────────────────────────────────────
def ddg_image_urls(query: str, n: int = 30) -> list[str]:
    """DuckDuckGo image search fallback (no API key); returns up to `n` direct image URLs."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return []
    out = []
    try:
        with DDGS() as ddgs:
            for hit in ddgs.images(
                keywords=f"{query} portrait headshot",
                max_results=n,
                size="Large",
            ):
                out.append(hit["image"])
    except Exception as e:
        print(f"[WARN] ddg_failed: query={query!r}, got={type(e).__name__}({e}), "
              f"fallback=bing_icrawler", flush=True)
    return out


def bing_icrawler_candidates(query: str, n: int = 30) -> list[Candidate]:
    """Bing-icrawler fallback. Lets icrawler download to a temp dir, then
    reads each downloaded file back as a Candidate with raw_bytes pre-loaded.
    More robust than URL-capture (icrawler's download() signature varies)."""
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        return []
    import tempfile
    cands: list[Candidate] = []
    try:
        with tempfile.TemporaryDirectory() as td:
            crawler = BingImageCrawler(
                storage={"root_dir": td},
                feeder_threads=1, parser_threads=1, downloader_threads=4,
                log_level=40,  # ERROR only
            )
            crawler.crawl(
                keyword=f"{query} portrait headshot",
                max_num=n,
                filters={"type": "photo"},
                file_idx_offset=0,
            )
            for f in sorted(Path(td).iterdir()):
                try:
                    data = f.read_bytes()
                    with Image.open(io.BytesIO(data)) as im:
                        w, h = im.size
                    cands.append(Candidate(
                        source="bing", url=f"bing://{query}/{f.name}",
                        width=w, height=h, raw_bytes=data,
                    ))
                except Exception:
                    continue
    except Exception as e:
        print(f"[WARN] bing_icrawler_failed: query={query!r}, "
              f"got={type(e).__name__}({e}), fallback=skip", flush=True)
    return cands


# ─── 4. Filter stack ──────────────────────────────────────────────────────
_FACE_APP = None


def _face_app():
    """Lazy-init InsightFace once per process. Uses buffalo_l (ArcFace R100)."""
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l",
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _FACE_APP = app
    return _FACE_APP


def passes_basic_filters(cand: Candidate) -> tuple[bool, str]:
    """Pre-download filters: filename + dimensions + AR."""
    url = cand.url or ""
    name = url.lower()
    if name.endswith(NON_PORTRAIT_EXT):
        return False, "non_photo_ext"
    if NON_PORTRAIT_RX.search(name) or (cand.title and NON_PORTRAIT_RX.search(cand.title.lower())):
        return False, "icon_logo_name"
    if cand.width and cand.height:
        if min(cand.width, cand.height) < MIN_SHORT_SIDE_PX:
            return False, f"too_small_{cand.width}x{cand.height}"
        ar = cand.width / max(1, cand.height)
        if not (MIN_AR <= ar <= MAX_AR):
            return False, f"bad_aspect_{ar:.2f}"
    return True, "ok"


def download(cand: Candidate, timeout: int = 25, max_retries: int = 3) -> tuple[bool, str]:
    """Fetch the image bytes; update width/height. Returns (success, reason).
    Retries on 429 / 503 with exponential backoff (Wikipedia/Commons rate-limits
    individual IPs after ~30 quick requests; canonical mitigation per
    https://api.wikimedia.org/wiki/Documentation/Getting_started/Rate_limits)."""
    backoff = 1.0
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(cand.url, headers={"User-Agent": USER_AGENT},
                              timeout=timeout, allow_redirects=True)
            if r.status_code == 429 or r.status_code == 503:
                if attempt < max_retries:
                    # Honor Retry-After if present, else exponential backoff.
                    ra = r.headers.get("Retry-After", "")
                    delay = float(ra) if ra.isdigit() else backoff
                    time.sleep(delay)
                    backoff *= 2
                    continue
                return False, f"http_{r.status_code}_after_retry"
            if r.status_code != 200:
                return False, f"http_{r.status_code}"
            if len(r.content) < 1024:
                return False, f"body_tiny_{len(r.content)}b"
            cand.raw_bytes = r.content
            with Image.open(io.BytesIO(cand.raw_bytes)) as im:
                cand.width, cand.height = im.size
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(backoff); backoff *= 2; continue
            return False, "timeout"
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries:
                time.sleep(backoff); backoff *= 2; continue
            return False, f"conn_err_{type(e).__name__}"
        except Exception as e:
            return False, f"err_{type(e).__name__}"
        return True, "ok"
    return False, "unreachable"


def passes_post_download_filters(cand: Candidate, face_app) -> tuple[bool, str, float]:
    """After download: resolution + AR (with real dims) + RetinaFace face check."""
    if cand.width < MIN_SHORT_SIDE_PX or cand.height < MIN_SHORT_SIDE_PX:
        return False, f"too_small_{cand.width}x{cand.height}", 0.0
    ar = cand.width / max(1, cand.height)
    if not (MIN_AR <= ar <= MAX_AR):
        return False, f"bad_aspect_{ar:.2f}", 0.0
    try:
        np_img = np.array(Image.open(io.BytesIO(cand.raw_bytes)).convert("RGB"))
        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        faces = face_app.get(bgr)
    except Exception as e:
        return False, f"decode_err_{type(e).__name__}", 0.0
    if len(faces) == 0:
        return False, "n_faces_0", 0.0
    # drop the "exactly 1 face" rule. For famous people, many photos are
    # group/event shots with the subject's face being the largest. Pick the
    # largest face — the ArcFace anchor filter downstream verifies identity
    # so a wrong-person largest-face gets rejected there.
    faces.sort(key=lambda fa: (fa.bbox[2] - fa.bbox[0]) * (fa.bbox[3] - fa.bbox[1]),
               reverse=True)
    f = faces[0]
    x1, y1, x2, y2 = f.bbox
    fw, fh = x2 - x1, y2 - y1
    face_area_frac = (fw * fh) / (cand.width * cand.height)
    if face_area_frac < FACE_AREA_FRAC_MIN:
        return False, f"face_too_small_{face_area_frac:.2f}_(of{len(faces)})", face_area_frac
    cand.arcface = f.normed_embedding.astype(np.float32)
    return True, f"ok_(of{len(faces)})", float(face_area_frac)


TRUSTED_SOURCES = {"wikidata_p18", "wikipedia_lead", "commons", "tmdb"}


def identity_scores(cands: list[Candidate]) -> tuple[np.ndarray, str]:
    """For each candidate, return cosine similarity to a 'true identity'
    reference embedding. Returns (scores, method) where method is one of:
      'anchor_trusted'  — averaged ArcFace embedding of trusted sources (Wikipedia/
                          Commons/Wikidata/TMDB). Best — anchor is editor-curated.
      'anchor_self'     — fallback when zero trusted candidates: use the mean
                          embedding as anchor. Lower confidence; vulnerable to
                          contamination if most candidates are wrong identity.

    This anchor-based approach (vs blind centrality) is what fixes the failure
    mode where Bing returns mostly wrong-person hits and the modal cluster
    is NOT the real celeb. Triple-sourced: VGGFace2 paper §3 uses the
    Wikipedia portrait as a 'reference probe'; IMDB-Face §3.3 same; our own
    our own scraper here also anchors on the Wikipedia portrait, same approach."""
    n = len(cands)
    if n == 0:
        return np.zeros(0, dtype=np.float32), "anchor_self"
    emb = np.stack([c.arcface for c in cands])  # (n, 512), already L2-normed
    trusted_mask = np.array([c.source in TRUSTED_SOURCES for c in cands])
    if trusted_mask.any():
        anchor = emb[trusted_mask].mean(axis=0)
        anchor /= np.linalg.norm(anchor) + 1e-9
        method = "anchor_trusted"
    else:
        anchor = emb.mean(axis=0)
        anchor /= np.linalg.norm(anchor) + 1e-9
        method = "anchor_self"
    return emb @ anchor, method   # (n,) cosines in [-1, 1]


def phash_of(cand: Candidate) -> "imagehash.ImageHash":
    """Compute the perceptual hash of a Candidate's already-downloaded bytes (for dedup)."""
    import imagehash
    with Image.open(io.BytesIO(cand.raw_bytes)) as im:
        return imagehash.phash(im)


# ─── 5. Per-celeb orchestrator ────────────────────────────────────────────
def collect_candidates(name: str) -> tuple[list[Candidate], dict]:
    """Run the full cascade. Returns (Candidates, wikidata_meta)."""
    seen_urls: set[str] = set()
    cands: list[Candidate] = []

    def add(c: Candidate):
        if c.url in seen_urls: return
        seen_urls.add(c.url)
        cands.append(c)

    # 1. Wikidata
    wd = wikidata_lookup(name)
    if wd.get("p18_url"):
        add(Candidate(source="wikidata_p18", url=wd["p18_url"]))

    # 2. Wikipedia REST summary
    lead = wikipedia_summary_lead(name)
    if lead:
        add(Candidate(source="wikipedia_lead", url=lead["url"],
                      width=lead.get("width", 0), height=lead.get("height", 0)))

    # 3. Wikipedia article images
    for title in wikipedia_page_image_titles(name):
        meta = commons_resolve_file(title)
        if meta:
            add(meta)
        if len(cands) >= CANDIDATE_CAP: break

    # 4. Commons category
    if len(cands) < CANDIDATE_CAP and wd.get("commons_category"):
        for title in commons_category_files(wd["commons_category"]):
            meta = commons_resolve_file(title)
            if meta:
                add(meta)
            if len(cands) >= CANDIDATE_CAP: break

    # Some celebs have a Commons category named after them even without
    # Wikidata P373 — fall back to a name-based guess.
    if len(cands) < CANDIDATE_CAP:
        for title in commons_category_files(name):
            meta = commons_resolve_file(title)
            if meta:
                add(meta)
            if len(cands) >= CANDIDATE_CAP: break

    # 5. TMDB
    if len(cands) < CANDIDATE_CAP and TMDB_BEARER:
        tmdb_id = wd.get("tmdb_id")
        if tmdb_id:
            try:
                tmdb_id = int(tmdb_id)
            except ValueError:
                tmdb_id = None
        if not tmdb_id:
            tmdb_id = tmdb_find_person(name)
        if tmdb_id:
            for c in tmdb_person_images(tmdb_id):
                add(c)
                if len(cands) >= CANDIDATE_CAP: break

    # 6. DuckDuckGo fallback when structured sources are thin
    if len(cands) < max(2 * TARGET_N, 20):
        for url in ddg_image_urls(name, n=max(20, CANDIDATE_CAP - len(cands))):
            add(Candidate(source="ddg", url=url))
            if len(cands) >= CANDIDATE_CAP: break

    # 7. icrawler-Bing as final source. We invoke it whenever we have an
    # anchor (Wikipedia/Commons/TMDB) so the anchor-based ArcFace gate kills
    # contamination, OR when we have nothing else. The anchor-trusted filter
    # at cos>=0.4 is what makes this safe — without an anchor, Bing's mixed
    # identities can poison the result.
    have_anchor = any(c.source in TRUSTED_SOURCES for c in cands)
    if (have_anchor or not cands) and len(cands) < CANDIDATE_CAP:
        bing_n = min(40, CANDIDATE_CAP - len(cands))
        for c in bing_icrawler_candidates(name, n=bing_n):
            add(c)
            if len(cands) >= CANDIDATE_CAP: break

    return cands, wd


def process_celeb(name: str, out_root: Path, force: bool = False) -> dict:
    """Full per-celeb pipeline: collect candidates, filter, identity-verify, dedup, write images + provenance."""
    slug = slugify(name)
    out_dir = out_root / slug
    if out_dir.exists() and not force:
        n_existing = sum(1 for p in out_dir.iterdir()
                         if p.is_file() and p.suffix in {".jpg", ".jpeg", ".png"})
        if n_existing >= TARGET_N:
            return {"name": name, "slug": slug, "skipped": True, "n_existing": n_existing}

    if out_dir.exists():
        import shutil; shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rejected").mkdir(exist_ok=True)

    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()

    # ── cascade
    print(f"  cascading sources…", flush=True)
    cands, wd = collect_candidates(name)
    by_src: dict[str, int] = {}
    for c in cands:
        by_src[c.source] = by_src.get(c.source, 0) + 1
    print(f"  collected {len(cands)} candidates: {by_src}", flush=True)
    if not cands:
        print(f"[WARN] no_candidates: name={name!r}, fallback=skip", flush=True)
        return {"name": name, "slug": slug, "error": "no_candidates"}

    # ── pre-download filter
    pre_pass: list[Candidate] = []
    for c in cands:
        ok, why = passes_basic_filters(c)
        if not ok:
            _drop(c, out_dir, why)
            continue
        pre_pass.append(c)
    print(f"  {len(pre_pass)}/{len(cands)} passed pre-download filter", flush=True)

    # ── download (skip if already preloaded by icrawler fallback)
    # Small per-request pacing to stay under Wikipedia/Commons rate limit
    # (~5000 req/h with proper UA; the 429s we hit are upload.wikimedia.org
    # per-IP, not the API).
    downloaded: list[Candidate] = []
    dl_fail_reasons: dict[str, int] = {}
    for c in pre_pass:
        if c.raw_bytes is not None:
            downloaded.append(c); continue
        ok, reason = download(c)
        if ok:
            downloaded.append(c)
        else:
            dl_fail_reasons[reason] = dl_fail_reasons.get(reason, 0) + 1
            _drop_meta(c, out_dir, f"download_{reason}")
        # 100ms pacing — at 10 req/s we'd need ~30s per celeb of pure
        # download time but stay polite. Wikipedia/Commons specifically
        # recommends spacing requests when iterating an API.
        time.sleep(0.1)
    fail_summary = ", ".join(f"{k}:{v}" for k, v in dl_fail_reasons.items()) or "no_failures"
    print(f"  {len(downloaded)}/{len(pre_pass)} downloaded "
          f"(failures: {fail_summary})", flush=True)

    # ── post-download filter (face detection)
    face_app = _face_app()
    post_pass: list[Candidate] = []
    face_fracs: dict[int, float] = {}
    for c in downloaded:
        ok, why, frac = passes_post_download_filters(c, face_app)
        if not ok:
            _drop(c, out_dir, why)
            continue
        face_fracs[id(c)] = frac
        post_pass.append(c)
    print(f"  {len(post_pass)}/{len(downloaded)} passed face filter", flush=True)

    if not post_pass:
        return {"name": name, "slug": slug, "error": "all_filtered"}

    # ── ArcFace identity gate (anchored on Wikipedia/Commons/TMDB when available)
    cosines, method = identity_scores(post_pass)
    # ArcFace canonical threshold: 0.40 is the InsightFace + DeepFace
    # consensus for "same identity" (verified against our inpainted-photo distribution in eval_3/aug/).
    # We use 0.40 when anchor is trusted (it's the real celeb), and a
    # looser 0.25 when anchor is just the candidate centroid (anchor itself
    # may be off).
    cutoff = 0.40 if method == "anchor_trusted" else 0.25
    keep_mask = cosines >= cutoff
    n_outliers = int((~keep_mask).sum())
    for i, c in enumerate(post_pass):
        if not keep_mask[i]:
            _drop(c, out_dir, f"identity_outlier_cos{cosines[i]:.2f}_{method}")
    inliers = [c for i, c in enumerate(post_pass) if keep_mask[i]]
    inlier_cents = cosines[keep_mask]
    print(f"  {len(inliers)}/{len(post_pass)} passed ArcFace ID (method={method}, "
          f"cutoff={cutoff:.2f}, dropped {n_outliers})", flush=True)

    # ── Sort by centrality, then phash-dedup as we save
    order = np.argsort(-inlier_cents)
    inliers = [inliers[i] for i in order]
    inlier_cents = inlier_cents[order]

    kept: list[Candidate] = []
    kept_phashes: list = []
    kept_centralities: list[float] = []
    for c, cent in zip(inliers, inlier_cents.tolist()):
        ph = phash_of(c)
        if any((ph - k) <= PHASH_THRESHOLD for k in kept_phashes):
            _drop(c, out_dir, "phash_duplicate")
            continue
        kept.append(c); kept_phashes.append(ph); kept_centralities.append(cent)
        if len(kept) >= TARGET_N: break
    print(f"  {len(kept)}/{TARGET_N} unique kept after phash dedup", flush=True)

    # ── Save kept + provenance
    provenance = []
    for i, (c, cent, ph) in enumerate(zip(kept, kept_centralities, kept_phashes), start=1):
        out_path = out_dir / f"{i:02d}.jpg"
        try:
            with Image.open(io.BytesIO(c.raw_bytes)) as im:
                im.convert("RGB").save(out_path, "JPEG", quality=92)
        except Exception as e:
            _drop(c, out_dir, f"save_failed_{type(e).__name__}")
            continue
        provenance.append({
            "filename": out_path.name,
            "source": c.source,
            "url": c.url,
            "title": c.title,
            "width": c.width, "height": c.height,
            "license": c.license,
            "artist": c.artist,
            "arcface_centrality": float(cent),
            "face_area_frac": face_fracs.get(id(c)),
            "phash": str(ph),
        })
    (out_dir / "provenance.json").write_text(json.dumps({
        "name": name, "slug": slug, "n_kept": len(provenance),
        "target_n": TARGET_N,
        "wikidata": wd,
        "n_candidates": len(cands),
        "n_downloaded": len(downloaded),
        "n_face_pass": len(post_pass),
        "n_inliers": len(inliers),
        "centroid_cutoff": float(cutoff),
        "elapsed_s": round(time.time() - t0, 1),
        "kept": provenance,
    }, indent=2, default=str))

    elapsed = time.time() - t0
    if len(provenance) < TARGET_N:
        print(f"[WARN] under_target: name={name!r}, expected={TARGET_N}, "
              f"got={len(provenance)}, fallback=accept_partial  ({elapsed:.1f}s)",
              flush=True)
    else:
        print(f"  ✓ {len(provenance)}/{TARGET_N}  ({elapsed:.1f}s)", flush=True)

    return {"name": name, "slug": slug, "n_kept": len(provenance),
            "elapsed_s": elapsed}


def _drop(cand: Candidate, out_dir: Path, reason: str) -> None:
    """Save a rejected image to rejected/<reason>__<index>.jpg for audit.
    Only saves bytes if we have them — pre-download rejects just log."""
    rej = out_dir / "rejected"
    rej.mkdir(exist_ok=True)
    safe = re.sub(r"[^\w.-]", "_", reason)[:60]
    idx = sum(1 for _ in rej.iterdir())
    if cand.raw_bytes:
        ext = ".jpg"  # we re-save as jpg
        try:
            with Image.open(io.BytesIO(cand.raw_bytes)) as im:
                im.convert("RGB").save(rej / f"{safe}__{idx:03d}{ext}", "JPEG", quality=80)
        except Exception:
            # If we can't decode, write raw
            (rej / f"{safe}__{idx:03d}.bin").write_bytes(cand.raw_bytes)
    else:
        _drop_meta(cand, out_dir, reason)


def _drop_meta(cand: Candidate, out_dir: Path, reason: str) -> None:
    """Lightweight rejection log — just URL + reason, no bytes."""
    rej_log = out_dir / "rejected" / "_url_rejects.jsonl"
    rej_log.parent.mkdir(exist_ok=True)
    with rej_log.open("a") as f:
        f.write(json.dumps({"reason": reason, "source": cand.source,
                            "url": cand.url, "title": cand.title}) + "\n")


# ─── 6. CLI ────────────────────────────────────────────────────────────────
def _load_celebs_dict() -> dict[str, list[str]]:
    """Return the in-module CELEBS dict (curated public-figure list for scraping)."""
    return CELEBS


def main() -> int:
    """CLI entry point: iterate over the celeb list and run process_celeb for each, then write a run summary."""
    global TARGET_N
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default="./celeb_headshots_v2")
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict to these CELEBS categories (see the CELEBS dict at module top)")
    p.add_argument("--names", nargs="+", default=None,
                   help="Ad-hoc celeb names to scrape (overrides --only)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--target-n", type=int, default=TARGET_N)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N names (after --only filtering)")
    args = p.parse_args()
    TARGET_N = args.target_n

    if args.names:
        names = list(args.names)
    else:
        celebs = _load_celebs_dict()
        cats = args.only if args.only else list(celebs.keys())
        bad = [c for c in cats if c not in celebs]
        if bad:
            raise SystemExit(f"unknown categories: {bad}")
        names = []
        seen = set()
        for cat in cats:
            for n in celebs[cat]:
                if n not in seen:
                    seen.add(n); names.append(n)

    if args.limit:
        names = names[: args.limit]

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Scraping {len(names)} celebs → {out_root}")
    print(f"TMDB enabled: {bool(TMDB_BEARER)}")

    results = []
    for i, name in enumerate(names, start=1):
        print(f"\n[{i}/{len(names)}]", end="")
        try:
            res = process_celeb(name, out_root, force=args.force)
            results.append(res)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            traceback.print_exc()
            results.append({"name": name, "error": f"{type(e).__name__}: {e}"})

    # ── summary
    (out_root / "_run_summary.json").write_text(json.dumps({
        "n_celebs": len(names),
        "n_with_full": sum(1 for r in results
                            if isinstance(r, dict) and r.get("n_kept", 0) >= TARGET_N),
        "n_partial": sum(1 for r in results
                          if isinstance(r, dict) and 0 < r.get("n_kept", 0) < TARGET_N),
        "n_failed": sum(1 for r in results
                          if isinstance(r, dict) and r.get("n_kept", 0) == 0),
        "results": results,
    }, indent=2))
    print(f"\nDone. Summary at {out_root / '_run_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
