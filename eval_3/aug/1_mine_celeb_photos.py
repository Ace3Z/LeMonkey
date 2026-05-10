#!/usr/bin/env python3
"""STAGE 1 — celebrity photo miner with identity verification.

Mines ≥ N photos per celebrity from the web, verifies each via InsightFace
ArcFace cosine similarity to a Wikipedia reference, and saves the survivors
under ~/LeMonkey/datasets/eval3_celebs/web/<celeb>/.

Pipeline per celebrity:
  1. Fetch a canonical Wikipedia primary headshot → ArcFace reference embedding.
  2. icrawler Bing engine: bulk-fetch ~3× the target count.
  3. Per candidate: load → detect face → must have exactly 1 face → ArcFace.
  4. Keep iff cosine >= --threshold (default 0.4).
  5. Save as <id>_<cosine>.jpg (cosine in filename for debug).

Resumable: each celeb's output dir is checked at start; we top up to --num
without redoing finished celebs.

Usage:
    python 1_mine_celeb_photos.py                              # IID three, 30 each
    python 1_mine_celeb_photos.py --celebs swift obama lecun federer merkel
    python 1_mine_celeb_photos.py --num 50 --threshold 0.45
    python 1_mine_celeb_photos.py --dry-run                    # plan + reference only

See STRATEGY.md §3.1 for design rationale.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

# Defer heavyweight imports so --help is fast and so we can fail nicely if
# the env is missing pieces.
try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None  # type: ignore

try:
    from icrawler.builtin import BingImageCrawler
except ImportError:
    BingImageCrawler = None  # type: ignore

# ─── Default celeb roster ────────────────────────────────────────────────────
# Short keys → display names. Wiki URL fragment is the celeb's Wikipedia page
# slug; we fetch a primary headshot from there as the ArcFace reference.
KNOWN_CELEBS: dict[str, dict[str, str]] = {
    # IID three (TOY tier + held-out IID tier)
    "swift":   {"name": "Taylor Swift",       "wiki": "Taylor_Swift"},
    "obama":   {"name": "Barack Obama",       "wiki": "Barack_Obama"},
    "lecun":   {"name": "Yann LeCun",         "wiki": "Yann_LeCun"},
    # OOD candidates likely to appear in TA list (popular public figures)
    "federer": {"name": "Roger Federer",      "wiki": "Roger_Federer"},
    "merkel":  {"name": "Angela Merkel",      "wiki": "Angela_Merkel"},
    "musk":    {"name": "Elon Musk",          "wiki": "Elon_Musk"},
    "messi":   {"name": "Lionel Messi",       "wiki": "Lionel_Messi"},
    "ronaldo": {"name": "Cristiano Ronaldo",  "wiki": "Cristiano_Ronaldo"},
    "beyonce": {"name": "Beyoncé",            "wiki": "Beyoncé"},
    "bezos":   {"name": "Jeff Bezos",         "wiki": "Jeff_Bezos"},
    "trump":   {"name": "Donald Trump",       "wiki": "Donald_Trump"},
    "harris":  {"name": "Kamala Harris",      "wiki": "Kamala_Harris"},
    "lebron":  {"name": "LeBron James",       "wiki": "LeBron_James"},
    "biden":   {"name": "Joe Biden",          "wiki": "Joe_Biden"},
}

UA = "LeMonkey-research/0.1 (mtajdini@student.ethz.ch)"


@dataclass
class CelebSpec:
    key: str
    name: str
    wiki_slug: str

    @property
    def wikimedia_filepath_url(self) -> str:
        return f"https://en.wikipedia.org/wiki/Special:FilePath/{self.wiki_slug}.jpg?width=1200"


# ─── Wikipedia primary headshot ──────────────────────────────────────────────
def fetch_wikipedia_reference(celeb: CelebSpec, *, timeout: float = 20.0) -> Image.Image | None:
    """Fetch a canonical 1200px-wide reference image. Tolerates redirects."""
    try:
        r = requests.get(
            celeb.wikimedia_filepath_url,
            headers={"User-Agent": UA},
            timeout=timeout,
            allow_redirects=True,
        )
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img = ImageOps.exif_transpose(img)
        return img
    except Exception as e:
        print(f"  [WARN] {celeb.key}: wikipedia reference fetch failed: {e}")
        return None


# ─── icrawler Bing miner ─────────────────────────────────────────────────────
def mine_via_bing(name: str, raw_dir: Path, max_n: int) -> int:
    """Bulk-fetch up to max_n images via icrawler Bing. Returns count saved."""
    if BingImageCrawler is None:
        raise RuntimeError("icrawler not installed (pip install icrawler)")
    raw_dir.mkdir(parents=True, exist_ok=True)
    crawler = BingImageCrawler(
        feeder_threads=1, parser_threads=1, downloader_threads=4,
        storage={"root_dir": str(raw_dir)},
    )
    # filter: large + photo-only is more robust than face-photo (which can miss)
    crawler.crawl(keyword=f"{name} portrait", max_num=max_n, file_idx_offset=0)
    return len(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.jpeg")) + list(raw_dir.glob("*.png")))


# ─── Identity verification ───────────────────────────────────────────────────
class FaceVerifier:
    """Wraps InsightFace buffalo_l for face detection + ArcFace embedding."""

    def __init__(self, det_size: int = 640):
        if FaceAnalysis is None:
            raise RuntimeError("insightface not installed (pip install insightface onnxruntime-gpu)")
        self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def detect(self, img_pil: Image.Image) -> list:
        # InsightFace expects BGR ndarray
        bgr = np.array(img_pil)[:, :, ::-1].copy()
        return self.app.get(bgr)

    def embed_single_face(self, img_pil: Image.Image) -> np.ndarray | None:
        faces = self.detect(img_pil)
        if len(faces) != 1:
            return None
        return faces[0].normed_embedding  # already L2-normalized

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b)  # both L2-normed


# ─── Per-celeb pipeline ──────────────────────────────────────────────────────
def process_celeb(
    celeb: CelebSpec,
    out_root: Path,
    *,
    target_n: int,
    threshold: float,
    verifier: FaceVerifier,
    dry_run: bool,
) -> dict:
    """Mine + verify until we have ≥ target_n photos. Returns a stats dict."""
    out_dir = out_root / celeb.key
    raw_dir = out_root / "_raw" / celeb.key
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.jpg")) + sorted(out_dir.glob("*.png"))
    if len(existing) >= target_n:
        print(f"  [{celeb.key}] already have {len(existing)} ≥ {target_n}; skipping.")
        return {"celeb": celeb.key, "kept": len(existing), "skipped": True}

    print(f"\n=== {celeb.key} ({celeb.name}) ===")

    # 1. Wikipedia reference
    print(f"  [1/4] fetching Wikipedia reference...")
    ref_img = fetch_wikipedia_reference(celeb)
    if ref_img is None:
        return {"celeb": celeb.key, "kept": 0, "error": "wiki ref fetch failed"}
    if dry_run:
        ref_img.save(out_dir / "__wiki_reference.jpg", quality=92)

    print(f"  [2/4] computing ArcFace embedding for reference...")
    ref_emb = verifier.embed_single_face(ref_img)
    if ref_emb is None:
        return {"celeb": celeb.key, "kept": 0, "error": "ref has 0 or >1 faces — pick a different wiki slug"}

    if dry_run:
        return {"celeb": celeb.key, "kept": len(existing), "dry_run": True}

    # 2. Bulk mine
    needed = target_n - len(existing)
    over_request = max(needed * 3, 30)
    print(f"  [3/4] mining ~{over_request} candidates via Bing (need {needed} verified)...")
    n_raw = mine_via_bing(celeb.name, raw_dir, max_n=over_request)
    print(f"        downloaded {n_raw} raw candidates")

    # 3. Verify each
    print(f"  [4/4] verifying each via ArcFace (threshold cosine ≥ {threshold:.2f})...")
    raw_files = sorted(p for p in raw_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    kept = len(existing)
    rejected_no_face = 0
    rejected_multi_face = 0
    rejected_low_cos = 0
    next_idx = (max((int(p.stem.split("_")[1]) for p in existing if "_" in p.stem
                     and p.stem.split("_")[1].isdigit()), default=-1)) + 1

    for raw in raw_files:
        if kept >= target_n:
            break
        try:
            img = Image.open(raw).convert("RGB")
            img = ImageOps.exif_transpose(img)
        except Exception:
            continue
        # downsize huge images for speed
        if max(img.size) > 1600:
            img.thumbnail((1600, 1600), Image.LANCZOS)
        faces = verifier.detect(img)
        if len(faces) == 0:
            rejected_no_face += 1
            continue
        if len(faces) > 1:
            # accept if there's a clearly dominant face (largest bbox > 2× the next)
            faces_sorted = sorted(faces, key=lambda f: -((f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])))
            a0 = (faces_sorted[0].bbox[2]-faces_sorted[0].bbox[0]) * (faces_sorted[0].bbox[3]-faces_sorted[0].bbox[1])
            a1 = (faces_sorted[1].bbox[2]-faces_sorted[1].bbox[0]) * (faces_sorted[1].bbox[3]-faces_sorted[1].bbox[1])
            if a0 < 2 * a1:
                rejected_multi_face += 1
                continue
            face = faces_sorted[0]
        else:
            face = faces[0]
        cos = float(face.normed_embedding @ ref_emb)
        if cos < threshold:
            rejected_low_cos += 1
            continue
        # save
        out_name = f"{celeb.key}_{next_idx:03d}_cos{int(cos*1000):03d}.jpg"
        img.save(out_dir / out_name, quality=92)
        kept += 1
        next_idx += 1

    stats = {
        "celeb": celeb.key,
        "kept": kept,
        "target": target_n,
        "raw_downloaded": n_raw,
        "rejected_no_face": rejected_no_face,
        "rejected_multi_face": rejected_multi_face,
        "rejected_low_cos": rejected_low_cos,
    }
    print(f"  → kept {kept}/{target_n}, "
          f"rejected: no-face={rejected_no_face} multi-face={rejected_multi_face} low-cos={rejected_low_cos}")
    return stats


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--celebs", nargs="+", default=["swift", "obama", "lecun"],
                   help="short keys from KNOWN_CELEBS or 'all'")
    p.add_argument("--num", type=int, default=30,
                   help="target verified photos per celeb (default 30)")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="ArcFace cosine threshold for accept (default 0.40)")
    p.add_argument("--out-root", default="/home/lemonkey/LeMonkey/datasets/eval3_celebs/web",
                   help="output root for verified photos")
    p.add_argument("--keep-raw", action="store_true",
                   help="don't delete the _raw bulk-download dir after verification")
    p.add_argument("--dry-run", action="store_true",
                   help="fetch wiki references only, skip bulk-mining")
    args = p.parse_args()

    if args.celebs == ["all"]:
        args.celebs = list(KNOWN_CELEBS.keys())

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    specs: list[CelebSpec] = []
    for k in args.celebs:
        if k not in KNOWN_CELEBS:
            print(f"[ERROR] unknown celeb key '{k}'. choices: {list(KNOWN_CELEBS)}", file=sys.stderr)
            return 1
        specs.append(CelebSpec(key=k, name=KNOWN_CELEBS[k]["name"], wiki_slug=KNOWN_CELEBS[k]["wiki"]))

    print(f"target: {args.num}/celeb × {len(specs)} celebs = {args.num * len(specs)} verified photos total")
    print(f"out:    {out_root}")
    print(f"thresh: cosine ≥ {args.threshold}")

    verifier: FaceVerifier | None = None
    if not args.dry_run:
        print("\nloading InsightFace buffalo_l...")
        t0 = time.time()
        verifier = FaceVerifier()
        print(f"  loaded in {time.time()-t0:.1f}s")

    summary: list[dict] = []
    for spec in specs:
        s = process_celeb(
            spec, out_root,
            target_n=args.num,
            threshold=args.threshold,
            verifier=verifier,
            dry_run=args.dry_run,
        )
        summary.append(s)

    # Cleanup raw bulk-download cache
    if not args.keep_raw and not args.dry_run:
        raw_root = out_root / "_raw"
        if raw_root.is_dir():
            shutil.rmtree(raw_root)

    # Persist stats
    if not args.dry_run:
        (out_root / "_stats.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 60)
    print(" summary")
    print("=" * 60)
    for s in summary:
        print(f"  {s.get('celeb','?'):10s}  kept={s.get('kept','?')}/"
              f"{s.get('target', args.num)}    "
              f"({'skipped' if s.get('skipped') else 'mined'})")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
