#!/usr/bin/env python3
"""Prepare a PaliGemma VQA training manifest from VGGFace2 (+ optionally our
own 193-celeb scraped bank for distribution alignment).

OUTPUT FORMAT
=============

Writes a parquet file with columns:
    image_path:  str    absolute path to a face image on disk
    prompt:      str    "<image>Who is the person in this image?\n"
    target:      str    canonical celeb name (e.g. "Yann LeCun")
    identity_id: str    source-stable identity hash (for splits)
    source:      str    "vggface2" | "scraped_eval3"

We don't pre-encode the images — PaliGemmaProcessor + a HF datasets loader does
that at training time. This keeps the manifest small (a few hundred MB) and lets
us re-sample/re-split without reprocessing pixels.

INPUTS
======

VGGFace2 expects the standard Oxford layout:
    <vggface2_root>/<identity_id>/<image_name>.jpg
    <vggface2_root>/identity_meta.csv     (id -> name mapping; optional but helps)

If you don't have identity_meta.csv, pass --names-jsonl with one
{"id": "n000001", "name": "Aaron Eckhart"} per line.

Scraped bank uses our 193-celeb dir layout:
    <scraped_root>/<celeb_slug>/<photo>.{jpg,png}
where the slug is the canonical name (lowercase, underscored).

EXAMPLE
=======

    python eval_3/scripts/warmstart/prepare_vggface2_vqa.py \\
        --vggface2-root /shared/datasets/vggface2/train \\
        --names-csv     /shared/datasets/vggface2/identity_meta.csv \\
        --scraped-root  ~/LeMonkey/datasets/eval3_celebs/scraped \\
        --max-per-identity 50 \\
        --out manifests/vggface2_vqa_train.parquet

Result: ~9131 identities × 50 = ~456k VGGFace2 rows + 193 × ~8 = ~1500 scraped
rows. Hub-style upload: `python ... --push-repo HBOrtiz/vggface2_vqa_paligemma`.
"""
from __future__ import annotations
import argparse, csv, json, random, sys
from pathlib import Path
from collections import defaultdict


def load_vggface2_names(csv_path: Path | None,
                         names_jsonl: Path | None) -> dict[str, str]:
    """Returns {identity_id: clean_name}.

    Note: the official Oxford VGGFace2 identity_meta.csv has the well-known
    bug that every column header EXCEPT Class_ID has a leading space:
        `Class_ID, Name, Sample_Num, Flag, Gender`
                  ^^^^^   leading space ^
    We strip whitespace from headers and values to survive this.
    """
    id_to_name: dict[str, str] = {}
    if csv_path is not None and csv_path.is_file():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
                cls_id = row.get("Class_ID") or row.get("class_id") or row.get("id")
                name = row.get("Name") or row.get("name")
                if cls_id and name:
                    id_to_name[cls_id] = name.replace("_", " ").strip()
        if not id_to_name:
            raise SystemExit(
                f"[FATAL] no identities parsed from {csv_path}. "
                "Header schema may have shifted — open the CSV and verify "
                "the Class_ID + Name columns are present."
            )
    if names_jsonl is not None and names_jsonl.is_file():
        with open(names_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                id_to_name[rec["id"]] = rec["name"]
    return id_to_name


def collect_vggface2(root: Path, id_to_name: dict[str, str],
                      max_per_identity: int, rng: random.Random,
                      ) -> list[dict]:
    rows: list[dict] = []
    n_no_name = 0
    n_identities = 0
    for ident_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ident_id = ident_dir.name
        name = id_to_name.get(ident_id)
        if not name:
            n_no_name += 1
            # Fall back to slug-as-name — but if EVERY id lacks a name
            # mapping, we hard-fail below.
            name = ident_id.replace("_", " ")
        photos = sorted(p for p in ident_dir.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not photos:
            continue
        rng.shuffle(photos)
        photos = photos[:max_per_identity]
        for ph in photos:
            rows.append({
                "image_path": str(ph),
                "prompt": "<image>Who is the person in this image?\n",
                "target": name,
                "identity_id": f"vggface2:{ident_id}",
                "source": "vggface2",
            })
        n_identities += 1
    if n_no_name:
        print(f"[WARN] vggface2_no_meta_match: expected=id_to_name lookup, "
              f"got={n_no_name}/{n_identities} unknown ids, "
              f"fallback=slug-as-name", flush=True)
    if n_no_name == n_identities and n_identities > 10:
        raise SystemExit(
            f"[FATAL] all {n_identities} vggface2 identities lack a name mapping. "
            "The CSV/JSONL is empty or schema-mismatched. Aborting — slug-as-name "
            "would train the model to predict 'n000001' instead of 'Aaron Eckhart'."
        )
    print(f"vggface2: {n_identities} identities, {len(rows)} rows", flush=True)
    return rows


def collect_scraped(root: Path, max_per_identity: int,
                     rng: random.Random) -> list[dict]:
    """Our 193-celeb scraped bank. Each subdir = one celeb, slug-named."""
    rows: list[dict] = []
    n_identities = 0
    for ident_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        slug = ident_dir.name
        # taylor_swift -> Taylor Swift
        name = " ".join(w.capitalize() for w in slug.replace("-", " ").split("_"))
        photos = sorted(p for p in ident_dir.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not photos:
            continue
        rng.shuffle(photos)
        photos = photos[:max_per_identity]
        for ph in photos:
            rows.append({
                "image_path": str(ph),
                "prompt": "<image>Who is the person in this image?\n",
                "target": name,
                "identity_id": f"scraped:{slug}",
                "source": "scraped_eval3",
            })
        n_identities += 1
    print(f"scraped: {n_identities} identities, {len(rows)} rows", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vggface2-root", type=Path,
                     help="VGGFace2 dir containing <id>/<image>.jpg (e.g. .../train/)")
    ap.add_argument("--names-csv", type=Path, default=None,
                     help="VGGFace2 identity_meta.csv (id -> name)")
    ap.add_argument("--names-jsonl", type=Path, default=None,
                     help="Alternate names file: one {id, name} per line")
    ap.add_argument("--scraped-root", type=Path, default=None,
                     help="Our 193-celeb scraped bank (optional)")
    ap.add_argument("--max-per-identity", type=int, default=50,
                     help="Cap per identity to keep the manifest manageable (default 50)")
    ap.add_argument("--scraped-max-per-identity", type=int, default=10,
                     help="Override max-per-identity for the scraped bank (default 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True,
                     help="Output parquet path (.parquet)")
    ap.add_argument("--shuffle", action="store_true", default=True,
                     help="Shuffle row order before write (default True)")
    args = ap.parse_args()

    if args.vggface2_root is None and args.scraped_root is None:
        ap.error("at least one of --vggface2-root / --scraped-root required")

    rng = random.Random(args.seed)
    rows: list[dict] = []

    if args.vggface2_root is not None:
        id_to_name = load_vggface2_names(args.names_csv, args.names_jsonl)
        rows.extend(collect_vggface2(args.vggface2_root, id_to_name,
                                       args.max_per_identity, rng))

    if args.scraped_root is not None:
        rows.extend(collect_scraped(args.scraped_root,
                                      args.scraped_max_per_identity, rng))

    if args.shuffle:
        rng.shuffle(rows)

    print(f"\ntotal rows: {len(rows)}", flush=True)
    by_source = defaultdict(int)
    by_name = defaultdict(int)
    for r in rows:
        by_source[r["source"]] += 1
        by_name[r["target"]] += 1
    for src, n in sorted(by_source.items()):
        print(f"  {src}: {n}", flush=True)
    print(f"unique target names: {len(by_name)}", flush=True)
    print(f"min/median/max examples per name: "
          f"{min(by_name.values())} / "
          f"{sorted(by_name.values())[len(by_name)//2]} / "
          f"{max(by_name.values())}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Use pyarrow to write parquet (no torch/datasets dep needed here)
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        k: [r[k] for r in rows]
        for k in ("image_path", "prompt", "target", "identity_id", "source")
    })
    pq.write_table(table, args.out, compression="snappy")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
