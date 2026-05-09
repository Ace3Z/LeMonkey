#!/usr/bin/env python
"""
Build a clean per-identity manifest from the hearfool VGGFace2 mirror on the
external SSD, joined with the upstream identity metadata CSV.

Inputs (defaults — overridable via CLI):
    /Volumes/externalSSD/datasets/vggface2_hearfool/{train,val}/n00xxxx/*.jpg
    /Volumes/externalSSD/datasets/vggface2_hearfool/meta/identity_meta_with_estimated_age.csv

Output:
    /Volumes/externalSSD/datasets/vggface2_hearfool/manifests/manifest.parquet

Manifest columns:
    class_id      str   "n000002"
    name          str   "A_Fine_Frenzy"
    n_images      int   315
    gender        str   "f"
    source_split  str   "train" | "val"
    image_paths   list[str]   absolute paths

Filtering:
    - drop identities with fewer than --min-images images
    - drop identities not present in identity_meta CSV (defensive — shouldn't happen)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

DEFAULT_DATA_ROOT = Path("/Volumes/externalSSD/datasets/vggface2_hearfool")
DEFAULT_META_CSV = DEFAULT_DATA_ROOT / "meta/identity_meta_with_estimated_age.csv"
DEFAULT_OUT = DEFAULT_DATA_ROOT / "manifests/manifest.parquet"


def load_metadata(csv_path: Path) -> dict[str, dict]:
    """Parse identity_meta CSV → {class_id: {name, gender, sample_num, ...}}.

    File format (no header), one row per identity:
        n000001, "14th_Dalai_Lama", 424, 0, m, 61
    """
    out: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if len(row) < 5:
                continue
            class_id, name, sample_num, flag, gender = row[:5]
            out[class_id.strip()] = {
                "name": name.strip().strip('"'),
                "gender": gender.strip(),
                "sample_num_meta": int(sample_num.strip()),
                "split_flag": int(flag.strip()),
            }
    return out


def scan_identities(split_root: Path) -> dict[str, list[str]]:
    """Return {class_id: [absolute_image_path, ...]} for all identity dirs in split_root."""
    out: dict[str, list[str]] = {}
    if not split_root.is_dir():
        return out
    for id_dir in sorted(split_root.iterdir()):
        if not id_dir.is_dir() or not id_dir.name.startswith("n"):
            continue
        imgs = sorted(str(p) for p in id_dir.glob("*.jpg"))
        if imgs:
            out[id_dir.name] = imgs
    return out


def build_manifest(data_root: Path, meta_csv: Path, min_images: int) -> pd.DataFrame:
    print(f"[build] data root: {data_root}")
    print(f"[build] meta csv:  {meta_csv}")
    print(f"[build] min imgs/id: {min_images}")

    meta = load_metadata(meta_csv)
    print(f"[build] metadata loaded: {len(meta)} identities")

    rows = []
    skipped_no_meta = 0
    skipped_low_count = 0
    for split in ("train", "val"):
        split_root = data_root / split
        ids = scan_identities(split_root)
        print(f"[build] {split}/ scanned: {len(ids)} identities, "
              f"{sum(len(v) for v in ids.values())} images")
        for class_id, paths in ids.items():
            if class_id not in meta:
                skipped_no_meta += 1
                continue
            if len(paths) < min_images:
                skipped_low_count += 1
                continue
            rows.append({
                "class_id": class_id,
                "name": meta[class_id]["name"],
                "n_images": len(paths),
                "gender": meta[class_id]["gender"],
                "source_split": split,
                "image_paths": paths,
            })

    df = pd.DataFrame(rows)
    print(f"[build] kept: {len(df)} identities, {df['n_images'].sum()} images")
    print(f"[build] skipped (no metadata): {skipped_no_meta}")
    print(f"[build] skipped (low image count <{min_images}): {skipped_low_count}")

    if not df.empty:
        print("\n[build] per-split breakdown:")
        print(df.groupby("source_split").agg(
            n_identities=("class_id", "count"),
            total_images=("n_images", "sum"),
            mean_imgs=("n_images", "mean"),
        ).round(1).to_string())
        print("\n[build] sample rows:")
        print(df[["class_id", "name", "n_images", "gender", "source_split"]].head().to_string(index=False))

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--meta-csv", type=Path, default=DEFAULT_META_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-images", type=int, default=30,
                        help="Drop identities with fewer than this many images")
    args = parser.parse_args()

    df = build_manifest(args.data_root, args.meta_csv, args.min_images)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\n[build] wrote {args.out}  ({args.out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
