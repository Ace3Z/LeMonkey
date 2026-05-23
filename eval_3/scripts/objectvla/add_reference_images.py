#!/usr/bin/env python3
"""Add the reference-channel image to the existing VL pairs manifest.

For each of the 9842 episodes (178 base + 9664 aug), the
`observation.images.reference` mp4 is a constant-frame video showing the
target celebrity's photo at 480x480. We extract frame 0 once per episode
and add a `reference_image_path` column to the manifest.

Output:
  <out-root>/
      images/...                    (unchanged, wrist-cam frames)
      references/<episode>__ref.jpg (NEW: 1 reference image per episode)
      manifest.parquet              (updated with reference_image_path column)
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def find_reference_video(ep_dir: Path) -> Path | None:
    """Locate the reference mp4."""
    ref_dir = ep_dir / "videos" / "observation.images.reference" / "chunk-000"
    if not ref_dir.is_dir():
        return None
    return next(iter(ref_dir.glob("file-*.mp4")), None)


def extract_reference_frame(args: tuple) -> dict:
    ep_dir, out_root_str = args
    out_root = Path(out_root_str)
    ep_name = ep_dir.name
    stats = {"ep": ep_name, "ok": False, "reason": ""}

    video = find_reference_video(ep_dir)
    if video is None:
        stats["reason"] = "no_reference_video"
        return stats

    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        stats["reason"] = "frame_decode_failed"
        return stats

    out_path = out_root / "references" / f"{ep_name}__ref.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    stats["ok"] = True
    stats["path"] = f"references/{ep_name}__ref.jpg"
    return stats


def discover_episodes(base_root: Path, aug_root: Path,
                       aug_pattern: str = "__var") -> list[Path]:
    base = sorted(p for p in base_root.iterdir()
                    if p.is_dir() and (p / "portrait_corners.json").is_file()
                    and (p / "reference.json").is_file()) if base_root.is_dir() else []
    aug = sorted(p for p in aug_root.iterdir()
                   if p.is_dir() and aug_pattern in p.name
                   and (p / "augmentation.json").is_file()) if aug_root.is_dir() else []
    return base + aug


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-root", type=Path,
                   default=Path.home() / "LeMonkey/datasets/eval3")
    p.add_argument("--aug-root", type=Path,
                   default=Path.home() / "LeMonkey/datasets/eval3_aug_v3_200celebs")
    p.add_argument("--out-root", type=Path,
                   default=Path.home() / "LeMonkey/datasets/eval3_objectvla_vl_pairs")
    p.add_argument("--num-workers", type=int, default=64)
    args = p.parse_args()

    cv2.setNumThreads(1)

    print("[1/3] extracting one reference frame per episode…", flush=True)
    eps = discover_episodes(args.base_root, args.aug_root)
    print(f"      {len(eps)} episodes", flush=True)

    ep_to_ref_path: dict[str, str] = {}
    n_ok = 0
    n_failed = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as exe:
        futures = [exe.submit(extract_reference_frame,
                                (ep, str(args.out_root))) for ep in eps]
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            if r["ok"]:
                ep_to_ref_path[r["ep"]] = r["path"]
                n_ok += 1
            else:
                n_failed += 1
                if n_failed < 5:
                    print(f"      [WARN] {r['ep']}: {r['reason']}", flush=True)
            if done % 2000 == 0:
                elapsed = time.time() - t0
                print(f"      {done}/{len(eps)} ({elapsed:.0f}s)", flush=True)
    print(f"      ok={n_ok} failed={n_failed} in {time.time()-t0:.1f}s", flush=True)

    # ── 2. Update manifest ──────────────────────────────────────────────
    print("[2/3] updating manifest with reference_image_path column…", flush=True)
    manifest_path = args.out_root / "manifest.parquet"
    df = pq.read_table(manifest_path).to_pandas()
    df["reference_image_path"] = df["episode"].map(ep_to_ref_path)
    missing = df["reference_image_path"].isna().sum()
    if missing:
        print(f"      [WARN] {missing} rows have no reference image (will be NaN)",
              flush=True)
    pq.write_table(pa.Table.from_pandas(df), manifest_path, compression="snappy")
    print(f"      rows: {len(df):,}, reference_image_path coverage: "
          f"{(len(df) - missing) * 100 / len(df):.1f}%", flush=True)

    print("[3/3] sanity check — first 3 rows with both image paths:")
    for i in range(3):
        r = df.iloc[i]
        print(f"  [{i}] ep={r.episode}")
        print(f"        image_path:           {r.image_path}")
        print(f"        reference_image_path: {r.reference_image_path}")

    print(f"\n==> done. references in {args.out_root / 'references'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
