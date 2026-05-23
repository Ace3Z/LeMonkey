#!/usr/bin/env python3
"""Build the ObjectVLA-style VL co-training manifest from our augmented dataset.

Uses the CACHED paper-bboxes from each base/aug episode's
`portrait_corners.json` (computed during aug generation) — no new inference,
just frame decode + math.

Per the 2026-05-20 cross-checked research (TRACK_OBJECTVLA.md, ObjectVLA paper
arxiv 2502.19250 §3, PaliGemma 2 arxiv 2412.03555, Kosmos-2 arxiv 2306.14824,
VGGFace2 arxiv 1710.08092):
  - whole-object (paper) bbox is the right granularity for our task
  - loose context around the identity-bearing region (face) HELPS identity
    discrimination (Banerjee 2022, LFW context-only ablation)
  - ObjectVLA's published recipe uses object-level boxes from DinoX, not
    sub-feature face boxes.

Per-frame, for each of the 3 visible portraits, we emit two VL pairs:
  1. Location-explicit (50%):
       prompt = "What is in this image?"
       target = "The printed photo of <Name> is at [x1, y1, x2, y2]."
  2. Q&A grounded by location (50%):
       prompt = "Who is in the photo at [x1, y1, x2, y2]?"
       target = "<Name>"

Frames are extracted as JPEGs at 90 quality. bbox = axis-aligned bounds of
the 4 portrait corners, normalized to [0, 1] in (W, H).

Output:
  <out-root>/
      images/<episode>__f<frame_idx>.jpg
      manifest.parquet          (one row per (frame, portrait, caption_type))
      _stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Frame sampling — 3 per episode at even intervals.
FRAME_FRACTIONS = [0.20, 0.50, 0.80]

# Caption types and their share among emitted pairs.
# Each (frame, portrait) emits one pair per caption type.
CAPTION_TYPES = ["location_explicit", "qa_grounded"]

# Camera-LMR slot -> celeb mapping is derived per-episode below.
SHORT_TO_FULL = {"swift": "taylor_swift",
                  "obama": "barack_obama",
                  "lecun": "yann_lecun"}


def slug_to_name(slug: str) -> str:
    """taylor_swift -> Taylor Swift. Special-case LeCun."""
    if slug in ("yann_lecun", "lecun"):
        return "Yann LeCun"
    return " ".join(w.capitalize() for w in slug.replace("-", "_").split("_"))


def format_bbox(bbox: list[float]) -> str:
    """[0.21, 0.34, 0.58, 0.72] → '[0.21, 0.34, 0.58, 0.72]'."""
    return "[" + ", ".join(f"{v:.2f}" for v in bbox) + "]"


def find_camera1_video(ep_dir: Path) -> Path | None:
    """Locate the wrist-cam mp4. Prefer H.264 sidecar if present."""
    cam1 = ep_dir / "videos" / "observation.images.camera1" / "chunk-000"
    if not cam1.is_dir():
        return None
    h264 = sorted(cam1.glob("*__h264.mp4"))
    if h264:
        return h264[0]
    return next(iter(cam1.glob("file-*.mp4")), None)


def pid_to_celeb_full_for_ep(ep_dir: Path) -> dict[str, str] | None:
    """Return {pid_str: celeb_slug_full}. Handles both aug variants
    (augmentation.json["pid_to_celeb"]) and base teleops (seeds + layout)."""
    aug_path = ep_dir / "augmentation.json"
    if aug_path.is_file():
        d = json.loads(aug_path.read_text())
        # v3 aug uses "pid_to_celeb" (full slug), track3 used "pid_to_celeb_full"
        return d.get("pid_to_celeb_full") or d.get("pid_to_celeb")
    # Base teleop fallback
    seeds_path = ep_dir / "portrait_seeds.json"
    if not seeds_path.is_file():
        return None
    seeds = json.loads(seeds_path.read_text())
    celebs_list = seeds.get("celebs", [])
    if len(celebs_list) != 3:
        return None
    # pid order matches list order
    return {str(i): SHORT_TO_FULL.get(s, s)
            for i, s in enumerate(celebs_list)}


def find_corners_path(ep_dir: Path, base_root: Path) -> Path | None:
    """Return path to portrait_corners.json. Aug variants don't have their
    own; we look up the source base teleop via augmentation.json['src_episode']."""
    own = ep_dir / "portrait_corners.json"
    if own.is_file():
        return own
    aug_path = ep_dir / "augmentation.json"
    if aug_path.is_file():
        d = json.loads(aug_path.read_text())
        src = d.get("src_episode")
        if src:
            cand = base_root / src / "portrait_corners.json"
            if cand.is_file():
                return cand
    return None


def process_episode(args: tuple) -> dict:
    ep_dir, out_root_str, base_root_str = args
    out_root = Path(out_root_str)
    base_root = Path(base_root_str)
    records: list[dict] = []
    stats = {"ep": ep_dir.name, "n_records": 0,
             "skipped_no_video": 0,
             "skipped_no_corners": 0,
             "skipped_no_pid_map": 0,
             "skipped_frame_decode": 0,
             "skipped_no_corners_for_pid": 0}

    video = find_camera1_video(ep_dir)
    if video is None:
        stats["skipped_no_video"] = 1
        return {"records": records, "stats": stats}

    corners_path = find_corners_path(ep_dir, base_root)
    if corners_path is None:
        stats["skipped_no_corners"] = 1
        return {"records": records, "stats": stats}
    corners = json.loads(corners_path.read_text())

    pid_to_full = pid_to_celeb_full_for_ep(ep_dir)
    if not pid_to_full:
        stats["skipped_no_pid_map"] = 1
        return {"records": records, "stats": stats}

    n_frames = int(corners.get("n_frames", 538))
    frame_indices = [int(n_frames * f) for f in FRAME_FRACTIONS]

    cap = cv2.VideoCapture(str(video))
    try:
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                stats["skipped_frame_decode"] += 1
                continue
            H, W = frame.shape[:2]

            # Save JPEG
            img_relpath = f"images/{ep_dir.name}__f{fi:04d}.jpg"
            img_abspath = out_root / img_relpath
            img_abspath.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_abspath), frame,
                          [cv2.IMWRITE_JPEG_QUALITY, 90])

            for pid_str, celeb_slug in pid_to_full.items():
                rec = corners["portraits"].get(pid_str, {}).get(str(fi))
                if rec is None or rec.get("corners") is None:
                    stats["skipped_no_corners_for_pid"] += 1
                    continue
                quad = np.asarray(rec["corners"], dtype=np.float32)
                x1f = float(quad[:, 0].min()) / W
                y1f = float(quad[:, 1].min()) / H
                x2f = float(quad[:, 0].max()) / W
                y2f = float(quad[:, 1].max()) / H
                # Clip to [0, 1]
                bbox = [max(0.0, min(1.0, v))
                        for v in (x1f, y1f, x2f, y2f)]
                celeb_name = slug_to_name(celeb_slug)
                bbox_str = format_bbox(bbox)

                for ctype in CAPTION_TYPES:
                    if ctype == "location_explicit":
                        prompt = "What is in this image?"
                        target = (f"The printed photo of {celeb_name} "
                                  f"is at {bbox_str}.")
                    else:  # qa_grounded
                        prompt = (f"Who is in the printed photo at "
                                  f"{bbox_str}?")
                        target = celeb_name
                    records.append({
                        "image_path": img_relpath,
                        "prompt": prompt,
                        "target": target,
                        "bbox_xyxy_norm": bbox,
                        "celeb_name": celeb_name,
                        "celeb_slug": celeb_slug,
                        "caption_type": ctype,
                        "episode": ep_dir.name,
                        "frame_idx": fi,
                        "pid": int(pid_str),
                    })
            stats["n_records"] = len(records)
    finally:
        cap.release()

    return {"records": records, "stats": stats}


def discover_episodes(base_root: Path, aug_root: Path,
                       aug_pattern: str = "__var") -> list[Path]:
    """Mirror the merger's discover() — sorted base teleops first, sorted aug after.
    Base teleops require their own portrait_corners.json; aug variants don't
    (we look up the source base's corners via augmentation.json)."""
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
    p.add_argument("--aug-pattern", default="__var")
    p.add_argument("--out-root", type=Path,
                   default=Path.home() / "LeMonkey/datasets/eval3_objectvla_vl_pairs")
    p.add_argument("--num-workers", type=int, default=64)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N episodes (smoke)")
    args = p.parse_args()

    # IMPORTANT: cv2 should be single-threaded per worker to avoid 64*N thread
    # explosion. Each worker handles ~150 episodes serially with frame decode +
    # JPEG write — pure CPU.
    cv2.setNumThreads(1)

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "images").mkdir(parents=True, exist_ok=True)

    print(f"==> discovering episodes…", flush=True)
    eps = discover_episodes(args.base_root, args.aug_root, args.aug_pattern)
    if args.limit:
        eps = eps[: args.limit]
    print(f"    {len(eps)} episodes; {args.num_workers} workers", flush=True)
    print(f"    expected records: {len(eps)} × {len(FRAME_FRACTIONS)} frames × "
          f"3 portraits × {len(CAPTION_TYPES)} captions = "
          f"~{len(eps) * len(FRAME_FRACTIONS) * 3 * len(CAPTION_TYPES):,}", flush=True)
    print(f"    expected jpegs:   {len(eps) * len(FRAME_FRACTIONS):,}", flush=True)
    print(flush=True)

    all_records: list[dict] = []
    agg_stats = {"n_records": 0, "n_episodes_processed": 0,
                 "skipped_no_video": 0, "skipped_no_corners": 0,
                 "skipped_no_pid_map": 0, "skipped_frame_decode": 0,
                 "skipped_no_corners_for_pid": 0}

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as exe:
        futures = {exe.submit(process_episode,
                                (ep, str(args.out_root), str(args.base_root))): ep
                    for ep in eps}
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            all_records.extend(result["records"])
            for k in agg_stats:
                if k == "n_episodes_processed":
                    continue
                agg_stats[k] = agg_stats.get(k, 0) + result["stats"].get(k, 0)
            agg_stats["n_episodes_processed"] += 1
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t_start
                eta = elapsed / done * (len(eps) - done)
                print(f"    {done}/{len(eps)} eps; "
                      f"{len(all_records):,} records; "
                      f"{elapsed:.0f}s elapsed; ETA {eta:.0f}s",
                      flush=True)

    print(f"\n==> writing manifest…", flush=True)
    df = pd.DataFrame(all_records)
    manifest_path = args.out_root / "manifest.parquet"
    pq.write_table(pa.Table.from_pandas(df), manifest_path, compression="snappy")

    stats_out = {
        **agg_stats,
        "n_records": len(all_records),
        "n_unique_celebs": df["celeb_name"].nunique() if len(df) else 0,
        "n_unique_images": df["image_path"].nunique() if len(df) else 0,
        "wall_seconds": time.time() - t_start,
    }
    (args.out_root / "_stats.json").write_text(json.dumps(stats_out, indent=2))

    print(f"\n==> done in {stats_out['wall_seconds']:.0f}s")
    print(f"    manifest    : {manifest_path}")
    print(f"    rows        : {stats_out['n_records']:,}")
    print(f"    unique imgs : {stats_out['n_unique_images']:,}")
    print(f"    unique celebs: {stats_out['n_unique_celebs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
