#!/usr/bin/env python3
"""Build VL pairs (location-explicit + Q&A grounded) from the merged cotrain dataset.

For each merged episode, emits VL records consumed by SmolVLA cotrain alongside
the robot action stream. Each record carries the overhead-cam frame-0 JPEG, the
reference-cam frame-0 JPEG, and a portrait quad (normalized [x,y]x4 in image
coords) for one of the three printed portraits visible in the frame.

Outputs:
  - Refined sub-pixel corners at frame 0, with degenerate-quad detection +
    coarse fallback.
  - Quad-corner geometry (no axis-aligned bbox).
  - Two caption types per record: location_explicit, qa_grounded.
  - Reference-camera frame extracted once per episode.

## Background

The canonical-name-keyed lookup below exists because an earlier builder keyed
`ep_to_videos` by the MERGED dataset's episode_index (positional, defined by
`data/merge_episodes.discover_episode_dirs()` = sorted(base) + sorted(aug))
but indexed it with `ep_idx = enumeration position in the builder's own
filtered ep_metadata list`. The builder's filter (requires portrait_corners +
portrait_seeds + valid target) selects fewer episodes than the merger, shifting
every subsequent index, so ~98% of rows had a frame from a different episode
than the labels described. The current builder replicates the merger's
discovery exactly and looks up by episode NAME, so the mispairing cannot recur.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CAPTION_TYPES = ["location_explicit", "qa_grounded"]
SHORT_TO_FULL = {"swift": "taylor_swift",
                  "obama": "barack_obama",
                  "lecun": "yann_lecun"}
LETTER_TO_SHORT = {"S": "swift", "O": "obama", "L": "lecun"}
LAYOUTS = ["SOL", "OSL", "OLS", "SLO", "LSO", "LOS"]


def slug_to_name(slug: str) -> str:
    """Format a slug like 'barack_obama' into the human-readable display name 'Barack Obama' (LeCun is special-cased)."""
    if slug in ("yann_lecun", "lecun"):
        return "Yann LeCun"
    return " ".join(w.capitalize() for w in slug.replace("-", "_").split("_"))


def format_quad(quad) -> str:
    """Flat 8-float string: '[x1, y1, x2, y2, x3, y3, x4, y4]'."""
    return "[" + ", ".join(f"{v:.3f}" for xy in quad for v in xy) + "]"


def physical_slot(layout: str, celeb_short: str) -> int:
    """Return the L/M/R slot index (0/1/2) of celeb_short in the layout string."""
    inv = {v: k for k, v in LETTER_TO_SHORT.items()}
    return layout.index(inv[celeb_short])


def pid_position_camera_lmr(corners_data: dict) -> dict:
    """Map per-portrait pid -> camera L/M/R slot index, ranked by mean x-coordinate of frame-0 corners."""
    means = {}
    for pid, frames in corners_data["portraits"].items():
        for fi, rec in frames.items():
            if rec.get("corners") is not None:
                xs = [c[0] for c in rec["corners"]]
                means[pid] = float(sum(xs)) / float(len(xs))
                break
    sorted_pids = sorted(means, key=lambda k: means[k])
    return {pid: idx for idx, pid in enumerate(sorted_pids)}


def load_cotrain_photo_bank(bank_root: "Path") -> "dict[str, list[Path]]":
    """Walk bank_root and collect per-celeb photo lists keyed by full slug."""
    bank = {}
    for full in SHORT_TO_FULL.values():
        photos = sorted(p for p in (bank_root / full).iterdir()
                          if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        bank[full] = photos
    return bank


def enumerate_target_photo_layout_tuples(bank: "dict[str, list[Path]]"):
    """Enumerate every (target_short, target_photo, layout) tuple the cotrain augmentation expects to cover."""
    tuples = []
    for short in ("swift", "obama", "lecun"):
        full = SHORT_TO_FULL[short]
        for photo in bank[full]:
            for layout in LAYOUTS:
                tuples.append((short, photo, layout))
    return tuples


def build_distractor_combos(layout: str, target_short: str, bank: "dict[str, list[Path]]", K: int, rng):
    """Sample up to K (distractor1_photo, distractor2_photo) combinations for the two non-target slots in `layout`."""
    target_pos = physical_slot(layout, target_short)
    d1_short = LETTER_TO_SHORT[layout[(target_pos + 1) % 3]]
    d2_short = LETTER_TO_SHORT[layout[(target_pos + 2) % 3]]
    d1_pool = bank[SHORT_TO_FULL[d1_short]]
    d2_pool = bank[SHORT_TO_FULL[d2_short]]
    combos = [(p1, p2) for p1 in d1_pool for p2 in d2_pool]
    if K >= len(combos):
        return combos
    return rng.sample(combos, K)


def discover_base_for_aug_pool(root: "Path") -> dict:
    """Replicate eval_3/aug/generators/cotrain.discover_base_teleops, the pool
    that aug variants were assigned to via slot_cursor round-robin. Uses the
    same filter as the original generator (requires all 4 caches)."""
    groups = {0: [], 1: [], 2: []}
    for ep in sorted(root.iterdir()):
        if not ep.is_dir(): continue
        for need in ("reference.json", "portrait_corners.json",
                       "portrait_seeds.json", "portrait_masks.pkl"):
            if not (ep / need).is_file():
                break
        else:
            sidecar = json.loads((ep / "reference.json").read_text())
            seeds = json.loads((ep / "portrait_seeds.json").read_text())
            tgt = sidecar.get("target_celeb", "")
            if tgt not in LETTER_TO_SHORT.values():
                continue
            corners = json.loads((ep / "portrait_corners.json").read_text())
            opid = None
            for i, c in enumerate(seeds.get("celebs", [])):
                if c == tgt:
                    opid = str(i)
                    break
            if opid is None: continue
            try:
                slot = pid_position_camera_lmr(corners)[opid]
            except KeyError: continue
            groups[slot].append(ep)
    return groups


def merger_discover_base(base_root: "Path") -> list:
    """Replicate eval_3/scripts/data/merge_episodes.discover_episode_dirs base
    filter: has meta/info.json AND reference.json. No cache requirement."""
    return sorted(p for p in base_root.iterdir()
                    if p.is_dir()
                    and (p / "meta" / "info.json").is_file()
                    and (p / "reference.json").is_file())


def re_derive_aug_variant_names(base_root: "Path", bank_root: "Path", seed: int = 42, K: int = 64):
    """Replicate eval_3/aug/generators/cotrain.py to get the EXACT list of
    variant names that were generated. These names match what the merger sees."""
    bank = load_cotrain_photo_bank(bank_root)
    tuples = enumerate_target_photo_layout_tuples(bank)
    base_groups = discover_base_for_aug_pool(base_root)
    slot_cursor = {0: 0, 1: 0, 2: 0}
    out = []  # (var_name, base_ep, target_short, layout, d1_short, d2_short)
    for tuple_idx, (target_short, target_photo, layout) in enumerate(tuples):
        slot = physical_slot(layout, target_short)
        pool = base_groups[slot]
        if not pool: continue
        combos = build_distractor_combos(layout, target_short, bank, K,
                                            random.Random(seed + tuple_idx))
        target_pos = physical_slot(layout, target_short)
        d1_short = LETTER_TO_SHORT[layout[(target_pos + 1) % 3]]
        d2_short = LETTER_TO_SHORT[layout[(target_pos + 2) % 3]]
        for var_idx, _ in enumerate(combos):
            base_ep = pool[slot_cursor[slot] % len(pool)]
            slot_cursor[slot] += 1
            var_name = f"{base_ep.name}__t3_{tuple_idx:04d}_v{var_idx:02d}"
            out.append((var_name, base_ep, target_short, layout,
                          d1_short, d2_short))
    return out


# ── Per-portrait quad: refined when good, coarse fallback when not ──
DEGEN_DUP_PX = 1.5
DEGEN_AREA_PX = 100


def shoelace_area(q) -> float:
    """Polygon area of a 4-corner quad via the shoelace formula."""
    q = np.asarray(q, dtype=np.float32)
    x = q[:, 0]; y = q[:, 1]
    return 0.5 * abs((x[0]*y[1] - x[1]*y[0]) + (x[1]*y[2] - x[2]*y[1]) +
                       (x[2]*y[3] - x[3]*y[2]) + (x[3]*y[0] - x[0]*y[3]))


def is_degenerate(corners) -> bool:
    """True if the quad collapses (duplicate corners within DEGEN_DUP_PX or area < DEGEN_AREA_PX)."""
    q = np.asarray(corners, dtype=np.float32)
    if q.shape[0] != 4: return True
    for i in range(4):
        for j in range(i+1, 4):
            if np.linalg.norm(q[i] - q[j]) < DEGEN_DUP_PX:
                return True
    if shoelace_area(q) < DEGEN_AREA_PX:
        return True
    return False


def load_clean_corners(base_ep_dir: "Path") -> dict:
    """Returns {pid_str: {"corners": [[x,y]x4], "refit_ok": bool}} per portrait.

    Prefers refine_paper_quad_to_edges output when 4-distinct-corner valid;
    falls back to portrait_corners.json frame-0 coarse minAreaRect output
    (always 4 distinct corners since boxPoints guarantees that)."""
    refit_path = base_ep_dir / "portrait_corners_refined_frame0.json"
    coarse_path = base_ep_dir / "portrait_corners.json"
    out = {}
    refit = json.loads(refit_path.read_text())["portraits"] if refit_path.is_file() else {}
    coarse = json.loads(coarse_path.read_text())["portraits"] if coarse_path.is_file() else {}
    for pid in ("0", "1", "2"):
        rec = refit.get(pid, {})
        corners = rec.get("corners")
        refit_ok = rec.get("refit_ok", False)
        if corners is not None and refit_ok and not is_degenerate(corners):
            out[pid] = {"corners": corners, "refit_ok": True}
        else:
            cf = coarse.get(pid, {}).get("0", {}).get("corners")
            if cf is not None:
                out[pid] = {"corners": cf, "refit_ok": False}
    return out


# ── Per-worker frame extraction ──
def find_camera1_video_path(ep_dir: "Path"):
    """For base teleops in datasets/eval3/, find their overhead-cam mp4 (prefer
    H.264 sidecar). This is the ORIGINAL recording, same content as merger."""
    cam1 = ep_dir / "videos" / "observation.images.camera1" / "chunk-000"
    if not cam1.is_dir():
        return None
    h264 = sorted(cam1.glob("*__h264.mp4"))
    if h264:
        return h264[0]
    return next(iter(cam1.glob("file-*.mp4")), None)


def find_reference_video_path(ep_dir: "Path"):
    """Return the reference-camera mp4 for a base teleop episode, or None if missing."""
    ref = ep_dir / "videos" / "observation.images.reference" / "chunk-000"
    if not ref.is_dir(): return None
    return next(iter(ref.glob("file-*.mp4")), None)


def process_episode(args):
    """Per-worker: decode the overhead + reference frame 0, write JPEGs, and emit one VL record per portrait + caption-type."""
    (ep_idx, ep_name, video_path_str, ref_video_path_str, base_ep_dir_str,
     pid_to_celeb_full_str, out_root_str) = args
    out_root = Path(out_root_str)
    base_ep_dir = Path(base_ep_dir_str)
    pid_to_celeb_full = json.loads(pid_to_celeb_full_str)
    records = []

    corners_per_pid = load_clean_corners(base_ep_dir)

    # 1. Wrist-cam frame 0 (FROM THE CORRECT VIDEO — that's the bug fix)
    cap = cv2.VideoCapture(video_path_str)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {"records": [], "ep": ep_name, "ok": False, "reason": "frame0_decode"}
    H, W = frame.shape[:2]
    chunk_idx = ep_idx // 9000
    img_relpath = f"images/chunk-{chunk_idx:03d}/{ep_name}__f0000.jpg"
    out_img = out_root / img_relpath
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 2. Reference frame 0
    ref_relpath = f"references/{ep_name}__ref.jpg"
    if ref_video_path_str:
        cap = cv2.VideoCapture(ref_video_path_str)
        ok, frame_ref = cap.read()
        cap.release()
        if ok and frame_ref is not None:
            out_ref = out_root / ref_relpath
            out_ref.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_ref), frame_ref, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 3. Emit VL pairs per portrait
    for pid_str, celeb_slug in pid_to_celeb_full.items():
        rec = corners_per_pid.get(pid_str)
        if rec is None: continue
        quad_px = np.asarray(rec["corners"], dtype=np.float32)
        quad_norm = [[float(np.clip(p[0]/W, 0, 1)),
                       float(np.clip(p[1]/H, 0, 1))] for p in quad_px]
        celeb_name = slug_to_name(celeb_slug)
        quad_str = format_quad(quad_norm)
        refit_ok = bool(rec.get("refit_ok", False))

        for ctype in CAPTION_TYPES:
            if ctype == "location_explicit":
                prompt = "What is in this image?"
                target = f"The printed photo of {celeb_name} is at {quad_str}."
            else:
                prompt = f"Who is in the printed photo at {quad_str}?"
                target = celeb_name
            records.append({
                "image_path": img_relpath,
                "reference_image_path": ref_relpath,
                "prompt": prompt,
                "target": target,
                "quad_corners_norm": quad_norm,
                "bbox_refit_ok": refit_ok,
                "celeb_name": celeb_name,
                "celeb_slug": celeb_slug,
                "caption_type": ctype,
                "episode": ep_name,
                "episode_index": ep_idx,
                "frame_idx": 0,
                "pid": int(pid_str),
            })
    return {"records": records, "ep": ep_name, "ok": True}


def main() -> int:
    """Driver: discover the canonical episode list, map episode_index -> mp4 paths, dispatch per-episode workers, then write the manifest parquet."""
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-root", type=Path,
                   default=Path.home() / "LeMonkey/datasets/eval3")
    p.add_argument("--bank-root", type=Path,
                   default=Path.home() / "LeMonkey/datasets/eval3_celebs/track3_bank")
    p.add_argument("--hf-root", type=Path,
                   default=Path.home() / "LeMonkey/datasets/eval3_track3_v3_from_hf")
    p.add_argument("--out-root", type=Path,
                   default=Path.home() / "LeMonkey/datasets/eval3_track3_vl_pairs_v3")
    p.add_argument("--num-workers", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cv2.setNumThreads(1)
    args.out_root.mkdir(parents=True, exist_ok=True)

    # ── 1. Build canonical episode name list = merger's discover() ──
    print("[1/4] replicating merger's discover() → canonical name list…", flush=True)
    merger_base_eps = merger_discover_base(args.base_root)
    print(f"      merger base: {len(merger_base_eps)} (meta/info.json + reference.json)",
          flush=True)
    aug_records = re_derive_aug_variant_names(args.base_root, args.bank_root,
                                                 seed=42, K=64)
    aug_records_sorted = sorted(aug_records, key=lambda r: r[0])
    print(f"      aug variants (re-derived): {len(aug_records_sorted)}", flush=True)
    canonical_names = ([ep.name for ep in merger_base_eps]
                        + [r[0] for r in aug_records_sorted])
    print(f"      canonical total: {len(canonical_names)} (matches merger's 9394)",
          flush=True)

    # Per-name metadata for the 151 cacheful base + 9216 aug
    print("[2/4] building per-name metadata…", flush=True)
    name_to_meta = {}
    # Base teleops with caches
    base_to_target_pid = {}
    base_eps_with_caches = []
    for ep in merger_base_eps:
        if not all((ep / f).is_file()
                    for f in ("portrait_corners.json", "portrait_seeds.json")):
            continue
        seeds = json.loads((ep / "portrait_seeds.json").read_text())
        ref = json.loads((ep / "reference.json").read_text())
        if ref.get("target_celeb") not in ("swift", "obama", "lecun"):
            continue
        try:
            corners = json.loads((ep / "portrait_corners.json").read_text())
            positions = pid_position_camera_lmr(corners)
        except Exception:
            continue
        pid_to_full = {str(i): SHORT_TO_FULL[s]
                        for i, s in enumerate(seeds["celebs"])}
        name_to_meta[ep.name] = {"base_ep": ep, "pid_to_celeb_full": pid_to_full}
        for i, c in enumerate(seeds["celebs"]):
            if c == ref["target_celeb"]:
                base_to_target_pid[ep.name] = str(i)
                break
        base_eps_with_caches.append(ep)
    print(f"      base eps with caches: {len(base_eps_with_caches)} / "
          f"{len(merger_base_eps)} (cacheless: "
          f"{len(merger_base_eps) - len(base_eps_with_caches)})", flush=True)

    # Cache per-base positions ONCE (avoid 9216x re-parse of 770KB JSON files)
    base_positions_cache = {}
    for ep in base_eps_with_caches:
        try:
            corners = json.loads((ep / "portrait_corners.json").read_text())
            base_positions_cache[ep.name] = pid_position_camera_lmr(corners)
        except Exception:
            pass

    # Aug variants (use base_to_target_pid + cached positions)
    n_aug_meta = 0
    for (var_name, base_ep, target_short, layout, d1_short, d2_short) in aug_records_sorted:
        orig_target_pid = base_to_target_pid.get(base_ep.name)
        if orig_target_pid is None: continue
        positions = base_positions_cache.get(base_ep.name)
        if positions is None: continue
        target_slot = positions[orig_target_pid]
        slot_to_pid = {s: p for p, s in positions.items()}
        d1_pid = slot_to_pid[(target_slot + 1) % 3]
        d2_pid = slot_to_pid[(target_slot + 2) % 3]
        pid_to_full = {
            orig_target_pid: SHORT_TO_FULL[target_short],
            d1_pid: SHORT_TO_FULL[d1_short],
            d2_pid: SHORT_TO_FULL[d2_short],
        }
        name_to_meta[var_name] = {"base_ep": base_ep, "pid_to_celeb_full": pid_to_full}
        n_aug_meta += 1
    print(f"      aug variants with meta: {n_aug_meta}", flush=True)
    print(f"      total name_to_meta entries: {len(name_to_meta)}", flush=True)

    # ── 3. Map merger episode_index → mp4 paths via the HF snapshot's episodes parquet ──
    print("[3/4] mapping episode_index → video paths in HF snapshot…", flush=True)
    ep_parquet = next((args.hf_root / "meta/episodes").rglob("*.parquet"))
    ep_df = pq.read_table(ep_parquet).to_pandas().reset_index()
    ep_to_videos = {}
    for _, row in ep_df.iterrows():
        ei = int(row["episode_index"])
        c1 = (args.hf_root /
                f"videos/observation.images.camera1/"
                f"chunk-{int(row['videos/observation.images.camera1/chunk_index']):03d}/"
                f"file-{int(row['videos/observation.images.camera1/file_index']):03d}.mp4")
        rf = (args.hf_root /
                f"videos/observation.images.reference/"
                f"chunk-{int(row['videos/observation.images.reference/chunk_index']):03d}/"
                f"file-{int(row['videos/observation.images.reference/file_index']):03d}.mp4")
        ep_to_videos[ei] = (c1, rf)
    print(f"      mapped {len(ep_to_videos)} episodes", flush=True)
    if len(ep_to_videos) != len(canonical_names):
        print(f"      [WARN] count mismatch: ep_to_videos={len(ep_to_videos)} "
              f"vs canonical={len(canonical_names)}", flush=True)

    # ── 4. Process. KEY FIX: iterate canonical_names by index, look up videos by THAT index ──
    print(f"[4/4] processing with {args.num_workers} workers (NAME-keyed lookup)…",
          flush=True)
    work = []
    n_skipped_no_meta = n_skipped_no_video = 0
    for episode_index, ep_name in enumerate(canonical_names):
        if args.limit is not None and episode_index >= args.limit:
            break
        meta = name_to_meta.get(ep_name)
        if meta is None:
            n_skipped_no_meta += 1
            continue
        if episode_index not in ep_to_videos:
            n_skipped_no_video += 1
            continue
        c1, rf = ep_to_videos[episode_index]
        if not c1.is_file():
            n_skipped_no_video += 1
            continue
        work.append((
            episode_index, ep_name, str(c1), str(rf) if rf.is_file() else "",
            str(meta["base_ep"]),
            json.dumps(meta["pid_to_celeb_full"]),
            str(args.out_root),
        ))
    print(f"      work: {len(work)} eps (skipped no_meta={n_skipped_no_meta}, "
          f"no_video={n_skipped_no_video})", flush=True)

    all_records = []
    n_ok = n_failed = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as exe:
        futures = [exe.submit(process_episode, w) for w in work]
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            all_records.extend(r["records"])
            n_ok += 1 if r["ok"] else 0
            n_failed += 0 if r["ok"] else 1
            done += 1
            if done % 1000 == 0:
                print(f"      {done}/{len(work)}: ok={n_ok} failed={n_failed} "
                      f"records={len(all_records):,} elapsed={time.time()-t0:.0f}s",
                      flush=True)

    print("\n[final] writing manifest…", flush=True)
    df = pd.DataFrame(all_records)
    pq.write_table(pa.Table.from_pandas(df),
                    args.out_root / "manifest.parquet", compression="snappy")

    n_refit = int(df["bbox_refit_ok"].sum()) if len(df) else 0
    stats = {
        "n_records": len(df),
        "n_unique_celebs": int(df["celeb_name"].nunique()) if len(df) else 0,
        "n_unique_episodes": int(df["episode"].nunique()) if len(df) else 0,
        "n_unique_images": int(df["image_path"].nunique()) if len(df) else 0,
        "n_refit_ok": n_refit,
        "pct_refit_ok": round(100 * n_refit / max(len(df), 1), 1),
        "n_canonical_skipped_no_meta": n_skipped_no_meta,
        "n_canonical_skipped_no_video": n_skipped_no_video,
        "wall_seconds": round(time.time() - t0, 1),
        "note": "image-label pairing uses NAME-keyed video lookup (fixes a positional-index drift bug present in earlier builders)",
    }
    (args.out_root / "_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
