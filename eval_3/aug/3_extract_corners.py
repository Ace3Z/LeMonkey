#!/usr/bin/env python3
"""STAGE 3 — convert per-frame portrait masks to ordered 4-corner quadrilaterals.

Reads <episode_dir>/portrait_masks.pkl produced by 2_segment_video.py and
writes <episode_dir>/portrait_corners.json.

Per frame, per portrait:
  1. Decode COCO RLE mask.
  2. cv2.findContours → largest external contour.
  3. cv2.minAreaRect → cv2.boxPoints → 4 corners.
  4. order_tl_tr_br_bl: disambiguate corner order by quadrant from centroid.
  5. Occlusion check: if SAM obj_score < 0 OR mask area < 0.5 × rolling
     median (window 15), mark frame "occluded".
  6. Linear interpolation: occluded runs (consecutive occluded frames)
     are filled by lerping corners between the last valid frame before
     and the first valid frame after the run.

Output schema:
    {
      "video_shape": [H, W],
      "n_frames": int,
      "portraits": {
        "0": {                                  # portrait_id
          "0":   {"corners": [[x,y]×4], "occluded": false, "score": 12.4, "interpolated": false},
          ...
          "599": {...}
        },
        "1": { ... },
        "2": { ... }
      }
    }

Usage:
    python 3_extract_corners.py /path/to/episode_dir
    python 3_extract_corners.py --root ~/LeMonkey/datasets/eval3_quick

See STRATEGY.md §3.3 for design rationale.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util


# ─── Corner ordering ─────────────────────────────────────────────────────────
def order_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """4×2 → 4×2 ordered top-left, top-right, bottom-right, bottom-left.

    Strategy: assign each point to a quadrant by angle from centroid.
    For a roughly axis-aligned rectangle this gives stable ordering;
    for highly-rotated rectangles (>45°) it still works since minAreaRect's
    orientation is consistent within an episode.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    centroid = pts.mean(0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    # quadrants:
    #   top-left  : angle ∈ (-π,-π/2)   -> slot 0
    #   top-right : angle ∈ (-π/2, 0)   -> slot 1
    #   bot-right : angle ∈ ( 0,  π/2)  -> slot 2
    #   bot-left  : angle ∈ ( π/2, π)   -> slot 3
    slots = np.where(angles < -np.pi/2, 0,
            np.where(angles <  0,        1,
            np.where(angles <  np.pi/2,  2, 3)))
    out = np.zeros((4, 2), dtype=np.float32)
    used = [False] * 4
    # First pass: assign each point to its slot (handle duplicates by next-best slot)
    for slot in (0, 1, 2, 3):
        idxs = np.where(slots == slot)[0]
        if len(idxs) == 0:
            continue
        # take the one closest to the slot's "ideal" angle to break ties
        ideal_angles = {0: -3*np.pi/4, 1: -np.pi/4, 2: np.pi/4, 3: 3*np.pi/4}
        d = np.abs(np.arctan2(np.sin(angles[idxs] - ideal_angles[slot]),
                              np.cos(angles[idxs] - ideal_angles[slot])))
        chosen = idxs[d.argmin()]
        out[slot] = pts[chosen]
        used[slot] = True
    # Fill any unused slots by leftover points (degenerate case)
    leftover_pts = [pts[i] for i in range(4) if not any(np.allclose(pts[i], out[s]) for s in range(4) if used[s])]
    leftover_iter = iter(leftover_pts)
    for s in range(4):
        if not used[s]:
            out[s] = next(leftover_iter)
    return out


# ─── Per-frame extraction ────────────────────────────────────────────────────
def extract_corners_one_frame(rle: dict, score: float, h: int, w: int) -> tuple[np.ndarray | None, float, int]:
    """Return (corners 4×2 or None on failure, score, mask_area)."""
    mask = mask_util.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    area = int(mask.sum())
    if area == 0:
        return None, score, 0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, score, 0
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:                   # ignore tiny shards
        return None, score, area
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_tl_tr_br_bl(box), score, area


# ─── Occlusion + interpolation ───────────────────────────────────────────────
def detect_and_interp_occlusion(
    raw: list[tuple[np.ndarray | None, float, int]],
    *, area_drop_ratio: float = 0.5, score_threshold: float = 0.0,
    rolling_window: int = 15,
) -> list[dict]:
    """
    raw: list (one per frame) of (corners | None, score, area)
    Returns list of dicts: {corners, occluded, score, interpolated}
    """
    n = len(raw)
    rolling_areas: list[int] = []
    occluded = [False] * n

    # First pass — flag occluded frames
    for i, (corners, score, area) in enumerate(raw):
        rolling_areas.append(area)
        if len(rolling_areas) > rolling_window:
            rolling_areas.pop(0)
        rolling_med = float(np.median([a for a in rolling_areas if a > 0])) if any(rolling_areas) else 0.0
        if (corners is None) or (score < score_threshold) or (rolling_med > 0 and area < area_drop_ratio * rolling_med):
            occluded[i] = True

    # Second pass — interpolate occluded runs
    out: list[dict] = []
    for i, (corners, score, _area) in enumerate(raw):
        if not occluded[i] and corners is not None:
            out.append({"corners": corners.tolist(), "occluded": False, "score": float(score), "interpolated": False})
            continue
        # find nearest valid neighbours
        prev_i, next_i = None, None
        for j in range(i - 1, -1, -1):
            if not occluded[j] and raw[j][0] is not None:
                prev_i = j
                break
        for j in range(i + 1, n):
            if not occluded[j] and raw[j][0] is not None:
                next_i = j
                break
        if prev_i is None and next_i is None:
            # whole video is bad; fall back to the raw corners we have (could be None)
            fallback = corners.tolist() if corners is not None else None
            out.append({"corners": fallback, "occluded": True, "score": float(score), "interpolated": False})
            continue
        if prev_i is None:
            out.append({"corners": raw[next_i][0].tolist(), "occluded": True, "score": float(score), "interpolated": True})
            continue
        if next_i is None:
            out.append({"corners": raw[prev_i][0].tolist(), "occluded": True, "score": float(score), "interpolated": True})
            continue
        # linear interp
        t = (i - prev_i) / (next_i - prev_i)
        a = np.asarray(raw[prev_i][0])
        b = np.asarray(raw[next_i][0])
        interp = ((1 - t) * a + t * b).tolist()
        out.append({"corners": interp, "occluded": True, "score": float(score), "interpolated": True})
    return out


# ─── Per-episode driver ──────────────────────────────────────────────────────
def process_episode(ep_dir: Path, *, force: bool) -> dict:
    pkl = ep_dir / "portrait_masks.pkl"
    if not pkl.is_file():
        return {"ep": ep_dir.name, "error": "portrait_masks.pkl missing — run 2_segment_video.py first"}
    out_json = ep_dir / "portrait_corners.json"
    if out_json.is_file() and not force:
        return {"ep": ep_dir.name, "skipped": True}

    with open(pkl, "rb") as f:
        cache = pickle.load(f)
    masks: dict[int, dict[int, dict]] = cache["masks"]
    n_frames = max(masks.keys()) + 1 if masks else 0

    # Probe first valid frame to learn H, W
    h = w = None
    for fi in sorted(masks.keys()):
        for pid, payload in masks[fi].items():
            mask = mask_util.decode(payload["rle"])
            h, w = mask.shape[:2]
            break
        if h is not None:
            break
    if h is None:
        return {"ep": ep_dir.name, "error": "no decodable masks in pkl"}

    portrait_ids = sorted({pid for f in masks.values() for pid in f.keys()})
    if len(portrait_ids) != 3:
        return {"ep": ep_dir.name, "error": f"expected 3 portraits, got {portrait_ids}"}

    out: dict = {"video_shape": [h, w], "n_frames": n_frames, "portraits": {}}
    interp_counts: dict[int, int] = {}
    for pid in portrait_ids:
        raw_per_frame: list[tuple[np.ndarray | None, float, int]] = []
        for fi in range(n_frames):
            payload = masks.get(fi, {}).get(pid)
            if payload is None:
                raw_per_frame.append((None, -1.0, 0))
            else:
                raw_per_frame.append(
                    extract_corners_one_frame(payload["rle"], payload["score"], h, w)
                )
        per_frame = detect_and_interp_occlusion(raw_per_frame)
        out["portraits"][str(pid)] = {str(i): rec for i, rec in enumerate(per_frame)}
        interp_counts[pid] = sum(1 for r in per_frame if r["interpolated"])

    out_json.write_text(json.dumps(out, indent=2))
    return {
        "ep": ep_dir.name, "saved": str(out_json),
        "n_frames": n_frames, "interp": interp_counts,
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("episode_dir", nargs="?", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if (args.episode_dir is None) == (args.root is None):
        print("[ERROR] specify exactly one of: episode_dir, --root", file=sys.stderr)
        return 2

    if args.episode_dir:
        ep_dirs = [Path(args.episode_dir)]
    else:
        root = Path(args.root)
        ep_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    results: list[dict] = []
    for ep_dir in ep_dirs:
        try:
            r = process_episode(ep_dir, force=args.force)
        except Exception as e:
            r = {"ep": ep_dir.name, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        if "saved" in r:
            print(f"  ✓ {r['ep']:50s}  {r['n_frames']:>4} frames, interp/portrait={r['interp']}")
        elif "skipped" in r:
            print(f"  - {r['ep']:50s}  (skipped)")
        else:
            print(f"  ✗ {r['ep']:50s}  {r.get('error','?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
