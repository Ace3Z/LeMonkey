#!/usr/bin/env python3
"""Recompute sub-pixel refined corners at frame 0 for every base teleop.

The legacy per-frame corner extractor (`_legacy/stage3_extract_corners.py`)
overwrote the refined corners with coarse per-frame `minAreaRect` output. The
original sub-pixel quad from `refine_paper_quad_to_edges()` only persists
implicitly as a rasterized mask in `portrait_masks.pkl["M_0_per_pid"]` —
re-extracting from that mask is lossy and sometimes degenerate.

This script re-runs the refinement using each base teleop's `frame_0.png`
+ SAM mask + the sibling `refine_paper_quad` module, saving the sub-pixel
refined corners as `<ep>/portrait_corners_refined_frame0.json`.

Aug variants share the base teleop's paper geometry (the inpainter does not
move the portraits), so refining at the base teleops covers every episode in
the merged dataset.

Output (per base teleop):
    <ep>/portrait_corners_refined_frame0.json:
        {
            "video_shape": [H, W],
            "pipeline_version": "refine_paper_quad_v10_recompute",
            "portraits": {
                "0": {"corners": [[x,y]×4], "refit_ok": true|false, "source": "..."},
                "1": {...},
                "2": {...}
            }
        }
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pycocotools.mask as mask_util


# Lazy import of refine_paper_quad (sibling module in aug/stages/).
_REFINE_FN = None


def get_refine_fn() -> Callable:
    global _REFINE_FN
    if _REFINE_FN is None:
        spec = ilu.spec_from_file_location(
            "refine_paper_quad",
            str(Path(__file__).resolve().parent / "refine_paper_quad.py"))
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _REFINE_FN = mod.refine_paper_quad_to_edges
    return _REFINE_FN


def order_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Reorder 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                       pts[np.argmax(s)], pts[np.argmax(d)]])


def refine_one_ep(ep_dir_str: str) -> dict:
    """Re-run sub-pixel paper-quad refinement for one base episode and write
    portrait_corners_refined_frame0.json. Returns a stats dict
    ``{ep, ok, reason, n_refit_ok, n_total}`` describing the outcome."""
    ep_dir = Path(ep_dir_str)
    out_path = ep_dir / "portrait_corners_refined_frame0.json"
    stats = {"ep": ep_dir.name, "ok": False, "reason": ""}

    frame0_path = ep_dir / "frame_0.png"
    pkl_path = ep_dir / "portrait_masks.pkl"
    if not (frame0_path.is_file() and pkl_path.is_file()):
        stats["reason"] = "missing_frame0_or_pkl"
        return stats

    frame = cv2.imread(str(frame0_path))
    if frame is None:
        stats["reason"] = "frame0_decode_failed"
        return stats
    H, W = frame.shape[:2]

    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)
    masks = pkl.get("masks", {}).get(0, {})  # frame 0
    if not masks or len(masks) < 3:
        stats["reason"] = "no_frame0_masks"
        return stats

    refine_fn = get_refine_fn()
    out = {
        "video_shape": [H, W],
        "pipeline_version": "refine_paper_quad_v10_recompute",
        "portraits": {},
    }
    for pid_int, payload in masks.items():
        if "rle" not in payload:
            out["portraits"][str(pid_int)] = {
                "corners": None, "refit_ok": False, "source": "no_rle",
            }
            continue
        sam_mask = mask_util.decode(payload["rle"]).astype(np.uint8)
        if sam_mask.ndim == 3:
            sam_mask = sam_mask[:, :, 0]
        # Coarse seed: minAreaRect of SAM mask
        contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
        if not contours:
            out["portraits"][str(pid_int)] = {
                "corners": None, "refit_ok": False, "source": "no_contour",
            }
            continue
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        coarse_corners = cv2.boxPoints(rect)
        coarse_corners = order_tl_tr_br_bl(coarse_corners)

        try:
            refined = refine_fn(frame, coarse_corners, sam_mask=sam_mask,
                                  verbose=False, debug_dir=None)
        except Exception as e:
            refined = None
            stats[f"err_pid{pid_int}"] = f"{type(e).__name__}: {str(e)[:60]}"

        if refined is not None:
            out["portraits"][str(pid_int)] = {
                "corners": refined.tolist(),
                "refit_ok": True,
                "source": "refine_paper_quad_v10",
            }
        else:
            out["portraits"][str(pid_int)] = {
                "corners": coarse_corners.tolist(),
                "refit_ok": False,
                "source": "minAreaRect_fallback",
            }

    out_path.write_text(json.dumps(out, indent=2))
    n_ok = sum(1 for v in out["portraits"].values() if v.get("refit_ok"))
    stats["ok"] = True
    stats["n_refit_ok"] = n_ok
    stats["n_total"] = len(out["portraits"])
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-root", type=Path,
                   default=Path.home() / "LeMonkey/datasets/eval3",
                   help="Root containing the base teleop episodes to refine")
    p.add_argument("--num-workers", type=int, default=16,
                   help="Refinement uses cv2 (Canny+Hough+cornerSubPix); moderate parallelism is fine.")
    p.add_argument("--force", action="store_true",
                   help="Recompute even when portrait_corners_refined_frame0.json already exists")
    args = p.parse_args()

    cv2.setNumThreads(1)

    base_eps = sorted(p for p in args.base_root.iterdir()
                        if p.is_dir() and (p / "portrait_masks.pkl").is_file()
                        and (p / "frame_0.png").is_file())
    if not args.force:
        base_eps = [ep for ep in base_eps
                      if not (ep / "portrait_corners_refined_frame0.json").is_file()]
    print(f"==> refining {len(base_eps)} base teleops with {args.num_workers} workers",
          flush=True)
    if not base_eps:
        print("    nothing to do (--force to recompute)")
        return 0

    t0 = time.time()
    n_ok = n_failed = 0
    all_refit_counts = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as exe:
        futures = [exe.submit(refine_one_ep, str(ep)) for ep in base_eps]
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            if r["ok"]:
                n_ok += 1
                all_refit_counts.append(r["n_refit_ok"])
            else:
                n_failed += 1
                print(f"  [WARN] {r['ep']}: {r['reason']}", flush=True)
            if (i + 1) % 20 == 0 or (i + 1) == len(base_eps):
                avg = sum(all_refit_counts) / max(len(all_refit_counts), 1)
                print(f"    {i+1}/{len(base_eps)}: ok={n_ok} failed={n_failed} "
                      f"avg_refit_ok_per_ep={avg:.2f}/3", flush=True)

    print(f"\n==> done in {time.time() - t0:.1f}s")
    print(f"    ok={n_ok}, failed={n_failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
