#!/usr/bin/env python3
"""Render a single composite showing every step of Stage 3 paper-quad refit.

`refine_paper_quad_to_edges` already dumps 5 progressive debug PNGs when
called with ``debug_dir``:

  01_band_and_coarse_quad.png             edge band (yellow ring) + SAM quad
  02_canny_edges_in_band.png              Canny edges in band + coarse quad
  03_hough_lines.png                      all HoughLinesP line segments
  04_oriented_lines_and_selected_sides.png 4 outermost lines per side bin
  05_final_refined_vs_coarse.png          refined quad (green) vs coarse (red)

This driver picks one portrait of one base teleop, calls the refit with a
temp ``debug_dir``, reads back the 5 PNGs, and stacks them into a single
composite for the aug README.

Usage:
    python eval_3/aug/dbg/refine_quad_panels.py \\
        --episode datasets/eval3/quick_lecun_LSO_ep01_20260511_205000 \\
        --pid 2 \\
        --out media/figures/aug/stage3_refine_pipeline.png
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import pickle
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _load_refine_fn():
    """Load refine_paper_quad_to_edges from the sibling stages/ module."""
    here = Path(__file__).resolve().parent
    target = here.parent / "stages" / "refine_paper_quad.py"
    spec = ilu.spec_from_file_location("refine_paper_quad", str(target))
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.refine_paper_quad_to_edges


def order_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Reorder 4 points as top-left, top-right, bottom-right, bottom-left
    (robust against rotated rhombuses; see stage4_steps_panel.py)."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    i_tl = int(np.argmin(s))
    i_br = int(np.argmax(s))
    others = [i for i in range(4) if i not in (i_tl, i_br)]
    a, b = others
    if pts[a, 0] >= pts[b, 0]:
        i_tr, i_bl = a, b
    else:
        i_tr, i_bl = b, a
    return np.array([pts[i_tl], pts[i_tr], pts[i_br], pts[i_bl]],
                     dtype=np.float32)


def crop_to_portrait(img: np.ndarray, corners: np.ndarray, *,
                      pad: int = 70) -> np.ndarray:
    """Tight crop around the portrait quad with ``pad`` px margin."""
    H, W = img.shape[:2]
    x0 = max(0, int(corners[:, 0].min()) - pad)
    y0 = max(0, int(corners[:, 1].min()) - pad)
    x1 = min(W, int(corners[:, 0].max()) + pad)
    y1 = min(H, int(corners[:, 1].max()) + pad)
    return img[y0:y1, x0:x1].copy()


def panel_header(img: np.ndarray, title: str, *, bg_h: int = 30) -> np.ndarray:
    """Stamp a dark title bar above the image."""
    H, W = img.shape[:2]
    bar = np.full((bg_h, W, 3), 28, dtype=np.uint8)
    cv2.putText(bar, title, (8, bg_h - 10),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main() -> int:
    """CLI entry point: run the Stage 3 refit on one portrait + assemble panel."""
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", required=True, type=Path,
                    help="Base teleop episode dir (with frame_0.png + "
                         "portrait_masks.pkl + portrait_corners.json)")
    p.add_argument("--pid", type=int, default=0,
                    help="Portrait index 0/1/2 (default: 0)")
    p.add_argument("--out", required=True, type=Path,
                    help="Output composite PNG path")
    args = p.parse_args()

    refine_fn = _load_refine_fn()

    frame = cv2.imread(str(args.episode / "frame_0.png"))
    if frame is None:
        raise SystemExit(f"cannot read {args.episode / 'frame_0.png'}")

    with open(args.episode / "portrait_masks.pkl", "rb") as f:
        cache = pickle.load(f)
    M_0 = cache["M_0_per_pid"][args.pid].astype(np.uint8)
    if M_0.max() <= 1:
        M_0 = (M_0 * 255).astype(np.uint8)

    corners_json = json.loads((args.episode / "portrait_corners.json").read_text())
    coarse_corners = order_tl_tr_br_bl(
        np.asarray(corners_json["portraits"][str(args.pid)]["0"]["corners"],
                    dtype=np.float32)
    )

    # Run the refit with debug_dir so the 5 PNGs get dumped.
    with tempfile.TemporaryDirectory(prefix="refine_dbg_") as td:
        dbg = Path(td)
        refined = refine_fn(frame, coarse_corners, sam_mask=M_0,
                              verbose=False, debug_dir=str(dbg))
        if refined is None:
            print(f"[WARN] refit failed for pid={args.pid}; falling back to coarse",
                  flush=True)
            refined = coarse_corners

        # Read back the 5 progressive PNGs in canonical order.
        steps = [
            ("01. SAM mask -> edge band (yellow) + coarse SAM quad (red)",
             dbg / "01_band_and_coarse_quad.png"),
            ("02. Canny edges (Otsu thresholds) restricted to band",
             dbg / "02_canny_edges_in_band.png"),
            ("03. HoughLinesP -> all candidate line segments",
             dbg / "03_hough_lines.png"),
            ("04. Outermost line per side (top/right/bottom/left)",
             dbg / "04_oriented_lines_and_selected_sides.png"),
            ("05. Refined quad (green, sub-pixel) vs coarse (red)",
             dbg / "05_final_refined_vs_coarse.png"),
        ]
        loaded: list[tuple[str, np.ndarray]] = []
        for caption, png in steps:
            if not png.is_file():
                print(f"[WARN] missing {png.name}; skipping", flush=True)
                continue
            img = cv2.imread(str(png))
            if img is None:
                print(f"[WARN] cannot read {png}; skipping", flush=True)
                continue
            loaded.append((caption, img))

    if not loaded:
        raise SystemExit("no debug PNGs produced; check refit run")

    # The refit's debug PNGs are full-frame (640x480) with a label bar
    # already drawn at the bottom. We crop them to the portrait region so
    # the panel actually shows the paper. Use the union of coarse + refined
    # corners so all 5 frames crop to the same region.
    union = np.vstack([coarse_corners, np.asarray(refined, dtype=np.float32)])
    pad = 60
    PANEL_W = 320

    # Crop each PNG to the portrait region + add a top caption bar.
    cells: list[np.ndarray] = []
    target_h = None
    for caption, img in loaded:
        crop = crop_to_portrait(img, union, pad=pad)
        h, w = crop.shape[:2]
        scale = PANEL_W / w
        new_h = int(round(h * scale))
        crop = cv2.resize(crop, (PANEL_W, new_h),
                           interpolation=cv2.INTER_LANCZOS4)
        panel = panel_header(crop, caption)
        cells.append(panel)
        target_h = panel.shape[0] if target_h is None else max(target_h, panel.shape[0])

    # Normalize all cells to the same height (pad the bottom with dark grey)
    norm_cells = []
    for c in cells:
        if c.shape[0] < target_h:
            extra = np.full((target_h - c.shape[0], c.shape[1], 3),
                              28, dtype=np.uint8)
            c = np.vstack([c, extra])
        norm_cells.append(c)

    # Layout: 3 panels in row 1, 2 centered in row 2.
    sep_h = 6
    sep_v = 6
    pad_color = 12
    row1 = norm_cells[:3]
    row2 = norm_cells[3:5] if len(norm_cells) >= 5 else norm_cells[3:]

    def hstack_with_sep(panels, sep_w=sep_v):
        out = [panels[0]]
        for p in panels[1:]:
            sep = np.full((p.shape[0], sep_w, 3), pad_color, dtype=np.uint8)
            out.extend([sep, p])
        return np.hstack(out)

    row1_img = hstack_with_sep(row1) if row1 else None
    row2_img = hstack_with_sep(row2) if row2 else None

    if row1_img is not None and row2_img is not None:
        # Pad row2 to row1's width, centered.
        gap = row1_img.shape[1] - row2_img.shape[1]
        if gap > 0:
            left = np.full((row2_img.shape[0], gap // 2, 3), pad_color, dtype=np.uint8)
            right = np.full((row2_img.shape[0], gap - gap // 2, 3), pad_color, dtype=np.uint8)
            row2_img = np.hstack([left, row2_img, right])
        elif gap < 0:
            row1_img = np.hstack([row1_img,
                                    np.full((row1_img.shape[0], -gap, 3),
                                             pad_color, dtype=np.uint8)])
        sep = np.full((sep_h, row1_img.shape[1], 3), pad_color, dtype=np.uint8)
        composite = np.vstack([row1_img, sep, row2_img])
    else:
        composite = row1_img if row1_img is not None else row2_img

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), composite)
    print(f"wrote {args.out} ({composite.shape[1]}x{composite.shape[0]})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
