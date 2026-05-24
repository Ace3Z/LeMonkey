#!/usr/bin/env python3
"""Render a 4-row before/after panel showing every Stage 4 inpainting step.

For one chosen portrait of one episode, this script:

  Row 1.  Lanczos warp          (source photo  ->  warped to dst quad)
  Row 2.  Gaussian MTF blur     (sharp warp    ->  sigma=0.8 px blur)
  Row 3.  Reinhard Lab transfer (blurred warp  ->  + ring-sampled color match)
  Row 4.  Poisson seamlessClone (naive paste   ->  NORMAL_CLONE composite)

Each row is rendered as a left/right pair on the same target frame's pixel
canvas, so the inserted patch sits at its real location and the boundary
treatments are visible against the rest of the scene. Output is a single
8-panel PNG written under media/figures/aug/.

Usage:
    python eval_3/aug/dbg/stage4_steps_panel.py \\
        --episode datasets/eval3/quick_lecun_LSO_ep01_20260511_205000 \\
        --pid 0 \\
        --new-photo datasets/eval3_celebs/scraped/taylor_swift/<photo.jpg> \\
        --out media/figures/aug/stage4_step_by_step.png

Reproducible: any episode dir under eval_3/ with portrait_masks.pkl +
portrait_corners.json + a photo from the scraped bank works.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


def panel_header(img: np.ndarray, title: str, *, bg_h: int = 32) -> np.ndarray:
    """Stamp a title bar above the image."""
    H, W = img.shape[:2]
    bar = np.full((bg_h, W, 3), 28, dtype=np.uint8)  # dark grey
    cv2.putText(bar, title, (10, bg_h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def order_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Reorder 4 points as top-left, top-right, bottom-right, bottom-left.

    Robust to rotated rhombuses where two corners can share the same y-x
    value (the naive ``argmax(y-x)`` then returns the same index as the
    corresponding ``argmax(x+y)``, producing a degenerate quad). We pick
    TL = smallest x+y and BR = largest x+y, then for the remaining two
    decide TR vs BL by x-coordinate (TR has the larger x).
    """
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    i_tl = int(np.argmin(s))
    i_br = int(np.argmax(s))
    others = [i for i in range(4) if i not in (i_tl, i_br)]
    # The remaining two are TR and BL; TR has the larger x.
    a, b = others
    if pts[a, 0] >= pts[b, 0]:
        i_tr, i_bl = a, b
    else:
        i_tr, i_bl = b, a
    return np.array([pts[i_tl], pts[i_tr], pts[i_br], pts[i_bl]],
                     dtype=np.float32)


def warp_to_quad(new_photo: np.ndarray, dst_corners: np.ndarray,
                  H: int, W: int) -> np.ndarray:
    """Lanczos warp the HD photo to the destination quad in the frame canvas."""
    h, w = new_photo.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                    dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst_corners.astype(np.float32))
    return cv2.warpPerspective(new_photo, M, (W, H), flags=cv2.INTER_LANCZOS4)


def reinhard_lab(src_bgr: np.ndarray, ref_bgr: np.ndarray,
                  sample_mask: np.ndarray, *, std_clip: tuple[float, float] = (0.3, 2.0)
                  ) -> np.ndarray:
    """Match (L,a,b) mean+std of src_bgr to those of ref_bgr sampled at
    sample_mask>0. Per-channel std ratio clamped to ``std_clip``."""
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = ref_lab[sample_mask > 0]
    if len(ring) < 10:
        return src_bgr
    src_mean = src_lab.mean((0, 1)); src_std = src_lab.std((0, 1)) + 1e-6
    ring_mean = ring.mean(0); ring_std = ring.std(0) + 1e-6
    ratio = np.clip(ring_std / src_std, std_clip[0], std_clip[1])
    out_lab = (src_lab - src_mean) * ratio + ring_mean
    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def alpha_paste(warped: np.ndarray, src_frame: np.ndarray,
                  mask: np.ndarray) -> np.ndarray:
    """Hard alpha paste — no feathering. Shows the seam crisply for the
    'before' panel of the Poisson row."""
    out = src_frame.copy()
    out[mask > 0] = warped[mask > 0]
    return out


def poisson_clone(warped: np.ndarray, src_frame: np.ndarray,
                    mask: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Replicate the inpaint_video.py 'poisson_normal' blend mode."""
    ring_pix = src_frame[ring > 0]
    dst = src_frame.copy()
    if len(ring_pix) >= 10:
        ring_mean = ring_pix.astype(np.float32).mean(0).astype(np.uint8)
        dst[mask > 0] = ring_mean
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return src_frame
    cx = int(round((xs.min() + xs.max()) / 2))
    cy = int(round((ys.min() + ys.max()) / 2))
    return cv2.seamlessClone(warped, dst, mask, (cx, cy), cv2.NORMAL_CLONE)


def crop_to_portrait(img: np.ndarray, corners: np.ndarray,
                      pad: int = 60) -> np.ndarray:
    """Return a tight crop around the portrait quad with `pad` px margin."""
    H, W = img.shape[:2]
    x0 = max(0, int(corners[:, 0].min()) - pad)
    y0 = max(0, int(corners[:, 1].min()) - pad)
    x1 = min(W, int(corners[:, 0].max()) + pad)
    y1 = min(H, int(corners[:, 1].max()) + pad)
    return img[y0:y1, x0:x1].copy()


def main() -> int:
    """CLI entry: render the 4-row before/after panel and write the PNG."""
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", required=True, type=Path,
                    help="Base teleop episode dir (must have portrait_masks.pkl "
                         "+ portrait_corners.json + frame_0.png)")
    p.add_argument("--pid", type=int, default=0,
                    help="Portrait index to swap (0/1/2; defaults to 0)")
    p.add_argument("--new-photo", required=True, type=Path,
                    help="HD photo to inpaint into the chosen portrait quad")
    p.add_argument("--out", required=True, type=Path,
                    help="Output PNG path (8-panel composite)")
    p.add_argument("--mtf-sigma", type=float, default=0.8,
                    help="Gaussian sigma for MTF blur (default 0.8 px)")
    p.add_argument("--ring-dilate-px", type=int, default=11,
                    help="Dilation for Reinhard ring sampling (default 11)")
    args = p.parse_args()

    # 1. Load inputs ----------------------------------------------------------
    frame = cv2.imread(str(args.episode / "frame_0.png"))
    if frame is None:
        raise SystemExit(f"cannot read {args.episode / 'frame_0.png'}")
    H, W = frame.shape[:2]
    new_photo = cv2.imread(str(args.new_photo))
    if new_photo is None:
        raise SystemExit(f"cannot read {args.new_photo}")

    with open(args.episode / "portrait_masks.pkl", "rb") as f:
        cache = pickle.load(f)
    M_0 = cache["M_0_per_pid"][args.pid].astype(np.uint8)
    if M_0.max() <= 1:
        M_0 = (M_0 * 255).astype(np.uint8)
    corners = json.loads((args.episode / "portrait_corners.json").read_text())
    dst_corners = order_tl_tr_br_bl(
        np.asarray(corners["portraits"][str(args.pid)]["0"]["corners"],
                    dtype=np.float32)
    )

    # 2. Step through the inpaint pipeline ------------------------------------
    warped = warp_to_quad(new_photo, dst_corners, H, W)
    blurred = cv2.GaussianBlur(warped, (0, 0), sigmaX=args.mtf_sigma)
    ring_dilated = cv2.dilate(M_0, np.ones((args.ring_dilate_px,
                                                 args.ring_dilate_px), np.uint8))
    ring = cv2.subtract(ring_dilated, M_0)
    color_matched = reinhard_lab(blurred, frame, sample_mask=ring)
    # Row 4 demonstrates the boundary-blend step. Production defaults to
    # alpha_feather with Reinhard OFF (apply_reinhard=False, see
    # inpaint_video.py:151 + the line 254-262 comment): Reinhard tends to
    # bleach on a white-table-dominated ring, so we use the BLURRED warp
    # (no color match) as the Poisson input here. This is the recipe a
    # reader sees actually deployed.
    naive_pasted = alpha_paste(blurred, frame, M_0)
    poisson = poisson_clone(blurred, frame, M_0, ring)

    # 3. Build per-step before/after pairs ------------------------------------
    # Row 1 "before" is the ORIGINAL frame (with the existing portrait still
    # there); every other "before" is the cumulative result of prior steps,
    # alpha-pasted at the portrait location so the surrounding scene is visible.
    # Row 4 shows the full composite of the final two blend variants.
    pairs = [
        ("Row 1.  Lanczos warp",
         "before: original frame",
         "after: warped to dst quad",
         frame.copy(),                                     # before: untouched frame
         alpha_paste(warped, frame, M_0)),                 # after: warped paste, no blur/color
        ("Row 2.  Gaussian MTF blur (sigma = 0.8 px)",
         "before: warped, full sharpness",
         "after: + Gaussian sigma 0.8",
         alpha_paste(warped, frame, M_0),
         alpha_paste(blurred, frame, M_0)),
        ("Row 3.  Reinhard Lab color transfer  (off by default in production)",
         "before: warped + blurred",
         "after: + ring-sampled color match",
         alpha_paste(blurred, frame, M_0),
         alpha_paste(color_matched, frame, M_0)),
        ("Row 4.  Boundary blend",
         "before: hard alpha paste (seam visible)",
         "after: Poisson NORMAL_CLONE",
         naive_pasted,
         poisson),
    ]

    # 4. Crop to the portrait region + assemble the 4x2 grid ------------------
    # Each panel is rendered at a fixed width so the final composite is
    # readable when embedded in a README.
    PANEL_W = 360
    rows = []
    for title, left_caption, right_caption, before, after in pairs:
        before_crop = crop_to_portrait(before, dst_corners, pad=70)
        after_crop = crop_to_portrait(after, dst_corners, pad=70)
        # Make them the same shape, then resize to a fixed panel width.
        h = min(before_crop.shape[0], after_crop.shape[0])
        w = min(before_crop.shape[1], after_crop.shape[1])
        before_crop = cv2.resize(before_crop, (w, h))
        after_crop = cv2.resize(after_crop, (w, h))
        scale = PANEL_W / w
        new_h = int(round(h * scale))
        before_crop = cv2.resize(before_crop, (PANEL_W, new_h), interpolation=cv2.INTER_LANCZOS4)
        after_crop = cv2.resize(after_crop, (PANEL_W, new_h), interpolation=cv2.INTER_LANCZOS4)
        before_panel = panel_header(before_crop, left_caption)
        after_panel = panel_header(after_crop, right_caption)
        row = np.hstack([before_panel, after_panel])
        sec_h = 38
        sec = np.full((sec_h, row.shape[1], 3), 12, dtype=np.uint8)
        cv2.putText(sec, title, (14, sec_h - 13),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 250, 250), 1,
                     cv2.LINE_AA)
        rows.append(np.vstack([sec, row]))

    # Pad all rows to the widest, then stack
    target_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < target_w:
            pad = np.full((r.shape[0], target_w - r.shape[1], 3), 0,
                            dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)
        # Thin separator between rows
        sep = np.full((4, target_w, 3), 50, dtype=np.uint8)
        padded.append(sep)
    composite = np.vstack(padded[:-1])  # drop trailing separator

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), composite)
    print(f"wrote {args.out} ({composite.shape[1]}x{composite.shape[0]})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
