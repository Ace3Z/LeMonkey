#!/usr/bin/env python3
"""Render a 3-row before/after panel showing every Stage 4 inpainting step.

For one chosen portrait of one episode, this script:

  Row 1.  Lanczos warp          (source photo  ->  warped to dst quad)
  Row 2.  Gaussian MTF blur     (sharp warp    ->  sigma=0.8 px blur)
  Row 3.  Alpha-feather paste   (blurred warp  ->  feathered composite,
                                 the deployed blend_mode="alpha_feather")

Each row is rendered as a left/right pair on the same target frame's pixel
canvas, so the inserted patch sits at its real location and the boundary
treatments are visible against the rest of the scene. Output is a single
6-panel PNG written under media/figures/aug/.

The Reinhard Lab color transfer and Poisson seamlessClone paths exist in
inpaint_video.py for ablation only; the deployed recipe is the 3-row
chain above (alpha-feather).

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


def alpha_paste(warped: np.ndarray, src_frame: np.ndarray,
                  mask: np.ndarray) -> np.ndarray:
    """Hard alpha paste with no feathering: every mask>0 pixel of
    src_frame is replaced by the corresponding warped pixel."""
    out = src_frame.copy()
    out[mask > 0] = warped[mask > 0]
    return out


def alpha_feather_blend(warped: np.ndarray, src_frame: np.ndarray,
                          mask: np.ndarray, *,
                          feather_sigma: float = 1.2) -> np.ndarray:
    """Replicate the inpaint_video.py 'alpha_feather' production blend.

    Mirrors inpaint_video.replace_portrait when blend_mode='alpha_feather'
    (the deployed default):
      1. Erode the mask by 1 px to drop one-pixel JPEG-halo artefacts.
      2. Gaussian-blur the binary mask for a soft inward transition.
      3. Clamp the feather to the un-eroded mask so alpha is exactly zero
         outside (otherwise the new photo would bleed onto adjacent
         gripper/can/hand pixels).
      4. Normalise the feather to [0, 1] and linearly blend.
    """
    erode_px = 1
    mask_eroded = cv2.erode(
        mask, np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
    )
    if mask_eroded.sum() == 0:
        return src_frame
    feather = cv2.GaussianBlur(
        mask_eroded.astype(np.float32), (0, 0), sigmaX=feather_sigma
    )
    feather = np.minimum(feather, mask.astype(np.float32))
    m = feather.max()
    if m > 1e-6:
        feather = feather / m
    feather = feather[:, :, None]
    out = (warped.astype(np.float32) * feather
            + src_frame.astype(np.float32) * (1 - feather))
    return np.clip(out, 0, 255).astype(np.uint8)


def quad_polygon_mask(H: int, W: int, dst_corners: np.ndarray) -> np.ndarray:
    """Filled quadrilateral mask from 4 corners (uint8, 0/255)."""
    poly = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(poly, dst_corners.astype(np.int32), 255)
    return poly


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
                    help="Output PNG path (3-row, 6-panel composite)")
    p.add_argument("--mtf-sigma", type=float, default=0.8,
                    help="Gaussian sigma for MTF blur (default 0.8 px)")
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
    # We show ONLY the steps that the deployed pipeline actually runs:
    #     Lanczos warp -> Gaussian MTF blur -> alpha-feather composite
    # (the optional Reinhard color transfer is OFF by default in production
    # because it bleaches on a white-table-dominated ring; the Poisson
    # NORMAL_CLONE blend is similarly an opt-in alternative. We don't
    # visualise either of them since neither is in the deployed recipe.)
    warped = warp_to_quad(new_photo, dst_corners, H, W)
    blurred = cv2.GaussianBlur(warped, (0, 0), sigmaX=args.mtf_sigma)

    # For the WARP and BLUR rows we want the new photo to fully replace the
    # original portrait, so we paste over the *dst_quad polygon* rather than
    # M_0. M_0 (the SAM mask) is typically smaller than the visible printed
    # portrait, so pasting only at M_0 lets the original face leak around the
    # edges of the swap. The polygon mask covers the full paper region.
    dst_poly = quad_polygon_mask(H, W, dst_corners)
    warp_paste = alpha_paste(warped, frame, dst_poly)
    blur_paste = alpha_paste(blurred, frame, dst_poly)

    # The FINAL composite row uses the deployed alpha-feather blend with
    # the actual M_0 mask, so the reader sees exactly what production
    # writes to disk (1-px erosion + Gaussian feather clamped to M_0).
    final_composite = alpha_feather_blend(blurred, frame, M_0)

    # 3. Build per-step before/after pairs ------------------------------------
    pairs = [
        ("Row 1.  Lanczos warp",
         "before: original frame",
         "after: new photo warped to dst quad",
         frame.copy(),                                     # before: untouched
         warp_paste),                                      # after: full-quad paste
        ("Row 2.  Gaussian MTF blur (sigma = 0.8 px)",
         "before: warped, full Lanczos sharpness",
         "after: + Gaussian sigma 0.8 (matches USB webcam MTF)",
         warp_paste,
         blur_paste),
        ("Row 3.  Alpha-feather composite (deployed blend)",
         "before: hard paste over dst quad (visible seam)",
         "after: alpha-feather at M_0 (1-px erosion + sigma feather)",
         blur_paste,
         final_composite),
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
