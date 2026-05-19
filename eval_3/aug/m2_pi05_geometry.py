"""Pi0.5 / PaliGemma geometry adapter for the M2 supervision builder.

Differences vs SmolVLA (which our existing `m2_alignment._resize_with_pad_box`
encodes):

|                          | SmolVLA              | Pi0.5 / PaliGemma           |
|--------------------------|----------------------|-----------------------------|
| Target resolution        | 512x512              | 224x224                     |
| Padding scheme           | left+top (asymmetric)| CENTER (symmetric)          |
| Vision patches           | 8x8 = 64 (pixel-shuffle 32x32→8x8) | 16x16 = 256 (patch_size=14)   |
| Source                   | modeling_smolvla.py:134 | modeling_pi05.py:204-216 |

This module exposes:
- `resize_with_pad_box_pi05(bbox_xyxy)` — map a 480x640 bbox into 224x224
  with symmetric pad, returning (x1, y1, x2, y2) in target pixels.
- `PI05_PATCH_GRID = 16` and `NUM_PI05_PATCHES = 256`.
- `bbox_to_patch_mask_pi05(bbox_target)` — quantise a 224x224 bbox to the
  16x16 patch grid, return a (256,) bool mask.

All three are used by `M2SupervisionBuilder` via dependency injection so the
existing builder can target either SmolVLA or Pi0.5 without changes.
"""
from __future__ import annotations

import numpy as np

# PaliGemma defaults from third_party/lerobot/.../configuration_pi05.py:26
PI05_IMG_HW = 224
PI05_PATCH_SIZE = 14            # SigLIP default for PaliGemma
PI05_PATCH_GRID = PI05_IMG_HW // PI05_PATCH_SIZE   # 16
NUM_PI05_PATCHES = PI05_PATCH_GRID * PI05_PATCH_GRID  # 256

# Source image (from the SO-101 camera1 stream, before resize).
RAW_IMG_H = 480
RAW_IMG_W = 640


def resize_with_pad_box_pi05(
    bbox_xyxy,
    orig_hw=(RAW_IMG_H, RAW_IMG_W),
    target_hw=(PI05_IMG_HW, PI05_IMG_HW),
):
    """Map a bbox in (640x480) pixel coords → (224x224) Pi0.5 input coords.

    Mirrors `resize_with_pad_torch` at modeling_pi05.py:204-216:
        ratio = max(W/tw, H/th)
        rh, rw = int(H/ratio), int(W/ratio)
        pad_h0, rem_h = divmod(th - rh, 2)
        pad_h1 = pad_h0 + rem_h
        pad_w0, rem_w = divmod(tw - rw, 2)
        pad_w1 = pad_w0 + rem_w

    For 480x640 → 224x224 specifically: ratio = 640/224 ≈ 2.857;
    rh, rw = 168, 224; pad_h0 = pad_h1 = 28 (no width pad).
    """
    x1, y1, x2, y2 = bbox_xyxy
    oh, ow = orig_hw
    th, tw = target_hw

    ratio = max(ow / tw, oh / th)
    rh = int(oh / ratio)
    rw = int(ow / ratio)

    pad_h0, rem_h = divmod(th - rh, 2)
    pad_w0, rem_w = divmod(tw - rw, 2)

    return (
        x1 / ratio + pad_w0,
        y1 / ratio + pad_h0,
        x2 / ratio + pad_w0,
        y2 / ratio + pad_h0,
    )


def bbox_to_patch_mask_pi05(bbox_xyxy_224) -> np.ndarray:
    """Quantise a 224x224 bbox to the 16x16 patch grid; return (256,) bool.

    Each patch is `PI05_PATCH_SIZE = 14` px wide. A patch is marked True iff
    the bbox overlaps it (i.e. integer-floor of x1 ≤ patch_col ≤ integer-ceil
    of x2/patch_size). Out-of-bounds is clipped to the grid.
    """
    x1, y1, x2, y2 = bbox_xyxy_224
    px1 = max(0, int(x1 // PI05_PATCH_SIZE))
    py1 = max(0, int(y1 // PI05_PATCH_SIZE))
    # Inclusive upper-bound: ceil(x2 / patch) gives the first patch fully
    # past the bbox; we use it as an exclusive end below.
    px2 = min(PI05_PATCH_GRID, int(np.ceil(x2 / PI05_PATCH_SIZE)))
    py2 = min(PI05_PATCH_GRID, int(np.ceil(y2 / PI05_PATCH_SIZE)))

    mask = np.zeros((PI05_PATCH_GRID, PI05_PATCH_GRID), dtype=bool)
    if px2 > px1 and py2 > py1:
        mask[py1:py2, px1:px2] = True
    return mask.flatten()  # (256,)
