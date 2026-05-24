#!/usr/bin/env python3
"""Synthetic-data smoke test for replace_portrait().

Builds a fake camera frame with a known "portrait" region at known corners,
runs replace_portrait() with a known replacement photo, and verifies:

  - the output shape matches input
  - the mask region's mean color is close to the replacement's mean color
    (homography + MTF blur + alpha-feather preserved the new content)
  - the surrounding (non-mask) region is unchanged
  - no NaN / Inf in the output
  - the replacement preserves the new photo's identity-relevant content
    (mean RGB shifted toward the new photo, not the original)

This file is the regression gate that runs before any real data is touched.
Lives under eval_3/aug/tests/ so the pipeline scripts proper stay clean.

Usage:
    python tests/test_replace_portrait.py
    pytest tests/test_replace_portrait.py -v   (if pytest is installed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Make the parent dir importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# Import the function we're validating. The aug/ dir uses leading-digit
# filenames so we use importlib.
import importlib.util
spec = importlib.util.spec_from_file_location("inpaint_mod", HERE.parent / "stages" / "inpaint_video.py")
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
replace_portrait = _mod.replace_portrait
reinhard_lab = _mod.reinhard_lab


def make_synthetic_scene(
    H: int = 480, W: int = 640,
    portrait_corners: np.ndarray | None = None,
    background_bgr: tuple[int, int, int] = (200, 200, 200),
    portrait_color: tuple[int, int, int] = (50, 50, 200),  # red-ish in BGR
    add_noise: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (frame, mask, ground_truth_corners_4x2_float32).

    add_noise=True adds Gaussian sensor noise (σ ≈ 8) to the whole frame,
    which is realistic for overhead-cam at 640×480 and exercises Reinhard's
    std-matching path — without noise, the outer ring is uniform and
    Reinhard would clamp to identity (correct behaviour, but not
    representative of real data).
    """
    if portrait_corners is None:
        portrait_corners = np.array([
            [220, 140], [430, 130],   # TL, TR  (slight tilt)
            [440, 380], [210, 370],   # BR, BL
        ], dtype=np.float32)

    frame = np.full((H, W, 3), background_bgr, dtype=np.uint8)
    pts = portrait_corners.astype(np.int32)
    cv2.fillPoly(frame, [pts], portrait_color)

    if add_noise:
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 8.0, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)

    return frame, mask, portrait_corners


def make_replacement_photo(h: int = 400, w: int = 280, *, textured: bool = True) -> np.ndarray:
    """A small high-res 'photo' with a clear distinct pattern + color.
    Aspect ratio ~0.7 = A5.

    textured=True adds a high-frequency checker so unsharp-mask has signal
    to sharpen (otherwise USM on flat color regions is ~no-op).
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (180, 60, 60)            # blue-heavy BGR
    img[h//3:2*h//3, :] = (40, 220, 220)  # yellow-heavy BGR
    if textured:
        # 8×8 checkerboard at low contrast — gives Lanczos something to
        # bandlimit and unsharp something to recover.
        ys, xs = np.mgrid[0:h, 0:w]
        checker = ((ys // 4 + xs // 4) % 2).astype(np.uint8) * 30
        img = np.clip(img.astype(np.int16) + checker[:, :, None] - 15, 0, 255).astype(np.uint8)
    return img


def assert_close(actual, expected, tol, label: str):
    if not (abs(actual - expected) <= tol):
        raise AssertionError(f"{label}: |{actual:.3f} - {expected:.3f}| > {tol}")


def test_basic_replacement():
    """The mask region's mean color should shift toward the new photo's color.
    The non-mask region should be unchanged.
    """
    frame, mask, corners = make_synthetic_scene()
    photo = make_replacement_photo()

    out = replace_portrait(frame, mask, photo, corners)

    # Sanity: shape + dtype + range
    assert out.shape == frame.shape, f"shape mismatch {out.shape} vs {frame.shape}"
    assert out.dtype == np.uint8
    assert not np.isnan(out).any() and not np.isinf(out).any()

    # Non-mask region unchanged
    non_mask = (mask == 0)
    diff_outside = np.abs(out[non_mask].astype(np.int16) - frame[non_mask].astype(np.int16)).mean()
    if diff_outside >= 5.0:                              # allow tiny Poisson bleed
        raise AssertionError(f"non-mask region drifted: mean abs diff = {diff_outside:.2f}")

    # Mask region color shifted away from original portrait color toward photo's mean
    orig_color = np.array([50, 50, 200])         # original portrait = red-ish
    photo_mean = photo.reshape(-1, 3).mean(0)    # blue/yellow-heavy mix
    out_mask_mean = out[mask > 0].astype(np.float32).mean(0)
    dist_to_orig = np.linalg.norm(out_mask_mean - orig_color)
    dist_to_photo = np.linalg.norm(out_mask_mean - photo_mean)
    if dist_to_photo >= dist_to_orig:
        raise AssertionError(
            f"output mask region didn't move toward replacement photo:\n"
            f"  out_mask_mean = {out_mask_mean}\n"
            f"  orig_portrait = {orig_color}    (dist {dist_to_orig:.1f})\n"
            f"  photo_mean    = {photo_mean}    (dist {dist_to_photo:.1f})"
        )
    print(f"  ✓ test_basic_replacement: out_mask_mean={out_mask_mean.round(1).tolist()}, "
          f"dist_to_orig={dist_to_orig:.1f}, dist_to_photo={dist_to_photo:.1f}")


def test_idempotent_on_uniform_replacement():
    """If you replace a portrait with a photo that already matches the
    surrounding background, the output should be ~identical to passing
    the photo straight through (Poisson preserves source gradients)."""
    frame, mask, corners = make_synthetic_scene(
        background_bgr=(128, 128, 128),
        portrait_color=(50, 50, 50),
    )
    photo = np.full((400, 280, 3), 128, dtype=np.uint8)  # uniform gray
    out = replace_portrait(frame, mask, photo, corners)

    out_inside = out[mask > 0].astype(np.float32).mean(0)
    target = np.array([128.0, 128.0, 128.0])
    dist = np.linalg.norm(out_inside - target)
    if dist >= 12.0:
        raise AssertionError(
            f"uniform-photo replacement deviated from target gray by {dist:.1f}\n"
            f"  out_inside = {out_inside}"
        )
    print(f"  ✓ test_idempotent_on_uniform_replacement: out_inside={out_inside.round(1).tolist()}, dist={dist:.1f}")


def test_no_op_on_zero_mask():
    """If the mask is empty, the output should equal the input."""
    frame, _, corners = make_synthetic_scene()
    empty_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    photo = make_replacement_photo()
    out = replace_portrait(frame, empty_mask, photo, corners)
    if not np.array_equal(out, frame):
        raise AssertionError("zero-mask should be a no-op")
    print(f"  ✓ test_no_op_on_zero_mask")


def test_reinhard_preserves_when_ref_eq_src():
    """When ref's sample pixels match src's pixel statistics, Reinhard is identity-ish."""
    src = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    # Reference is the same image; sample mask = entire image
    sample = np.ones(src.shape[:2], dtype=np.uint8)
    out = reinhard_lab(src, src, sample)
    # mean/std preserved → output ≈ input
    diff = np.abs(out.astype(np.int16) - src.astype(np.int16)).mean()
    if diff >= 3.0:
        raise AssertionError(f"reinhard_lab(x, x, full) should be ~identity; mean abs diff {diff:.2f}")
    print(f"  ✓ test_reinhard_preserves_when_ref_eq_src: mean abs diff = {diff:.3f}")


def test_unsharp_increases_high_freq():
    """The optional unsharp pass should sharpen edges (increase high-freq energy)."""
    frame, mask, corners = make_synthetic_scene()
    photo = make_replacement_photo()
    soft = replace_portrait(frame, mask, photo, corners, apply_unsharp=False)
    sharp = replace_portrait(frame, mask, photo, corners, apply_unsharp=True)
    # Compare Laplacian-magnitude of the mask region
    soft_lap = np.abs(cv2.Laplacian(soft, cv2.CV_32F))[mask > 0].mean()
    sharp_lap = np.abs(cv2.Laplacian(sharp, cv2.CV_32F))[mask > 0].mean()
    if sharp_lap <= soft_lap:
        raise AssertionError(
            f"apply_unsharp=True did not increase high-freq energy: "
            f"sharp={sharp_lap:.2f} ≤ soft={soft_lap:.2f}"
        )
    print(f"  ✓ test_unsharp_increases_high_freq: sharp={sharp_lap:.2f} > soft={soft_lap:.2f}")


def main() -> int:
    print(f"Synthetic smoke test for replace_portrait()")
    print(f"  module: {_mod.__file__}")
    print()
    tests = [
        test_basic_replacement,
        test_idempotent_on_uniform_replacement,
        test_no_op_on_zero_mask,
        test_reinhard_preserves_when_ref_eq_src,
        test_unsharp_increases_high_freq,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ {fn.__name__}: {e}")
            failures += 1
    print()
    if failures:
        print(f"FAILED: {failures}/{len(tests)} tests")
        return 1
    print(f"ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
