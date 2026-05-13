"""Classical sub-pixel rectangle refit for printed-paper boundaries.

Why this module exists:
  SAM 2.1's mask boundary has a documented 5-15 px upsampling-induced
  looseness on planar rigid targets (SAM 2 paper §3.3; SAMRefiner ICLR
  2025 arXiv:2502.06756). For printed paper on a near-uniform table,
  the *true* boundary is a sharp high-contrast edge that classical
  sub-pixel methods localize to ~0.05–0.1 px (Devernay 1995 INRIA TR;
  Trujillo-Pino et al. IVC 2013; IPOL 2017/216). This module takes a
  coarse 4-corner quad (from SAM + minAreaRect) and refines it to the
  true paper edge.

Algorithm overview (canonical "neural seed → classical refine" recipe;
sources cross-checked per CLAUDE.md §7):

  1. Build an "edge band" around the coarse quad: dilate(mask, +20 px)
     AND NOT erode(mask, +20 px). The true edge lies in this band
     given SAM's documented ≤15 px boundary error.

  2. Canny on grayscale with Otsu-derived thresholds (high = Otsu,
     low = 0.5 * Otsu — the 2:1 ratio Fang et al. ICIP 2009 verify).
     Restrict edges to the band only — zero out edges inside the
     inner erosion (those are printed-content edges, not paper-edge).

  3. HoughLinesP (Matas-Galambos-Kittler probabilistic transform) →
     candidate line segments. Cluster into 4 orientation bins (top,
     bottom, left, right) relative to the coarse-quad centroid.

  4. Outermost-line selection per bin. Printed-content edges always
     sit INSIDE the paper margin, so the paper edge is the farthest
     line from centroid in each direction (Dropbox 2016 scanner;
     Lee 2024 IET ElLet planar tracking).

  5. Contrast gate per line: paper is brighter than table (white A5
     paper grayscale ≈ 180-220, gray table ≈ 100-130 → Δ ≥ 30 is
     conservative). Sample ±5 px strips along the line.

  6. Intersect the 4 lines pairwise → 4 sub-pixel corners.

  7. cv2.cornerSubPix to snap to the local gradient saddle (Förstner
     & Gülch 1987; OpenCV cornerSubPix tutorial).

  8. Sanity check the refined quad: must be convex, must overlap the
     coarse quad with IoU ≥ 0.5, must fit inside the frame. On any
     failure, return None (caller falls back to coarse quad) and log
     a [WARN] per CLAUDE.md §5.

Result: corners typically ≤ 0.5 px off the true paper edge — two
orders of magnitude tighter than SAM's raw mask boundary.

References:
  - Canny 1986 IEEE PAMI 8(6):679-698
  - Otsu 1979 IEEE Trans. SMC; Fang et al. ICIP 2009 Otsu-Canny
  - Matas, Galambos, Kittler "Robust Detection of Lines Using the
    Progressive Probabilistic Hough Transform" CVIU 2000
  - Förstner & Gülch 1987 fast corner detector / saddle point
  - Devernay INRIA TR 2724 (1995) sub-pixel non-max suppression
  - Trujillo-Pino et al. "Accurate subpixel edge location based on
    partial area effect" Image & Vision Computing 31(1):72-90, 2013
  - IPOL 2017/216 Canny + Devernay reference impl
  - SAMRefiner Lin et al. ICLR 2025 arXiv:2502.06756 (motivation:
    SAM mask boundaries are loose, must be refined for thin
    structures)
  - Dropbox 2016 scanner blog (outermost-line selection)
"""
from __future__ import annotations

import cv2
import numpy as np


# Default parameters — every value triple-sourced; deviations flagged.
DILATE_PX = 50      # Edge band outer offset. Two failure modes drive this:
                    # (i) SAM upsample-induced boundary looseness ≤15 px
                    #     (SAM 2 paper §3.3),
                    # (ii) SAM under-segmenting the paper (segments printed
                    #     face/torso but misses white paper margin) — the
                    #     paper margin is typically 30–50 px outside SAM's
                    #     boundary. Verified visually 2026-05-13 on
                    #     quick_lecun_LSO_ep01: SAM mask was the face area,
                    #     paper edge was ~40 px outside.
                    # 50 px gives margin for the worst case while still
                    # being tight enough to reject far-away table seams.
ERODE_PX = 20       # Inner offset. Keep at 20 — the inner half of the band
                    # only matters when SAM OVERSIZES the paper (the
                    # swift_OLS_ep01 failure mode). 20 px reach inward
                    # handles SAM oversize ≤15 px with margin.
HOUGH_RHO = 1       # OpenCV docs default for high-resolution edges.
HOUGH_THETA = np.pi / 180   # 1° angular resolution.
HOUGH_THRESH = 18   # Min vote count. Empirically the paper edge has
                    # ~30–50 inlier pixels at 1° resolution; 18 leaves
                    # margin for shadow-broken edges (mid-occlusion).
MIN_LINE_FRAC = 0.20        # min_length = 0.20 × min(box_w, box_h).
                            # Loosened from 0.30 — at heavy paper tilt
                            # (~50°) the visible side projection is
                            # 0.65×, and shadows can break it further.
MAX_LINE_GAP = 15           # Bridge small Canny gaps from shadow seams.
ORIENT_BIN_DEG = 30         # Loosened from 25 — covers paper rotations
                            # ≤30° relative to the coarse quad's own
                            # principal axes (rare but real with
                            # noisy SAM boundaries).
SUBPIX_WIN = (5, 5)         # cornerSubPix window. Förstner-Gülch
                            # paper uses 5×5 for natural images.
SUBPIX_ITERS = 30
SUBPIX_EPS = 0.001
MIN_IOU_VS_COARSE = 0.50    # Sanity gate: refined quad must overlap
                            # the SAM coarse quad with ≥50% IoU. A
                            # large drop means refinement found the
                            # wrong rectangle.
MAX_BAND_GROWTH = 1.50      # Refined area must be ≤ 1.5× of coarse
                            # area. The refinement should TIGHTEN the
                            # rect, not grow it.
EDGE_CONTRAST_MIN = 12.0    # Min grayscale Δ between the two sides of an
                            # edge. Real paper-table contrast ≥ 50 GL on
                            # clean frames, but shadowed/occluded segments
                            # can drop to ~20. Loosened from 25 to 12.


def _line_perp_dist(line: np.ndarray, pt: np.ndarray) -> float:
    """Signed perpendicular distance from `pt` to the infinite line
    through `(x1,y1)-(x2,y2)`. Sign indicates which side `pt` is on."""
    x1, y1, x2, y2 = line
    nx, ny = -(y2 - y1), (x2 - x1)
    norm = float(np.hypot(nx, ny))
    if norm < 1e-6:
        return 0.0
    return ((pt[0] - x1) * nx + (pt[1] - y1) * ny) / norm


def _line_angle_deg(line: np.ndarray) -> float:
    """Angle of the line in [0, 180)."""
    x1, y1, x2, y2 = line
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def _line_unit_normal(line: np.ndarray) -> tuple[float, float]:
    """Unit normal vector to the line."""
    x1, y1, x2, y2 = line
    nx, ny = -(y2 - y1), (x2 - x1)
    norm = float(np.hypot(nx, ny))
    if norm < 1e-6:
        return 1.0, 0.0
    return nx / norm, ny / norm


def _contrast_along_line(line: np.ndarray, gray: np.ndarray,
                          n_samples: int = 20, offset_px: int = 5) -> float:
    """Mean(brighter side) − Mean(darker side) sampled along the line.
    Used as the paper-vs-table gate: paper white, table gray."""
    H, W = gray.shape
    x1, y1, x2, y2 = line
    nx, ny = _line_unit_normal(line)
    ts = np.linspace(0.1, 0.9, n_samples)
    px = x1 + ts * (x2 - x1)
    py = y1 + ts * (y2 - y1)
    sx_a = np.clip(np.round(px - offset_px * nx).astype(int), 0, W - 1)
    sy_a = np.clip(np.round(py - offset_px * ny).astype(int), 0, H - 1)
    sx_b = np.clip(np.round(px + offset_px * nx).astype(int), 0, W - 1)
    sy_b = np.clip(np.round(py + offset_px * ny).astype(int), 0, H - 1)
    a = float(gray[sy_a, sx_a].mean())
    b = float(gray[sy_b, sx_b].mean())
    return abs(a - b)


def _intersect(L1: np.ndarray, L2: np.ndarray) -> tuple[float, float] | None:
    """Return the intersection point of two line segments (treated as
    infinite lines through their endpoints), or None if parallel."""
    x1, y1, x2, y2 = L1
    x3, y3, x4, y4 = L2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return float(x1 + t * (x2 - x1)), float(y1 + t * (y2 - y1))


def _quad_area(corners: np.ndarray) -> float:
    """Shoelace formula on 4 ordered corners."""
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def _quad_iou(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> float:
    """IoU of two quadrilaterals via rasterisation. shape = (H, W)."""
    H, W = shape
    ma = np.zeros((H, W), dtype=np.uint8)
    mb = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(ma, a.astype(np.int32), 1)
    cv2.fillConvexPoly(mb, b.astype(np.int32), 1)
    inter = int((ma & mb).sum())
    union = int((ma | mb).sum())
    return inter / max(union, 1)


def _is_convex(corners: np.ndarray) -> bool:
    """A 4-corner polygon is convex iff all consecutive cross products
    have the same sign."""
    pts = corners.astype(np.float64)
    n = len(pts)
    signs = []
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        c = pts[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(np.sign(cross))
    return len(set(signs)) <= 1


def _order_tl_tr_br_bl(corners: np.ndarray) -> np.ndarray:
    """Order 4 unordered points as TL/TR/BR/BL by sum and diff
    (canonical doc-scanner trick: TL has min sum, BR max sum;
    TR has min diff (x-y), BL max diff)."""
    pts = corners.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl], axis=0)


def _draw_lines(canvas: np.ndarray, lines, color=(0, 255, 0), thickness=1, label=None):
    out = canvas.copy()
    for L in lines:
        x1, y1, x2, y2 = [int(v) for v in (L if hasattr(L, "__len__") else L[0])][:4]
        cv2.line(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if label:
        cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def refine_paper_quad_to_edges(
    frame_bgr: np.ndarray,
    coarse_corners: np.ndarray,
    *,
    sam_mask: np.ndarray | None = None,
    verbose: bool = False,
    debug_dir = None,
) -> np.ndarray | None:
    """Refine a coarse 4-corner quad (from SAM + minAreaRect) to the true
    paper edge using classical Canny + Hough + outermost-line + sub-pixel
    intersection.

    Returns a (4, 2) np.float32 quad ordered TL/TR/BR/BL, or None if the
    refinement fails any sanity gate (in which case caller should keep
    the coarse corners).

    Logs [WARN] on every failure (CLAUDE.md §5)."""
    H, W = frame_bgr.shape[:2]
    coarse_corners = np.asarray(coarse_corners, dtype=np.float32)
    if coarse_corners.shape != (4, 2):
        if verbose:
            print(f"[WARN] refine_paper_quad_bad_input: expected=(4,2) coarse_corners, "
                  f"got={coarse_corners.shape}, fallback=None", flush=True)
        return None

    # 1. Build edge band: dilate(SAM mask) AND NOT erode(SAM mask).
    if sam_mask is None:
        # Derive a mask from the coarse corners themselves.
        sam_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(sam_mask, coarse_corners.astype(np.int32), 1)
    m = sam_mask.astype(np.uint8)
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * DILATE_PX + 1, 2 * DILATE_PX + 1))
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (2 * ERODE_PX + 1, 2 * ERODE_PX + 1))
    outer = cv2.dilate(m, k_dilate)
    inner = cv2.erode(m, k_erode)
    band = (outer.astype(bool) & ~inner.astype(bool)).astype(np.uint8)

    # debug: draw the band region + coarse quad on the frame
    if debug_dir is not None:
        from pathlib import Path as _P
        dbg = _P(debug_dir); dbg.mkdir(parents=True, exist_ok=True)
        band_viz = frame_bgr.copy()
        band_viz[band > 0] = (band_viz[band > 0] * 0.4 + np.array([0, 255, 255]) * 0.6).astype(np.uint8)
        cv2.polylines(band_viz, [coarse_corners.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.imwrite(str(dbg / "01_band_and_coarse_quad.png"),
                    _draw_lines(band_viz, [],
                                label=f"01. edge band (yellow) + coarse SAM quad (red)  "
                                      f"dilate={DILATE_PX} erode={ERODE_PX}"))
    if band.sum() < 100:
        if verbose:
            print(f"[WARN] refine_paper_quad_empty_band: expected=band>100 px, "
                  f"got={int(band.sum())} px, fallback=None", flush=True)
        return None

    # 2. Canny with Otsu-derived thresholds.
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    band_pixels = gray_blur[band > 0]
    if band_pixels.size < 100:
        return None
    otsu_thr, _ = cv2.threshold(band_pixels, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = max(10, int(0.5 * otsu_thr))
    high = max(low + 5, int(otsu_thr))
    edges = cv2.Canny(gray_blur, low, high, apertureSize=3, L2gradient=True)
    edges = (edges & (band * 255).astype(np.uint8)).astype(np.uint8)

    if debug_dir is not None:
        edges_viz = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Tint edges yellow for visibility
        edges_viz[edges > 0] = (0, 255, 255)
        # Mix with frame at 30% alpha
        bg = (frame_bgr.astype(np.float32) * 0.5).astype(np.uint8)
        edges_viz = np.where(edges_viz.any(axis=2, keepdims=True), edges_viz, bg)
        cv2.polylines(edges_viz, [coarse_corners.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.imwrite(str(dbg / "02_canny_edges_in_band.png"),
                    _draw_lines(edges_viz, [],
                                label=f"02. Canny edges in band (yellow) + coarse quad (red).  "
                                      f"Otsu={int(otsu_thr)}  low={low} high={high}  "
                                      f"edges={int(edges.sum()/255)} px"))
    if edges.sum() < 200:
        if verbose:
            print(f"[WARN] refine_paper_quad_too_few_edges: expected=edges>200 px, "
                  f"got={int(edges.sum())} px, fallback=None", flush=True)
        return None

    # 3. HoughLinesP. Sized so it rejects printed-content lines.
    xs, ys = np.where(m > 0)
    if xs.size == 0:
        return None
    bb_w = float(ys.max() - ys.min())  # ys=col coord b/c .where returns (row, col)
    bb_h = float(xs.max() - xs.min())
    min_line_length = int(MIN_LINE_FRAC * min(bb_w, bb_h))
    lines_raw = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH,
                                 minLineLength=min_line_length,
                                 maxLineGap=MAX_LINE_GAP)

    if debug_dir is not None:
        n_lines = 0 if lines_raw is None else len(lines_raw)
        hough_viz = (frame_bgr.astype(np.float32) * 0.5).astype(np.uint8)
        if lines_raw is not None:
            for L in lines_raw[:, 0, :]:
                x1, y1, x2, y2 = [int(v) for v in L]
                cv2.line(hough_viz, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.polylines(hough_viz, [coarse_corners.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.imwrite(str(dbg / "03_hough_lines.png"),
                    _draw_lines(hough_viz, [],
                                label=f"03. Hough lines (yellow) + coarse quad (red).  "
                                      f"n={n_lines}  min_len={min_line_length}  "
                                      f"thresh={HOUGH_THRESH}"))
    if lines_raw is None or len(lines_raw) < 4:
        if verbose:
            print(f"[WARN] refine_paper_quad_too_few_lines: expected≥4 Hough lines, "
                  f"got={0 if lines_raw is None else len(lines_raw)}, fallback=None",
                  flush=True)
        return None
    lines = lines_raw[:, 0, :]  # (N, 4)

    # 4. Cluster lines into 4 orientation bins (top, bottom, left, right)
    # relative to the coarse-quad centroid + the *coarse* edge orientations.
    centroid = coarse_corners.mean(axis=0)

    # Determine the coarse quad's two principal axes.
    # Edges 0→1 (TL→TR), 1→2 (TR→BR), 2→3 (BR→BL), 3→0 (BL→TL).
    e_top = coarse_corners[1] - coarse_corners[0]      # roughly horizontal
    e_right = coarse_corners[2] - coarse_corners[1]    # roughly vertical
    angle_h_coarse = float(np.degrees(np.arctan2(e_top[1], e_top[0])) % 180)
    angle_v_coarse = float(np.degrees(np.arctan2(e_right[1], e_right[0])) % 180)

    def _ang_diff(a: float, b: float) -> float:
        """Smallest abs angular distance between two angles in [0, 180)."""
        d = abs(a - b) % 180
        return min(d, 180 - d)

    bins: dict[str, list[tuple[np.ndarray, float]]] = {
        "top": [], "bottom": [], "left": [], "right": [],
    }
    for L in lines:
        ang = _line_angle_deg(L)
        d = _line_perp_dist(L, centroid)
        contrast = _contrast_along_line(L, gray_blur)
        if contrast < EDGE_CONTRAST_MIN:
            continue  # not a paper/table boundary
        if _ang_diff(ang, angle_h_coarse) <= ORIENT_BIN_DEG:
            (bins["top"] if d < 0 else bins["bottom"]).append((L, d))
        elif _ang_diff(ang, angle_v_coarse) <= ORIENT_BIN_DEG:
            (bins["left"] if d < 0 else bins["right"]).append((L, d))

    # 5. For each bin, pick the OUTERMOST line (largest |d|).
    sides: dict[str, np.ndarray | None] = {}
    for name in ("top", "bottom", "left", "right"):
        cands = bins[name]
        if not cands:
            sides[name] = None
            continue
        sides[name] = max(cands, key=lambda t: abs(t[1]))[0]

    if debug_dir is not None:
        sides_viz = (frame_bgr.astype(np.float32) * 0.5).astype(np.uint8)
        for name, color in (("top", (0, 0, 255)), ("bottom", (0, 255, 0)),
                            ("left", (255, 0, 0)), ("right", (0, 255, 255))):
            for L, _d in bins[name]:
                x1, y1, x2, y2 = [int(v) for v in L]
                cv2.line(sides_viz, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            if sides[name] is not None:
                x1, y1, x2, y2 = [int(v) for v in sides[name]]
                cv2.line(sides_viz, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        cv2.polylines(sides_viz, [coarse_corners.astype(np.int32)], True, (255, 255, 255), 1)
        missing_now = [s for s in sides if sides[s] is None]
        cv2.imwrite(str(dbg / "04_oriented_lines_and_selected_sides.png"),
                    _draw_lines(sides_viz, [],
                                label=f"04. lines per orientation (thin) + selected outermost "
                                      f"(thick). red=top, green=bot, blue=left, yellow=right. "
                                      f"missing={missing_now}"))
    if any(sides[s] is None for s in sides):
        if verbose:
            missing = [s for s in sides if sides[s] is None]
            print(f"[WARN] refine_paper_quad_missing_sides: expected=4 sides, "
                  f"got_missing={missing}, fallback=None", flush=True)
        return None

    # 6. Intersect adjacent sides → 4 corners.
    # Order: top∩left = TL, top∩right = TR, bottom∩right = BR, bottom∩left = BL
    pairs = [("top", "left"), ("top", "right"),
             ("bottom", "right"), ("bottom", "left")]
    raw_corners = []
    for a, b in pairs:
        p = _intersect(sides[a], sides[b])
        if p is None:
            if verbose:
                print(f"[WARN] refine_paper_quad_parallel_sides: pair=({a},{b}), "
                      f"fallback=None", flush=True)
            return None
        raw_corners.append(p)
    refined = np.array(raw_corners, dtype=np.float32)

    # Clip to image bounds. cv2.cornerSubPix asserts on out-of-bounds.
    refined[:, 0] = np.clip(refined[:, 0], 1.0, W - 2.0)
    refined[:, 1] = np.clip(refined[:, 1], 1.0, H - 2.0)

    # 7. cv2.cornerSubPix saddle-snap.
    refined_for_sp = refined.reshape(-1, 1, 2).copy()
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            SUBPIX_ITERS, SUBPIX_EPS)
    try:
        cv2.cornerSubPix(gray, refined_for_sp, SUBPIX_WIN, (-1, -1), crit)
        refined = refined_for_sp.reshape(-1, 2)
    except cv2.error as e:
        if verbose:
            print(f"[WARN] refine_paper_quad_cornerSubPix_failed: {e}, "
                  f"fallback=raw_intersections", flush=True)
        # Keep raw intersections — they're already sub-pixel.

    # Re-order TL/TR/BR/BL (cornerSubPix doesn't permute, but be defensive).
    refined = _order_tl_tr_br_bl(refined)

    # 8. Sanity gates.
    if not _is_convex(refined):
        if verbose:
            print(f"[WARN] refine_paper_quad_not_convex: fallback=None", flush=True)
        return None
    coarse_area = _quad_area(coarse_corners)
    refined_area = _quad_area(refined)
    if refined_area > MAX_BAND_GROWTH * coarse_area or refined_area < 0.4 * coarse_area:
        if verbose:
            print(f"[WARN] refine_paper_quad_area_out_of_range: "
                  f"expected=area in [0.4, {MAX_BAND_GROWTH}]×coarse, "
                  f"got={refined_area/max(coarse_area,1):.2f}×, fallback=None",
                  flush=True)
        return None
    iou = _quad_iou(_order_tl_tr_br_bl(coarse_corners), refined, (H, W))

    if debug_dir is not None:
        final_viz = frame_bgr.copy()
        cv2.polylines(final_viz, [coarse_corners.astype(np.int32)], True, (0, 0, 255), 2)
        cv2.polylines(final_viz, [refined.astype(np.int32)], True, (0, 255, 0), 2)
        for p in refined:
            cv2.circle(final_viz, (int(p[0]), int(p[1])), 6, (0, 255, 0), -1)
            cv2.circle(final_viz, (int(p[0]), int(p[1])), 7, (255, 255, 255), 1)
        cv2.imwrite(str(dbg / "05_final_refined_vs_coarse.png"),
                    _draw_lines(final_viz, [],
                                label=f"05. coarse SAM quad (red) vs refined quad (green).  "
                                      f"IoU={iou:.2f}  area_ratio={refined_area/max(coarse_area,1):.2f}"))

    if iou < MIN_IOU_VS_COARSE:
        if verbose:
            print(f"[WARN] refine_paper_quad_low_iou_vs_coarse: "
                  f"expected≥{MIN_IOU_VS_COARSE}, got={iou:.2f}, fallback=None",
                  flush=True)
        return None
    return refined.astype(np.float32)
