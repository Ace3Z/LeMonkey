"""Render diagnostic panels for stage 2 outputs.

Two PNGs land in each eval3_aug variant folder:
  - dbg_stage2_portraits.png  (3 panels: GroundingDINO bboxes, SAM masks, final quads)
  - dbg_stage2_occluders.png  (5 sampled frames showing occluder regions per portrait)

Usage:
    python dbg_stage2_panels.py <variant_dir>
    python dbg_stage2_panels.py --root <eval3_aug_root>
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mu

# BGR per portrait id
COLORS = [(80, 80, 255), (80, 255, 80), (255, 130, 80)]
OCCLUDER_COLOR = (40, 220, 255)


def _local_lemonkey_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "LeMonkey" and (parent / "datasets").exists():
            return parent
    return None


def find_source_episode(variant_dir: Path) -> Path | None:
    aug = json.loads((variant_dir / "augmentation.json").read_text())
    src = aug["src_episode"]
    root = _local_lemonkey_root()
    if root is None:
        return None
    for sub in ("datasets/eval3", "datasets/eval3_quick"):
        cand = root / sub / src
        if cand.is_dir():
            return cand
    return None


def _label(img, text, x, y, color):
    cv2.rectangle(img, (x, y - 18), (x + 8 * len(text) + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)


def _panel_header(img, title):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _overlay_mask(img, mask_bool, color, alpha=0.45):
    overlay = img.copy()
    overlay[mask_bool] = color
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def make_portrait_panels(ep_dir: Path) -> np.ndarray:
    frame0 = cv2.imread(str(ep_dir / "frame_0.png"))
    if frame0 is None:
        raise RuntimeError(f"missing frame_0.png in {ep_dir}")
    seeds = json.loads((ep_dir / "portrait_seeds.json").read_text())
    corners_data = json.loads((ep_dir / "portrait_corners.json").read_text())
    masks_pkl = pickle.load(open(ep_dir / "portrait_masks.pkl", "rb"))
    M_0 = masks_pkl["M_0_per_pid"]

    # ── Panel A: GroundingDINO portrait bboxes ─────────────────────────
    a = frame0.copy()
    for i, (box, score, celeb, cos) in enumerate(zip(
            seeds["boxes_xyxy"], seeds["box_scores"],
            seeds["celebs"], seeds["arcface_cosines"])):
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(a, (x0, y0), (x1, y1), COLORS[i], 2)
        _label(a, f"pid{i} {celeb} gdino={score:.2f} arc={cos:.2f}", x0, y0, COLORS[i])
    trusted = seeds.get("arcface_trusted")
    a = _panel_header(a, f"A. GroundingDINO portrait bboxes  (ArcFace trusted={trusted})")

    # ── Panel B: SAM portrait masks (M_0) ──────────────────────────────
    b = frame0.copy()
    for i in range(3):
        m = M_0[i].astype(bool)
        b = _overlay_mask(b, m, COLORS[i], alpha=0.45)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(b, cnts, -1, COLORS[i], 2)
    mask_src = seeds.get("mask_sources", {})
    src_line = "  ".join(f"pid{i}: {mask_src.get(str(i), '?')}" for i in range(3))
    b = _panel_header(b, f"B. SAM 2.1 portrait masks   {src_line}")

    # ── Panel C: final 4-corner quads ─────────────────────────────────
    c = frame0.copy()
    for i in range(3):
        quad = corners_data["portraits"][str(i)]["0"]["corners"]
        pts = np.array(quad, dtype=np.int32)
        cv2.polylines(c, [pts], isClosed=True, color=COLORS[i], thickness=2)
        for j, v in enumerate(pts):
            cv2.circle(c, tuple(v), 5, COLORS[i], -1)
            cv2.circle(c, tuple(v), 6, (255, 255, 255), 1)
            cv2.putText(c, "TL TR BR BL".split()[j], (v[0] + 7, v[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    for k, fc in (corners_data.get("face_centers") or {}).items():
        if fc is None:
            continue
        cv2.drawMarker(c, (int(fc[0]), int(fc[1])), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
    c = _panel_header(c, "C. Final quads (minAreaRect + v9.5 face-aware reorder; white cross = face center)")

    return np.hstack([a, b, c])


def make_occluder_panels(ep_dir: Path, n_samples: int = 5) -> np.ndarray:
    masks_pkl = pickle.load(open(ep_dir / "portrait_masks.pkl", "rb"))
    M_0 = masks_pkl["M_0_per_pid"]
    per_frame = masks_pkl["masks"]
    n_frames = max(per_frame.keys()) + 1
    h264 = masks_pkl.get("video_path")

    sample_idx = np.linspace(0, n_frames - 1, n_samples, dtype=int).tolist()
    cap = cv2.VideoCapture(h264) if h264 else None
    out_panels = []
    for fi in sample_idx:
        frame = None
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                frame = None
        if frame is None:
            frame = cv2.imread(str(ep_dir / "frame_0.png")).copy()

        # Occluder per pid = M_0_pid AND NOT visible_paper_pid
        occluder_total = np.zeros(frame.shape[:2], dtype=bool)
        for i in range(3):
            visible = mu.decode(per_frame[fi][i]["rle"]).astype(bool)
            occ = M_0[i].astype(bool) & ~visible
            occluder_total |= occ
            cnts, _ = cv2.findContours(M_0[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cnts, -1, COLORS[i], 1)
        if occluder_total.any():
            frame = _overlay_mask(frame, occluder_total, OCCLUDER_COLOR, alpha=0.55)
            cnts, _ = cv2.findContours(occluder_total.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cnts, -1, OCCLUDER_COLOR, 1)
        frame = _panel_header(frame, f"frame {fi} / {n_frames - 1}  "
                                     f"(yellow = occluder over portrait; colored outlines = portraits)")
        out_panels.append(frame)
    if cap is not None:
        cap.release()
    return np.hstack(out_panels)


def _render_occluder_frame(frame: np.ndarray, M_0: dict, per_frame_masks: dict,
                            fi: int, n_frames: int) -> np.ndarray:
    """Annotate `frame` with portrait outlines + occluder regions (yellow).
    Returns side-by-side (orig | annotated) for video output."""
    annotated = frame.copy()
    occluder_total = np.zeros(frame.shape[:2], dtype=bool)
    for i in range(3):
        visible = mu.decode(per_frame_masks[fi][i]["rle"]).astype(bool)
        occ = M_0[i].astype(bool) & ~visible
        occluder_total |= occ
        cnts, _ = cv2.findContours(M_0[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, cnts, -1, COLORS[i], 1)
    if occluder_total.any():
        annotated = _overlay_mask(annotated, occluder_total, OCCLUDER_COLOR, alpha=0.55)
        cnts, _ = cv2.findContours(occluder_total.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, cnts, -1, OCCLUDER_COLOR, 1)
    # Side-by-side
    H, W = frame.shape[:2]
    sbs = np.zeros((H, 2 * W, 3), dtype=np.uint8)
    sbs[:, :W] = frame
    sbs[:, W:] = annotated
    cv2.rectangle(sbs, (0, 0), (2 * W, 24), (0, 0, 0), -1)
    cv2.putText(sbs, "ORIGINAL", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (80, 255, 80), 1, cv2.LINE_AA)
    cv2.putText(sbs, f"OCCLUDERS (yellow) + portrait outlines  frame {fi}/{n_frames - 1}",
                (W + 8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 255), 1, cv2.LINE_AA)
    return sbs


def make_occluder_video(ep_dir: Path, out_path: Path) -> dict:
    masks_pkl = pickle.load(open(ep_dir / "portrait_masks.pkl", "rb"))
    M_0 = masks_pkl["M_0_per_pid"]
    per_frame = masks_pkl["masks"]
    n_frames = max(per_frame.keys()) + 1
    h264 = masks_pkl.get("video_path")
    if not h264 or not Path(h264).is_file():
        return {"error": f"video_path missing or unreadable: {h264}"}

    cap = cv2.VideoCapture(h264)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_w = 2 * W + (2 * W) % 2
    out_h = H + H % 2

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if fi not in per_frame:
                fi += 1
                continue
            sbs = _render_occluder_frame(frame, M_0, per_frame, fi, n_frames)
            if sbs.shape[0] != out_h or sbs.shape[1] != out_w:
                sbs = cv2.copyMakeBorder(sbs, 0, out_h - sbs.shape[0],
                                          0, out_w - sbs.shape[1],
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cv2.imwrite(str(td / f"f{fi:06d}.png"), sbs)
            fi += 1
        cap.release()
        # Encode via system ffmpeg (libx264 + yuv420p for playback compat)
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-framerate", str(int(round(fps))),
            "-i", str(td / "f%06d.png"),
            "-c:v", "libx264", "-crf", "20", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_path),
        ]
        rc = subprocess.run(cmd, check=False, stdin=subprocess.DEVNULL)
        if rc.returncode != 0 or not out_path.is_file():
            return {"error": "ffmpeg encode failed"}
    return {"ok": True, "n_frames": fi, "fps": fps, "out": str(out_path)}


def render_one(variant_dir: Path) -> dict:
    if not (variant_dir / "augmentation.json").is_file():
        return {"variant": variant_dir.name, "error": "augmentation.json missing"}
    ep_dir = find_source_episode(variant_dir)
    if ep_dir is None:
        return {"variant": variant_dir.name, "error": "source episode dir not found"}
    needed = ["frame_0.png", "portrait_seeds.json", "portrait_corners.json", "portrait_masks.pkl"]
    missing = [n for n in needed if not (ep_dir / n).is_file()]
    if missing:
        return {"variant": variant_dir.name, "error": f"missing in src: {missing}"}
    try:
        portraits_img = make_portrait_panels(ep_dir)
        occ_img = make_occluder_panels(ep_dir)
        out_p = variant_dir / "dbg_stage2_portraits.png"
        out_o = variant_dir / "dbg_stage2_occluders.png"
        out_v = variant_dir / "dbg_stage2_occluders.mp4"
        cv2.imwrite(str(out_p), portraits_img)
        cv2.imwrite(str(out_o), occ_img)
        vid_res = make_occluder_video(ep_dir, out_v)
        return {"variant": variant_dir.name,
                "portraits": str(out_p),
                "occluders_png": str(out_o),
                "occluders_mp4": str(out_v) if vid_res.get("ok") else f"FAILED: {vid_res.get('error')}"}
    except Exception as e:
        traceback.print_exc()
        return {"variant": variant_dir.name, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variant_dir", nargs="?", default=None)
    p.add_argument("--root", default=None,
                   help="iterate over all variant dirs under this root (eval3_aug)")
    args = p.parse_args()

    if args.variant_dir:
        targets = [Path(args.variant_dir)]
    elif args.root:
        root = Path(args.root)
        targets = sorted(p for p in root.iterdir()
                         if p.is_dir() and (p / "augmentation.json").is_file())
    else:
        p.print_usage(sys.stderr)
        return 2

    rc = 0
    for v in targets:
        r = render_one(v)
        if r.get("error"):
            print(f"  ✗ {v.name:<55} {r['error']}", flush=True)
            rc = 1
        else:
            print(f"  ✓ {v.name:<55} → portraits + occluders", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
