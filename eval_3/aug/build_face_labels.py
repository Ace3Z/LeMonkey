#!/usr/bin/env python3
"""Build per-source-episode face_labels.json from the camera1 videos.

Why this exists:
  The augmented variant dirs in `eval3_track3_aug/` have everything we
  need to train an M2 face-alignment loss EXCEPT the per-frame face
  bounding boxes in camera1. The aug pipeline used `portrait_corners.json`
  from each source recording to paste photos, but those source-corner
  files are not propagated into the variant dirs. This script recovers
  them by running RetinaFace on one representative variant per source
  episode (since variants from the same source share identical camera
  trajectories → identical face pixel positions).

What it does:
  1. Group variants by source-episode prefix (e.g. `quick_lecun_LSO_ep01_20260511_205000`).
  2. For each unique source episode, take the FIRST variant (`v00` ideally)
     as the representative and read its camera1.mp4.
  3. For each frame, run RetinaFace (via InsightFace `buffalo_l`).
  4. Keep top-3 faces by bbox area, sort left-to-right by x-center.
  5. Write `<output_dir>/<source_episode>.face_labels.json` with
     per-frame {n_visible_faces, bboxes[, ...3...]} where each bbox has
     {x1,y1,x2,y2,score,x_center} and a "position" tag (left/middle/right).

Why no ArcFace identity matching here:
  augmentation.json already tells the training dataloader which celeb
  sits at left/middle/right via `new_layout_camera_lmr` (e.g. "OLS" →
  Obama-Left, LeCun-Middle, Swift-Right). face_labels.json is purely
  spatial; identity is joined at training time. This keeps the artifact
  small and reusable across all variants of a given source.

Performance budget:
  151 source episodes × 538 frames × ~10ms RetinaFace per frame ≈ 13.5 min
  on a single CUDA GPU (qolam). On Mac CPU: ~1.5-2 h. Use --stride N to
  process every Nth frame and linearly interpolate the rest (cuts wall
  time by a factor of N at modest accuracy cost — the bboxes move
  smoothly because the wrist cam moves smoothly).

Usage (qolam, recommended):

    python eval_3/aug/build_face_labels.py \
        --aug-root /path/to/eval3_track3_aug \
        --output-dir eval_3/aug/stats/face_labels \
        --stride 1

Use --dry-run to print the source-episode grouping without doing any
inference. Use --limit N to process only the first N source episodes for
testing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np


VARIANT_PREFIX_RE = re.compile(r"^(.+)__t3_\d+_v\d+$")


def _build_arcface_app(det_size: int, use_cuda: bool):
    """Load buffalo_l. We use the FaceAnalysis pipeline but only the
    detector is needed — ArcFace recognition is skipped at build time."""
    from insightface.app import FaceAnalysis

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if use_cuda else ["CPUExecutionProvider"])
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],  # recognition not needed here
        providers=providers,
    )
    app.prepare(ctx_id=0 if use_cuda else -1, det_size=(det_size, det_size))
    return app


def _group_variants_by_source(aug_root: Path) -> dict[str, list[Path]]:
    """Return {source_episode_prefix → [variant_dir, ...]} sorted."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for entry in aug_root.iterdir():
        if not entry.is_dir():
            continue
        m = VARIANT_PREFIX_RE.match(entry.name)
        if m is None:
            continue
        groups[m.group(1)].append(entry)
    for src in groups:
        groups[src].sort()
    return dict(sorted(groups.items()))


def _pick_representative(group: list[Path]) -> Path | None:
    """Return the variant whose camera1.mp4 we'll segment. Prefer `_v00`."""
    for v in group:
        mp4 = v / "videos/observation.images.camera1/chunk-000/file-000.mp4"
        if mp4.is_file():
            return v
    return None


def _decode_frames(mp4_path: Path):
    """Yield (frame_idx, BGR np.ndarray) for every frame in the video."""
    import cv2

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 failed to open {mp4_path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1
    cap.release()


def _detect_faces(app, frame_bgr, max_n: int = 3, score_thresh: float = 0.4):
    """Return up to max_n bboxes, sorted left-to-right by x-center."""
    faces = app.get(frame_bgr)
    faces = [f for f in faces if float(f.det_score) >= score_thresh]
    # Largest by area first, take top max_n
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    faces = faces[:max_n]
    # Then sort left-to-right
    faces.sort(key=lambda f: 0.5 * (f.bbox[0] + f.bbox[2]))
    out = []
    for f in faces:
        x1, y1, x2, y2 = (float(v) for v in f.bbox)
        out.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "score": float(f.det_score),
            "x_center": 0.5 * (x1 + x2),
        })
    return out


def _process_one_source(app, source: str, rep_variant: Path,
                        stride: int, score_thresh: float) -> dict:
    """Run RetinaFace on the representative variant of `source`."""
    mp4 = rep_variant / "videos/observation.images.camera1/chunk-000/file-000.mp4"

    frames_data: list[dict] = []
    keyframe_data: dict[int, list[dict]] = {}

    for frame_idx, frame in _decode_frames(mp4):
        if frame_idx % stride == 0:
            bboxes = _detect_faces(app, frame, max_n=3, score_thresh=score_thresh)
            keyframe_data[frame_idx] = bboxes

    if not keyframe_data:
        return {
            "source_episode": source,
            "n_frames": 0,
            "stride": stride,
            "frames": [],
            "error": "no keyframes decoded",
        }

    n_frames = max(keyframe_data) + 1
    if stride > 1:
        # Linear-interpolate bboxes between keyframes for the in-between frames.
        # If a keyframe has fewer than 3 bboxes, mark the missing positions as None
        # and let the dataloader skip those frames.
        sorted_keys = sorted(keyframe_data.keys())
        for fidx in range(n_frames):
            if fidx in keyframe_data:
                bboxes = keyframe_data[fidx]
            else:
                # find nearest keyframes
                lo = max((k for k in sorted_keys if k <= fidx), default=None)
                hi = min((k for k in sorted_keys if k >= fidx), default=None)
                if lo is None:
                    bboxes = keyframe_data[hi]
                elif hi is None or hi == lo:
                    bboxes = keyframe_data[lo]
                else:
                    a, b = keyframe_data[lo], keyframe_data[hi]
                    if len(a) != len(b):  # transitions through occlusion — just copy lo
                        bboxes = a
                    else:
                        t = (fidx - lo) / (hi - lo)
                        bboxes = []
                        for ba, bb in zip(a, b):
                            bboxes.append({
                                "x1": ba["x1"] + t * (bb["x1"] - ba["x1"]),
                                "y1": ba["y1"] + t * (bb["y1"] - ba["y1"]),
                                "x2": ba["x2"] + t * (bb["x2"] - ba["x2"]),
                                "y2": ba["y2"] + t * (bb["y2"] - ba["y2"]),
                                "score": min(ba["score"], bb["score"]),
                                "x_center": 0.5 * ((ba["x1"] + ba["x2"]) + t *
                                                   ((bb["x1"] + bb["x2"]) -
                                                    (ba["x1"] + ba["x2"]))),
                                "interpolated": True,
                            })
            frames_data.append({"frame_idx": fidx,
                                "n_visible_faces": len(bboxes),
                                "bboxes": bboxes})
    else:
        for fidx in range(n_frames):
            bboxes = keyframe_data.get(fidx, [])
            frames_data.append({"frame_idx": fidx,
                                "n_visible_faces": len(bboxes),
                                "bboxes": bboxes})

    return {
        "source_episode": source,
        "representative_variant": rep_variant.name,
        "n_frames": n_frames,
        "stride": stride,
        "score_thresh": score_thresh,
        "schema_version": 1,
        "frames": frames_data,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug-root", type=Path, required=True,
                        help="Root containing the 9217 variant directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1,
                        help="Run RetinaFace every Nth frame, interpolate rest")
    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N source episodes (debug)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (default: use CUDA if available)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just print the source-episode grouping")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing face_labels.json")
    args = parser.parse_args()

    if not args.aug_root.is_dir():
        print(f"[ERR] aug-root not found: {args.aug_root}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] scanning {args.aug_root} for variants ...")
    groups = _group_variants_by_source(args.aug_root)
    print(f"[info] found {len(groups)} unique source episodes "
          f"across {sum(len(v) for v in groups.values())} variants")

    if args.dry_run:
        for src, group in list(groups.items())[:20]:
            print(f"  {src}: {len(group)} variants  (rep={group[0].name})")
        if len(groups) > 20:
            print(f"  ... ({len(groups) - 20} more)")
        return 0

    # Try CUDA; fall back to CPU with [WARN]
    use_cuda = not args.cpu
    if use_cuda:
        try:
            import onnxruntime  # noqa: F401
            providers = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                print(f"[WARN] CUDA requested but not available: "
                      f"expected=CUDAExecutionProvider, got={providers}, "
                      f"fallback=CPUExecutionProvider", flush=True)
                use_cuda = False
        except ImportError:
            print("[WARN] onnxruntime not importable for capability probe: "
                  "expected=installed, got=ImportError, fallback=CPU", flush=True)
            use_cuda = False

    print(f"[info] loading buffalo_l (det_size={args.det_size}, "
          f"providers={'CUDA' if use_cuda else 'CPU'})")
    app = _build_arcface_app(det_size=args.det_size, use_cuda=use_cuda)

    t0 = time.time()
    items = list(groups.items())
    if args.limit is not None:
        items = items[: args.limit]

    processed = 0
    skipped = 0
    failed = []

    for i, (src, group) in enumerate(items, start=1):
        out_path = args.output_dir / f"{src}.face_labels.json"
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        rep = _pick_representative(group)
        if rep is None:
            failed.append(f"{src} (no readable camera1.mp4 in any variant)")
            continue
        try:
            t_one = time.time()
            result = _process_one_source(app, src, rep,
                                         stride=args.stride,
                                         score_thresh=args.score_thresh)
            out_path.write_text(json.dumps(result))
            processed += 1
            print(f"[{i:3d}/{len(items)}] {src:50s} "
                  f"frames={result['n_frames']:4d} "
                  f"wall={time.time() - t_one:5.1f}s")
        except Exception as e:
            failed.append(f"{src} ({type(e).__name__}: {e})")
            print(f"[ERR] {src}: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\n[done] processed={processed} cached={skipped} failed={len(failed)} "
          f"wall={elapsed:.1f}s avg={elapsed / max(1, processed):.1f}s/source")
    if failed:
        print(f"[fail-list]")
        for f in failed:
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
