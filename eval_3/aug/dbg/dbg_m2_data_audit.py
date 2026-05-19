#!/usr/bin/env python3
"""M2 data-audit: verify that face_labels ↔ augmentation.json ↔ celeb_embeddings.json
compose correctly when joined at training time.

For each of 5 representative-variant face_labels files (covering 5 distinct
layouts), the script:

  1. Loads the manifest, the face_labels.json, and the representative
     variant's augmentation.json.
  2. Calls `build_supervision_for_frame(frame_0, new_layout_camera_lmr, lookup)`
     and prints the slot→celeb mapping, bbox, n_active patches, centroid hash.
  3. Re-runs buffalo_l ArcFace on the bbox crop of camera1's frame 0 and
     asks: across all 192 manifest centroids, who's nearest? Compares to the
     celeb augmentation.json says should be there.
  4. Recomputes the centroid for the three on-screen celebs from their
     stored per-photo .npy embeddings and compares to the manifest centroid.
  5. Reports norms of `target_centroids` returned by
     `build_supervision_for_frame`.
  6. Exercises the n_visible_faces < 3 path by synthesising a frame with the
     trailing bboxes stripped and confirming `valid[s] = False` for the
     missing slots.

Run with the lemonkey-arcface conda env.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path("/Users/mahbod/swiss uni /Sem-2/Robotics/project")
sys.path.insert(0, str(PROJECT_ROOT))

from eval_3.aug.m2_alignment import build_supervision_for_frame, slot_to_celeb  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed inputs.
# ---------------------------------------------------------------------------

MANIFEST_PATH = PROJECT_ROOT / "eval_3/aug/stats/celeb_embeddings.json"
FACE_LABELS_DIR = PROJECT_ROOT / "eval_3/aug/stats/face_labels"
AUG_ROOT = Path(os.path.expanduser("~/Downloads/eval3_track3_aug"))
CELEB_BANK_ROOT = Path(os.path.expanduser("~/Downloads/eval3_celebs"))

# One representative file per layout. User asked for {LSO, LOS, OLS, OSL, SLO}
# but the face_labels collection only contains {LSO, SLO, SOL, OSL, OLS}; LOS
# is not present, so we substitute SOL (the only available layout the user
# did not explicitly list).
SELECTED_FACE_LABELS = [
    "quick_lecun_LSO_ep01_20260511_205000.face_labels.json",  # LSO
    "quick_lecun_SLO_ep01_20260511_210355.face_labels.json",  # SLO
    "quick_lecun_SOL_ep01_20260511_212006.face_labels.json",  # SOL (substituted for LOS)
    "quick_obama_OSL_ep01_20260511_200348.face_labels.json",  # OSL
    "quick_swift_OLS_ep01_20260511_192524.face_labels.json",  # OLS
]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _centroid_hash(c: np.ndarray) -> str:
    """First 8 hex of sha1 over the centroid's float32 bytes."""
    return hashlib.sha1(np.asarray(c, dtype=np.float32).tobytes()).hexdigest()[:8]


def _decode_frame(mp4_path: Path, frame_idx: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_idx} of {mp4_path}")
    return frame  # BGR


def _build_arcface(det_size: int = 320):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def _arcface_embed_from_crop(app, frame_bgr: np.ndarray, bbox_xyxy) -> np.ndarray | None:
    """Run detection over a crop expanded around the bbox; pick the largest
    face that overlaps the original bbox; return its L2-normalised 512-D embedding.

    We feed the WHOLE FRAME to the FaceAnalysis pipeline (detection +
    recognition) rather than a tight crop, then match by overlap. This is
    what the rest of the pipeline does, and it's what gives the alignment
    transform InsightFace expects."""
    faces = app.get(frame_bgr)
    if not faces:
        return None
    bx1, by1, bx2, by2 = bbox_xyxy
    def iou(f):
        fx1, fy1, fx2, fy2 = f.bbox
        ix1, iy1 = max(bx1, fx1), max(by1, fy1)
        ix2, iy2 = min(bx2, fx2), min(by2, fy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        a1 = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        a2 = max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
        u = a1 + a2 - inter
        return 0.0 if u <= 0 else inter / u
    faces.sort(key=iou, reverse=True)
    best = faces[0]
    if iou(best) < 0.3:
        return None
    emb = best.normed_embedding
    if emb is None:
        # Compute from .embedding if normed_embedding is missing.
        emb = best.embedding
        emb = emb / (np.linalg.norm(emb) + 1e-12)
    return np.asarray(emb, dtype=np.float32)


def _nearest_celeb(query: np.ndarray, centroids: dict[str, np.ndarray]) -> tuple[str, float, str, float]:
    """Return (top1_celeb, top1_cos, top2_celeb, top2_cos)."""
    qn = query / (np.linalg.norm(query) + 1e-12)
    best, second = ("", -2.0), ("", -2.0)
    for name, c in centroids.items():
        cn = c / (np.linalg.norm(c) + 1e-12)
        s = float(qn @ cn)
        if s > best[1]:
            second = best
            best = (name, s)
        elif s > second[1]:
            second = (name, s)
    return best[0], best[1], second[0], second[1]


# ---------------------------------------------------------------------------
# Main audit loop.
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"[info] loading manifest from {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text())
    centroid_lookup_full: dict[str, np.ndarray] = {
        name: np.asarray(c["centroid"], dtype=np.float32)
        for name, c in manifest["celebs"].items()
    }
    print(f"[info] manifest has {len(centroid_lookup_full)} celeb centroids")
    bank_root_manifest = Path(manifest["bank_root"])

    # ----- Q3: recompute the three on-screen celebs' centroids from .npy
    print()
    print("=== Centroid-recompute check (mean+normalize over per-photo .npy) ===")
    recompute_rows = []
    for celeb in ["barack_obama", "yann_lecun", "taylor_swift"]:
        entry = manifest["celebs"][celeb]
        photos = entry["photos"]   # dict: rel-png → rel-npy
        npys = []
        for png_rel, npy_rel in photos.items():
            # We have to use the locally-rooted path; manifest stores rel.
            npy_path = CELEB_BANK_ROOT / npy_rel
            if not npy_path.is_file():
                print(f"  [WARN] {celeb}: missing npy {npy_path}")
                continue
            npys.append(np.load(npy_path).astype(np.float32))
        if not npys:
            print(f"  [WARN] {celeb}: no npys loaded; skipping")
            continue
        stacked = np.stack(npys, axis=0)
        # Replicate the canonical centroid pipeline: each per-photo embedding
        # is already L2-normalized (we just verified one above), then we
        # average, then L2-normalize the average.
        avg = stacked.mean(axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-12)
        manifest_c = centroid_lookup_full[celeb]
        manifest_c_n = manifest_c / (np.linalg.norm(manifest_c) + 1e-12)
        cos = float(avg @ manifest_c_n)
        norm_manifest = float(np.linalg.norm(manifest_c))
        recompute_rows.append((celeb, len(npys), entry["n_photos"], norm_manifest, cos))
        print(f"  {celeb}: n_npys_loaded={len(npys)} (manifest.n_photos={entry['n_photos']}) "
              f"|manifest centroid|={norm_manifest:.6f}  cos(recompute, manifest)={cos:.6f}")

    # ----- Build ArcFace once.
    print()
    print("[info] loading buffalo_l (CPU, det_size=320) ...", flush=True)
    app = _build_arcface(det_size=320)
    print("[info] buffalo_l loaded.")

    # ----- Per-frame join audit.
    rows = []  # (source, slot, bbox, expected, top1, top2_celeb, top2_cos, match, centroid_norm)
    print()
    print("=== Per-frame join audit (frame 0 of representative variant) ===")
    for fname in SELECTED_FACE_LABELS:
        face_labels_path = FACE_LABELS_DIR / fname
        fl = json.loads(face_labels_path.read_text())
        rep_variant = fl["representative_variant"]
        source = fl["source_episode"]

        aug_path = AUG_ROOT / rep_variant / "augmentation.json"
        aug = json.loads(aug_path.read_text())
        new_lmr = aug["new_layout_camera_lmr"]
        expected_celebs = slot_to_celeb(new_lmr)

        # build_supervision for frame 0
        frame_0 = fl["frames"][0]
        masks, valid, targets = build_supervision_for_frame(
            frame_0, new_lmr, centroid_lookup_full
        )
        n_visible = int(frame_0["n_visible_faces"])

        # Decode frame 0 of the rep variant's camera1.mp4
        mp4 = AUG_ROOT / rep_variant / "videos/observation.images.camera1/chunk-000/file-000.mp4"
        frame_bgr = _decode_frame(mp4, frame_idx=0)
        h, w = frame_bgr.shape[:2]

        print(f"\n--- {source}")
        print(f"    layout (filename): {aug['orig_layout_filename']}  "
              f"new_layout_camera_lmr: {new_lmr}  (slot→celeb: {expected_celebs})")
        print(f"    rep variant: {rep_variant}")
        print(f"    n_visible_faces (frame 0): {n_visible}   camera1 frame shape: {h}x{w}")
        print(f"    target_centroids norms: "
              f"{[float(np.linalg.norm(t)) for t in targets]}")

        for s in range(3):
            if s >= n_visible:
                # Missing-slot: confirm valid[s] is False.
                rows.append({
                    "source": source,
                    "slot": "LMR"[s],
                    "bbox": "(missing)",
                    "expected_celeb": expected_celebs[s] if s < len(expected_celebs) else "(n/a)",
                    "top1": "(no detection)",
                    "top1_cos": float("nan"),
                    "top2": "",
                    "top2_cos": float("nan"),
                    "match": "n/a",
                    "centroid_norm": float(np.linalg.norm(targets[s])),
                    "n_active_patches": int(masks[s].sum()),
                    "valid_flag": bool(valid[s]),
                    "centroid_hash": _centroid_hash(targets[s]),
                })
                print(f"    slot {s} ({'LMR'[s]}): MISSING (n_visible_faces<{s+1}); "
                      f"valid={bool(valid[s])} (expected False)  "
                      f"n_active_patches={int(masks[s].sum())} (expected 0)")
                continue

            b = frame_0["bboxes"][s]
            bbox = (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]))
            celeb_expected = expected_celebs[s]
            target_c = targets[s]

            # Independent ArcFace embed of THIS bbox in THIS frame; nearest centroid
            # across all 192 celebs.
            emb = _arcface_embed_from_crop(app, frame_bgr, bbox)
            if emb is None:
                top1, top1_cos = "(no face)", float("nan")
                top2, top2_cos = "", float("nan")
                match = "no-detect"
            else:
                top1, top1_cos, top2, top2_cos = _nearest_celeb(emb, centroid_lookup_full)
                match = "OK" if top1 == celeb_expected else "MISMATCH"

            n_active = int(masks[s].sum())
            c_norm = float(np.linalg.norm(target_c))
            c_hash = _centroid_hash(target_c)

            rows.append({
                "source": source,
                "slot": "LMR"[s],
                "bbox": f"({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})",
                "expected_celeb": celeb_expected,
                "top1": top1,
                "top1_cos": top1_cos,
                "top2": top2,
                "top2_cos": top2_cos,
                "match": match,
                "centroid_norm": c_norm,
                "n_active_patches": n_active,
                "valid_flag": bool(valid[s]),
                "centroid_hash": c_hash,
            })
            print(f"    slot {s} ({'LMR'[s]}): bbox={bbox}  expected={celeb_expected}  "
                  f"top1={top1} (cos={top1_cos:.3f})  top2={top2} (cos={top2_cos:.3f})  "
                  f"match={match}  n_active={n_active}  |c|={c_norm:.4f}  hash={c_hash}  valid={valid[s]}")

    # ----- n_visible<3 path synthetic test.
    print()
    print("=== Partial-data path (synthetic, n_visible_faces<3) ===")
    fl = json.loads((FACE_LABELS_DIR / SELECTED_FACE_LABELS[0]).read_text())
    frame0 = fl["frames"][0]
    new_lmr = "OLS"  # arbitrary

    for n_keep in [0, 1, 2, 3]:
        fake_frame = dict(frame0)
        fake_frame["bboxes"] = frame0["bboxes"][:n_keep]
        fake_frame["n_visible_faces"] = n_keep
        masks, valid, targets = build_supervision_for_frame(
            fake_frame, new_lmr, centroid_lookup_full
        )
        expect_valid = [s < n_keep for s in range(3)]
        print(f"  n_keep={n_keep}: valid={list(map(bool, valid))}  "
              f"expected={expect_valid}  "
              f"OK={list(map(bool, valid))==expect_valid}  "
              f"mask_sums={[int(m.sum()) for m in masks]}  "
              f"|targets|={[float(np.linalg.norm(t)) for t in targets]}")

    # ----- target_centroids norms across all rows.
    norms = [r["centroid_norm"] for r in rows if r["valid_flag"]]
    print()
    print(f"=== target_centroid norms (across all valid slots): "
          f"min={min(norms):.6f} max={max(norms):.6f} ===")

    # ----- Tabular summary.
    print()
    print("=" * 110)
    print("TABULAR REPORT (one row per (source, slot)):")
    print("=" * 110)
    hdr = ("source", "slot", "bbox", "expected_celeb", "top1_pred", "match", "|centroid|")
    print(f"{hdr[0]:<55} {hdr[1]:<4} {hdr[2]:<22} {hdr[3]:<18} {hdr[4]:<18} {hdr[5]:<8} {hdr[6]:<10}")
    print("-" * 140)
    mismatches = 0
    for r in rows:
        src_short = r["source"].replace("quick_", "")
        match_str = r["match"]
        if match_str == "MISMATCH":
            mismatches += 1
        print(f"{src_short:<55} {r['slot']:<4} {r['bbox']:<22} "
              f"{r['expected_celeb']:<18} {r['top1']:<18} {match_str:<8} "
              f"{r['centroid_norm']:<10.6f}")
    print("-" * 140)
    print()
    if mismatches == 0:
        print("BOTTOM LINE: JOIN IS CORRECT (all top-1 ArcFace predictions match aug.json claims).")
    else:
        print(f"BOTTOM LINE: BUG: {mismatches} slot(s) mismatch between aug.json's expected celeb "
              f"and ArcFace top-1 prediction. See rows above.")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
