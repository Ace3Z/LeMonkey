#!/usr/bin/env python3
"""Visual gate for M2 toolkit outputs.

Two artifacts to inspect:

1) `eval_3/aug/stats/celeb_embeddings.json` — per-photo ArcFace embeddings
   plus per-celeb centroids built from heldout + scraped photos.

2) `eval_3/aug/stats/face_labels/<source>.face_labels.json` — per-frame
   face bboxes detected from one representative variant per source episode.

This script produces a `dbg/` directory of PNGs the user can open in
Preview / Finder to verify the toolkit is doing the right thing.

Outputs (under --output-dir):
  centroid_similarity_iid_plus_ood.png
      Cosine-similarity heatmap for 3 IID celebs + N random OOD.
      Off-diagonal cells should be near 0; diagonal at 1.0.

  source_overlays/<source_episode>_frame{0,mid,occl}.png
      Per-source-episode panels: frame 0, mid-trajectory, and one
      occlusion frame (if any). Each detected face bbox is drawn with
      a slot label (L/M/R) + the celeb name that should be there per
      the variant's augmentation.json + optional ArcFace cosine.

  matcher_sanity_<source>.png (one per --n-sources)
      Bbox crops side-by-side with the 3 candidate held-out photos
      and the nearest-centroid identity prediction. Confirms RetinaFace
      bboxes contain the right celeb and ArcFace can identify them.

Usage:

    python eval_3/aug/dbg/dbg_face_labels.py \
        --bank-root ~/Downloads/eval3_celebs \
        --aug-root ~/Downloads/eval3_track3_aug \
        --face-labels-dir eval_3/aug/stats/face_labels \
        --manifest eval_3/aug/stats/celeb_embeddings.json \
        --output-dir eval_3/aug/stats/face_labels_dbg \
        --n-sources 6

Requires the same conda env as cache_arcface_embeddings.py / build_face_labels.py
(opencv-python, insightface, numpy, matplotlib).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_manifest(p: Path) -> dict:
    return json.loads(p.read_text())


def _grab_frame(mp4: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _find_occluded_frame(frames_data: list[dict]) -> int | None:
    """Find the first frame where n_visible_faces < 3."""
    for f in frames_data:
        if f["n_visible_faces"] < 3:
            return f["frame_idx"]
    return None


def _layout_to_celeb_per_slot(aug: dict) -> list[str]:
    """Given augmentation.json, return [celeb_at_left, celeb_at_mid, celeb_at_right]."""
    # new_layout_camera_lmr is a 3-letter code like "OLS" meaning Obama-Left, LeCun-Middle, Swift-Right.
    letter_to_short = {"O": "obama", "L": "lecun", "S": "swift"}
    short_to_full = {"obama": "barack_obama", "lecun": "yann_lecun", "swift": "taylor_swift"}
    lmr = aug["new_layout_camera_lmr"]
    return [short_to_full[letter_to_short[c]] for c in lmr]


def _put_label(img, text, x, y, fg=(255, 255, 255), bg=(0, 0, 0), scale=0.5):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)


def _draw_overlay(frame_bgr: np.ndarray, bboxes: list[dict], celebs_per_slot: list[str],
                  arcface_cos: list[float] | None = None) -> np.ndarray:
    out = frame_bgr.copy()
    colors = [(60, 200, 60), (60, 200, 200), (60, 60, 220)]  # BGR (L=green, M=yellow, R=red)
    slot_names = ["L", "M", "R"]
    for i, b in enumerate(bboxes):
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        color = colors[i % 3]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        slot = slot_names[i] if i < 3 else f"?{i}"
        celeb = celebs_per_slot[i] if i < len(celebs_per_slot) else "?"
        cos_str = f"  cos={arcface_cos[i]:.2f}" if arcface_cos and i < len(arcface_cos) else ""
        label = f"{slot}: {celeb}  score={b['score']:.2f}{cos_str}"
        _put_label(out, label, x1, max(y1, 14), bg=color)
    return out


def _build_arcface_app(det_size: int = 320):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "recognition"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def _arcface_for_bbox(app, frame_bgr: np.ndarray, bbox: dict) -> np.ndarray | None:
    """Run buffalo_l on a small crop around the bbox; return the 512-D embedding."""
    # Expand bbox 25% to give RetinaFace some context for landmark fitting
    h, w = frame_bgr.shape[:2]
    bw = bbox["x2"] - bbox["x1"]
    bh = bbox["y2"] - bbox["y1"]
    px = int(0.25 * bw)
    py = int(0.25 * bh)
    x1 = max(0, int(bbox["x1"]) - px)
    y1 = max(0, int(bbox["y1"]) - py)
    x2 = min(w, int(bbox["x2"]) + px)
    y2 = min(h, int(bbox["y2"]) + py)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    faces = app.get(crop)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return np.asarray(f.normed_embedding, dtype=np.float32)


def _build_centroid_lookup(manifest: dict) -> dict[str, np.ndarray]:
    out = {}
    for celeb, info in manifest["celebs"].items():
        if info["centroid"]:
            out[celeb] = np.asarray(info["centroid"], dtype=np.float32)
    return out


def render_centroid_heatmap(manifest: dict, output_path: Path, n_ood: int = 8) -> None:
    import matplotlib.pyplot as plt

    iid = ["taylor_swift", "barack_obama", "yann_lecun"]
    celebs = manifest["celebs"]
    pool = [c for c in celebs if c not in iid and celebs[c]["n_photos"] >= 5]
    random.seed(0)
    ood = random.sample(pool, min(n_ood, len(pool)))
    rows = iid + ood

    cents = np.stack([np.asarray(celebs[c]["centroid"]) for c in rows])
    mat = cents @ cents.T

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(rows)), max(5, 0.5 * len(rows))))
    im = ax.imshow(mat, vmin=-0.2, vmax=1.0, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(len(rows)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(rows, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(rows)):
            t = f"{mat[i,j]:+.2f}"
            ax.text(j, i, t, ha="center", va="center",
                    fontsize=6, color="white" if mat[i,j] > 0.4 or mat[i,j] < -0.05 else "black")
    ax.set_title("ArcFace centroid cosine similarity — IID + random OOD\n"
                 "diagonal=1.0 (same celeb), off-diagonal ≈ 0 (different celebs)")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"[png] {output_path}")


def render_source_overlays(face_labels_path: Path, aug_root: Path, bank_root: Path,
                            centroid_lookup: dict[str, np.ndarray],
                            output_dir: Path, app) -> dict:
    """Render frame 0, mid, and one occluded frame for a source episode."""
    d = json.loads(face_labels_path.read_text())
    rep_var_name = d["representative_variant"]
    rep_var = aug_root / rep_var_name
    mp4 = rep_var / "videos/observation.images.camera1/chunk-000/file-000.mp4"
    aug = json.loads((rep_var / "augmentation.json").read_text())
    celebs_per_slot = _layout_to_celeb_per_slot(aug)

    n_frames = d["n_frames"]
    targets = [
        ("frame0", 0),
        ("mid", n_frames // 2),
    ]
    occluded = _find_occluded_frame(d["frames"])
    if occluded is not None:
        targets.append((f"occl{occluded}", occluded))

    summary = {"source": d["source_episode"], "panels": [], "matcher_acc": None}
    src_dir = output_dir / "source_overlays"
    src_dir.mkdir(parents=True, exist_ok=True)
    src = d["source_episode"]

    matcher_results = []
    for tag, fidx in targets:
        frame = _grab_frame(mp4, fidx)
        if frame is None:
            print(f"  [skip] {src} frame {fidx}: decode failed")
            continue
        # Find the entry in frames list
        entry = next((f for f in d["frames"] if f["frame_idx"] == fidx), None)
        if entry is None or not entry["bboxes"]:
            continue

        # For frame0 only, run ArcFace per bbox and compute cosines to celebs_per_slot's centroids
        cos_for_overlay = None
        if tag == "frame0":
            cos_for_overlay = []
            for slot_idx, b in enumerate(entry["bboxes"]):
                emb = _arcface_for_bbox(app, frame, b)
                if emb is None:
                    cos_for_overlay.append(float("nan"))
                    matcher_results.append(None)
                    continue
                expected = celebs_per_slot[slot_idx] if slot_idx < len(celebs_per_slot) else None
                # Nearest-centroid across all 192 celebs
                all_celebs = list(centroid_lookup.keys())
                all_cents = np.stack([centroid_lookup[c] for c in all_celebs])
                sims = all_cents @ emb
                winner_idx = int(np.argmax(sims))
                winner = all_celebs[winner_idx]
                cos_expected = float(centroid_lookup[expected] @ emb) if expected in centroid_lookup else float("nan")
                cos_for_overlay.append(cos_expected)
                matcher_results.append({
                    "slot": slot_idx, "expected": expected, "winner": winner,
                    "cos_expected": cos_expected, "cos_winner": float(sims[winner_idx]),
                    "correct": (winner == expected),
                })

        overlay = _draw_overlay(frame, entry["bboxes"], celebs_per_slot, cos_for_overlay)
        out_path = src_dir / f"{src}__{tag}.png"
        cv2.imwrite(str(out_path), overlay)
        summary["panels"].append(str(out_path.relative_to(output_dir)))
        print(f"  [png] {out_path}")

    n_correct = sum(1 for m in matcher_results if m and m["correct"])
    n_total = sum(1 for m in matcher_results if m is not None)
    summary["matcher_acc"] = f"{n_correct}/{n_total}" if n_total else "no-faces-readable"
    summary["matcher_detail"] = matcher_results
    return summary


def render_matcher_sanity(face_labels_path: Path, aug_root: Path, bank_root: Path,
                          centroid_lookup: dict[str, np.ndarray],
                          manifest: dict, output_dir: Path, app) -> None:
    """For one source: show frame0's 3 bbox crops next to each candidate held-out
    photo, plus the predicted nearest-centroid label."""
    import matplotlib.pyplot as plt

    d = json.loads(face_labels_path.read_text())
    rep_var = aug_root / d["representative_variant"]
    mp4 = rep_var / "videos/observation.images.camera1/chunk-000/file-000.mp4"
    aug = json.loads((rep_var / "augmentation.json").read_text())
    celebs_per_slot = _layout_to_celeb_per_slot(aug)
    frame = _grab_frame(mp4, 0)
    if frame is None:
        return
    entry0 = next((f for f in d["frames"] if f["frame_idx"] == 0), None)
    if not entry0 or not entry0["bboxes"]:
        return

    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    fig.suptitle(f"Matcher sanity: {d['source_episode']}\n"
                 f"(left = camera1 bbox crop, right = expected celeb scraped/heldout)",
                 fontsize=10)

    for i, b in enumerate(entry0["bboxes"][:3]):
        # Left: bbox crop
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(crop)
        axes[i, 0].set_title(f"slot {i} ({'LMR'[i]}) bbox crop", fontsize=9)
        axes[i, 0].axis("off")

        # Right: one of the expected celeb's photos
        expected = celebs_per_slot[i] if i < len(celebs_per_slot) else None
        if expected and expected in manifest["celebs"]:
            photos = list(manifest["celebs"][expected]["photos"].keys())
            if photos:
                # prefer a heldout photo if available
                heldout = [p for p in photos if "heldout/" in p]
                rel = heldout[0] if heldout else photos[0]
                ref_img = cv2.imread(str(bank_root / rel))
                if ref_img is not None:
                    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                    axes[i, 1].imshow(ref_img)
                    axes[i, 1].set_title(f"expected: {expected}\n({rel})", fontsize=8)
                    axes[i, 1].axis("off")
                else:
                    axes[i, 1].text(0.5, 0.5, f"can't load {rel}",
                                    transform=axes[i, 1].transAxes, ha="center")
                    axes[i, 1].axis("off")

        # Add nearest-centroid prediction
        emb = _arcface_for_bbox(app, frame, b)
        if emb is not None:
            all_celebs = list(centroid_lookup.keys())
            all_cents = np.stack([centroid_lookup[c] for c in all_celebs])
            sims = all_cents @ emb
            top3 = np.argsort(-sims)[:3]
            top3_text = "Top-3 nearest-centroid:\n" + "\n".join(
                f"  {all_celebs[idx]}: cos={sims[idx]:.3f}"
                for idx in top3
            )
            axes[i, 0].text(0, 1.02, top3_text, transform=axes[i, 0].transAxes,
                            fontsize=7, verticalalignment="bottom", family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / f"matcher_sanity__{d['source_episode']}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[png] {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--aug-root", type=Path, required=True)
    parser.add_argument("--face-labels-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-sources", type=int, default=6,
                        help="Number of source episodes to visualise")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(args.manifest)
    centroid_lookup = _build_centroid_lookup(manifest)

    # 1) Centroid heatmap
    print("[step 1/3] rendering centroid similarity heatmap ...")
    render_centroid_heatmap(manifest, args.output_dir / "centroid_similarity_iid_plus_ood.png")

    # 2) Source overlays
    jsons = sorted(args.face_labels_dir.glob("*.face_labels.json"))
    if not jsons:
        print(f"[ERR] no face_labels.json files under {args.face_labels_dir}", file=sys.stderr)
        return 2
    print(f"[step 2/3] found {len(jsons)} face_labels.json files")
    random.seed(args.seed)
    # Pick first one always for reproducibility, then random others.
    picked = [jsons[0]] + random.sample(jsons[1:], min(args.n_sources - 1, len(jsons) - 1))
    print(f"[step 2/3] rendering overlays for {len(picked)} sources ...")
    app = _build_arcface_app()

    summaries = []
    for jp in picked:
        print(f"  source: {jp.stem.replace('.face_labels', '')}")
        s = render_source_overlays(jp, args.aug_root, args.bank_root,
                                    centroid_lookup, args.output_dir, app)
        summaries.append(s)

    # 3) Matcher sanity grid for the first picked source
    print("[step 3/3] rendering matcher-sanity grid for first source ...")
    render_matcher_sanity(picked[0], args.aug_root, args.bank_root,
                          centroid_lookup, manifest, args.output_dir, app)

    # Summary
    print("\n=== SUMMARY ===")
    for s in summaries:
        print(f"  {s['source']}: matcher {s['matcher_acc']}")
        for m in s.get("matcher_detail") or []:
            if m is None:
                continue
            tag = "ok" if m["correct"] else "MISMATCH"
            print(f"    slot {m['slot']}: expected={m['expected']:<18s} "
                  f"winner={m['winner']:<18s} "
                  f"cos_expected={m['cos_expected']:+.3f} cos_winner={m['cos_winner']:+.3f} [{tag}]")

    print(f"\n[done] open the PNGs under {args.output_dir}/")
    print(f"       - centroid_similarity_iid_plus_ood.png   (cosine heatmap)")
    print(f"       - source_overlays/<source>__<frame>.png  (bbox + label overlays)")
    print(f"       - matcher_sanity__<source>.png            (bbox vs reference comparison)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
