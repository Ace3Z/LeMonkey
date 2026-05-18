#!/usr/bin/env python3
"""OOD validation for the ArcFace celeb bank.

We can't validate OOD celebs *in the workspace* — the camera1 videos
only show the 3 IID celebs (Swift/Obama/LeCun). The relevant
generalisation question is: **if Day-3 / Day-4 eval introduces a
celebrity we've only seen via scraped/heldout photos, can the bank
identify them?**

Leave-one-out (LOO) test:
  For each celeb with >=2 photos, for each of that celeb's photos:
    - Recompute the centroid WITHOUT that photo.
    - Compute cosine of the held-out photo's embedding against every
      celeb's LOO centroid (own celeb uses LOO; others use full).
    - Predict the nearest-centroid celeb.
    - Score correct if predicted == own celeb.

Aggregate metrics:
  - LOO top-1 accuracy across the full bank (~1400 photo trials).
  - LOO top-1 accuracy broken down by photos-per-celeb bucket
    (celebs with few photos suffer more from removing one — small bank,
     less stable centroid).
  - Per-celeb confusion list for misclassifications.

Visual artifacts:
  - ood_loo_summary.png — bar chart of per-bucket LOO accuracy + scatter
    of confidence margin (top1 - top2 cosine) per photo.
  - ood_sample_grid.png — 12 random OOD celebs: for each, show one
    of their photos + the top-3 nearest-centroid predictions
    (with that photo removed from its own celeb's centroid).

Usage:

    python eval_3/aug/dbg/dbg_ood_matcher.py \
        --bank-root ~/Downloads/eval3_celebs \
        --manifest eval_3/aug/stats/celeb_embeddings.json \
        --output-dir eval_3/aug/stats/face_labels_dbg

Requires matplotlib + cv2 + numpy. No extra GPU / model calls — the
manifest cache already has all embeddings.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


IID_CELEBS = {"taylor_swift", "barack_obama", "yann_lecun"}


def _load_manifest(p: Path) -> dict:
    return json.loads(p.read_text())


def _build_full_centroids(manifest: dict, bank_root: Path) -> dict[str, np.ndarray]:
    out = {}
    for celeb, info in manifest["celebs"].items():
        if info["centroid"]:
            out[celeb] = np.asarray(info["centroid"], dtype=np.float32)
    return out


def _photo_embeddings(manifest: dict, bank_root: Path, celeb: str) -> dict[str, np.ndarray]:
    info = manifest["celebs"][celeb]
    out = {}
    for rel, npy_rel in info["photos"].items():
        out[rel] = np.load(bank_root / npy_rel)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-grid", type=int, default=12,
                        help="Number of OOD celebs to show in the visual grid")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(args.manifest)
    full_centroids = _build_full_centroids(manifest, args.bank_root)
    all_celebs = list(full_centroids.keys())
    cent_matrix = np.stack([full_centroids[c] for c in all_celebs])
    celeb_to_idx = {c: i for i, c in enumerate(all_celebs)}

    # ---- LOO eval over the entire bank ----
    print(f"[loo] running leave-one-out top-1 over {len(all_celebs)} celebs ...")
    rows = []
    for celeb in all_celebs:
        info = manifest["celebs"][celeb]
        if info["n_photos"] < 2:
            continue
        embs = {rel: np.load(args.bank_root / npy_rel)
                for rel, npy_rel in info["photos"].items()}
        own_idx = celeb_to_idx[celeb]
        for rel, e in embs.items():
            # Recompute centroid WITHOUT this photo.
            others = np.stack([emb for r, emb in embs.items() if r != rel])
            loo_cent = others.mean(axis=0)
            loo_cent = loo_cent / max(float(np.linalg.norm(loo_cent)), 1e-6)
            # Patch the centroid matrix in-place for this query
            cent_matrix[own_idx] = loo_cent
            sims = cent_matrix @ e
            top_idx = int(np.argmax(sims))
            top2_idx = int(np.argsort(-sims)[1])
            rows.append({
                "celeb": celeb,
                "photo": rel,
                "n_photos": info["n_photos"],
                "is_iid": celeb in IID_CELEBS,
                "top1_celeb": all_celebs[top_idx],
                "top1_cos": float(sims[top_idx]),
                "top2_celeb": all_celebs[top2_idx],
                "top2_cos": float(sims[top2_idx]),
                "own_cos_loo": float(sims[own_idx]),
                "correct": all_celebs[top_idx] == celeb,
            })
            # restore the centroid matrix
            cent_matrix[own_idx] = full_centroids[celeb]

    # Aggregate
    n_iid = sum(1 for r in rows if r["is_iid"])
    n_ood = sum(1 for r in rows if not r["is_iid"])
    n_iid_correct = sum(1 for r in rows if r["is_iid"] and r["correct"])
    n_ood_correct = sum(1 for r in rows if not r["is_iid"] and r["correct"])

    print(f"\n[loo] overall: {sum(r['correct'] for r in rows)}/{len(rows)} "
          f"= {100*sum(r['correct'] for r in rows)/len(rows):.1f}%")
    print(f"[loo] IID:     {n_iid_correct}/{n_iid} = {100*n_iid_correct/n_iid:.1f}%")
    print(f"[loo] OOD:     {n_ood_correct}/{n_ood} = {100*n_ood_correct/n_ood:.1f}%")

    # By n_photos bucket (the small-bank effect)
    buckets = [(2, 2), (3, 4), (5, 7), (8, 12), (13, 100)]
    bucket_stats: list[tuple[str, int, int, float]] = []
    print("\n[loo] by photos-per-celeb bucket:")
    print(f"  {'bucket':>10s}  {'n_photos':>10s}  {'correct':>8s}  {'acc':>6s}  {'mean_top1':>9s}  {'mean_margin':>11s}")
    for lo, hi in buckets:
        sub = [r for r in rows if lo <= r["n_photos"] <= hi]
        if not sub:
            continue
        ncor = sum(1 for r in sub if r["correct"])
        mean_top1 = np.mean([r["top1_cos"] for r in sub])
        mean_margin = np.mean([r["top1_cos"] - r["top2_cos"] for r in sub])
        print(f"  {f'{lo}-{hi}':>10s}  {len(sub):>10d}  {ncor:>8d}  "
              f"{100*ncor/len(sub):>5.1f}%  {mean_top1:>+8.3f}  {mean_margin:>+11.3f}")
        bucket_stats.append((f"{lo}-{hi}", len(sub), ncor, 100 * ncor / len(sub)))

    # Misclassification top examples
    misses = [r for r in rows if not r["correct"] and not r["is_iid"]]
    misses.sort(key=lambda r: r["top1_cos"] - r["top2_cos"], reverse=True)
    print(f"\n[loo] {len(misses)} OOD misclassifications. Top-10 highest-confidence misses:")
    for r in misses[:10]:
        print(f"  {r['celeb']:>22s} ({r['photo'].split('/')[-1]:<40s}) "
              f"→ {r['top1_celeb']:>22s}  cos={r['top1_cos']:+.3f}  own_cos={r['own_cos_loo']:+.3f}")

    # Write rows to disk for any further audit
    out_rows = args.output_dir / "ood_loo_rows.json"
    out_rows.write_text(json.dumps(rows, indent=2))
    print(f"\n[json] {out_rows}")

    # ---- Visuals ----
    import cv2
    import matplotlib.pyplot as plt

    # 1) Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # bar of per-bucket acc
    if bucket_stats:
        names = [b[0] for b in bucket_stats]
        accs = [b[3] for b in bucket_stats]
        ns = [b[1] for b in bucket_stats]
        bars = axes[0].bar(names, accs, color="#3a6ea0")
        for bar, n in zip(bars, ns):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"n={n}", ha="center", fontsize=9)
        axes[0].set_ylim(0, 105)
        axes[0].set_ylabel("Top-1 LOO accuracy (%)")
        axes[0].set_xlabel("Photos per celeb (bucket)")
        axes[0].set_title("LOO top-1 accuracy by bank size per celeb")
        axes[0].axhline(100, color="gray", linewidth=0.5, linestyle="--")
    # scatter of top1 - top2 margin per photo, by correct/incorrect
    cor = [(r["top1_cos"], r["top1_cos"] - r["top2_cos"]) for r in rows if r["correct"]]
    inc = [(r["top1_cos"], r["top1_cos"] - r["top2_cos"]) for r in rows if not r["correct"]]
    if cor:
        axes[1].scatter([x for x, _ in cor], [y for _, y in cor],
                        s=8, alpha=0.4, color="#1e7a30", label=f"correct (n={len(cor)})")
    if inc:
        axes[1].scatter([x for x, _ in inc], [y for _, y in inc],
                        s=18, alpha=0.9, color="#b22a2a", label=f"INCORRECT (n={len(inc)})", marker="x")
    axes[1].set_xlabel("Top-1 cosine")
    axes[1].set_ylabel("Top-1 − Top-2 margin")
    axes[1].set_title("Per-photo confidence vs margin (LOO)")
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("OOD identity-matcher: leave-one-out validation on the celeb bank", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = args.output_dir / "ood_loo_summary.png"
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"[png] {out1}")

    # 2) Sample grid: pick N OOD celebs, show one photo + top-3 predictions
    random.seed(args.seed)
    ood_pool = [c for c in all_celebs if c not in IID_CELEBS and manifest["celebs"][c]["n_photos"] >= 3]
    picked = random.sample(ood_pool, min(args.n_grid, len(ood_pool)))

    n_cols = 4
    n_rows = (len(picked) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i, celeb in enumerate(picked):
        ax = axes[i // n_cols, i % n_cols]
        info = manifest["celebs"][celeb]
        rels = list(info["photos"].keys())
        # use the FIRST photo as query, build LOO centroid without it
        rel = rels[0]
        e = np.load(args.bank_root / info["photos"][rel])
        others = np.stack([np.load(args.bank_root / info["photos"][r]) for r in rels[1:]])
        loo_cent = others.mean(axis=0)
        loo_cent = loo_cent / max(float(np.linalg.norm(loo_cent)), 1e-6)
        own_idx = celeb_to_idx[celeb]
        cent_matrix[own_idx] = loo_cent
        sims = cent_matrix @ e
        top3 = np.argsort(-sims)[:3]
        cent_matrix[own_idx] = full_centroids[celeb]

        # render the photo
        img = cv2.imread(str(args.bank_root / rel))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        ax.set_title(f"actual: {celeb}\nphoto: {rel.split('/')[-1]}\nn_photos={info['n_photos']}",
                     fontsize=8)
        # Below the image: top-3 predictions
        pred_text = "Top-3 nearest-centroid (LOO):"
        for k, idx in enumerate(top3):
            mark = "✓" if all_celebs[idx] == celeb else "✗"
            pred_text += f"\n  {k+1}. {mark} {all_celebs[idx]}  cos={sims[idx]:+.3f}"
        ax.text(0.0, -0.02, pred_text, transform=ax.transAxes,
                fontsize=7, family="monospace", verticalalignment="top",
                color="#1e7a30" if all_celebs[top3[0]] == celeb else "#b22a2a")
        ax.axis("off")

    # Hide unused subplots
    for j in range(len(picked), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.suptitle(f"OOD nearest-centroid sanity (leave-one-out, n={len(picked)})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = args.output_dir / "ood_sample_grid.png"
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"[png] {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
