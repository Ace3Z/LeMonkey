#!/usr/bin/env python3
"""Pre-compute celeb-vs-celeb ArcFace cosine confusion matrix.

For each pair (a, b) of celebs in Mahbod's manifest, compute the cosine
between their centroids. Output a dense 192×192 matrix + a per-celeb
"top-K most confusable" list.

Why we need this (face-binding focus):
- Enhancement B-3 (hard-negative oversampling) needs to identify which
  variants have visually-confusable distractors visible.
- Enhancement B-5 (curriculum) needs a notion of "difficulty" per celeb.
- At training-data-prep time, we look up: "for target celeb X, which other
  celebs are visually similar (high centroid cosine)?" — those are the
  confusers we want to oversample variants for.

Run ONCE after pulling celeb_embeddings.json — output is static and
reusable across all data audits.

Per CLAUDE.md §5: no silent fallbacks (broken centroids logged).
Per CLAUDE.md §7: triple-source defaults.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Confusable threshold: two celebs whose centroid cosine > this are flagged.
# Triple-source rationale:
#  - For different identities on clean LFW: centroid cosine usually < 0.3.
#  - Same identity (different photos) usually 0.5+.
#  - 0.30 chosen as "above noise floor for different celebs, well below
#    same-identity threshold" — flags genuinely visually-similar pairs
#    without false positives.
DEFAULT_CONFUSABLE_THRESHOLD = 0.30

# Top-K most confusable celebs per target (for hard-neg mining lookup).
DEFAULT_TOP_K = 5

# Known broken centroid (from Mahbod's M2 audit) — flag but don't drop here.
BROKEN_SLUGS = {"oier_mees"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb-manifest", type=Path,
                        default=Path("data/arcface_toolkit/celeb_embeddings.json"))
    parser.add_argument("--output-matrix", type=Path,
                        default=Path("eval_3/scripts/track_2/confusion_matrix.npy"))
    parser.add_argument("--output-slugs", type=Path,
                        default=Path("eval_3/scripts/track_2/confusion_slugs.json"))
    parser.add_argument("--output-topk", type=Path,
                        default=Path("eval_3/scripts/track_2/confusable_topk.json"))
    parser.add_argument("--threshold", type=float,
                        default=DEFAULT_CONFUSABLE_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    if not args.celeb_manifest.is_file():
        print(f"[ERR] manifest not found: {args.celeb_manifest}", file=sys.stderr)
        return 2

    manifest = json.loads(args.celeb_manifest.read_text())
    celebs = manifest["celebs"]
    slugs = sorted(celebs.keys())
    n = len(slugs)
    print(f"[info] loaded {n} celeb centroids")

    # Stack centroids into a matrix.
    centroids = np.zeros((n, 512), dtype=np.float32)
    broken_indices = []
    null_indices = []
    for i, slug in enumerate(slugs):
        c = celebs[slug].get("centroid")
        if c is None:
            print(f"[WARN] slug={slug!r}: expected centroid, got None, fallback=zero-row",
                  flush=True)
            null_indices.append(i)
            continue  # zero row
        c = np.asarray(c, dtype=np.float32)
        norm = float(np.linalg.norm(c))
        if norm < 1e-6:
            print(f"[WARN] slug={slug!r}: expected unit norm, got {norm:.2e}, "
                  f"fallback=zero-row", flush=True)
            null_indices.append(i)
            continue
        centroids[i] = c / norm  # safety renormalize
        if slug in BROKEN_SLUGS:
            broken_indices.append(i)

    # All-pairs cosine (since centroids are L2-normed, just inner product).
    cos_matrix = centroids @ centroids.T  # (n, n) float32, diagonal = 1.0
    np.fill_diagonal(cos_matrix, -1.0)    # mask self → no self in top-K

    # Save matrix + slug ordering.
    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_matrix, cos_matrix)
    args.output_slugs.write_text(json.dumps({
        "slugs": slugs,
        "matrix_shape": list(cos_matrix.shape),
        "broken_slug_indices": broken_indices,
        "broken_slugs": [slugs[i] for i in broken_indices],
        "null_slug_indices": null_indices,
        "null_slugs": [slugs[i] for i in null_indices],
        "threshold_used": args.threshold,
    }, indent=2))
    print(f"[done] matrix {cos_matrix.shape} → {args.output_matrix}")
    print(f"[done] slug ordering → {args.output_slugs}")

    # Per-celeb top-K most confusable.
    topk = {}
    n_with_confuser = 0
    for i, slug in enumerate(slugs):
        if i in null_indices:
            topk[slug] = {"top_k": [], "n_above_threshold": 0,
                          "note": "null centroid"}
            continue
        # Sort descending by cosine (skip self via -1.0 diagonal).
        order = np.argsort(-cos_matrix[i])
        top_rows = order[:args.top_k]
        entry = []
        for j in top_rows:
            entry.append({
                "slug": slugs[int(j)],
                "cos": float(cos_matrix[i, int(j)]),
            })
        n_above = int((cos_matrix[i] > args.threshold).sum())
        if n_above > 0:
            n_with_confuser += 1
        topk[slug] = {
            "top_k": entry,
            "n_above_threshold": n_above,
        }
    args.output_topk.write_text(json.dumps(topk, indent=2))
    print(f"[done] top-{args.top_k} confusables → {args.output_topk}")
    print(f"[summary]")
    print(f"  celebs with ≥1 confuser above {args.threshold}: "
          f"{n_with_confuser}/{n} ({n_with_confuser/n*100:.1f}%)")

    # Bird's-eye stats.
    upper = cos_matrix[np.triu_indices(n, k=1)]
    print(f"  pairwise cosine: mean={upper.mean():+.3f}  "
          f"std={upper.std():.3f}  "
          f"p50={np.percentile(upper, 50):+.3f}  "
          f"p90={np.percentile(upper, 90):+.3f}  "
          f"p99={np.percentile(upper, 99):+.3f}")
    print(f"  pairs above threshold {args.threshold}: "
          f"{(upper > args.threshold).sum()}/{len(upper)} "
          f"({(upper > args.threshold).mean()*100:.2f}%)")

    # Print 5 worst confusable pairs.
    flat_indices = np.argsort(-upper)[:5]
    print(f"  top-5 most-confusable pairs:")
    for k in flat_indices:
        # Reverse-engineer (i, j) from flat upper-triangle index k.
        # Slow but only 5 iterations.
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                if cnt == k:
                    print(f"    {slugs[i]:30s} vs {slugs[j]:30s}: cos={upper[k]:+.3f}")
                    break
                cnt += 1
            else:
                continue
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
