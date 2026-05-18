#!/usr/bin/env python3
"""Push the M2 ArcFace toolkit artifacts to a distinct HF dataset.

Target: `HBOrtiz/eval3_m2_arcface_toolkit` (dataset repo).
Distinct from the main training dataset `HBOrtiz/so101_eval3_track3_v3_baseline`
so M2-specific artifacts don't pollute that namespace.

Pushes:
  README.md                              — what this repo is, how to use it
  celeb_embeddings.json                  — manifest with per-photo paths and per-celeb centroids
  arcface_embeddings/<heldout|scraped>/<celeb>/<photo>.arcface.npy
                                         — 1,445 cached 512-D L2-norm float32 embeddings
  face_labels/<source_episode>.face_labels.json
                                         — 151 per-source-episode bbox tracks (positions only)

Usage (from project root):
  python eval_3/aug/push_m2_toolkit_to_hf.py [--what {embeddings,labels,all}]

Reads HF_TOKEN from .env. Will create the repo if missing.
The actual training-time bbox/identity join still happens via each
variant's `augmentation.json` (we don't bake celeb identities into
face_labels — keeps them reusable across variants).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import dotenv
from huggingface_hub import HfApi, create_repo


REPO_ID = "HBOrtiz/eval3_m2_arcface_toolkit"
REPO_TYPE = "dataset"

README = """\
# Eval 3 — M2 ArcFace toolkit data

Training-time-only artifacts for the M2 ArcFace cosine-distillation loss
(`dev/m2-arcface-toolkit` branch).

**Inference contract is unchanged.** This data is consumed only inside
the training-time dataloader. The deployed policy graph contains
`SmolVLA` alone — no ArcFace, no RetinaFace, no InsightFace.

## Contents

| Path | What | Size |
|---|---|---|
| `celeb_embeddings.json` | Manifest: per-photo paths + per-celeb 512-D centroid (mean over the celeb's photos, L2-normalized) | ~3 MB |
| `arcface_embeddings/heldout/<celeb>/<photo>.arcface.npy` | Per-photo ArcFace `buffalo_l` embeddings for the 3 IID celebs' workspace photos (14 photos × 3 celebs) | ~30 KB |
| `arcface_embeddings/scraped/<celeb>/<photo>.arcface.npy` | Per-photo embeddings for 189 OOD celebs from the web-scraped bank | ~2.8 MB |
| `face_labels/<source_episode>.face_labels.json` | Per-frame face bboxes for one representative variant per source episode (151 sources × ~250 KB each) | ~39 MB |

## How it was built

1. **ArcFace cache:** `eval_3/aug/cache_arcface_embeddings.py` ran InsightFace
   `buffalo_l` on every photo in `eval3_celebs/{heldout,scraped}/`,
   keeping the largest detected face. 1,445 embeddings, 0 failures.
   Leave-one-out top-1 identity recall on the full bank: **99.5 %**.
2. **Face labels:** `eval_3/aug/build_face_labels.py` grouped the 9,216
   augmented variants by source-episode prefix (151 unique sources, ~60
   variants each share camera trajectory because the camera is
   fixed), ran RetinaFace at `det_size=640` and `stride=5` (linear bbox
   interpolation between keyframes) on one representative camera1 video
   per source, and emitted per-frame bboxes sorted left-to-right by
   x-center.

## How to use at training time

```python
import json, numpy as np
from pathlib import Path

# Step 1: load the manifest
manifest = json.loads((repo_root / "celeb_embeddings.json").read_text())

# Step 2: per-celeb centroid for the alignment-loss target
def celeb_centroid(slug: str) -> np.ndarray:
    return np.asarray(manifest["celebs"][slug]["centroid"], dtype=np.float32)

# Step 3: per-source bboxes
def load_face_labels(source_episode: str) -> dict:
    p = repo_root / "face_labels" / f"{source_episode}.face_labels.json"
    return json.loads(p.read_text())

# Step 4: at each training step, join via the variant's augmentation.json:
#   - read augmentation.json[new_layout_camera_lmr] (e.g. "OLS")
#   - look up "OLS" → [obama@left, lecun@middle, swift@right]
#   - load face_labels[source].frames[frame_idx].bboxes (sorted left-to-right)
#   - per-bbox supervision target = ArcFace centroid of the celeb at that slot
```

## Identity-matching reliability

The matcher (RetinaFace bbox → ArcFace embedding → nearest centroid in
the 192-celeb gallery) was validated under leave-one-out:

| Bucket (photos/celeb) | LOO top-1 | Mean top1−top2 margin |
|---|---|---|
| 2 | 100 % | +0.36 |
| 3–4 | 100 % | +0.59 |
| 5–7 | 100 % | +0.60 |
| 8–12 | 99.2 % | +0.60 |
| 13+ | 100 % | +0.68 |

The 7 misses (out of 1,445) cluster on one celeb (`oier_mees`) whose
scraped bank is structurally broken (intra-celeb own-photo cosines
0.05–0.13). Flag him before any reliance on OOD coverage.

## Branch & docs

- Source branch: [`dev/m2-arcface-toolkit`](https://github.com/Ace3Z/LeMonkey/tree/dev/m2-arcface-toolkit)
- Experiment log: `docs/experiments/2026-05-19_m2_data_foundation.md`
- Validation report: `docs/report/2026-05-18_m2_arcface_validation.md`
"""


def stage_embeddings(bank_root: Path, manifest_path: Path, stage_root: Path) -> int:
    """Copy .arcface.npy files preserving heldout/scraped/<celeb>/ structure.
    Returns the count of files staged."""
    import json
    manifest = json.loads(manifest_path.read_text())
    stage_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for celeb, info in manifest["celebs"].items():
        for rel_photo, rel_npy in info["photos"].items():
            src = bank_root / rel_npy
            if not src.exists():
                print(f"[WARN] missing embedding: expected={src}, "
                      f"got=missing, fallback=skip-and-warn", flush=True)
                continue
            dest = stage_root / "arcface_embeddings" / rel_npy
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            n += 1
    # Also copy the manifest itself
    shutil.copy2(manifest_path, stage_root / "celeb_embeddings.json")
    return n


def stage_face_labels(labels_dir: Path, stage_root: Path) -> int:
    """Copy face_labels JSONs into stage/face_labels/."""
    dest = stage_root / "face_labels"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for j in sorted(labels_dir.glob("*.face_labels.json")):
        shutil.copy2(j, dest / j.name)
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", type=Path, default=Path.home() / "Downloads/eval3_celebs")
    parser.add_argument("--manifest", type=Path, default=Path("eval_3/aug/stats/celeb_embeddings.json"))
    parser.add_argument("--face-labels-dir", type=Path, default=Path("eval_3/aug/stats/face_labels"))
    parser.add_argument("--what", choices=["embeddings", "labels", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[ERR] HF_TOKEN missing in .env", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    if not args.dry_run:
        try:
            create_repo(REPO_ID, repo_type=REPO_TYPE, token=token, exist_ok=True, private=False)
            print(f"[hf] repo ready: https://huggingface.co/datasets/{REPO_ID}")
        except Exception as e:
            print(f"[ERR] create_repo failed: {e}", file=sys.stderr)
            return 1

    with tempfile.TemporaryDirectory(prefix="m2_toolkit_stage_") as td:
        stage = Path(td)
        (stage / "README.md").write_text(README)

        n_emb = n_lbl = 0
        if args.what in ("embeddings", "all"):
            n_emb = stage_embeddings(args.bank_root, args.manifest, stage)
            print(f"[stage] {n_emb} arcface embeddings + manifest")
        if args.what in ("labels", "all"):
            if args.face_labels_dir.is_dir() and any(args.face_labels_dir.iterdir()):
                n_lbl = stage_face_labels(args.face_labels_dir, stage)
                print(f"[stage] {n_lbl} face_labels JSONs")
            else:
                print(f"[WARN] face_labels: expected dir={args.face_labels_dir} with content, "
                      f"got=empty/missing, fallback=skip-labels-this-run", flush=True)

        # Show what's staged
        total_files = sum(1 for _ in stage.rglob("*") if _.is_file())
        total_size = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file())
        print(f"[stage] total {total_files} files, {total_size/1e6:.1f} MB at {stage}")

        if args.dry_run:
            print("[dry-run] not pushing")
            for p in sorted(stage.rglob("*"))[:30]:
                print(f"  {p.relative_to(stage)}")
            return 0

        # Push
        print(f"[hf] uploading folder → https://huggingface.co/datasets/{REPO_ID}")
        api.upload_folder(
            folder_path=str(stage),
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"add M2 toolkit data: {n_emb} embeddings, {n_lbl} face_labels",
            token=token,
        )
        print(f"[done] https://huggingface.co/datasets/{REPO_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
