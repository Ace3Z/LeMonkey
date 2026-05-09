#!/usr/bin/env python
"""
Convert the per-identity manifest into JSONL training files for SmolVLM2 LoRA SFT.

Inputs:
    /Volumes/externalSSD/datasets/vggface2_hearfool/manifests/manifest.parquet

Outputs:
    /Volumes/externalSSD/datasets/vggface2_hearfool/manifests/train.jsonl
    /Volumes/externalSSD/datasets/vggface2_hearfool/manifests/val.jsonl

Each JSONL line:
    {
      "class_id":   "n000002",
      "name":       "A Fine Frenzy",
      "prompt":     "Who is shown in this photo?",
      "response":   "A Fine Frenzy",
      "image_path": "/Volumes/.../n000002/0001_01.jpg"
    }

The trainer takes each row and wraps it into the conversational format `trl`
SFTTrainer expects (image content block + text content block in a user turn,
plain text in the assistant turn).

Splits: hearfool's natural train/val split (identity-disjoint) is preserved.
Image sampling: per-identity cap (default 100 train / 30 val). Random with
fixed seed for reproducibility.

Prompt templates: identity-grounding only (no eval-style "place coke on ..."
phrasing — that's the VLA's job at downstream training time).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

DEFAULT_DATA_ROOT = Path("/Volumes/externalSSD/datasets/vggface2_hearfool")
# --manifest and --out-dir default to <data-root>/manifests/... resolved at parse time

PROMPT_TEMPLATES = [
    "Who is shown in this photo?",
    "Identify the person in this image.",
    "This is a photo of",
    "What is the name of the person pictured?",
    "Name the individual in the picture.",
]


def clean_name(raw: str) -> str:
    """VGGFace2 stores names like 'A_Fine_Frenzy' — convert to 'A Fine Frenzy'."""
    return raw.replace("_", " ").strip()


def make_row(class_id: str, name: str, img_path: str, rng: random.Random) -> dict:
    return {
        "class_id": class_id,
        "name": name,
        "prompt": rng.choice(PROMPT_TEMPLATES),
        "response": name,
        "image_path": img_path,
    }


def build_disjoint_splits(
    df: pd.DataFrame,
    n_train_per_id: int,
    n_val_per_id: int,
    max_identities: int | None,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Legacy mode: hearfool's identity-disjoint train/val partition.

    Use only as a zero-shot OOD probe — NOT as a primary val signal during SFT
    (the val task becomes "name an identity you've never seen" → impossible by
    construction → eval loss climbs while train loss falls).
    """
    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for split, n_per_id, bucket, sub_seed in [
        ("train", n_train_per_id, train_rows, seed),
        ("val", n_val_per_id, val_rows, seed + 1),
    ]:
        sub = df[df["source_split"] == split].copy()
        if max_identities is not None:
            sub = sub.head(max_identities)
        rng_local = random.Random(sub_seed)
        for _, row in sub.iterrows():
            paths = list(row["image_paths"])
            chosen = rng_local.sample(paths, k=min(n_per_id, len(paths)))
            name = clean_name(row["name"])
            for p in chosen:
                bucket.append(make_row(row["class_id"], name, p, rng_local))
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def build_per_identity_splits(
    df: pd.DataFrame,
    n_train_per_id: int,
    n_val_per_id: int,
    max_identities: int | None,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Recommended mode: same identities in train and val, different images.

    For each identity in the manifest (both hearfool train/ and val/ are used —
    no point wasting data), sample (n_train_per_id + n_val_per_id) images, take
    the first n_train_per_id for train and the next n_val_per_id for val.

    Val loss now measures "did the LoRA learn to name an identity it WAS
    trained on" — the actually-meaningful signal for SFT progress.
    """
    rng = random.Random(seed)
    sub = df.copy()
    if max_identities is not None:
        sub = sub.head(max_identities)

    train_rows, val_rows = [], []
    for _, row in sub.iterrows():
        paths = list(row["image_paths"])
        n_total = n_train_per_id + n_val_per_id
        chosen = rng.sample(paths, k=min(n_total, len(paths)))
        name = clean_name(row["name"])
        for p in chosen[:n_train_per_id]:
            train_rows.append(make_row(row["class_id"], name, p, rng))
        for p in chosen[n_train_per_id:n_train_per_id + n_val_per_id]:
            val_rows.append(make_row(row["class_id"], name, p, rng))
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Anchor for default --manifest and --out-dir paths")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="default: <data-root>/manifests/manifest.parquet")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <data-root>/manifests")
    parser.add_argument("--train-imgs-per-id", type=int, default=100)
    parser.add_argument("--val-imgs-per-id", type=int, default=30)
    parser.add_argument("--max-identities", type=int, default=None,
                        help="Cap on identities per split (use small value for smoke tests)")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix on output filenames (e.g. '.smoke' → train.smoke.jsonl)")
    parser.add_argument("--val-strategy", choices=["per-identity", "disjoint"],
                        default="per-identity",
                        help="per-identity (default): same IDs in train and val, different images "
                             "— measures whether LoRA learned trained identities. "
                             "disjoint (legacy): hearfool's identity-disjoint partition — "
                             "structurally measures impossible zero-shot identification, "
                             "use only as an OOD probe.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.manifest is None:
        args.manifest = args.data_root / "manifests/manifest.parquet"
    if args.out_dir is None:
        args.out_dir = args.data_root / "manifests"

    df = pd.read_parquet(args.manifest)
    print(f"[json] manifest: {len(df)} identities")
    print(f"[json] templates: {len(PROMPT_TEMPLATES)} prompts")
    print(f"[json] cap: {args.train_imgs_per_id} train imgs/id, {args.val_imgs_per_id} val imgs/id")
    print(f"[json] strategy: {args.val_strategy}")
    if args.max_identities is not None:
        print(f"[json] max-identities: {args.max_identities} per split")
    print()

    if args.val_strategy == "per-identity":
        train_rows, val_rows = build_per_identity_splits(
            df, args.train_imgs_per_id, args.val_imgs_per_id,
            args.max_identities, args.seed,
        )
    else:  # disjoint
        train_rows, val_rows = build_disjoint_splits(
            df, args.train_imgs_per_id, args.val_imgs_per_id,
            args.max_identities, args.seed,
        )

    suf = args.out_suffix
    train_out = args.out_dir / f"train{suf}.jsonl"
    val_out = args.out_dir / f"val{suf}.jsonl"
    write_jsonl(train_rows, train_out)
    write_jsonl(val_rows, val_out)

    print(f"[json] train: {len(train_rows)} examples → {train_out}")
    print(f"[json] val:   {len(val_rows)} examples → {val_out}")

    n_train_ids = len({r["class_id"] for r in train_rows})
    n_val_ids = len({r["class_id"] for r in val_rows})
    train_id_set = {r["class_id"] for r in train_rows}
    val_id_set = {r["class_id"] for r in val_rows}
    overlap = train_id_set & val_id_set
    if args.val_strategy == "disjoint":
        print(f"[json] train identities: {n_train_ids} | val identities: {n_val_ids} | "
              f"overlap: {len(overlap)} (must be 0)")
        assert len(overlap) == 0, "train/val identities must be disjoint"
    else:
        # per-identity: every val id MUST also be a train id (and image-paths disjoint)
        print(f"[json] train identities: {n_train_ids} | val identities: {n_val_ids} | "
              f"overlap: {len(overlap)} (must equal {n_val_ids} for per-identity strategy)")
        assert val_id_set.issubset(train_id_set), \
            "per-identity strategy: val identities must be a subset of train identities"
        train_paths = {r["image_path"] for r in train_rows}
        val_paths = {r["image_path"] for r in val_rows}
        path_overlap = train_paths & val_paths
        assert len(path_overlap) == 0, \
            f"per-identity strategy: image paths must be disjoint, got {len(path_overlap)} overlapping"

    print("\n[json] sample train rows (first 3):")
    for r in train_rows[:3]:
        print(f"  - {r['name']:25s}  prompt: {r['prompt']!r:50s}  img: ...{r['image_path'][-30:]}")


if __name__ == "__main__":
    main()
