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
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "manifests/manifest.parquet"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "manifests"

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


def emit_rows_for_identity(
    class_id: str,
    name: str,
    image_paths: list[str],
    n_per_id: int,
    rng: random.Random,
) -> list[dict]:
    chosen = rng.sample(image_paths, k=min(n_per_id, len(image_paths)))
    rows = []
    for img_path in chosen:
        prompt = rng.choice(PROMPT_TEMPLATES)
        rows.append({
            "class_id": class_id,
            "name": name,
            "prompt": prompt,
            "response": name,
            "image_path": img_path,
        })
    return rows


def build_split(
    df: pd.DataFrame,
    split: str,
    n_per_id: int,
    max_identities: int | None,
    seed: int,
) -> list[dict]:
    sub = df[df["source_split"] == split].copy()
    if max_identities is not None:
        sub = sub.head(max_identities)

    rng = random.Random(seed)
    rows: list[dict] = []
    for _, row in sub.iterrows():
        rows.extend(emit_rows_for_identity(
            class_id=row["class_id"],
            name=clean_name(row["name"]),
            image_paths=list(row["image_paths"]),
            n_per_id=n_per_id,
            rng=rng,
        ))
    rng.shuffle(rows)
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-imgs-per-id", type=int, default=100)
    parser.add_argument("--val-imgs-per-id", type=int, default=30)
    parser.add_argument("--max-identities", type=int, default=None,
                        help="Cap on identities per split (use small value for smoke tests)")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix on output filenames (e.g. '.smoke' → train.smoke.jsonl)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.manifest)
    print(f"[json] manifest: {len(df)} identities")
    print(f"[json] templates: {len(PROMPT_TEMPLATES)} prompts")
    print(f"[json] cap: {args.train_imgs_per_id} train imgs/id, {args.val_imgs_per_id} val imgs/id")
    if args.max_identities is not None:
        print(f"[json] max-identities: {args.max_identities} per split")
    print()

    train_rows = build_split(df, "train", args.train_imgs_per_id,
                             args.max_identities, args.seed)
    val_rows = build_split(df, "val", args.val_imgs_per_id,
                           args.max_identities, args.seed + 1)

    suf = args.out_suffix
    train_out = args.out_dir / f"train{suf}.jsonl"
    val_out = args.out_dir / f"val{suf}.jsonl"
    write_jsonl(train_rows, train_out)
    write_jsonl(val_rows, val_out)

    print(f"[json] train: {len(train_rows)} examples → {train_out}")
    print(f"[json] val:   {len(val_rows)} examples → {val_out}")

    n_train_ids = len({r["class_id"] for r in train_rows})
    n_val_ids = len({r["class_id"] for r in val_rows})
    overlap = {r["class_id"] for r in train_rows} & {r["class_id"] for r in val_rows}
    print(f"[json] train identities: {n_train_ids} | val identities: {n_val_ids} | "
          f"overlap: {len(overlap)} (must be 0)")
    assert len(overlap) == 0, "train/val identities must be disjoint"

    print("\n[json] sample train rows (first 3):")
    for r in train_rows[:3]:
        print(f"  - {r['name']:25s}  prompt: {r['prompt']!r:50s}  img: ...{r['image_path'][-30:]}")


if __name__ == "__main__":
    main()
