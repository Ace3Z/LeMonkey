#!/usr/bin/env python3
"""Push the merged eval3 dataset to HBOrtiz/so101_eval3_broad on HF Hub.

Uses upload_large_folder (multi-part, resumable, retries on transient
failures) and the HBOrtiz token stored at
secrets/huggingface/token_hbortiz.

Usage:
    push_dataset_to_hf.py [--local datasets/eval3_merged]
                          [--repo  HBOrtiz/so101_eval3_broad]
                          [--token-file secrets/huggingface/token_hbortiz]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    """Push --local to HF dataset repo --repo via huggingface_hub.upload_large_folder using a token read from --token-file."""
    p = argparse.ArgumentParser()
    p.add_argument("--local", type=Path, default=Path("datasets/eval3_merged"),
                   help="Local dataset root to upload (default: datasets/eval3_merged).")
    p.add_argument("--repo", default="HBOrtiz/so101_eval3_broad",
                   help="HF dataset repo id to push to (default: HBOrtiz/so101_eval3_broad).")
    p.add_argument("--token-file", type=Path,
                   default=Path("secrets/huggingface/token_hbortiz"),
                   help="File containing the HF token (must begin with 'hf_').")
    args = p.parse_args()

    if not args.local.is_dir():
        print(f"[FATAL] {args.local} not found", file=sys.stderr)
        return 2

    token = args.token_file.read_text().strip()
    if not token.startswith("hf_"):
        print(f"[FATAL] token in {args.token_file} doesn't look right",
              file=sys.stderr)
        return 2

    api = HfApi(token=token)
    print(f"==> creating dataset repo {args.repo} (if it doesn't exist)",
          flush=True)
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=False)

    print(f"==> uploading {args.local} → {args.repo}", flush=True)
    print(f"    size: {sum(p.stat().st_size for p in args.local.rglob('*') if p.is_file())/2**30:.2f} GB",
          flush=True)

    # num_workers cap: HF rate-limits aggressively when 64 workers slam the
    # API for many small files (29k JPEGs hit ~143×429 + 49×503 retries with
    # default 64 workers, stalled for 4+ min). 8 workers stays under the limit.
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(args.local),
        ignore_patterns=["*.tmp", "__pycache__", "*.pyc"],
        num_workers=8,
    )

    print(f"\n==> done. dataset at https://huggingface.co/datasets/{args.repo}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
