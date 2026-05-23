#!/usr/bin/env bash
# Run the full Pi0.5 VL cotrain data-side pipeline when the bboxes arrive.
#
# Sequence: schema verify → audit (~1h) → keep_list+weights (~5min)
#
# Usage:
#   ./run_audit_pipeline.sh <path-or-hf-repo of the bbox parquet>
#
# Examples:
#   ./run_audit_pipeline.sh /path/to/roham_bboxes.parquet
#   ./run_audit_pipeline.sh HBOrtiz/eval3_200celeb_bboxes        # HF dataset
#
# Per: bail on errors; no silent fallback.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <bbox-parquet-path-or-HF-repo>" >&2
    exit 2
fi

INPUT="$1"
HERE="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(dirname "$(dirname "$HERE")")"
cd "$REPO_ROOT"

WORK_DIR="${WORK_DIR:-$HOME/.cache/pi05_vl_cotrain_audit}"
mkdir -p "$WORK_DIR"

# Step 0: resolve INPUT to a local parquet path.
if [ -f "$INPUT" ]; then
    BBOX_PARQUET="$INPUT"
    echo "==> using local parquet: $BBOX_PARQUET"
else
    echo "==> downloading from HF: $INPUT"
    LOCAL_DIR="$WORK_DIR/$(echo "$INPUT" | tr / _)"
    mkdir -p "$LOCAL_DIR"
    python3 -c "
from huggingface_hub import snapshot_download
import os
p = snapshot_download(
    repo_id='$INPUT',
    repo_type='dataset',
    allow_patterns=['*.parquet'],
    local_dir='$LOCAL_DIR',
    token=os.environ.get('HF_TOKEN'),
)
print(p)
"
    # Find the parquet inside.
    BBOX_PARQUET=$(find "$LOCAL_DIR" -name '*.parquet' -type f | head -1)
    if [ -z "$BBOX_PARQUET" ]; then
        echo "[ERR] no parquet found in downloaded repo $INPUT" >&2
        exit 1
    fi
    echo "==> resolved to: $BBOX_PARQUET"
fi

# Step 1: verify schema (~5 sec).
NORMALIZED_PARQUET="$WORK_DIR/bboxes_normalized.parquet"
echo
echo "==> Step 1/3: verify schema"
python3 eval_3/scripts/pi05_vl_cotrain/verify_bbox_schema.py \
    --bbox-parquet "$BBOX_PARQUET" \
    --write-normalized "$NORMALIZED_PARQUET"

# Step 2: ArcFace audit (~1 h on CPU; ~10 min on GPU).
AUDIT_PARQUET="eval_3/scripts/pi05_vl_cotrain/audit_200celeb.parquet"
echo
echo "==> Step 2/3: ArcFace audit (this is the long step — go get coffee)"
python3 eval_3/scripts/pi05_vl_cotrain/arcface_audit_200celeb.py \
    --bbox-parquet "$NORMALIZED_PARQUET" \
    --celeb-manifest data/arcface_toolkit/celeb_embeddings.json \
    --output "$AUDIT_PARQUET"

# Step 3: build keep_list + sample weights (~5 min).
echo
echo "==> Step 3/3: build keep_list + sample weights"
python3 eval_3/scripts/pi05_vl_cotrain/build_keep_list_and_weights.py \
    --audit-parquet "$AUDIT_PARQUET" \
    --output-dir eval_3/scripts/pi05_vl_cotrain/

# Summary.
echo
echo "==> Done. Artifacts ready for Brev launch:"
echo "    - $AUDIT_PARQUET"
echo "    - eval_3/scripts/pi05_vl_cotrain/keep_episodes.txt"
echo "    - eval_3/scripts/pi05_vl_cotrain/hardneg_weights.npy"
echo "    - eval_3/scripts/pi05_vl_cotrain/build_keep_list_summary.json"
echo
echo "==> Next: launch Pi0.5 VL cotrain smoke test on the training VM:"
echo "    STEPS=200 bash eval_3/scripts/brev/train_pi05_vl_cotrain.sh"
