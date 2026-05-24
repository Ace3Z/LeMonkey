#!/usr/bin/env bash
# One-time pre-download of the VL pairs dataset with max_workers=1.
# Workaround for the HF Hub multi-worker 429-rate-limit when the VL
# dataset's ~30k JPEGs are downloaded in parallel.
# Run with HF_TOKEN exported: bash eval_3/scripts/smolvla_cotrain/predownload_vl_dataset.sh
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lemonkey

echo "==> pre-downloading HBOrtiz/so101_eval3_cotrain_grounding (max_workers=1) ..."
python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="HBOrtiz/so101_eval3_cotrain_grounding",
    repo_type="dataset",
    token=os.environ.get("HF_TOKEN"),
    max_workers=1,
)
print(f"Done: {path}", flush=True)
PYEOF
echo "==> VL dataset ready."
