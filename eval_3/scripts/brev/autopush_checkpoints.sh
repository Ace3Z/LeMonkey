#!/usr/bin/env bash
# Watcher: for every new checkpoint, push pretrained_model to HF as a
# step-tagged revision, run the attention probe on the LOCAL checkpoint
# (no re-download), then delete the whole local checkpoint dir.
#
# Disk-safety: Pi0.5 checkpoints are ~25 GB each (16.6 GB model + ~9 GB
# optimizer state). On a 97 GB disk with ~52 GB baseline (weights +
# dataset + OS) only one checkpoint fits at a time, so we delete each
# checkpoint immediately after push+probe (KEEP_LOCAL=0 default).
#
# Usage:
#   bash autopush_checkpoints.sh <output_dir> <hf_repo_id>
#
# Background:
#   nohup bash autopush_checkpoints.sh <dir> <repo> > ~/autopush.log 2>&1 &

set -uo pipefail

OUTPUT_DIR=${1:?usage: autopush_checkpoints.sh <output_dir> <hf_repo_id>}
HF_REPO=${2:?usage: autopush_checkpoints.sh <output_dir> <hf_repo_id>}
KEEP_LOCAL=${KEEP_LOCAL:-0}      # how many recent checkpoints to keep locally
# RUN_PROBE default off: the probe needs a standalone-PaliGemma loader, but
# the checkpoint is a full PI05Policy. Probe manually on demand instead.
RUN_PROBE=${RUN_PROBE:-0}

cd ~/LeMonkey
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pi05 2>/dev/null || conda activate lemonkey 2>/dev/null || true
set -a; source ~/LeMonkey/.env 2>/dev/null || source ~/.env; set +a

CKPT_BASE="$OUTPUT_DIR/checkpoints"
PUSHED_LOG="$OUTPUT_DIR/.autopush_pushed.txt"
PROBE_BASE="$OUTPUT_DIR/probes"
mkdir -p "$PROBE_BASE"
touch "$PUSHED_LOG"

echo "[autopush] watching $CKPT_BASE → $HF_REPO  (KEEP_LOCAL=$KEEP_LOCAL)"

while true; do
  if [ ! -d "$CKPT_BASE" ]; then
    sleep 30; continue
  fi
  for ckpt_dir in $(ls -1 "$CKPT_BASE" 2>/dev/null | grep -E '^[0-9]+$' | sort -n); do
    pretrained="$CKPT_BASE/$ckpt_dir/pretrained_model"
    [ -f "$pretrained/model.safetensors" ] || continue
    grep -qx "$ckpt_dir" "$PUSHED_LOG" && continue

    step_num=$((10#$ckpt_dir))
    rev="step-$step_num"
    echo "[autopush] $(date -u +%H:%M:%S) step=$step_num → $HF_REPO@$rev"

    # 1. Push pretrained_model to HF.
    python - <<PYTHON
import os, json
from huggingface_hub import HfApi, create_branch
api = HfApi(token=os.environ['HF_TOKEN'])
try:
    create_branch('$HF_REPO', branch='$rev', token=os.environ['HF_TOKEN'])
except Exception:
    pass
cfg = os.path.join('$pretrained', 'config.json')
if os.path.exists(cfg):
    d = json.loads(open(cfg).read())
    if d.get('type') != 'pi05':
        d = {'type': 'pi05', **{k: v for k, v in d.items() if k != 'type'}}
        open(cfg, 'w').write(json.dumps(d, indent=2))
api.upload_folder(folder_path='$pretrained', repo_id='$HF_REPO',
                  repo_type='model', revision='$rev',
                  commit_message='Pi0.5 + M2 + KLAL: step $step_num',
                  token=os.environ['HF_TOKEN'])
print('[autopush] uploaded $rev')
PYTHON

    # 2. Probe the LOCAL checkpoint (no re-download).
    if [ "$RUN_PROBE" = "1" ]; then
      PROBE_OUT="$PROBE_BASE/step_$step_num"
      mkdir -p "$PROBE_OUT"
      echo "[autopush] probing local checkpoint → $PROBE_OUT"
      python eval_3/scripts/attention_map_probe_paligemma.py \
        --model "$pretrained" \
        --image /tmp/probe_input.png \
        --layers 6 10 14 17 \
        --out "$PROBE_OUT" 2>&1 | tail -20 | tee "$PROBE_OUT/summary.txt" || \
        echo "[autopush] probe failed (non-fatal)"
    fi

    echo "$ckpt_dir" >> "$PUSHED_LOG"

    # 3. Delete old checkpoint dirs past KEEP_LOCAL.
    all=( $(ls -1 "$CKPT_BASE" | grep -E '^[0-9]+$' | sort -n) )
    n=${#all[@]}
    if (( n > KEEP_LOCAL )); then
      for d in "${all[@]:0:$((n - KEEP_LOCAL))}"; do
        echo "[autopush] deleting local $CKPT_BASE/$d"
        rm -rf "${CKPT_BASE:?}/$d"
      done
    fi
  done
  sleep 45
done
