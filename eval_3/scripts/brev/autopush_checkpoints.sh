#!/usr/bin/env bash
# Watcher: push every new `pretrained_model` checkpoint to HF as a step-tagged
# revision, then delete the local copy to free disk. Also runs a quick
# attention-probe on each new checkpoint so you can confirm training is
# actually working mid-run rather than waiting until the end.
#
# Usage:
#   bash eval_3/scripts/brev/autopush_checkpoints.sh \
#       <output_dir> <hf_repo_id>
#
# Example:
#   bash autopush_checkpoints.sh \
#       ~/outputs/train/pi05_track_E_m2_3celeb \
#       HBOrtiz/pi05_eval3_track_E_m2_mahbod
#
# Background usage:
#   nohup bash autopush_checkpoints.sh <dir> <repo> > ~/autopush.log 2>&1 &

set -euo pipefail

OUTPUT_DIR=${1:?usage: autopush_checkpoints.sh <output_dir> <hf_repo_id>}
HF_REPO=${2:?usage: autopush_checkpoints.sh <output_dir> <hf_repo_id>}
KEEP_LOCAL=${KEEP_LOCAL:-2}   # how many recent local checkpoints to keep
PROBE_FRAME_EPISODE=${PROBE_FRAME_EPISODE:-100}
PROBE_FRAME_INDEX=${PROBE_FRAME_INDEX:-10}

cd ~/LeMonkey
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pi05 2>/dev/null || conda activate lemonkey 2>/dev/null

set -a; source ~/LeMonkey/.env; set +a

CKPT_BASE="$OUTPUT_DIR/checkpoints"
PUSHED_LOG="$OUTPUT_DIR/.autopush_pushed.txt"
PROBE_BASE="$OUTPUT_DIR/probes"
mkdir -p "$PROBE_BASE"
touch "$PUSHED_LOG"

echo "[autopush] watching $CKPT_BASE  →  $HF_REPO  (keep $KEEP_LOCAL local)"

while true; do
  if [ ! -d "$CKPT_BASE" ]; then
    sleep 30; continue
  fi
  for ckpt_dir in $(ls -1 "$CKPT_BASE" 2>/dev/null | grep -E '^[0-9]+$' | sort -n); do
    step=$ckpt_dir
    pretrained="$CKPT_BASE/$ckpt_dir/pretrained_model"
    [ -f "$pretrained/model.safetensors" ] || continue
    grep -qx "$step" "$PUSHED_LOG" && continue

    echo "[autopush] step=$step → pushing to $HF_REPO@step-$step"
    rev="step-$step"

    # Push.
    python - <<PYTHON
import os, json
from huggingface_hub import HfApi, create_branch
api = HfApi(token=os.environ['HF_TOKEN'])
try:
    create_branch('$HF_REPO', branch='$rev', token=os.environ['HF_TOKEN'])
except Exception as e:
    pass  # branch may already exist
# Patch config.json with type=pi05 discriminator (mirror of the SmolVLA fix)
cfg_path = os.path.join('$pretrained', 'config.json')
if os.path.exists(cfg_path):
    cfg = json.loads(open(cfg_path).read())
    if cfg.get('type') != 'pi05':
        cfg = {'type': 'pi05', **{k: v for k, v in cfg.items() if k != 'type'}}
        open(cfg_path, 'w').write(json.dumps(cfg, indent=2))
        print(f'[autopush] patched type=pi05 into config.json')
api.upload_folder(
    folder_path='$pretrained', repo_id='$HF_REPO', repo_type='model',
    revision='$rev',
    commit_message=f"Pi0.5 + M2 + KLAL: step $step",
    token=os.environ['HF_TOKEN'],
)
print(f'[autopush] uploaded $HF_REPO@$rev')
PYTHON

    # Auto-validation probe (runs the Pi0.5 attention probe on this fresh ckpt).
    PROBE_OUT="$PROBE_BASE/step_$step"
    mkdir -p "$PROBE_OUT"
    echo "[autopush] running attention probe on step-$step → $PROBE_OUT"
    python eval_3/scripts/attention_map_probe_pi05.py \
      --repo "$HF_REPO" --revision "$rev" \
      --layers 6 10 14 17 \
      --episode "$PROBE_FRAME_EPISODE" --frame "$PROBE_FRAME_INDEX" \
      --out "$PROBE_OUT" 2>&1 | tail -15 | tee "$PROBE_OUT/summary.txt" || \
        echo "[autopush] probe failed; continuing"

    echo "$step" >> "$PUSHED_LOG"

    # Keep only the last KEEP_LOCAL checkpoint dirs locally.
    all_ckpts=( $(ls -1 "$CKPT_BASE" | grep -E '^[0-9]+$' | sort -n) )
    n_total=${#all_ckpts[@]}
    if (( n_total > KEEP_LOCAL )); then
      n_delete=$((n_total - KEEP_LOCAL))
      for to_delete in "${all_ckpts[@]:0:$n_delete}"; do
        echo "[autopush] deleting local $CKPT_BASE/$to_delete"
        rm -rf "$CKPT_BASE/$to_delete"
      done
    fi
  done
  sleep 60
done
