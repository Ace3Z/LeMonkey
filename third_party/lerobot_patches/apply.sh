#!/usr/bin/env bash
# Apply third_party/lerobot patches needed to run with the dependency
# versions our environment actually installs.
#
# Idempotent: skips patches that look already-applied. Safe to re-run.
#
# Invoked automatically by the Eval 3 setup scripts (which install the
# vendored lerobot fork and need these patches):
#   - eval_3/scripts/training_vm/setup_pi05.sh
#   - eval_3/scripts/training_vm/setup_paligemma_warmstart.sh
#   - eval_3/scripts/smolvla_cotrain/setup_env.sh
# Note: scripts/setup_smolvla_env.sh installs lerobot==0.5.1 from PyPI
# and does NOT need to call this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="$HERE/../lerobot"
if [ ! -d "$LEROBOT_DIR/src/lerobot" ]; then
  echo "[FATAL] expected lerobot submodule at $LEROBOT_DIR; got missing" >&2
  exit 1
fi

cd "$LEROBOT_DIR"

apply_one() {
  local patch="$1"
  local marker_file="$2"
  local marker_pat="$3"
  if grep -q -- "$marker_pat" "$marker_file" 2>/dev/null; then
    echo "  [skip] $patch (already applied)"
    return 0
  fi
  if git apply --check "$HERE/$patch" 2>/dev/null; then
    git apply "$HERE/$patch"
    echo "  [ok]   $patch"
  else
    echo "  [warn] $patch did not apply cleanly; the file may have been edited locally"
    echo "         (run 'git status' in $LEROBOT_DIR and reconcile manually)"
  fi
}

echo "==> applying lerobot patches"
apply_one "01-groot-skip-strict.patch" \
  "src/lerobot/policies/groot/groot_n1.py" \
  "Upstream applies .@strict. here without .@dataclass"
apply_one "02-untagged-dataset-fallback.patch" \
  "src/lerobot/datasets/utils.py" \
  "dataset_version: expected=tagged dataset"
echo "==> done"
