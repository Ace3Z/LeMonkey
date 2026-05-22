#!/usr/bin/env bash
# Prepare a Track-B-ready local copy of the augmented eval3 dataset.
#
# Materialises HBOrtiz/so101_eval3_cotrain locally (camera1 + meta
# + data only), swaps in the corrected stats.json from
# HBOrtiz/so101_eval3_track3_v3_pi05, and applies the five schema renames
# Pi0.5 needs (drop reference, rename camera1 -> right_wrist_0_rgb, patch
# total_frames). Idempotent — safe to re-run.
#
# See eval_3/tracks/TRACK_B_DEVBOX_HANDOVER.md §1 for the full rationale.

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-$HOME/datasets/eval3_track3_v3_pi05}"
BASELINE_REPO="${BASELINE_REPO:-HBOrtiz/so101_eval3_cotrain}"
PI05_REPO="${PI05_REPO:-HBOrtiz/so101_eval3_track3_v3_pi05}"

# Resolve HF token from common locations
if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ -f "$HOME/LeMonkey/secrets/huggingface/token_hbortiz" ]]; then
        HF_TOKEN="$(cat "$HOME/LeMonkey/secrets/huggingface/token_hbortiz")"
    elif [[ -f "$HOME/ETH_Uni/LeMonkey/secrets/huggingface/token_hbortiz" ]]; then
        HF_TOKEN="$(cat "$HOME/ETH_Uni/LeMonkey/secrets/huggingface/token_hbortiz")"
    else
        echo "[WARN] HF_TOKEN not in env: expected=token from secrets/huggingface/token_hbortiz, got=missing, fallback=anonymous-HF-which-likely-fails-for-private-or-rate-limited-repos" >&2
    fi
fi
export HF_TOKEN

# Activate conda env if not already in one
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV:-}" != "lemonkey" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate lemonkey
    else
        echo "[WARN] conda.sh not at \$HOME/miniconda3/etc/profile.d/conda.sh: expected=lemonkey conda env, got=no-conda, fallback=hope-python-on-PATH-is-the-right-one" >&2
    fi
fi

echo "==> prepare_dataset_track_B"
echo "    target root  : $DATASET_ROOT"
echo "    baseline repo: $BASELINE_REPO"
echo "    pi05 repo    : $PI05_REPO"
echo

mkdir -p "$DATASET_ROOT"

# ------------------------------------------------------------------------------
# Step 1 — snapshot_download baseline (camera1 + meta + data only).
# ------------------------------------------------------------------------------
echo "==> [1/5] snapshot_download (camera1 + meta + data, ~5 min, ~13 GB)"
python - <<PY
import os, pathlib
from huggingface_hub import snapshot_download
local = snapshot_download(
    repo_id="$BASELINE_REPO",
    repo_type="dataset",
    local_dir=os.environ.get("DATASET_ROOT", "$DATASET_ROOT"),
    token=os.environ.get("HF_TOKEN") or None,
    allow_patterns=[
        "meta/*",
        "data/*",
        "videos/observation.images.camera1/**",
        ".gitattributes",
    ],
    max_workers=4,
)
print("snapshot_download OK:", local)
PY

# ------------------------------------------------------------------------------
# Step 2 — pull corrected stats.json from the _pi05 repo (LFS-aware).
# ------------------------------------------------------------------------------
echo "==> [2/5] fetch corrected stats.json from $PI05_REPO"
python - <<PY
import os, shutil
from huggingface_hub import hf_hub_download
src = hf_hub_download(
    repo_id="$PI05_REPO",
    filename="meta/stats.json",
    repo_type="dataset",
    token=os.environ.get("HF_TOKEN") or None,
)
dst = os.path.join("$DATASET_ROOT", "meta", "stats.json")
# Backup the baseline stats first (if not yet backed up)
backup = dst + ".baseline"
if os.path.exists(dst) and not os.path.exists(backup):
    shutil.copy2(dst, backup)
    print("  backed up baseline stats.json -> stats.json.baseline")
shutil.copy2(src, dst)
print(f"  swapped in pi05 stats.json ({os.path.getsize(dst):,} bytes)")
PY

# ------------------------------------------------------------------------------
# Step 3 — strip observation.images.reference from info.json + stats.json.
# ------------------------------------------------------------------------------
echo "==> [3/5] strip 'reference' feature from info.json + stats.json"
python - <<PY
import json, pathlib
root = pathlib.Path("$DATASET_ROOT")
ref_key = "observation.images.reference"
for fname in ("meta/info.json", "meta/stats.json"):
    p = root / fname
    d = json.load(open(p))
    container = d.get("features", d)
    if ref_key in container:
        del container[ref_key]
        json.dump(d, open(p, "w"), indent=2)
        print(f"  {fname}: removed {ref_key}")
    else:
        print(f"  {fname}: already clean")
PY

# ------------------------------------------------------------------------------
# Step 4 — rename camera1 -> right_wrist_0_rgb everywhere.
#   - info.json features
#   - stats.json top-level
#   - meta/episodes/*.parquet columns (videos/.../* and stats/.../*)
#   - videos/ directory
# ------------------------------------------------------------------------------
echo "==> [4/5] rename observation.images.camera1 -> observation.images.right_wrist_0_rgb"
python - <<PY
import json, pathlib, pyarrow.parquet as pq
root = pathlib.Path("$DATASET_ROOT")
old, new = "observation.images.camera1", "observation.images.right_wrist_0_rgb"

# info.json
p = root / "meta/info.json"
info = json.load(open(p))
if old in info.get("features", {}):
    info["features"][new] = info["features"].pop(old)
    json.dump(info, open(p, "w"), indent=2)
    print(f"  info.json features: {old} -> {new}")
elif new in info.get("features", {}):
    print(f"  info.json features: already renamed")
else:
    print(f"  info.json features: neither key present?!")

# stats.json
p = root / "meta/stats.json"
stats = json.load(open(p))
if old in stats:
    stats[new] = stats.pop(old)
    json.dump(stats, open(p, "w"), indent=2)
    print(f"  stats.json: {old} -> {new}")
elif new in stats:
    print(f"  stats.json: already renamed")

# episodes parquet columns
ep_files = sorted((root / "meta/episodes").rglob("*.parquet"))
for fp in ep_files:
    t = pq.read_table(fp)
    rename = {c: c.replace(old, new) for c in t.column_names if old in c}
    if not rename:
        print(f"  {fp.name}: already renamed (or no matching cols)")
        continue
    new_cols = [rename.get(c, c) for c in t.column_names]
    t2 = t.rename_columns(new_cols)
    backup = fp.with_suffix(".parquet.bak")
    if not backup.exists():
        fp.rename(backup)
    pq.write_table(t2, fp, compression="snappy")
    print(f"  {fp.name}: renamed {len(rename)} columns")

# Videos directory
old_dir = root / f"videos/{old}"
new_dir = root / f"videos/{new}"
if old_dir.exists() and not new_dir.exists():
    old_dir.rename(new_dir)
    print(f"  videos/: {old} -> {new}")
elif new_dir.exists():
    print(f"  videos/: already renamed")
PY

# ------------------------------------------------------------------------------
# Step 5 — patch total_frames to actual parquet row count.
# ------------------------------------------------------------------------------
echo "==> [5/5] patch info.json total_frames to actual parquet row count"
python - <<PY
import json, pathlib, pyarrow.parquet as pq
root = pathlib.Path("$DATASET_ROOT")
p = root / "meta/info.json"
info = json.load(open(p))
real = sum(pq.read_metadata(f).num_rows for f in sorted((root / "data").rglob("*.parquet")))
old = info["total_frames"]
if old != real:
    info["total_frames"] = real
    json.dump(info, open(p, "w"), indent=2)
    print(f"  total_frames: {old} -> {real}")
else:
    print(f"  total_frames: already correct ({real})")
PY

echo
echo "==> Done. Local pi05-ready dataset at $DATASET_ROOT"
echo "    use --dataset.root=$DATASET_ROOT in run_training_track_B.sh"
