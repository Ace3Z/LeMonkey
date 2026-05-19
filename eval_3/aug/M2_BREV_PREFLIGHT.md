# M2 Brev pre-flight checklist

Before launching `eval_3/scripts/brev/run_training_track_D_m2.sh` on Brev,
verify every item below. Each one catches a known failure mode at zero cost.

## On the Brev VM

```
# 1. Repo is checked out at ~/LeMonkey on dev/m2-arcface-toolkit
cd ~/LeMonkey
git fetch origin
git checkout dev/m2-arcface-toolkit
git pull
git log --oneline -5

# 2. Conda env is set up with all M2 deps
conda activate lemonkey
python -c "import insightface, onnxruntime, transformers, lerobot; print('OK')"
# Expected: prints OK without ImportError.

# 3. Base teleop dirs + aug variant dirs are present
ls ~/LeMonkey/datasets/eval3 | head -5
ls ~/LeMonkey/datasets/eval3_track3_aug | head -5
ls ~/LeMonkey/datasets/eval3_track3_aug | wc -l    # should be 9216

# 4. HF token is set
echo "HF_TOKEN=$HF_TOKEN" | head -c 20    # should print "HF_TOKEN=hf_..."

# 5. CUDA is healthy
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 6. SmolVLA + Hans's warm-VLM are reachable
python -c "
from huggingface_hub import HfApi
api = HfApi()
for r in ['lerobot/smolvla_base', 'HansOrtiz/smolvlm2_celeb_warm']:
    info = api.model_info(r)
    print(f'{r}: {info.modelId} OK')
"
```

## M2 toolkit pre-flight

```
# 7. Download the M2 toolkit data from HF (one-time)
mkdir -p ~/eval3_m2_toolkit
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='HBOrtiz/eval3_m2_arcface_toolkit',
    repo_type='dataset',
    local_dir='/root/eval3_m2_toolkit',  # adjust to your $HOME
)
"
ls ~/eval3_m2_toolkit/   # should contain: README.md, celeb_embeddings.json,
                          # arcface_embeddings/, face_labels/
ls ~/eval3_m2_toolkit/face_labels/ | wc -l    # should be 151

# 8. Build the episode_index → variant mapping
python eval_3/aug/m2_episode_mapping.py \
    --base-root ~/LeMonkey/datasets/eval3 \
    --aug-root ~/LeMonkey/datasets/eval3_track3_aug \
    --output ~/eval3_m2_toolkit/episode_mapping.json
# Expected output: n_base=178  n_aug=9216  total=9394
# If n_base != 178 or n_aug != 9216, the merged HF dataset may have been
# rebuilt with different content — pause and reconcile with Roham.

# 9. Run the local hook probe to confirm SmolVLM2 dimensions match
python eval_3/aug/dbg/dbg_m2_hook_probe.py
# Expected: ALL 5/5 CHECKS PASSED — hook captures what we expect
```

## Smoke test (~10 min)

```
# 10. Run a 100-step M2 training session on a tiny subset to validate end-to-end
M2_FACE_LABELS_DIR=~/eval3_m2_toolkit/face_labels \
M2_MANIFEST_PATH=~/eval3_m2_toolkit/celeb_embeddings.json \
M2_AUG_ROOT=~/LeMonkey/datasets/eval3_track3_aug \
M2_EPISODE_MAPPING=~/eval3_m2_toolkit/episode_mapping.json \
M2_LAMBDA=0.2 \
M2_CAPTURE_LAYER=9 \
M2_LOG_EVERY=10 \
python -u eval_3/scripts/lerobot_train_with_m2.py \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.vlm_model_name=HansOrtiz/smolvlm2_celeb_warm \
    --policy.freeze_vision_encoder=True \
    --policy.train_expert_only=False \
    --policy.empty_cameras=1 \
    --policy.optimizer_lr=5e-5 \
    --policy.compile_model=False \
    --policy.push_to_hub=False \
    --dataset.repo_id=HBOrtiz/so101_eval3_track3_v3_baseline \
    --dataset.video_backend=pyav \
    --batch_size=8 --steps=100 \
    --output_dir=/tmp/m2_smoke \
    --job_name=m2_smoke \
    --wandb.enable=false

# Expected log lines:
#   [m2 launcher] face_labels=151 celebs=192 episodes=9394 lam=0.2 capture_layer=9
#   [m2 launcher] wrapped policy: NNN.NM frozen, NN.NM trainable
#   [m2] step=     0  m2_loss=±0.0NNN  n_valid=NN/24  mean_cos=±0.0NNN  base=NN
#   ... (every 10 steps)
#   loss decreases over the 100 steps
```

If step 10 passes, the full 30k-step run will work — launch the real one:

```
bash eval_3/scripts/brev/run_training_track_D_m2.sh
```

## Failure-mode quick-checks

| Symptom | Likely cause | Fix |
|---|---|---|
| `[WARN] M2WrappedPolicy: hook did not fire` | `compile_model=True` or some other config that breaks hook semantics | Confirm `--policy.compile_model=False` in launch flags |
| `KeyError: M2SupervisionBuilder ... no face_labels` | Mapping points to a source not in `face_labels/` (e.g. a base teleop accidentally not flagged as `is_base`) | Re-run `m2_episode_mapping.py`; check `n_base` and `n_aug` match expectations |
| `mean_cos` stays ~0 after 1k steps | Frozen projector + frozen layers 0-8 leave no path for VLM to align (only layers 9-15 can update) | Confirm `apply_partial_freeze` ran (check the "frozen / trainable" line at startup). If yes, try lower `M2_LAMBDA=0.1` or shallower capture layer 8. |
| OOM at bs=64 | bf16 activation memory for layers 9-15 backward + AdamW state | Drop to bs=48 first, then bs=32. A100 80GB should hold bs=64 comfortably so OOM signals a config issue |
| Action loss diverges after step 5k | M2 loss dominates and corrupts action-head input | Drop `M2_LAMBDA` to 0.1 and resume from the last checkpoint |

## What gets written where

| Artifact | Path |
|---|---|
| Training log | `~/outputs/train/smolvla_track_D_m2.log` |
| Final checkpoint (local) | `~/outputs/train/smolvla_track_D_m2/` |
| Final checkpoint (HF) | `HBOrtiz/smolvla_eval3_track_D_m2_mahbod` |
| Intermediate ckpts (every 5k steps) | `~/outputs/train/smolvla_track_D_m2/checkpoints/` |

## After training finishes

```
# Confirm the HF push succeeded
python -c "
from huggingface_hub import HfApi
files = HfApi().list_repo_files('HBOrtiz/smolvla_eval3_track_D_m2_mahbod')
print(f'{len(files)} files at HBOrtiz/smolvla_eval3_track_D_m2_mahbod')
for f in files[:8]: print(f' ', f)
"

# Hand off to Darius for Strix deployment (TODO.md step "Hand off to Darius
# for Strix deployment + 3-rollout test").
```
