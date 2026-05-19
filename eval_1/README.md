# eval_1 — Direct color-conditioned pick-and-place

Runtime artifacts and scripts for **Eval 1** of the project: SmolVLA picks a
banana and places it in a colored bowl (`blue` / `red` / `green`) on prompt.

The deployed model is **`HBOrtiz/smolvla_eval1_v2`**, step `025000` — fine-tuned
from `lerobot/smolvla_base` on a merged 153-episode dataset (118 clean BC
demos + 35 HG-DAgger correction demos) with image augmentation enabled.

## Table of contents

- [What's on Hugging Face Hub](#whats-on-hugging-face-hub)
- [Default checkpoint](#default-checkpoint)
- [Quick reference — which script to use](#quick-reference--which-script-to-use)
- [Hardware assumptions](#hardware-assumptions)
- [Inference command pattern](#inference-command-pattern)
- [Layout](#layout)
- [Known limitations](#known-limitations)

## What's on Hugging Face Hub

| Repo | Type | Contents |
|---|---|---|
| [`smolvla_eval1_v2`](https://huggingface.co/HBOrtiz/smolvla_eval1_v2) | model | **Deployed policy** — 450M params, 25k steps, image augmentation, 5 intermediate checkpoints |
| [`so101_eval1_all_v2`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_all_v2) | dataset | Merged BC + DAgger training data, 153 ep, 44.6k frames |
| [`so101_eval1_blue`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_blue) · [`_red`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_red) · [`_green`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_green) | datasets | Per-color BC demos (39 / 39 / 40 ep) |
| [`so101_eval1_dagger_blue`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_dagger_blue) · [`_red`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_dagger_red) · [`_green`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_dagger_green) | datasets | Per-color HG-DAgger correction demos |

## Default checkpoint

All scripts default to **v2 / `025000`** — the most-converged step from the
final training run.

```bash
./scripts/run_rollout.sh                          # v2 / 025000
MODEL=v2 ./scripts/eval_checkpoint.sh 025000 42   # eval the default with seed 42
MODEL=v2 ./scripts/eval_checkpoint.sh 020000      # try a different intermediate
```

Pass a different step as the first positional argument to override.

## Quick reference — which script to use

### Run a single rollout with a typed or spoken prompt

```bash
./scripts/run_rollout.sh           # type the prompt
./scripts/run_rollout_voice.sh     # speak the prompt (Whisper) or 't' to type
```

Each rollout captures the arm's starting pose, runs the policy for 40 s
(press right-arrow to end the episode early — built into `lerobot-record`),
then drives the arm back to the starting pose for the next take.

### Run a rollout on macOS (Apple Silicon)

```bash
./scripts/dry_run_mac.py            # headless: load on MPS, time inference (no robot/cam)
./scripts/run_rollout_mac.sh        # actual rollout — needs SO-101 + USB cam attached
```

`run_rollout_mac.sh` is the macOS sibling of `run_rollout.sh`. It targets the
`lemonkey` conda env under `~/miniforge3`, auto-detects the follower at the first
`/dev/cu.usbmodem*` it finds, uses an integer camera index (not `/dev/video0`),
loads the SO-101 calibration from the in-repo `calibration/robots/so_follower/`,
and defaults the checkpoint to
`/Volumes/T7/LeMonkey/models/smolvla_eval1_v2/checkpoints/<step>/pretrained_model`.
Override any of those with the `PYBIN`, `FOLLOWER_PORT`, `CAMERA_INDEX`,
`CALIBRATION_DIR`, `CKPT`, or `ROLLOUT_DIR` env vars. First run requires Camera
permission for Terminal in macOS System Settings → Privacy & Security → Camera.

`dry_run_mac.py` loads the policy + pre/postprocessors (overriding the
`device_processor` step from `cuda` to `mps`) and runs `predict_action` against
a synthetic observation. Measured on M-series MPS: ~5 ms per cached action +
~850 ms per chunk-recompute (every 50th frame), effective ~46 Hz — above the
SO-101's 30 Hz control loop.

### Run a 30-rollout structured eval

```bash
MODEL=v2 ./scripts/eval_checkpoint.sh             # 30 rollouts, v2/025000, random seed
MODEL=v2 ./scripts/eval_checkpoint.sh 025000 42   # same, fixed prompt-shuffle seed
./scripts/compare_evals.py                        # aggregate eval CSVs across sessions
```

The eval shuffles 30 prompts (5 verbatim training phrasings + 5 OOD paraphrases
per color), asks Y/N after each rollout, and writes
`evals/v2_ckpt025000_<ts>.csv`. `compare_evals.py` picks up every CSV in
`evals/` and prints aggregate per-color and per-prompt-type success rates.

### Record DAgger correction episodes

```bash
./scripts/dagger_record.py \
  --dataset-root /home/lemonkey/LeMonkey/datasets/eval1_dagger/blue \
  --dataset-repo-id ${HF_USER}/so101_eval1_dagger_blue \
  --task "Put the banana in the blue colored bowl." \
  --num-episodes 12 --episode-time-s 30
```

Press `SPACE` to toggle teleop ON / OFF (anchored delta — no jerk).
`n` ends an episode early after a successful pick. `d` marks the last-saved
episode for deletion at end-of-run. `r` releases torques mid-session for
manual homing.

### Set up a fresh Brev VM

```bash
bash ~/LeMonkey/eval_1/scripts/brev_setup.sh   # idempotent — installs miniconda, lerobot, ffmpeg
```

### Release torques and manually home both arms

```bash
./scripts/rest_arms.py                  # release torques, ENTER to re-engage
./scripts/rest_arms.py --hold-after     # re-engage follower torque at the new pose
```

### Probe whether the policy is conditioning on language

```bash
./scripts/analyze_memorization.py            # static dataset metrics (~5 s)
./scripts/probe_language_conditioning.py     # behavioral probe with varied prompts (~3 min)
./scripts/probe_compositional.py             # compositional-prompt probe (~3 min)
```

The language-conditioning probe's **`wrong_color`** distance is the decisive
signal: if swapping `blue` ↔ `red` in the prompt barely changes the predicted
action chunk, the policy is not really conditioning on the color word.
Reference healthy range: `wrong_color > 30`, `paraphrase < 15`,
`empty/nonsense > 30`.

## Hardware assumptions

- **Robot**: SO-101 follower on `/dev/so101-follower` (udev-stable symlink),
  calibrated, in home pose. Leader on `/dev/so101-leader` (udev-stable).
- **Camera**: USB camera at `/dev/video0` (640×480 @ 30 fps), wrist-mounted.
  Same physical mount used during training.
- **GPU**: any NVIDIA GPU with ≥6 GB VRAM. GTX 1660 SUPER tested for inference.
  H100 / RTX 6000 used for training. Apple Silicon (MPS) also supported for
  inference via `run_rollout_mac.sh`; CPU-only is too slow for 30 Hz control.
- **Microphone**: only for `run_rollout_voice.sh` (defaults to `plughw:1,0`).

The udev symlinks `so101-follower` / `so101-leader` are created by
`/etc/udev/rules.d/99-so101.rules` (host-specific, not in the repo).
Without them, override with `--follower-port /dev/ttyACMx`
`--leader-port /dev/ttyACMy` on `dagger_record.py` and `rest_arms.py`.

## Inference command pattern

The shell scripts wrap this `lerobot-record` invocation:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/so101-follower --robot.id=my_follower \
  --robot.cameras="{ camera1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.repo_id=local/eval_<name> \
  --dataset.root=/path/to/output \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=40 \
  --dataset.single_task="<prompt>" \
  --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
  --dataset.push_to_hub=false \
  --policy.path=/home/lemonkey/LeMonkey/eval_1/train/smolvla_eval1_v2/checkpoints/025000/pretrained_model
```

The camera key is `camera1` — the policy expects `observation.images.camera1`.

## Layout

```
eval_1/
├── README.md                                     ← this file
├── scripts/                                      ← all runnables (tracked in git)
│   ├── brev_setup.sh                             idempotent bootstrap for fresh Brev VMs
│   │
│   ├── data collection (robot-side)
│   │   ├── run_rollout.sh                        single rollout, typed prompt (Linux deploy box)
│   │   ├── run_rollout_mac.sh                    macOS sibling of run_rollout.sh (MPS, /dev/cu.usbmodem*)
│   │   ├── dry_run_mac.py                        headless MPS inference smoke test, no hardware needed
│   │   ├── run_rollout_voice.sh                  single rollout, Whisper voice prompt
│   │   ├── voice_transcribe.py                   Whisper helper
│   │   ├── dagger_record.py                      HG-DAgger correction recorder
│   │   ├── auto_home.py                          captures + drives back to a saved pose (override port via SO101_FOLLOWER_PORT)
│   │   └── rest_arms.py                          release torques, manually home both arms
│   │
│   ├── evaluation
│   │   ├── eval_checkpoint.sh                    30-rollout structured eval (MODEL=v1|v2)
│   │   └── compare_evals.py                      aggregate eval CSVs across sessions
│   │
│   ├── analysis (offline, no robot needed)
│   │   ├── analyze_memorization.py               dataset-level memorization-risk metrics
│   │   ├── probe_language_conditioning.py        does the policy listen to the color word?
│   │   └── probe_compositional.py                does the policy respond to spatial language?
│   │
│   ├── normalize_dagger_to_bc_schema.py          one-off used to merge DAgger into BC v3 schema
│   └── residual/                                 research artifact (CR-DAgger residual head, superseded by v2)
│
├── train/                                        ← model checkpoints (gitignored)
├── rollouts/                                     ← per-rollout dataset dumps (gitignored)
└── evals/                                        ← per-session eval CSVs (gitignored)
```

`scripts/residual/` contains the original CR-DAgger residual-head approach we
explored before v2. The v2 from-scratch retrain (which absorbs the same DAgger
data) supersedes it for deployment, but the files remain for research notes
and reproducibility.

## Known limitations

- **Single camera**: `empty_cameras=2` pads the two missing wrist views with
  zeros (matches `smolvla_aloha_sim` style).
- **English-only prompts**: 5 verbatim training phrasings + 5 OOD paraphrases
  per color used in eval. Out-of-language behaviour is undefined.
- **Inference latency**: ~150 ms / frame on a GTX 1660 SUPER, well within
  what the SO-101 control loop tolerates at 30 Hz.

See [`HBOrtiz/smolvla_eval1_v2`](https://huggingface.co/HBOrtiz/smolvla_eval1_v2)
for the full model card.
