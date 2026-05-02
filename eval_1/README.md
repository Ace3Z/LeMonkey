# eval_1 — SO-101 SmolVLA deployment

Runtime artifacts and scripts for **Eval 1** of the project: color-conditioned
banana → bowl pick-and-place with a fine-tuned SmolVLA policy + a residual
correction head trained on HG-DAgger data.

## Layout

```
eval_1/
├── README.md                    ← you are here
├── scripts/                     ← all runnables (tracked in git)
│   ├── setup / bootstrap
│   │   └── brev_setup.sh                    idempotent bootstrap for fresh Brev VMs
│   │
│   ├── data collection (robot-side, run on the laptop with arm + camera)
│   │   ├── run_rollout.sh                   single rollout, typed prompt
│   │   ├── run_rollout_voice.sh             single rollout, Whisper voice prompt
│   │   ├── voice_transcribe.py              Whisper helper (called by run_rollout_voice.sh)
│   │   ├── dagger_record.py                 HG-DAgger correction recorder (SPACE toggle, leader bilateral, 'd' delete, 'n' next, 'r' rest)
│   │   └── rest_arms.py                     release torques, manually home both arms
│   │
│   ├── evaluation (robot-side)
│   │   ├── eval_checkpoint.sh               30-rollout structured eval of the BASE policy
│   │   └── compare_evals.py                 aggregate eval CSVs across sessions, pick winner
│   │
│   ├── analysis (offline, no robot needed)
│   │   ├── analyze_memorization.py          dataset-level memorization-risk metrics
│   │   └── probe_language_conditioning.py   probe whether the policy is conditioning on language
│   │
│   └── residual/                ← residual policy subsystem (CR-DAgger style)
│       ├── residual_head.py                 the small MLP module (~384K params)
│       ├── train_residual.py                trains the residual on DAgger data, frozen base
│       ├── inference_residual.py            ResidualWrapper: base + residual at deploy time
│       └── eval_residual.py                 30-rollout evaluator using ResidualWrapper
│
├── train/                       ← model checkpoints  (gitignored)
├── rollouts/                    ← per-rollout dataset dumps  (gitignored)
├── evals/                       ← per-session eval CSVs  (gitignored)
├── bench/                       ← Brev batch-size benchmark output  (gitignored)
└── SETUP.md                     ← Brev VM runbook  (gitignored, local only)
```

## What's on Hugging Face Hub (under `HBOrtiz/`)

| Repo | Type | Contents |
|---|---|---|
| [`smolvla_eval1`](https://huggingface.co/HBOrtiz/smolvla_eval1) | model | Fine-tuned base policy (450M params, 20k steps) + model card |
| [`so101_eval1_blue`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_blue)   | dataset | 39 ep, banana → blue bowl |
| [`so101_eval1_red`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_red)     | dataset | 39 ep, banana → red bowl |
| [`so101_eval1_green`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_green) | dataset | 40 ep, banana → green bowl |
| `so101_eval1_dagger_blue` | dataset | HG-DAgger corrections, blue corner positions |
| `so101_eval1_dagger_red` | dataset | HG-DAgger corrections, red |
| `so101_eval1_dagger_green` | dataset | HG-DAgger corrections, green |

## Default checkpoint

All scripts default to **`020000`** — the most-converged step (final
cosine-decay LR ≈ 2.5% of base, action-expert weight delta from 015000 was
only ~1.02). Override with the first positional argument or `--ckpt-step`.

```bash
./scripts/run_rollout.sh 015000                  # try 15k checkpoint
./scripts/eval_checkpoint.sh 015000 42           # eval the 15k ckpt with seed 42
```

---

## Quick reference — which script to use

### "I want to record DAgger correction episodes for the policy"
```bash
./scripts/dagger_record.py \
  --dataset-root /home/lemonkey/LeMonkey/datasets/eval1_dagger/blue \
  --dataset-repo-id ${HF_USER}/so101_eval1_dagger_blue \
  --task "Put the banana in the blue colored bowl." \
  --num-episodes 12 --episode-time-s 30
```
Press SPACE to toggle teleop ON/OFF (anchored delta — no jerk). Press `n`
to end an episode early after a successful pick. Press `d` to mark the
last-saved episode for deletion at end-of-run. Press `r` to release
torques mid-session for manual homing.

### "I want to evaluate the BASE policy"
```bash
./scripts/eval_checkpoint.sh                    # 30 rollouts, base 020000, random seed
./scripts/eval_checkpoint.sh 020000 42          # same, fixed seed
./scripts/compare_evals.py                      # aggregate all eval CSVs, print winner
```

### "I want to evaluate the BASE + RESIDUAL composite policy"
```bash
./scripts/residual/eval_residual.py \
  --base-path     /home/lemonkey/LeMonkey/eval_1/train/smolvla_eval1/checkpoints/020000/pretrained_model \
  --residual-path /home/lemonkey/LeMonkey/eval_1/train/residual/last \
  --num-episodes 30
./scripts/compare_evals.py                      # picks up the residual CSV alongside base CSVs
```

### "I want to train the residual head on DAgger data" (run on Brev H100)
```bash
python scripts/residual/train_residual.py \
  --dataset-root ~/LeMonkey/datasets/eval1_dagger/blue \
  --dataset-root ~/LeMonkey/datasets/eval1_dagger/red \
  --dataset-root ~/LeMonkey/datasets/eval1_dagger/green \
  --policy-path  ~/LeMonkey/eval_1/train/smolvla_eval1/checkpoints/020000/pretrained_model \
  --out          ~/outputs/residual \
  --steps 5000 --batch-size 32 --lr 3e-4 --intervention-weight 2.0
```

### "I want to set up a fresh Brev VM"
```bash
bash ~/LeMonkey/eval_1/scripts/brev_setup.sh   # idempotent — installs miniconda, lerobot, ffmpeg
```

### "I want to release torques and manually home both arms"
```bash
./scripts/rest_arms.py                  # release torques, ENTER when done
./scripts/rest_arms.py --hold-after     # re-engage follower torque at new pose
```

### "I want to check whether the policy is memorizing or learning language"
```bash
./scripts/analyze_memorization.py            # static dataset metrics (~5 s)
./scripts/probe_language_conditioning.py     # behavioral probe with varied prompts (~3 min)
```

The probe's **wrong_color distance** is the decisive signal: if swapping
`blue` ↔ `red` in the prompt barely changes the predicted action chunk,
the policy isn't really conditioning on the color word. Reference healthy
range: wrong_color > 30, paraphrase < 15, empty/nonsense > 30.

---

## Architecture overview

**Two-component policy at deploy time:**

```
   observation
       │
       ▼
   ┌────────────────────┐
   │  base SmolVLA      │   frozen, 450M params, on HF Hub
   │  (HBOrtiz/smolvla_eval1)
   └────────────────────┘
       │ base_action (6,)
       ▼
   ┌────────────────────┐
   │  residual MLP      │   ~384K params, trained on DAgger frames only
   │  (scripts/residual/)
   └────────────────────┘
       │ residual (6,) ←→ clipped to ±5°/joint, ±10 gripper
       ▼
   final_action = base_action + clip(residual)
       │
       ▼
   robot
```

The base learns the bulk of the task from clean teleop demos. The residual
learns to correct base mistakes on the failure cases captured by HG-DAgger
(banana in corner positions, etc.) without disturbing the base's correct
behaviour. Trained on correction data only, with non-intervention frames
having near-zero target deltas.

Strategy doc with full reasoning: `docs/report/residual_strategy.md`
(local-only, gitignored).

---

## Hardware assumptions

- **Robot**: SO-101 follower on `/dev/so101-follower` (udev-stable symlink),
  calibrated, in home pose. Leader on `/dev/so101-leader` (udev-stable).
- **Camera**: USB camera at `/dev/video0` (640×480 @ 30 fps), wrist-mounted.
  Same physical mount used during training.
- **GPU**: any NVIDIA GPU with ≥6 GB VRAM. GTX 1660 SUPER tested for inference
  (~683 ms/frame for residual+base — slow but functional). H100 for training.
- **Microphone**: only for `run_rollout_voice.sh` (defaults to `plughw:1,0`).

The udev symlinks `so101-follower` / `so101-leader` are created by
`/etc/udev/rules.d/99-so101.rules` (host-specific, not in the repo).
Without them, override with `--follower-port /dev/ttyACMx`
`--leader-port /dev/ttyACMy` on `dagger_record.py` and `rest_arms.py`.

---

## Inference command pattern (for reference)

The scripts wrap this `lerobot-record` invocation:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/so101-follower --robot.id=my_follower \
  --robot.cameras="{ camera1: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.repo_id=local/eval_<name> \
  --dataset.root=/path/to/output \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=20 \
  --dataset.single_task="<prompt>" \
  --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
  --dataset.push_to_hub=false \
  --policy.path=/home/lemonkey/LeMonkey/eval_1/train/smolvla_eval1/checkpoints/020000/pretrained_model
```

The camera key is `camera1` (not `front`) — the policy expects
`observation.images.camera1`; using `front` triggers a feature-mismatch
error because `--dataset.rename_map` is applied after feature validation
in `lerobot-record`.

For residual deployment, `lerobot-record` doesn't know about the residual
wrapper — use `scripts/residual/eval_residual.py` instead, which loads
both base + residual and runs the rollout loop directly.

---

## Known limitations

- **No image augmentation during the original base training** — the base
  may struggle under different lighting. The residual fine-tunes on
  augmentation-friendly DAgger data which partially mitigates.
- **Single camera, `empty_cameras=2` pads** the two missing wrist views
  with zeros (matches `smolvla_aloha_sim` style).
- **English-only prompts** — 12 phrasings + 5 OOD paraphrases per color
  used in eval. Out-of-language behaviour is undefined.
- **Inference latency**: residual stack ~683 ms/frame on GTX 1660 SUPER
  because we reset the base per-frame for train/inference parity. Robot
  motors interpolate smoothly between commands; this works for the 20 s
  Eval-1 rollouts.

See [`HBOrtiz/smolvla_eval1`](https://huggingface.co/HBOrtiz/smolvla_eval1)
for the full base-model card.
