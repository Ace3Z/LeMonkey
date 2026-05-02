# eval_1 — SO-101 SmolVLA deployment

Runtime artifacts and scripts for **Eval 1** of the project: color-conditioned
banana → bowl pick-and-place with a fine-tuned SmolVLA policy.

## Layout

```
eval_1/
├── README.md                ← you are here
├── scripts/                 ← all runnables (tracked in git)
│   ├── run_rollout.sh           interactive single rollout (typed prompt)
│   ├── run_rollout_voice.sh     voice prompt via Whisper large-v3-turbo
│   ├── voice_transcribe.py      faster-whisper helper (SO-101 vocab bias)
│   ├── eval_checkpoint.sh       structured per-checkpoint eval harness
│   └── compare_evals.py         aggregate eval CSVs across sessions
├── train/                   ← model checkpoints  (gitignored, ~6 GB)
├── rollouts/                ← per-rollout dataset dumps  (gitignored)
├── evals/                   ← per-session eval CSVs  (gitignored)
├── bench/                   ← Brev batch-size benchmark output  (gitignored)
└── SETUP.md                 ← Brev VM runbook  (gitignored, local only)
```

## What's on Hugging Face Hub

| Repo | Type | Contents |
|---|---|---|
| [`HBOrtiz/smolvla_eval1`](https://huggingface.co/HBOrtiz/smolvla_eval1) | model | The trained policy (final 020000 checkpoint, 450M params) + model card |
| [`HBOrtiz/so101_eval1_blue`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_blue)   | dataset | 39 episodes, banana → blue bowl |
| [`HBOrtiz/so101_eval1_red`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_red)     | dataset | 39 episodes, banana → red bowl |
| [`HBOrtiz/so101_eval1_green`](https://huggingface.co/datasets/HBOrtiz/so101_eval1_green) | dataset | 40 episodes, banana → green bowl |

## Default checkpoint

All scripts default to **`020000`** — the most-converged step (final cosine-decay
LR ≈ 2.5% of base, action-expert weight delta from 015000 was only ~1.02 — negligible).

Override per script:
```bash
./scripts/run_rollout.sh 015000              # try 15k checkpoint
./scripts/eval_checkpoint.sh 015000 9 voice  # 9 voice-driven rollouts on 15k
```

## Quick start

### Single rollout (typed prompt)
```bash
./scripts/run_rollout.sh
```

### Voice-driven rollout (Whisper)
```bash
./scripts/run_rollout_voice.sh
```
Press ENTER to start recording, ENTER to stop. The transcript is shown for
confirmation before launching the rollout.

### Reset both arms to rest pose
```bash
./scripts/rest_arms.py                 # release torque, ENTER when done
./scripts/rest_arms.py --hold-after    # re-engage follower torque at new pose
```
Connects to follower + leader, disables torque on both so the arms go limp
and can be moved by hand. Useful before recording sessions, after a Ctrl+C
that left the follower in an awkward pose, or any time you want a manual
home reset.

### HG-DAgger correction recording (for compounding-error failures)
```bash
./scripts/dagger_record.py \
  --dataset-root /home/lemonkey/LeMonkey/datasets/eval1_dagger/blue \
  --dataset-repo-id ${HF_USER}/so101_eval1_dagger_blue \
  --task "Put the banana in the blue colored bowl." \
  --num-episodes 5 \
  --episode-time-s 30
```
Runs the policy by default; **hold SPACEBAR** to take over via the leader arm
(release to return to policy control). Frames are saved with
`is_intervention=1` on takeover frames so corrective demonstrations can be
filtered/upweighted during fine-tuning. Implements Human-Gated DAgger
(Kelly et al. 2019) on top of LeRobot v3 datasets — needed because lerobot's
own recorder is policy-XOR-teleop, not hybrid.

### Memorization-vs-learning offline analyses
```bash
./scripts/analyze_memorization.py            # static dataset analysis
./scripts/probe_language_conditioning.py     # actual policy behavior probe
```

**`analyze_memorization.py`** — loads the 3 color datasets and reports
trajectory diversity within-prompt vs across-color, plus a frames/params
overparameterization sanity check. Cheap, runs in 5 s.

**`probe_language_conditioning.py`** — runs inference on training images
with deliberately varied prompts (verbatim trained / paraphrased /
wrong-color / empty / nonsense) and compares predicted action chunks. The
**wrong-color distance** is the decisive signal: if swapping `blue` ↔ `red`
in the prompt barely changes the action, the policy isn't really listening
to the color word. Takes ~3 min on a GTX 1660 SUPER.

Reference numbers from a healthy run:
- wrong_color distance > 30 (color word strongly steers action)
- paraphrase distance < 15 (semantically equivalent phrasings produce similar actions)
- empty/nonsense distance > 30 (language is being read, not ignored)

The current trained checkpoint (020000) shows wrong_color ≈ 15 and high paraphrase
variance (17–112), indicating partial language conditioning that's brittle to
phrasing — see the probe output for full details.

### Structured per-checkpoint evaluation
```bash
./scripts/eval_checkpoint.sh                   # 30 rollouts on 020000 (10/color, shuffled)
./scripts/eval_checkpoint.sh 015000             # same but on a different ckpt
./scripts/eval_checkpoint.sh 020000 42          # fixed seed (reproducible shuffle)
./scripts/compare_evals.py                      # aggregate all sessions, print winner
```

Each session runs 30 rollouts: **10 per color**, with **5 in-distribution
prompts** (verbatim from training data) and **5 out-of-distribution prompts**
(plausible paraphrases the policy never saw). Order is shuffled.

For each rollout the harness:
1. Displays the target color, prompt type, and the exact prompt
2. Waits for ENTER to start
3. Runs `lerobot-record` for 20 s with that prompt
4. Asks `Success? [y/n]`

Results land in `evals/ckpt<step>_<timestamp>.csv` with a `prompt_type`
column so `compare_evals.py` can split in-distribution vs OOD success rates.

## Hardware assumptions

- **Robot**: SO-101 follower on `/dev/ttyACM1`, calibrated, in home pose
- **Camera**: USB camera at `/dev/video0` (640×480 @ 30 fps), wrist-mounted
- **GPU**: any NVIDIA GPU with ≥6 GB VRAM (GTX 1660 SUPER tested; SmolVLA fp32 fits at batch 1)
- **Microphone**: required only for voice mode (defaults to `plughw:1,0`)

## Inference command pattern

The scripts wrap this `lerobot-record` invocation:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_follower \
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

Note: the camera key is **`camera1`** (not `front`). The policy expects
`observation.images.camera1`; using `front` triggers a feature-mismatch error
because `--dataset.rename_map` is applied after feature validation in
`lerobot-record`.

## Known limitations

- **No image augmentation during training** — the model has only seen home
  lighting + table color. May struggle at HG (different tables, different lighting).
- **Single camera** — `empty_cameras=2` pads the missing wrist views with zeros.
- **English-only prompts** — 12 phrasings used during training, all English.

See [`HBOrtiz/smolvla_eval1`](https://huggingface.co/HBOrtiz/smolvla_eval1) for
the full model card.
