# eval_3 — SO-101 π0.5, Coke can on celebrity image

Runtime artifacts and scripts for **Eval 3** (50 pts): the policy must place a
Coke can on top of the printed celebrity image named by the prompt.

Per `docs/PROJECT.md` §2:
- DIN A5 color portraits placed in a semicircle. **All images are head/shoulder
  portraits — not full-body.**
- A normal 330 ml Coke can in the middle (empty for Eval 3, no Coke Zero).
  May be crumbled at the sides for grip; must still stand on its own.
- Prompt: `"Place the coke on [celebrity name]"`.
- **In-distribution celebrities:** Taylor Swift, Barack Obama, Yann LeCun.
- **9 rollouts / team**, **5.55 pts each** (50 / 9), **20 s / rollout**, split
  into three groups of three:

  | Runs | Setup | Image source |
  |---|---|---|
  | **1–3** known IID | one Swift + one Obama + one LeCun on the table in random order | **Exact images from the TOY PDF** ([`docs/Eval_3_TOY_Celebrity_Images.pdf`](../docs/Eval_3_TOY_Celebrity_Images.pdf) — 5 of each celebrity, 15 total) |
  | **4–6** held-out IID | same three celebrities | Different photos of Swift / Obama / LeCun the TAs did NOT hand out |
  | **7–9** OOD | popular OOD celebs (e.g. Roger Federer, Angela Merkel) | Drawn from a TA candidate list (Slack) |

  Exact image setups (positions, OOD identity) are undisclosed in advance but
  identical across groups.

> **Action item:** print [`docs/Eval_3_TOY_Celebrity_Images.pdf`](../docs/Eval_3_TOY_Celebrity_Images.pdf)
> in color and **cut the images out so there is no white border**. These cut-outs
> are the exact images used in runs 1–3.

**Constraint (PROJECT.md §3, updated rule):** VLA-only **at inference time**.
- At demo day: no YOLO, face-ID models, cloud-VLM API calls, or other
  foundation models in the policy itself.
- At training time: **other models *are* allowed offline to create labelled
  data** (e.g. run a face-recognition model to auto-label demos with celebrity
  bounding boxes; generate synthetic backgrounds via SDXL). The "helper" model
  must not run at inference — only its outputs end up baked into the VLA's
  weights.

This loosening is **new** vs. the original brief and opens up data-augmentation
routes that the architecture doc's "primary π0.5 + name-only" plan did not
assume.

## Architecture

Backbone is fixed: **`lerobot/pi05_base`** (Pi0.5, ~3.3 B params, PaliGemma-3B
VLM + 300 M action expert, flow-matching head). PROJECT.md §3 explicitly
allows different models per eval, so SmolVLA stays for Eval 1/2 and Pi0.5
takes Eval 3 — costs the smallest-model bonus but is the only path that even
has a chance at the OOD celebrity tier (Pi0.5's PaliGemma backbone has
WebLI-derived public-figure knowledge; SmolVLM-500M does not).

The plan **branches** based on a 10-min zero-shot probe (Phase 1 below).
Two concrete paths:

### Path A — image-as-prompt + co-train (primary, recommended)

Adopt the **Interleave-VLA pattern** ([arxiv 2505.02152](https://arxiv.org/abs/2505.02152) —
2–3× OOD generalization gain over text-only, single-VLA at inference):

- At every training step AND inference step, the prompt is interleaved:
  `[reference photo of <name>] "Place the coke on <name>"`
- The VLA matches the reference photo to one of the 3 prints in the scene —
  it does **visual matching**, not name-recall. This sidesteps PaliGemma's
  weak / undocumented zero-shot celebrity recognition entirely.
- **Co-train** with VQA pairs from VGGFace2 / IMDB-Wiki (~5–10k pairs from
  ~100 candidates) to keep PaliGemma's face features sharp. Mix ratio per
  Pi0.5-KI paper: ≥1 VQA sample for every 2 robot demos.
- `train_expert_only=False` (must train VLM gradients to ground name → face).
- Single-VLA-at-inference compliant: the reference photo lookup is data, not
  a model. The face-ID model used to clean VGGFace2 only runs offline (per
  PROJECT.md §3 loosening: "helper" models allowed at training time only).

Triggers: probe accuracy < 80 % on TOY OR < 50 % on a sample of popular OOD
celebs, OR (regardless of probe) we want robust runs 7–9.

### Path B — text-only Pi0.5 fine-tune (conservative)

The original scaffolded plan: `lerobot/pi05_base`, prompt is just
`"Place the coke on <name>"`, train action expert only, freeze PaliGemma.

Triggers: probe accuracy ≥ 80 % on TOY AND ≥ 50 % on OOD sample.

Risk: PaliGemma forgets what little celebrity grounding it has during
sequential fine-tune (catastrophic forgetting documented in Pi0.5-KI and
"Don't Blind Your VLA" arxiv 2510.25616). Mitigated by `train_expert_only=True`
default — the VLM weights are not touched, so the celebrity-recognition
ability we measured pre-train is exactly what we get post-train.

### Why π0.5 over π0

- `tokenizer_max_length=200` (π0 = 48) — fits longer prompts and OOD names.
- State/action **quantile** normalisation — more robust on small fine-tune sets.
- Pi0.5 already co-trained on web data + robot data, the same shape Path A needs.

## Pre-training risk-check (run BEFORE burning Brev hours)

1. **Zero-shot PaliGemma probe on the TOY PDF + a sample of popular OOD candidates.**
   Run `eval_3/scripts/probe_paligemma.py` (no Brev — local GPU, ~10 min).
   The script: extracts the 15 TOY images from `docs/Eval_3_TOY_Celebrity_Images.pdf`,
   pulls reference photos for ~10 popular OOD candidates (Federer, Merkel,
   Musk, Beyoncé, etc.) from Wikipedia, and asks
   `google/paligemma-3b-pt-224` "Who is this person?" for each. Decision rule:
   - **≥ 80 % TOY AND ≥ 50 % OOD** → Path B (text-only Pi0.5) is viable.
     Path A still safer for OOD tier; pick based on time budget.
   - **anything below** → Path A (image-as-prompt + co-train) is mandatory.
2. **Compute budget check.** Pi0.5 fine-tuning on Brev: ~10–15 h on RTX Pro
   6000 / A100 with batch 32 × 25–30k steps. ~$15–20 of credit. With $170+
   left after Eval 2's two runs, well within budget for one Path A run + 2h
   diagnostics.
3. **Post-train generalisation probes.** Three failure modes the 9-rollout
   split exposes — diagnose each separately:
   - TOY collapse → memorisation problem; more semicircle position variety.
   - Held-out IID collapse → photo-specific overfit; more photos per IID celeb.
   - OOD collapse → reference-photo grounding broken; check VQA co-train ratio.

## Layout

```
eval_3/
├── README.md                    ← this file
├── scripts/
│   └── brev/                    scripts to scp to the Brev VM for training
│       ├── setup_pi05.sh           env bootstrap (lerobot[pi0]==0.5.1 + paligemma deps)
│       ├── run_training.sh         the actual lerobot-train command
│       ├── start_training.sh       wraps run_training.sh in a transient systemd user service
│       ├── follow_training.sh      live tail of the log
│       └── training_status.sh      one-shot status snapshot
├── state/                       plan.json — persistent recording state (gitignored)
├── train/                       model checkpoints (gitignored)
├── rollouts/                    per-rollout dataset dumps (gitignored)
└── evals/                       per-session eval CSVs (gitignored)
```

Datasets live under `~/LeMonkey/datasets/eval3/` (one dir per episode).

## Status

**Not yet trained.** Brev scripts are scaffolded but **not executed**.
Recording protocol (record_eval3.py) and rollout script (run_rollout_eval3.py)
not yet implemented. Build order:

**Phase 0 — physical prep (no compute, today):**
- ☐ Print + cut TOY PDF (no white border, per PROJECT.md)
- ☐ Find ≥ 4 extra photos per IID celeb (Swift / Obama / LeCun) for held-out training data
- ☐ Decide camera mount → **shoulder mount** for Eval 3 (sees all prints throughout the descent; wrist loses them). PROJECT.md §4 allows per-eval mount choice.

**Phase 1 — gating probe (no compute, ~1 h):**
- ☐ Build `eval_3/scripts/probe_paligemma.py`
- ☐ Run probe → branch: Path A (image-as-prompt + co-train) OR Path B (text-only)

**Phase 2 — data pipeline (~1–2 h coding + ~10–15 h teleop):**
- ☐ `record_eval3.py` — recorder, image-as-prompt format if Path A
- ☐ Collect ~150 demos (semicircle layouts varied, lighting varied, include some HG)
- ☐ `merge_eval3_episodes.py` → one LeRobot v3 dataset
- ☐ Push dataset to `HBOrtiz/so101_eval3_all`
- ☐ (Path A only) build VQA co-train stream from VGGFace2 / IMDB-Wiki

**Phase 3 — training (~12 h on Brev RTX Pro 6000):**
- ☐ Update `run_training.sh` for Path A or B; image augmentation enabled (color + illumination)
- ☐ Launch via `start_training.sh`; checkpoint every 5 k steps

**Phase 4 — pull-back + validation (~30 min, no robot):**
- ☐ Pull final checkpoint from HF, run `probe_paligemma.py` again on the trained model
- ☐ Verify ≥ 80 % held-out IID accuracy before committing to robot tests

**Phase 5 — real-robot eval:**
- ☐ Build `run_rollout_eval3.py` (mirror of `eval_2/scripts/run_rollout_freeplay.py` with image-as-prompt input if Path A)
- ☐ Run all 9 protocol rollouts; iterate before demo day

## Training pipeline (scaffolded, not yet run)

The recipe mirrors Eval 2's Brev pipeline, with the policy / batch-size /
schedule swapped for π0.5. **Training is parked — do not run yet.**

### What to copy to Brev (when ready)

```
~/LeMonkey/datasets/eval3_merged/                       ← merged dataset (TBD)
~/LeMonkey/eval_3/scripts/brev/setup_pi05.sh            → ~/setup_pi05.sh
~/LeMonkey/eval_3/scripts/brev/run_training.sh          → ~/run_training.sh
~/LeMonkey/eval_3/scripts/brev/start_training.sh        → ~/start_training.sh
~/LeMonkey/eval_3/scripts/brev/follow_training.sh       → ~/follow_training.sh
~/LeMonkey/eval_3/scripts/brev/training_status.sh       → ~/training_status.sh
```

### On Brev (when ready)

```bash
# 1. (only on a fresh VM) install miniconda + lerobot[pi0]==0.5.1 + ffmpeg + HF auth
bash ~/LeMonkey/eval_3/scripts/brev/setup_pi05.sh
hf auth login   # paste write token

# 2. Linger so training survives SSH disconnect
sudo loginctl enable-linger $USER

# 3. Launch training as a systemd user service
chmod +x ~/run_training.sh ~/start_training.sh ~/follow_training.sh ~/training_status.sh
~/start_training.sh
~/follow_training.sh    # live progress (Ctrl-C to detach; training keeps running)
```

### Training config (planned)

| Param | Value | Why |
|---|---|---|
| `--policy.path` | `lerobot/pi05_base` | π0.5 with PaliGemma-3B backbone |
| `--policy.push_to_hub` | `false` | local-only; we push via `hf upload` after |
| `--policy.empty_cameras` | `2` | match v2 — pads two missing cameras with zeros |
| `--dataset.repo_id` | `local/so101_eval3_all` | local-only |
| `--dataset.root` | `~/LeMonkey/datasets/eval3_merged` | the merged dataset (TBD) |
| `--dataset.image_transforms.enable` | `true` | color jitter + **random illumination** (per PROJECT.md §7 TA tip — demo-day lighting is unpredictable). **No horizontal flip** (would mirror celebrity faces). |
| `--batch_size` | `32` | π0.5 = ~6 × SmolVLA VRAM; eval_2 used 192 on H100. Drop to 16 if OOM. |
| `--steps` | `30000` | larger model + smaller batch → more steps than eval_2's 25k |
| `--save_freq` | `5000` | 6 intermediates (5k/10k/15k/20k/25k/30k) |
| `--output_dir` | `/home/shadeform/outputs/train/pi05_eval3` | per-eval output |
| `--job_name` | `pi05_eval3` | per-eval job name |
| `--policy.device` | `cuda` | |
| `--wandb.enable` | `false` | |

Defaults left untouched (relying on `lerobot/pi05_base` config):
- `train_expert_only=true` → freeze PaliGemma, only action expert trains.
- `freeze_vision_encoder=true` → SigLIP vision encoder frozen.
- `optimizer_lr=2.5e-5` → π0/0.5 default.
- `num_inference_steps=10` → flow-matching default.

## Hardware

Same as Eval 1/2 — see `eval_1/README.md`. Leader on `/dev/so101-leader`,
follower on `/dev/so101-follower`, camera on `/dev/video0`.

For Eval 3 specifically, **camera placement** is a per-eval decision
([PROJECT.md §4](../docs/PROJECT.md#4-hardware--objects)). Wrist-mounted
(default) sees the can but loses sight of all the portraits during the
descent; a self-built shoulder mount keeps every portrait in frame
throughout the rollout. Decision deferred until the zero-shot PaliGemma
probe — if PaliGemma needs a clean overview shot of the workspace to
recognise faces, shoulder-mount; if wrist-down works fine, leave it
mounted to match Eval 1/2 data.
