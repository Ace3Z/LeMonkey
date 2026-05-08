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

Per [`docs/VLA_ARCHITECTURES.md`](../docs/VLA_ARCHITECTURES.md) §3 ("Eval 3"):

| | Choice | Why |
|---|---|---|
| Primary VLA | [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) | Only plug-and-play VLA in LeRobot whose VLM has serious world knowledge of public figures. |
| VLM backbone | PaliGemma-3B (`google/paligemma-3b-pt-224`) | WebLI pretraining → meaningful prior on public figures. SmolVLM-500M (used in Eval 1/2) does not have this. |
| Action expert | gemma_300m | π0.5 default. Flow-matching head, 10 inference steps. |
| Total params | ~3.3 B | Costs the small-model bonus, but it's the only path with strong celeb recognition. |
| Fallback | Image-as-prompt SmolVLA (Interleave-VLA-style) | Single-VLA, so still complies with §3. Triggered only if PaliGemma's zero-shot celeb recognition collapses or 20 s rollout cap is breached. |

### Why π0.5 over π0

- `tokenizer_max_length=200` (π0 = 48) — fits longer prompts and OOD names.
- State/action **quantile** normalisation — more robust on small fine-tune sets.

### Fine-tuning strategy

Default: **train action expert only**, freeze PaliGemma entirely
(`train_expert_only=true`, `freeze_vision_encoder=true`). PaliGemma already
has the celebrity → face association from web pretraining; the action expert
just needs to learn the geometry of "go to the position where I see Taylor
Swift's face."

Pivot to unfreezing PaliGemma (`train_expert_only=false`) only if the
zero-shot probe (below) shows PaliGemma can't ID the printed A5 portraits at
the workspace lighting / angle.

## Pre-training risk-check (run BEFORE burning Brev hours)

1. **Zero-shot PaliGemma probe on the TOY PDF.** Hand
   `google/paligemma-3b-pt-224` the 15 cut-out TOY images (already in the repo
   at [`docs/Eval_3_TOY_Celebrity_Images.pdf`](../docs/Eval_3_TOY_Celebrity_Images.pdf))
   and ask "Who is in this image?" for each. Then repeat on a sample of OOD
   names from the TA candidate list. Pass: ≥ 80 % accuracy on TOY. Fail →
   π0.5 alone won't suffice; pivot to image-as-prompt or to data-augmentation
   route enabled by the loosened §3 rule (offline face-ID labelling).
2. **Compute budget check.** π0.5 fine-tuning on Brev: ~10–15 h on A100 vs
   ~4 h for SmolVLA. With $200 of Brev credit and Eval 1/2 already trained,
   budget for one full π0.5 run + ~2 h headroom for diagnostics.
3. **Held-out & OOD generalisation probes.** The 9-rollout split gives us
   three test regimes with different difficulties (TOY → held-out IID → OOD).
   After fine-tune, evaluate on each separately:
   - TOY collapse → memorisation problem; need more diverse training
     positions for the same images.
   - Held-out IID collapse → photo-specific overfitting; data augmentation
     (different framings/lighting of the 3 celebs) is the lever.
   - OOD collapse → name → face binding not transferring; pivot to
     image-as-prompt or to offline face-ID labelling at training time.

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

1. ☐ Zero-shot PaliGemma probe (gate on ≥ 80 % in-distribution accuracy)
2. ☐ Recording protocol + balanced plan
3. ☐ Collect demos in HG (per PROJECT.md "include HG recordings")
4. ☐ Merge episodes into LeRobot v3 dataset
5. ☐ Push dataset to HF Hub (`HBOrtiz/so101_eval3_all`)
6. ☐ Train on Brev (this directory's scripts)
7. ☐ Probe OOD generalisation
8. ☐ Real-robot rollout

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
