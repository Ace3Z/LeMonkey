# Brev runbook — Pi0.5 + M2 + KLAL training

This page tells you the exact sequence of commands to run on a fresh Brev
VM (e.g. `a-toy-pi05`) to launch Track E (Pi0.5 + M2 + KLAL). Two-stage
gate-then-train, plus an auto-push watcher so you don't lose disk to
checkpoints.

VM resources expected: H100 80GB, 28 cores, ≥150 GB RAM, ≥97 GB disk.

## 0. SSH in (with forwarded agent so `git clone` from GitHub works)

```
ssh -A a-toy-pi05
```

## 1. One-shot provisioning

```
# This installs miniconda, the pi05 conda env, lerobot from source, the
# working transformers/hf_hub pin, and pre-downloads weights + M2 toolkit.
curl -sL https://raw.githubusercontent.com/Ace3Z/LeMonkey/dev/m2-arcface-toolkit/eval_3/scripts/brev/setup_brev_pi05.sh | bash
```

Then create the HF token file:

```
echo "HF_TOKEN=hf_..." > ~/LeMonkey/.env
```

## 2. Upload the 3-celeb augmentation.json bundle from your laptop

The 9216 `augmentation.json` files (one per aug variant) carry the
new_layout_camera_lmr that the M2 builder needs. They live on the
laptop, NOT on HF. From your laptop:

```
cd ~/Downloads/eval3_track3_aug
find . -name 'augmentation.json' | tar -czf /tmp/aug_jsons_3celeb.tar.gz -T -
scp /tmp/aug_jsons_3celeb.tar.gz a-toy-pi05:/tmp/
ssh a-toy-pi05 'mkdir -p ~/LeMonkey/datasets/eval3_track3_aug && \
                cd ~/LeMonkey/datasets/eval3_track3_aug && \
                tar -xzf /tmp/aug_jsons_3celeb.tar.gz'
```

You also need the base teleop directory listing (for the
episode-mapping build). The simplest is to upload its metadata too:

```
# Replace ~/Downloads/eval3 with where the base teleop dirs live.
cd ~/Downloads
tar -czf /tmp/eval3_base_listing.tar.gz \
    eval3/quick_*/   # adjust glob to your base-teleop dir name
scp /tmp/eval3_base_listing.tar.gz a-toy-pi05:/tmp/
ssh a-toy-pi05 'mkdir -p ~/LeMonkey/datasets/eval3 && \
                cd ~/LeMonkey/datasets/eval3 && \
                tar -xzf /tmp/eval3_base_listing.tar.gz --strip-components=1'
```

(The base-teleop videos themselves are NOT needed at training time —
the merged HF dataset has those. Only the directory layout is needed
so `m2_episode_mapping.py` can match base→aug.)

## 3. Run the gating experiments (~5 min total)

These tell us *before any training* whether Pi0.5+M2+KLAL has a path
forward.

```
cd ~/LeMonkey
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pi05
set -a && source ~/LeMonkey/.env && set +a
```

### G1: Does vanilla PaliGemma already have cross-modal attention?

```
python eval_3/scripts/attention_map_probe_pi05.py \
    --repo lerobot/pi05_base \
    --layers 6 10 14 17 \
    --out ~/pi05_attn_vanilla
scp -r a-toy-pi05:~/pi05_attn_vanilla ./   # pull to your laptop to inspect
```

**Pass:** argmax patch DIFFERS across the 3 prompts at one or more layers.
**Fail:** argmax constant across prompts → reconsider plan (VQA pretrain first).

### G2: Does PaliGemma know Swift/Obama/LeCun by name?

```
# Optional: upload the celeb headshots first.
scp -r ~/Downloads/eval3_celebs/track3_bank a-toy-pi05:~/eval3_celebs_bank
ssh a-toy-pi05

python eval_3/scripts/probe_paligemma_celeb_vqa.py \
    --celeb-bank ~/eval3_celebs_bank \
    --out ~/pg_vqa.json
```

**Pass (≥50% avg hit rate):** WebLI prior alive; M2+KLAL on action data alone is sufficient.
**Marginal (20-50%):** add Stage-3 VQA fine-tune before action training.
**Fail (<20%):** VQA fine-tune is mandatory.

## 4. Launch training (in tmux so it survives disconnect)

```
# Kill any prior session.
tmux kill-session -t pi05 2>/dev/null

# Launch.
tmux new-session -d -s pi05 'cd ~/LeMonkey && \
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate pi05 && \
    set -a && source .env && set +a && \
    bash eval_3/scripts/brev/run_training_track_E_pi05_3celeb.sh; \
    echo "=== END (exit=$?) ==="; sleep 100000'

# Start the autopush watcher in another tmux window so each new
# checkpoint hits HF (and frees local disk).
tmux new-session -d -s autopush 'cd ~/LeMonkey && \
    bash eval_3/scripts/brev/autopush_checkpoints.sh \
        ~/outputs/train/pi05_track_E_m2_3celeb \
        HBOrtiz/pi05_eval3_track_E_m2_mahbod; \
    sleep 100000'

# Verify both up.
tmux ls
```

## 5. Reconnect later

```
# Don't press Ctrl+C while attached — that kills training. Use Ctrl+B D.
TERM=xterm-256color tmux attach -t pi05
TERM=xterm-256color tmux attach -t autopush

# Or just tail the logs without attaching.
tail -f ~/outputs/train/pi05_track_E_m2_3celeb.log
tail -f ~/autopush.log
```

## 6. Mid-run verification (the "is it actually working?" check)

The autopush watcher runs the Pi0.5 attention probe on every new
checkpoint and saves heatmaps under
`~/outputs/train/pi05_track_E_m2_3celeb/probes/step_NNNNN/`.

To inspect:

```
ssh a-toy-pi05 'ls ~/outputs/train/pi05_track_E_m2_3celeb/probes/'
scp -r a-toy-pi05:~/outputs/train/pi05_track_E_m2_3celeb/probes ./
```

Read `step_NNNNN/summary.txt` for the gating decision per checkpoint;
open the overlay PNGs to see *where* the model attends. Healthy:
argmax shifts to the prompted celeb's face as training progresses.

## 7. Disk safety

The autopush watcher keeps the **last `KEEP_LOCAL=2` checkpoints** locally
and deletes the rest after pushing. Override:

```
KEEP_LOCAL=3 bash eval_3/scripts/brev/autopush_checkpoints.sh ...
```

## 8. Handoff to Darius

When you're satisfied with a checkpoint:

```
# The HF artifact is at HBOrtiz/pi05_eval3_track_E_m2_mahbod@step-NNNNN
# Darius uses:
huggingface-cli download HBOrtiz/pi05_eval3_track_E_m2_mahbod \
    --revision step-NNNNN --local-dir /tmp/pi05_E_NNNNN
# Then bash eval_3/scripts/run_rollout.sh /tmp/pi05_E_NNNNN
# (run_rollout.sh accepts a local path as arg)
```
