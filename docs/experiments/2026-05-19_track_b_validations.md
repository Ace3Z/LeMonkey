# 2026-05-19 — Track B (Pi0.5) pre-flight validations

Three parallel agents launched to validate load-bearing claims before kicking off the 24h Pi0.5 training run. Findings changed the recipe in three places.

## What I asked

1. **PaliGemma celebrity prior + frozen-VLM viability** — does Pi0.5's WebLI prior survive `train_expert_only=True`?
2. **Camera count** — what does `lerobot/pi05_base` actually expect for `empty_cameras`?
3. **Quantile stats + VRAM** — is our merger's quantile aggregation correct? Will Pi0.5 fit on Brev + Strix?

## What the agents found

### 1 — Frozen VLM is contradicted by the paper itself

Citing Pi0.5-KI (arxiv 2505.23705 §4 + Fig 4a/8 + pi.website/research/knowledge_insulation):

> "Fully freezing the VLM yields ~0% task performance."

Pi0.5-KI's recipe is **not** "freeze VLM" — it's "update VLM via discrete FAST action tokens AND stop-gradients from the continuous action expert." Web/VQA co-training protects OOD generalization (Fig 8 caption).

Plus: PaliGemma has no published celebrity-recognition benchmarks (PaliGemma 2 paper arxiv 2412.03555). WebLI was DLP-filtered for sensitive identifiers. Long-tail entity literature (Parashar et al. arxiv 2401.12425) shows web-scale VLMs systematically fail on long-tail named entities. Our 2026-05-09 0/14 zero-shot probe is consistent with this.

**Fix:** drop `train_expert_only=True`. Use LoRA via lerobot's `--peft` to update the PaliGemma LLM via low-rank adapters. Splits the difference between full SFT (catastrophic forgetting) and frozen (~0% per the paper).

### 2 — Pi0.5_base trained with 3 camera slots, not 4

From [`lerobot/pi05_base` HF config.json](https://huggingface.co/lerobot/pi05_base/raw/main/config.json):
- `observation.images.base_0_rgb`
- `observation.images.left_wrist_0_rgb`
- `observation.images.right_wrist_0_rgb`

**Fix:**
- `--policy.empty_cameras=2` (fills 2 missing slots with masked zeros)
- `--dataset.rename_map='{"observation.images.camera1": "observation.images.right_wrist_0_rgb"}'` (maps our 1 wrist cam onto a pretrained slot embedding)

Verified at `configuration_pi05.py:125-133` (validate_features adds empty_camera_{i} keys) and `modeling_pi05.py:1199-1204` (fills missing keys with `-1` tensor + zero attention mask).

### 3 — Quantile stats wrong; VRAM fits

**Quantile stats:** Our custom merger aggregates per-ep quantiles with `q01=min, q99=max, q50=median` across episodes. This is **wrong**. Example failure: 50 eps with q99=0.5 + 1 ep with q99=10.0 → merger reports global q99=10.0; true global q99 over the joined frames is ~0.52. Pi0.5 reads quantiles from `meta/stats.json` and clips by them — bad clipping = training instability.

**Fix:** run `third_party/lerobot/src/lerobot/scripts/augment_dataset_quantile_stats.py --repo-id local/eval3_track3_v3 --root .../eval3_track3_v3_merged --overwrite`. Computes exact quantiles over all 5,053,972 frames. Then re-push dataset to HF.

**VRAM:** training ~7 GB on Brev's 96 GB (fits with batch=24). Inference ~4.8 GB on Strix 16 GB (fits with headroom). Bf16 weights + LoRA grads + AdamW state + activations w/ grad_ckpt.

## What this means for the team

- Track B is still worth running — the team's "Pi0.5 is safer than SmolVLA" instinct is consistent with capacity arguments, but the original recipe was wrong on three points.
- LoRA fine-tune of PaliGemma during training is the research-grounded middle ground. Avoids the failure mode the Pi0.5-KI paper documents while preserving most of the pretrained prior.
- We're now expected to FINISH Track B around Day 3 morning per the plan. Day 4 dry-run still feasible.

## Files updated

- `eval_3/tracks/TRACK_B.md` — full Track B explanation with all 3 validation findings
- `eval_3/scripts/brev/run_training_track_B.sh` — corrected launch script
- This file (`docs/experiments/2026-05-19_track_b_validations.md`) — durable log

## Branch

Work happening on `track-b-pi05` off `main`.

## Agent transcripts (raw)

Full transcripts available at:
- `/tmp/claude-1000/.../tasks/a58dc8ae8bdd5428f.output` — PaliGemma prior agent
- `/tmp/claude-1000/.../tasks/aef33071d6aaf9464.output` — camera count agent
- `/tmp/claude-1000/.../tasks/a2901db079fac3eca.output` — quantile + VRAM agent

These are ephemeral; the synthesised findings above are the durable record.
