# 2026-05-20 — Track 2 ObjectVLA Brev integration day

## What was run

Brev instance `cotrainerbboxarct2` (H100 PCIe 80 GB, driver 570.195.03 → CUDA 12.8 cap).
Track 2 wrapper integration following BREV_INTEGRATION_PLAYBOOK Phase A–D.

## Surprises and fixes

### 1. `torch 2.11.0+cu130` shipped in `lemonkey` conda env — incompatible with H100 driver

The Brev preinstalled env shipped torch compiled for CUDA 13.0. The H100 driver
570.195.03 caps at CUDA 12.8. `torch.cuda.is_available()` raised
`RuntimeError: The NVIDIA driver on your system is too old (found version 12080)`.

Fix: `pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall torch==2.11.0 torchvision==0.26.0`. lerobot's own pyproject pins
`pytorch-cu128` as the default linux index, so this matches the maintainers'
intent — the cu130 wheel was installed accidentally during env build.

Lingering version-conflict warnings (`numpy 2.4.4 vs lerobot wants <2.3.0`,
`setuptools 70.2.0 vs lerobot wants 71.0.0+`) — harmless so far, not chased.

### 2. Warm-PG v2 vision_tower silently loses warm weights via the HF Hub repo

The Hub upload `HBOrtiz/pi05_paligemma_celeb_warm_v2` was saved with the
**old** PaliGemma SigLIP layout — keys like
`...vision_tower.vision_model.embeddings.patch_embedding.weight` (with the
`.vision_model.` intermediate). lerobot 0.5.2 + transformers 5.9.0 expects
the **new** layout (no `.vision_model.` segment).

Load behaviour:

- **Hub snapshot**: strict load fails → non-strict fallback → vision_tower
  silently randomly initialised. Warm-PG vision weights are lost. The probe
  printed `loaded ok: 4.14B params` but missed all vision encoder weights.
- **Local copy** at `/home/shadeform/ckpts/warm_pg_v2_patched/`: lerobot's
  loader emits `WARNING:root:Vision embedding key might need handling` and
  **does** remap the keys. Probe `weight_norm=38.67` confirms loaded weights
  (random init for a SigLIP Conv2d would be O(1e-2)).

Fix: point `PRETRAINED=` in `eval_3/scripts/brev/run_training_track_2.sh` at
`/home/shadeform/ckpts/warm_pg_v2_patched`. Hub repo is unusable until
re-uploaded with the remapped state dict.

**Action item:** re-save the warm-PG with current-layout keys and re-push to
the Hub so the run is reproducible from a fresh box.

### 3. `lerobot[dataset]` extras missing from the preinstalled env

`from lerobot.datasets import make_dataset` raised `ImportError: 'datasets'
is required`. Fixed with `pip install datasets av`. The brev env was
installed without `lerobot[dataset]` extras.

### 4. The wrapper was a scaffold — until today, `run_training_track_2.sh` did nothing

`eval_3/scripts/track_2/lerobot_train_with_vl_cotrain.py` had 4
`[BREV_INTEGRATE]` markers and a `main()` that printed a checklist and
returned 0. The shell launcher correctly invoked it; the Python did no work.

Today's commit (`e8b9d10`) replaces the scaffold with a real loop:

- draccus-parse lerobot config from forwarded `--*` flags
- `make_dataset` / `make_policy` / `make_pre_post_processors` /
  `make_optimizer_and_scheduler` reused from lerobot
- `wrap_with_peft(peft_config=our_LoraConfig)` with optional `rank_pattern`
  per-layer (Enhancement B-4 wired)
- VL dataloader: PaliGemma processor + `VLPairsDataset` +
  `make_vl_collator` with suffix masking
- Modulo alternation: `step % (vl_ratio+1) == 0` → VQA, else flow.
  `vl_ratio=10` → period 11 (do NOT improvise; ObjectVLA published).
- `pi05_vqa_loss` now handles PEFT-wrapped traversal
  (`policy.base_model.model.model.paligemma_with_expert.paligemma`) and
  fills in the manual-splice fallback (no more `NotImplementedError`).
- EMA shadow updates per step, saved alongside checkpoints.
- `save_checkpoint` + `update_last_checkpoint` at `save_freq`; hub push at end.

## Smoke gate (in flight)

Running on `cotrainerbboxarct2` with `STEPS=200 BATCH_SIZE=8 NUM_WORKERS=8`.
Watching for:

- Both `flow_loss` and `vqa_loss` log lines fire (modulo 11)
- No `create_causal_mask got attention_mask=<dict>` crash
- Loss trending down by step 200
- VRAM peak under 90 GB

Will write a follow-up entry when the smoke + 24-h run land.

## Outstanding

- Re-upload warm-PG v2 with correct vision_tower keys.
- Run the ArcFace audit pipeline once Roham's bbox parquet is on HF
  (currently skipped → B-2/B-3 [WARN] fallback).
- Validate VL gradients actually flow through LoRA-on-language-model layers
  (parallel review agent is auditing this concern explicitly).
