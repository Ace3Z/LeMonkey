# Track 2 smoke test plan

**Owner:** Sejohn · **Gate for:** Task #19 (full 24 h Brev launch) · **VM:** brev_instance2

Per [`TRACK_OBJECTVLA.md`](../../tracks/TRACK_OBJECTVLA.md) §4 + the enhanced
spec [`TRACK_OBJECTVLA_ENHANCED.md`](../../tracks/TRACK_OBJECTVLA_ENHANCED.md).
This doc enumerates the exact checks a 200-step smoke run must pass before
the 30 k step production launch.

---

## 0 · Pre-flight gates (must be green before running ANY smoke)

| Gate | Owner | Verify |
|---|---|---|
| Darius's Strix VRAM probe | Darius | Pi0.5 fits 16 GB, latency < 20 s on cold `pi05_base`. If RED → abort Track 2 entirely, pivot to SmolVLA. |
| Darius's VL pairs manifest | Darius | `HBOrtiz/eval3_vl_pairs_broad` on HF, parquet schema matches `lerobot_train_with_vl_cotrain.py` `VLPairsDataset.__init__` required cols (`image_path`, `prompt`, `target`) |
| Roham's robot-frame bboxes | Roham | parquet with `episode_idx`, `frame_idx`, `bbox_xyxy`, `target_celeb` columns for 200-celeb dataset |
| brev_instance2 access | - | SSH works; conda `lemonkey` env activates; `python -c "import lerobot, peft, transformers, torch; print(...)"` reports versions |
| warm-PG checkpoint accessible | Roham | `hf download HBOrtiz/pi05_paligemma_celeb_warm_v2` succeeds (or local cache present) |

---

## 1 · Data prep verification (run on dev box BEFORE Brev)

Run the ArcFace audit pipeline locally - if it doesn't smoke clean here, it
won't on Brev either.

### 1.1 - Pull Roham's bboxes (~30 s)

```bash
hf download HBOrtiz/<roham-bbox-repo> --repo-type dataset \
    --local-dir data/roham_bboxes_200celeb
```

Verify column schema matches the audit script's expected input.

### 1.2 - Run audit (~1 h)

```bash
python eval_3/scripts/track_2/arcface_audit_200celeb.py \
    --bbox-parquet data/roham_bboxes_200celeb/200celebs.parquet \
    --celeb-manifest data/arcface_toolkit/celeb_embeddings.json \
    --output eval_3/scripts/track_2/audit_200celeb.parquet
```

**PASS criteria:**
- No `[ERR]` messages
- Summary shows ≥ 85% of rows valid (target_cos not NaN)
- `would-keep at cos>=0.50` ≥ 85%
- `would-mark hard at gap<0.10` between 5–25%

**FAIL modes:**
- < 85% retention → inpainting noise worse than expected; lower threshold OR
  investigate (`[WARN]` lines list which celebs are failing).
- > 50% marked hard → centroid noise; either skip B-3 or raise gap threshold.

### 1.3 - Build keep_list + sample_weights (~5 min)

```bash
python eval_3/scripts/track_2/build_keep_list_and_weights.py \
    --audit-parquet eval_3/scripts/track_2/audit_200celeb.parquet \
    --output-dir eval_3/scripts/track_2/
```

**PASS criteria:**
- Exit code 0 (script aborts if retention < 70%, see `--min-retention`)
- `keep_episodes.txt` has ≥ 85% of source episodes
- `hardneg_weights.npy` writes successfully
- Summary JSON shows `frac_hard_of_kept` is 5–25%

---

## 2 · Wrapper scaffolding verification (run on dev box)

These checks confirm the Python wrapper imports and parses arguments correctly
WITHOUT touching the GPU.

### 2.1 - Argparser smoke (~30 s)

```bash
python eval_3/scripts/track_2/lerobot_train_with_vl_cotrain.py --help
```

**PASS criteria:** all Track 2 extra flags (`--vl_dataset.manifest`,
`--vl_ratio`, `--dataset.episodes_file`, etc.) appear in the help output.

### 2.2 - Component imports (~10 s)

```bash
python -c "
import sys; sys.path.insert(0, 'eval_3/scripts/track_2')
from lerobot_train_with_vl_cotrain import (
    VLPairsDataset, make_vl_collator, pi05_vqa_loss, pi05_flow_loss,
    apply_layer_wise_lora, EMAShadow,
)
print('OK all components importable')
"
```

### 2.3 - Curriculum sampler smoke (~5 s)

```bash
python -c "
import sys; sys.path.insert(0, 'eval_3/scripts/track_2')
from curriculum_sampler import build_curriculum_sampler
from pathlib import Path
s = build_curriculum_sampler(
    Path('eval_3/scripts/track_2/hardneg_weights.npy'),
    Path('eval_3/scripts/track_2/audit_200celeb.parquet'),
    switch_step=5000, num_samples=100,
)
s.set_step(0); phase1 = list(s)[:10]
s.set_step(6000); phase2 = list(s)[:10]
print(f'phase 1 sample: {phase1}')
print(f'phase 2 sample: {phase2}')
"
```

**PASS:** both phases produce non-empty index lists. Phase 1 may have fewer
unique indices (only easy episodes); phase 2 should be broader.

---

## 3 · The 200-step smoke run (on brev_instance2)

### 3.1 - Launch with reduced steps (~20 min on RTX PRO 6000)

```bash
# On brev_instance2 after env + dataset sync:
STEPS=200 BATCH_SIZE=8 bash eval_3/scripts/brev/run_training_track_2.sh
```

Watch the first 50 steps closely.

### 3.2 - Gates during the smoke run

| Step | Check | PASS criterion |
|---|---|---|
| 0–10 | Both loss types fire | log shows BOTH `flow_loss=X.X` and `vqa_loss=X.X` lines (modulo 11 = 10 robot + 1 VL) |
| 0–50 | No dict-attention-mask crash | no `TypeError: create_causal_mask got attention_mask=<dict>` |
| 50–200 | Loss curves trending down | `flow_loss` decreasing monotonically (small noise OK); `vqa_loss` also decreasing |
| 50 | GPU memory snapshot | `nvidia-smi`: VRAM peak < 90 GB (RTX PRO 6000 has 96 GB) |
| 100 | Layer-wise LoRA active | `requires_grad` on Gemma layer 10 params is True; param count > uniform r=32 baseline if B-4 wired |
| 100 | Curriculum sampler hint | phase 1 active (step < 5000 default); log line `[curriculum] phase 1` printed |
| 200 | EMA shadow weights tracked | `[ema] tracking N tensors` printed at startup; periodic `[ema]` updates OK |

### 3.3 - Fallback: dict-attention-mask crash

If step ~10 raises a TypeError about `create_causal_mask` and a dict-typed
`attention_mask`:

1. Confirm transformers version: `python -c "import transformers; print(transformers.__version__)"`. If ≥ 5.0, this is the known issue.
2. Don't restart cold. Inside `pi05_vqa_loss`, set `_fallback_state["use_manual_splice"] = True` and re-run the failing batch.
3. The manual-splice code path needs to:
   - Get image embeddings from `policy.model.paligemma_with_expert.paligemma.vision_tower(pixel_values)`
   - Project them through `policy.model.paligemma_with_expert.paligemma.multi_modal_projector`
   - Splice into the language input embeds at the `<image>` token positions
   - Build a tensor `attention_mask` of shape `(B, total_seq_len)`
   - Call `policy.model.paligemma_with_expert.paligemma.model.language_model(inputs_embeds=..., attention_mask=..., labels=...)` directly
   - Use the returned `.loss`

   **The exact attribute path verified on brev_instance2** is the integration
   work - see Roham's `train_paligemma_vqa.py` for the analogous offline pattern.

4. Verify the splice fallback produces the SAME magnitude `vqa_loss` as the
   primary path would have, on a synthetic 2-sample batch.

---

## 4 · Pi0.5 inference smoke (Strix-side, BEFORE the 24h training run)

Run this BEFORE committing the 24h Brev launch - Darius's Strix VRAM probe
already covers this, but verify your specific checkpoint loads cleanly:

```bash
# On Strix:
hf download HBOrtiz/pi05_paligemma_celeb_warm_v2 --local-dir /tmp/pi05_warm

# Simulate the rollout runner's first action with the warm-PG checkpoint
# as a dry-run for Track 2's eventual checkpoint format.
bash eval_3/scripts/run_rollout_track_2.sh /tmp/pi05_warm
# Type a test prompt: "Place the can on the photo of Yann LeCun."
```

**PASS:** rollout completes 25 s without crashing. Robot moves (won't pick
the right celeb - warm-PG isn't action-trained - but the inference loop
shouldn't crash).

---

## 5 · Go/no-go for 24 h launch

If ALL of the following are green, launch the 30 k step production run:

- [ ] Data prep (§1) ran successfully - keep_list.txt + hardneg_weights.npy exist
- [ ] Wrapper components import (§2)
- [ ] 200-step smoke ran with both loss types firing (§3.1, 3.2)
- [ ] No dict-attention-mask crash OR fallback splice path validated (§3.3)
- [ ] GPU memory peak < 90 GB
- [ ] EMA + curriculum + layer-rank all signal active in logs
- [ ] Strix can load a Pi0.5 checkpoint (§4)

Then launch:

```bash
# On brev_instance2:
nohup bash eval_3/scripts/brev/run_training_track_2.sh \
    > ~/outputs/track_2_full.log 2>&1 &
echo $! > ~/outputs/track_2.pid
```

Tail the log, check every ~hour for the first 4 hours, then every 4 hours.

---

## 6 · Abort gates DURING the 24 h run

Don't let a bad run consume the full budget. Abort early if:

| Symptom | Action |
|---|---|
| `flow_loss` not decreasing after step 2 k | abort, investigate dataset format / quantile stats |
| `vqa_loss` plateaued > 3.0 after step 5 k | abort, debug VQA collator / token format |
| GPU OOM | reduce `BATCH_SIZE` from 48 → 24 → 16, restart |
| Brev VM disconnected | reconnect via tmux/screen; check checkpoint files exist; resume from latest save |
| Random NaN losses | check EMA + bf16 interactions; may need to disable EMA |
| Loss curve identical to vanilla Track B's (no improvement) | confirm VL batches are actually firing (count `vqa_loss=` lines vs `flow_loss=` in log; should be ~1:10 ratio) |

---

*Maintained 2026-05-20. Update after each smoke run with observed metrics.*
