# Track 2 - Brev integration playbook

**Audience:** Sejohn (or whoever has brev_instance2 SSH), working through the
4 `[BREV_INTEGRATE]` markers in `lerobot_train_with_vl_cotrain.py`.

**Estimated total time:** ~4–5 h active work + 200-step smoke + then 24 h training.

**Goal:** convert the wrapper from "scaffold prints checklist and exits" → "fires
training with mixed 10:1 robot:VL batches on Pi0.5 + warm-PG + 200-celeb."

---

## Phase A - Setup (~30 min)

### A.1 - SSH + sync branch

```bash
ssh <brev_instance2>      # check HANDOVER_BREV_INSTANCE2.md for the host
cd ~/LeMonkey
git fetch origin
git checkout dev/SjohnU/track_2_objectvla
git pull origin dev/SjohnU/track_2_objectvla
```

### A.2 - Activate env + verify versions

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lemonkey

# Verify the version stack:
python <<'EOF'
import lerobot, peft, transformers, torch
print(f"lerobot      = {lerobot.__version__}")
print(f"peft         = {peft.__version__}")
print(f"transformers = {transformers.__version__}")
print(f"torch        = {torch.__version__}")
print(f"cuda         = {torch.version.cuda}")
print(f"gpu          = {torch.cuda.get_device_name(0)}")
EOF
```

**Record these numbers.** transformers version determines whether the
dict-attention-mask fallback in Phase D2 is needed.

### A.3 - Confirm Pi0.5 can load

```bash
python <<'EOF'
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
import torch
p = PI05Policy.from_pretrained("HBOrtiz/pi05_paligemma_celeb_warm_v2")
print(f"loaded ok, {sum(x.numel() for x in p.parameters())/1e9:.2f}B params")
print(f"layers: {len(p.model.paligemma_with_expert.paligemma.model.language_model.layers)}")
EOF
```

If the import path is wrong, try `lerobot.common.policies.pi05.modeling_pi05`
(older lerobot used `common.`).

**Record the exact attribute path** that works - the wrapper uses this to
attach hooks.

---

## Phase B - VL images pre-extraction (~30 min, mostly waiting)

### B.1 - Download Roham's VL dataset

```bash
mkdir -p ~/data/vl_pairs
python <<'EOF'
from huggingface_hub import snapshot_download
import os
p = snapshot_download(
    repo_id="HBOrtiz/eval3_vl_pairs_broad",
    repo_type="dataset",
    local_dir="~/data/vl_pairs",
    token=os.environ["HF_TOKEN"],
)
print(f"downloaded to {p}")
EOF
```

Roughly 1.15 GB. On brev_instance2's pipe, ~3–5 min.

### B.2 - Extract images.tar.zst

```bash
cd ~/data/vl_pairs
# zstd is the compressor; usually pre-installed.
tar --use-compress-program=unzstd -xf images.tar.zst
ls images/chunk-000/ | head -5
ls images/chunk-000/ | wc -l   # should be 29,445
```

### B.3 - Sanity-check the VL manifest loads + image_root resolution works

```bash
cd ~/LeMonkey
python <<'EOF'
import sys; sys.path.insert(0, "eval_3/scripts/track_2")
from lerobot_train_with_vl_cotrain import VLPairsDataset
# Construct without a processor - just verify schema + image lookup.
class DummyProcessor: pass
ds = VLPairsDataset(
    manifest_path_or_id="HBOrtiz/eval3_vl_pairs_broad",
    processor=DummyProcessor(),
    image_root=__import__("pathlib").Path("~/data/vl_pairs/images").expanduser(),
)
print(f"\nrows: {len(ds)}")
print(f"sample row 0:")
sample = ds[0]
print(f"  image: {sample['image'].size}")
print(f"  prompt: {sample['prompt'][:80]}...")
print(f"  target: {sample['target'][:80]}...")
print(f"  celeb_slug: {sample['celeb_slug']}")
EOF
```

PASS = sample row loads, image is non-blank (not the 224×224 gray fallback),
prompt has `<image>` token prepended.

If you see the gray-blank fallback `[WARN]` → image_root path is wrong, fix it.

---

## Phase C - Wrapper integration: the 4 markers (~3–4 h)

The wrapper currently does steps 1–5 below as scaffolding. You need to flesh
out steps 6–10 with the actual lerobot training loop integration.

### C.1 - Marker #1: locate the lerobot training entry point

The wrapper needs to hook into how lerobot starts training. Mahbod's M2 wrapper
did this via monkey-patching `lerobot.policies.factory.make_policy`. Find the
equivalent in the installed version:

```bash
python -c "import lerobot; print(lerobot.__file__)"
# Then grep for the train entry point:
find $(python -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))") -name "train.py" | head -3
find $(python -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))") -name "factory*.py" | head -3
```

Read those files. The training loop is typically a function called `train()`
or in a `Trainer` class. Look for:
- Where the dataloader is built (you'll add a second one for VL batches)
- Where the per-step `forward()` happens (you'll add the modulo alternation)
- Where the loss is `.backward()`d (you'll add EMA update after the step)

### C.2 - Marker #2: build the dual-dataloader + step alternation

This is the core of Track 2. Pseudocode (adapt to your lerobot version):

```python
# After lerobot builds its standard robot dataloader (robot_loader),
# we add a parallel VL dataloader:
vl_dataset = VLPairsDataset(args.vl_manifest, processor=paligemma_processor,
                             image_root=Path("~/data/vl_pairs/images").expanduser())
vl_loader = DataLoader(vl_dataset, batch_size=args.batch_size,
                       collate_fn=make_vl_collator(paligemma_processor),
                       shuffle=True, num_workers=4)
vl_iter = iter(vl_loader)

# Modify the training loop:
for step in range(args.steps):
    if step % (args.vl_ratio + 1) == 0:                # 1 VL batch per 10 robot
        try:
            batch = next(vl_iter)
        except StopIteration:
            vl_iter = iter(vl_loader)
            batch = next(vl_iter)
        loss = pi05_vqa_loss(policy.model, batch, _fallback_state)
        # Log it distinctly so the smoke gate can verify both fire:
        if step % 10 == 0:
            print(f"step {step:5d}  vqa_loss={loss.item():.4f}", flush=True)
    else:
        batch = next(robot_iter)
        loss = pi05_flow_loss(policy, batch)
        if step % 10 == 0:
            print(f"step {step:5d}  flow_loss={loss.item():.4f}", flush=True)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if ema is not None:
        ema.update(policy.model)
```

**Sacred constraint:** `vl_ratio + 1 = 11`. Do NOT change to 10 or 12 - the
modulo arithmetic produces the published 10:1 robot:VL ratio when
`step % 11 == 0` for VL batches.

### C.3 - Marker #3: layer-wise LoRA rank (B-4)

PEFT's `LoraConfig` supports `rank_pattern` (a `{regex: rank}` dict). The
wrapper already builds this dict from `layer_rank_track2.json`. To apply it:

```python
from peft import LoraConfig, get_peft_model

# Build rank_pattern from layer_rank_track2.json
rank_pattern = {
    rf".*language_model\.layers\.{i}\..*\.{tm}$": int(r)
    for i_str, r in json.load(open("eval_3/scripts/track_2/layer_rank_track2.json"))["layer_rank"].items()
    for i in [int(i_str)]
    for tm in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

cfg = LoraConfig(
    r=32,                            # default for layers not in rank_pattern
    rank_pattern=rank_pattern,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
peft_model = get_peft_model(base_model, cfg)
```

**If PEFT's version doesn't support `rank_pattern`** (it was added in PEFT
0.10.0), fall back to uniform `r=32` and skip B-4. Don't block the launch on
it.

### C.4 - Marker #4: dict-attention-mask fallback (only if smoke crashes)

The `pi05_vqa_loss` function tries the primary HF `model.forward()` path
first. If transformers ≥5.0 builds a dict-typed `attention_mask` that crashes
`PiGemmaModel.forward`, the function catches the error and sets a flag to
take the manual-splice fallback path - which is currently
`raise NotImplementedError`.

If smoke (Phase D) crashes with this error, here's the splice template (from
TRACK_B_WARMSTART.md §6, adapted for Pi0.5):

```python
def pi05_vqa_loss_manual_splice(model, batch):
    paligemma = model.model.paligemma_with_expert.paligemma
    vision_tower = paligemma.vision_tower
    multi_modal_projector = paligemma.multi_modal_projector
    language_model = paligemma.model.language_model
    lm_head = paligemma.lm_head

    # 1. Image embeddings.
    image_features = vision_tower(batch["pixel_values"]).last_hidden_state
    image_features = multi_modal_projector(image_features)   # (B, n_img_tokens, d_model)

    # 2. Text embeddings.
    inputs_embeds = language_model.embed_tokens(batch["input_ids"])

    # 3. Splice image features at <image> token positions.
    image_token_id = paligemma.config.image_token_index
    image_mask = (batch["input_ids"] == image_token_id)
    inputs_embeds[image_mask] = image_features.flatten(0, 1).to(inputs_embeds.dtype)

    # 4. Direct call with TENSOR attention_mask.
    outputs = language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return outputs.loss
```

Replace the `raise NotImplementedError` in `pi05_vqa_loss` with this. The
exact attribute path may need tweaking depending on the lerobot version's
Pi0.5 wrapper - use Roham's `train_paligemma_vqa.py` on
`origin/track-b-warmstart-vqa` as the canonical reference.

### C.5 - EMA shadow weights (B-7)

Already implemented as `EMAShadow` class. Just construct it and call `.update()`
once per step:

```python
ema = EMAShadow(policy.model, alpha=args.ema_alpha) if args.use_ema else None
# (inside the training loop, after optimizer.step():)
if ema is not None:
    ema.update(policy.model)

# At save time, dump shadow weights too:
ema.save(Path(args.output_dir) / "ema_shadow.pt")
```

For HF push, decide whether to use the EMA shadow weights or the live weights
as the deployable checkpoint. Standard practice: use EMA weights at inference.

---

## Phase D - Smoke test (~30 min, gates Phase E)

### D.1 - Launch 200-step smoke

```bash
cd ~/LeMonkey
mkdir -p ~/outputs
STEPS=200 BATCH_SIZE=8 bash eval_3/scripts/brev/run_training_track_2.sh \
    > ~/outputs/track_2_smoke.log 2>&1 &
SMOKE_PID=$!
tail -f ~/outputs/track_2_smoke.log
```

### D.2 - Verify the gates (from SMOKE_TEST.md §3.2)

In the live log, watch for:

| Gate | What to look for |
|---|---|
| Both losses fire | Lines like `step  10  flow_loss=...` AND `step  11  vqa_loss=...` (modulo 11) |
| No dict-mask crash | No `TypeError: create_causal_mask got attention_mask=<dict>` |
| Loss trending down | At step 200, `flow_loss` should be ~30% lower than step 10 |
| VRAM peak | `nvidia-smi --query-gpu=memory.used` < 90 GB |
| EMA active | `[ema] tracking N tensors` at startup |

### D.3 - If dict-mask crashes

```bash
kill $SMOKE_PID
# Edit pi05_vqa_loss to use the manual splice (Phase C.4).
# Re-run smoke.
```

### D.4 - If VRAM OOMs at bs=8

Drop further:
```bash
STEPS=200 BATCH_SIZE=4 bash eval_3/scripts/brev/run_training_track_2.sh
```

If even bs=4 OOMs, the wrapper has a memory leak somewhere (likely the EMA
shadow). Disable EMA:
```bash
USE_EMA=false STEPS=200 BATCH_SIZE=8 bash eval_3/scripts/brev/run_training_track_2.sh
```

---

## Phase E - Full 24 h launch (~30 sec to fire, then walk away)

### E.1 - Launch in tmux/nohup

```bash
nohup bash eval_3/scripts/brev/run_training_track_2.sh \
    > ~/outputs/track_2_full.log 2>&1 &
echo $! > ~/outputs/track_2.pid
echo "PID=$(cat ~/outputs/track_2.pid); ETA ~24h"
```

### E.2 - Monitoring schedule

| Hour | Check |
|---|---|
| +1 h | `flow_loss` decreasing, `vqa_loss` decreasing, no errors |
| +6 h | Curriculum switch happened at step 5000 (log line `[curriculum] phase 2`) |
| +12 h | Checkpoint pushed at step 15000 (about midway) |
| +24 h | Final checkpoint pushed to `HBOrtiz/pi05_eval3_objectvla` |

### E.3 - Abort gates during the run (from SMOKE_TEST.md §6)

If at any point:
- Loss curves flat or rising
- Sustained OOM warnings
- NaN losses appear

Then: kill the job, check `train_paligemma_vqa.py` for debugging patterns,
restart with adjusted hyperparams.

---

## Phase F - Strix deploy (~2 h, after E completes)

This is Task #20, separate from this playbook. Once `HBOrtiz/pi05_eval3_objectvla`
is pushed:

```bash
# On Strix:
bash eval_3/scripts/run_rollout_track_2.sh main
# Type a test prompt, verify rollout works.
```

---

## Quick-reference: integration ordering

```
Phase A setup            ─ 30 min
Phase B image extract    ─ 30 min   (background while you do Phase C)
Phase C.1 train.py read  ─ 30 min   (read installed lerobot training loop)
Phase C.2 step alt loop  ─ 1-1.5 h  (modulo, dataloaders, loss routing)
Phase C.3 layer-rank     ─ 30 min   (PEFT rank_pattern, or fall back)
Phase C.4 dict-mask      ─ 1 h ONLY IF smoke crashes (Phase D.3)
Phase C.5 EMA            ─ 15 min   (already templated)
Phase D   smoke          ─ 30 min   (run + watch logs)
Phase E   launch + wait  ─ 30 sec + 24h
                          ────────
Active work:              ~4-5h before smoke
Total wall to demo ready:  ~30h
```

---

## When to ask for help

- Phase C.1: if you can't find the lerobot training entry point, paste
  `python -c "import lerobot; print(lerobot.__file__)"` output to chat.
- Phase C.2: if step alternation has weird gradient interactions (e.g.,
  vqa_loss spikes after a robot batch), paste the loss curves.
- Phase C.4: dict-mask splice failing on a specific attribute path → paste
  the AttributeError.
- Phase D: any smoke gate failing → paste the relevant log lines.

The wrapper scaffold IS the spec - if you find yourself rewriting beyond
the markers, you're probably solving the wrong problem. Stick to the
modulo-alternation structure.

---

*Maintained 2026-05-20. Update after each Brev run with observed metrics
+ corrections to this playbook.*
