# VLM Backbone LoRA Fine-Tune for Eval 3 (Celebrity Recognition)

> **Audience:** an implementing agent. This doc is a self-contained gameplan — choose a dataset, fine-tune the SmolVLM backbone with LoRA so it grounds celebrity *names* to faces, then retrain the VLA on top. Read [`PROJECT.md`](PROJECT.md) §"Eval 3" first if you don't have the constraints in your head.

---

## 1. Goal

Make the VLA's vision-language backbone reliably recognize the **TOY celebrities** (Taylor Swift, Barack Obama, Yann LeCun) and a broader pool of **popular celebrities** (for OOD runs 7–9 of Eval 3) so that prompts like `"Place the coke on Taylor Swift"` ground correctly to the face printed on the card on the table.

Constraints from [`PROJECT.md`](PROJECT.md):

- **VLA-only at inference.** No YOLO / face-recognition model / cloud VLM at demo day. The deployed policy is the VLA alone (see PROJECT.md lines 106–109).
- **Foundation models allowed at training time.** You may use them to label / curate data offline. Their outputs may be baked into the VLA's weights; the helper model itself must not run at inference.
- **Different weights per eval allowed** (PROJECT.md §3, lines 96–97 and 112–113). You can ship "celeb-LoRA-on" for Eval 3 and "LoRA-off" for Evals 1 / 2.

---

## 2. Approach (and why)

**Pick:** LoRA fine-tune of the VLM (vision tower + LM) on a name-labeled celebrity dataset, then retrain the VLA on the resulting backbone (LoRA merged into the weights).

Why **LoRA** over full fine-tune:

1. **Catastrophic forgetting.** The same backbone has to do Eval 1 (color naming) and Eval 2 (left/right/spatial). Full fine-tune on faces drifts the backbone toward face-discriminative features and silently regresses Eval 1/2. LoRA freezes the base and adds a low-rank delta — base behavior is preserved.
2. **Toggleability.** A LoRA adapter is a small file you can enable per eval. Full fine-tune commits you to one set of weights.
3. **Compute.** $200 Brev budget (PROJECT.md §6). LoRA on rank 8–16 is ~1 order of magnitude cheaper in VRAM and wall-clock than full fine-tune of a SmolVLM-class backbone.
4. **Sample efficiency.** Curated celeb sets are ~10k–50k images. LoRA tolerates this scale; full fine-tune wants more data and more careful LR scheduling.

> The HuggingFace SmolVLM2 blog claims full fine-tune > LoRA for the **500M** variant. That's correct *in their context* (in-domain video captioning, model is the final artifact). It does **not** transfer to our case (narrow celeb fine-tune, model is a backbone for a downstream VLA needing broad visual reasoning). Action-conditioned VLA training won't restore generic visual capabilities the backbone forgot — there's no caption loss to re-amortize them.

Why **not** auto-relabel teleop demos with a face recognizer (rejected approach, kept here so it doesn't get re-litigated):

- Requires recording teleop episodes for many celebrities. Real bottleneck — we don't have hundreds of teleop sessions.
- Adds a fragile pipeline (face-rec + label injection) for marginal benefit over giving the backbone direct celebrity grounding.

---

## 2.A How the LoRA interacts with SmolVLA's truncated backbone

A non-obvious architectural detail dictates *where* the LoRA is allowed to live and *how* we train it. From the SmolVLA paper (§3.1, §4.3, Fig. 1) and the lerobot source (`smolvlm_with_expert.py:88-90`):

1. **SmolVLA discards the upper LM layers of the VLM.** Default `num_vlm_layers = 16` — at load time it does `self.vlm.text_model.layers = self.vlm.text_model.layers[:16]`. Layers 16–31 are physically deleted, not just bypassed.
2. **The VLM is frozen during SmolVLA training.** Only the ~100M-parameter Action Expert (a separate flow-matching transformer) is trained, cross-attending to the truncated VLM's hidden states.
3. **The vision tower (SigLIP) is kept whole** — only the LM stack is truncated.

Implications for our LoRA:

- **`target_modules` must be restricted to LM layers 0–15 + the full vision tower.** A LoRA placed on layers 16–31 is silently discarded the moment SmolVLA loads the model. This is not a "distribution shift" issue — those weights cease to exist in the deployed model.
- **The chat-template format used at training time is largely irrelevant** to whether the knowledge transfers. SmolVLA bypasses the chat template entirely (see `processor_smolvla.py` — raw tokenize + concat, no `apply_chat_template`). What survives the format change is the *intermediate-layer encoding* of "this image contains person X" — which is exactly what gets baked into the LoRA's weight delta and what the Action Expert can later attend to.
- **After our LoRA lands, the Action Expert must be retrained** against the modified backbone (Step 6 below), because it was originally tuned against the *unmodified* layer-15 representations.

### Two training topologies (Approach A is what we ship; B is the documented fallback)

**Approach A — Full VLM + LoRA on layers 0–15 + next-token loss. ✅ Selected.**

- Load the full 32-layer SmolVLM2-500M; attach LoRA only to LM layers 0–15 attention/MLP projections + vision-tower attention projections.
- Train with standard supervised fine-tuning: `image + prompt → name` next-token prediction loss against the model's normal output head.
- Layers 16–31 act as **training-time scaffolding**: gradient flows back through them to give layers 0–15 a meaningful learning signal, even though those upper layers will be discarded after training.
- After training: drop layers 16–31, drop the output head, merge LoRA into layers 0–15 → produces exactly the shape SmolVLA loads.
- **Why selected:** standard `trl SFTTrainer` + `peft` tooling, easy to monitor (interpretable name outputs at every eval), low risk of subtle bugs, and the loss signal is well-understood. Right tool for the prototype phase.
- **Cost:** forward + backward through layers 16–31 is "wasted" compute at training time (paid in GPU-hours, not in correctness).

**Approach B — Truncated VLM (16 layers) + LoRA + custom downstream head + custom loss. 🔄 Documented fallback.**

- Truncate the VLM to 16 layers *first* (mirror SmolVLA's exact deployed shape).
- Attach a small custom head on layer-15's hidden state — e.g., a `Linear(hidden_dim → num_celebs)` classifier with cross-entropy loss, or a contrastive head with InfoNCE.
- Train LoRA on layers 0–15 + the new head jointly. After training, discard the head; merge LoRA into the truncated backbone.
- **Why this exists:** the layer-15 hidden states get directly optimized for the downstream task — no scaffolding indirection. Philosophically aligned with how SmolVLA itself uses the truncated VLM (frozen VLM + trainable downstream consumer).
- **When to switch to B:** if Approach A's TOY-PDF eval (Step 5 below) is weak — i.e., name accuracy is low even though the LoRA training loss converges nicely. That symptom means "the LoRA is encoding face→name in late-layer-friendly representations that don't survive truncation," and Approach B fixes the root cause.
- **Cost:** custom head + loss to design and validate; less off-the-shelf tooling; harder to debug; need to choose head architecture and loss objective.

### Honorable mention — Approach C (rejected)

Keep the full model + LoRA on *all* 32 layers. Standard out-of-the-box `peft` behavior. **Do not do this** — half of the trained LoRA delta lives in layers 16–31 and is deleted by SmolVLA's truncation, wasting half the compute and producing inconsistent results that depend on which layer the model happened to encode the celebrity knowledge in.

---

## 3. Dataset plan

### Primary: VGGFace2 (filtered)

- Source: <https://github.com/ox-vgg/vgg_face2> (canonical), Kaggle mirror typically named `vggface2` (the official VGG download has been gated; expect to use a mirror).
- ~9,131 named identities, ~3.3M images.
- Filter down to **the ~500–1,000 most-photographed identities** before training. Reasons: the long tail has heavy label noise, and the OOD candidate list TAs publish will skew toward famous people anyway.

### Supplements (must-do)

- **Yann LeCun:** scrape ~30–50 web images. He won't be in VGGFace2. Diverse poses / lighting / age. Save as a single identity folder named `yann_lecun` matching VGGFace2 structure.
- **Taylor Swift / Obama augmentation:** scrape ~20 extra recent images each — the VGGFace2 set may skew old. Helps held-out IID (Eval 3 runs 4–6).
- **OOD anchor identities:** once the TAs publish the OOD candidate list (watch [`project-1-vla`](https://robot-course-ethz.slack.com/archives/C0AULTPSDHS) per PROJECT.md §10), make sure each candidate is in the filtered identity set. Scrape if missing.

### Held-out eval set (do not train on)

- **LFW (Labeled Faces in the Wild):** <https://vis-www.cs.umass.edu/lfw/>. ~5.7k named identities, 13k images, contains Obama. Free, permissive.
- Use as a sanity-check name-grounding accuracy benchmark during training.

### Hard rejects (do not use)

- **CelebA** — identities are anonymized integer IDs, **no names**. Useless for name grounding. Common confusion.
- **MS-Celeb-1M** — withdrawn by Microsoft; legally problematic.
- **WebFace260M** — too large + license-gated for our budget.

---

## 4. Training repo

Two stages:

### Stage A — Prototype (notebook, fast iteration)

Use the HF cookbook SFT notebook to validate the data pipeline end-to-end on a tiny subset (e.g. 5 identities × 20 images):

- <https://github.com/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_smol_vlm_sft_trl.ipynb>

This shakes out tokenizer / chat-template / image-format issues in minutes.

> Skip these two cookbook notebooks for our task:
> - `fine_tuning_vlm_trl.ipynb` — generic VLM (Idefics-style), not SmolVLM-shaped.
> - `fine_tuning_vlm_dpo_smolvlm_instruct.ipynb` — DPO is preference alignment; you can't inject visual knowledge ("this face = Yann LeCun") with DPO. Wrong tool.

### Stage B — Real training run

Use [2U1/SmolVLM-Finetune](https://github.com/2U1/SmolVLM-Finetune). Reasons:

- Native LoRA / DoRA support (`--lora_enable`, `--vision_lora`).
- Separate learning rates for vision tower / projector / LM (`--vision_lr`, `--connector_lr`, `--learning_rate`) — the README explicitly recommends `vision_lr` ≈ 5–10× smaller than LM LR.
- DeepSpeed zero2/zero3, flash-attn, multi-image support.
- Caveat: as of the README snapshot (Jan 2025) it lists `Add support smolvlm2` in TODO. **It supports SmolVLM v1**. If you want SmolVLM**2** for the smallest-model bonus, follow the SmolVLM2 fine-tune Colab linked from the HF blog instead.

Local repo path (already cloned next to LeMonkey): `../SmolVLM-Finetune/`. See its [`README.md`](../../SmolVLM-Finetune/README.md).

---

## 5. Data format

`SmolVLM-Finetune` expects LLaVA-style JSON. One entry per (image, caption) pair:

```json
[
  {
    "id": "obama_0001",
    "image": "obama/0001.jpg",
    "conversations": [
      { "from": "human", "value": "<image>\nWho is shown in this photo?" },
      { "from": "gpt",   "value": "Barack Obama" }
    ]
  }
]
```

Vary the prompts so the model doesn't overfit to one phrasing:

- `"Who is shown in this photo?"`
- `"Identify the person in this image."`
- `"This is a photo of"` *(model completes name)*
- `"Place the coke on the photo of <name>"` — include this exact eval-style prompt for half the data so language conditioning matches demo day.

Image preprocessing:

- **Resize to 256×256** (PROJECT.md §9 / TA tip line 255). Faster, no quality penalty for VLAs at this resolution.
- **Random brightness / contrast / color jitter** as augmentation (PROJECT.md TA tip line 256). Demo-day lighting is unpredictable.

---

## 6. Step-by-step gameplan

### Step 1 — Acquire VGGFace2 + supplements

```
data/
  vggface2_filtered/        # ~500–1000 identities, top-N images each
    n000001/  (= one celeb)
    ...
  yann_lecun/               # ~30–50 manually scraped
  obama_extra/              # ~20 recent
  taylor_swift_extra/       # ~20 recent
  ood_candidates/           # populate once TA list is out
  lfw_eval/                 # held-out only
```

Filter VGGFace2 by image count per identity, keep top-N. Drop identities with <30 clean images.

### Step 2 — Build LLaVA JSON

Write a small Python script: walk the data tree, for each image emit one JSON entry with a randomly chosen prompt template. Output `train.json` and `val.json` (10% holdout, **disjoint identities** in val so you measure generalization, not memorization). Also build `lfw_eval.json` from the LFW identities that overlap the candidate set.

### Step 3 — Stage A smoke test (HF cookbook notebook)

Train on 5 identities × 20 images for 100 steps. Verify the model output flips from generic ("a person") to correct names. If this fails, the data format or prompt is wrong — fix before scaling up.

### Step 4 — Stage B real run (2U1/SmolVLM-Finetune)

Concrete config (LoRA on both LM and vision tower):

```bash
deepspeed src/train.py \
  --deepspeed scripts/zero2.json \
  --model_id HuggingFaceTB/SmolVLM-Instruct \
  --data_path data/train.json \
  --image_folder data/ \
  --output_dir out/celeb_lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --lora_enable True \
  --vision_lora True \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --learning_rate 1e-4 \
  --vision_lr 1e-5 \
  --connector_lr 1e-4 \
  --bf16 True \
  --max_seq_length 2048 \
  --report_to wandb
```

Notes:
- `vision_lr` 10× smaller than LM LR per the 2U1 README guidance.
- `lora_alpha = 2 × lora_rank` is a reasonable default.
- Use a single A100/H100 — multi-GPU costs more credits than it saves on this dataset size (PROJECT.md §6.4).

### Step 5 — Validate before VLA retraining

Run inference on:
- Held-out **VGGFace2 val** identities → expect strong generalization (same dataset distribution).
- **LFW** subset → cross-dataset sanity. Looser bar.
- The **TOY PDF images** themselves ([`docs/Eval_3_TOY_Celebrity_Images.pdf`](Eval_3_TOY_Celebrity_Images.pdf)) — the actual demo-day images. **This is the bar.** If the LoRA-fine-tuned model can't name all 15 TOY images correctly from the printed crops, do not advance to VLA retraining; iterate on the data first.
- **Eval 1 / 2 regression check:** prompt the fine-tuned VLM on a few generic color/spatial questions about scene images. If accuracy drops vs. the base model, your LoRA is over-specialized — reduce rank, lower LR, add general-image replay data.

### Step 6 — Merge LoRA, retrain VLA

```bash
bash scripts/merge_lora.sh   # produces a standalone backbone with LoRA baked in
```

Use the merged checkpoint as the VLA's backbone init and retrain the VLA from your existing pipeline. Keep the **base** (non-celeb) checkpoint around for Eval 1 / 2 if you observe regressions there — PROJECT.md explicitly allows different weights per eval.

---

## 7. Risks & fallbacks

| Risk | Symptom | Fallback |
|------|---------|----------|
| LoRA underfits — celebs not learned | Validation: model can't name TOY images | Increase rank to 32, alpha to 64. Train more epochs. Add more images per identity. |
| Catastrophic forgetting on Eval 1/2 | Color/spatial accuracy regresses | Lower rank, lower vision_lr, mix in 30–50% LAION/COCO general-image batches. |
| OOD generalization fails (runs 7–9) | Held-out OOD celebs not recognized | After TA list is out, add explicit OOD-candidate identities to training set. Don't expect zero-shot on identities we never showed the model. |
| VLA quality drops with new backbone | Eval 1/2 success rate drops in robot rollouts | Ship merged-LoRA backbone for Eval 3 only; use original backbone for Eval 1/2 (PROJECT.md §3). |
| LoRA results plateau below target | LFW accuracy stuck below ~70% | Try the **full fine-tune with replay** route: very low LR (1e-6 vision, 1e-5 LM), 30–50% non-face replay batches per step, early-stop on Eval 1/2 regression. Only if LoRA exhausted. |

---

## 8. References

- [`PROJECT.md`](PROJECT.md) — Eval 3 spec, training-time foundation-model rule, smallest-model bonus, compute budget.
- [`Eval_3_TOY_Celebrity_Images.pdf`](Eval_3_TOY_Celebrity_Images.pdf) — the 15 in-distribution images. Print and cut for demo day; also use directly as a held-out validation set during fine-tuning.
- VGGFace2 (canonical): <https://github.com/ox-vgg/vgg_face2>
- LFW: <https://vis-www.cs.umass.edu/lfw/>
- 2U1/SmolVLM-Finetune: <https://github.com/2U1/SmolVLM-Finetune> (also at `../SmolVLM-Finetune/`)
- HF cookbook SmolVLM SFT notebook: <https://github.com/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_smol_vlm_sft_trl.ipynb>
- SmolVLM (v1): <https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct>
- SmolVLM2 collection (newer, smallest-model-bonus candidate): see HF blog post linked from the SmolVLM2 announcement.
- LeRobot dataset v3 (downstream VLA training format): <https://huggingface.co/docs/lerobot/lerobot-dataset-v3>
