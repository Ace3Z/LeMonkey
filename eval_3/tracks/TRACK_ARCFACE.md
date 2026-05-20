# Track 3 — ArcFace ↔ SigLIP alignment (Pi0.5)

**Owner:** Mahbod
**Backbone:** `lerobot/pi05_base` + frozen ArcFace `buffalo_l/w600k_r50` · **Output:** `HBOrtiz/pi05_eval3_arcface_aligned`
**ETA:** ~1 h projector pretrain + ~24 h Pi0.5 training.

This is the **novel research-contribution** track. Full strategy context: [`docs/report/EVAL_3_FINAL_PLAN.html`](../../docs/report/EVAL_3_FINAL_PLAN.html) §3 Track 3, research synthesis in [`docs/experiments/2026-05-20_pivot_research.md`](../../docs/experiments/2026-05-20_pivot_research.md).

Mahbod already built the M2 ArcFace cosine-distillation toolkit for SmolVLA (`HBOrtiz/eval3_m2_arcface_toolkit`). This track ports that ArcFace expertise to Pi0.5 — but as an alignment objective rather than a distillation drop-in.

---

## 1 · Why this track + why it's novel

Track 2 (ObjectVLA) hypothesizes the failure is *SigLIP doesn't know **where** the face is*. Track 3 hypothesizes the failure is *SigLIP doesn't have **face-discriminative geometry** at all*. Both can be true; running them independently lets the team decompose the failure.

**The mechanism is paper-validated; the teacher is not.** BlindVLA ([arxiv 2510.25616](https://arxiv.org/abs/2510.25616)) validated: a frozen MLP projector + per-token cosine alignment loss (λ=0.2) between LM hidden states and a frozen teacher's features improves VLA OOD generalization. But their teacher was C-RADIOv3 (a general-purpose ViT). **No published paper has used ArcFace specifically as the alignment teacher for a VLM/VLA.** Every face-MLLM paper (Face-LLaVA, FaceInsight, FaceLLM, Face-MLLM) uses landmark detectors or face-pretrained CLIP variants instead. If this works, it's a real contribution.

---

## 2 · Architecture

```
   wrist-cam frame
        │
        ├──────────────▶ InsightFace RetinaFace ──▶ face bbox + crop
        │                        │
        │                        ▼
        │                ArcFace w600k_r50 (FROZEN)
        │                        │  512-dim embedding
        │                        ▼
        │                2-layer MLP projector (FROZEN after warmup)
        │                512 → 2048 → 2048, GELU + LayerNorm
        │                        │  z_proj  (2048-dim)
        ▼                        │
   PaliGemma SigLIP ──▶ LM layers ──┐
        │                          │  h_pg = LM hidden state at the
        ▼                          │         face-token position,
   Gemma action expert             │         middle layer (~layer 9 of Gemma-2B)
        │                          ▼
   flow-matching         L_align = -cos( normalize(z_proj), normalize(h_pg) )
        │                          │
        ▼                          ▼
   L_flow_matching  +  λ · L_align       (λ = 0.2)
```

### Loss

$$L_{total} = L_{flow\_matching} + \lambda \cdot L_{align}, \qquad \lambda = 0.2$$

$$L_{align} = -\frac{1}{k}\sum_{j=1}^{k}\cos\Big(F.normalize(\text{MLP}(z^{arcface}_j)),\ F.normalize(h^{pg}_j)\Big)$$

Every choice cites a precedent:
- **λ = 0.2** — BlindVLA ablation sweet spot.
- **Frozen MLP projector, 512→2048→2048** — LLaVA-1.5 (MLP > linear projector); dims chosen so the projector lands in PaliGemma's LM hidden dim (2048, verified). BlindVLA used a frozen 3-layer; 2-layer is the LLaVA default and sufficient.
- **Mid-layer injection (~layer 9 of 18)** — BlindVLA injects at layer 16 of 32; we mirror the ~½-depth ratio.
- **Frozen ArcFace teacher** — unanimous in the literature (BlindVLA, Arc2Face, FR-distillation). ArcFace's hypersphere geometry was trained on ~10M faces; fine-tuning it on ~10k demos would destroy it.

---

## 3 · Step-by-step

### Step 1 — Projector pretrain dataset (~30 min, edna)

Pairs of `(face crop, identity label)` from the 192-celeb scraped bank. edna's `aug` conda env already has InsightFace. For each photo: detect + crop the largest face, store the crop path + celeb label.

**Deliverable:** `HBOrtiz/eval3_arcface_projector_pretrain` — small parquet, ~1.5k rows.

### Step 2 — Pretrain the MLP projector (~1 h, any GPU incl. dev box RTX 5090)

The projector learns to map ArcFace embeddings into PaliGemma's face-token feature space:
1. For each face crop: compute ArcFace `w600k_r50` 512-dim embedding (frozen).
2. Run a vanilla `lerobot/pi05_base` forward on the same crop, mean-pool the SigLIP image tokens that cover the face → target feature `h_pg^(0)` (the un-trained reference).
3. Train the MLP (512→2048→2048, GELU, LayerNorm) for ~1k steps to minimize `1 - cos(MLP(z), h_pg^(0))`.
4. Freeze the MLP. Save it.

**Deliverable:** `HBOrtiz/eval3_arcface_projector` — the frozen MLP state dict.

### Step 3 — Patch `modeling_pi05.py` for the auxiliary loss

In `PI05Policy.forward()` (the training path):
1. At each training step, for each frame, detect the face bbox via InsightFace (or precompute bboxes offline and load them with the batch — **strongly preferred for speed**).
2. Compute the frozen ArcFace embedding of the face crop.
3. Project through the frozen MLP → `z_proj`.
4. Capture the LM hidden state at the face-token position in layer ~9 (`PaliGemmaWithExpertModel` already exposes per-layer states via `compute_layer_complete` — see `modeling_pi05.py:226-296`).
5. Compute `L_align` (per-token cosine), add `0.2 * L_align` to the flow-matching loss.

**Precompute the bboxes + ArcFace embeddings offline** and ship them as extra columns in the dataset — running InsightFace inside the training loop would be a ~30-50 ms/frame bottleneck. Mahbod's M2 toolkit already has the ArcFace-embedding caching code; reuse it.

### Step 4 — Train (~24 h)

Same LoRA recipe as Track 1 + the auxiliary head. Needs a Brev VM — if a 3rd VM isn't provisioned, this runs on brev_instance2 *after* Track 2 finishes (likely too late for eval; flag this scheduling risk to Roham early).

```bash
--policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base
--policy.freeze_vision_encoder=True --policy.train_expert_only=False
--peft.method_type=LORA --peft.r=32
--peft.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
--policy.arcface_align=True --policy.arcface_lambda=0.2          # new flags
--policy.arcface_projector=HBOrtiz/eval3_arcface_projector
--dataset.repo_id=HBOrtiz/so101_eval3_aug_v3_200celebs
--batch_size=48 --steps=30000 --optimizer_lr=1e-5
--output_dir=outputs/pi05_arcface --policy.push_to_hub=True
--policy.repo_id=HBOrtiz/pi05_eval3_arcface_aligned
```

### Step 5 — Strix test

3-rollout protocol (TOY / IID / OOD) — coordinate with Darius.

---

## 4 · Risks

| Risk | Mitigation |
|---|---|
| Running InsightFace inside the training loop is too slow | Precompute bboxes + ArcFace embeddings offline, ship as dataset columns |
| Face bbox needed at **inference** too | Run RetinaFace at inference (~30-50 ms on Strix — well within the 20 s/rollout budget). Or train a no-bbox variant aligning all patches (lossier). |
| Printed-photo eval condition vs the celeb's clean web photos → ArcFace embeddings may not match | Apply print-style augmentation (the v3 aug pipeline's Lab/MTF/dither stack) to the projector-pretrain face crops |
| Aligning at the wrong LM layer | Layer 9 is a starting point (½-depth, BlindVLA-mirrored). If loss plateaus, sweep {6, 9, 12}. |
| 3rd Brev VM not provisioned → Track 3 can't run in time | Flag to Roham on day 1. Track 3 is the most schedule-fragile of the three. |

---

## 5 · Sources

- BlindVLA — [arxiv 2510.25616](https://arxiv.org/abs/2510.25616) + [code](https://github.com/CognitiveAISystems/BlindVLA)
- ArcFace — [arxiv 1801.07698](https://arxiv.org/abs/1801.07698)
- Arc2Face (ArcFace → cross-attention precedent) — [arxiv 2403.11641](https://arxiv.org/abs/2403.11641)
- LLaVA-1.5 (MLP > linear projector) — improved LLaVA paper
- Research synthesis — [`docs/experiments/2026-05-20_pivot_research.md`](../../docs/experiments/2026-05-20_pivot_research.md)
- Mahbod's prior M2 toolkit — `HBOrtiz/eval3_m2_arcface_toolkit`

---

*Scaffolded 2026-05-20. Owner: Mahbod. Status: assigned, not started.*
