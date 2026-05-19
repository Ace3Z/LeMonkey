# Pi0.5 gating experiments — 2026-05-20

Two gating experiments to decide whether to commit Brev hours to a
Pi0.5 + M2 + KLAL training run on `a-toy-pi05` (H100 80GB).

## TL;DR — proceed with training

- **G1 (vanilla PaliGemma attention probe): PASS** — argmax patch SHIFTS
  across the 3 celeb prompts at layers 6, 10, and 14. The cross-modal
  attention channel exists; M2+KLAL has something real to supervise.
  Qualitatively different from SmolVLM2's pathology (constant argmax
  at (1,7) across all prompts).
- **G2 (zero-shot PaliGemma celeb VQA): MARGINAL** — Obama recognized
  (1/3 verification), Swift and LeCun NOT recognized. Positional VQA
  fails (0/3 — model only says "person"). PaliGemma's WebLI prior
  knows Obama but not the other two as named entities.

**Combined diagnosis:** PaliGemma has the architectural cross-modal
attention machinery (G1) but lacks pre-existing name-knowledge for 2/3
celebs (G2). This is fine for M2+KLAL training because KLAL **teaches
the name-to-face binding directly from bbox-supervised attention** — it
doesn't need PaliGemma to already know the celebrity names. The action
dataset has 9.4k episodes × 5 paraphrases per name = thousands of
gradient steps where KLAL pushes the right name-token → right face-patch.

## Numerical findings — G1

argmax patch (row, col) from name-token attention to camera1 patches:

| layer | swift  | obama  | lecun  | shifts? |
|-------|--------|--------|--------|---------|
| 6     | (9,13) | (2,13) | (2,13) | ✓ (swift differs) |
| 10    | (8,12) | (9,8)  | (9,15) | ✓ (all 3 differ) |
| 14    | (9,13) | (15,0) | (15,3) | ✓ (all 3 differ) |
| 17    | (15,3) | (15,3) | (15,3) | ✗ (converges to sink) |

LeCun is at columns 0-5, Obama 5-10, Swift 10-15. Layer 10:
- Swift argmax col 12 → **correct quadrant** (right side, where Swift is)
- Obama argmax col 8 → **correct quadrant** (middle, where Obama is)
- LeCun argmax col 15 → **wrong** (Swift's side, not LeCun's)

So 2 of 3 prompts already roughly point at the right celeb without
any training. KLAL should sharpen the correct ones and fix LeCun.

Max attention values are 0.005-0.04 — well above the sink threshold
but still diffuse. KLAL with λ=1.0 should peak them on the target.

## Numerical findings — G2

Scene VQA on the same input frame (3 celebs visible):

| question | answer | hit |
|---|---|---|
| Who is in this image? | "person" | — |
| Who is on the LEFT? (expect LeCun) | "person" | ✗ |
| Who is in the MIDDLE? (expect Obama) | "person" | ✗ |
| Who is on the RIGHT? (expect Swift) | "person" | ✗ |
| Is Taylor Swift visible? | "no" | ✗ |
| Is Barack Obama visible? | "yes" | ✓ |
| Is Yann LeCun visible? | "no" | ✗ |

Open-ended VQA: 0/3 correct. Verification VQA: 1/3 correct.

The model is conservative — defaults to "no" or "person". Even Obama
recognition is binary (yes/no), not name-localization.

## Why we still proceed

KLAL's supervised loss does NOT depend on PaliGemma already knowing
the names. Its bbox-derived target distribution `P` is built from our
face_labels (face detector + per-variant `new_layout_camera_lmr`), not
from any name-knowledge inside the model. KLAL says:

> "When the prompt contains 'Taylor Swift' and Swift's face is at this
> bbox, the name-token's attention to image patches MUST be peaked
> around that bbox."

This trains the model to **bind name→face from scratch via the action
dataset**, using bbox-supervised attention regularization. The fact
that PaliGemma already has the architectural attention bandwidth (G1
pass) means there's a working channel to shape.

## Recommended training launch

```bash
ssh a-toy-pi05
# In tmux to survive disconnect:
tmux new-session -d -s pi05 'cd ~/LeMonkey && \
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate pi05 && \
    set -a && source .env && set +a && \
    bash eval_3/scripts/brev/run_training_track_E_pi05_3celeb.sh; \
    echo END; sleep 100000'

# Auto-push + auto-probe watcher (saves disk + checks model is learning):
tmux new-session -d -s autopush 'cd ~/LeMonkey && \
    bash eval_3/scripts/brev/autopush_checkpoints.sh \
        ~/outputs/train/pi05_track_E_m2_3celeb \
        HBOrtiz/pi05_eval3_track_E_m2_mahbod; sleep 100000'
```

## Mid-run pass criterion

The autopush watcher runs `attention_map_probe_paligemma.py` on every
new checkpoint. **By step ~10k**, expect:

- argmax across the 3 prompts hits 3 DISTINCT (row, col) positions
  at layer ≥10
- argmax falls **within the correct celeb's bbox** (or within 1 patch)
  on ≥2/3 prompts
- KLAL loss decreasing steadily (logged inline)
- m2_loss approaching -0.7 (mean_cos ≥ 0.7)

**If step 10k fails the discrimination test, kill the run** and
consider:
1. Increase KLAL λ from 1.0 to 2.0
2. Add VQA pretraining stage (LoRA on celeb-face VQA dataset)
3. Lower M2_LAMBDA so M2 doesn't compete with KLAL

## Files referenced

- `eval_3/scripts/attention_map_probe_paligemma.py` (G1 — direct
  PaliGemma probe via standard transformers API, bypasses lerobot's
  pi05 wrapper compat issues)
- `eval_3/scripts/probe_paligemma_celeb_vqa_scene.py` (G2 — scene VQA)
- Heatmaps under this dir: `<celeb>_layer<NN>_heatmap.png` and
  `_overlay.png` for layers 6/10/14/17 across the 3 prompts.
