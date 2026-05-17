# DAgger collection + retraining strategy — SO-101 / SmolVLA / Eval 1

Local-only working doc (gitignored under `docs/report/`).
Synthesized from the recent literature on real-robot DAgger to maximize
the chance of fixing our "banana-in-corners → policy doesn't grasp"
failure mode under our time / compute budget.

## Problem we're solving

- Trained SmolVLA (450M, fine-tuned from `lerobot/smolvla_base` on 118
  teleop episodes) succeeds on the central bowl-picking case but **fails
  in corner zones**: the policy approaches the corner banana, lands the
  gripper in a near-but-wrong pose, and can't recover. This is the
  textbook **compounding-error** failure DAgger was invented to address
  (Ross et al. 2011).

- Pure additional teleop demos starting from home pose (Path A in our
  earlier discussion) don't fix this — they don't cover *recovery from
  off-target intermediate states*, which is exactly what's missing.

## What the recent literature actually says

Most relevant single source: **"What Matters in DAgger? An Empirical
Study on Improving Real-World Robot Learning with Human Corrections"**
(Wang et al., June 2025). They ran a controlled comparison on real
robot manipulation. Key findings:

1. **Compliant residual delta corrections (CR-DAgger) > take-over (HG-DAgger)** by 30–45% success rate.
   - Take-over has problems: force discontinuity at transitions,
     distribution shift from large corrections, indirect-correction
     errors via teleoperation.
   - Delta corrections layered on the policy maintain smooth transitions
     and distribution consistency.

2. **~50 episodes is enough** to get +50% success on contact-rich tasks.
   The CR-DAgger paper used **50–100 correction episodes total**.

3. **Residual-policy training >> fine-tuning the base policy.**
   - Residual policy (~2 MB add-on, learns corrections only): best.
   - Retrain from scratch on aggregated (BC + DAgger) data: stable but slow.
   - **Fine-tuning the base BC checkpoint on DAgger data DROPPED success rate by 30%**.
     Catastrophic forgetting + distribution shift kill it.

4. **Sample more densely right after intervention onset.**
   The first frames of an intervention contain the most information about
   where/how the policy was going wrong. Sampling 4× denser there
   improves correction reactivity.

5. **Intervene sparingly and at the failure moment** —
   not preemptively, not for the whole episode.

## What we have (and what's the gap)

| Element | Our setup | Gap vs. literature |
|---|---|---|
| Base policy | SmolVLA 450M fine-tuned 20k steps | ✓ |
| Hardware | SO-101 follower + leader, 1 camera, no force feedback | No force ⇒ can't do *true* CR-DAgger |
| DAgger interface | `dagger_record.py` with anchored-delta toggle | ≈ "delta correction" semantically — closest practical match to CR-DAgger we can do |
| Episode count plan | 8 / color = 24 total | Undershoots 50; need to scale to ~36–45 |
| Training plan (was) | Fine-tune from `HBOrtiz/smolvla_eval1` for 5k steps | Literature explicitly warns this drops accuracy ⇒ change |
| Residual-policy infrastructure | None | Add as future work; for now use option below |

## Plan

### Phase 1 — Collect ~36–45 DAgger episodes (~2.5 hrs)

12–15 episodes per color, **30 s** each.

Per-color recording protocol:

- **5 distinct failure zones** identified in Phase 0 (run a few diagnostic
  rollouts first if not already done): front-left, front-right, back-left,
  back-right, far-front. About 2–3 episodes per zone.
- **Mix the 4 trained phrasings** plus 1 unseen paraphrase (5 prompts
  total per color, balanced across episodes).
- **Intervene only when failure is imminent or happening**, not preemptively.
- **Cap intervention at ~30 % of episode time.** The recent paper found
  too-long interventions degrade to BC and lose DAgger's value.
- Use the **anchored-delta toggle** (SPACE press = on, SPACE press = off)
  in `dagger_record.py` — leader's incremental motion adds to follower's
  current pose; no need to physically pre-align the leader.
- **Discard wobbly / poorly-corrected episodes** rather than save them.

Target intervention rate per episode: **15–25 %** (warn / discard >40 %).

### Phase 2 — Training (Brev H100, ~2.5 hrs)

**Decision: retrain from `lerobot/smolvla_base`, not from our trained checkpoint.**

The "What Matters in DAgger?" paper showed fine-tuning a trained BC
policy on DAgger data hurts. Retraining from scratch on the merged
dataset (118 BC + ~40 DAgger episodes) is slower but stable and avoids
catastrophic forgetting.

Command (run on Brev):

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.empty_cameras=2 \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --dataset.repo_id=HBOrtiz/so101_eval1_all_v2 \
  --dataset.root=/home/shadeform/LeMonkey/datasets/eval1/all_v2 \
  --dataset.image_transforms.enable=true \
  --batch_size=192 \
  --steps=20000 \
  --save_freq=5000 \
  --output_dir=/home/shadeform/outputs/train/smolvla_eval1_v2 \
  --job_name=smolvla_eval1_v2 \
  --policy.device=cuda \
  --wandb.enable=true
```

Key changes vs the original training run:
- Same base (`lerobot/smolvla_base`), same step count (20k), same batch (192).
- **`--dataset.image_transforms.enable=true`** (was off before — fixes
  the no-augmentation gap we identified earlier).
- **New merged dataset `so101_eval1_all_v2`** = original 6 datasets
  (3 colors of 118 ep) + 3 DAgger color datasets, merged via
  `lerobot-edit-dataset --operation.type=merge`.

Optional sample-weight tweak (advanced): if lerobot supports per-frame
sampling weights, weight `is_intervention=1` frames 3× higher to
implement the "dense sampling near intervention onset" finding.

### Phase 3 — Evaluate

```bash
./scripts/probe_language_conditioning.py 020000   # on the v2 model
./scripts/eval_checkpoint.sh 020000 42            # 30-rollout eval
./scripts/compare_evals.py                        # vs the v1 model
```

Expected improvement signal:

| Metric | v1 (current) | v2 target |
|---|---|---|
| Overall success rate | TBD baseline | +20–40 pp on corner cases |
| Wrong-color distance (probe) | 15 | ≥30 |
| Trained-vs-OOD gap | TBD | ≤15 pp |

If v2 success rate ≤ v1: the residual-policy approach is the next escalation
(custom code, ~1 day of work).

## Reference numbers from collection so far

(fill in as we go)

| Color | Episodes | Avg intervention % | Notes |
|---|---|---|---|
| Blue  | _ | _ | _ |
| Red   | _ | _ | _ |
| Green | _ | _ | _ |

## Open questions / TBDs

- [ ] Does `lerobot-train` accept per-frame sample weights? If yes, set
      `is_intervention` frames to 3×.
- [ ] Do we need to upweight interventions or is uniform sampling fine?
      Literature suggests upweighting helps but isn't strictly required.
- [ ] Residual-policy implementation if v2 underperforms — defer until needed.

## Sources

- Wang et al. "What Matters in DAgger?" (June 2025) — arXiv 2506.16685
- Kelly et al. "HG-DAgger" (2019) — arXiv 1810.02890
- Ross et al. "A Reduction of Imitation Learning..." (2011) — original DAgger paper
- Hoque et al. "ThriftyDAgger" (2022) — context-switching rate concept
