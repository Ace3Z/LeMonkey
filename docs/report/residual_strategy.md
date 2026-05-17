# Residual policy strategy — SmolVLA + SO-101 / Eval 1

Local-only working doc (gitignored under `docs/report/`).
Synthesized from CR-DAgger paper (arXiv 2506.16685 v5), upstream lerobot
audit, and OSS prior art (yifan-hou/cr-dagger, ankile/robust-rearrangement,
tongzhoumu/policy_decorator).

## 1. Goal & success criteria

Train a small residual head on top of the **frozen** `HBOrtiz/smolvla_eval1`
base policy that learns to correct its outputs on the failure modes our
DAgger episodes captured (banana in corners → policy doesn't grasp).

**Success metric:** rollout success rate on the 30-rollout
`eval_checkpoint.sh` harness goes up by ≥ +15 percentage points vs the v1
base model alone, especially on out-of-distribution prompts and corner
banana positions.

**Soft target** (per CR-DAgger paper): the residual achieves the +50 to
+60 pp gains they reported on their tasks. Realistic for us:
+20 to +40 pp.

## 2. Architecture

```
                    ┌──────────────────────────────┐
       observation ─┤   FROZEN  base_smolvla       ├─→ base_action_chunk[50,6]
                    │   (450M, never updated)      │
                    └──────────────────────────────┘
                               │
                       (extract base_action[step_idx])
                               │
                    ┌──────────────────────────────┐
       state[6]   ──┤  vision_encoder(image)       │
       image      ──┤   (frozen, reused from base) │
                    │                              │
       base_action─→│   MLP (small, ~2M params)    │
                    │                              │
                    └──────────────────────────────┘
                               │
                          residual[6]
                               │
                          clip(±BOUND)
                               │
       base_action ────────────┼──→  + ──→ final_action[6] → robot
                               ▼
```

### Inputs to the residual

- **Joint state** (`observation.state`): (6,) float — current arm pose.
- **Image features** (`observation.images.camera1`): extracted from the
  frozen SmolVLM2 vision encoder used by SmolVLA. We tap into the
  pre-pooled image embedding (~512-dim feature vector after the encoder).
- **Base action**: (6,) float — what SmolVLA wants the joints to be next.

### Output

- **Residual delta**: (6,) float — joint-position correction added
  elementwise to the base action.

### Clipping

Following Policy Decorator (Mu et al., NeurIPS 2024): the residual is
clipped per-joint to a small bound (default ±5°/joint and ±5 for gripper)
to prevent the residual from going feral when its inputs are OOD.
Critical for stability — log a `[WARN]` if clipping triggers more than
50% of frames in an episode (signal that the residual is mistraining).

### Why image-aware (not state-only)

Our failure mode is the policy mis-perceiving the corner banana position.
A state-only residual sees the wrong base action but has no signal about
where the banana actually is, so it can't direct the correction
spatially. CR-DAgger's residual reuses the frozen vision encoder for
exactly this reason; we follow suit.

## 3. Training

### Data

Train on the **merged DAgger episodes only** (35 episodes / ~13.3k
frames). Per CR-DAgger paper §5.1: residual policy is "trained from
scratch on correction data only" — including zero-residual labels for
the non-intervention frames within those episodes.

```
For each frame in DAgger dataset:
  obs        = (image, state)
  recorded_a = action that was actually sent (teleop OR policy)
  is_int     = 1 if user was driving via leader, 0 if policy was driving
  base_a     = frozen_base_smolvla(obs)         # forward pass
  target_residual = recorded_a - base_a         # signed delta
  Loss = MSE( residual_net(image, state, base_a), target_residual )
```

For `is_int=0` frames, `recorded_a ≈ base_a` so the target is naturally
near-zero — the residual learns "be quiet when the policy is right."
For `is_int=1` frames, the human's action diverges from the base, giving
a non-trivial target.

### Hyperparameters (initial picks; revise after dry runs)

| Parameter | Value | Source |
|---|---|---|
| Architecture | MLP, 4 layers, hidden=256, GELU | borrowed from Policy Decorator + intuition |
| Trainable params | ~2M (matches CR-DAgger spirit) | |
| Optimizer | AdamW(lr=3e-4, β=0.9/0.95, wd=1e-4) | DETR/CR-DAgger style |
| Batch size | 64 | small enough for any GPU |
| Steps | 5,000 | one full pass through 13k frames at bs=64 ≈ 200 steps; we want ~25 epochs |
| LR schedule | cosine to 1e-5 | standard |
| Loss | MSE on action delta | most direct |
| Weighting | up-weight is_int=1 frames 2× | CR-DAgger §A.2 hinted at the value of correction-frame emphasis |
| Residual clip | ±5° per arm joint, ±5 (range_0_100) for gripper | Policy Decorator finding |

### Hardware

Training is small: ~2M params, ~13k frames, ~5k steps. Should run on
the GTX 1660 SUPER locally (~30 min) OR Brev H100 (~5 min). Local is
fine for residuals.

## 4. Inference

```python
class ResidualSmolVLAWrapper:
    """Inference-only wrapper that adds a residual to a frozen SmolVLA's actions."""
    def __init__(self, base_smolvla_path, residual_ckpt):
        self.base = SmolVLAPolicy.from_pretrained(base_smolvla_path).eval().cuda()
        self.residual = ResidualHead.from_pretrained(residual_ckpt).eval().cuda()
        # freeze everything; we only do forward
        for p in self.base.parameters(): p.requires_grad = False
        for p in self.residual.parameters(): p.requires_grad = False

    def select_action(self, obs):
        with torch.inference_mode():
            base_a = self.base.select_action(obs)            # (6,)
            img_feat = self._extract_image_features(obs)     # (512,)
            state = obs["observation.state"]
            r = self.residual(img_feat, state, base_a)       # (6,)
            r = torch.clamp(r, -CLIP, CLIP)
            return base_a + r
```

Hooks into the existing `lerobot-record --policy.path=...` flow only if
we register the wrapper as a recognized policy type; alternatively, we
write a small standalone rollout runner (matches our `dagger_record.py`
pattern) that imports this directly. **Standalone runner is simpler and
chosen.**

## 5. Plan & verification steps

| Step | Build | Verify |
|---|---|---|
| 1 | Write `residual_head.py` — small MLP module + load/save | `python -c "from residual_head import ...; m = ResidualHead(); print(sum(p.numel() for p in m.parameters()))"` returns ~2M |
| 2 | Write `train_residual.py` — loads merged DAgger dataset, runs frozen base, computes target deltas, trains MLP, saves checkpoint | Test on a tiny 50-frame subset; loss should drop below 1.0 (rough — check by visualization) |
| 3 | Write `inference_residual.py` — wrapper that loads base + residual, exposes `select_action` | Smoke test: same image+state through wrapper produces an action close to base alone (residual should be ~0 for in-dist inputs) |
| 4 | Modify or fork `eval_checkpoint.sh` to use the wrapper for rollouts | Rollout starts and runs without crash |
| 5 | Train on Brev (or local) | wandb shows loss decreasing |
| 6 | Eval the residual+base composite vs base alone using `eval_checkpoint.sh` | Success rate ≥ +15 pp on corner positions (target +30) |

Each step has an explicit verifier per CLAUDE.md §4.

## 6. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Image-feature extraction breaks (SmolVLA's encoder is private) | medium | Read modeling_smolvla.py, find the encoder, extract pre-pooled features. Fallback (with `[WARN]`): state-only residual. |
| Residual goes feral on OOD inputs | medium | Per-joint clipping (already in design); also log when clip activates >50% of frames. |
| Trained residual doesn't help (or hurts) | medium | Compare against v1 base in `compare_evals.py`; ship whichever wins. |
| Training instability (hidden=256 too big) | low | If loss diverges, drop to hidden=128. |
| 2M params overkill for 13k frames | possible | Final residual size is verifiable; if overfitting, reduce. |

## 7. Rollout / eval criterion

Single source of truth: rerun `eval_checkpoint.sh 020000 42` with the
**residual wrapper** as the policy, then `compare_evals.py`. The harness
already logs trained vs OOD performance, success by color, etc.

If residual+base **wins by ≥ 5 successes**: ship residual.
If within ±2: tie, prefer base for simplicity.
If residual loses: revert; we still have the v1 base on HF Hub.

## 8. Source references

- CR-DAgger v5: https://arxiv.org/html/2506.16685v5
- yifan-hou/cr-dagger: https://github.com/yifan-hou/cr-dagger
- Policy Decorator: https://github.com/tongzhoumu/policy_decorator (residual clipping)
- ankile/robust-rearrangement: https://github.com/ankile/robust-rearrangement (per-step-on-chunk)
- lerobot policy factory (where we'd register if we wanted to): src/lerobot/policies/factory.py:85-219
- SmolVLA modeling: src/lerobot/policies/smolvla/modeling_smolvla.py
