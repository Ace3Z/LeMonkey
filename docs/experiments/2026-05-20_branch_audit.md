# 2026-05-20 — Four-branch documentation audit

Audited `track-b-pi05`, `main` (= `dev/SjohnU/track_2_objectvla`, same commit),
and `dev/m2-arcface-toolkit` with four parallel read-only agents. Goal: find
what the team's strategy docs have missed.

---

## THE HEADLINE: KLAL already exists and is training

**`dev/m2-arcface-toolkit` already has a complete, audited, co-trained KLAL
implementation.** The team's strategy docs (`TRACK_ARCFACE.md`, the attention-
routing diagnosis) describe KLAL as something to *build*. It is built.

- Code: `eval_3/aug/m2_klal.py` (232 lines) + integrated in
  `eval_3/aug/m2_pi05_policy_wrapper.py`.
- It is **"Track E"** = Pi0.5 + M2 (ArcFace distillation) + KLAL, introduced
  commit `dd1d981`.
- Loss: `total = action_loss + lam_m2*m2 + klal`, one backward pass —
  co-trained, exactly as the strategy prescribes.
- KLAL: hooks `q_proj`/`k_proj` on PaliGemma LM layers, recomputes
  `softmax(QK^T/√d)` (GQA-aware), head-averages, slices name-token rows × 256
  image-patch cols, KL-divergence against a Gaussian (σ=1.5 patches) on the
  face-bbox centroid. Bbox source = the M2 toolkit's `face_labels/`.
- Status: smoke-tested (`mean_cos` +0.011→+0.562 over 800 steps), 3-agent
  audited, retrain config finalized, **mid-flight on Brev**. The decisive
  step-~10k face-binding probe has not yet returned.

**Implication:** do NOT re-build KLAL. `TRACK_ARCFACE.md` (which assigns Mahbod
to build it from scratch) is obsolete — Track E *is* that work, further along.

---

## CRITICAL — open risks / unimplemented work

1. **Track 2's co-train training loop is still a scaffold.**
   `lerobot_train_with_vl_cotrain.py` `main()` prints an integration checklist
   and `return 0` — it does not train. The modulo-11 robot/VL alternation loop
   does not exist. 4 `[BREV_INTEGRATE]` blockers remain (lerobot train hook,
   version check, layer-wise LoRA `rank_pattern` not wired, dict-attention-mask
   splice = `raise NotImplementedError`). ~4-5 h Brev work. **Track 2 is not
   runnable yet.**

2. **The Strix VRAM + latency probe has never been run.**
   `probe_pi05_strix.py` exists; no results. It is a kill-switch — thresholds
   <14 GB VRAM and <20 s p95 latency (the TA rule). If Pi0.5 fails either, every
   Pi0.5 track is disqualified or must pivot to SmolVLA. **This is the biggest
   unmeasured risk in the whole project.**

3. **transformers 4.55 breaks Pi0.5 without compat patches.**
   `pi05_inference_patch.apply()` must run before any `PI05Policy` construction
   (two patches: `embed_image` pooler_output, `create_causal_mask` kwarg
   rename). Team-wide gotcha — affects rollout and any Pi0.5 launch.

---

## CRITICAL — corroborations and invalidations

4. **The attention-routing diagnosis is independently corroborated.**
   Mahbod's step-10000 attention probe: name-token argmax = patch `(1,7)`
   (top-right background corner) for *every* prompt at *every* probed layer
   (9/11/13/15); attention below uniform (1/64). The VLM face-detection probes
   show warm-VLM v1 *and* v2 are sink-locked — "Path B (frozen VLM) is dead."
   SmolVLA Track D step-25000 identical. Our diagnosis is solid and now has
   mechanistic evidence from a second source.

5. **M2-detach bug invalidates pre-fix Pi0.5 M2 numbers.** Before commit
   `5c65ce2` (2026-05-20 12:12) the Pi0.5 M2 capture hook stored
   `h.detach().clone()` — M2 loss was logged but trained zero parameters. Any
   Pi0.5 `mean_cos` before that commit is invalid (incl. the step-1000 Track E
   probe's M2 numbers). **The 0.88/≈0.85 mean-cosine is from SmolVLA Track D,
   which used a correct hook — that number is real and defensible.** KLAL was
   unaffected (separate hookset).

6. **The 2026-05-18 M2 validation report predicted M2 would fail.** It listed
   4 blockers and cited BlindVLA §7.6 (the method does not help fine-grained
   under-represented concepts — exactly this task). M2 was built anyway; the
   step-10000 probe + failed Strix rollout vindicated the prediction. M2-alone
   is a confirmed dead end.

---

## MEDIUM — doc contradictions and gotchas

7. **`TRACK_B.md` is internally contradictory** — r=16 vs r=32, attention-only
   vs 7-projection LoRA, batch 24 vs 48, PaliGemma 1 vs 2, "3 celebs" vs "9
   celebs". The "source of truth" doc would make someone launch the wrong
   recipe. `TRACK_B_BREV_HANDOVER.md` is stale on the same points.

8. **`lora_alpha` is silently ignored by lerobot-train.** lerobot's `PeftConfig`
   exposes only 5 fields; `lora_alpha`/`lora_dropout` fall back to PEFT defaults
   (alpha=r). The documented alpha=64 was never applied to the Pi0.5 LoRA runs.

9. **PaliGemma 1, not 2** — `pivot_research.md` corrected this but `TRACK_B.md`
   and `TRACK_B_WARMSTART.md` still cite the PaliGemma 2 paper. Not propagated.

10. **Two conflicting Track 2 VL datasets.** Sejohn's headshot-bank generator
    (`generate_vl_pairs.py`) is effectively dead; the wired-in dataset is
    Roham's teleop-frame manifest `HBOrtiz/eval3_objectvla_vl_pairs`
    (176,670 rows, 192 celebs). `VLPairsDataset` was schema-patched for it but
    not yet validated against real images.

11. **`TRACK_OBJECTVLA_ENHANCED.md` supersedes the baseline runbook.** Sejohn's
    enhanced spec stacks B-1..B-7 (warm-PG start, ArcFace data filter, hard-neg
    oversampling, layer-wise LoRA rank, ArcFace curriculum, EMA). The baseline
    `TRACK_OBJECTVLA.md` is now out of date.

12. **Augmentation data bug** — 4/151 source episodes have miscoded slot-R
    supervision (the aug pipeline failed to replace a Swift portrait with
    LeCun); for one episode 17/17 LeCun-at-R variants render Swift. ~260 of
    9,216 variants suspect. Upstream bug in `generate_aug_track3.py`, not yet
    swept across the full set.

13. **Face detection ran undersized.** `det_size=320` caught 3 faces in only
    50.4% of frames; re-run queued at `det_size=640`. Whether M2/KLAL labels
    used the undersized run needs confirming.

14. **Rollout gotchas** (commit `0ae95e4`, undocumented in TRACK_B docs):
    `PI05Policy.from_pretrained` on a PEFT adapter dir **silently loads random
    weights** if PEFT isn't installed (observed live — "~100-scale random
    actions"); `compile_model=True` must be OFF for rollout or the arm gets 0
    actions. `TRACK_B_ROLLOUT_HANDOVER.md` is referenced but does not exist.

15. **KLAL open verification items** — recomputes attention without RoPE
    (deliberate, but flagged to check vs arXiv:2511.12738 before the final
    run); the `target_slot_idx` path must be confirmed populated, else KLAL
    supervises the union of all 3 faces, not the prompted one.

16. **`TORCHCODEC_OOM_REPORT.md`** concluded the fix is
    `--dataset.video_backend=pyav`; the Track B Brev run used torchcodec anyway
    and hit the OOM, patched with `num_workers=2`. Two contradictory mitigations
    for the same root cause; never reconciled, never filed upstream.

---

## Bottom line

- **KLAL is built (Track E), co-trained, mid-flight.** Adopt it; don't rebuild.
  `TRACK_ARCFACE.md` is obsolete.
- The real blockers are **(a)** Track 2's co-train loop is unimplemented
  scaffold and **(b)** the Strix kill-switch probe has never run.
- The attention-routing diagnosis is confirmed by a second independent probe.
- M2-alone is a confirmed dead end (predicted, then observed).
- `TRACK_B.md` and the baseline `TRACK_OBJECTVLA.md` are stale and contradictory
  — superseded by the devbox-handover recipe and `TRACK_OBJECTVLA_ENHANCED.md`.

*Audit by 4 parallel read-only agents, 2026-05-20.*
