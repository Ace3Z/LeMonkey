# 2026-05-21 — KLAL + LoRA co-training import + review

## What was done

Imported mahbod's KLAL (KL-divergence attention loss) + LoRA co-training work
from `dev/mahbod/kl-divergence` onto `main` (commit `526073f`). Mahbod's
`cotrain.py` becomes the **canonical** trainer — KLAL and LoRA are gated behind
`--enable_klal` / `--enable_lora`, so with neither flag it runs as plain
robot+VL co-training (Darius's behaviour); with the flags it adds LoRA
adaptation and the KLAL attention-supervision loss on VL steps.

Imported (byte-identical, selective copy — no merge):
- `eval_3/aug/m2_klal.py`, `m2_klal_vl.py`, `m2_klal_smolvla.py`, `m2_lora.py`
- `eval_3/aug/tests/test_klal_lora_smoke.py`
- `eval_3/scripts/smolvla_cotrain/{cotrain.py,launch.sh,setup_env.sh,README.md,run_cluster.sh,RUN_ON_CLUSTER.md,.gitignore}`
- `docs/experiments/2026-05-20_klal_lora_smolvla_cotrain.md`

Not imported: `m2_klal_data.tar.zst` (9.4 MB, stale — the M2-bundle data path
was removed when KLAL moved to the VL forward).

## Review (agents D + E + F)

**KLAL implementation — sound.** Forward-KL direction correct, causal masking
correct, RoPE/GQA faithful to SmolVLM internals, bbox→target-mask construction
valid, no silent fallbacks. **Smoke test 33/33 PASS** on the current env
(transformers 5.3.0 / lerobot 0.5.1) — including a real end-to-end check where
KLAL's gradient reaches a LoRA adapter. No 4.55→5.3.0 API drift in the m2
modules.

**cotrain.py — structurally avoids Darius's blockers.** No `meta.info.data_path`
(C3 absent — but only because the episode-filter function was dropped entirely).
VL collator uses `do_image_splitting=False` → 64 image tokens, no truncation
(C4 absent). KLAL/LoRA gating is clean.

## KNOWN GAP — 5 Darius fixes missing (reconcile before a real run)

Mahbod's branch forked from Darius's early (`92a11d0`) and re-implemented later
fixes independently. Missing, in priority order:

1. **eval-day defaults** — `--vl_ratio` default is 10 (eval-day used 5);
   docstring/help still references `eval3_objectvla_vl_pairs` (should be
   `HBOrtiz/eval3_track3_vl_pairs`). Trivial.
2. **`empty_cameras` + `reference→camera2` rename** — mahbod sets
   `empty_cameras=0` and omits the rename map; Darius hard-set `1` + patched
   `RenameObservationsProcessorStep`. Silent-failure risk — verify against the
   Track-3 robot dataset's actual camera schema.
3. **`HF_HUB_OFFLINE` flip in save/push** — without it, checkpoint upload
   silently fails when `launch.sh` exports `HF_HUB_OFFLINE=1`.
4. **robot-dataset partial-cache handling** — no `_episodes_with_complete_files`
   and no `--robot_max_episodes`; a partial cache crashes mid-run. (Port
   Darius's filter but with the C3 fix: `meta.data_path`, not `meta.info.data_path`.)
5. **lerobot `snapshot_download`/`get_safe_version` monkey-patch** — mahbod's
   `main_process_first()` barrier helps multi-GPU but not the single-GPU 429
   stall the patch addressed.

Also: `_patch_smolvlm_vision_embeddings()` runs unconditionally and reverts an
upstream transformers-5.3.0 rewrite — remove or version-gate it. `--enable_klal`
help text says "robot steps"; KLAL actually applies on VL steps.

## Status

KLAL co-training is on `main` as a flag-gated option, verified to run (smoke
33/33). Before any real training run the 5 gaps above must be reconciled and
the trainer smoke-tested on the AWS node in all 3 modes (plain / `--enable_lora`
/ `--enable_klal`).
