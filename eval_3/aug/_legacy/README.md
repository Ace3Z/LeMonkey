# eval_3/aug/_legacy

Historical artifacts preserved for reproducibility. None of these files is
part of the deployed Eval 3 recipe.

| File | What it was |
|---|---|
| `STRATEGY_v1.md` | First augmentation strategy doc (2026-05-10). Superseded by `eval_3/aug/STRATEGY.md` (was `STRATEGY_v3.md`). |
| `RESEARCH_v2.md` | Research note that motivated the v2 stage-2 rewrite (GroundingDINO + Lucas-Kanade tracking). Itself superseded by the v6 static-camera approach. |
| `pipeline_v1.py` | v1 orchestrator that chained `stage2_v1_sam_video.py` -> `stage3_extract_corners.py` -> `inpaint` -> `stage5_verify_identity.py`. |
| `stage2_v1_sam_video.py` | v1 stage 2: SAM 2.1 video predictor on interactively-clicked frame-0 prompts. Produced jittery masks. |
| `stage2_v2_grounding_lk.py` | v2 stage 2: GroundingDINO frame-0 detect + SAM + Lucas-Kanade corner tracking with Kalman smoothing. |
| `stage3_extract_corners.py` | v1 stage 3: convert per-frame `portrait_masks.pkl` (from v1 stage 2) into `portrait_corners.json`. Made redundant by v6 static-camera stage 2, which writes corners directly. |
| `stage5_verify_identity.py` | v1 stage 5: per-variant ArcFace identity QA. The v3 generators (`broad.py`, `cotrain.py`) decide identity at sampling time, so this post-hoc check is not invoked. |

The deployed (v6 / v3) recipe lives at `eval_3/aug/{stages,generators,merge_prep,training}/`.
