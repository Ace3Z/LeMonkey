# 2026-05-21 — Track-3 robot dataset: 20-episode visual verification

## What was run

Rendered 20 randomly sampled Track-3 augmented episodes (`seed=20`) from
`datasets/eval3_track3_aug/` as annotated videos: full 538-frame trajectory,
all 3 portrait quads drawn per-frame from the base episode's
`portrait_corners.json`, each quad labelled with its celeb, the prompt and the
target celeb shown in a banner, target quad highlighted green / distractors red.

Output: `eval_3/attention_steering/dataset_verify/` — `ep_01..ep_20_*.mp4`,
`strip_ep_NN_.jpg` (5-keyframe strips), `_montage.jpg`.

## What was checked, per episode

For each episode, that all four agree:
1. `augmentation.json` prompt celeb
2. the green TARGET quad's label
3. the actual face inpainted inside that quad
4. where the coke can physically lands at the end of the trajectory

## Result — 20/20 correct

Every episode: prompt celeb == target quad label == face inside the quad ==
can-landing portrait. All 60 portrait labels (3 × 20) match the inpainted
face. The quads track the rotated portraits cleanly across the trajectory.

| ep | prompt target | target pid | can lands on target |
|----|---------------|-----------|---------------------|
| 01 | Barack Obama  | pid0 | yes |
| 02 | Taylor Swift  | pid0 | yes |
| 03 | Barack Obama  | pid1 | yes |
| 04 | Taylor Swift  | pid1 | yes |
| 05 | Taylor Swift  | pid2 | yes |
| 06 | Barack Obama  | pid2 | yes |
| 07 | Barack Obama  | pid0 | yes |
| 08 | Taylor Swift  | pid0 | yes |
| 09 | Barack Obama  | pid1 | yes |
| 10 | Yann LeCun    | pid1 | yes |
| 11 | Barack Obama  | pid1 | yes |
| 12 | Yann LeCun    | pid0 | yes |
| 13 | Yann LeCun    | pid1 | yes |
| 14 | Barack Obama  | pid0 | yes |
| 15 | Barack Obama  | pid0 | yes |
| 16 | Yann LeCun    | pid2 | yes |
| 17 | Barack Obama  | pid2 | yes |
| 18 | Taylor Swift  | pid0 | yes |
| 19 | Yann LeCun    | pid1 | yes |
| 20 | Barack Obama  | pid1 | yes |

Target pids span all three slots (0/1/2) and all three celebs — the sample is
not degenerate.

## Conclusion

The Track-3 robot dataset (`HBOrtiz/so101_eval3_track3_v3_baseline` /
`datasets/eval3_track3_aug`) is correctly labelled on this 20-episode sample:
the inpainted identity, the prompt, the bbox, and the executed trajectory are
mutually consistent. Combined with the `eval3_track3_vl_pairs` v3 validation
(see `2026-05-21_vl_pairs_image_mispairing.md`), both co-training datasets are
verified train-ready.
