# LeMonkey

**Vision-Language-Action manipulation policy** for the ETH Robot Learning FS26 course (Project 1).

Pick-and-place on a 6-DOF SO-101 arm, driven by a natural-language prompt and a pretrained vision-language backbone. Evaluated on three setups of increasing reasoning difficulty — direct color lookup, compositional instruction following, and celebrity-image matching — with a bonus for the smallest-parameter model.

## Status

🚧 Early planning. Captain nomination, Brev setup, and hardware pickup in progress.

## Repository layout

```
.
├── README.md                                        # this file
├── .gitignore
└── docs/
    ├── PROJECT.md                                   # full brief: eval spec, constraints, workflow, resources
    ├── RELATED_WORK.md                              # prior work ranked per eval (repos, datasets, checkpoints)
    ├── VLA_ARCHITECTURES.md                         # VLA × VLM lit review, per-eval recommendation, tunable knobs
    ├── Robot_Learning_FS26_Brev_Instruction.pdf     # Brev GPU setup instructions (original PDF)
    └── vla_slide_image.png                          # course project-overview slide
```

## Start here

**Read [`docs/PROJECT.md`](docs/PROJECT.md) first.** It consolidates everything: the three eval tasks, architecture constraints, hardware list, LeRobot data format, Brev GPU setup, sanity-check tasks, recommended workflow, and all external references.

For concrete starting points (repos, datasets, checkpoints) ranked per eval, see [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md). For the VLA / VLM choice itself — backbones, parameter counts, fine-tuning knobs, and the per-eval recommendation — see [`docs/VLA_ARCHITECTURES.md`](docs/VLA_ARCHITECTURES.md).

## Quick links

| | |
|---|---|
| Slack — course channel | [`project-1-vla`](https://robot-course-ethz.slack.com/archives/C0AULTPSDHS) ([join workspace](https://join.slack.com/t/robotlearning-wht4341/shared_invite/zt-3vjghtb1w-K3k7b7amUr37y39IF9dL3g)) |
| GPU compute | [NVIDIA Brev](https://brev.nvidia.com) · [docs](https://docs.nvidia.com/brev/latest) |
| Robot | [SO-101 build & calibration](https://huggingface.co/docs/lerobot/so101) |
| Data format | [LeRobot dataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) |
| Reference code | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| Related work | [VLABench](https://vlabench.github.io/) |
| Course doc | [Google Docs](https://docs.google.com/document/d/1YsQ_Qe4vEwDp1dJdqn3l9vSt7oJBkc6JazjbmWLxAXg/edit?tab=t.0) |

## Team

See [`docs/PROJECT.md` §11](docs/PROJECT.md#11-team).

## License

Course coursework — internal use by the team only. Not for redistribution.
