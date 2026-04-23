# Robot Learning FS26 — Project 1: Reasoning Pick & Place (VLA)

> Build a **Vision-Language-Action (VLA)** manipulation policy that picks up an object and places it at a target location specified by a natural-language prompt. Evaluated on three setups of increasing difficulty. Smallest-model bonus available.

---

## TL;DR

| Item | Value |
|------|-------|
| **Project** | Reasoning Pick & Place with a VLA policy |
| **Robot** | SO-101 arm (hardware provided by the course) |
| **Data format** | [LeRobot dataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) |
| **Compute** | NVIDIA Brev — $200 team credit, H100 (A100 fallback) |
| **Max score** | 150 pts main eval + 50 pts bonus |
| **Slack channel** | `project-1-vla` (course workspace) |
| **Eval location** | HG (collect at least some training data in this setup) |

---

## 1. Project Goal

Build a vision-language-conditioned manipulation policy that:

1. Observes the scene through the robot's camera.
2. Reads a natural-language prompt.
3. Picks up the target object and places it at the location the prompt specifies.

The policy must be a **VLA built on a pretrained vision-language backbone**. The central capability we are evaluated on is the policy's ability to *reason* about the prompt — not just map colors to positions.

---

## 2. Evaluation — 150 pts main + 50 pts bonus

All evals use the same physical setup: the SO-101 is mounted at the edge of a white table, objects placed in front of it. You choose exact positions on eval day, but all objects must remain movable by **±5 cm** from those positions. The gripper must start **≥15 cm** from the banana, and the camera must see all three bowls and the banana.

### Eval 1 — Direct color-conditioned pick-and-place (50 pts)

- Three bowls placed in a semicircle: **blue, red, green** (left → right).
- Squishy toy banana placed horizontally (smiley orientation) directly in front of the arm.
- Prompt: `"Put the banana in the [blue/red/green] colored bowl."`
- **Success:** banana ends up inside the correct bowl within **15 seconds**.
- **9 rollouts / team** (3 per color), ~5.55 pts per success.

### Eval 2 — Compositional instruction following (50 pts)

- Same bowl setup as Eval 1 but **varying colors**.
- Prompts require reasoning beyond direct color lookup, e.g.:
  - `"Put the banana into the 2nd bowl from the left."`
  - `"Put the banana into the bowl colored by mixing red and blue."` (→ purple)
  - `"Put the banana into the bowl that is not green and not blue."`
- Exact prompts not disclosed in advance but identical across groups; multiple prompts, mix of easy and hard.
- **15 s / rollout.** Points distributed evenly across rollouts.

### Eval 3 — Coke can on celebrity image (50 pts)

- DIN A5 color prints of celebrities placed in a semicircle; a 330 ml slim coke can stands in the middle.
- Prompt: `"Place the coke on [celebrity name]"` — policy must place the can on top of the correct image.
- **In-distribution celebrities:** Taylor Swift, Barack Obama, Yann LeCun.
- **OOD celebrities** used in some rollouts (e.g. Roger Federer, Angela Merkel).
- Exact images and setups undisclosed but identical across groups.
- **10 rollouts / team**, 5 pts each. 15 s / rollout.

### Bonus (up to 50 pts) — Smallest model

Ranked by **total active parameter count** of the models used during inference for each eval.

| Eval | 1st | 2nd | 3rd | … |
|------|-----|-----|-----|---|
| Eval 1 | 10 | 9 | 8 | … |
| Eval 2 | 20 | 18 | 16 | … |
| Eval 3 | 20 | 18 | 16 | … |

Different models may be used across the three eval setups, or the same model with different weights/checkpoints.

---

## 3. Architecture & Procedure Constraints

- **Must be a VLA.** Architecture for each task (or combined) must use a **pretrained vision-language backbone**. It does *not* need a flow-matching action head or other extensions.
- **Must be learned.** You must demonstrate you trained / fine-tuned the model yourself on the provided compute.
- **Training data**: publicly available datasets, teleoperation, or synthetic generation — all allowed.
- **Optional extensions allowed**: RL post-training, synthetic data generation, etc.
- **Hardware**: no custom hardware beyond what was provided in the course box. Return it fully functional.

**Recommended starting points for VLAs:** *FlowerVLA*, *SmolVLA*.

---

## 4. Hardware & Objects

### Robot

- **SO-101 arm** (course-provided).
- Build & calibration guide: <https://huggingface.co/docs/lerobot/so101>

### Manipulation objects (provided by the course)

| Object | Quantity | Link |
|--------|---------:|------|
| Bowls (one per color) | 6 | [Amazon.de](https://www.amazon.de/-/en/gp/product/B0DWSCZNM1?smid=A31GQFT58NDMJG&psc=1) |
| Squishy banana | 1 | [Amazon.de](https://www.amazon.de/-/en/gp/product/B0F53JN82R?smid=A2F4BB4XD540GR&th=1) |
| Coke can (330 ml slim) | 1 | [Migros](https://www.migros.ch/en/product/120271700000) |

Objects can be picked up early the week after the announcement or in the **TA session on April 30th**. Monitor Slack for current info.

---

## 5. Data

- **Format:** [LeRobot dataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) for **all** demonstrations recorded throughout the course.
- **Source:** teleoperation on the SO-101, plus any publicly available or synthetic data you want to mix in.
- **Diversity:** collect in **different lighting conditions, different tables**, etc. The final evals are in **HG** — include HG recordings. The HG tables are **not perfectly white**.
- **Validation:** always replay a recorded episode on the real robot before committing to training, to catch broken demos early.

### LeRobot reference repo

- [huggingface/lerobot](https://github.com/huggingface/lerobot) — models, training scripts, example code.

---

## 6. Compute — NVIDIA Brev

NVIDIA provides compute via **Brev**. Baseline: **$200 credit per team** (not per student).

### 6.1 Setup flow

1. **Nominate a Team Captain.** Captain replies (from ETH email) to **all** people CC'd in the setup email. The coupon is only issued once the captain is known.
2. **Captain: redeem the coupon.**
   1. Sign up at <https://brev.nvidia.com> using your ETH email.
   2. Billing tab → Redeem Code → paste code → Redeem.
   3. Team tab → Generate Invite Link → share with teammates.
3. **Teammates** click the invite link and sign in with their ETH email.
4. **Add the TAs to your Brev org** (required): via Team tab, same invite link:
   - Zador Pataki — <patakiz@ethz.ch>
   - Nicole Damblon — <ndamblon@ethz.ch>

### 6.2 Captain responsibilities

- Create the Brev organization.
- Redeem the coupon code.
- Invite teammates.
- Share GPU instances (SSH / Jupyter) with the team.
- Monitor team usage.

### 6.3 Sharing a GPU instance

- **SSH:** instance in GPU tab → Share SSH Access → add teammate usernames (instance must be running).
- **Jupyter:** under Using secure links → Edit access → add teammate email/username → Share (container must be fully built first).

### 6.4 Hardware expectations

- **Target GPU:** H100. **Fallback:** A100 when H100 capacity is tight.
- Size the instance to the job — smaller instances last proportionally longer against the $200 budget.
- **While prototyping, stick to a single GPU.** Most bugs reproduce on 1 GPU; 8×H100 debugging burns credits fast.
- Brev pulls from multiple cloud providers; availability is dynamic.
- **Pick one provider and stick with it.** Swapping mid-project introduces subtle env differences. Write down which provider you used once a setup works.

### 6.5 Using credits responsibly

- **Stop or terminate instances the moment you're done.** Idle GPUs burn credits fast.
- **Pause/resume is provider-dependent.** On providers that don't support stop/start, terminating means reuploading data — data is tied to the instance.
- **Going away for a weekend?** `scp` your data out and terminate.
- **Negative-balance instances are deleted after 96 h.** Monitor the Billing tab; if an instance is deleted, its data goes with it.

### 6.6 Getting help

- **Brev platform issues** (SSH, billing, providers, `scp`, …): <brev-support@nvidia.com>
- **Brev docs & in-product AI search:** <https://docs.nvidia.com/brev/latest> (also under the Documentation tab in the Brev console). Read these first.
- **Course-specific / coupon questions:** Zador (<patakiz@ethz.ch>), Nicole (<ndamblon@ethz.ch>).

---

## 7. Sanity-Check Tasks (required — do these first)

Before tackling the main project, every team must complete the following as a setup sanity check:

1. **Collect 20 demonstrations** of the project's task (or a simplified version) with the SO-101. Repeat the same simple motion — eliminate variation between demos. Use **LeRobot dataset v3**.
2. **Replay** some of the demos with objects in the positions they were recorded in, to verify correct recording. **No training yet.**
3. **Upload demonstrations to Brev**, pick any simple policy (e.g. from [huggingface/lerobot](https://github.com/huggingface/lerobot)), and train it via **behavior cloning to overfit** to those demos.
4. **Deploy** the trained policy. Match the background and object positions seen in demos. Verify the robot replicates the recorded motion. If nobody on the team has a local GPU for deployment, ask other groups or the TAs on Slack.

Report any issues in the group's Slack channel.

---

## 8. Recommended Workflow (getting started)

1. Calibrate the robot.
2. Get teleoperation working.
3. Record and replay a teleop episode; verify it's correct (LeRobot format).
4. Research small VLA architectures.
5. Implement a basic VLA training loop and VLA model loading.
6. Set up compute access; train any small model on any data.
7. Train on a **single** teleop episode and verify it learns — **overfit test**.
8. Collect teleop data for **Eval 1** and train.

---

## 9. Tips & Tricks

- **Budget compute wisely.** $200 disappears fast on idle H100s.
- **Validate teleop data before training.** Bad demos = bad policies.
- **Debug with a single episode first.** Only scale up once the pipeline runs end-to-end.
- **Use the LeRobot / HuggingFace ecosystem.** Saves a lot of time.
- **Reuse OSS code.** Simple VLA training repos + pretrained VLMs / VLAs.
- **Collect data in varied conditions:** lighting, tables, backgrounds — for a robust policy.
- **Collect at least some data in HG.** That's where the final evals happen and the tables aren't perfectly white.

---

## 10. Logistics & Communication

- **Slack workspace:** course workspace, channel `project-1-vla`. All future clarifications go there.
- **Weekly update:** each group sends a short **Slack update before every Thursday session**, describing progress and current issues, so TAs can focus on the hard debugging problems in-session.
- **TAs:** Zador Pataki (<patakiz@ethz.ch>), Nicole Damblon (<ndamblon@ethz.ch>).

---

## 11. Team

| Name | Role | Brev / ETH email | Notes |
|------|------|------------------|-------|
| TBD  | Team Captain | TBD | Will redeem Brev coupon, own shared GPU instance |
| TBD  |       |     |       |
| TBD  |       |     |       |
| TBD  |       |     |       |

> **Action:** nominate the team captain and fill in the table. Captain replies to the TA email from their ETH address.

---

## 12. References & Links

### Course / platform
- [Brev (NVIDIA)](https://brev.nvidia.com) — GPU compute portal
- [Brev documentation](https://docs.nvidia.com/brev/latest)
- [Brev support](mailto:brev-support@nvidia.com)
- Full course doc (Google Docs): <https://docs.google.com/document/d/1YsQ_Qe4vEwDp1dJdqn3l9vSt7oJBkc6JazjbmWLxAXg/edit?tab=t.0>

### Robot & data
- [SO-101 build & calibration](https://huggingface.co/docs/lerobot/so101)
- [LeRobot dataset v3 format](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [huggingface/lerobot (models, code)](https://github.com/huggingface/lerobot)

### Research / related work
- [VLABench — benchmark paper](https://vlabench.github.io/) *(recommended starting read)*
- **FlowerVLA, SmolVLA** — recommended small-VLA starting points

### Objects (for reference / reorder)
- [Bowls (Amazon.de)](https://www.amazon.de/-/en/gp/product/B0DWSCZNM1?smid=A31GQFT58NDMJG&psc=1)
- [Squishy banana (Amazon.de)](https://www.amazon.de/-/en/gp/product/B0F53JN82R?smid=A2F4BB4XD540GR&th=1)
- [Coke can 330 ml slim (Migros)](https://www.migros.ch/en/product/120271700000)

### Local files in this repo
- [`Robot_Learning_FS26_Brev_Instruction.pdf`](Robot_Learning_FS26_Brev_Instruction.pdf) — Brev setup instructions (original)
- [`vla_slide_image.png`](vla_slide_image.png) — course project-overview slide

---

## 13. Open Items / TBD

- [ ] Nominate team captain and notify the TAs
- [ ] Redeem Brev coupon (captain) and share with team
- [ ] Add Zador (<patakiz@ethz.ch>) and Nicole (<ndamblon@ethz.ch>) to the Brev org
- [ ] Pick up hardware (TA session **April 30**, or earlier if available)
- [ ] Decide on VLA backbone (FlowerVLA / SmolVLA / other)
- [ ] Divide responsibilities: teleop / data / training / eval / deployment
- [ ] Complete sanity-check tasks (20 demos → replay → overfit → deploy)
- [ ] Agree on a weekly Slack update owner (rotating?)
