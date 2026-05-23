#!/usr/bin/env python3
"""Record Eval 2 teleop episodes against a balanced 180-episode plan.

Eval 2 task (compositional instruction following):
  - Banana stays in its Eval-1 position throughout.
  - 3 colored bowls reshuffled across 3 positions per session.
  - Compositional prompts (spatial / relational / negation).
  - 20 s per rollout.

Design:
  • Plan = 180 episodes = 6 arrangements × 6 families × 5 reps.
  • Plan is generated once, saved to disk, and survives script restarts.
  • Episodes are batched by arrangement (~30 per arrangement) so you reshuffle
    the bowls only ~6 times across the whole 180-episode collection.
  • Within each batch the 30 episodes are shuffled across the 6 families so
    the order of prompts is unbiased.
  • Bowl arrangement and prompt-family are *independently* balanced — each
    arrangement appears in exactly 30 episodes; each family in exactly 30.
  • Each episode picks a random valid target_idx and a random phrasing
    variant for the family, so the recorded prompts are highly diverse
    (~150+ distinct strings across the 180 episodes).

Usage:
    record_episodes.py                          # use existing plan or create one
    record_episodes.py --regenerate-plan        # discard plan, build a fresh one
    record_episodes.py --regenerate-plan --seed 42
    record_episodes.py --dry-run                # walk the plan without recording

Controls during recording:
    ENTER   record the next planned episode
    d       delete the LAST recorded episode AND re-queue its prompt
    p       show progress
    q       quit (state is auto-saved every episode)

Outputs:
    ~/LeMonkey/datasets/eval2/ep_NNNN_<arr>_<family>_<ts>/
    eval_2/state/plan.json                   # persistent plan + progress
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# ─── Arrangement spec ────────────────────────────────────────────────────────
COLOR_NAMES = {"B": "blue", "R": "red", "G": "green"}
ARRANGEMENTS = ["BRG", "BGR", "RBG", "RGB", "GBR", "GRB"]


def is_valid_arrangement(s: str) -> bool:
    return len(s) == 3 and sorted(s) == ["B", "G", "R"]


# ─── Phrasing pools (deliberately broad to combat phrasing-overfit) ──────────
DIRECT_COLOR_PHR = [
    "Put the banana in the {c} bowl",
    "Put the banana in the {c} colored bowl.",
    "Place the banana in the {c} bowl",
    "pick the banana and put it in the {c} bowl",
    "Place the banana in the {c} colored bowl",
    "Drop the banana in the {c} bowl",
    "Move the banana to the {c} bowl",
    "Put the banana into the {c} bowl",
]

ABS_PHR = {
    "left": [
        "Put the banana in the leftmost bowl.",
        "Put the banana in the left bowl.",
        "Put the banana in the bowl on the far left.",
        "Put the banana in the bowl all the way to the left.",
        "Put the banana in the bowl on the very left.",
        "Place the banana in the leftmost bowl.",
        "Drop the banana in the leftmost bowl.",
        "Put the banana in the bowl furthest to the left.",
    ],
    "middle": [
        "Put the banana in the middle bowl.",
        "Put the banana in the center bowl.",
        "Put the banana in the bowl in the middle.",
        "Put the banana in the bowl in the center.",
        "Place the banana in the middle bowl.",
        "Drop the banana in the center bowl.",
        "Put the banana in the centre bowl.",
        "Put the banana in the bowl that is in the middle.",
    ],
    "right": [
        "Put the banana in the rightmost bowl.",
        "Put the banana in the right bowl.",
        "Put the banana in the bowl on the far right.",
        "Put the banana in the bowl all the way to the right.",
        "Put the banana in the bowl on the very right.",
        "Place the banana in the rightmost bowl.",
        "Drop the banana in the rightmost bowl.",
        "Put the banana in the bowl furthest to the right.",
    ],
}

ORDINAL_WORDS = {1: ["1st", "first"], 2: ["2nd", "second"], 3: ["3rd", "third"]}
ORD_PHR = [
    "Put the banana into the {ord} bowl from the {ref}.",
    "Put the banana in the {ord} bowl from the {ref}.",
    "Place the banana in the {ord} bowl from the {ref}.",
    "Drop the banana in the {ord} bowl from the {ref}.",
    "Put the banana in the bowl that is the {ord} from the {ref}.",
]

REL_LR_PHR = [
    "Put the banana in the bowl on the {side} of the {ref} bowl.",
    "Put the banana in the bowl to the {side} of the {ref} bowl.",
    "Put the banana in the bowl just on the {side} of the {ref} bowl.",
    "Put the banana in the bowl immediately {side} of the {ref} bowl.",
    "Place the banana in the bowl on the {side} of the {ref} bowl.",
    "Put the banana in the bowl directly to the {side} of the {ref} bowl.",
]

REL_BETWEEN_PHR = [
    "Put the banana in the bowl between the {a} and {b} bowls.",
    "Put the banana in the bowl that is between the {a} and {b} bowls.",
    "Put the banana in the bowl that sits between the {a} and {b} bowls.",
    "Place the banana in the bowl between the {a} and {b} bowls.",
    "Put the banana in the middle bowl between the {a} and {b} bowls.",
]

NEG_PHR = [
    "Put the banana in the bowl that is not {a} and not {b}.",
    "Put the banana in the bowl that is neither {a} nor {b}.",
    "Put the banana in the bowl that is not {a} and also not {b}.",
    "Place the banana in the bowl that is not {a} and not {b}.",
    "Drop the banana in the bowl that is not {a} and not {b}.",
    "Put the banana in the bowl that isn't {a} or {b}.",
    "Put the banana in the bowl that is not the {a} bowl and not the {b} bowl.",
]


# ─── Per-family generators ───────────────────────────────────────────────────
# Each: returns (target_idx, prompt) or (None, None) if the combo is invalid.

def gen_direct(arrangement: str, target_idx: int) -> tuple[int, str]:
    color = COLOR_NAMES[arrangement[target_idx]]
    return target_idx, random.choice(DIRECT_COLOR_PHR).format(c=color)


def gen_absolute(arrangement: str, target_idx: int) -> tuple[int, str]:
    pos = ["left", "middle", "right"][target_idx]
    return target_idx, random.choice(ABS_PHR[pos])


def gen_ordinal(arrangement: str, target_idx: int) -> tuple[int, str]:
    side = random.choice(["left", "right"])
    n = target_idx + 1 if side == "left" else 3 - target_idx
    word = random.choice(ORDINAL_WORDS[n])
    return target_idx, random.choice(ORD_PHR).format(ord=word, ref=side)


def gen_relational_lr(arrangement: str, target_idx: int) -> tuple[int | None, str | None]:
    options = []
    for side, dx in [("right", -1), ("left", +1)]:
        ref_idx = target_idx + dx
        if 0 <= ref_idx <= 2:
            ref_color = COLOR_NAMES[arrangement[ref_idx]]
            options.append((side, ref_color))
    if not options:
        return None, None
    side, ref = random.choice(options)
    return target_idx, random.choice(REL_LR_PHR).format(side=side, ref=ref)


def gen_relational_between(arrangement: str, target_idx: int) -> tuple[int | None, str | None]:
    if target_idx != 1:
        return None, None
    a = COLOR_NAMES[arrangement[0]]
    b = COLOR_NAMES[arrangement[2]]
    if random.random() < 0.5:
        a, b = b, a
    return target_idx, random.choice(REL_BETWEEN_PHR).format(a=a, b=b)


def gen_negation(arrangement: str, target_idx: int) -> tuple[int, str]:
    target_color = COLOR_NAMES[arrangement[target_idx]]
    others = [c for c in ["blue", "red", "green"] if c != target_color]
    if random.random() < 0.5:
        others = others[::-1]
    a, b = others
    return target_idx, random.choice(NEG_PHR).format(a=a, b=b)


GENERATORS: dict[str, callable] = {
    "direct":               gen_direct,
    "spatial_absolute":     gen_absolute,
    "spatial_ordinal":      gen_ordinal,
    "relational_lr":        gen_relational_lr,
    "relational_between":   gen_relational_between,
    "negation":             gen_negation,
}
FAMILIES = list(GENERATORS.keys())


def _valid_target_indices(family: str, arrangement: str) -> list[int]:
    """Which target_idx values produce a valid prompt for this (family, arr)."""
    valid = []
    for ti in range(3):
        # Ask the generator without committing — we re-call later to actually
        # pick the phrasing once we've decided on target_idx.
        idx, _ = GENERATORS[family](arrangement, ti)
        if idx is not None:
            valid.append(ti)
    return valid


# ─── Plan generation + persistence ───────────────────────────────────────────

@dataclass
class Episode:
    idx: int                 # original plan position
    arrangement: str
    family: str
    target_idx: int
    target_color: str
    prompt: str
    status: str = "pending"  # pending | recorded | deleted
    recorded_path: str | None = None
    timestamp: str | None = None


@dataclass
class Plan:
    episodes: list[Episode] = field(default_factory=list)
    seed: int = 42

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "episodes": [asdict(e) for e in self.episodes],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Plan":
        return cls(
            seed=data.get("seed", 42),
            episodes=[Episode(**e) for e in data["episodes"]],
        )


def generate_plan(seed: int) -> Plan:
    """Build a balanced 6×6×5 = 180-episode plan, batched by arrangement."""
    rng = random.Random(seed)
    plan_episodes: list[Episode] = []

    # Decide outer order of arrangements (random permutation of the 6).
    arr_order = list(ARRANGEMENTS)
    rng.shuffle(arr_order)

    for arr in arr_order:
        batch: list[Episode] = []
        for fam in FAMILIES:
            for _ in range(5):
                # Pick a target_idx within the family's valid set
                valid = _valid_target_indices(fam, arr)
                if not valid:
                    raise RuntimeError(f"family {fam} has no valid target on {arr}")
                # Use module-level random for in-cell randomness, seeded from the rng
                random.seed(rng.random())
                target_idx = random.choice(valid)
                idx_resolved, prompt = GENERATORS[fam](arr, target_idx)
                assert idx_resolved is not None
                batch.append(Episode(
                    idx=-1,  # filled in after batch shuffle
                    arrangement=arr,
                    family=fam,
                    target_idx=target_idx,
                    target_color=COLOR_NAMES[arr[target_idx]],
                    prompt=prompt,
                ))
        # Shuffle within batch so families interleave; batches stay arrangement-fixed
        rng.shuffle(batch)
        plan_episodes.extend(batch)

    for i, ep in enumerate(plan_episodes, 1):
        ep.idx = i

    return Plan(episodes=plan_episodes, seed=seed)


def load_or_create_plan(path: Path, seed: int, regenerate: bool) -> Plan:
    if path.exists() and not regenerate:
        try:
            return Plan.from_json(json.loads(path.read_text()))
        except Exception as e:
            print(f"[WARN] failed to load existing plan ({e}); regenerating", flush=True)
    plan = generate_plan(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_json(), indent=2))
    return plan


def save_plan(plan: Plan, path: Path) -> None:
    path.write_text(json.dumps(plan.to_json(), indent=2))


def next_pending(plan: Plan) -> Episode | None:
    for ep in plan.episodes:
        if ep.status == "pending":
            return ep
    return None


def progress_summary(plan: Plan) -> str:
    n_total = len(plan.episodes)
    by_status: dict[str, int] = {}
    by_arr: dict[str, dict[str, int]] = {a: {"pending": 0, "recorded": 0, "deleted": 0} for a in ARRANGEMENTS}
    by_fam: dict[str, dict[str, int]] = {f: {"pending": 0, "recorded": 0, "deleted": 0} for f in FAMILIES}
    by_color: dict[str, dict[str, int]] = {c: {"pending": 0, "recorded": 0, "deleted": 0} for c in ["blue", "red", "green"]}
    for ep in plan.episodes:
        by_status[ep.status] = by_status.get(ep.status, 0) + 1
        by_arr[ep.arrangement][ep.status] += 1
        by_fam[ep.family][ep.status] += 1
        by_color[ep.target_color][ep.status] += 1
    n_done = by_status.get("recorded", 0)
    out = [f"  total: {n_total}    recorded: {n_done}    pending: {by_status.get('pending', 0)}    deleted: {by_status.get('deleted', 0)}"]
    out.append("")
    out.append(f"  {'arrangement':12s}  recorded / total")
    for a in ARRANGEMENTS:
        c = by_arr[a]
        out.append(f"  {a:12s}  {c['recorded']:>2} / {c['recorded']+c['pending']+c['deleted']}")
    out.append("")
    out.append(f"  {'family':20s}  recorded / total")
    for f in FAMILIES:
        c = by_fam[f]
        out.append(f"  {f:20s}  {c['recorded']:>2} / {c['recorded']+c['pending']+c['deleted']}")
    out.append("")
    out.append(f"  {'target color':12s}  recorded / total")
    for c in ["blue", "red", "green"]:
        d = by_color[c]
        out.append(f"  {c:12s}  {d['recorded']:>2} / {d['recorded']+d['pending']+d['deleted']}")
    return "\n".join(out)


# ─── Delete the last recorded episode (and re-queue its prompt) ──────────────

def delete_last_recorded(plan: Plan) -> Episode | None:
    """Find the most-recently recorded episode, delete its dataset dir,
    and flip its plan slot back to pending so it gets re-recorded next."""
    last = None
    for ep in plan.episodes:
        if ep.status == "recorded":
            key = (ep.timestamp or "", ep.idx)
            if last is None or key > (last.timestamp or "", last.idx):
                last = ep
    if last is None:
        return None
    if last.recorded_path:
        p = Path(last.recorded_path)
        if p.exists():
            shutil.rmtree(p)
            print(f"  🗑  removed {p}")
        else:
            print(f"  [WARN] recorded path missing: {p}")
    # Re-queue: status pending, clear recording metadata
    last.status = "pending"
    last.recorded_path = None
    last.timestamp = None
    # Move it to the front of the pending queue so the next ENTER re-records it
    plan.episodes.remove(last)
    # Insert at the position of the first remaining pending episode
    insert_at = next((i for i, e in enumerate(plan.episodes) if e.status == "pending"), len(plan.episodes))
    plan.episodes.insert(insert_at, last)
    return last


# ─── Main loop ───────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan-path", default="/home/lemonkey/LeMonkey/eval_2/state/plan.json")
    p.add_argument("--regenerate-plan", action="store_true",
                   help="Discard existing plan and start over (this resets all progress)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode-time-s", type=float, default=20.0)
    p.add_argument("--reset-time-s",   type=float, default=10.0)
    p.add_argument("--root", default="/home/lemonkey/LeMonkey/datasets/eval2")
    p.add_argument("--leader-port",   default="/dev/so101-leader")
    p.add_argument("--leader-id",     default="my_leader")
    p.add_argument("--follower-port", default="/dev/so101-follower")
    p.add_argument("--follower-id",   default="my_follower")
    p.add_argument("--cam-path",      default="/dev/video0")
    p.add_argument("--dry-run", action="store_true",
                   help="Walk through the plan without recording (no robot needed)")
    args = p.parse_args()

    Path(args.root).mkdir(parents=True, exist_ok=True)
    plan_path = Path(args.plan_path)

    if args.regenerate_plan and plan_path.exists():
        try:
            ans = input(f"⚠  --regenerate-plan will discard the existing plan at {plan_path}\n"
                        f"   recorded data on disk is NOT deleted, but progress will be lost.\n"
                        f"   continue? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return 0
        if ans != "y":
            print("aborted.")
            return 0

    plan = load_or_create_plan(plan_path, seed=args.seed, regenerate=args.regenerate_plan)
    save_plan(plan, plan_path)

    print("=" * 72)
    print(f"  Eval 2 teleop recording (180-episode balanced plan)")
    print(f"  plan file   : {plan_path}")
    print(f"  output root : {args.root}")
    print(f"  dry run     : {args.dry_run}")
    print("=" * 72)
    print()
    print(progress_summary(plan))
    print()

    last_arr = None
    while True:
        ep = next_pending(plan)
        if ep is None:
            print("\n🎉  plan complete — no pending episodes remaining.")
            print(progress_summary(plan))
            return 0

        # Announce arrangement change loudly so user reshuffles bowls
        if ep.arrangement != last_arr:
            print()
            print("╔" + "═" * 70 + "╗")
            print(f"║  ARRANGEMENT CHANGE → please set bowls to: {ep.arrangement:<25}".ljust(71) + " ║")
            print(f"║  left={COLOR_NAMES[ep.arrangement[0]]:<6}  middle={COLOR_NAMES[ep.arrangement[1]]:<6}  right={COLOR_NAMES[ep.arrangement[2]]:<6}".ljust(71) + " ║")
            print("╚" + "═" * 70 + "╝")
            last_arr = ep.arrangement

        n_total = len(plan.episodes)
        n_done = sum(1 for e in plan.episodes if e.status == "recorded")
        print()
        print("─" * 72)
        print(f" Episode {ep.idx:>3} of {n_total}    ({n_done} recorded so far)")
        print(f"   arrangement : {ep.arrangement}    family : {ep.family}")
        print(f"   prompt      : \"{ep.prompt}\"")
        print(f"   TARGET BOWL : {ep.target_color.upper()} (idx {ep.target_idx} = {['LEFT','MIDDLE','RIGHT'][ep.target_idx]})")
        print("─" * 72)

        try:
            ans = input("ENTER=record / 'd'=delete last / 'p'=progress / 'q'=quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nbye (state saved).")
            return 0

        if ans == "q":
            return 0
        if ans == "p":
            print(progress_summary(plan))
            continue
        if ans == "d":
            removed = delete_last_recorded(plan)
            if removed is None:
                print("  (nothing recorded yet to delete)")
            else:
                print(f"  re-queued episode {removed.idx} ({removed.family} on {removed.arrangement}, prompt: \"{removed.prompt}\")")
                save_plan(plan, plan_path)
            continue

        # Record
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ep_{ep.idx:04d}_{ep.arrangement}_{ep.family}_{ts}"
        run_path = Path(args.root) / run_name

        if args.dry_run:
            print(f"  [dry-run] would record → {run_path}")
            ep.status = "recorded"
            ep.recorded_path = str(run_path)
            ep.timestamp = ts
            save_plan(plan, plan_path)
            continue

        cmd = [
            "lerobot-record",
            "--robot.type=so101_follower",
            f"--robot.port={args.follower_port}",
            f"--robot.id={args.follower_id}",
            f"--robot.cameras={{ camera1: {{type: opencv, index_or_path: {args.cam_path}, width: 640, height: 480, fps: 30}}}}",
            "--display_data=true",
            "--teleop.type=so101_leader",
            f"--teleop.port={args.leader_port}",
            f"--teleop.id={args.leader_id}",
            f"--dataset.repo_id=local/eval2_{run_name}",
            f"--dataset.root={run_path}",
            "--dataset.num_episodes=1",
            f"--dataset.episode_time_s={args.episode_time_s}",
            f"--dataset.reset_time_s={args.reset_time_s}",
            f"--dataset.single_task={ep.prompt}",
            "--dataset.streaming_encoding=true",
            "--dataset.encoder_threads=2",
            "--dataset.push_to_hub=false",
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  ⚠️  lerobot-record exited rc={rc} — episode NOT marked recorded; press 'd' if you want to remove a partial dataset on disk.")
            continue

        ep.status = "recorded"
        ep.recorded_path = str(run_path)
        ep.timestamp = ts
        save_plan(plan, plan_path)
        print(f"  ✓ saved → {run_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
