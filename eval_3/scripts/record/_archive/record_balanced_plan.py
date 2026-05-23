#!/usr/bin/env python3
"""Record Eval 3 teleop episodes against a balanced 144-episode plan.

Per PROJECT.md §2 (Eval 3) and eval_3/README.md (Path A: image-as-prompt
+ co-train, decided after Phase 1 PaliGemma probe yielded 0% name recall):

  - Workspace shows 3 A5 portraits (Swift / Obama / LeCun) arranged
    left/middle/right on the table.
  - Coke can in front of robot.
  - Per episode: a target celebrity and a *held-out* reference photo of that
    celebrity (NOT the photo that's on the workspace) are embedded in the
    prompt at inference time. The recorder writes the reference path into a
    sidecar  reference.json  next to the LeRobot dataset, so the data loader
    at training time can read it.
  - Prompt phrasing varies across episodes to combat phrasing-overfit.
  - 20 s per rollout (matches Eval 2 / PROJECT.md §2).

Design:
  • Plan = 3 targets × 6 layouts × 8 reps = 144 episodes.
  • Batched by layout so you reshuffle portraits ~6 times across the whole
    collection (not 144 times).
  • Within each batch, the 24 episodes are shuffled across targets so the
    target sequence is unbiased.
  • Reference photo cycles through whatever held-out files exist under
    datasets/eval3_celebs/heldout/<celeb>/  — fail loudly if a celeb has none.

Usage:
    record_eval3.py
    record_eval3.py --regenerate-plan --seed 42
    record_eval3.py --dry-run                # walk plan without recording

Controls during recording (matches record_episodes.py):
    ENTER   record next episode
    d       delete last recorded episode AND re-queue
    p       progress summary
    q       quit (state auto-saved)
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
from itertools import permutations
from pathlib import Path

# ─── Celebrity spec ──────────────────────────────────────────────────────────
CELEB_KEYS = ["swift", "obama", "lecun"]   # short keys used in layout codes
CELEB_INITIALS = {"swift": "S", "obama": "O", "lecun": "L"}
CELEB_NAMES = {"swift": "Taylor Swift", "obama": "Barack Obama", "lecun": "Yann LeCun"}
INITIAL_TO_KEY = {v: k for k, v in CELEB_INITIALS.items()}

# All 3! = 6 layouts. A 3-letter code S/O/L per left/middle/right position.
LAYOUTS = ["".join(p) for p in permutations("SOL")]   # ['SOL','SLO','OSL','OLS','LSO','LOS']

# Per-celeb held-out photo directory (sourced separately by the user / a downloader script).
HELDOUT_ROOT = Path("/home/lemonkey/LeMonkey/datasets/eval3_celebs/heldout")

# ─── Phrasing pool (broad to combat phrasing-overfit) ────────────────────────
PROMPT_PHR = [
    "Place the coke on {name}.",
    "Place the coke can on {name}.",
    "Put the coke on {name}.",
    "Put the can on {name}.",
    "Place the can on the photo of {name}.",
    "Place the coke on the picture of {name}.",
    "Put the coke can on the picture of {name}.",
    "Drop the coke on {name}.",
    "Place the coke on top of {name}.",
    "Set the coke can on {name}.",
]


# ─── Plan + episode model ───────────────────────────────────────────────────
@dataclass
class Episode:
    idx: int
    target_celeb: str        # "swift" | "obama" | "lecun"
    layout: str              # 3 letters from {S,O,L}, order = left/middle/right
    target_idx: int          # 0/1/2; derived from layout + target
    reference_photo: str     # absolute path to a held-out photo of target_celeb
    prompt: str
    status: str = "pending"  # pending | recorded | deleted
    recorded_path: str | None = None
    timestamp: str | None = None


@dataclass
class Plan:
    episodes: list[Episode] = field(default_factory=list)
    seed: int = 42

    def to_json(self) -> dict:
        return {"seed": self.seed, "episodes": [asdict(e) for e in self.episodes]}

    @classmethod
    def from_json(cls, data: dict) -> "Plan":
        return cls(seed=data.get("seed", 42), episodes=[Episode(**e) for e in data["episodes"]])


# ─── Reference-photo discovery ──────────────────────────────────────────────
def list_heldout(celeb_key: str) -> list[Path]:
    d = HELDOUT_ROOT / celeb_key
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def assert_heldout_photos_exist() -> None:
    """Hard check before generating a plan — refuse to proceed without photos."""
    missing = []
    for k in CELEB_KEYS:
        if not list_heldout(k):
            missing.append(k)
    if missing:
        msg = (
            f"\n[ERROR] No held-out photos found for: {missing}\n"
            f"Expected at least one .jpg/.png per celebrity under:\n"
            + "\n".join(f"   {HELDOUT_ROOT}/{k}/" for k in CELEB_KEYS)
            + "\nCreate the dirs and add a few photos (5+ each is recommended).\n"
        )
        raise SystemExit(msg)


# ─── Layout helpers ─────────────────────────────────────────────────────────
def target_position_in_layout(layout: str, target_celeb: str) -> int:
    initial = CELEB_INITIALS[target_celeb]
    return layout.index(initial)


def layout_to_human(layout: str) -> str:
    """SOL → 'Swift / Obama / LeCun  (L/M/R)'"""
    parts = [CELEB_NAMES[INITIAL_TO_KEY[c]] for c in layout]
    return "  /  ".join(parts) + "    (left / middle / right)"


# ─── Plan generation ────────────────────────────────────────────────────────
def generate_plan(seed: int, reps_per_cell: int = 8) -> Plan:
    """Build a balanced 3 × 6 × reps-episode plan, batched by layout."""
    rng = random.Random(seed)

    # Per-celeb cycling iterator over held-out photos (deterministic given seed)
    photo_pools: dict[str, list[Path]] = {k: list_heldout(k) for k in CELEB_KEYS}
    if any(not v for v in photo_pools.values()):
        raise RuntimeError(f"missing photos: {[k for k,v in photo_pools.items() if not v]}")
    # Shuffle each pool once with the seed so cycling order is deterministic
    for k in CELEB_KEYS:
        rng_pool = random.Random(seed + sum(ord(c) for c in k))
        rng_pool.shuffle(photo_pools[k])
    photo_cursor: dict[str, int] = {k: 0 for k in CELEB_KEYS}

    def next_photo(celeb: str) -> Path:
        pool = photo_pools[celeb]
        p = pool[photo_cursor[celeb] % len(pool)]
        photo_cursor[celeb] += 1
        return p

    plan_episodes: list[Episode] = []

    # Outer order of layouts (random permutation of the 6)
    layout_order = list(LAYOUTS)
    rng.shuffle(layout_order)

    for layout in layout_order:
        batch: list[Episode] = []
        for celeb in CELEB_KEYS:
            for _ in range(reps_per_cell):
                ti = target_position_in_layout(layout, celeb)
                phrasing = rng.choice(PROMPT_PHR).format(name=CELEB_NAMES[celeb])
                ref = next_photo(celeb)
                batch.append(Episode(
                    idx=-1,  # filled after batch shuffle
                    target_celeb=celeb,
                    layout=layout,
                    target_idx=ti,
                    reference_photo=str(ref),
                    prompt=phrasing,
                ))
        # Shuffle within batch so target order is unbiased; layout stays fixed
        rng.shuffle(batch)
        plan_episodes.extend(batch)

    for i, ep in enumerate(plan_episodes, 1):
        ep.idx = i

    return Plan(episodes=plan_episodes, seed=seed)


# ─── Plan persistence ────────────────────────────────────────────────────────
def load_or_create_plan(path: Path, seed: int, regenerate: bool, reps: int) -> Plan:
    if path.exists() and not regenerate:
        try:
            return Plan.from_json(json.loads(path.read_text()))
        except Exception as e:
            print(f"[WARN] failed to load existing plan ({e}); regenerating", flush=True)
    plan = generate_plan(seed, reps)
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


# ─── Progress summary ───────────────────────────────────────────────────────
def progress_summary(plan: Plan) -> str:
    n_total = len(plan.episodes)
    by_status: dict[str, int] = {}
    by_layout: dict[str, dict[str, int]] = {l: {"pending": 0, "recorded": 0, "deleted": 0} for l in LAYOUTS}
    by_target: dict[str, dict[str, int]] = {c: {"pending": 0, "recorded": 0, "deleted": 0} for c in CELEB_KEYS}
    for ep in plan.episodes:
        by_status[ep.status] = by_status.get(ep.status, 0) + 1
        by_layout[ep.layout][ep.status] += 1
        by_target[ep.target_celeb][ep.status] += 1
    n_done = by_status.get("recorded", 0)
    out = [f"  total: {n_total}    recorded: {n_done}    pending: {by_status.get('pending', 0)}    deleted: {by_status.get('deleted', 0)}"]
    out.append("")
    out.append(f"  {'layout':12s}  recorded / total")
    for l in LAYOUTS:
        c = by_layout[l]
        out.append(f"  {l:12s}  {c['recorded']:>2} / {c['recorded']+c['pending']+c['deleted']}")
    out.append("")
    out.append(f"  {'target':12s}  recorded / total")
    for c in CELEB_KEYS:
        d = by_target[c]
        out.append(f"  {CELEB_NAMES[c]:14s}  {d['recorded']:>2} / {d['recorded']+d['pending']+d['deleted']}")
    return "\n".join(out)


# ─── Delete-last-recorded helper ────────────────────────────────────────────
def delete_last_recorded(plan: Plan) -> Episode | None:
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
    last.status = "pending"
    last.recorded_path = None
    last.timestamp = None
    plan.episodes.remove(last)
    insert_at = next((i for i, e in enumerate(plan.episodes) if e.status == "pending"), len(plan.episodes))
    plan.episodes.insert(insert_at, last)
    return last


# ─── Sidecar reference.json ─────────────────────────────────────────────────
def write_reference_sidecar(run_path: Path, ep: Episode) -> None:
    """Write the reference-photo metadata next to the LeRobot dataset.

    The training data loader will read this to know which photo to feed as
    the image-as-prompt for this episode.
    """
    payload = {
        "episode_idx": ep.idx,
        "target_celeb": ep.target_celeb,
        "target_celeb_name": CELEB_NAMES[ep.target_celeb],
        "layout": ep.layout,
        "target_idx": ep.target_idx,
        "reference_photo": ep.reference_photo,
        "prompt": ep.prompt,
    }
    sidecar = run_path / "reference.json"
    sidecar.write_text(json.dumps(payload, indent=2))


# ─── Main ───────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan-path", default="/home/lemonkey/LeMonkey/eval_3/state/plan.json")
    p.add_argument("--regenerate-plan", action="store_true",
                   help="Discard existing plan and start over (this resets all progress)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reps-per-cell", type=int, default=8,
                   help="Episodes per (target × layout) cell. 8 → 144 total; 9 → 162.")
    p.add_argument("--episode-time-s", type=float, default=20.0)
    p.add_argument("--reset-time-s",   type=float, default=10.0)
    p.add_argument("--root", default="/home/lemonkey/LeMonkey/datasets/eval3")
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

    # Hard-fail loudly if held-out photos aren't there yet.
    if not plan_path.exists() or args.regenerate_plan:
        assert_heldout_photos_exist()

    plan = load_or_create_plan(plan_path, seed=args.seed,
                               regenerate=args.regenerate_plan,
                               reps=args.reps_per_cell)
    save_plan(plan, plan_path)

    print("=" * 72)
    print(f"  Eval 3 teleop recording (Path A: image-as-prompt + co-train)")
    print(f"  plan file   : {plan_path}")
    print(f"  output root : {args.root}")
    print(f"  dry run     : {args.dry_run}")
    print("=" * 72)
    print()
    print(progress_summary(plan))
    print()

    last_layout = None
    while True:
        ep = next_pending(plan)
        if ep is None:
            print("\n🎉  plan complete — no pending episodes remaining.")
            print(progress_summary(plan))
            return 0

        # Loud banner on layout change so user reshuffles portraits
        if ep.layout != last_layout:
            print()
            print("╔" + "═" * 70 + "╗")
            print(f"║  LAYOUT CHANGE → please set portraits to: {ep.layout:<25}".ljust(71) + " ║")
            print(f"║  {layout_to_human(ep.layout)}".ljust(71) + " ║")
            print("╚" + "═" * 70 + "╝")
            last_layout = ep.layout

        n_total = len(plan.episodes)
        n_done = sum(1 for e in plan.episodes if e.status == "recorded")
        print()
        print("─" * 72)
        print(f" Episode {ep.idx:>3} of {n_total}    ({n_done} recorded so far)")
        print(f"   layout      : {ep.layout}    target : {CELEB_NAMES[ep.target_celeb]}")
        print(f"   prompt      : \"{ep.prompt}\"")
        print(f"   TARGET POS  : idx {ep.target_idx} = {['LEFT','MIDDLE','RIGHT'][ep.target_idx]}")
        print(f"   reference   : {ep.reference_photo}")
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
                print(f"  re-queued episode {removed.idx} ({CELEB_NAMES[removed.target_celeb]} on {removed.layout}, prompt: \"{removed.prompt}\")")
                save_plan(plan, plan_path)
            continue

        # Record
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ep_{ep.idx:04d}_{ep.layout}_{ep.target_celeb}_{ts}"
        run_path = Path(args.root) / run_name

        if args.dry_run:
            print(f"  [dry-run] would record → {run_path}")
            run_path.mkdir(parents=True, exist_ok=True)
            write_reference_sidecar(run_path, ep)
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
            f"--dataset.repo_id=local/eval3_{run_name}",
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

        write_reference_sidecar(run_path, ep)
        ep.status = "recorded"
        ep.recorded_path = str(run_path)
        ep.timestamp = ts
        save_plan(plan, plan_path)
        print(f"  ✓ saved → {run_path}")
        print(f"  ✓ wrote reference sidecar → {run_path / 'reference.json'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
