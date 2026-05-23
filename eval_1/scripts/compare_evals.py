#!/usr/bin/env python3
"""Compare results across all eval CSVs and pick the best checkpoint.

Reads every ckpt*_*.csv file under eval_1/evals/, prints per-session results
(broken down by color and by prompt type if available), then aggregates by
checkpoint and names the winner.

Usage:
    compare_evals.py               # scan eval_1/evals/*.csv
    compare_evals.py path/to/dir   # scan custom dir
"""
import csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    """Aggregate per-session eval CSVs and report the best-performing checkpoint."""
    base = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "LeMonkey/eval_1/evals")
    files = sorted(glob.glob(os.path.join(base, "ckpt*_*.csv")))
    if not files:
        print(f"No CSVs found under {base}")
        return 1

    print(f"Found {len(files)} eval session(s)\n")
    summary = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        ckpt = name.split("_")[0].replace("ckpt", "")
        rows = list(csv.DictReader(open(fp)))
        rows = [r for r in rows if r["success"] not in ("", "skipped")]
        n = len(rows)
        if n == 0:
            continue
        ok = sum(1 for r in rows if r["success"] == "1")

        per_color = defaultdict(lambda: [0, 0])
        for r in rows:
            per_color[r["color_target"]][0] += int(r["success"])
            per_color[r["color_target"]][1] += 1

        per_kind = defaultdict(lambda: [0, 0])
        has_kind = "prompt_type" in (rows[0].keys() if rows else {})
        if has_kind:
            for r in rows:
                k = r.get("prompt_type") or "unknown"
                per_kind[k][0] += int(r["success"])
                per_kind[k][1] += 1

        summary.append({
            "session": name,
            "ckpt": ckpt,
            "n": n,
            "ok": ok,
            "rate": ok / n if n else 0,
            "per_color": dict(per_color),
            "per_kind": dict(per_kind),
            "has_kind": has_kind,
        })

    # Per-session table
    has_any_kind = any(s["has_kind"] for s in summary)
    if has_any_kind:
        print(f"{'session':45s} {'ckpt':>7s} {'rate':>10s} {'blue':>8s} {'red':>8s} {'green':>8s} {'trained':>10s} {'untrained':>10s}")
    else:
        print(f"{'session':45s} {'ckpt':>7s} {'rate':>10s} {'blue':>8s} {'red':>8s} {'green':>8s}")
    print("-" * 120)
    for s in summary:
        pc = s["per_color"]
        b = f"{pc.get('blue', [0,0])[0]}/{pc.get('blue', [0,0])[1]}"
        r = f"{pc.get('red',  [0,0])[0]}/{pc.get('red',  [0,0])[1]}"
        g = f"{pc.get('green',[0,0])[0]}/{pc.get('green',[0,0])[1]}"
        rate_str = f"{s['ok']}/{s['n']} ({100*s['rate']:.0f}%)"
        if has_any_kind:
            pk = s["per_kind"]
            tr = f"{pk.get('trained',[0,0])[0]}/{pk.get('trained',[0,0])[1]}"
            ut = f"{pk.get('untrained',[0,0])[0]}/{pk.get('untrained',[0,0])[1]}"
            print(f"{s['session']:45s} {s['ckpt']:>7s} {rate_str:>10s} {b:>8s} {r:>8s} {g:>8s} {tr:>10s} {ut:>10s}")
        else:
            print(f"{s['session']:45s} {s['ckpt']:>7s} {rate_str:>10s} {b:>8s} {r:>8s} {g:>8s}")

    # Aggregate by checkpoint
    print("\n=== AGGREGATED BY CHECKPOINT ===")
    by_ckpt = defaultdict(lambda: [0, 0])
    by_ckpt_color = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    by_ckpt_kind = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for s in summary:
        by_ckpt[s["ckpt"]][0] += s["ok"]
        by_ckpt[s["ckpt"]][1] += s["n"]
        for color, (o, t) in s["per_color"].items():
            by_ckpt_color[s["ckpt"]][color][0] += o
            by_ckpt_color[s["ckpt"]][color][1] += t
        for kind, (o, t) in s["per_kind"].items():
            by_ckpt_kind[s["ckpt"]][kind][0] += o
            by_ckpt_kind[s["ckpt"]][kind][1] += t

    if has_any_kind:
        print(f"{'ckpt':>7s} {'rate':>14s} {'blue':>8s} {'red':>8s} {'green':>8s} {'trained':>10s} {'untrained':>10s}")
    else:
        print(f"{'ckpt':>7s} {'rate':>14s} {'blue':>8s} {'red':>8s} {'green':>8s}")
    print("-" * 80)
    ranked = sorted(by_ckpt.items(), key=lambda x: -x[1][0] / max(x[1][1], 1))
    for ckpt, (o, t) in ranked:
        cd = by_ckpt_color[ckpt]
        b = f"{cd['blue'][0]}/{cd['blue'][1]}"
        r = f"{cd['red'][0]}/{cd['red'][1]}"
        g = f"{cd['green'][0]}/{cd['green'][1]}"
        rate_str = f"{o}/{t} ({100*o/t:.0f}%)" if t else "n/a"
        if has_any_kind:
            kd = by_ckpt_kind[ckpt]
            tr = f"{kd['trained'][0]}/{kd['trained'][1]}"
            ut = f"{kd['untrained'][0]}/{kd['untrained'][1]}"
            print(f"{ckpt:>7s} {rate_str:>14s} {b:>8s} {r:>8s} {g:>8s} {tr:>10s} {ut:>10s}")
        else:
            print(f"{ckpt:>7s} {rate_str:>14s} {b:>8s} {r:>8s} {g:>8s}")

    if ranked and ranked[0][1][1] > 0:
        winner = ranked[0][0]
        print(f"\n🏆 BEST CHECKPOINT: {winner}  ({ranked[0][1][0]}/{ranked[0][1][1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
