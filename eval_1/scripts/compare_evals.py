#!/usr/bin/env python3
"""Compare results across all eval CSVs and pick the best checkpoint.

Usage:
    compare_evals.py               # scan eval_1/evals/*.csv
    compare_evals.py path/to/dir   # scan custom dir
"""
import csv
import glob
import os
import sys
from collections import defaultdict


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/lemonkey/LeMonkey/eval_1/evals"
    files = sorted(glob.glob(os.path.join(base, "ckpt*_*.csv")))
    if not files:
        print(f"No CSVs found under {base}")
        return 1

    print(f"Found {len(files)} eval session(s)\n")
    summary = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        # parse "ckpt020000_20260502_120000"
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
        summary.append({
            "session": name,
            "ckpt": ckpt,
            "n": n,
            "ok": ok,
            "rate": ok / n if n else 0,
            "per_color": dict(per_color),
        })

    # Print per-session
    print(f"{'session':45s} {'ckpt':>7s} {'rate':>10s} {'blue':>10s} {'red':>10s} {'green':>10s}")
    print("-" * 100)
    for s in summary:
        pc = s["per_color"]
        b = f"{pc.get('blue', [0,0])[0]}/{pc.get('blue', [0,0])[1]}"
        r = f"{pc.get('red',  [0,0])[0]}/{pc.get('red',  [0,0])[1]}"
        g = f"{pc.get('green',[0,0])[0]}/{pc.get('green',[0,0])[1]}"
        print(f"{s['session']:45s} {s['ckpt']:>7s} {s['ok']}/{s['n']:>4d} ({100*s['rate']:>3.0f}%) {b:>10s} {r:>10s} {g:>10s}")

    # Aggregate by checkpoint (in case multiple sessions for same ckpt)
    print("\n=== AGGREGATED BY CHECKPOINT ===")
    by_ckpt = defaultdict(lambda: [0, 0])
    by_ckpt_color = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for s in summary:
        by_ckpt[s["ckpt"]][0] += s["ok"]
        by_ckpt[s["ckpt"]][1] += s["n"]
        for color, (o, t) in s["per_color"].items():
            by_ckpt_color[s["ckpt"]][color][0] += o
            by_ckpt_color[s["ckpt"]][color][1] += t

    print(f"{'ckpt':>7s} {'rate':>12s} {'blue':>10s} {'red':>10s} {'green':>10s}")
    print("-" * 60)
    ranked = sorted(by_ckpt.items(), key=lambda x: -x[1][0] / max(x[1][1], 1))
    for ckpt, (o, t) in ranked:
        cd = by_ckpt_color[ckpt]
        b = f"{cd['blue'][0]}/{cd['blue'][1]}"
        r = f"{cd['red'][0]}/{cd['red'][1]}"
        g = f"{cd['green'][0]}/{cd['green'][1]}"
        rate_str = f"{o}/{t} ({100*o/t:.0f}%)" if t else "n/a"
        print(f"{ckpt:>7s} {rate_str:>12s} {b:>10s} {r:>10s} {g:>10s}")

    if ranked and ranked[0][1][1] > 0:
        winner = ranked[0][0]
        print(f"\n🏆 BEST CHECKPOINT: {winner}  ({ranked[0][1][0]}/{ranked[0][1][1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
