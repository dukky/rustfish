#!/usr/bin/env python3
"""Fit a CCRL-scale rating from fastchess PGNs played against scripts/anchors.json.

Every game must involve the engine under test (default name "rustfish") and one
anchor whose PGN name matches an anchor "name". Results are aggregated per
anchor and fed to the same maximum-likelihood fit as fit_rating.py.

Usage:
  python3 scripts/ccrl_ladder.py --anchors scripts/anchors.json \
      --summary summary.md --json results.json games/*.pgn
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_rating import fit  # noqa: E402

TAG_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')


def iter_games(path):
    """Yield the tag dict of each game in a PGN file."""
    tags = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = TAG_RE.match(line)
            if m:
                if m.group(1) == "Event" and tags:
                    yield tags
                    tags = {}
                tags[m.group(1)] = m.group(2)
    if tags:
        yield tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--engine", default="rustfish")
    ap.add_argument("--summary", help="write a markdown summary here")
    ap.add_argument("--json", help="write machine-readable results here")
    ap.add_argument("pgns", nargs="+")
    args = ap.parse_args()

    with open(args.anchors) as fh:
        anchors = {a["name"]: a["ccrl"] for a in json.load(fh)["anchors"]}

    stats = {}
    time_losses = 0
    for path in args.pgns:
        for g in iter_games(path):
            white, black, result = g.get("White"), g.get("Black"), g.get("Result")
            if args.engine not in (white, black) or result not in ("1-0", "0-1", "1/2-1/2"):
                continue
            opp = black if white == args.engine else white
            if opp not in anchors:
                sys.exit(f"unknown opponent {opp!r} in {path}")
            s = stats.setdefault(opp, [0, 0, 0])
            if result == "1/2-1/2":
                s[1] += 1
            elif (result == "1-0") == (white == args.engine):
                s[0] += 1
            else:
                s[2] += 1
                if "time" in g.get("Termination", "").lower():
                    time_losses += 1

    if not stats:
        sys.exit("no finished games found")

    runs = [(anchors[name], *wdl) for name, wdl in stats.items()]
    best, lo, hi = fit(runs)

    rows = []
    for name, (w, d, l) in sorted(stats.items(), key=lambda kv: anchors[kv[0]]):
        n = w + d + l
        rows.append({"anchor": name, "ccrl": anchors[name], "wins": w, "draws": d,
                     "losses": l, "games": n, "score": (w + 0.5 * d) / n})
    total = sum(r["games"] for r in rows)

    lines = [
        f"## {args.engine}: CCRL Blitz-scale estimate **{best:.0f}** (95% CI {lo:.0f} .. {hi:.0f})",
        "",
        f"{total} games against {len(rows)} anchors; {time_losses} losses on time.",
        "",
        "| Anchor | CCRL | W | D | L | Games | Score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['anchor']} | {r['ccrl']} | {r['wins']} | {r['draws']} | "
                     f"{r['losses']} | {r['games']} | {100 * r['score']:.1f}% |")
    lines += [
        "",
        "CI covers game noise only; each anchor's own CCRL rating carries about +-20,",
        "which averages down across anchors but is not included above.",
    ]
    text = "\n".join(lines)
    print(text)

    if args.summary:
        with open(args.summary, "a") as fh:
            fh.write(text + "\n")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"engine": args.engine, "rating": best, "ci": [lo, hi],
                       "games": total, "time_losses": time_losses, "anchors": rows},
                      fh, indent=2)


if __name__ == "__main__":
    main()
