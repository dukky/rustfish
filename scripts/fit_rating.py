#!/usr/bin/env python3
"""Fit a single maximum-likelihood rating from gauntlet JSONL ladder results.

Each input file is a gauntlet run against one fixed-rating anchor. Draws count
as half wins. The fit maximizes the binomial log-likelihood of the observed
scores under the logistic expected-score model, and reports a ~95% CI from the
likelihood ratio (2*dlnL <= 3.84).

Usage:
  python3 scripts/fit_rating.py --run 1900=tmp/ladder500_sf1900.jsonl \
      --run 2050=tmp/ladder500_sf2050.jsonl ...
"""
import argparse
import json
import math
import sys


def load_run(path):
    wins = draws = losses = 0
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "run_end":
                wins = rec["wins"]
                draws = rec["draws"]
                losses = rec["losses"]
    return wins, draws, losses


def expected(r, anchor):
    return 1.0 / (1.0 + 10.0 ** ((anchor - r) / 400.0))


def log_lik(r, runs):
    total = 0.0
    for anchor, wins, draws, losses in runs:
        n = wins + draws + losses
        if n == 0:
            continue
        s = (wins + 0.5 * draws) / n
        e = min(max(expected(r, anchor), 1e-9), 1.0 - 1e-9)
        total += n * (s * math.log(e) + (1.0 - s) * math.log(1.0 - e))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help="anchor_rating=path/to/gauntlet.jsonl (repeatable)",
    )
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        anchor_str, _, path = spec.partition("=")
        wins, draws, losses = load_run(path)
        anchor = float(anchor_str)
        runs.append((anchor, wins, draws, losses))
        n = wins + draws + losses
        score = (wins + 0.5 * draws) / n if n else float("nan")
        print(
            f"anchor {anchor:6.0f}: {wins}W {draws}D {losses}L "
            f"({n} games, score {100 * score:.1f}%)"
        )

    if not any(w + d + l for _, w, d, l in runs):
        sys.exit("no finished games found")

    grid = [r / 10.0 for r in range(10000, 40001)]  # 1000..4000 in 0.1 steps
    best = max(grid, key=lambda r: log_lik(r, runs))
    best_ll = log_lik(best, runs)

    lo = hi = best
    for r in grid:
        if 2.0 * (best_ll - log_lik(r, runs)) <= 3.84:
            lo = min(lo, r)
            hi = max(hi, r)

    print()
    print(f"ML rating estimate: {best:.0f}  (95% CI approx {lo:.0f} .. {hi:.0f})")
    print("Scale caveat: anchored to the opponents' nominal ratings; only as")
    print("accurate as those anchors at the tested time control.")


if __name__ == "__main__":
    main()
