#!/usr/bin/env python3
"""Pentanomial SPRT on logistic Elo, plus helpers for fastchess output.

A "pair" is the two games played on one opening with colors swapped. Its
score is 0, 0.25, 0.5, 0.75 or 1, and the pentanomial counts how many pairs
landed on each. Counting pairs instead of games removes most of the noise that
opening choice adds, which roughly halves the games an SPRT needs.

The LLR uses the generalized-SPRT normal approximation used by OpenBench:
    LLR = N * (s1 - s0) * (2*mu - s0 - s1) / (2 * var)
with mu/var the mean/variance of the pair score and s0/s1 the expected
scores under H0 (elo0) and H1 (elo1).

Usage:
  python3 scripts/sprt.py --ptnml 10,120,400,130,12 --elo0 0 --elo1 5
  python3 scripts/sprt.py --fastchess-log shard1.log shard2.log --elo0 0 --elo1 5
"""
import argparse
import json
import math
import re
import sys

PAIR_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)
PTNML_RE = re.compile(r"Ptnml\(0-2\):\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]")


def elo_to_score(elo):
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score):
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


def stats(ptnml):
    n = sum(ptnml)
    if n == 0:
        return 0, 0.5, 0.0
    # Regularize empty buckets so a short, lopsided run has nonzero variance.
    counts = [c if c > 0 else 1e-3 for c in ptnml]
    total = sum(counts)
    mu = sum(c * s for c, s in zip(counts, PAIR_SCORES)) / total
    var = sum(c * (s - mu) ** 2 for c, s in zip(counts, PAIR_SCORES)) / total
    return n, mu, var


def llr(ptnml, elo0, elo1):
    n, mu, var = stats(ptnml)
    if n == 0 or var <= 0:
        return 0.0
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return n * (s1 - s0) * (2.0 * mu - s0 - s1) / (2.0 * var)


def bounds(alpha=0.05, beta=0.05):
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def elo_estimate(ptnml):
    """Return (elo, lo95, hi95) from the pair-score mean and its standard error."""
    n, mu, var = stats(ptnml)
    if n == 0:
        return 0.0, 0.0, 0.0
    se = math.sqrt(var / n)
    return score_to_elo(mu), score_to_elo(mu - 1.96 * se), score_to_elo(mu + 1.96 * se)


def evaluate(ptnml, elo0, elo1, alpha=0.05, beta=0.05):
    lo, hi = bounds(alpha, beta)
    value = llr(ptnml, elo0, elo1)
    if value >= hi:
        status = "H1"   # candidate is at least elo1-ish better: accept
    elif value <= lo:
        status = "H0"   # candidate is not better than elo0: reject
    else:
        status = "continue"
    elo, elo_lo, elo_hi = elo_estimate(ptnml)
    pairs = sum(ptnml)
    return {
        "status": status, "llr": round(value, 3), "llr_bounds": [round(lo, 3), round(hi, 3)],
        "elo0": elo0, "elo1": elo1, "ptnml": list(ptnml), "pairs": pairs, "games": 2 * pairs,
        "elo": round(elo, 2), "elo_ci95": [round(elo_lo, 2), round(elo_hi, 2)],
    }


def ptnml_from_fastchess_log(path):
    """Return the last Ptnml(0-2) line in a fastchess log (its final cumulative count)."""
    last = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = PTNML_RE.search(line)
            if m:
                last = [int(x) for x in m.groups()]
    if last is None:
        raise ValueError(f"no Ptnml line in {path}")
    return last


def parse_ptnml(text):
    vals = [int(x) for x in text.split(",")]
    if len(vals) != 5:
        raise ValueError(f"pentanomial needs 5 counts, got {text!r}")
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptnml", action="append", default=[], help="a,b,c,d,e (repeatable; summed)")
    ap.add_argument("--fastchess-log", nargs="*", default=[], help="fastchess stdout logs to sum")
    ap.add_argument("--elo0", type=float, required=True)
    ap.add_argument("--elo1", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args()

    total = [0] * 5
    for spec in args.ptnml:
        total = [a + b for a, b in zip(total, parse_ptnml(spec))]
    for path in args.fastchess_log:
        total = [a + b for a, b in zip(total, ptnml_from_fastchess_log(path))]
    if sum(total) == 0:
        sys.exit("no pairs given")

    res = evaluate(total, args.elo0, args.elo1, args.alpha, args.beta)
    if args.json:
        print(json.dumps(res))
        return
    print(f"pentanomial {res['ptnml']}  ({res['games']} games)")
    print(f"Elo {res['elo']:+.1f}  (95% CI {res['elo_ci95'][0]:+.1f} .. {res['elo_ci95'][1]:+.1f})")
    print(f"LLR {res['llr']:+.2f}  bounds {res['llr_bounds']}  [{res['elo0']}, {res['elo1']}]  -> {res['status']}")


if __name__ == "__main__":
    main()
