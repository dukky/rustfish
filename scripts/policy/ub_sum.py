#!/usr/bin/env python3
"""Summarise `oracletest ... v=<d>` output (policy upper bounds: LMR, deep
ordering, root ordering).

usage: ub_sum.py <shard files...> [--base 0:0:100:100]
Per arm: nodes vs base (sum ratio; mean per-position log ratio, se), root
agreement with the deep search, and value loss of the chosen root move: each
distinct chosen move is re-searched (child at depth v, fresh state); loss =
best verified score among the candidates - the arm's move's verified score
(cp, capped at 300), mean and paired difference vs base with its se.
LMR arms also show coverage: share of LMR-reduced moves that are the oracle
best / oracle-bad.
"""
import math
import sys


def parse(paths):
    pos, cur = [], None
    for p in paths:
        for line in open(p):
            t = line.split()
            if not t:
                continue
            if t[0] == "POS":
                cur = {"arms": {}, "ver": {}}
                pos.append(cur)
            elif t[0] == "DEEP" and cur is not None:
                cur["deep"] = dict(kv.split("=", 1) for kv in t[1:])
            elif t[0] == "ARM" and cur is not None:
                f = dict(kv.split("=", 1) for kv in t[2:])
                cur["arms"][t[1]] = (int(f["nodes"]), f["bm"], [int(x) for x in f.get("lmr", "0,0,0").split(",")])
            elif t[0] == "VER" and cur is not None:
                cur["ver"][t[1]] = int(t[2])
    return [p for p in pos if "deep" in p and p["ver"]]


def ms(xs):
    n = len(xs)
    m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else 0
    return m, se


def main():
    args = sys.argv[1:]
    base = "0:0:100:100"
    if "--base" in args:
        i = args.index("--base")
        base = args[i + 1]
        del args[i:i + 2]
    pos = parse(args)
    arms = []
    for p in pos:
        for a in p["arms"]:
            if a not in arms:
                arms.append(a)
    pos = [p for p in pos if all(a in p["arms"] for a in arms) and all(p["arms"][a][1] in p["ver"] for a in arms)]
    n = len(pos)
    print(f"positions: {n}")

    def loss(p, a):
        best = max(p["ver"].values())
        return min(300, best - p["ver"][p["arms"][a][1]])

    bsum = sum(p["arms"][base][0] for p in pos)
    print(f"{'arm':<22}{'nodes':>8}{'logr':>8}{'se':>7}{'agree%':>8}{'loss':>7}{'dloss':>7}{'se':>6}{'>50cp%':>8}{'lmr best%':>10}{'bad%':>6}")
    for a in arms:
        tot = sum(p["arms"][a][0] for p in pos)
        lr, lse = ms([math.log(p["arms"][a][0] / p["arms"][base][0]) for p in pos])
        agr = 100 * sum(p["arms"][a][1] == p["deep"]["bm"] for p in pos) / n
        ls = [loss(p, a) for p in pos]
        dl, dse = ms([loss(p, a) - loss(p, base) for p in pos])
        big = 100 * sum(x > 50 for x in ls) / n
        lm = [sum(p["arms"][a][2][i] for p in pos) for i in range(3)]
        cov = f"{100 * lm[1] / lm[0]:>10.2f}{100 * lm[2] / lm[0]:>6.1f}" if lm[0] else ""
        print(f"{a:<22}{tot / bsum:>8.4f}{lr:>8.4f}{lse:>7.4f}{agr:>8.1f}{sum(ls) / n:>7.2f}{dl:>7.2f}{dse:>6.2f}{big:>8.1f}{cov}")


if __name__ == "__main__":
    main()
