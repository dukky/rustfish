#!/usr/bin/env python3
"""Analyse an `nnue-trainer uncert-dump` CSV: does the uncertainty head u
predict the net's actual error beyond trivial baselines?

Error: t = |sigmoid(score/400) - sigmoid(eval/400)| (WDL space, what u is
trained on) and tcp = |score - eval| in centipawns. "Large" errors:
t > sigmoid(100/400) - 0.5 (= 100 cp at equality) and tcp > 100.

Baselines are fitted (cell means) on the even rows and every predictor is
scored on the odd rows, so the cell model cannot win by memorising.

    python3 -I scripts/uncert_analysis.py [--mul 0.5] val16.csv [--mul 0.75 val.csv ...]

`--mul` converts the dumped raw net output to the units of the record's
search score (0.5 = the engine's halved static eval; applies to the files
after it).
"""
import csv
import math
import random
import sys

LARGE_T = 1 / (1 + math.exp(-0.25)) - 0.5  # 0.0622


def sig(x):
    return 1 / (1 + math.exp(-x / 400))


def ranks(xs):
    order = sorted(range(len(xs)), key=xs.__getitem__)
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")


def auc(pred_ranks, labels):
    """Mann-Whitney AUC of a predictor (given its ranks) for binary labels."""
    pos = sum(labels)
    neg = len(labels) - pos
    s = sum(r for r, l in zip(pred_ranks, labels) if l)
    # ranks are 0-based
    return (s - pos * (pos - 1) / 2) / (pos * neg)


def top_capture(pred, labels, frac=0.1, seed=1):
    """Share of all positives that fall in the top `frac` by pred (ties broken at random)."""
    rnd = random.Random(seed)
    keyed = sorted(range(len(pred)), key=lambda i: (pred[i], rnd.random()), reverse=True)
    k = int(len(pred) * frac)
    pos = sum(labels)
    return sum(labels[i] for i in keyed[:k]) / pos


def bin_of(x, edges):
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def load(path, mul):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            e, s = float(r["eval"]) * mul, float(r["score"])
            rows.append(
                {
                    "eval": e,
                    "score": s,
                    "u": float(r["u"]),
                    "pieces": int(r["pieces"]),
                    "phase": int(r["phase"]),
                    "material": int(r["material"]),
                    "ep_us": int(r["enprise_us"]),
                    "ep_them": int(r["enprise_them"]),
                    "att": int(r["attacked_us"]) + int(r["attacked_them"]),
                    "t": abs(sig(s) - sig(e)),
                    "tcp": min(abs(s - e), 2000.0),
                }
            )
    return rows


def cell(r):
    return (
        bin_of(abs(r["eval"]), [25, 50, 100, 150, 200, 300, 500, 800]),
        r["pieces"] // 4,
        min(r["ep_us"], 2),
        min(r["ep_them"], 2),
        bin_of(abs(r["material"] * 100 - r["eval"]), [50, 100, 200, 400]),
    )


def cell_model(fit_rows, key, target):
    sums = {}
    for r in fit_rows:
        c = key(r)
        a = sums.setdefault(c, [0.0, 0])
        a[0] += r[target]
        a[1] += 1
    glob = sum(r[target] for r in fit_rows) / len(fit_rows)
    # shrink sparse cells toward the global mean
    return {c: (s + 50 * glob) / (n + 50) for c, (s, n) in sums.items()}, glob


def analyse(path, mul):
    rows = load(path, mul)
    print(f"\n=== {path}: {len(rows)} positions, eval = {mul} x net output")
    n = len(rows)
    mt = sum(r["t"] for r in rows) / n
    mu = sum(r["u"] for r in rows) / n
    print(f"mean t {mt:.5f}  mean u {mu:.5f}  mean |score-eval| {sum(r['tcp'] for r in rows)/n:.1f}cp")
    print(f"large (t>{LARGE_T:.4f}) {sum(r['t'] > LARGE_T for r in rows)/n:.3%}   large (tcp>100) {sum(r['tcp'] > 100 for r in rows)/n:.3%}")

    # calibration by u decile (all rows)
    order = sorted(range(n), key=lambda i: rows[i]["u"])
    print("\nu decile | mean u | mean t | mean |s-e| cp | P(t large) | P(cp>100)")
    for d in range(10):
        idx = order[d * n // 10 : (d + 1) * n // 10]
        m = len(idx)
        print(
            f"  {d+1:2d}     | {sum(rows[i]['u'] for i in idx)/m:.4f} | {sum(rows[i]['t'] for i in idx)/m:.4f} |"
            f" {sum(rows[i]['tcp'] for i in idx)/m:7.1f}     | {sum(rows[i]['t'] > LARGE_T for i in idx)/m:7.2%}    | {sum(rows[i]['tcp'] > 100 for i in idx)/m:7.2%}"
        )

    fit = rows[0::2]
    test = rows[1::2]
    # baseline cell model, and cell model x u-quintile (both fit on even rows)
    model, glob = cell_model(fit, cell, "t")
    uq_edges = sorted(r["u"] for r in fit)
    uq_edges = [uq_edges[len(uq_edges) * k // 5] for k in range(1, 5)]
    model_u, glob_u = cell_model(fit, lambda r: cell(r) + (bin_of(r["u"], uq_edges),), "t")

    preds = {
        "u (head)": [r["u"] for r in test],
        "-|eval|": [-abs(r["eval"]) for r in test],
        "pieces": [r["pieces"] for r in test],
        "phase": [r["phase"] for r in test],
        "en prise (both)": [r["ep_us"] + r["ep_them"] for r in test],
        "attacked (both)": [r["att"] for r in test],
        "|material-eval|": [abs(r["material"] * 100 - r["eval"]) for r in test],
        "baseline cells": [model.get(cell(r), glob) for r in test],
        "cells x u-quintile": [model_u.get(cell(r) + (bin_of(r["u"], uq_edges),), glob_u) for r in test],
    }
    t = [r["t"] for r in test]
    tcp = [r["tcp"] for r in test]
    lt = [r["t"] > LARGE_T for r in test]
    lcp = [r["tcp"] > 100 for r in test]
    rt, rtcp = ranks(t), ranks(tcp)
    print(f"\nscored on odd rows ({len(test)}); cells fit on even rows")
    print("predictor           | spearman t | pearson t | spearman cp | AUC t>100cp-eq | top-10% capture t | top-10% capture cp>100")
    for name, p in preds.items():
        rp = ranks(p)
        print(
            f"  {name:18s}| {pearson(rp, rt):+.4f}    | {pearson(p, t):+.4f}   | {pearson(rp, rtcp):+.4f}     |"
            f" {auc(rp, lt):.4f}         | {top_capture(p, lt):.3f}             | {top_capture(p, lcp):.3f}"
        )
    # does u add information beyond the cells? correlation with the cell model's residual
    base = preds["baseline cells"]
    resid = [a - b for a, b in zip(t, base)]
    print(f"\nspearman(u, t - cellpred) = {pearson(ranks(preds['u (head)']), ranks(resid)):+.4f}")
    mse = lambda p: sum((a - b) ** 2 for a, b in zip(p, t)) / len(t)
    var = mse([sum(t) / len(t)] * len(t))
    for name in ["baseline cells", "u (head)", "cells x u-quintile"]:
        print(f"R^2 vs t, {name:18s}: {1 - mse(preds[name]) / var:.4f}")


if __name__ == "__main__":
    mul = 0.5  # src/eval.rs halves the net output
    args = sys.argv[1:]
    while args:
        a = args.pop(0)
        if a == "--mul":
            mul = float(args.pop(0))
        else:
            analyse(a, mul)
