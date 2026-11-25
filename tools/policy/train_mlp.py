#!/usr/bin/env python3
"""Tiny numpy MLP ranker for policy-order v0 experiments.

Pairwise logistic loss over (best move, sampled alternatives). This is a
prototype only: the feature set is offline and lacks live-search signals.
"""
import argparse
import json
import os
import sys

import numpy as np


def load(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not rows:
        sys.exit("empty dataset")
    return rows


def split(rows):
    games = sorted({r["game_id"] for r in rows})
    test = {g for g in games if (hash(g) & 0x7FFFFFFF) % 5 == 0}
    if not test and len(games) > 1:
        test = {games[-1]}
    tr = [r for r in rows if r["game_id"] not in test]
    te = [r for r in rows if r["game_id"] in test]
    if not tr:
        tr, te = rows[::2], rows[1::2]
    return tr, te


def pairs(rows, max_neg, rng):
    P, N = [], []
    for r in rows:
        pos = [m for m in r["moves"] if m["label"]]
        neg = [m for m in r["moves"] if not m["label"]]
        if not pos or not neg:
            continue
        p = np.asarray(pos[0]["x"], dtype=np.float64)
        idx = rng.choice(len(neg), size=min(max_neg, len(neg)), replace=False)
        for j in idx:
            P.append(p)
            N.append(np.asarray(neg[int(j)]["x"], dtype=np.float64))
    if not P:
        sys.exit("no pairs")
    return np.asarray(P), np.asarray(N)


def standardize(x, *others):
    mean = x.mean(0)
    std = x.std(0)
    std[std < 1e-6] = 1.0
    return [(a - mean) / std for a in others], mean, std


def forward(x, W1, b1, w2, b2):
    h = np.maximum(0.0, x @ W1 + b1)
    return h, h @ w2 + b2


def evaluate(rows, W1, b1, w2, b2, mean, std):
    top1 = 0
    ranks = []
    for r in rows:
        moves = r["moves"]
        if len(moves) < 2:
            continue
        x = (np.asarray([m["x"] for m in moves], dtype=np.float64) - mean) / std
        _, y = forward(x, W1, b1, w2, b2)
        order = np.argsort(-y)
        best = next((i for i, m in enumerate(moves) if m["label"]), None)
        if best is None:
            continue
        top1 += int(order[0] == best)
        ranks.append(int(np.where(order == best)[0][0]) + 1)
    if not ranks:
        return 0.0, 0.0, 0
    return top1 / len(ranks), float(np.mean(ranks)), len(ranks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--negatives", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261124)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = load(args.data)
    tr, te = split(rows)
    P, N = pairs(tr, args.negatives, rng)
    (P, N), mean, std = standardize(P, P, N)
    d = P.shape[1]
    W1 = rng.normal(0, 1 / np.sqrt(d), (d, args.hidden))
    b1 = np.zeros(args.hidden)
    w2 = rng.normal(0, 1 / np.sqrt(args.hidden), args.hidden)
    b2 = 0.0

    n = len(P)
    for epoch in range(args.epochs):
        hp, yp = forward(P, W1, b1, w2, b2)
        hn, yn = forward(N, W1, b1, w2, b2)
        margin = yp - yn
        sig = 1.0 / (1.0 + np.exp(np.clip(margin, -60, 60)))
        # d loss / d margin = -(1 - sig)
        dm = -(1.0 - sig)
        # gradients of output
        gw2 = (dm[:, None] * (hp - hn)).mean(0)
        gb2 = dm.mean()
        # ReLU backprop
        dp = (dm[:, None] * w2[None, :]) * (hp > 0)
        dn = (-dm[:, None] * w2[None, :]) * (hn > 0)
        gW1 = (P.T @ dp + N.T @ dn) / n
        gb1 = (dp + dn).mean(0)
        W1 -= args.lr * gW1
        b1 -= args.lr * gb1
        w2 -= args.lr * gw2
        b2 -= args.lr * gb2
        if epoch % 100 == 0:
            print(f"epoch {epoch} loss {np.logaddexp(0, -margin).mean():.4f}", file=sys.stderr)

    top1, rank, ntest = evaluate(te, W1, b1, w2, b2, mean, std)
    metrics = {
        "positions_train": len(tr),
        "positions_test": len(te),
        "pairs_train": int(len(P)),
        "test_positions_scored": ntest,
        "top1": round(top1, 4),
        "mean_rank": round(rank, 3),
        "hidden": args.hidden,
        "epochs": args.epochs,
        "negatives": args.negatives,
    }
    model = {
        "type": "mlp_pairwise",
        "features": rows[0]["features"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "w2": w2.tolist(),
        "b2": float(b2),
        "metrics": metrics,
        "note": "Prototype; not loaded by the engine.",
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(model, fh)
        fh.write("\n")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
