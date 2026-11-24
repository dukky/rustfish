#!/usr/bin/env python3
"""Train a small pairwise linear ranker from policy-data.jsonl.

This is a Phase-1 prototype, not an engine integration. The model is a
pairwise logistic ranker: the engine's best move is the positive and legal
alternatives are sampled as negatives.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

FEATURE_NAMES = None


def load_data(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        sys.exit("empty dataset")
    return rows


def split_rows(rows):
    """Group by game id; put roughly 20% of games in test."""
    games = sorted({r["game_id"] for r in rows})
    test_games = {g for g in games if (hash(g) & 0x7FFFFFFF) % 5 == 0}
    if not test_games and len(games) > 1:
        test_games = {games[-1]}
    train = [r for r in rows if r["game_id"] not in test_games]
    test = [r for r in rows if r["game_id"] in test_games]
    if not train:
        # Tiny datasets: fall back to a deterministic row split.
        train = rows[::2]
        test = rows[1::2]
    return train, test


def rows_to_pairs(rows, negatives, rng):
    """Return positive/negative feature pairs for pairwise training."""
    pos, neg = [], []
    for row in rows:
        moves = row["moves"]
        positives = [m for m in moves if m["label"]]
        negatives_all = [m for m in moves if not m["label"]]
        if not positives or not negatives_all:
            continue
        p = positives[0]
        take = min(negatives, len(negatives_all))
        idx = rng.choice(len(negatives_all), size=take, replace=False)
        for j in idx:
            pos.append(p["x"])
            neg.append(negatives_all[int(j)]["x"])
    if not pos:
        sys.exit("no pairwise examples")
    return np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)


def standardize(train_x, *others):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return [(x - mean) / std for x in others], mean, std


def top1_and_rank(rows, weights, mean, std, features):
    top1 = 0
    rank_sum = 0
    n = 0
    for row in rows:
        moves = row["moves"]
        if len(moves) < 2:
            continue
        x = np.asarray([m["x"] for m in moves], dtype=np.float64)
        z = ((x - mean) / std) @ weights
        order = np.argsort(-z)
        best = next((i for i, m in enumerate(moves) if m["label"]), None)
        if best is None:
            continue
        n += 1
        if int(order[0]) == best:
            top1 += 1
        rank_sum += int(np.where(order == best)[0][0]) + 1
    if n == 0:
        return 0.0, 0.0, 0
    return top1 / n, rank_sum / n, n


def heuristic_scores(moves, features):
    idx = {name: i for i, name in enumerate(features)}
    out = []
    for m in moves:
        x = m["x"]
        score = (
            100.0 * x[idx["capture_value"]]
            + 25.0 * x[idx["is_check"]]
            + 5.0 * x[idx["to_center"]]
            + 4.0 * x[idx["is_castle"]]
            + 8.0 * x[idx["promotion_value"]]
            + 2.0 * x[idx["relative_rank"]]
            - 1.0 * x[idx["to_attacked"]]
        )
        out.append(score)
    return np.asarray(out, dtype=np.float64)


def heuristic_top1(rows, features):
    top1 = 0
    n = 0
    for row in rows:
        moves = row["moves"]
        if len(moves) < 2:
            continue
        scores = heuristic_scores(moves, features)
        best = next((i for i, m in enumerate(moves) if m["label"]), None)
        if best is None:
            continue
        n += 1
        if int(np.argmax(scores)) == best:
            top1 += 1
    return top1 / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--negatives", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261124)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = load_data(args.data)
    features = rows[0].get("features")
    if not features:
        sys.exit("dataset missing feature names")
    train_rows, test_rows = split_rows(rows)
    pos, neg = rows_to_pairs(train_rows, args.negatives, rng)
    (pos_s, neg_s), mean, std = standardize(pos, pos, neg)
    # Standardize negatives with the positive statistics (same feature space).

    w = np.zeros(pos_s.shape[1], dtype=np.float64)
    n = len(pos_s)
    for epoch in range(args.epochs):
        margin = (pos_s - neg_s) @ w
        # pairwise logistic loss: log(1 + exp(-margin))
        # gradient = -sigmoid(-margin) * (pos - neg)
        sig = 1.0 / (1.0 + np.exp(np.clip(margin, -60, 60)))
        grad = -((1.0 - sig)[:, None] * (pos_s - neg_s)).mean(axis=0)
        grad += args.l2 * w
        w -= args.lr * grad
        if epoch % 250 == 0:
            print(f"epoch {epoch} w_norm {np.linalg.norm(w):.4f}", file=sys.stderr)

    top1, mean_rank, test_n = top1_and_rank(test_rows, w, mean, std, features)
    base_top1 = heuristic_top1(test_rows, features)
    metrics = {
        "positions_train": len(train_rows),
        "positions_test": len(test_rows),
        "pairs_train": int(len(pos)),
        "test_positions_scored": test_n,
        "top1": round(top1, 4),
        "mean_rank": round(mean_rank, 3),
        "heuristic_top1": round(base_top1, 4),
        "epochs": args.epochs,
        "lr": args.lr,
        "l2": args.l2,
        "negatives": args.negatives,
    }
    model = {
        "type": "pairwise_logistic",
        "features": features,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": w.tolist(),
        "metrics": metrics,
        "note": "Phase-1 prototype; not loaded by the engine.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
        fh.write("\n")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
