#!/usr/bin/env python3
"""LMR recall test on `oracletest ... 21:0` dumps (stdlib only).

usage: recall.py <shard files...> [--mlp] [--weights out.txt]
Rows (`R` lines): every LMR-reduced quiet at nodes where the oracle best is
among them (positive nodes), plus all reduced quiets at a 5% sample of
nodes. Positions are split in half by a hash of the POS line: rankers are
fitted on the train half and scored on the test half.

Reported per ranker on the test half:
- top-1 / top-2 recall: share of positive nodes where the oracle best ranks
  first / in the first two among the node's reduced moves;
- per-move threshold curve: protect a move if score > t. Recall = protected
  positives / positives; protect rate = protected share of all reduced moves
  and precision = positives / protected (both from the 5% node sample,
  weight 20).
"""
import math
import random
import sys
import zlib

COLS = "node sampled label depth nt idx r h ch k0 k1 cm chk see kind fromatt toatt se ce improving ttb hasoracle".split()
C = {c: i for i, c in enumerate(COLS)}


def load(paths):
    nodes = {}  # (pos, node) -> list of rows
    pos = -1
    split = {}
    for p in paths:
        for line in open(p):
            if line.startswith("POS "):
                pos += 1
                split[pos] = zlib.crc32(line.encode()) % 2
            elif line.startswith("R "):
                v = [int(x) for x in line.split()[1:]]
                nodes.setdefault((pos, v[0]), []).append(v)
    return nodes, split


def clip(x, a):
    return max(-a, min(a, x))


def feats(v):
    f = [
        v[C["h"]] / 4000, v[C["ch"]] / 4000, v[C["k0"]], v[C["k1"]], v[C["cm"]], v[C["chk"]],
        clip(v[C["see"]], 900) / 300, float(v[C["see"]] < 0), v[C["fromatt"]], v[C["toatt"]],
        clip(v[C["ce"]], 600) / 200, float(v[C["ce"]] > -25),
        clip(v[C["se"]] if v[C["se"]] > -90000 else -600, 600) / 200,
        v[C["improving"]], math.log(1 + v[C["idx"]]), v[C["r"]] / 3, v[C["depth"]] / 10,
    ]
    k = v[C["kind"]]
    f += [float(k == i) for i in range(6)]
    f += [float(v[C["nt"]] == i) for i in range(3)]
    f += [float(v[C["ttb"]] == i) for i in range(4)]
    return f


NF = len(feats([0] * len(COLS)))


def linear_fit(train, epochs=6, lr=0.05, l2=1e-4):
    """Listwise softmax over each positive node's reduced moves."""
    w = [0.0] * NF
    data = [([feats(v) for v in rows], [v[C["label"]] for v in rows]) for rows in train]
    rnd = random.Random(1)
    for ep in range(epochs):
        rnd.shuffle(data)
        for X, y in data:
            s = [sum(a * b for a, b in zip(w, x)) for x in X]
            m = max(s)
            e = [math.exp(z - m) for z in s]
            z = sum(e)
            for x, ei, yi in zip(X, e, y):
                g = ei / z - yi
                if g:
                    for j in range(NF):
                        w[j] -= lr * (g * x[j] + l2 * w[j])
        lr *= 0.6
    return lambda v: sum(a * b for a, b in zip(w, feats(v))), w


def mlp_fit(train, hid=16, epochs=6, lr=0.02):
    rnd = random.Random(2)
    W1 = [[rnd.gauss(0, 0.3) for _ in range(NF)] for _ in range(hid)]
    b1 = [0.0] * hid
    W2 = [rnd.gauss(0, 0.3) for _ in range(hid)]
    data = [([feats(v) for v in rows], [v[C["label"]] for v in rows]) for rows in train]

    def fwd(x):
        h = [max(0.0, sum(a * b for a, b in zip(W1[k], x)) + b1[k]) for k in range(hid)]
        return sum(a * b for a, b in zip(W2, h)), h

    for ep in range(epochs):
        rnd.shuffle(data)
        for X, y in data:
            outs = [fwd(x) for x in X]
            s = [o[0] for o in outs]
            m = max(s)
            e = [math.exp(z - m) for z in s]
            z = sum(e)
            for x, (sc, h), ei, yi in zip(X, outs, e, y):
                g = ei / z - yi
                if abs(g) < 1e-4:
                    continue
                for k in range(hid):
                    if h[k] > 0:
                        gk = g * W2[k]
                        row = W1[k]
                        for j in range(NF):
                            row[j] -= lr * gk * x[j]
                        b1[k] -= lr * gk
                    W2[k] -= lr * g * h[k]
        lr *= 0.6
    return lambda v: fwd(feats(v))[0]


def evaluate(name, score, test_nodes, rnd):
    pos_nodes = [rows for rows in test_nodes if any(v[C["label"]] for v in rows)]
    t1 = t2 = 0
    for rows in pos_nodes:
        sc = [(score(v), rnd.random(), v[C["label"]]) for v in rows]
        sc.sort(reverse=True)
        t1 += sc[0][2]
        t2 += sc[0][2] or (len(sc) > 1 and sc[1][2])
    npos = len(pos_nodes)
    # threshold curve
    P = sorted((score(v) + 1e-9 * rnd.random() for rows in pos_nodes for v in rows if v[C["label"]]), reverse=True)
    S = sorted(((score(v) + 1e-9 * rnd.random(), v[C["label"]]) for rows in test_nodes if rows[0][C["sampled"]] for v in rows), reverse=True)
    nsamp = len(S)
    out = []
    for target in (0.5, 0.8):
        k = max(1, int(target * len(P)))
        t = P[k - 1]
        prot = sum(1 for s, _ in S if s >= t)
        rate = prot / max(1, nsamp)
        # precision: positives protected (exact) / protected moves (sample x 20)
        prec = k / max(1, 20 * prot)
        out.append((target, rate, prec, t))
    # top-1 at every node: protect rate = nodes / reduced moves (from the sample)
    samp_nodes = [rows for rows in test_nodes if rows[0][C["sampled"]]]
    rate1 = len(samp_nodes) / max(1, nsamp)
    print(f"{name:<26}{100 * t1 / npos:>7.1f}{100 * t2 / npos:>7.1f}"
          + "".join(f"{100 * r:>10.2f}{100 * p:>8.2f}" for _, r, p, _ in out) + f"{100 * rate1:>10.1f}"
          + "   t@50/80 " + " ".join(f"{t:.3f}" for *_, t in out))


def main():
    args = sys.argv[1:]
    use_mlp = "--mlp" in args
    wout = None
    if "--weights" in args:
        i = args.index("--weights")
        wout = args[i + 1]
        del args[i:i + 2]
    args = [a for a in args if a != "--mlp"]
    nodes, split = load(args)
    train = [rows for (p, _), rows in nodes.items() if split[p] == 0]
    test = [rows for (p, _), rows in nodes.items() if split[p] == 1]
    pos_train = [r for r in train if any(v[C["label"]] for v in r)]
    pos_test = [r for r in test if any(v[C["label"]] for v in r)]
    samp = [r for r in test if r[0][C["sampled"]]]
    nred = sum(len(r) for r in samp)
    npos_s = sum(v[C["label"]] for r in samp for v in r)
    print(f"positions {len(split)}; nodes train {len(train)} test {len(test)}; positive nodes train {len(pos_train)} test {len(pos_test)}")
    print(f"test sample: {len(samp)} nodes, {nred} reduced moves, oracle-best share {100 * npos_s / max(1, nred):.2f}%, "
          f"mean reduced moves per positive node {sum(len(r) for r in pos_test) / max(1, len(pos_test)):.2f}")
    print(f"\n{'ranker':<26}{'top1%':>7}{'top2%':>7}{'rate@50':>10}{'prec':>8}{'rate@80':>10}{'prec':>8}{'top1 rate':>10}")
    print("  (rate = share of all reduced moves protected; prec = share of protected that are the oracle best)")
    rnd = random.Random(7)
    g = lambda c: (lambda v: v[C[c]])
    evaluate("random", lambda v: 0, test, rnd)
    evaluate("picker order (-idx)", lambda v: -v[C["idx"]], test, rnd)
    evaluate("history h", g("h"), test, rnd)
    evaluate("cont history ch", g("ch"), test, rnd)
    evaluate("h + ch", lambda v: v[C["h"]] + v[C["ch"]], test, rnd)
    evaluate("child eval - alpha", g("ce"), test, rnd)
    evaluate("child eval > alpha-25", lambda v: float(v[C["ce"]] > -25), test, rnd)
    evaluate("SEE", g("see"), test, rnd)
    evaluate("gives check", g("chk"), test, rnd)
    evaluate("killer/counter", lambda v: v[C["k0"]] + v[C["k1"]] + v[C["cm"]], test, rnd)
    evaluate("main's base r (low first)", lambda v: -v[C["r"]], test, rnd)
    lin, w = linear_fit(pos_train)
    evaluate("linear (listwise, fit)", lin, test, rnd)
    evaluate("linear on TRAIN half", lin, train, rnd)
    if wout:
        with open(wout, "w") as f:
            f.write(" ".join(f"{x:.6f}" for x in w) + "\n")
    print("  linear weights:", " ".join(f"{x:+.2f}" for x in w))
    if use_mlp:
        mlp = mlp_fit(pos_train)
        evaluate("MLP 16 (listwise, fit)", mlp, test, rnd)
        evaluate("MLP 16 on TRAIN half", mlp, train, rnd)


if __name__ == "__main__":
    main()
