#!/usr/bin/env python3
"""Summaries for the tree-gap runs (docs/tree-gap-2026-11-26.md).

  analyze.py bb <dir>        black-box rustfish vs peer tables (rf-*/peer-* jsonl)
  analyze.py stats <file>    where rustfish's nodes go (treestats jsonl)
  analyze.py kn <dir>        knob counterfactuals: nodes vs base per depth
Python stdlib only.
"""
import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

VAL = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}


def final_board(line):
    """Piece placement after `fen ... moves ...` (enough to count material)."""
    t = line.split()
    if t[0] == "position":
        t = t[1:]
    assert t[0] == "fen"
    rows = t[1].split("/")
    sq = {}
    for r, row in enumerate(rows):
        f = 0
        for c in row:
            if c.isdigit():
                f += int(c)
            else:
                sq[(f, 7 - r)] = c
                f += 1
    moves = t[t.index("moves") + 1:] if "moves" in t else []
    for m in moves:
        a = (ord(m[0]) - 97, int(m[1]) - 1)
        b = (ord(m[2]) - 97, int(m[3]) - 1)
        p = sq.pop(a)
        if p in "pP" and a[0] != b[0] and b not in sq:  # en passant
            sq.pop((b[0], a[1]), None)
        if p in "kK" and abs(a[0] - b[0]) == 2:  # castling
            rf, rt = (7, 5) if b[0] > a[0] else (0, 3)
            sq[(rt, a[1])] = sq.pop((rf, a[1]))
        if len(m) == 5:
            p = m[4].upper() if p.isupper() else m[4]
        sq[b] = p
    return sq


def phase(line):
    sq = final_board(line)
    npm = sum(VAL[p.lower()] for p in sq.values() if p.lower() not in "pk")
    n = len(sq)
    if npm >= 50:
        return "opening/early-mg"
    if npm >= 30:
        return "middlegame"
    if npm >= 12:
        return "late-mg"
    return "endgame"


def load(path):
    return [json.loads(l) for l in open(path)]


def gmean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def nodes_at(rec, d):
    for it in rec["iters"]:
        if it["depth"] == d:
            return it["nodes"]
    return None


def field_at(rec, d, f):
    for it in rec["iters"]:
        if it["depth"] == d:
            return it.get(f)
    return None


def merge(dirpath, pattern):
    out = {}
    for p in sorted(glob.glob(os.path.join(dirpath, "**", pattern), recursive=True)):
        for r in load(p):
            out.setdefault(r["pos"], []).append(r)
    return out


def bb(d):
    rf = merge(d, "rf-d*-r1.jsonl")
    pe = merge(d, "peer-d*-r1.jsonl")
    rf2 = merge(d, "rf-d*-r2.jsonl")
    pe2 = merge(d, "peer-d*-r2.jsonl")
    keys = sorted(set(rf) & set(pe))
    print(f"positions: {len(keys)}")
    maxd = 20
    print("\n| depth | n | rf nodes (sum) | peer nodes (sum) | ratio of sums | geomean ratio | median ratio | rf EBF | peer EBF | rf time s | peer time s | rf seldepth | peer seldepth |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    prev = None
    for dd in range(4, maxd + 1):
        ks = [k for k in keys if nodes_at(rf[k][0], dd) and nodes_at(pe[k][0], dd)]
        # constant set: positions that completed this depth for both engines
        if len(ks) < 0.9 * len(keys):
            break
        a = sum(nodes_at(rf[k][0], dd) for k in ks)
        b = sum(nodes_at(pe[k][0], dd) for k in ks)
        rats = [nodes_at(rf[k][0], dd) / nodes_at(pe[k][0], dd) for k in ks]
        ta = sum(field_at(rf[k][0], dd, "time") or 0 for k in ks) / 1000
        tb = sum(field_at(pe[k][0], dd, "time") or 0 for k in ks) / 1000
        sa = st.mean(field_at(rf[k][0], dd, "seldepth") or 0 for k in ks)
        sb = st.mean(field_at(pe[k][0], dd, "seldepth") or 0 for k in ks)
        ebf_a = ebf_b = float("nan")
        ks2 = [k for k in ks if nodes_at(rf[k][0], dd - 1) and nodes_at(pe[k][0], dd - 1)]
        if ks2:
            ebf_a = gmean([nodes_at(rf[k][0], dd) / nodes_at(rf[k][0], dd - 1) for k in ks2])
            ebf_b = gmean([nodes_at(pe[k][0], dd) / nodes_at(pe[k][0], dd - 1) for k in ks2])
        print(f"| {dd} | {len(ks)} | {a:,} | {b:,} | {a/b:.2f} | {gmean(rats):.2f} | {st.median(rats):.2f} | {ebf_a:.2f} | {ebf_b:.2f} | {ta:.1f} | {tb:.1f} | {sa:.1f} | {sb:.1f} |")
    # determinism check (repeat 2)
    for name, x, y in (("rustfish", rf, rf2), ("peer", pe, pe2)):
        same = tot = 0
        for k in keys:
            if k in y:
                a, b = x[k][0]["iters"][-1]["nodes"], y[k][0]["iters"][-1]["nodes"]
                tot += 1
                same += a == b
        if tot:
            print(f"\n{name}: repeat run identical final nodes in {same}/{tot} positions")
    # by phase at depth 12 and 14
    for dd in (12, 14):
        byp = defaultdict(list)
        for k in keys:
            a, b = nodes_at(rf[k][0], dd), nodes_at(pe[k][0], dd)
            if a and b:
                byp[phase(k)].append((a, b))
        print(f"\nDepth {dd} by phase:\n\n| phase | n | ratio of sums | geomean | p10 | p90 |")
        print("|---|---:|---:|---:|---:|---:|")
        for ph in ("opening/early-mg", "middlegame", "late-mg", "endgame"):
            v = byp.get(ph, [])
            if not v:
                continue
            r = sorted(a / b for a, b in v)
            print(f"| {ph} | {len(v)} | {sum(a for a,_ in v)/sum(b for _,b in v):.2f} | {gmean(r):.2f} | {r[len(r)//10]:.2f} | {r[(9*len(r))//10]:.2f} |")
    # equal time
    rt = merge(d, "rf-mt1000.jsonl")
    pt = merge(d, "peer-mt1000.jsonl")
    ks = sorted(set(rt) & set(pt))
    if ks:
        def summ(rec):
            its = [i for i in rec["iters"] if "pv" in i]
            dmax = its[-1]["depth"] if its else 0
            sel = its[-1].get("seldepth", 0) if its else 0
            nodes = its[-1]["nodes"] if its else 0
            ch = 0
            last = None
            lastchg = 0
            for i in its:
                if i["depth"] < 5:
                    last = i["pv"]
                    continue
                if last is not None and i["pv"] != last:
                    ch += 1
                    lastchg = i["depth"]
                last = i["pv"]
            frac = lastchg / dmax if dmax else 0
            return dmax, sel, nodes, ch, frac, rec["best"]
        A = [summ(rt[k][0]) for k in ks]
        B = [summ(pt[k][0]) for k in ks]
        agree = sum(a[5] == b[5] for a, b in zip(A, B))
        print(f"\nEqual time (movetime 1000, {len(ks)} positions):\n")
        print("| engine | mean depth | mean seldepth | mean nodes | best-move changes from depth 5 (mean) | positions with >= 1 change | last change at depth fraction (mean, changers) |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for name, X in (("rustfish", A), ("peer", B)):
            ch = [x for x in X if x[3] > 0]
            print(f"| {name} | {st.mean(x[0] for x in X):.1f} | {st.mean(x[1] for x in X):.1f} | {st.mean(x[2] for x in X):,.0f} | {st.mean(x[3] for x in X):.2f} | {len(ch)} | {st.mean(x[4] for x in ch) if ch else 0:.2f} |")
        print(f"\nsame final move: {agree}/{len(ks)}")
        # rustfish depth at which it first matches peer's final move and keeps it
        # (equal-time depth semantics): compare rf depth-d best with peer's 1 s move
    # depth semantics: at which rustfish depth does its best move agree with
    # the peer's depth-d move as often as the peer's own depth-(d+2) does?
    print("\nAgreement of each engine's depth-d move with the peer's deepest move (same position):\n")
    print("| d | rustfish d vs peer deepest | peer d vs peer deepest | rustfish d vs rustfish deepest | peer d vs rustfish deepest |")
    print("|---|---:|---:|---:|---:|")
    for dd in range(6, 21, 2):
        c = [0, 0, 0, 0]
        n = 0
        for k in keys:
            ra, pa = rf[k][0]["iters"], pe[k][0]["iters"]
            if not ra or not pa:
                continue
            rd = {i["depth"]: i.get("pv") for i in ra}
            pd = {i["depth"]: i.get("pv") for i in pa}
            if dd not in rd or dd not in pd:
                continue
            pdeep, rdeep = pa[-1].get("pv"), ra[-1].get("pv")
            n += 1
            c[0] += rd[dd] == pdeep
            c[1] += pd[dd] == pdeep
            c[2] += rd[dd] == rdeep
            c[3] += pd[dd] == rdeep
        if n:
            print(f"| {dd} | {c[0]/n:.3f} | {c[1]/n:.3f} | {c[2]/n:.3f} | {c[3]/n:.3f} | ({n})")
    print("\n(final depths reached: rustfish", st.mean(rf[k][0]["iters"][-1]["depth"] for k in keys),
          "peer", st.mean(pe[k][0]["iters"][-1]["depth"] for k in keys), ")")


def sumstats(recs):
    tot = {}
    for r in recs:
        s = r.get("stats")
        if not s:
            continue
        for k, v in s.items():
            if isinstance(v, list):
                if k not in tot:
                    tot[k] = [0] * len(v)
                tot[k] = [a + b for a, b in zip(tot[k], v)]
            else:
                tot[k] = tot.get(k, 0) + v
    return tot


def stats(path):
    recs = []
    for p in sorted(glob.glob(path, recursive=True)):
        recs += load(p)
    s = sumstats(recs)
    total = sum(r["iters"][-1]["nodes"] for r in recs)
    main = sum(sum(s[f"nodes_t{t}"]) for t in range(3))
    qs = s["qs_nodes"]
    print(f"positions {len(recs)}, nodes {total:,} (negamax d>0 {main:,} = {main/total:.1%}, qsearch {qs:,} = {qs/total:.1%}, other negamax exits {total-main-qs:,})")
    print(f"qsearch: in check {s['qs_incheck']}, stand-pat cutoffs {s['qs_standpat_cut']/qs:.1%}, entries from negamax {s['qs_root']:,} (qs nodes per entry {qs/s['qs_root']:.2f})")
    print(f"aspiration: {s['asp_fails']} fails over {s['asp_iters']} root searches; nodes in failed root searches {sum(s['sub8']):,} = {sum(s['sub8'])/total:.1%}")
    D = len(s["nodes_t0"])
    print("\nNodes by depth remaining and node type (negamax, depth > 0):\n")
    print("| depth | PV | CUT | ALL | total | share | TT hit | TT cut | NMP tried/ok | LMR reduced | LMR re-search (rate) | re-search beats alpha | check ext | SE ext/tries | neg ext |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in range(1, D):
        n = [s[f"nodes_t{t}"][d] for t in range(3)]
        if sum(n) == 0:
            continue
        ttcut = sum(s[f"exit_t{t}_x0"][d] for t in range(3))
        print(f"| {d} | {n[0]:,} | {n[1]:,} | {n[2]:,} | {sum(n):,} | {sum(n)/main:.1%} | {s['tt_hit'][d]/max(1,s['tt_probe'][d]):.0%} | {ttcut/max(1,sum(n)):.1%} | "
              f"{s['nmp_tries'][d]}/{s['nmp_ok'][d]} | {s['lmr_red'][d]:,} | {s['lmr_re'][d]:,} ({s['lmr_re'][d]/max(1,s['lmr_red'][d]):.1%}) | {s['lmr_re_ok'][d]/max(1,s['lmr_re'][d]):.0%} | "
              f"{s['ext_check'][d]:,} | {s['ext_se'][d]}/{s['se_tries'][d]} | {s['ext_neg'][d]} |")
    X = ["TT", "razor", "RFP", "NMP", "ProbCut", "multicut", "fail-high", "fail-low", "exact", "other"]
    print("\nNode exits by type (all depths):\n")
    print("| type | nodes | " + " | ".join(X) + " |")
    print("|---" * (len(X) + 2) + "|")
    for t, nm in enumerate(("PV", "CUT", "ALL")):
        n = sum(s[f"nodes_t{t}"])
        print(f"| {nm} | {n:,} | " + " | ".join(f"{sum(s[f'exit_t{t}_x{x}'])/max(1,n):.1%}" for x in range(len(X))) + " |")
    # expected type vs actual
    print("\nMoves searched per move-loop node (mean) by type and depth bands:\n")
    print("| type | d1-3 | d4-6 | d7-9 | d10+ |")
    print("|---|---:|---:|---:|---:|")
    for t, nm in enumerate(("PV", "CUT", "ALL")):
        row = []
        for lo, hi in ((1, 3), (4, 6), (7, 9), (10, D - 1)):
            ms = sum(s[f"ms_t{t}"][lo:hi + 1])
            ln = sum(s[f"loop_t{t}"][lo:hi + 1])
            row.append(f"{ms/max(1,ln):.2f}")
        print(f"| {nm} | " + " | ".join(row) + " |")
    MI = ["0", "1", "2", "3-4", "5-7", "8-12", "13-20", "21+"]
    print("\nSubtree nodes by the parent's move index (share of all nodes, inclusive), all depths:\n")
    print("| type | " + " | ".join(MI) + " |")
    print("|---" * (len(MI) + 1) + "|")
    for t, nm in enumerate(("PV", "CUT", "ALL")):
        print(f"| {nm} | " + " | ".join(f"{sum(s[f'cmi_t{t}_m{m}'])/total:.1%}" for m in range(len(MI))) + " |")
    cuts = [sum(s[f"cut_m{m}"]) for m in range(len(MI))]
    print("\nFail-high move index distribution: " + ", ".join(f"{MI[m]}: {cuts[m]/sum(cuts):.1%}" for m in range(len(MI))))
    SUB = ["NMP", "ProbCut", "singular verify", "first move", "late reduced", "late unreduced", "LMR re-search", "PVS re-search", "aspiration fail", "checking moves"]
    print("\nInclusive subtree nodes by mechanism (share of all nodes; nested searches count in each):\n")
    print("| mechanism | nodes | share | count |")
    print("|---|---:|---:|---:|")
    for i, nm in enumerate(SUB):
        print(f"| {nm} | {sum(s[f'sub{i}']):,} | {sum(s[f'sub{i}'])/total:.1%} | {sum(s[f'subc{i}']):,} |")
    pl = s["plies"]
    cum = 0
    print("\nNodes by ply from root: " + ", ".join(f"{p}: {pl[p]/total:.1%}" for p in range(0, 40, 2) if pl[p]))


def kn(d):
    files = sorted(glob.glob(os.path.join(d, "**", "kn-d*.jsonl"), recursive=True))
    sets = defaultdict(dict)
    for p in files:
        name = os.path.basename(p)[:-6]
        depth, ks = name.split("-", 2)[1], name.split("-", 2)[2]
        for r in load(p):
            sets[(depth, ks)][r["pos"]] = r
    for depth in sorted({k[0] for k in sets}):
        base = sets.get((depth, "base"))
        if not base:
            continue
        print(f"\nKnob counterfactuals at go {depth} ({len(base)} positions; ratio = variant/base):\n")
        dmax = int(depth[1:])
        cols = [d for d in (8, 10, 12, dmax) if d <= dmax]
        print("| knobs | " + " | ".join(f"nodes d{c} (sum)" for c in cols) + " | geomean d" + str(dmax) + " | same move as base | qsearch share |")
        print("|---|" + "---:|" * (len(cols) + 3))
        for (dp, ks), recs in sorted(sets.items(), key=lambda x: x[0][1]):
            if dp != depth:
                continue
            ps = [p for p in base if p in recs]
            row = []
            for c in cols:
                a = sum(nodes_at(recs[p], c) or 0 for p in ps)
                b = sum(nodes_at(base[p], c) or 0 for p in ps)
                row.append(f"{a/b:.3f}" if b else "-")
            g = gmean([(nodes_at(recs[p], dmax) or 1) / (nodes_at(base[p], dmax) or 1) for p in ps])
            same = sum(recs[p]["best"] == base[p]["best"] for p in ps)
            sa = sumstats(recs[p] for p in ps)
            tot = sum(recs[p]["iters"][-1]["nodes"] for p in ps)
            print(f"| {ks} | " + " | ".join(row) + f" | {g:.3f} | {same}/{len(ps)} | {sa.get('qs_nodes',0)/max(1,tot):.1%} |")


if __name__ == "__main__":
    {"bb": bb, "stats": stats, "kn": kn}[sys.argv[1]](sys.argv[2])
