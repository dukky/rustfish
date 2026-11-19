#!/usr/bin/env python3
"""Search statistics: run a `--features stats` rustfish build over bench or
an EPD/FEN list at fixed depth, merge the counters and print markdown tables.

  cargo build --release --features stats --target-dir target-st
  python3 scripts/search_stats.py --engine target-st/release/rustfish bench
  python3 scripts/search_stats.py --engine target-st/release/rustfish \
      --positions mid.epd --depth 12 --parallel 4 --out tmp/stats.json

`--features stats-verify` also re-searches 1/16 of pruned nodes and moves;
its tables add pruning error rates. See docs/search-diagnosis-2026-11-19.md.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict

DB = 16
K = 16
NT = ["PV", "CUT", "ALL"]
GROUPS = [("1", [1]), ("2", [2]), ("3", [3]), ("4-5", [4, 5]), ("6-8", [6, 7, 8]), ("9+", list(range(9, DB)))]
STAGES = ["TT", "winning cap", "killer1", "counter", "killer2", "quiet hist>0", "quiet other", "bad cap", "equal cap"]


def run_engine(engine, cmds, dump):
    p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    out, _ = p.communicate("\n".join(cmds + ["statsdump " + dump, "quit"]) + "\n")
    return out


def run(args):
    tmp = tempfile.mkdtemp(prefix="sstats-")
    if args.mode == "bench":
        jobs = [["bench %d" % args.depth if args.depth else "bench"]]
    else:
        fens = [l.strip() for l in open(args.positions) if l.strip() and not l.startswith("#")]
        fens = fens[: args.limit] if args.limit else fens
        jobs = [[] for _ in range(args.parallel)]
        for i, f in enumerate(fens):
            if f.startswith(("fen ", "startpos")):
                pos = "position " + f
            else:
                pos = "position fen " + (" ".join(f.split()[:6]) if ";" not in f else f.split(";")[0].strip())
            jobs[i % args.parallel] += ["ucinewgame", pos, "go depth %d" % args.depth]
        jobs = [j for j in jobs if j]
    outs = [None] * len(jobs)

    def work(i):
        cmds = ["setoption name Hash value %d" % args.hash, "isready"] + jobs[i]
        outs[i] = run_engine(args.engine, cmds, os.path.join(tmp, "s%d.json" % i))

    ts = [threading.Thread(target=work, args=(i,)) for i in range(len(jobs))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    merged = defaultdict(list)
    for i in range(len(jobs)):
        d = json.load(open(os.path.join(tmp, "s%d.json" % i)))
        for k, v in d.items():
            m = merged[k]
            if len(m) < len(v):
                m.extend([0] * (len(v) - len(m)))
            for j, x in enumerate(v):
                m[j] += x
    # Iteration node counts for EBF (per search: nodes by completed depth).
    iters = []
    for out in outs:
        cur = {}
        for line in out.splitlines():
            t = line.split()
            if line.startswith("info depth") and "nodes" in t:
                if int(t[2]) == 1 and cur:
                    iters.append(cur)
                    cur = {}
                cur[int(t[2])] = int(t[t.index("nodes") + 1])
            elif line.startswith("bestmove"):
                if cur:
                    iters.append(cur)
                cur = {}
    merged["_iters"] = iters
    return dict(merged)


def g(st, key, i):
    v = st.get(key, [])
    return v[i] if i < len(v) else 0


def nd(nt, d):
    return nt * DB + d


def ndk(nt, d, k):
    return nd(nt, d) * K + k


def S(st, key, nts, ds, k=None):
    return sum(g(st, key, nd(n, d) if k is None else ndk(n, d, k)) for n in nts for d in ds)


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "-"


def f2(a, b):
    return "%.2f" % (a / b) if b else "-"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def report(st):
    out = []
    A = [0, 1, 2]
    alld = list(range(DB))
    nm = S(st, "nm_nodes", A, alld)
    q = g(st, "q_nodes", 0)
    out.append("### Tree shape\n")
    out.append("negamax nodes %d, qsearch nodes %d (qsearch share %s of all)\n" % (nm, q, pct(q, nm + q)))
    rows = []
    for name, ds in GROUPS:
        rows.append([name] + [S(st, "nm_nodes", [n], ds) for n in range(3)] + [pct(S(st, "nm_nodes", A, ds), nm)])
    out.append(table(rows, ["depth", "PV nodes", "CUT nodes", "ALL nodes", "share of nm"]))
    iters = st.get("_iters", [])
    if iters:
        rows = []
        maxd = max(max(i) for i in iters)
        for d in range(2, maxd + 1):
            r = [i[d] / i[d - 1] for i in iters if d in i and d - 1 in i and i[d - 1] > 0]
            if r:
                gm = math.exp(sum(math.log(x) for x in r) / len(r))
                rows.append([d, len(r), "%.2f" % gm])
        out.append("\nEBF (geometric mean of cumulative nodes(d)/nodes(d-1)):\n")
        out.append(table(rows, ["iter depth", "searches", "EBF"]))
    out.append("\n### Node outcomes (move-loop nodes)\n")
    rows = []
    for n in range(3):
        for name, ds in GROUPS:
            ln = S(st, "loop_nodes", [n], ds)
            if not ln:
                continue
            fh, fl, ex = S(st, "fh_nodes", [n], ds), S(st, "fl_nodes", [n], ds), S(st, "ex_nodes", [n], ds)
            rows.append([NT[n], name, ln, pct(fh, ln), pct(fl, ln), pct(ex, ln),
                         f2(S(st, "loop_legal", [n], ds), ln),
                         f2(S(st, "fh_searched", [n], ds), fh), f2(S(st, "fl_searched", [n], ds), fl),
                         pct(S(st, "loop_ttmove", [n], ds), ln)])
    out.append(table(rows, ["type", "depth", "nodes", "fail high", "fail low", "exact", "legal/node",
                            "searched @FH", "searched @FL", "TT move avail"]))
    out.append("\n### Ordering: index of the cutoff move (CUT+ALL+PV)\n")
    rows = []
    for name, ds in GROUPS + [("all", alld)]:
        fh = S(st, "fh_nodes", A, ds)
        if not fh:
            continue
        h = [S(st, "cut_idx", A, ds, k) for k in range(K)]
        rows.append([name, fh, pct(h[0], fh), pct(h[1], fh), pct(h[2], fh), pct(h[3], fh),
                     pct(sum(h[4:8]), fh), pct(sum(h[8:]), fh), f2(S(st, "fh_searched", A, ds), fh)])
    out.append(table(rows, ["depth", "fail highs", "idx0", "idx1", "idx2", "idx3", "idx4-7", "idx8+", "avg searched"]))
    rows = []
    for n in range(3):
        fh = S(st, "fh_nodes", [n], alld)
        rows.append([NT[n], fh, pct(S(st, "cut_idx", [n], alld, 0), fh)])
    out.append("\nFirst-move cutoff rate by node type: " + "; ".join("%s %s of %d" % (r[0], r[2], r[1]) for r in rows) + "\n")
    out.append("\n### Ordering: which move class cuts (share of cutoffs) and its cut rate when searched\n")
    rows = []
    fh = S(st, "fh_nodes", A, alld)
    for k, nm_ in enumerate(STAGES):
        c = S(st, "cut_stage", A, alld, k)
        srch = S(st, "searched_stage", A, alld, k)
        gen = S(st, "gen_stage", A, alld, k)
        c2 = S(st, "cut_stage", A, list(range(6, DB)), k)
        s2 = S(st, "searched_stage", A, list(range(6, DB)), k)
        rows.append([nm_, pct(c, fh), gen, srch, pct(c, srch), pct(c2, s2)])
    out.append(table(rows, ["class", "share of cutoffs", "visited", "searched", "cut rate if searched", "cut rate d6+"]))
    if "first_stage" in st:
        out.append("\n### CUT nodes: class of the first move searched, and how often it cuts\n")
        rows = []
        for name, ds in GROUPS + [("all", alld)]:
            tot = S(st, "loop_nodes", [1], ds)
            if not tot:
                continue
            r = [name]
            for k in range(len(STAGES)):
                fs = S(st, "first_stage", [1], ds, k)
                r.append("%s / %s" % (pct(fs, tot), pct(S(st, "first_cut", [1], ds, k), fs)) if fs else "-")
            rows.append(r)
        out.append(table(rows, ["depth"] + STAGES))
        out.append("\n(share of CUT nodes whose first move is this class / that move's cut rate)\n")
        rows = []
        lc = S(st, "fh_nodes", [1], alld) - S(st, "cut_idx", [1], alld, 0)
        for k, nm_ in enumerate(STAGES):
            rows.append([nm_, pct(S(st, "late_cut_first", [1], alld, k), lc), pct(S(st, "late_cut_stage", [1], alld, k), lc)])
        out.append("Late cutoffs at CUT nodes (cutoff at idx>=1, %d): class of the failed first move vs class of the cutting move\n" % lc)
        out.append(table(rows, ["class", "failed first move", "late cutting move"]))
    out.append("\n### Late move reductions (null-window moves after the first)\n")
    rows = []
    for name, ds in GROUPS:
        nq, nn = S(st, "nw_quiet", A, ds), S(st, "nw_noisy", A, ds)
        ap = S(st, "lmr_applied", A, ds)
        fhr = S(st, "lmr_fh", A, ds)
        rows.append([name, nq, nn, pct(ap, nq), f2(S(st, "lmr_rsum", A, ds), ap), pct(fhr, ap),
                     pct(S(st, "lmr_confirm", A, ds), fhr), S(st, "pvs_research", A, ds)])
    out.append(table(rows, ["depth", "nw quiets", "nw noisy", "LMR applied (of quiets)", "avg R",
                            "reduced fails high", "re-search confirms", "PVS re-searches"]))
    rows = []
    for n in range(3):
        ap = S(st, "lmr_applied", [n], alld)
        fhr = S(st, "lmr_fh", [n], alld)
        rows.append([NT[n], ap, f2(S(st, "lmr_rsum", [n], alld), ap), pct(fhr, ap), pct(S(st, "lmr_confirm", [n], alld), fhr)])
    out.append("\n" + table(rows, ["type", "reduced", "avg R", "fails high", "confirmed"]))
    out.append("\n### Pruning (fires; rate = fires / attempts or / candidate moves)\n")
    rows = []
    for name, ds in GROUPS:
        def tf(t, f):
            return "%d (%s)" % (S(st, f, A, ds), pct(S(st, f, A, ds), S(st, t, A, ds)))
        quiet_vis = S(st, "gen_stage", A, ds, 6)
        bad_vis = S(st, "gen_stage", A, ds, 7)
        rows.append([name, tf("razor_try", "razor_fire"), tf("rfp_try", "rfp_fire"), tf("nmp_try", "nmp_fire"),
                     tf("probcut_try", "probcut_fire"),
                     "%d (%s)" % (S(st, "lmp", A, ds), pct(S(st, "lmp", A, ds), quiet_vis)),
                     "%d (%s)" % (S(st, "fut_move", A, ds), pct(S(st, "fut_move", A, ds), quiet_vis)),
                     "%d (%s)" % (S(st, "see_quiet", A, ds), pct(S(st, "see_quiet", A, ds), quiet_vis)),
                     "%d (%s)" % (S(st, "see_cap", A, ds), pct(S(st, "see_cap", A, ds), bad_vis)),
                     S(st, "iir", A, ds)])
    out.append(table(rows, ["depth", "fut->qs", "RFP", "NMP", "ProbCut", "LMP (of plain quiets)",
                            "move futility", "SEE quiet", "SEE capture (of bad caps)", "IIR"]))
    se = [S(st, k, A, alld) for k in ("se_try", "se_ext", "se_multicut", "se_neg")]
    out.append("\nSingular: %d tries, extend %s, multicut %s, negative ext %s\n" % (
        se[0], pct(se[1], se[0]), pct(se[2], se[0]), pct(se[3], se[0])))
    hq = S(st, "gen_stage", A, alld, 5)
    pq = sum(S(st, "gen_stage", A, alld, k) for k in (2, 3, 4, 5))
    plain = S(st, "gen_stage", A, alld, 6)
    out.append("Quiet classes visited: priority (killer/counter/history>0) %d, plain %d (priority share %s; exempt from LMP/futility/SEE pruning)\n" % (
        pq, plain, pct(pq, pq + plain)))
    out.append("\n### TT\n")
    rows = []
    for n in range(3):
        nn = S(st, "nm_nodes", [n], alld)
        rows.append([NT[n], nn, pct(S(st, "tt_hit", [n], alld), nn), pct(S(st, "tt_hit_move", [n], alld), nn),
                     pct(S(st, "tt_hit_deep", [n], alld), nn), pct(S(st, "tt_cut", [n], alld), nn)])
    out.append(table(rows, ["type", "nodes", "hit", "hit w/ move", "hit depth>=d", "TT cutoff"]))
    out.append("\nqsearch TT hit rate %s\n" % pct(g(st, "q_tt_hit", 0), q))
    vk = [k for k in ("vrazor", "vrfp", "vnmp", "vprobcut", "vlmp", "vfut", "vseeq", "vseec") if k in st]
    if vk:
        out.append("\n### Pruning error rate (sampled re-searches; error = full search disagrees)\n")
        rows = []
        for k in vk:
            v = st[k]
            r = [k]
            for name, ds in GROUPS + [("all", alld)]:
                n_ = sum(v[2 * d] for d in ds if 2 * d < len(v))
                e = sum(v[2 * d + 1] for d in ds if 2 * d + 1 < len(v))
                r.append("%s (%d)" % (pct(e, n_), n_) if n_ else "-")
            rows.append(r)
        out.append(table(rows, ["rule"] + [n for n, _ in GROUPS] + ["all"]))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "positions", "report"])
    ap.add_argument("--engine")
    ap.add_argument("--positions")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--out")
    ap.add_argument("--json", help="report mode: stats JSON written by --out")
    args = ap.parse_args()
    if args.mode == "report":
        st = json.load(open(args.json))
    else:
        st = run(args)
        if args.out:
            json.dump(st, open(args.out, "w"))
    print(report(st))


if __name__ == "__main__":
    main()
