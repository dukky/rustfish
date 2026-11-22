"""Analyse qrun stats JSON(s) (merged across seeds by mean)."""
import json, sys
from collections import defaultdict
SB = 81
CLS = ["TT(SEE>0)", "win cap", "equal cap", "queen promo", "underpromo", "evasion cap", "evasion quiet"]
C2 = ["SEE>0", "SEE=0", "promo"]
QG = ["qp0", "qp1-2", "qp3+"]
def load(paths):
    ds = [json.load(open(p)) for p in paths]
    m = defaultdict(list)
    for d in ds:
        for k, v in d.items():
            if k.startswith("_"): continue
            x = m[k]
            if len(x) < len(v): x.extend([0] * (len(v) - len(x)))
            for i, y in enumerate(v): x[i] += y / len(ds)
    return m, ds
def g(m, k, i):
    v = m.get(k, []); return v[i] if i < len(v) else 0
def slack(b): return b * 25 - 1000
def pct(a, b): return "%.2f%%" % (100 * a / b) if b else "-"
def main():
    m, ds = load(sys.argv[1:])
    tot = [sum(r["nodes"] for r in d["_rows"]) for d in ds]
    qn = sum(m["q_nodes"])
    print("total nodes per seed:", tot, " qsearch nodes %.0f = %s" % (qn, pct(qn, sum(tot) / len(tot))))
    print("\n## qsearch nodes by qply / in check / stand-pat raise")
    for q in range(16):
        n = g(m, "q_nodes", q)
        if n < 1: continue
        rs = [g(m, "q_ret", r * 16 + q) for r in range(7)]
        print("qp%-2d %10.0f (%5.1f%%)  check %5.1f%%  draw %4.1f TTcut %4.1f standpat-cut %4.1f nomoves %4.1f movecut %4.1f faillow %4.1f exact %4.1f  sp-raise %4.1f" % (
            q, n, 100 * n / qn, 100 * g(m, "q_check", q) / n, *[100 * r / n for r in rs], 100 * g(m, "q_spraise", q) / n))
    print("\n## moves by class x qply: searched / raise / cut / determinant / subtree nodes (share of qsearch)")
    for c in range(7):
        s = sum(g(m, "q_mv", (c * 16 + q) * 4) for q in range(16))
        if s < 1: continue
        r = sum(g(m, "q_mv", (c * 16 + q) * 4 + 1) for q in range(16))
        cu = sum(g(m, "q_mv", (c * 16 + q) * 4 + 2) for q in range(16))
        de = sum(g(m, "q_det", c * 16 + q) for q in range(16))
        sub = sum(g(m, "q_sub", c * 16 + q) for q in range(16))
        print("%-14s searched %10.0f raise %s cut %s det %s  subtree %s" % (CLS[c], s, pct(r, s), pct(cu, s), pct(de, s), pct(sub, qn)))
        line = []
        for q in range(8):
            sq = g(m, "q_mv", (c * 16 + q) * 4)
            if sq: line.append("qp%d n=%.0f r=%.1f%% c=%.1f%% d=%.1f%%" % (q, sq, 100 * g(m, "q_mv", (c * 16 + q) * 4 + 1) / sq, 100 * g(m, "q_mv", (c * 16 + q) * 4 + 2) / sq, 100 * g(m, "q_det", c * 16 + q) / sq))
        print("    " + " | ".join(line))
    for key, dk, title in [("q_sl", "q_detsl", "slack = static eval + victim(+promo) - alpha"), ("q_sls", "q_detsls", "slack = static eval + SEE - alpha")]:
        print("\n## %s: per bucket searched / raise / cut / det, cumulative det-rate for prune 'slack <= s'" % title)
        for gq in range(3):
            for c2 in range(3):
                base = (gq * 3 + c2) * SB
                tots = sum(g(m, key, (base + b) * 4) for b in range(SB))
                if tots < 1: continue
                print("-- %s %s (searched %.0f)" % (QG[gq], C2[c2], tots))
                cs = cr = cd = csub = 0
                for b in range(SB):
                    s = g(m, key, (base + b) * 4); r = g(m, key, (base + b) * 4 + 1); cu = g(m, key, (base + b) * 4 + 2); d = g(m, dk, base + b)
                    cs += s; cr += r; cd += d
                    if key == "q_sl": csub += g(m, "q_slsub", base + b)
                    sl = slack(b)
                    if s >= 1 and -600 <= sl <= 300:
                        print("   s=%+5d n=%9.0f raise %6s cut %6s det %6s | cum n=%9.0f (%5.1f%% of class) raise %6s det %6s%s" % (
                            sl, s, pct(r, s), pct(cu, s), pct(d, s), cs, 100 * cs / tots, pct(cr, cs), pct(cd, cs), ("  subtree %s of qs" % pct(csub, qn)) if key == "q_sl" else ""))
    print("\n## gives check (q_chk), SEE>0+equal, slack<=-100: n / det")
    for gc in range(2):
        for c2 in range(2):
            base = (gc * 3 + c2) * SB
            s = sum(g(m, "q_chk", (base + b) * 4) for b in range(SB) if slack(b) <= -100)
            d = sum(g(m, "q_detchk", base + b) for b in range(SB) if slack(b) <= -100)
            print("  check=%d %s: n=%.0f det %s" % (gc, C2[c2], s, pct(d, s)))
    print("\n## excess = score - (static eval + victim), distribution per group (cum share >= x)")
    for gq in range(3):
        for c2 in range(2):
            base = (gq * 3 + c2) * SB
            v = [g(m, "q_ex", base + b) for b in range(SB)]
            t = sum(v)
            if not t: continue
            out = []
            for thr in [-200, -100, -50, 0, 50, 100, 150, 200, 300, 500]:
                out.append("%+d:%.2f%%" % (thr, 100 * sum(v[b] for b in range(SB) if slack(b) >= thr) / t))
            print("  %s %s  P(excess>=x) " % (QG[gq], C2[c2]) + " ".join(out))
    print("\n## SEE bucket (x100) by qply group: n / raise / det")
    for gq in range(3):
        out = []
        for sb in range(11):
            s = g(m, "q_see", (gq * 11 + sb) * 4)
            if s >= 1: out.append("%d:n=%.0f r=%.1f%% d=%.1f%%" % (sb, s, 100 * g(m, "q_see", (gq * 11 + sb) * 4 + 1) / s, 100 * g(m, "q_detsee", gq * 11 + sb) / s))
        print("  %s " % QG[gq] + " | ".join(out))
    print("\n## move index x node slack (stand pat - alpha), det rate")
    for nb in range(SB):
        sl = slack(nb)
        if sl > 0 or sl < -800 or sl % 100: continue
        out = []
        for ix in range(8):
            s = sum(g(m, "q_idx", ((b) * 16 + ix) * 4) for b in range(nb, min(nb + 4, SB)))
            d = sum(g(m, "q_detidx", b * 16 + ix) for b in range(nb, min(nb + 4, SB)))
            if s >= 1: out.append("i%d n=%.0f d=%.1f%%" % (ix, s, 100 * d / s))
        print("  sp-alpha in [%+d,%+d): " % (sl, sl + 100) + " | ".join(out))
main()
