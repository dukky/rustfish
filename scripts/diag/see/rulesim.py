"""Replay candidate thresholds on a see-diag SEELOG: moves that change
decision, their verify error rate and the subtree nodes involved (direct
effect only; the tree's reaction is not modelled).
usage: rulesim.py LOG"""
import sys
from collections import defaultdict
qui, caps, pcf, pct, pcn = [], [], [], [], []
for l in open(sys.argv[1]):
    t = l.split()
    if not t:
        continue
    v = list(map(int, t[1:]))
    {"qui": qui, "cap": caps, "pcf": pcf, "pct": pct, "pcn": pcn}.get(t[0], []).append(v)
TOTAL = 155e6

def report(name, rows, rate, newly_pruned, newly_kept, err_i, cost_i):
    p = [r for r in rows if newly_pruned(r)]
    k = [r for r in rows if newly_kept(r)]
    def s(x):
        return "%7d moves, err %.2f%%, nodes %.2fM (%.2f%% of tree)" % (
            len(x) * rate, 100.0 * sum(r[err_i] for r in x) / len(x) if x else 0, sum(r[cost_i] for r in x) * rate / 1e6,
            100 * sum(r[cost_i] for r in x) * rate / TOTAL)
    print("%-34s newly pruned: %s\n%-34s newly kept:   %s" % (name, s(p), "", s(k)))

# qui: d nt s sev pruned err cost idx gc
def qthr(d, cap=0, m=20):
    t = -m * d * d
    return max(t, -cap) if cap else t
for qc in (300, 400, 500):
    report("quiet cap %d" % qc, qui, 8, lambda r, c=qc: not r[4] and r[2] < qthr(r[0], c), lambda r: False, 5, 6)
report("quiet cap 400, d<=6 only", qui, 8, lambda r: not r[4] and r[0] <= 6 and r[2] < qthr(r[0], 400), lambda r: False, 5, 6)
# cap: d nt s sev ch pruned err cost idx gc
def cthr(d, ch, m=100, cap=0, div=64):
    b = -m * d
    if cap:
        b = max(b, -cap)
    return b - (ch // div if div else 0)
for m in (80, 125):
    report("capture margin %d*d" % m, caps, 4, lambda r, m=m: not r[5] and r[2] < cthr(r[0], r[4], m), lambda r, m=m: r[5] and r[2] >= cthr(r[0], r[4], m), 6, 7)
report("capture cap 400", caps, 4, lambda r: not r[5] and r[2] < cthr(r[0], r[4], cap=400), lambda r: False, 6, 7)
report("capture: exchange sacs (-200..-150) kept at d1", caps, 4, lambda r: False, lambda r: r[5] and r[0] == 1 and -200 <= r[2] <= -150, 6, 7)
# ProbCut filter loosening: filtered good-stage moves with slack in [-X, 0)
rest = defaultdict(list)
for d, nt, sl, tr, na, rs, fh in pcn:
    rest[d].append(rs)
mrest = {d: sum(v) / len(v) for d, v in rest.items()}
for X in (50, 100, 150, 200, 300):
    rows = [r for r in pcf if not r[5] and -X <= r[2] - r[4] < 0]
    n = len(rows) * 4
    cost = sum(r[8] for r in rows) * 4
    succ = [r for r in rows if r[7] >= 0]
    save = sum(mrest.get(r[0], 0) for r in succ) * 4
    print("ProbCut filter -%d: %d extra tries, %.1f%% succeed, try cost %.2fM, est. saving %.2fM (net %.2f%% of tree)" % (
        X, n, 100.0 * len(succ) / max(len(rows), 1), cost / 1e6, save / 1e6, 100 * (cost - save) / TOTAL))
