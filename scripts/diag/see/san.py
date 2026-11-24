"""Analyse a see-diag SEELOG (stats build, `go depth 13` over pos962).
usage: san.py LOG [LOG2 ...]   (several logs: tables per log side by side as a/b)
Records (sample rates in the stats build): pcf 1/4, pct all, pcv 1/4 of cuts,
pcn all, qui 1/8 (SEE < 0 only), cap 1/4."""
import sys
from collections import defaultdict

RATE = {"pcf": 4, "pct": 1, "pcv": 4, "pcn": 1, "qui": 8, "cap": 4}


def load(path):
    recs = defaultdict(list)
    for l in open(path):
        t = l.split()
        if t and t[0] in RATE:
            recs[t[0]].append([int(x) for x in t[1:]])
    return recs


def bk(x, w=50, lo=-600, hi=600):
    x = max(lo, min(hi, x))
    return (x // w) * w


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def probcut(R, tag):
    print("\n## ProbCut (%s)" % tag)
    tried = R["pct"]; filt = R["pcf"]
    # population: tried (all) + filtered (x4)
    for name, idx in (("piece SEE slack (s - thr)", 2), ("eval-scale SEE slack (sev - thr)", 3)):
        print("\n%s: bucket | tried n, succ%%, cost | filtered n(est), succ%%, cost" % name)
        rows = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
        for r in tried:
            d, nt, s, sev, thr, q, red, cost = r
            k = bk((s if idx == 2 else sev) - thr)
            rows[k][0] += 1; rows[k][1] += red >= 0; rows[k][2] += cost
        for r in filt:
            d, nt, s, sev, thr, bad, q, red, cost = r
            k = bk((s if idx == 2 else sev) - thr)
            rows[k][3] += 1; rows[k][4] += red >= 0; rows[k][5] += cost
        for k in sorted(rows):
            a = rows[k]
            print("  %+5d | %7d %5.1f%% %7.0f | %7d %5.1f%% %7.0f" % (
                k, a[0], pct(a[1], a[0]), a[2] / max(a[0], 1), a[3] * 4, pct(a[4], a[3]), a[5] / max(a[3], 1)))
    # filtered by class (bad capture stage)
    for bad in (0, 1):
        sub = [r for r in filt if r[5] == bad]
        print("filtered %s: n(est) %d, success %.2f%%, mean cost %.0f" % (
            "bad-stage" if bad else "good-stage", len(sub) * 4, pct(sum(r[7] >= 0 for r in sub), len(sub)),
            sum(r[8] for r in sub) / max(len(sub), 1)))
    # node outcomes after a failed ProbCut
    print("\nnodes whose ProbCut failed: depth | n, tried/node, attempt nodes, rest nodes, fail-high%")
    by = defaultdict(lambda: [0, 0, 0, 0, 0])
    for d, nt, sl, tr, na, rest, fh in R["pcn"]:
        a = by[min(d, 12)]; a[0] += 1; a[1] += tr; a[2] += na; a[3] += rest; a[4] += fh
    for d in sorted(by):
        a = by[d]
        print("  d%-3d %7d %4.2f %8.0f %9.0f %5.1f%%" % (d, a[0], a[1] / a[0], a[2] / a[0], a[3] / a[0], pct(a[4], a[0])))
    # cut verification
    print("\nProbCut cuts verified at full depth (ProbCut off at the node): depth | n, error% (full search < beta), cost")
    by = defaultdict(lambda: [0, 0, 0])
    for d, nt, s, sev, thr, sc, err, cost in R["pcv"]:
        a = by[(min(d, 12), nt)]; a[0] += 1; a[1] += err; a[2] += cost
    for k in sorted(by):
        a = by[k]
        print("  d%-3d %s %6d %5.2f%% %8.0f" % (k[0], "CUT" if k[1] == 1 else "ALL", a[0], pct(a[1], a[0]), a[2] / a[0]))
    by = defaultdict(lambda: [0, 0])
    for d, nt, s, sev, thr, sc, err, cost in R["pcv"]:
        a = by[bk(sev - thr, 100, -300, 900)]; a[0] += 1; a[1] += err
    print("  by eval-scale slack: " + "  ".join("%+d: %d %.2f%%" % (k, a[0], pct(a[1], a[0])) for k, a in sorted(by.items())))
    by = defaultdict(lambda: [0, 0])
    for d, nt, s, sev, thr, sc, err, cost in R["pcv"]:
        a = by[bk(sc, 50, 0, 400)]; a[0] += 1; a[1] += err
    print("  by reduced-search excess over probcut_beta: " + "  ".join("%+d: %d %.2f%%" % (k, a[0], pct(a[1], a[0])) for k, a in sorted(by.items())))
    tot_tried = sum(r[7] for r in tried); tot_cuts = sum(1 for r in tried if r[6] >= 0)
    print("tried %d (cuts %d, %.1f%%), attempt nodes %d" % (len(tried), tot_cuts, pct(tot_cuts, len(tried)), tot_tried))


def prune(R, key, tag, sidx, evidx, pruned_idx, err_idx, cost_idx, gc_idx):
    rate = RATE[key]
    recs = R[key]
    print("\n## %s (%s): n(est) %d" % (key, tag, len(recs) * rate))
    for name, xi in (("piece SEE", sidx), ("eval-scale SEE", evidx)):
        print("\n%s by depth: bucket: n(est) pruned%% err%% | ..." % name)
        by = defaultdict(lambda: [0, 0, 0, 0])
        for r in recs:
            d = r[0]
            k = (d, bk(r[xi], 50 if d <= 3 else 100, -1200, 0))
            a = by[k]; a[0] += 1; a[1] += r[pruned_idx]; a[2] += r[err_idx]; a[3] += r[cost_idx]
        for d in sorted({k[0] for k in by}):
            cells = ["%+d:%d/%.0f%%/%.2f%%" % (k[1], by[k][0] * rate, pct(by[k][1], by[k][0]), pct(by[k][2], by[k][0]))
                     for k in sorted(by) if k[0] == d]
            print("  d%d " % d + "  ".join(cells))
    print("\nby depth: n(est), pruned%, err% pruned, err% kept, err% pruned non-check / check, cost pruned (mean), cost kept")
    by = defaultdict(list)
    for r in recs:
        by[r[0]].append(r)
    for d in sorted(by):
        rs = by[d]
        p = [r for r in rs if r[pruned_idx]]; k = [r for r in rs if not r[pruned_idx]]
        pn = [r for r in p if not r[gc_idx]]; pc = [r for r in p if r[gc_idx]]
        print("  d%d %8d %5.1f%% %5.2f%% %5.2f%%  %5.2f%% / %5.2f%% (n %d)  %6.0f %6.0f" % (
            d, len(rs) * rate, pct(len(p), len(rs)), pct(sum(r[err_idx] for r in p), len(p)),
            pct(sum(r[err_idx] for r in k), len(k)), pct(sum(r[err_idx] for r in pn), len(pn)),
            pct(sum(r[err_idx] for r in pc), len(pc)), len(pc), sum(r[cost_idx] for r in p) / max(len(p), 1),
            sum(r[cost_idx] for r in k) / max(len(k), 1)))
    # separation: for each depth, ROC-style: errors captured at equal prune share by piece vs eval SEE
    print("\nscale test: prune the lowest-X% by each SEE (ties broken randomly-ish by order); errors among pruned")
    for d in sorted(by):
        rs = by[d]
        out = []
        for share in (0.2, 0.4, 0.6):
            n = int(len(rs) * share)
            e1 = sum(r[err_idx] for r in sorted(rs, key=lambda r: r[sidx])[:n])
            e2 = sum(r[err_idx] for r in sorted(rs, key=lambda r: r[evidx])[:n])
            out.append("%d%%: piece %.2f%% eval %.2f%%" % (share * 100, pct(e1, n), pct(e2, n)))
        print("  d%d  " % d + " | ".join(out))


def main():
    for path in sys.argv[1:]:
        R = load(path)
        probcut(R, path)
        # qui: d nt s sev pruned err cost idx gc
        prune(R, "qui", path, 2, 3, 4, 5, 6, 8)
        # cap: d nt s sev ch pruned err cost idx gc
        prune(R, "cap", path, 2, 3, 5, 6, 7, 9)


main()
