"""Depth-13 node totals per arm and seed against the no-op of the same seed.
usage: ncmp.py RUNDIR ARM...   (files RUNDIR/n-ARM-sSEED.json, no-op n-noopSEED.json)"""
import json, sys, os, statistics
d = sys.argv[1]
seeds = (100, 99, 101)
def tot(p):
    return sum(r["nodes"] for r in json.load(open(p))["_rows"])
base = {s: tot(os.path.join(d, "n-noop%d.json" % s)) for s in seeds}
m = statistics.mean(base.values())
print("no-op: " + " ".join("s%d %.2fM (%+.2f%%)" % (s, base[s] / 1e6, 100 * (base[s] / m - 1)) for s in seeds))
for a in sys.argv[2:]:
    r = []
    for s in seeds:
        p = os.path.join(d, "n-%s-s%d.json" % (a, s))
        if os.path.exists(p):
            r.append(100 * (tot(p) / base[s] - 1))
    print("%-10s %s  mean %+.2f%%" % (a, " ".join("%+.2f%%" % x for x in r), statistics.mean(r) if r else float("nan")))
