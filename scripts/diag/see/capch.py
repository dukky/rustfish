"""Capture SEE prune: error rate by depth x exact SEE value x capture-history sign/band.
usage: capch.py LOG"""
import sys
from collections import defaultdict
by = defaultdict(lambda: [0, 0, 0])
Z = [0, 0, 0]
for l in open(sys.argv[1]):
    if not l.startswith("cap "):
        continue
    d, nt, s, sev, ch, pr, err, cost, idx, gc = map(int, l.split()[1:])
    if s < -700:
        s = -700
    band = "ch<-4k" if ch < -4096 else "ch<0" if ch < 0 else "ch<4k" if ch < 4096 else "ch>=4k"
    a = by[(d, s, band)]; a[0] += 1; a[1] += err; a[2] += pr
for d in range(1, 7):
    vals = sorted({k[1] for k in list(by) if k[0] == d and sum(by.get((d, k[1], b), Z)[0] for b in ("ch<-4k", "ch<0", "ch<4k", "ch>=4k")) >= 400}, reverse=True)
    print("d%d  SEE: [n err%% pruned%%] for ch<-4k | ch<0 | ch<4k | ch>=4k" % d)
    for s in vals:
        cells = []
        for b in ("ch<-4k", "ch<0", "ch<4k", "ch>=4k"):
            a = by.get((d, s, b), Z)
            cells.append("%6d %5.2f%% %3.0f%%" % (a[0] * 4, 100.0 * a[1] / a[0] if a[0] else float("nan"), 100.0 * a[2] / a[0] if a[0] else float("nan")))
        print("  %5d  %s" % (s, " | ".join(cells)))
