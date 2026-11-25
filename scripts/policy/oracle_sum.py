#!/usr/bin/env python3
"""Summarise `oracletest` output (policy-order v0 gate A).

usage: oracle_sum.py <shard files...> [--base 0:0:100:100]
Per arm: total nodes vs the base arm (sum ratio, and the mean of per-position
log ratios with its standard error), first-move cut rate, moves searched per
fail high, oracle coverage, root-move agreement with the deep search. For the
base arm: which class the cutting move is in when it is not searched first.
"""
import math
import sys
from collections import defaultdict

CLASSES = ["tt", "goodcap", "killer/counter", "quiet h|ch>0", "plain quiet", "badcap", "oracle-placed", "-"]
DB = ["d1-3", "d4-7", "d8+"]
NT = ["pv", "cut", "all"]


def vec(s):
    return [int(x) for x in s.split(",")]


def mat(s):
    return [vec(r) for r in s.split(";")]


def cube(s):
    return [[vec(c) for c in r.split("/")] for r in s.split(";")]


def parse(paths):
    pos = []  # list of dict arm -> fields
    cur = None
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line.startswith("POS "):
                cur = {"_pos": line[4:]}
                pos.append(cur)
            elif line.startswith("DEEP ") and cur is not None:
                f = dict(kv.split("=", 1) for kv in line.split()[1:])
                cur["_deep"] = f
            elif line.startswith("ARM ") and cur is not None:
                parts = line.split()
                arm = parts[1]
                f = dict(kv.split("=", 1) for kv in parts[2:])
                d = {
                    "nodes": int(f["nodes"]), "bm": f["bm"], "score": int(f["score"]),
                    "loops": vec(f["loops"]), "avail": vec(f["avail"]), "istt": vec(f["istt"]),
                    "qnott": vec(f["qnott"]), "applied": vec(f["applied"]),
                    "fh": mat(f["fh"]), "fhfirst": mat(f["fhfirst"]), "fhmoves": mat(f["fhmoves"]),
                    "late": cube(f["late"]), "waste": cube(f["waste"]), "latehit": cube(f["latehit"]),
                    "fhwo": vec(f["fhwo"]), "fhco": vec(f["fhco"]),
                    "loopnodes": mat(f["loopnodes"]), "fhloopnodes": mat(f["fhloopnodes"]),
                }
                cur[arm] = d
    return [p for p in pos if "_deep" in p]


def main():
    args = sys.argv[1:]
    base = "0:0:100:100"
    if "--base" in args:
        i = args.index("--base")
        base = args[i + 1]
        del args[i:i + 2]
    pos = parse(args)
    arms = []
    for p in pos:
        for k in p:
            if not k.startswith("_") and k not in arms:
                arms.append(k)
    pos = [p for p in pos if all(a in p for a in arms)]
    n = len(pos)
    print(f"positions: {n}; deep nodes mean {sum(int(p['_deep']['nodes']) for p in pos) / max(n, 1):.0f}")
    bsum = sum(p[base]["nodes"] for p in pos)
    print(f"\n{'arm':<14}{'nodes':>8}{'logr':>8}{'se':>6}{'fmc%':>7}{'d1-3':>6}{'d4-7':>6}{'d8+':>6}{'mv/fh':>7}{'cov%':>6}{'appl%':>6}{'top1%':>6}{'rootagr%':>9}{'|ds|':>6}")
    for a in arms:
        tot = sum(p[a]["nodes"] for p in pos)
        lr = [math.log(p[a]["nodes"] / p[base]["nodes"]) for p in pos]
        m = sum(lr) / n
        se = math.sqrt(sum((x - m) ** 2 for x in lr) / (n - 1) / n)
        fh = [[sum(p[a]["fh"][d][t] for p in pos) for t in range(3)] for d in range(3)]
        ff = [[sum(p[a]["fhfirst"][d][t] for p in pos) for t in range(3)] for d in range(3)]
        fm = sum(p[a]["fhmoves"][d][t] for p in pos for d in range(3) for t in range(3))
        fht = sum(map(sum, fh))
        fft = sum(map(sum, ff))
        fmd = [100 * sum(ff[d]) / max(1, sum(fh[d])) for d in range(3)]
        loops = sum(sum(p[a]["loops"]) for p in pos)
        avail = sum(sum(p[a]["avail"]) for p in pos)
        appl = sum(sum(p[a]["applied"]) for p in pos)
        fhwo = sum(sum(p[a]["fhwo"]) for p in pos)
        fhco = sum(sum(p[a]["fhco"]) for p in pos)
        agr = sum(p[a]["bm"] == p["_deep"]["bm"] for p in pos)
        ds = sum(abs(p[a]["score"] - int(p["_deep"]["score"])) for p in pos) / n
        print(f"{a:<14}{tot / bsum:>8.4f}{m:>8.4f}{se:>6.3f}{100 * fft / fht:>7.2f}{fmd[0]:>6.2f}{fmd[1]:>6.2f}{fmd[2]:>6.2f}{fm / fht:>7.3f}"
              f"{100 * avail / loops:>6.1f}{100 * appl / loops:>6.1f}{100 * fhco / max(1, fhwo):>6.1f}{100 * agr / n:>9.1f}{ds:>6.1f}")

    b = base
    print(f"\nBase arm {b}: late cutters (cutter not first) by class; share of count / waste / oracle-named")
    for d in range(3):
        for t in range(3):
            late = [sum(p[b]["late"][d][t][c] for p in pos) for c in range(8)]
            waste = [sum(p[b]["waste"][d][t][c] for p in pos) for c in range(8)]
            hit = [sum(p[b]["latehit"][d][t][c] for p in pos) for c in range(8)]
            fh = sum(p[b]["fh"][d][t] for p in pos)
            ln = sum(p[b]["loopnodes"][d][t] for p in pos)
            tl, tw = sum(late), sum(waste)
            if fh == 0:
                continue
            print(f"  {DB[d]} {NT[t]}: fh {fh}, late {tl} ({100 * tl / fh:.1f}%), waste {100 * tw / max(1, ln):.1f}% of loop nodes")
            for c in range(7):
                if late[c]:
                    print(f"      {CLASSES[c]:<16} {100 * late[c] / tl:5.1f}% of late, {100 * waste[c] / max(1, tw):5.1f}% of waste, oracle named {100 * hit[c] / late[c]:5.1f}%")
    tree = sum(p[b]["nodes"] for p in pos)
    print("\n  waste nodes (pre-cutter) as % of total tree nodes, by depth bucket (overlapping across levels):")
    for d in range(3):
        w = sum(p[b]["waste"][d][t][c] for p in pos for t in range(3) for c in range(8))
        print(f"    {DB[d]}: {100 * w / tree:.1f}%")
    loops = [sum(p[b]["loops"][d] for p in pos) for d in range(3)]
    avail = [sum(p[b]["avail"][d] for p in pos) for d in range(3)]
    istt = [sum(p[b]["istt"][d] for p in pos) for d in range(3)]
    qn = [sum(p[b]["qnott"][d] for p in pos) for d in range(3)]
    print("\n  oracle coverage in the base tree (avail / =TT / quiet!=TT, % of move-loop nodes):")
    for d in range(3):
        print(f"    {DB[d]}: {100 * avail[d] / loops[d]:.1f} / {100 * istt[d] / loops[d]:.1f} / {100 * qn[d] / loops[d]:.1f}")


if __name__ == "__main__":
    main()
