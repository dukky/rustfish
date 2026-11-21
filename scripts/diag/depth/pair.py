"""Pairwise judge: for positions where the candidate's d14 move differs from
base d14's move A, compare both moves' depth-20 scores (same judge, both
moves). Prints mean delta (cand - base) over all positions, wins/losses >= 25 cp,
and a bootstrap 95% interval."""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
orc = {r["i"]: r for r in (json.loads(l) for l in open(os.path.join(HERE, "oracle-base.jsonl")))}
cache = {}
for l in open(os.path.join(HERE, "d20cache.jsonl")):
    x = json.loads(l); cache[(x["i"], x["m"])] = x["s"]


def s20(i, m):
    o = orc[i]
    if m == o["b"]:
        return o["sb20"]
    if m == o["a"] and "sa20" in o:
        return o["sa20"]
    return cache.get((i, m))


clip = lambda v: max(-300, min(300, v))
for name in sys.argv[1:]:
    rows = [json.loads(l) for l in open(os.path.join(HERE, "cand-%s.jsonl" % name))]
    d = []
    nodes = bn = 0
    for r in rows:
        o = orc[r["i"]]
        nodes += r["n"]; bn += o["n14"]
        if r["m"] == o["a"]:
            d.append(0); continue
        a, m = s20(r["i"], o["a"]), s20(r["i"], r["m"])
        if a is None or m is None or abs(a) > 20000 or abs(m) > 20000:
            d.append(0); continue
        d.append(clip(m - a))
    random.seed(1)
    bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(2000))
    print("%-12s n=%d nodes x%.3f  delta %+.2f cp [%+.2f, %+.2f]  differ %d  better>=25 %d  worse>=25 %d" % (
        name, len(d), nodes / bn, sum(d) / len(d), bs[50], bs[1950], sum(1 for r in rows if r["m"] != orc[r["i"]]["a"]),
        sum(1 for x in d if x >= 25), sum(1 for x in d if x <= -25)))
