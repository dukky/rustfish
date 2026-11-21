import json, re, sys
from collections import Counter
rows = [json.loads(l) for l in open(sys.argv[1])]
print("misses localized:", len(rows))
kv = lambda s: dict(re.findall(r"(\w+)=(\S+)", s))
root_idx = Counter(); root_red = 0; stops = Counter(); exacta = 0
ext_opp = Counter(); ext_our = Counter(); our_r = Counter(); our_idx = Counter(); depthlost = []
ply1_tt = 0; lengths = Counter()
for r in rows:
    rt = r["root"]
    if rt:
        i = int(rt["idx"]); root_idx[min(i, 10)] += 1
        if i >= 4: root_red += 1
        if rt["s"] == rt["a"]: exacta += 1
    st = r["stop"].split(":")[0] if r["stop"].startswith("node") else r["stop"]
    stops[r["stop"].split(":")[1] if r["stop"].startswith("node:") else r["stop"]] += 1
    lengths[len(r["line"])] += 1
    for c in r["chain"]:
        d = kv(c)
        ply = int(d["ply"])
        if "ext" in d:
            # the move at node ply belongs to the side to move at ply: odd = opponent
            (ext_opp if ply % 2 == 1 else ext_our)[d["ext"]] += 1
            if ply % 2 == 0:
                our_r[d["r"]] += 1; our_idx[min(int(d["idx"]), 8)] += 1
        if "PRUNED" in c:
            (ext_our if ply % 2 == 0 else ext_opp)["PRUNED"] += 1
print("root idx of B:", sorted(root_idx.items()), " root-LMR reduced (idx>=4):", root_red, " B score == alpha:", exacta)
print("stop reasons:", stops.most_common())
print("line length:", sorted(lengths.items()))
print("opp line moves ext:", ext_opp, " our line moves ext:", ext_our)
print("our line moves r:", sorted(our_r.items()), " idx:", sorted(our_idx.items()))
# net depth: ply + d at the deepest node vs 14
nets = []
for r in rows:
    if r["chain"]:
        d = kv(r["chain"][-1]); nets.append(int(d["ply"]) + int(d["d"]) - 14)
print("net depth at deepest line node (ply+d-14):", sorted(Counter(nets).items()))
