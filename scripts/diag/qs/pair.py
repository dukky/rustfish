import json, sys, math, random
O="/private/tmp/claude-501/-Users-andreasholley-dev-stockfish-challenge/4c85bf7b-729e-4ca9-80d9-7d20684e957e/scratchpad/tree/orc/cand-%s.jsonl"
def ld(n): return {r["i"]: r for r in (json.loads(l) for l in open(O % n))}
base = ld(sys.argv[1])
for n in sys.argv[2:]:
    c = ld(n); ids = sorted(set(c) & set(base))
    d = [min(c[i]["regret"],300) - min(base[i]["regret"],300) for i in ids]
    m = sum(d)/len(d); sd = math.sqrt(sum((x-m)**2 for x in d)/(len(d)-1)); se = sd/math.sqrt(len(d))
    same = sum(1 for i in ids if c[i]["m"] == base[i]["m"])
    print("%-12s n=%d mean diff %+.2f cp (se %.2f, regret capped 300)  same move %.1f%%  nodes x%.3f" % (n, len(ids), m, se, 100*same/len(ids), sum(c[i]["n"] for i in ids)/sum(base[i]["n"] for i in ids)))
