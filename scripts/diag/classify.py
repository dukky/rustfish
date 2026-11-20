import json, sys
from collections import Counter
rs = [json.loads(l) for l in open(sys.argv[1])]
n = len(rs)
diff = [r for r in rs if r["a"] != r["b"]]
sig = [r for r in diff if "a14" in r]
print(f"positions {n}, d14 != d20 {len(diff)} ({100*len(diff)/n:.0f}%), significant (gap >= 25) {len(sig)}")
gaps = sorted(r["gap"] for r in sig)
if gaps:
    print("gap median", gaps[len(gaps)//2])
cause = Counter()
sole = Counter()
anyflip = Counter()
deps = Counter()
root_sub = Counter()
for r in sig:
    if r["b14"] > r["a14"]:
        cause["root"] += 1
        # did B appear as d14-iteration best earlier? last iteration change?
        bests = [x[2] for x in r["it14"]]
        root_sub["B was best at an earlier iteration" if r["b"] in bests else "B never best"] += 1
        continue
    fl = [k for k, (x, y, _) in r["toggles"].items() if y > x]
    for k in fl:
        anyflip[k] += 1
    if len(fl) == 1:
        sole[fl[0]] += 1
        cause["one mechanism"] += 1
    elif fl:
        cause["several mechanisms"] += 1
    else:
        d = [int(k) for k, (x, y) in r["depth"].items() if y > x]
        cause["depth (no toggle)" if d else "none up to d18"] += 1
        if d:
            deps[min(d)] += 1
print("cause:", dict(cause))
print("root sub:", dict(root_sub))
print("toggle flips (any):", anyflip.most_common())
print("toggle flips (sole):", sole.most_common())
print("depth at which B overtakes A (no toggle):", sorted(deps.items()))
