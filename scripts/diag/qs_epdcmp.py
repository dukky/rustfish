import re, sys, statistics
def load(p):
    d = {}
    for l in open(p):
        m = re.match(r'(\S+) (\w+)\s+leaves am at depth\s+(\S+)\s+nodes\s+(\S+)', l)
        if m:
            dep = int(m.group(3)) if m.group(3) != '-' else 26
            n = int(m.group(4)) if m.group(4) != '-' else 10_000_000
            d[m.group(1)] = (m.group(2), dep, n)
    return d
base = load(sys.argv[1])
print(f"{'run':40}{'solved':>7}{'blunder med d':>14}{'blunder gmean nodes':>20}{'better/worse(>=2ply)':>22}")
for p in sys.argv[1:]:
    r = load(p)
    bl = [k for k in r if r[k][0] == 'blunder' and base[k][1] > 2]
    solved = sum(1 for v in r.values() if v[1] < 26)
    med = statistics.median(r[k][1] for k in bl)
    gm = statistics.geometric_mean(r[k][2] for k in bl)
    better = sum(1 for k in bl if r[k][1] <= base[k][1] - 2)
    worse = sum(1 for k in bl if r[k][1] >= base[k][1] + 2)
    print(f"{p.rsplit('/',1)[-1]:40}{solved:>7}{med:>14}{gm:>20.0f}{better:>10}/{worse}")
