"""Fixed-node runs vs reference: best-move agreement and final-score error.
python3 ana2.py REF RUN...
"""
import json, sys, statistics

def load(p):
    return {r['id']: r for r in map(json.loads, open(p))}

def final(r):
    ds = sorted(int(d) for d in r['rows'])
    return r['rows'][str(ds[-1])][0], ds[-1]

ref = load(sys.argv[1])
REF = {k: (final(r)[0], r['best']) for k, r in ref.items() if r['rows'] and final(r)[0] is not None and abs(final(r)[0]) <= 800}
base = None
print(f"{'run':24}{'agree%':>8}{'R':>7}{'thr200+':>8}{'|err|':>7}{'R|err|':>7}{'depth':>7}{'=base%':>8}")
for p in sys.argv[2:]:
    run = load(p)
    if base is None:
        base = run
    ag, agR, agT, err, errR, dep, same = [], [], [], [], [], [], []
    for k, (rs, rb) in REF.items():
        r = run[k]
        s, d = final(r)
        a = r['best'] == rb
        ag.append(a); err.append(abs(max(-1500, min(1500, s)) - rs)); dep.append(d)
        same.append(r['best'] == base[k]['best'])
        if r['cls'] == 'R':
            agR.append(a); errR.append(err[-1])
        if ref[k]['q']['threat'] > 120:
            agT.append(a)
    pct = lambda v: 100 * sum(v) / len(v)
    print(f"{p:24}{pct(ag):>8.2f}{pct(agR):>7.2f}{pct(agT):>8.2f}{statistics.mean(err):>7.2f}{statistics.mean(errR):>7.2f}{statistics.mean(dep):>7.2f}{pct(same):>8.1f}")
