"""Compare runs to the reference.

python3 ana.py REF.jsonl RUN.jsonl [RUN2.jsonl ...]  (RUN may equal REF)
"""
import json, sys, statistics

def load(p):
    return {r['id']: r for r in map(json.loads, open(p))}

ref = load(sys.argv[1])
REF = {}
for k, r in ref.items():
    ds = sorted(int(d) for d in r['rows'])
    if not ds:
        continue
    s, mv, _ = r['rows'][str(ds[-1])]
    if s is None or abs(s) > 800:
        continue
    REF[k] = (s, r['best'], ds[-1])
print(f"reference positions |score|<=800: {len(REF)}; median ref depth "
      f"{statistics.median(v[2] for v in REF.values())}")

def bucket(r):
    q = r['q']
    if q['incheck']:
        return 'check'
    t = q['threat']
    return 'thr0' if t <= 0 else ('thr100' if t <= 120 else 'thr200+')

DEPTHS = [1, 2, 4, 6, 8, 10]
for path in sys.argv[2:]:
    run = load(path)
    print(f"\n== {path}")
    groups = {}
    for k, (rs, rbm, _) in REF.items():
        r = run.get(k)
        if not r:
            continue
        for g in ('all', r['cls'], bucket(r)):
            groups.setdefault(g, []).append((r, rs, rbm))
    hdr = f"{'group':9}{'n':>6}{'st-ref':>8}{'qs-ref':>8}{'|st|':>6}{'|qs|':>6}" + ''.join(f"{'|d'+str(d)+'|':>7}" for d in DEPTHS) + ''.join(f"{'mv'+str(d):>6}" for d in (4, 8, 10)) + f"{'qsN':>6}{'nodes10':>9}"
    print(hdr)
    for g in ['all', 'R', 'F', 'thr0', 'thr100', 'thr200+', 'check']:
        L = groups.get(g, [])
        if not L:
            continue
        def mae(f):
            v = [abs(f(r) - rs) for r, rs, _ in L if f(r) is not None]
            return statistics.mean(v) if v else float('nan')
        def sgn(f):
            return statistics.mean(f(r) - rs for r, rs, _ in L)
        def dsc(r, d):
            x = r['rows'].get(str(d))
            return None if x is None or x[0] is None else max(-1500, min(1500, x[0]))
        def agree(d):
            v = [(r['rows'].get(str(d)) or [None, None])[1] == rbm for r, _, rbm in L]
            return 100 * sum(v) / len(v)
        line = f"{g:9}{len(L):>6}{sgn(lambda r: r['q']['static']):>8.1f}{sgn(lambda r: max(-1500, min(1500, r['q']['qs']))):>8.1f}"
        line += f"{mae(lambda r: r['q']['static']):>6.0f}{mae(lambda r: max(-1500, min(1500, r['q']['qs']))):>6.0f}"
        line += ''.join(f"{mae(lambda r, d=d: dsc(r, d)):>7.1f}" for d in DEPTHS)
        line += ''.join(f"{agree(d):>6.1f}" for d in (4, 8, 10))
        line += f"{statistics.mean(r['q']['nodes'] for r, _, _ in L):>6.1f}"
        n10 = [r['rows'].get('10', [0, 0, 0])[2] for r, _, _ in L]
        line += f"{statistics.mean(n10):>9.0f}"
        print(line)
