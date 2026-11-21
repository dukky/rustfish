import json, statistics, sys
rows = []
for r in map(json.loads, open(sys.argv[1])):
    ds = sorted(int(d) for d in r['rows'])
    s = r['rows'][str(ds[-1])][0]
    if s is None or abs(s) > 800 or r['q']['incheck']:
        continue
    q = r['q']
    rows.append((q['qs'], s, q['threat'], q['hang'], q['static']))
print('n', len(rows))
for feat, idx in (('threat', 2), ('hang', 3)):
    print(f'-- {feat}')
    for lo, hi in ((-9999, 0), (1, 120), (121, 350), (351, 99999)):
        L = [x for x in rows if lo <= x[idx] <= hi]
        if L:
            print(f'  {feat} {lo}..{hi}: n {len(L)} mean(qs-ref) {statistics.mean(a - b for a, b, *_ in L):6.1f} '
                  f'mean feat {statistics.mean(x[idx] for x in L):6.0f}  |qs-ref| {statistics.mean(abs(a - b) for a, b, *_ in L):5.1f}')
    for k in (0, 1/32, 1/16, 1/8, 1/4, 1/2):
        mae = statistics.mean(abs(a - k * max(0, x[idx]) - b) for x in rows for a, b in [(x[0], x[1])])
        print(f'  k={k:.4f} MAE {mae:.2f}')
