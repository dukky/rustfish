#!/usr/bin/env python3
"""Calibrate the suite: run every binary in a directory, compare each with the base,
and print a table next to the known SPRT Elo of each change.

    python3 scripts/diag/suite/calib.py --bins tmp/suite/bins --base main \
        --nodes 1000000 --ref-engine tmp/suite/rustfish-ref --out-dir tmp/suite/calib-1m

Binary names: `main` (the base), `main2` (a byte-identical copy: run-to-run noise),
`dummy*` (meaningless perturbations: the noise floor), `R-<change>` (a merged change
reverted, so expected Elo = -SPRT), `F-<change>` (a rejected change applied, expected = +SPRT).
"""
import argparse, json, os, statistics, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as suite_run  # noqa: E402

# Elo of the change itself at STC (10+0.1) from tests/sprt-log.jsonl / plan.md.
SPRT = {
    'pv-no-tt-cut': 24.8, 'tt-failow-nomove': 16.7, 'qsearch-fail-soft': 14.9,
    'prune-exempt-killers': 11.1, 'history-bonus-linear': 2.6, 'iir-depth3': -6.9,
}


def expected(name):
    if name.startswith('R-'):
        return -SPRT.get(name[2:], float('nan'))
    if name.startswith('F-'):
        return SPRT.get(name[2:], float('nan'))
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bins', required=True)
    ap.add_argument('--base', default='main')
    ap.add_argument('--suite', default=str(HERE / 'suite.jsonl'))
    ap.add_argument('--nodes', type=int, default=1_000_000)
    ap.add_argument('--parallel', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument('--ref-engine')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--only', nargs='*', help='subset of binary names')
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(os.listdir(a.bins))
    names = [a.base] + [n for n in names if n != a.base and (not a.only or n in a.only)]
    for n in names:
        res = out / f'{n}.jsonl'
        if res.exists():
            continue
        cmd = [sys.executable, str(HERE / 'run.py'), 'run', '--engine', str(Path(a.bins) / n),
               '--suite', a.suite, '--nodes', str(a.nodes), '--parallel', str(a.parallel),
               '--out', str(res)]
        if a.ref_engine:
            cmd += ['--ref-engine', a.ref_engine]
        print('running', n, flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    base = out / f'{a.base}.jsonl'
    rows = []
    for n in names[1:]:
        c = suite_run.compare(str(base), str(out / f'{n}.jsonl'), quiet=True)
        meta = json.loads(open(out / f'{n}.jsonl').readline())['meta']
        rows.append((n, expected(n), c, meta['seconds']))
    print(f'\n{len(rows)} binaries vs {a.base}, {a.nodes} nodes; deltas positive = better than {a.base}')
    hdr = f'{"binary":26} {"SPRT":>6} {"chg":>4} {"+/-":>7} {"d solved ladder":>22} {"d loss cp":>22} {"d mwpl":>22} {"d log nts":>22} {"sec":>5}'
    print(hdr)
    fmt = lambda t, s=1: f'{t[0]*s:+6.2f} [{t[1]*s:+6.2f},{t[2]*s:+6.2f}]'
    for n, e, c, sec in rows:
        print(f'{n:26} {e:+6.1f} {c["changed"]:4d} {"+%d/-%d" % (c["gained"], c["lost"]):>7} '
              f'{fmt(c["solved"], c["n"]):>22} {fmt(c["loss_cp"]):>22} {fmt(c["mwpl"]):>22} {fmt(c["log_nts"]):>22} {sec:5.0f}')
    dummies = [c for n, e, c, _ in rows if n.startswith('dummy')]
    if len(dummies) >= 2:
        print('\nnoise floor (SD over dummies): ' + ', '.join(
            f'{k} {statistics.stdev(d[k][0] * (d["n"] if k == "solved" else 1) for d in dummies):.2f}'
            for k in ('solved', 'loss_cp', 'mwpl', 'log_nts')))
    real = [(e, c) for n, e, c, _ in rows if n[:2] in ('R-', 'F-') and e == e]
    if real:
        for k in ('solved', 'loss_cp', 'mwpl', 'log_nts'):
            agree = sum((c[k][0] > 0) == (e > 0) for e, c in real)
            print(f'sign agreement with SPRT on {k}: {agree}/{len(real)}; Spearman rho '
                  f'{spearman([e for e, _ in real], [c[k][0] for _, c in real]):+.2f}')


def spearman(x, y):
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    if n < 2:
        return float('nan')
    return 1 - 6 * sum((a - b) ** 2 for a, b in zip(rx, ry)) / (n * (n * n - 1))


if __name__ == '__main__':
    main()
