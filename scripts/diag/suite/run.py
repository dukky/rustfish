#!/usr/bin/env python3
"""Run the critical-position suite on one binary, or compare two result files.

    # one binary: fixed nodes, fresh process per position, 1 thread, Hash 64
    python3 scripts/diag/suite/run.py run --engine ./target/release/rustfish \
        --nodes 1000000 --ref-engine tmp/suite/rustfish-ref --out tmp/suite/cand.jsonl
    # paired comparison (bootstrap CIs over positions)
    python3 scripts/diag/suite/run.py compare tmp/suite/base.jsonl tmp/suite/cand.jsonl

Per position one search at the top budget N. From its iteration trace the move the
engine would play at N/2, N/4, ... (`--ladder`, default 4 points) is also read off,
which averages out some of the chaotic move flipping at a single budget for free.
Metrics (labels in suite.jsonl, see README.md):
  solved   chosen move == reference best move (at the top budget, and averaged over the ladder)
  loss     reference score(best) - reference score(chosen), cp, clamped to [0, 500], ladder mean;
           also as win-probability loss (mwpl, milli; logistic with a 400 cp scale)
  nts      nodes from which the reference move stays best to the end (unsolved: 2x budget)
A chosen move with no reference score (not in the labels or suite-extra.jsonl) is
searched once with --ref-engine (the label binary) and cached in suite-extra.jsonl;
without --ref-engine it is scored with the position's `gap` (the smallest known loss).
"""
import argparse, json, math, os, random, statistics, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from uci import Engine  # noqa: E402

LOSS_CAP = 500


def wp(cp):
    return 1.0 / (1.0 + 10 ** (-cp / 400.0))


def load_suite(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def load_extra(path):
    extra = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            extra[(r['fen'], r['move'])] = r['score']
    return extra


def solve_point(rows, best):
    ds = sorted(rows)
    for d in ds:
        if all(rows[e][0] == best for e in ds if e >= d):
            return d, rows[d][2]
    return None, None


def ladder_points(a):
    if a.depth:
        return [['depth', a.depth - k] for k in range(a.ladder)]
    return [['nodes', a.nodes >> k] for k in range(a.ladder)]


def move_at(rows, final, kind, v, top):
    """Move the engine would play at ladder point (kind, v): the final bestmove at the top
    point, else the PV move of the deepest completed iteration within the budget (a
    fixed-node search is deterministic, so a truncated trace equals a smaller search up
    to its last completed iteration; a partial last iteration is ignored)."""
    if [kind, v] == top:
        return final
    ds = [d for d in sorted(rows) if (d <= v if kind == 'depth' else rows[d][2] <= v)]
    return rows[ds[-1]][0] if ds else None


def cmd_run(a):
    suite = load_suite(a.suite)
    if a.limit:
        suite = suite[:a.limit]
    extra_path = Path(a.suite).with_name('suite-extra.jsonl')
    extra = load_extra(extra_path)
    lock = threading.Lock()
    opts = [tuple(o.split('=', 1)) for o in a.option]
    eng = Engine(a.engine, hash_mb=a.hash, options=opts)
    ref = Engine(a.ref_engine, hash_mb=a.hash) if a.ref_engine else None
    pts = ladder_points(a)
    go = f'{pts[0][0]} {pts[0][1]}'

    def ref_score(p, m):
        if m == p['best']:
            return p['score']
        if m in p['alts']:
            return p['alts'][m]
        with lock:
            v = extra.get((p['fen'], m))
        if v is None and ref is not None and m:
            c = ref.search(p['fen'], f'nodes {a.child_nodes}', moves=[m])
            if c['rows']:
                v = -c['rows'][max(c['rows'])][1]
                with lock:
                    extra[(p['fen'], m)] = v
                    with open(extra_path, 'a') as f:
                        f.write(json.dumps(dict(fen=p['fen'], move=m, score=v, nodes=a.child_nodes)) + '\n')
        return v

    def one(p):
        r = eng.search(p['fen'], go)
        rows, final = r['rows'], r['bestmove']
        moves, losses, exact = [], [], 0
        for kind, v in pts:
            m = move_at(rows, final, kind, v, pts[0])
            sc = ref_score(p, m) if m else None
            exact += sc is not None
            loss = 0 if m == p['best'] else (
                max(0, min(LOSS_CAP, p['score'] - sc)) if sc is not None else min(LOSS_CAP, p['gap']))
            moves.append(m)
            losses.append(loss)
        sd, sn = solve_point(rows, p['best']) if final == p['best'] else (None, None)
        return dict(id=p['id'], bucket=p['bucket'], score=p['score'], moves=moves, losses=losses,
                    solved=[m == p['best'] for m in moves], solve_depth=sd, solve_nodes=sn,
                    nodes=max((v[2] for v in rows.values()), default=0), depth=max(rows, default=0),
                    exact=exact)

    t0 = time.time()
    with ThreadPoolExecutor(a.parallel) as ex:
        res = list(ex.map(one, suite))
    dt = time.time() - t0
    budget = None if a.depth else a.nodes
    meta = dict(engine=a.engine, go=go, ladder=pts, options=a.option, positions=len(res),
                seconds=round(dt, 1), budget=budget)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, 'w') as f:
            f.write(json.dumps(dict(meta=meta)) + '\n')
            for r in res:
                f.write(json.dumps(r) + '\n')
    summarize(res, meta, a.boot)
    print(f'time {dt:.0f}s ({a.parallel} parallel)')


def nts(r, budget):
    if r['solved'][0] and r['solve_nodes']:
        return r['solve_nodes']
    return 2 * (budget or max(r['nodes'], 1))


def lad_wpl(r):
    return statistics.fmean(wp(r['score']) - wp(r['score'] - l) for l in r['losses'])


def boot_ci(vals, n, f=statistics.fmean, seed=1):
    rng = random.Random(seed)
    k = len(vals)
    s = sorted(f([vals[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return s[int(0.025 * n)], s[int(0.975 * n) - 1]


def summarize(res, meta, boot):
    budget = meta.get('budget')
    print(f'positions {len(res)}  ladder {meta["ladder"]}  '
          f'(scored moves {sum(r["exact"] for r in res)}/{len(res) * len(meta["ladder"])}, rest use gap)')
    cols = (('solve rate @top', [1.0 * r['solved'][0] for r in res], '{:.3f}'),
            ('solve rate ladder', [statistics.fmean(r['solved']) for r in res], '{:.3f}'),
            ('loss cp ladder', [statistics.fmean(r['losses']) for r in res], '{:.1f}'),
            ('mwpl ladder', [1000 * lad_wpl(r) for r in res], '{:.1f}'))
    for name, v, fmt in cols:
        lo, hi = boot_ci(v, boot)
        print(f'  {name:18} ' + fmt.format(statistics.fmean(v)) + '  [' + fmt.format(lo) + ', '
              + fmt.format(hi) + ']')
    lnts = [math.log(nts(r, budget)) for r in res]
    lo, hi = boot_ci(lnts, boot)
    print(f'  {"geo nts":18} {math.exp(statistics.fmean(lnts)):.0f}  [{math.exp(lo):.0f}, {math.exp(hi):.0f}]'
          '  (nodes-to-solve at the top budget, unsolved = 2x budget)')
    by = {}
    for r in res:
        by.setdefault(r['bucket'], []).append(r)
    print('  @top by bucket: ' + ', '.join(
        f'{k} {sum(x["solved"][0] for x in v)}/{len(v)}' for k, v in sorted(by.items())))


def read_res(path):
    lines = [json.loads(l) for l in open(path)]
    return lines[0]['meta'], {r['id']: r for r in lines[1:]}


METRICS = ('top', 'solved', 'loss_cp', 'mwpl', 'log_nts')


def compare(base_path, cand_path, boot=4000, quiet=False):
    mb, b = read_res(base_path)
    mc, c = read_res(cand_path)
    ids = sorted(set(b) & set(c))
    budget = mb.get('budget')
    sol = lambda r: statistics.fmean(r['solved'])
    los = lambda r: statistics.fmean(r['losses'])
    d = {
        'top': [c[i]['solved'][0] - b[i]['solved'][0] for i in ids],
        'solved': [sol(c[i]) - sol(b[i]) for i in ids],
        'loss_cp': [los(b[i]) - los(c[i]) for i in ids],                   # positive = candidate better
        'mwpl': [1000 * (lad_wpl(b[i]) - lad_wpl(c[i])) for i in ids],     # milli-wp, positive = better
        'log_nts': [math.log(nts(b[i], budget)) - math.log(nts(c[i], budget)) for i in ids],
    }
    changed = sum(b[i]['moves'][0] != c[i]['moves'][0] for i in ids)
    gain = sum(c[i]['solved'][0] and not b[i]['solved'][0] for i in ids)
    lost = sum(b[i]['solved'][0] and not c[i]['solved'][0] for i in ids)
    out = dict(n=len(ids), changed=changed, gained=gain, lost=lost)
    for k in METRICS:
        lo, hi = boot_ci(d[k], boot)
        out[k] = (statistics.fmean(d[k]), lo, hi)
    if not quiet:
        print(f'{cand_path} vs {base_path}: {len(ids)} positions, top-budget move changed in {changed}, '
              f'+{gain} / -{lost} solved')
        for k, label, sc in (('top', 'd solved @top', len(ids)), ('solved', 'd solved ladder', len(ids)),
                             ('loss_cp', 'd loss cp', 1), ('mwpl', 'd mwpl', 1), ('log_nts', 'd log nts', 1)):
            m, lo, hi = out[k]
            print(f'  {label:16} {m * sc:+8.2f}  [{lo * sc:+.2f}, {hi * sc:+.2f}]')
        print('  (positive = candidate better; solved in positions, loss in cp, mwpl in milli win-prob,'
              ' log nts: candidate needs exp(-x) times the nodes)')
    return out


def cmd_compare(a):
    compare(a.base, a.cand, a.boot)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--engine', required=True)
    r.add_argument('--suite', default=str(HERE / 'suite.jsonl'))
    r.add_argument('--nodes', type=int, default=1_000_000)
    r.add_argument('--depth', type=int, default=0, help='fixed depth instead of nodes')
    r.add_argument('--ladder', type=int, default=4,
                   help='also score the move at budget/2, /4, ... (or depth-1, -2, ...) from the same trace')
    r.add_argument('--option', action='append', default=[], help='UCI option Name=Value')
    r.add_argument('--hash', type=int, default=64)
    r.add_argument('--parallel', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    r.add_argument('--ref-engine', help='reference (label) binary for scoring moves with no label')
    r.add_argument('--child-nodes', type=int, default=3_000_000)
    r.add_argument('--limit', type=int, default=0)
    r.add_argument('--boot', type=int, default=2000)
    r.add_argument('--out')
    c = sp.add_parser('compare')
    c.add_argument('base')
    c.add_argument('cand')
    c.add_argument('--boot', type=int, default=4000)
    a = ap.parse_args()
    {'run': cmd_run, 'compare': cmd_compare}[a.cmd](a)


if __name__ == '__main__':
    main()
