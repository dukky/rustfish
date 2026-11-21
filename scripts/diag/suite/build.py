#!/usr/bin/env python3
"""Build the critical-position suite.

  extract  PGNs -> candidates.jsonl   (positions where an engine's reported eval swung)
  label    candidates -> labels.jsonl (deep rustfish searches; resumable, parallel)
  select   labels -> suite.jsonl      (keep positions where the best move clearly matters)

PGN evals (fastchess comments, either engine) are only used to *propose* candidates;
every label comes from our own engine (see README.md).
"""
import argparse, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chesslite import Board  # noqa: E402
from uci import Engine, score_cp  # noqa: E402

TAG = re.compile(r'\[(\w+) "([^"]*)"\]')
TOKEN = re.compile(r'\{[^}]*\}|\d+\.(?:\.\.)?|[^\s{}]+')
EVAL = re.compile(r'^\{\s*([+-]?(?:\d+\.\d+|M\d+))/(\d+)')


def parse_eval(tok):
    m = EVAL.match(tok)
    if not m:
        return None
    v = m.group(1)
    if 'M' in v:
        return 3000 if not v.startswith('-') else -3000
    return round(float(v) * 100)


def games(path):
    text = Path(path).read_text(errors='replace')
    for chunk in re.split(r'\n(?=\[Event )', text):
        tags = dict(TAG.findall(chunk))
        body = '\n'.join(l for l in chunk.splitlines() if not l.startswith('['))
        if 'FEN' not in tags and 'White' not in tags:
            continue
        yield tags, body


def extract(a):
    rng = random.Random(20261121)
    seen, out = set(), []
    ngames = 0
    for pgn in a.pgn:
        src = Path(pgn).parent.parent.name + '/' + Path(pgn).name
        for gi, (tags, body) in enumerate(games(pgn)):
            ngames += 1
            b = Board(tags.get('FEN', 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'))
            moves = []  # (fen, uci, mover_name, eval_mover_pov or None)
            for tok in TOKEN.findall(body):
                if tok.startswith('{'):
                    if moves and moves[-1][3] is None:
                        moves[-1][3] = parse_eval(tok)
                    continue
                if tok[0].isdigit() and tok.endswith('.') or tok in ('1-0', '0-1', '1/2-1/2', '*'):
                    continue
                try:
                    m = b.parse_san(tok)
                except Exception:
                    break
                mover = tags.get('White') if b.white else tags.get('Black')
                moves.append([b.fen(), Board.uci(m), mover, None, b.material_phase()])
                b.push(m)
            pool_random, game_swings = [], []
            for i, (fen, mv, mover, e, ph) in enumerate(moves):
                if e is None or abs(e) > a.max_eval:
                    continue
                key = ' '.join(fen.split()[:4])
                if key in seen:
                    continue
                opp_before = moves[i - 1][3] if i >= 1 else None   # opponent POV
                opp_after = moves[i + 1][3] if i + 1 < len(moves) else None
                self_after = moves[i + 2][3] if i + 2 < len(moves) else None
                sig = []
                if opp_after is not None:
                    sig.append(e + opp_after)                       # mover overestimated
                    if opp_before is not None:
                        sig.append(opp_after - opp_before)          # opponent's view of the loss
                if self_after is not None:
                    sig.append(e - self_after)                      # mover's own view two plies later
                # require two independent signals (or both engines) to agree
                sig.sort(reverse=True)
                swing = sig[1] if len(sig) >= 2 else 0
                rec = dict(fen=fen, game_move=mv, mover=mover, eval=e, swing=swing, phase=ph,
                           src=f'{src}#{gi + 1}@{i}', tc=tags.get('TimeControl', ''))
                if swing >= a.min_swing:
                    game_swings.append(rec)
                elif i >= 4:
                    pool_random.append(rec)
            game_swings.sort(key=lambda r: -r['swing'])
            for r in game_swings[:a.per_game]:
                seen.add(' '.join(r['fen'].split()[:4]))
                out.append(dict(r, kind='swing'))
            # one quiet control position per game, so the suite is not only blunders
            if pool_random:
                r = rng.choice(pool_random)
                key = ' '.join(r['fen'].split()[:4])
                if key not in seen:
                    seen.add(key)
                    out.append(dict(r, kind='random'))
    # Shuffled, interleaved 2 swing : 1 random, so `label --limit N` labels a fair sample.
    rng.shuffle(out)
    sw = [r for r in out if r['kind'] == 'swing']
    rd = [r for r in out if r['kind'] == 'random']
    out = []
    while sw or rd:
        out += sw[:2] + rd[:1]
        sw, rd = sw[2:], rd[1:]
    with open(a.out, 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    kinds = {}
    for r in out:
        kinds[r['kind']] = kinds.get(r['kind'], 0) + 1
    print(f'{ngames} games -> {len(out)} candidates {kinds}')


def solve_point(rows, best):
    """First (depth, nodes) from which every later completed depth's best move is `best`."""
    ds = sorted(rows)
    for d in ds:
        if all(rows[e][0] == best for e in ds if e >= d):
            return d, rows[d][2]
    return None, None


def label_one(a, rec):
    eng = Engine(a.engine, hash_mb=a.hash)
    try:
        d3 = eng.search(rec['fen'], f'depth {a.trivial_depth}')
        deep = eng.search(rec['fen'], f'nodes {a.root_nodes}')
        best, rows = deep['bestmove'], deep['rows']
        if not rows:
            return None
        sd, sn = solve_point(rows, best)
        top = max(rows)
        res = dict(rec, d3=d3['bestmove'], best=best, root_score=rows[top][1], depth=top,
                   solve_depth=sd, solve_nodes=sn,
                   iter_moves=[[d, rows[d][0], rows[d][1], rows[d][2]] for d in sorted(rows)])
        # Children: the best move plus other moves the deep search or the game considered,
        # each searched to the same node budget so their scores are comparable.
        alts = []
        for m in [rec['game_move'], d3['bestmove']] + [rows[d][0] for d in sorted(rows, reverse=True)]:
            if m and m != best and m not in alts:
                alts.append(m)
        res['alts'] = {}
        if sd is None or sd < a.min_solve_depth:
            res['skip'] = 'trivial'
            return res
        if sd > top - a.stable_iters:
            res['skip'] = 'unstable'
            return res
        if abs(rows[top][1]) > a.max_abs_score:
            res['skip'] = 'decided'
            return res
        if not alts:
            res['skip'] = 'noalt'
            return res
        for m in [best] + alts[:a.max_alts]:
            c = eng.search(rec['fen'], f'nodes {a.child_nodes}', moves=[m])
            if not c['rows']:
                continue
            res['alts'][m] = -c['rows'][max(c['rows'])][1]
        res['score'] = res['alts'].pop(best, None)
        if res['score'] is None or not res['alts']:
            res['skip'] = 'nochild'
        return res
    finally:
        eng.close()


def label(a):
    cands = [json.loads(l) for l in open(a.candidates)]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            done.add(json.loads(l)['fen'])
    todo = [c for c in cands if c['fen'] not in done and (not a.kind or c['kind'] == a.kind)]
    if a.limit:
        todo = todo[:a.limit]
    print(f'{len(todo)} to label ({len(done)} done)', flush=True)
    with open(a.out, 'a') as f, ThreadPoolExecutor(a.parallel) as ex:
        for i, r in enumerate(ex.map(lambda c: label_one(a, c), todo)):
            if r is not None:
                f.write(json.dumps(r) + '\n')
                f.flush()
            if i % 50 == 0:
                print(f'  {i}/{len(todo)}', flush=True)


def select(a):
    rows = [json.loads(l) for l in open(a.labels)]
    keep = []
    for r in rows:
        if r.get('skip') or not r['alts']:
            continue
        gap = r['score'] - max(r['alts'].values())
        if gap < a.min_gap:
            continue
        r['gap'] = gap
        keep.append(r)
    buckets = {'opening': [], 'middlegame': [], 'endgame': []}
    for r in keep:
        ph = r['phase']
        buckets['opening' if ph >= 20 else 'middlegame' if ph >= 10 else 'endgame'].append(r)
    rng = random.Random(7)
    out = []
    for k, v in buckets.items():
        rng.shuffle(v)
        out += [dict(r, bucket=k) for r in v[:a.per_bucket]]
    out.sort(key=lambda r: r['src'])
    with open(a.out, 'w') as f:
        for i, r in enumerate(out):
            r = {k: r[k] for k in ('fen', 'best', 'score', 'alts', 'gap', 'depth', 'solve_depth',
                                   'solve_nodes', 'root_score', 'd3', 'game_move', 'mover', 'swing', 'kind',
                                   'bucket', 'phase', 'src', 'tc')}
            f.write(json.dumps(dict(id=f'cp{i:04d}', **r)) + '\n')
    print(f'{len(rows)} labelled, {len(keep)} pass, buckets ' +
          ', '.join(f'{k} {len(v)}' for k, v in buckets.items()) + f' -> {len(out)} written')


def peek(a):
    """Yield summary of a labels file (how many pass each filter)."""
    rows = [json.loads(l) for l in open(a.labels)]
    stat = {}
    for r in rows:
        gap = (r['score'] - max(r['alts'].values())) if r.get('score') is not None and r['alts'] else None
        if a.verbose:
            print(r['kind'][:3], r['phase'], r['best'], r['d3'], r['game_move'], r.get('score'), r.get('root_score'), r['depth'],
                  r['solve_depth'], r.get('skip', ''), gap, r['alts'])
        k = (r['kind'], 'skip' if r.get('skip') else 'noalt' if gap is None else
             'gap>=50' if gap >= 50 else 'gap<50')
        stat[k] = stat.get(k, 0) + 1
    for k in sorted(stat):
        print(k, stat[k])


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    e = sp.add_parser('extract')
    e.add_argument('pgn', nargs='+')
    e.add_argument('--out', default='candidates.jsonl')
    e.add_argument('--min-swing', type=int, default=100)
    e.add_argument('--max-eval', type=int, default=250)
    e.add_argument('--per-game', type=int, default=3)
    l = sp.add_parser('label')
    l.add_argument('--engine', required=True)
    l.add_argument('--candidates', default='candidates.jsonl')
    l.add_argument('--out', default='labels.jsonl')
    l.add_argument('--root-nodes', type=int, default=6_000_000)
    l.add_argument('--child-nodes', type=int, default=3_000_000)
    l.add_argument('--trivial-depth', type=int, default=3)
    l.add_argument('--min-solve-depth', type=int, default=5)
    l.add_argument('--max-alts', type=int, default=4)
    l.add_argument('--hash', type=int, default=64)
    l.add_argument('--parallel', type=int, default=3)
    l.add_argument('--limit', type=int, default=0)
    l.add_argument('--kind', default='', help='only label candidates of this kind (swing/random)')
    l.add_argument('--stable-iters', type=int, default=2,
                   help='best move must already be best this many iterations before the last')
    l.add_argument('--max-abs-score', type=int, default=400)
    s = sp.add_parser('select')
    s.add_argument('--labels', default='labels.jsonl')
    s.add_argument('--out', default='suite.jsonl')
    s.add_argument('--min-gap', type=int, default=50)
    s.add_argument('--per-bucket', type=int, default=400)
    p = sp.add_parser('peek')
    p.add_argument('labels')
    p.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()
    {'extract': extract, 'label': label, 'select': select, 'peek': peek}[a.cmd](a)


if __name__ == '__main__':
    main()
