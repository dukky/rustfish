#!/usr/bin/env python3
"""For each EPD position with `am` (and the game move in UCI as c4), run one fresh
rustfish search (1 thread) and report the depth/nodes from which the best move is no
longer the avoid-move for every later depth. Lower is better.

    python3 scripts/diag/epd_depth.py --engine ./target/release/rustfish \
        --epd scripts/diag/clover-misses.epd --nodes 20000000
"""
import argparse, re, subprocess, statistics
from concurrent.futures import ThreadPoolExecutor


def search(engine, fen, nodes, hash_mb):
    p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    p.stdin.write(f'setoption name Hash value {hash_mb}\nsetoption name OwnBook value false\n'
                  f'ucinewgame\nisready\nposition fen {fen}\ngo nodes {nodes}\n')
    p.stdin.flush()
    rows = {}
    for line in p.stdout:
        if line.startswith('info') and ' pv ' in line:
            d = int(re.search(r' depth (\d+)', line).group(1))
            n = int(re.search(r' nodes (\d+)', line).group(1))
            rows[d] = (line.split(' pv ')[1].split()[0], n)
        elif line.startswith('bestmove'):
            break
    p.stdin.write('quit\n'); p.stdin.flush(); p.wait()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True)
    ap.add_argument('--epd', default='scripts/diag/clover-misses.epd')
    ap.add_argument('--nodes', type=int, default=20_000_000)
    ap.add_argument('--hash', type=int, default=64)
    ap.add_argument('--parallel', type=int, default=2)
    a = ap.parse_args()
    pos = []
    for line in open(a.epd):
        if not line.strip() or line.startswith('#'):
            continue
        f = line.split()
        fen = ' '.join(f[:4]) + ' 0 1'
        am = re.search(r'c4 "([a-h1-8qrbn]+)"', line).group(1)
        pid = re.search(r'id "([^"]+)"', line).group(1)
        pos.append((pid, fen, am))

    def run(p):
        rows = search(a.engine, p[1], a.nodes, a.hash)
        ds = sorted(rows)
        leave = next((d for d in ds if all(rows[e][0] != p[2] for e in ds if e >= d)), None)
        return p[0], leave, rows[leave][1] if leave else None, ds[-1] if ds else 0

    with ThreadPoolExecutor(a.parallel) as ex:
        res = list(ex.map(run, pos))
    for pid, leave, n, md in res:
        print(f'{pid:32} leaves am at depth {leave if leave else "-":>3}  nodes {n if n else "-":>10}  (max depth {md})')
    found = [r for r in res if r[1]]
    print(f'solved {len(found)}/{len(res)}; median depth {statistics.median(r[1] for r in found) if found else "-"}, '
          f'median nodes {statistics.median(r[2] for r in found) if found else "-"}')


if __name__ == '__main__':
    main()
