"""Run every position of a set: qsinfo + one fresh search; write JSONL.

python3 run.py ENGINE SET OUT.jsonl --go "nodes 4000000" [--qsdiag N] [--parallel 5] [--limit K]
"""
import argparse, json, re, subprocess, threading

ap = argparse.ArgumentParser()
ap.add_argument('engine'); ap.add_argument('set'); ap.add_argument('out')
ap.add_argument('--go', default='depth 10')
ap.add_argument('--qsdiag', type=int, default=0)
ap.add_argument('--parallel', type=int, default=5)
ap.add_argument('--hash', type=int, default=64)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--opt', action='append', default=[])
ap.add_argument('--noqs', action='store_true', help='engine has no qsinfo command')
a = ap.parse_args()

pos = [l.split(' ', 2) for l in open(a.set) if l.strip()]
if a.limit:
    pos = pos[:a.limit]
res = [None] * len(pos)

def score_of(line):
    m = re.search(r' score (cp|mate) (-?\d+)', line)
    if not m:
        return None
    if m.group(1) == 'cp':
        return int(m.group(2))
    v = int(m.group(2))
    return 3000 if v > 0 else -3000

def work(w):
    p = subprocess.Popen([a.engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    def send(s):
        p.stdin.write(s + '\n'); p.stdin.flush()
    send(f'setoption name Hash value {a.hash}')
    send('setoption name OwnBook value false')
    send(f'setoption name QsDiag value {a.qsdiag}')
    for o in a.opt:
        k, v = o.split('=')
        send(f'setoption name {k} value {v}')
    for i in range(w, len(pos), a.parallel):
        cls, pid, spec = pos[i]
        q = {}
        if not a.noqs:
            send('ucinewgame'); send('position ' + spec); send('qsinfo')
        while not a.noqs:
            l = p.stdout.readline()
            if l.startswith('qsinfo'):
                t = l.split()[1:]
                q = {t[j]: int(t[j + 1]) for j in range(0, len(t), 2)}
                break
        send('ucinewgame'); send('position ' + spec); send('go ' + a.go)
        rows = {}
        while True:
            l = p.stdout.readline()
            if l.startswith('info') and ' pv ' in l and ' depth ' in l:
                d = int(re.search(r' depth (\d+)', l).group(1))
                n = int(re.search(r' nodes (\d+)', l).group(1))
                rows[d] = (score_of(l), l.split(' pv ')[1].split()[0], n)
            elif l.startswith('bestmove'):
                bm = l.split()[1]
                break
        res[i] = {'id': pid, 'cls': cls, 'q': q, 'rows': rows, 'best': bm}
    send('quit'); p.wait()

ts = [threading.Thread(target=work, args=(w,)) for w in range(a.parallel)]
[t.start() for t in ts]; [t.join() for t in ts]
with open(a.out, 'w') as f:
    for r in res:
        f.write(json.dumps(r) + '\n')
