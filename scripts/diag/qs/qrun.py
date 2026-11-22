"""Run an engine over pos962 at fixed depth (3 procs), merge statsdump counters.
usage: [ENV=..] qrun.py ENGINE OUT.json [--depth 13] [--par 3] [--limit N] [--nostats]
Records total nodes, per-search wall time (ms from the final info line) and bestmoves."""
import argparse, json, os, subprocess, sys, tempfile, threading, time
from collections import defaultdict
POS = "/private/tmp/claude-501/-Users-andreasholley-dev-stockfish-challenge/4c85bf7b-729e-4ca9-80d9-7d20684e957e/scratchpad/hist/pos962.txt"
ap = argparse.ArgumentParser()
ap.add_argument("engine"); ap.add_argument("out")
ap.add_argument("--depth", type=int, default=13); ap.add_argument("--par", type=int, default=1)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--nostats", action="store_true")
ap.add_argument("--hash", type=int, default=64)
a = ap.parse_args()
fens = [l.strip() for l in open(POS) if l.strip()]
if a.limit: fens = fens[:a.limit]
tmp = tempfile.mkdtemp(prefix="qrun-", dir=os.path.dirname(os.path.abspath(__file__)))
res = [None] * a.par
def work(w):
    cmds = ["setoption name Hash value %d" % a.hash, "isready"]
    ids = list(range(w, len(fens), a.par))
    for i in ids:
        cmds += ["ucinewgame", "position " + fens[i], "go depth %d" % a.depth]
    if not a.nostats: cmds.append("statsdump %s/s%d.json" % (tmp, w))
    cmds.append("quit")
    p = subprocess.Popen([a.engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    out, _ = p.communicate("\n".join(cmds) + "\n")
    rows = []; last = None; k = 0
    for line in out.splitlines():
        t = line.split()
        if line.startswith("info depth") and "nodes" in t:
            last = (int(t[2]), int(t[t.index("nodes") + 1]), int(t[t.index("time") + 1]) if "time" in t else 0)
        elif line.startswith("bestmove"):
            rows.append({"i": ids[k], "best": t[1], "d": last[0], "nodes": last[1], "ms": last[2]}); k += 1
    res[w] = rows
ts = [threading.Thread(target=work, args=(w,)) for w in range(a.par)]
t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
merged = defaultdict(list)
if not a.nostats:
    for w in range(a.par):
        d = json.load(open("%s/s%d.json" % (tmp, w)))
        for k, v in d.items():
            m = merged[k]
            if len(m) < len(v): m.extend([0] * (len(v) - len(m)))
            for j, x in enumerate(v): m[j] += x
rows = sorted(sum(res, []), key=lambda r: r["i"])
merged["_rows"] = rows
merged["_wall"] = time.time() - t0
json.dump(merged, open(a.out, "w"))
print("%s: %d searches, nodes %d, sum ms %d, wall %.0fs" % (a.out, len(rows), sum(r["nodes"] for r in rows), sum(r["ms"] for r in rows), merged["_wall"]))
