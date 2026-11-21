"""Run the stats build with DiagOff verify bit over positions; merge counters.
usage: vrun.py ENGINE POSFILE N DEPTH PAR OUT.json"""
import json, os, subprocess, sys, threading
eng, posf, n, depth, par, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eng import load_positions
pos = load_positions(posf)[:n]
tmp = out + ".parts"
os.makedirs(tmp, exist_ok=True)


def work(w):
    cmds = ["setoption name Hash value 64", "setoption name DiagOff value %d" % (1 << 20)]
    for p in pos[w::par]:
        cmds += ["ucinewgame", p, "go depth %d" % depth]
    cmds += ["statsdump %s/%d.json" % (tmp, w), "quit"]
    subprocess.run([eng], input="\n".join(cmds) + "\n", text=True, stdout=subprocess.DEVNULL)


ts = [threading.Thread(target=work, args=(w,)) for w in range(par)]
[t.start() for t in ts]
[t.join() for t in ts]
m = {}
for w in range(par):
    for k, v in json.load(open("%s/%d.json" % (tmp, w))).items():
        a = m.setdefault(k, [])
        a.extend([0] * (len(v) - len(a)))
        for i, x in enumerate(v):
            a[i] += x
json.dump(m, open(out, "w"))
