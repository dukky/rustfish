"""Depth-14 vs depth-20 oracle on a position list (resumable).
usage: oracle.py ENGINE POSFILE OUT.jsonl [PAR]"""
import json, sys, threading, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eng import Eng, load_positions

ENG, POS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PAR = int(sys.argv[4]) if len(sys.argv) > 4 else 3
D, OD = 14, 20
pos = load_positions(POS)
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        done.add(json.loads(l)["i"])
lock = threading.Lock()
out = open(OUT, "a")


def work(w):
    e = Eng(ENG)
    for i in range(w, len(pos), PAR):
        if i in done:
            continue
        p = pos[i]
        a = e.search(p, D)
        b = e.search(p, OD)
        r = {"i": i, "pos": p, "a": a["best"], "sa14": a["score"], "n14": a["nodes"],
             "b": b["best"], "sb20": b["score"], "n20": b["nodes"], "pv20": b["info"][-1]["pv"],
             "it14": [(x["d"], x["s"], x["pv"][0], x["n"]) for x in a["info"]],
             "it20": [(x["d"], x["s"], x["pv"][0], x["n"]) for x in b["info"]]}
        if a["best"] != b["best"]:
            x = e.search(p, OD, [a["best"]])
            r["sa20"] = x["score"]
            r["gap"] = b["score"] - x["score"]
        with lock:
            out.write(json.dumps(r) + "\n")
            out.flush()
    e.quit()


ts = [threading.Thread(target=work, args=(w,)) for w in range(PAR)]
[t.start() for t in ts]
[t.join() for t in ts]
