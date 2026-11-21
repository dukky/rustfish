"""Score a candidate engine against the depth-20 oracle.

For every oracle position: fresh `go depth D` with the candidate; the chosen
move is scored at depth 20 by the BASE engine (searchmoves, cached in
d20cache.jsonl). Regret = oracle best score - score of the chosen move.

usage: evalcand.py CAND_ENGINE NAME [--depth 14] [--par 3] [--opt Name=V ...]
Writes cand-NAME.jsonl; prints the summary (also: evalcand.py --summary NAME...).
"""
import argparse, json, os, sys, threading, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eng import Eng

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "rf-base")
ORC = os.path.join(HERE, "oracle-base.jsonl")
CACHE = os.path.join(HERE, "d20cache.jsonl")
GAP = 25


def load_oracle():
    return {r["i"]: r for r in (json.loads(l) for l in open(ORC))}


def load_cache():
    c = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            x = json.loads(l)
            c[(x["i"], x["m"])] = x["s"]
    return c


def summary(name, orc):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "cand-%s.jsonl" % name))]
    rows = [r for r in rows if r["i"] in orc]
    reg = [r["regret"] for r in rows]
    nodes = sum(r["n"] for r in rows)
    bnodes = sum(orc[r["i"]]["n14"] for r in rows)
    sig = {i for i, o in orc.items() if o.get("gap", 0) >= GAP and abs(o["sb20"]) < 20000}
    fixed = sum(1 for r in rows if r["i"] in sig and r["regret"] < GAP)
    picksB = sum(1 for r in rows if r["i"] in sig and r["m"] == orc[r["i"]]["b"])
    broke = sum(1 for r in rows if r["i"] not in sig and r["regret"] >= GAP)
    nsig = sum(1 for r in rows if r["regret"] >= GAP)
    agree = sum(1 for r in rows if r["m"] == orc[r["i"]]["b"])
    print("%-22s n=%d nodes x%.3f  mean regret %.2f cp  sig misses %d  (fixed %d/%d, picks B %d, new %d)  agree %.1f%%" % (
        name, len(rows), nodes / max(bnodes, 1), statistics.mean(reg), nsig, fixed, len(sig & {r["i"] for r in rows}),
        picksB, broke, 100.0 * agree / len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", nargs="?")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--summary", nargs="*")
    a = ap.parse_args()
    orc = load_oracle()
    if a.summary is not None:
        for n in a.summary:
            summary(n, orc)
        return
    out_path = os.path.join(HERE, "cand-%s.jsonl" % a.name)
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["i"] for l in open(out_path)}
    cache = load_cache()
    lock = threading.Lock()
    out = open(out_path, "a")
    cf = open(CACHE, "a")
    ids = sorted(i for i in orc if i not in done)

    def work(w):
        e = Eng(a.engine, a.opt)
        b = None
        for i in ids[w::a.par]:
            o = orc[i]
            x = e.search(o["pos"], a.depth)
            m = x["best"]
            if m == o["b"]:
                s = o["sb20"]
            elif m == o["a"] and "sa20" in o:
                s = o["sa20"]
            else:
                with lock:
                    s = cache.get((i, m))
                if s is None:
                    if b is None:
                        b = Eng(BASE)
                    s = b.search(o["pos"], 20, [m])["score"]
                    with lock:
                        cache[(i, m)] = s
                        cf.write(json.dumps({"i": i, "m": m, "s": s}) + "\n")
                        cf.flush()
            regret = max(0, o["sb20"] - s) if abs(o["sb20"]) < 20000 else (0 if s == o["sb20"] else min(1000, abs(o["sb20"] - s)))
            with lock:
                out.write(json.dumps({"i": i, "m": m, "n": x["nodes"], "s14": x["score"], "regret": regret}) + "\n")
                out.flush()
        e.quit()
        if b:
            b.quit()

    ts = [threading.Thread(target=work, args=(w,)) for w in range(a.par)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    out.close()
    summary(a.name, orc)


main()
