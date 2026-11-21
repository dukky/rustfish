"""Localize where a depth-14 search goes wrong on an oracle miss.

For each significant miss (oracle move B better than d14's A by >= 25 cp at
depth 20), follow the line the d14 search used to reject B: at opponent nodes
take the move that cut (d14's refutation), at our nodes take the oracle's move
(a depth 20-ply search of that position), and trace how the d14 search handled
each line move (index, stage, reduction, extension, score).

usage: localize.py STATS_ENGINE ORACLE.jsonl OUT.jsonl [PAR]
"""
import json, os, re, sys, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eng import Eng

ENG, ORC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PAR = int(sys.argv[4]) if len(sys.argv) > 4 else 3
D, OD, GAP, MAXPLY = 14, 20, 25, 12

rows = [json.loads(l) for l in open(ORC)]
miss = [r for r in rows if r.get("gap", 0) >= GAP and abs(r["sb20"]) < 20000]
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["i"] for l in open(OUT)}
lock = threading.Lock()
out = open(OUT, "a")


def with_moves(pos, moves):
    if not moves:
        return pos
    return pos + (" " if " moves " in pos else " moves ") + " ".join(moves)


def kv(line):
    return dict(re.findall(r"(\w+)=(\S+)", line))


def trace(e, pos, line):
    """Run d14 with the line traced; return the decisive visit at each ply."""
    e.send("ucinewgame")
    e.send(pos)
    e.send("traceline " + " ".join(line))
    e.send("go depth %d" % D)
    lines = []
    while True:
        l = e.p.stdout.readline()
        if l.startswith(("TRN", "TRR")):
            lines.append(l.rstrip())
        if l.startswith("bestmove"):
            break
    # Walk back from the last root visit of the last iteration.
    root_i = max(i for i, l in enumerate(lines) if l.startswith("TRR")) if any(l.startswith("TRR") for l in lines) else None
    if root_i is None:
        return None, []
    root = kv(lines[root_i])
    chain = []
    end = root_i
    for ply in range(1, len(line) + 1):
        found = None
        for i in range(end - 1, -1, -1):
            l = lines[i]
            if l.startswith("TRR"):
                break
            d = kv(l)
            if l.startswith("TRN") and int(d["ply"]) == ply and d.get("ex") == "0":
                found = i
                break
            if l.startswith("TRN") and int(d["ply"]) < ply:
                break
        if found is None:
            break
        d = kv(lines[found])
        d["raw"] = lines[found]
        chain.append(d)
        end = found
    return root, chain


def work(w):
    e = Eng(ENG)
    for k in range(w, len(miss), PAR):
        r = miss[k]
        if r["i"] in done:
            continue
        pos, line = r["pos"], [r["b"]]
        steps, stop = [], "maxply"
        root = None
        while len(line) <= MAXPLY:
            root, chain = trace(e, pos, line)
            if root is None:
                stop = "B not searched in last iteration"
                break
            if len(chain) < len(line):
                # The deepest line node was not reached on the decisive path.
                stop = "line node not visited (ply %d)" % len(chain)
                break
            node = chain[-1]
            why = node["why"]
            ply = len(line)
            steps.append({"ply": ply, "node": node["raw"]})
            if why.startswith("cut:"):
                mv = why[4:].split("@")[0]
                if ply % 2 == 0:
                    # Our node cut: our line move succeeded at this visit.
                    stop = "our node cuts"
                    break
                line.append(mv)
                continue
            if why.startswith(("fl:", "exact:")):
                if ply % 2 == 1:
                    stop = "opp node fails low"
                    break
                if int(node["d"]) <= 1:
                    stop = "horizon"
                    break
                o = e.search(with_moves(pos, line), max(OD - ply, 1))
                line.append(o["best"])
                continue
            stop = "node:" + why
            break
        # Final trace so the last appended move's handling is recorded.
        _, chain = trace(e, pos, line)
        res = {"i": r["i"], "pos": pos, "a": r["a"], "b": r["b"], "gap": r["gap"], "line": line,
               "root": root, "stop": stop, "chain": [c["raw"] for c in chain]}
        with lock:
            out.write(json.dumps(res) + "\n")
            out.flush()
    e.quit()


ts = [threading.Thread(target=work, args=(w,)) for w in range(PAR)]
[t.start() for t in ts]
[t.join() for t in ts]
