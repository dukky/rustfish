"""d14 agreement with the depth-20 oracle (base engine) over the miss-scan positions.
Usage: agree.py <engine> <misses.jsonl> <out.jsonl> [DiagOff]"""
import json, sys, subprocess, threading
ENG, IN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
OFF = int(sys.argv[4]) if len(sys.argv) > 4 else 0
rs = [json.loads(l) for l in open(IN)]
res = [None] * len(rs)


def work(w, n):
    p = subprocess.Popen([ENG], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    def send(s):
        p.stdin.write(s + "\n"); p.stdin.flush()
    send("setoption name Hash value 64")
    if OFF:
        send("setoption name DiagOff value %d" % OFF)
    for i in range(w, len(rs), n):
        send("ucinewgame"); send(rs[i]["pos"]); send("go depth 14")
        nodes = 0
        while True:
            l = p.stdout.readline(); t = l.split()
            if l.startswith("info depth") and "nodes" in t:
                nodes = int(t[t.index("nodes") + 1])
            if l.startswith("bestmove"):
                res[i] = (t[1], nodes); break
    send("quit")


ts = [threading.Thread(target=work, args=(w, 4)) for w in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
agree = sum(res[i][0] == r["b"] for i, r in enumerate(rs))
sig = [i for i, r in enumerate(rs) if "a14" in r]
fixed = sum(res[i][0] == rs[i]["b"] for i in sig)
broke = sum(res[i][0] != r["b"] for i, r in enumerate(rs) if r["a"] == r["b"])
nodes = sum(x[1] for x in res)
print(f"{ENG} off={OFF}: agree {agree}/{len(rs)} ({100*agree/len(rs):.1f}%), sig misses fixed {fixed}/{len(sig)}, base-agreeing now differ {broke}, nodes {nodes/len(rs)/1e3:.0f}k")
json.dump(res, open(OUT, "w"))
