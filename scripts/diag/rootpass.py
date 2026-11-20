"""Second pass: for root-category misses, which switch makes the full depth-14 search pick the oracle move."""
import json, sys, threading
ENG, IN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
BITS = {0: "base", 1: "lmr", 2: "lmp", 4: "fut", 8: "see", 16: "nmp", 32: "rfp", 64: "razor", 128: "probcut", 256: "se", 512: "iir", 1024: "ttcut", 2048: "cutlmr", 4096: "rootcut", 8192: "asp"}
import subprocess


class Eng:
    def __init__(self):
        self.p = subprocess.Popen([ENG], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("setoption name Hash value 64")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def search(self, p, depth, off=0):
        self.send("setoption name DiagOff value %d" % off)
        self.send("ucinewgame")
        self.send(p)
        self.send("go depth %d" % depth)
        nodes = 0
        while True:
            l = self.p.stdout.readline()
            t = l.split()
            if l.startswith("info depth") and "nodes" in t:
                nodes = int(t[t.index("nodes") + 1])
            if l.startswith("bestmove"):
                return t[1], nodes


rs = [json.loads(l) for l in open(IN)]
todo = [r for r in rs if "a14" in r and r["b14"] > r["a14"]]
lock = threading.Lock()
out = open(OUT, "w")


def work(w, n):
    e = Eng()
    for i in range(w, len(todo), n):
        r = todo[i]
        res = {}
        for bit, name in BITS.items():
            bm, nodes = e.search(r["pos"], 14, bit)
            res[name] = [bm == r["b"], bm == r["a"], nodes]
        with lock:
            out.write(json.dumps({"pos": r["pos"], "a": r["a"], "b": r["b"], "res": res}) + "\n")
            out.flush()
    e.send("quit")


ts = [threading.Thread(target=work, args=(w, 4)) for w in range(4)]
[t.start() for t in ts]
[t.join() for t in ts]
