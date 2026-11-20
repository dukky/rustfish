"""Depth-14 misses vs a depth-20 oracle, classified by which mechanism hid the oracle move."""
import subprocess, sys, json, threading
ENG, OUT = sys.argv[1], sys.argv[2]
files = sys.argv[3:]
D, OD, GAP = 14, 20, 25
BITS = {1: "lmr", 2: "lmp", 4: "fut", 8: "see", 16: "nmp", 32: "rfp", 64: "razor", 128: "probcut", 256: "se", 512: "iir", 1024: "ttcut"}
pos = []
for f in files:
    for l in open(f):
        l = l.strip()
        if l and not l.startswith("#"):
            pos.append("position " + l if l.startswith(("fen ", "startpos")) else "position fen " + l)
lock = threading.Lock()
out = open(OUT, "a")
done = set()
try:
    for l in open(OUT):
        done.add(json.loads(l)["pos"])
except Exception:
    pass


class Eng:
    def __init__(self):
        self.p = subprocess.Popen([ENG], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.off = None
        self.send("setoption name Hash value 64")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def search(self, p, depth, moves=None, off=0):
        if off != self.off:
            self.send("setoption name DiagOff value %d" % off)
            self.off = off
        self.send("ucinewgame")
        self.send(p)
        self.send("go depth %d" % depth + (" searchmoves " + " ".join(moves) if moves else ""))
        info = []
        while True:
            l = self.p.stdout.readline()
            t = l.split()
            if l.startswith("info depth") and "score" in t:
                i = t.index("score")
                sc = int(t[i + 2]) if t[i + 1] == "cp" else (30000 - abs(int(t[i + 2]))) * (1 if int(t[i + 2]) > 0 else -1)
                info.append((int(t[2]), sc, t[t.index("pv") + 1], int(t[t.index("nodes") + 1])))
            if l.startswith("bestmove"):
                return t[1], (info[-1][1] if info else 0), info


def work(w, n):
    e = Eng()
    for i in range(w, len(pos), n):
        p = pos[i]
        if p in done:
            continue
        a, sa, info14 = e.search(p, D)
        b, sb, info20 = e.search(p, OD)
        r = {"pos": p, "a": a, "b": b, "sa14": sa, "sb20": sb, "it14": [(x[0], x[1], x[2]) for x in info14], "it20": [(x[0], x[2]) for x in info20]}
        if a != b:
            _, sa20, _ = e.search(p, OD, [a])
            r["sa20"] = sa20
            r["gap"] = sb - sa20
            if sb - sa20 >= GAP and abs(sb) < 20000:
                _, a14, _ = e.search(p, D, [a])
                _, b14, _ = e.search(p, D, [b])
                r["a14"], r["b14"] = a14, b14
                if b14 <= a14:
                    fl = {}
                    for bit, name in BITS.items():
                        _, x, ia = e.search(p, D, [a], bit)
                        _, y, ib = e.search(p, D, [b], bit)
                        fl[name] = [x, y, ia[-1][3] + ib[-1][3]]
                    r["toggles"] = fl
                    dep = {}
                    for d in (15, 16, 17, 18):
                        _, x, _ = e.search(p, d, [a])
                        _, y, _ = e.search(p, d, [b])
                        dep[d] = [x, y]
                        if y > x:
                            break
                    r["depth"] = dep
        with lock:
            out.write(json.dumps(r) + "\n")
            out.flush()
    e.send("quit")


N = int(__import__("os").environ.get("PAR", "4"))
ts = [threading.Thread(target=work, args=(w, N)) for w in range(N)]
[t.start() for t in ts]
[t.join() for t in ts]
