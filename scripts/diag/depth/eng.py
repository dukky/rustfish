"""Small UCI driver shared by the depth scripts."""
import subprocess


def parse_score(t):
    i = t.index("score")
    v = int(t[i + 2])
    if t[i + 1] == "mate":
        v = (30000 - abs(v)) * (1 if v > 0 else -1)
    return v


class Eng:
    def __init__(self, path, opts=()):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.send("setoption name Hash value 64")
        for o in opts:
            n, v = o.split("=", 1)
            self.send("setoption name %s value %s" % (n, v))

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def search(self, pos, depth=None, moves=None, nodes=None, extra=()):
        self.send("ucinewgame")
        for e in extra:
            self.send(e)
        self.send(pos)
        lim = ("depth %d" % depth) if depth else ("nodes %d" % nodes)
        self.send("go " + lim + (" searchmoves " + " ".join(moves) if moves else ""))
        info = []
        lines = []
        while True:
            l = self.p.stdout.readline()
            if not l:
                raise RuntimeError("engine died")
            t = l.split()
            if l.startswith("info depth") and "score" in t and "pv" in t:
                pv = t[t.index("pv") + 1:]
                info.append({"d": int(t[2]), "s": parse_score(t), "n": int(t[t.index("nodes") + 1]), "pv": pv})
            elif not l.startswith("info depth"):
                lines.append(l.rstrip())
            if l.startswith("bestmove"):
                return {"best": t[1], "score": info[-1]["s"] if info else 0,
                        "nodes": info[-1]["n"] if info else 0, "info": info, "other": lines}

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(5)
        except Exception:
            self.p.kill()


def load_positions(path):
    out = []
    for l in open(path):
        l = l.strip()
        if l and not l.startswith("#"):
            out.append("position " + l if l.startswith(("fen ", "startpos")) else "position fen " + l)
    return out
