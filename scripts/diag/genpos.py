import subprocess, sys, re, glob, threading, random
ENG = sys.argv[1]
OUT = sys.argv[2]
fens = set()
for f in glob.glob("/private/tmp/claude-501/-Users-andreasholley-dev-stockfish-challenge/4c85bf7b-729e-4ca9-80d9-7d20684e957e/scratchpad/games/s*/games-*.pgn"):
    for m in re.finditer(r'^\[FEN "([^"]+)"\]', open(f).read(), re.M):
        fens.add(m.group(1))
fens = sorted(fens)
rng = random.Random(20261120)
jobs = [(f, 8 + rng.randrange(23)) for f in fens]
res = [None] * len(jobs)


class Eng:
    def __init__(self):
        self.p = subprocess.Popen([ENG], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("setoption name Hash value 16")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def go(self, pos, cmd):
        self.send(pos)
        self.send(cmd)
        while True:
            l = self.p.stdout.readline()
            if l.startswith("bestmove"):
                return l.split()[1]


def work(w, n):
    e = Eng()
    for i in range(w, len(jobs), n):
        fen, L = jobs[i]
        moves = []
        e.send("ucinewgame")
        for _ in range(L):
            bm = e.go("position fen %s moves %s" % (fen, " ".join(moves)), "go depth 7")
            if bm in ("(none)", "0000"):
                break
            moves.append(bm)
        res[i] = "fen %s moves %s" % (fen, " ".join(moves))
    e.send("quit")


ts = [threading.Thread(target=work, args=(w, 4)) for w in range(4)]
[t.start() for t in ts]
[t.join() for t in ts]
open(OUT, "w").write("\n".join(r for r in res if r) + "\n")
