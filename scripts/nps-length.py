#!/usr/bin/env python3
"""NPS against search length and hash size (one engine build per name).

    scripts/nps-length.py --bin main=./main --bin aa=./main2 --hash 16 \
        --movetimes 100,300,1000,3000 --fens 25 [--json out.json]

Each binary is one UCI process (Hash set once, 1 thread); for every FEN of
scripts/peer_nps.py's set (the first --fens of them, every other one) and every
movetime it runs `ucinewgame`, `position fen`, `go movetime T`, waiting for
bestmove; the binaries alternate per search in rotated order. Reports total
nodes / total time per (binary, movetime), and, if the binary prints
`info string stats k=v ...` (the speed-diag-stats branch), those counters
summed per movetime.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peer_nps import FENS  # noqa: E402


class Engine:
    def __init__(self, path, hash_mb):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci")
        self.wait("uciok")
        self.send(f"setoption name Hash value {hash_mb}")
        self.send("setoption name Threads value 1")
        self.send("isready")
        self.wait("readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def wait(self, tok):
        lines = []
        while True:
            l = self.p.stdout.readline()
            if not l:
                raise SystemExit("engine died")
            lines.append(l.strip())
            if l.startswith(tok):
                return lines

    def search(self, fen, movetime):
        self.send("ucinewgame")
        self.send(f"position fen {fen}")
        self.send("isready")
        self.wait("readyok")
        self.send(f"go movetime {movetime}")
        nodes = ms = depth = 0
        stats = {}
        for l in self.wait("bestmove"):
            t = l.split()
            if l.startswith("info") and "nodes" in t and "time" in t:
                nodes = int(t[t.index("nodes") + 1])
                ms = int(t[t.index("time") + 1])
                if "depth" in t:
                    depth = int(t[t.index("depth") + 1])
            if l.startswith("info string stats"):
                for kv in t[3:]:
                    k, v = kv.split("=")
                    stats[k] = int(v)
        return nodes, ms, depth, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", action="append", required=True)
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--movetimes", default="100,300,1000,3000")
    ap.add_argument("--fens", type=int, default=25)
    ap.add_argument("--json")
    a = ap.parse_args()
    bins = [b.split("=", 1) for b in a.bin]
    mts = [int(x) for x in a.movetimes.split(",")]
    fens = FENS[::2][: a.fens]
    eng = {n: Engine(p, a.hash) for n, p in bins}
    res = {n: {mt: {"nodes": 0, "ms": 0, "depth": 0, "n": 0, "stats": {}} for mt in mts} for n, _ in bins}
    k = 0
    for fen in fens:
        for mt in mts:
            order = bins[k % len(bins):] + bins[: k % len(bins)]
            k += 1
            for n, _ in order:
                nodes, ms, depth, stats = eng[n].search(fen, mt)
                r = res[n][mt]
                r["nodes"] += nodes
                r["ms"] += ms
                r["depth"] += depth
                r["n"] += 1
                for kk, v in stats.items():
                    r["stats"][kk] = r["stats"].get(kk, 0) + v
    for n, _ in bins:
        eng[n].send("quit")
    print(f"hash {a.hash} MB, {len(fens)} FENs")
    for n, _ in bins:
        for mt in mts:
            r = res[n][mt]
            nps = r["nodes"] * 1000 / max(r["ms"], 1)
            line = f"{n:>12} movetime {mt:>5}: nps {nps:>9.0f} avg depth {r['depth'] / r['n']:.1f} nodes/search {r['nodes'] / r['n']:.0f}"
            st = r["stats"]
            if st:
                nodes = st.get("negamax", 0) + st.get("qnodes", 0)
                per = lambda x: st.get(x, 0) / max(nodes, 1)
                line += (f" | q {per('qnodes'):.3f} tt_hit/probe {st.get('tt_hit', 0) / max(st.get('tt_probe', 1), 1):.3f}"
                         f" ec_hit/probe {st.get('ec_hit', 0) / max(st.get('ec_probe', 1), 1):.3f} nnue/node {per('nnue_eval'):.3f}"
                         f" pend1 {per('pend1'):.3f} pend2 {per('pend2'):.3f} pend3+ {per('pend3p'):.3f} refresh/node {per('refresh'):.4f}")
            print(line, flush=True)
    if a.json:
        json.dump({"hash": a.hash, "res": res}, open(a.json, "w"))


if __name__ == "__main__":
    main()
