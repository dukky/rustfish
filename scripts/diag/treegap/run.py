#!/usr/bin/env python3
"""Tree-gap runner (docs/tree-gap-2026-11-26.md): one UCI engine over a
position file, `go depth D` or `go movetime M` per position (ucinewgame
first, fresh hash), recording every `info` line that carries a depth
(depth, seldepth, nodes, time, score, first PV move) plus rustfish's
`info string treestats` / `iter` lines. Output: JSONL, one record per position.

  run.py --engine engines/peer --positions positions.txt --depth 16 \
      --option Hash=16 --jobs 4 --out clover.jsonl

Black-box use of other engines only: nothing but their UCI output is read.
Python stdlib only.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def parse_info(line):
    t = line.split()
    rec = {}
    i = 1
    while i < len(t):
        w = t[i]
        if w in ("depth", "seldepth", "nodes", "time", "nps", "multipv", "hashfull", "tbhits"):
            if i + 1 < len(t):
                try:
                    rec[w] = int(t[i + 1])
                except ValueError:
                    pass
            i += 2
        elif w == "score" and i + 2 < len(t):
            kind, v = t[i + 1], t[i + 2]
            try:
                rec["score"] = int(v) if kind == "cp" else (100000 - abs(int(v))) * (1 if int(v) > 0 else -1)
            except ValueError:
                pass
            i += 3
            while i < len(t) and t[i] in ("lowerbound", "upperbound"):
                rec["bound"] = t[i]
                i += 1
        elif w == "pv":
            rec["pv"] = t[i + 1] if i + 1 < len(t) else None
            break
        elif w == "string":
            break
        else:
            i += 1
    return rec


def run_position(engine, options, knobs, spec, line, timeout):
    p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(s):
        p.stdin.write(s + "\n")
        p.stdin.flush()

    send("uci")
    for o in options:
        n, v = o.split("=", 1)
        send(f"setoption name {n} value {v}")
    if knobs:
        send(f"setoption name DiagKnobs value {knobs}")
    send("isready")
    for l in p.stdout:
        if l.startswith("readyok"):
            break
    send("ucinewgame")
    send("isready")
    for l in p.stdout:
        if l.startswith("readyok"):
            break
    pos = line if line.startswith("position") else "position " + line
    send(pos)
    timer = threading.Timer(timeout, lambda: send("stop"))
    t0 = time.time()
    send("go " + spec)
    timer.start()
    iters, stats, sel = [], None, {}
    best = None
    for l in p.stdout:
        if l.startswith("bestmove"):
            best = l.split()[1] if len(l.split()) > 1 else None
            break
        if l.startswith("info string treestats"):
            stats = {}
            for kv in l.split()[3:]:
                k, v = kv.split("=", 1)
                stats[k] = [int(x) for x in v.split(",")] if "," in v else int(v)
            continue
        if l.startswith("info string iter"):
            w = l.split()
            sel[int(w[3])] = int(w[5])
            continue
        if l.startswith("info") and " depth " in l and " nodes " in l:
            r = parse_info(l)
            if "bound" in r or r.get("multipv", 1) != 1:
                continue
            iters.append(r)
    timer.cancel()
    wall = time.time() - t0
    try:
        send("quit")
    except BrokenPipeError:
        pass
    p.wait(timeout=10)
    # keep the last line per depth (Clover may print several per depth)
    byd = {}
    for r in iters:
        byd[r["depth"]] = r
    for d, s in sel.items():
        if d in byd:
            byd[d]["seldepth"] = s
    return {"pos": line, "best": best, "wall": wall, "iters": [byd[d] for d in sorted(byd)], "stats": stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--movetime", type=int)
    ap.add_argument("--nodes", type=int)
    ap.add_argument("--option", action="append", default=[])
    ap.add_argument("--knobs", default="")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N positions only")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = [l.strip() for l in open(a.positions) if l.strip() and not l.startswith("#")]
    if a.limit:
        lines = lines[: a.limit]
    spec = f"depth {a.depth}" if a.depth else f"movetime {a.movetime}" if a.movetime else f"nodes {a.nodes}"
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda l: run_position(a.engine, a.option, a.knobs, spec, l, a.timeout), lines))
    with open(a.out, "w") as f:
        for i, r in enumerate(res):
            r["i"] = i
            f.write(json.dumps(r) + "\n")
    tot = sum(r["iters"][-1]["nodes"] for r in res if r["iters"])
    print(f"{a.out}: {len(res)} positions, {tot} nodes, {sum(r['wall'] for r in res):.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
