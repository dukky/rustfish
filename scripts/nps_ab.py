#!/usr/bin/env python3
"""Interleaved NPS A/B (temporary, release-test branch). Usage: nps_ab.py base:path cand:path ...
The first binary is the base; list it twice (second name e.g. base2) for an A/A control."""
import platform, re, statistics, subprocess, sys

BINS = [a.split(":", 1) for a in sys.argv[1:]]
ROUNDS, MT_ROUNDS = 12, 6
FENS = ["r1bq1rk1/pppn1ppp/3bpn2/3p4/2PP4/2N1PN2/PPQ2PPP/R1B1KB1R w KQ - 2 8",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
        "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/8 b - - 0 1"]
pin = ["taskset", "-c", "1"] if platform.system() == "Linux" else []

def run(path, inp):
    return subprocess.run(pin + [path], input=inp, capture_output=True, text=True, check=True).stdout

def bench(path):
    out = run(path, "bench 15\nquit\n")
    return int(re.search(r"Nodes/second\s*:\s*(\d+)", out).group(1))

def movetime(path):
    inp = "setoption name Hash value 64\n"
    for f in FENS:
        inp += f"ucinewgame\nposition fen {f}\ngo movetime 1000\nisready\n"
    out = run(path, inp + "quit\n")
    nodes = time = 0
    for block in out.split("bestmove")[:-1]:
        last = [l for l in block.splitlines() if " nodes " in l and " time " in l][-1]
        nodes += int(re.search(r" nodes (\d+)", last).group(1))
        time += int(re.search(r" time (\d+)", last).group(1))
    return nodes * 1000 // max(time, 1)

def table(title, fn, rounds):
    res = {n: [] for n, _ in BINS}
    for r in range(rounds):
        k = r % len(BINS)
        for n, p in BINS[k:] + BINS[:k]:
            res[n].append(fn(p))
    base = BINS[0][0]
    print(f"\n#### {title}, {rounds} interleaved rounds\n")
    print("| binary | median nps | median ratio to base | min..max |\n|---|---|---|---|")
    for n, _ in BINS:
        rat = [x / y for x, y in zip(res[n], res[base])]
        print(f"| {n} | {statistics.median(res[n]):.0f} | {statistics.median(rat):.4f} | {min(rat):.3f}..{max(rat):.3f} |")
    sys.stdout.flush()

print(f"### {platform.system()} {platform.processor() or ''}")
table("bench 15", bench, ROUNDS)
table("go movetime 1000 x 4 FENs (Hash 64)", movetime, MT_ROUNDS)
