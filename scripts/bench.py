#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time

POSITIONS = [
    "startpos",
    "fen r1bq1rk1/pppn1ppp/3bpn2/3p4/2PP4/2N1PN2/PPQ2PPP/R1B1KB1R w KQ - 2 8",
    "fen r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1",
    "fen 8/2p5/3p4/KP5r/1R3pk1/4P1p1/5P2/8 w - - 0 1",
]

def send(p, s):
    p.stdin.write(s + "\n")
    p.stdin.flush()


def read_until(p, prefix, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        line = p.stdout.readline()
        if not line:
            continue
        line = line.strip()
        if line.startswith(prefix):
            return line
    raise RuntimeError(f"timeout waiting for {prefix}")


def bench(engine_cmd, depth, options):
    p = subprocess.Popen(
        engine_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    try:
        send(p, "uci")
        read_until(p, "uciok")
        send(p, "setoption name OwnBook value false")
        for option in options:
            if "=" not in option:
                raise RuntimeError(f"option must be NAME=VALUE, got: {option}")
            name, value = option.split("=", 1)
            send(p, f"setoption name {name} value {value}")
        send(p, "isready")
        read_until(p, "readyok")

        total_nodes = 0
        total_ms = 0

        for i, pos in enumerate(POSITIONS, 1):
            if pos == "startpos":
                send(p, "position startpos")
            else:
                send(p, f"position {pos}")

            send(p, f"go depth {depth}")
            nodes = 0
            tms = 0
            while True:
                line = p.stdout.readline()
                if not line:
                    continue
                line = line.strip()
                if line.startswith("info "):
                    parts = line.split()
                    for j, tok in enumerate(parts):
                        if tok == "nodes" and j + 1 < len(parts):
                            try:
                                nodes = int(parts[j + 1])
                            except ValueError:
                                pass
                        if tok == "time" and j + 1 < len(parts):
                            try:
                                tms = int(parts[j + 1])
                            except ValueError:
                                pass
                elif line.startswith("bestmove"):
                    break

            total_nodes += nodes
            total_ms += tms
            nps = int(nodes * 1000 / max(1, tms))
            print(f"pos {i}/{len(POSITIONS)}: nodes={nodes} time_ms={tms} nps={nps}")

        total_nps = int(total_nodes * 1000 / max(1, total_ms))
        print(f"\nTOTAL: nodes={total_nodes} time_ms={total_ms} nps={total_nps}")
    finally:
        try:
            send(p, "quit")
        except Exception:
            pass
        p.terminate()


def main():
    ap = argparse.ArgumentParser(description="No-deps UCI bench runner")
    ap.add_argument("--engine", default="./target/release/rustfish")
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument(
        "--engine-option",
        action="append",
        default=[],
        help="UCI option in NAME=VALUE form, repeatable",
    )
    args = ap.parse_args()
    bench([args.engine], args.depth, args.engine_option)


if __name__ == "__main__":
    sys.exit(main())
