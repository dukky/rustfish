#!/usr/bin/env python3
"""Check the engine's bench node count against the latest `Bench: N` commit trailer.

The bench total is a signature of search behavior. Rules:
  - If HEAD's commit message has `Bench: N`, the engine must produce exactly N.
  - Otherwise it must match the most recent `Bench: N` in history (a
    non-functional change: refactor, speedup, docs, tooling).
  - With no `Bench:` anywhere in history, the check only reports.

Usage:
  python3 scripts/check_bench.py [--engine ./target/release/rustfish]
"""
import argparse
import re
import subprocess
import sys

BENCH_RE = re.compile(r"^Bench:\s*(\d+)\s*$", re.MULTILINE)


def run_bench(engine):
    out = subprocess.run([engine], input="bench\nquit\n", capture_output=True,
                         text=True, check=True).stdout
    m = re.search(r"Nodes searched\s*:\s*(\d+)", out)
    if not m:
        sys.exit(f"no 'Nodes searched' line in bench output:\n{out[-2000:]}")
    nps = re.search(r"Nodes/second\s*:\s*(\d+)", out)
    return int(m.group(1)), int(nps.group(1)) if nps else 0


def expected_bench():
    """Return (bench, commit, is_head) for the newest commit carrying a Bench trailer."""
    log = subprocess.run(["git", "log", "-z", "--format=%H%n%B"], capture_output=True,
                         text=True, check=True).stdout
    for i, entry in enumerate(e for e in log.split("\0") if e.strip()):
        sha, _, body = entry.strip().partition("\n")
        m = BENCH_RE.findall(body)
        if m:
            return int(m[-1]), sha[:7], i == 0
    return None, None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./target/release/rustfish")
    args = ap.parse_args()

    nodes, nps = run_bench(args.engine)
    want, sha, at_head = expected_bench()
    print(f"bench: {nodes} nodes ({nps} nps)")
    if want is None:
        print("no 'Bench:' trailer in history yet; nothing to compare")
        return
    if nodes == want:
        print(f"matches Bench: {want} from {sha}")
        return
    if at_head:
        sys.exit(f"HEAD says 'Bench: {want}' but the engine searched {nodes} nodes")
    sys.exit(
        f"search behavior changed: {nodes} nodes, expected {want} (from {sha}).\n"
        f"If the change is intended, add 'Bench: {nodes}' to the commit message."
    )


if __name__ == "__main__":
    main()
