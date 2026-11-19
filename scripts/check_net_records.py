#!/usr/bin/env python3
"""List net branches with no row in net/experiments.jsonl.

Net branches = net rows in the queue log (`log.jsonl` on queue-state) after
the trainer-v2 pivot, the set the dashboard lists, plus `nnue-*` branches on
origin committed after the pivot that change net/nnue.bin (not `-wip`
work in progress, not `nnue-scale*` eval-scale tests) (engine-only
branches such as `*-wip` inference work have no net of their own). Exit 1 if any
lacks a training row (add one with `scripts/netloop.py record`).
Usage: python3 scripts/check_net_records.py [--local-only]
"""
import json, subprocess, sys
from pathlib import Path

PIVOT = "2026-11-06T08:00:00"  # trainer v2 (deploy/dashboard/server.py NET_PIVOT)
NET_MODES = ("net-gate", "stc", "ltc")
REPO = Path(__file__).resolve().parent.parent


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout


def recorded(rows):
    """Branch names covered by rows (same rule as the dashboard's by_branch)."""
    return {r.get("branch") or f"nnue-{str(r['id']).lower()}" for r in rows if r.get("branch") or r.get("id")}


def missing(branches, rows):
    return sorted(set(branches) - recorded(rows))


def main():
    rows = [json.loads(l) for l in (REPO / "net/experiments.jsonl").read_text().splitlines() if l.strip()]
    branches = set()
    if "--local-only" not in sys.argv:
        git("fetch", "--quiet", "origin", "queue-state")
        git("fetch", "--quiet", "origin", "+refs/heads/nnue-*:refs/remotes/origin/nnue-*")
        for line in git("for-each-ref", "--format=%(refname:short) %(committerdate:iso-strict)",
                        "refs/remotes/origin/nnue-*").splitlines():
            ref, when = line.split()
            if when < PIVOT or ref.endswith("-wip") or ref.startswith("origin/nnue-scale"):
                continue
            base = git("merge-base", "origin/main", ref).strip()
            if git("diff", "--name-only", base, ref, "--", "net/nnue.bin").strip():
                branches.add(ref.split("/", 1)[1])
        for line in git("show", "origin/queue-state:log.jsonl").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            b = r.get("branch") or ""
            if b.startswith("nnue-") and not b.startswith("nnue-scale") and (r.get("logged_at") or "") >= PIVOT and r.get("mode") in NET_MODES:
                branches.add(b)
    else:
        branches = {b.strip() for b in git("branch", "--format=%(refname:short)", "--list", "nnue-*").split()}
    gap = missing(branches, rows)
    for b in gap:
        print(f"no training record: {b}")
    print(f"{len(branches) - len(gap)}/{len(branches)} net branches have a row")
    return 1 if gap else 0


if __name__ == "__main__":
    sys.exit(main())
