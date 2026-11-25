#!/usr/bin/env python3
"""Interleaved bench-NPS comparison of rustfish binaries that must search
identically (exact speedups).

    scripts/speed-compare.py --base main=./main cand1=./c1 [cand2=./c2 ...]
        [--rounds 12] [--depth 13] [--env NAME=VALUE] [--json out.json]

Every candidate must give the base's bench node count and `evalhash` output;
a mismatching one is reported and dropped. Each round runs every binary once,
in an order rotated per round, so drift on a shared runner hits all binaries
alike. Reports per candidate the median and mean of the per-round paired
ratios cand/base (the base run of the same round).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys


def run(binary, cmds, env):
    out = subprocess.run([binary], input="\n".join(cmds + ["quit"]) + "\n", capture_output=True, text=True,
                         env=env, check=True).stdout
    return out


def field(out, key):
    for line in out.splitlines():
        if line.startswith(key):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"no '{key}' in output:\n{out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("cands", nargs="+")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--depth", type=int, default=13)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--json")
    ap.add_argument("--label", default="")
    ap.add_argument("--allow-diff", action="store_true",
                    help="search changes: skip the identity check, also report time-to-depth and nodes")
    a = ap.parse_args()
    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    bins = [tuple(x.split("=", 1)) for x in [a.base] + a.cands]
    base = bins[0][0]

    ident = {}
    for name, path in bins:
        b = run(path, ["bench"], env)
        e = run(path, ["evalhash"], env)
        eh = " ".join(l.split(":", 1)[1].strip() for l in e.splitlines() if "hash" in l.lower() and ":" in l)
        ident[name] = (field(b, "Nodes searched"), eh, field(b, "NNUE SIMD"))
        print(f"{name}: bench {ident[name][0]} evalhash {eh} simd {ident[name][2]}", flush=True)
    same = (lambda n: True) if a.allow_diff else (lambda n: ident[n][:2] == ident[base][:2])
    ok = [(n, p) for n, p in bins if same(n)]
    bad = [n for n, _ in bins if not same(n)]
    for n in bad:
        print(f"MISMATCH: {n} does not search identically to {base}", flush=True)

    nps = {n: [] for n, _ in ok}
    ms = {n: [] for n, _ in ok}
    nodes = {}
    for r in range(a.rounds):
        # Plain rotation: over len(ok) rounds every binary runs once in every
        # position (use a multiple of the binary count as --rounds).
        k = r % len(ok)
        order = ok[k:] + ok[:k]
        for n, p in order:
            out = run(p, [f"bench {a.depth}"], env)
            nps[n].append(int(field(out, "Nodes/second")))
            ms[n].append(int(field(out, "Total time (ms)")))
            nodes[n] = int(field(out, "Nodes searched"))
        print(f"round {r + 1}: " + " ".join(f"{n}={nps[n][-1]}" for n, _ in ok), flush=True)

    res = {"label": a.label, "simd": ident[base][2], "bench": ident[base][0], "mismatch": bad,
           "base_nps_median": statistics.median(nps[base]), "cands": {}}
    for n, _ in ok:
        if n == base:
            continue
        ratios = [c / b - 1 for c, b in zip(nps[n], nps[base])]
        tr = [c / b - 1 for c, b in zip(ms[n], ms[base])]
        res["cands"][n] = {"median": statistics.median(ratios), "mean": statistics.fmean(ratios),
                           "min": min(ratios), "max": max(ratios), "nps_median": statistics.median(nps[n]),
                           "time_median": statistics.median(tr), "nodes": nodes[n], "base_nodes": nodes[base]}
        print(f"{a.label} {n} vs {base}: time-to-depth {a.depth} median {statistics.median(tr) * 100:+.2f}% "
              f"nodes {nodes[n]} vs {nodes[base]} ({(nodes[n] / nodes[base] - 1) * 100:+.2f}%)", flush=True)
        print(f"{a.label} {n} vs {base}: median {res['cands'][n]['median'] * 100:+.2f}% "
              f"mean {res['cands'][n]['mean'] * 100:+.2f}% "
              f"[{res['cands'][n]['min'] * 100:+.1f}, {res['cands'][n]['max'] * 100:+.1f}] "
              f"(base {res['base_nps_median']:.0f} nps)", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
