#!/usr/bin/env python3
"""Recompute the CI of logged CCRL ladders with the current method (read-only).

Rows logged before the pentanomial CI carry only per-anchor W/D/L, so their
pairs are not recoverable from the log. With --download the run's
ccrl-ladder-results artifact (all-games.pgn; GitHub keeps it 90 days) is
fetched with `gh run download` and pairs are rebuilt from the PGN; otherwise
the row falls back to a trinomial CI (per-game draw-aware variance, no pair
correlation) plus the anchors' CCRL errors. tests/ladder-log.jsonl is never
modified.

  python3 scripts/ladder_recompute.py --last 2 --download tmp/ladder-pgns
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ccrl_ladder as cl  # noqa: E402

# Broken anchor builds the logged ratings were already refit without
# (rows keep their games, plus rating_with_black_marlin).
EXCLUDED = {"black-marlin-4.0"}


def anchor_errors():
    """ccrl_err by anchor name from every scripts/anchors*.json (None if absent)."""
    errs = {}
    for path in sorted(glob.glob(os.path.join(HERE, "anchors*.json"))):
        with open(path) as fh:
            for a in json.load(fh)["anchors"]:
                errs[a["name"]] = a.get("ccrl_err") or errs.get(a["name"])
    return errs


def fetch_pgn(row, out_dir):
    run_id = (row.get("url") or "").rstrip("/").rsplit("/", 1)[-1]
    if not run_id.isdigit():
        return None
    dest = os.path.join(out_dir, run_id)
    pgn = os.path.join(dest, "all-games.pgn")
    if not os.path.exists(pgn):
        os.makedirs(dest, exist_ok=True)
        r = subprocess.run(["gh", "run", "download", run_id, "-R", "dukky/rustfish",
                            "-n", "ccrl-ladder-results", "-D", dest], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(pgn):
            return None
    return pgn


def recompute(row, errs, pgn=None, engine="rustfish"):
    """Return (analyse() result, method note) for one log row."""
    ccrl = {a["anchor"]: a["ccrl"] for a in row["anchors"]}
    if row["anchors"] and row["anchors"][0].get("penta") is not None:
        rows = row["anchors"]
        note = "logged pairs"
    elif pgn:
        stats, _ = cl.collect([pgn], engine, ccrl)
        rows = []
        for name, s in stats.items():
            w, d, l = s["wdl"]
            n = w + d + l
            rows.append({"anchor": name, "ccrl": ccrl[name], "wins": w, "draws": d, "losses": l,
                         "games": n, "score": (w + 0.5 * d) / n, "penta": s["penta"],
                         "unpaired": s["unpaired"]})
        note = "pairs from PGN"
    else:
        rows = [dict(a) for a in row["anchors"] if a["games"]]
        note = "trinomial (no pairs)"
    rows = [r for r in rows if r["anchor"] not in EXCLUDED]
    for r in rows:
        if not r.get("ccrl_err"):
            r["ccrl_err"] = errs.get(r["anchor"])
    return cl.analyse(rows), note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(HERE, "..", "tests", "ladder-log.jsonl"))
    ap.add_argument("--last", type=int, default=0, help="only the last N rows (default all)")
    ap.add_argument("--download", metavar="DIR", help="fetch each run's PGNs into DIR via gh")
    args = ap.parse_args()

    with open(args.log) as fh:
        log = [json.loads(line) for line in fh if line.strip()]
    if args.last:
        log = log[-args.last:]
    errs = anchor_errors()
    print(f"{'sha':<8} {'logged':<16} {'games':>5} {'rating':>7} {'logged CI':>18} {'new':>7} "
          f"{'new CI (+-)':>22} {'binomial (+-)':>22}  method")
    for row in log:
        pgn = fetch_pgn(row, args.download) if args.download else None
        res, note = recompute(row, errs, pgn)
        lo, hi = row["ci"]
        nlo, nhi = res["ci"]
        blo, bhi = res["ci_binomial"]
        print(f"{row['sha'][:7]:<8} {row['logged_at']:<16} {row['games']:>5} {row['rating']:>7.1f} "
              f"{lo:>8.1f}..{hi:<8.1f} {res['rating']:>7.1f} {nlo:>7.1f}..{nhi:<7.1f}({(nhi - nlo) / 2:4.1f}) "
              f"{blo:>7.1f}..{bhi:<7.1f}({(bhi - blo) / 2:4.1f})  {note}")


if __name__ == "__main__":
    main()
