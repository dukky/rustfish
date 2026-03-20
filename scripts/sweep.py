#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAUNTLET = os.path.join(ROOT, "scripts", "gauntlet.py")


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def parse_variant(raw: str) -> tuple[str, list[str]]:
    if ":" not in raw:
        return raw.strip(), []
    name, opts = raw.split(":", 1)
    options = [opt.strip() for opt in opts.split(",") if opt.strip()]
    return name.strip(), options


def load_run_end(path: str) -> dict:
    last = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = json.loads(line)
    if not isinstance(last, dict) or last.get("event") != "run_end":
        raise RuntimeError(f"missing run_end record in {path}")
    return last


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep candidate variants against a fixed baseline using gauntlet.py")
    ap.add_argument("--baseline", required=True, help="Baseline engine path")
    ap.add_argument("--candidate", required=True, help="Candidate engine path")
    ap.add_argument("--baseline-name", default="baseline")
    ap.add_argument("--candidate-name", default="candidate")
    ap.add_argument("--baseline-option", action="append", default=[])
    ap.add_argument("--candidate-option", action="append", default=[])
    ap.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant spec: NAME:Opt=Val,Other=Val . Can be repeated.",
    )
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--movetime", type=int, default=30)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--max-plies", type=int, default=180)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--openings-file", default=None)
    ap.add_argument("--jsonl-dir", default=os.path.join(ROOT, "tmp", "sweeps"))
    ap.add_argument("--no-sprt", action="store_true", help="Disable SPRT and run fixed-length matches")
    ap.add_argument("--sprt-h0-elo", type=float, default=-5.0)
    ap.add_argument("--sprt-h1-elo", type=float, default=5.0)
    ap.add_argument("--sprt-alpha", type=float, default=0.05)
    ap.add_argument("--sprt-beta", type=float, default=0.05)
    ap.add_argument("--sprt-min-games", type=int, default=24)
    args = ap.parse_args()

    variants = [parse_variant(raw) for raw in args.variant] or [("default", [])]
    os.makedirs(args.jsonl_dir, exist_ok=True)

    summary: list[dict] = []
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    for name, variant_options in variants:
        jsonl_path = os.path.join(args.jsonl_dir, f"{stamp}_{name}.jsonl")
        cmd = [
            sys.executable,
            GAUNTLET,
            "--engine-a",
            args.candidate,
            "--engine-b",
            args.baseline,
            "--engine-a-name",
            name,
            "--engine-b-name",
            args.baseline_name,
            "--games",
            str(args.games),
            "--max-plies",
            str(args.max_plies),
            "--parallel",
            str(args.parallel),
            "--jsonl",
            jsonl_path,
        ]
        if not args.no_sprt:
            cmd.extend(
                [
                    "--sprt-h0-elo",
                    str(args.sprt_h0_elo),
                    "--sprt-h1-elo",
                    str(args.sprt_h1_elo),
                    "--sprt-alpha",
                    str(args.sprt_alpha),
                    "--sprt-beta",
                    str(args.sprt_beta),
                    "--sprt-min-games",
                    str(args.sprt_min_games),
                ]
            )
        if args.depth is not None:
            cmd.extend(["--depth", str(args.depth)])
        else:
            cmd.extend(["--movetime", str(args.movetime)])
        if args.openings_file:
            cmd.extend(["--openings-file", args.openings_file])
        for opt in args.candidate_option + variant_options:
            cmd.extend(["--engine-a-option", opt])
        for opt in args.baseline_option:
            cmd.extend(["--engine-b-option", opt])

        code = run(cmd)
        if code != 0:
            return code
        result = load_run_end(jsonl_path)
        result["variant_name"] = name
        result["jsonl"] = jsonl_path
        summary.append(result)

    summary.sort(key=lambda row: row.get("approx_elo_diff", float("-inf")), reverse=True)
    print("\nSweep Summary")
    for row in summary:
        print(
            f"{row['variant_name']}: "
            f"score={row.get('score_pct', 0.0):.1f}% "
            f"elo={row.get('approx_elo_diff', 0.0):.1f} "
            f"games={row.get('games_played', 0)} "
            f"sprt={row.get('sprt_state', 'n/a')} "
            f"log={row['jsonl']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
