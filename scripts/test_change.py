#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAUNTLET = os.path.join(ROOT, "scripts", "gauntlet.py")
BENCH = os.path.join(ROOT, "scripts", "bench.py")
OPENINGS = os.path.join(ROOT, "scripts", "openings.txt")


def run_step(name: str, cmd: list[str], allow_fail: bool = False) -> int:
    print(f"\n== {name} ==", flush=True)
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(proc.returncode)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="One-command correctness/performance/benchmark validation")
    ap.add_argument("--baseline", default=None, help="Optional frozen baseline engine for A/B smoke")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--movetime", type=int, default=20)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--depth", type=int, default=5, help="Bench depth")
    ap.add_argument("--hash", type=int, default=256)
    ap.add_argument("--jsonl-dir", default=os.path.join(ROOT, "tmp", "test-change"))
    ap.add_argument("--openings-file", default=OPENINGS)
    args = ap.parse_args()

    os.makedirs(args.jsonl_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    run_step("cargo check", ["cargo", "check"])
    run_step("perfttest", ["/bin/zsh", "-lc", "printf 'perfttest\\nquit\\n' | cargo run --quiet"])
    run_step("selftest", ["/bin/zsh", "-lc", "printf 'selftest 100\\nquit\\n' | cargo run --quiet"])
    run_step("release build", ["cargo", "build", "--release"])
    run_step(
        "uci smoke",
        ["/bin/zsh", "-lc", "printf 'uci\\nisready\\nposition startpos\\ngo depth 4\\nquit\\n' | ./target/release/rustfish"],
    )
    run_step("bench", [sys.executable, BENCH, "--engine", "./target/release/rustfish", "--depth", str(args.depth)])

    if args.baseline:
        jsonl_path = os.path.join(args.jsonl_dir, f"{stamp}_ab_smoke.jsonl")
        gauntlet_cmd = [
            sys.executable,
            GAUNTLET,
            "--engine-a",
            "./target/release/rustfish",
            "--engine-b",
            args.baseline,
            "--engine-a-name",
            "candidate",
            "--engine-b-name",
            "baseline",
            "--engine-a-option",
            "OwnBook=false",
            "--engine-b-option",
            "OwnBook=false",
            "--engine-a-option",
            f"Hash={args.hash}",
            "--engine-b-option",
            f"Hash={args.hash}",
            "--games",
            str(args.games),
            "--movetime",
            str(args.movetime),
            "--max-plies",
            str(args.max_plies),
            "--parallel",
            str(args.parallel),
            "--jsonl",
            jsonl_path,
        ]
        if args.openings_file:
            gauntlet_cmd.extend(["--openings-file", args.openings_file])
        run_step("A/B smoke", gauntlet_cmd)
        print(f"\nA/B JSONL: {jsonl_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
