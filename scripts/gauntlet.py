#!/usr/bin/env python3
import argparse
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class EngineCfg:
    name: str
    cmd: list[str]


class UCIEngine:
    def __init__(self, cfg: EngineCfg):
        self.cfg = cfg
        self.p = subprocess.Popen(
            cfg.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _send(self, line: str) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def _read_until(self, prefix: str, timeout: float = 30.0) -> str:
        assert self.p.stdout is not None
        end = time.time() + timeout
        while time.time() < end:
            line = self.p.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith(prefix):
                return line
        raise RuntimeError(f"{self.cfg.name}: timeout waiting for {prefix}")

    def init(self) -> None:
        self._send("uci")
        self._read_until("uciok")
        self._send("isready")
        self._read_until("readyok")

    def set_option(self, name: str, value: str | None = None) -> None:
        if value is None or value == "":
            self._send(f"setoption name {name}")
        else:
            self._send(f"setoption name {name} value {value}")

    def sync_ready(self) -> None:
        self._send("isready")
        self._read_until("readyok")

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def bestmove(self, moves: list[str], movetime_ms: int | None, depth: int | None) -> str:
        pos = "position startpos"
        if moves:
            pos += " moves " + " ".join(moves)
        self._send(pos)

        if depth is not None:
            self._send(f"go depth {depth}")
            timeout = 15.0
        else:
            assert movetime_ms is not None
            self._send(f"go movetime {movetime_ms}")
            timeout = max(5.0, movetime_ms / 1000.0 + 10.0)

        line = self._read_until("bestmove", timeout=timeout)
        parts = line.split()
        if len(parts) < 2:
            return "0000"
        mv = parts[1]
        if mv in {"(none)", "none", "0000"}:
            return "0000"
        return mv

    def quit(self) -> None:
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self.p.terminate()
        except Exception:
            pass


def play_game(
    white: UCIEngine,
    black: UCIEngine,
    movetime_ms: int | None,
    depth: int | None,
    max_plies: int,
) -> tuple[float, list[str], str]:
    white.new_game()
    black.new_game()
    moves: list[str] = []

    for ply in range(max_plies):
        eng = white if ply % 2 == 0 else black
        mv = eng.bestmove(moves, movetime_ms, depth)
        if mv == "0000":
            # side to move has no legal move; we don't distinguish mate/stalemate here
            if ply % 2 == 0:
                return 0.0, moves, "white_no_move"
            return 1.0, moves, "black_no_move"
        moves.append(mv)

    return 0.5, moves, "max_plies"


def resolve_engine_path(path_or_name: str) -> str:
    if os.path.isfile(path_or_name) and os.access(path_or_name, os.X_OK):
        return path_or_name
    found = shutil.which(path_or_name)
    if found:
        return found
    return path_or_name


def parse_option_kv(raw: str) -> tuple[str, str | None]:
    if "=" in raw:
        k, v = raw.split("=", 1)
        return k.strip(), v.strip()
    return raw.strip(), None


def main() -> int:
    ap = argparse.ArgumentParser(description="No-deps UCI gauntlet runner")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--movetime", type=int, default=100)
    ap.add_argument("--depth", type=int, default=None, help="If set, use fixed search depth instead of movetime")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--rustfish", default="./target/release/rustfish")
    ap.add_argument("--stockfish", required=True)
    ap.add_argument(
        "--stockfish-option",
        action="append",
        default=[],
        help="UCI option for Stockfish, e.g. Skill Level=5 or UCI_Elo=1800",
    )
    ap.add_argument(
        "--rustfish-option",
        action="append",
        default=[],
        help="UCI option for rustfish, e.g. Hash=128",
    )
    args = ap.parse_args()

    rust_path = resolve_engine_path(args.rustfish)
    if rust_path == args.rustfish and not os.path.exists(rust_path):
        debug_candidate = "./target/debug/rustfish"
        if os.path.exists(debug_candidate):
            rust_path = debug_candidate

    sf_path = resolve_engine_path(args.stockfish)

    if not os.path.exists(rust_path):
        print(
            f"error: rustfish binary not found: {rust_path}\n"
            f"hint: run `cargo build --release` or pass --rustfish ./target/debug/rustfish",
            file=sys.stderr,
        )
        return 2

    if not os.path.exists(sf_path):
        print(
            f"error: stockfish binary not found: {sf_path}\n"
            f"hint: install stockfish and/or pass --stockfish /full/path/to/stockfish",
            file=sys.stderr,
        )
        return 2

    rust_cfg = EngineCfg("rustfish", [rust_path])
    sf_cfg = EngineCfg("stockfish", [sf_path])

    try:
        a = UCIEngine(rust_cfg)
        b = UCIEngine(sf_cfg)
    except FileNotFoundError as e:
        print(f"error: failed to launch engine: {e}", file=sys.stderr)
        return 2

    try:
        a.init()
        b.init()

        for raw in args.rustfish_option:
            name, value = parse_option_kv(raw)
            if name:
                a.set_option(name, value)
        a.sync_ready()

        for raw in args.stockfish_option:
            name, value = parse_option_kv(raw)
            if name:
                b.set_option(name, value)
        b.sync_ready()

        wins = draws = losses = 0
        plies_list: list[int] = []
        for g in range(args.games):
            rust_white = (g % 2 == 0)
            try:
                if rust_white:
                    score, moves, reason = play_game(a, b, args.movetime, args.depth, args.max_plies)
                else:
                    score, moves, reason = play_game(b, a, args.movetime, args.depth, args.max_plies)
                    score = 1.0 - score
            except RuntimeError as e:
                score, moves, reason = 0.0, [], f"error:{e}"

            if score > 0.75:
                wins += 1
                res = "1-0"
            elif score < 0.25:
                losses += 1
                res = "0-1"
            else:
                draws += 1
                res = "1/2-1/2"

            plies_list.append(len(moves))

            print(
                f"game {g+1}/{args.games}: result={res} reason={reason} plies={len(moves)} "
                f"WDL={wins}/{draws}/{losses}",
                flush=True,
            )

        total = args.games
        score_frac = (wins + 0.5 * draws) / total if total else 0.0
        pct = score_frac * 100.0
        print(f"\nFinal WDL: {wins}/{draws}/{losses}  score={pct:.1f}%")

        if plies_list:
            print(
                "Plies: "
                f"avg={sum(plies_list)/len(plies_list):.1f} "
                f"median={statistics.median(plies_list):.1f} "
                f"min={min(plies_list)} max={max(plies_list)}"
            )

        eps = 1e-9
        s = min(max(score_frac, eps), 1.0 - eps)
        elo_diff = -400.0 * math.log10(1.0 / s - 1.0)
        print(f"Approx Elo diff (rustfish - opponent): {elo_diff:.1f}")
        return 0
    finally:
        a.quit()
        b.quit()


if __name__ == "__main__":
    sys.exit(main())
