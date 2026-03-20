#!/usr/bin/env python3
import argparse
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineCfg:
    name: str
    cmd: list[str]


@dataclass(frozen=True)
class GameResult:
    game_index: int
    score_a: float
    plies: int
    reason: str
    opening: str
    opening_index: int
    a_white: bool
    base_position: str
    opening_moves: list[str]


@dataclass(frozen=True)
class SprtConfig:
    h0_elo: float
    h1_elo: float
    alpha: float
    beta: float
    min_games: int


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

    def bestmove(
        self,
        base_position: str,
        moves: list[str],
        movetime_ms: int | None,
        depth: int | None,
    ) -> str:
        pos = "position " + base_position
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


def score_to_result(score_a: float) -> str:
    if score_a > 0.75:
        return "1-0"
    if score_a < 0.25:
        return "0-1"
    return "1/2-1/2"


class JsonlLogger:
    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def play_game(
    white: UCIEngine,
    black: UCIEngine,
    base_position: str,
    opening_moves: list[str],
    movetime_ms: int | None,
    depth: int | None,
    max_plies: int,
) -> tuple[float, list[str], str]:
    white.new_game()
    black.new_game()
    moves: list[str] = list(opening_moves)

    for ply in range(max_plies):
        eng = white if ply % 2 == 0 else black
        mv = eng.bestmove(base_position, moves, movetime_ms, depth)
        if mv == "0000":
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


def configure_engine(engine: UCIEngine, raw_options: list[str]) -> None:
    for raw in raw_options:
        name, value = parse_option_kv(raw)
        if name:
            engine.set_option(name, value)
    engine.sync_ready()


def load_openings(path: str | None) -> list[str]:
    if not path:
        return ["startpos"]

    openings: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            openings.append(line)

    if not openings:
        raise ValueError(f"no usable openings found in {path}")
    return openings


def split_opening_spec(spec: str) -> tuple[str, list[str]]:
    if " moves " in spec:
        base, moves = spec.split(" moves ", 1)
        move_list = [mv for mv in moves.split() if mv]
    else:
        base = spec
        move_list = []
    return base.strip(), move_list


def split_game_indices(total_games: int, workers: int) -> list[list[int]]:
    if total_games <= 0:
        return []

    workers = max(1, min(workers, total_games))
    base = total_games // workers
    extra = total_games % workers
    batches: list[list[int]] = []
    start = 0
    for worker in range(workers):
        size = base + (1 if worker < extra else 0)
        if size <= 0:
            continue
        batches.append(list(range(start, start + size)))
        start += size
    return batches


def run_game_batch(
    game_indices: list[int],
    a_cfg: EngineCfg,
    b_cfg: EngineCfg,
    a_options: list[str],
    b_options: list[str],
    openings: list[str],
    movetime_ms: int | None,
    depth: int | None,
    max_plies: int,
) -> list[GameResult]:
    try:
        a = UCIEngine(a_cfg)
        b = UCIEngine(b_cfg)
    except FileNotFoundError as e:
        raise RuntimeError(f"failed to launch engine: {e}") from e

    results: list[GameResult] = []
    try:
        a.init()
        b.init()
        configure_engine(a, a_options)
        configure_engine(b, b_options)

        for game_index in game_indices:
            a_white = game_index % 2 == 0
            opening_index = (game_index // 2) % len(openings)
            opening = openings[opening_index]
            base_position, opening_moves = split_opening_spec(opening)
            try:
                if a_white:
                    score, moves, reason = play_game(
                        a,
                        b,
                        base_position,
                        opening_moves,
                        movetime_ms,
                        depth,
                        max_plies,
                    )
                else:
                    score, moves, reason = play_game(
                        b,
                        a,
                        base_position,
                        opening_moves,
                        movetime_ms,
                        depth,
                        max_plies,
                    )
                    score = 1.0 - score
            except RuntimeError as e:
                score, moves, reason = 0.0, [], f"error:{e}"

            results.append(
                GameResult(
                    game_index,
                    score,
                    len(moves),
                    reason,
                    opening,
                    opening_index,
                    a_white,
                    base_position,
                    opening_moves,
                )
            )
        return results
    finally:
        a.quit()
        b.quit()


def print_result_line(
    game_index: int,
    total_games: int,
    score_a: float,
    reason: str,
    plies: int,
    opening: str,
    wins: int,
    draws: int,
    losses: int,
) -> None:
    print(
        f"game {game_index + 1}/{total_games}: result={score_to_result(score_a)} reason={reason} plies={plies} "
        f"WDL={wins}/{draws}/{losses} opening={opening}",
        flush=True,
    )


def build_matchup(args: argparse.Namespace) -> tuple[EngineCfg, EngineCfg, list[str], list[str]]:
    if args.engine_a or args.engine_b:
        if not args.engine_a or not args.engine_b:
            raise ValueError("both --engine-a and --engine-b are required in generic engine mode")

        a_path = resolve_engine_path(args.engine_a)
        b_path = resolve_engine_path(args.engine_b)
        a_name = args.engine_a_name or os.path.basename(a_path)
        b_name = args.engine_b_name or os.path.basename(b_path)
        return (
            EngineCfg(a_name, [a_path]),
            EngineCfg(b_name, [b_path]),
            args.engine_a_option,
            args.engine_b_option,
        )

    if not args.stockfish:
        raise ValueError("either pass --engine-a/--engine-b or provide --stockfish")

    rust_path = resolve_engine_path(args.rustfish)
    if rust_path == args.rustfish and not os.path.exists(rust_path):
        debug_candidate = "./target/debug/rustfish"
        if os.path.exists(debug_candidate):
            rust_path = debug_candidate

    sf_path = resolve_engine_path(args.stockfish)
    return (
        EngineCfg("rustfish", [rust_path]),
        EngineCfg("stockfish", [sf_path]),
        args.rustfish_option,
        args.stockfish_option,
    )


def ensure_engine_exists(cfg: EngineCfg) -> None:
    path = cfg.cmd[0]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{cfg.name} binary not found: {path}")


def score_to_elo(score_frac: float) -> float:
    eps = 1e-9
    s = min(max(score_frac, eps), 1.0 - eps)
    return -400.0 * math.log10(1.0 / s - 1.0)


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def sprt_bounds(cfg: SprtConfig) -> tuple[float, float]:
    lower = math.log(cfg.beta / (1.0 - cfg.alpha))
    upper = math.log((1.0 - cfg.beta) / cfg.alpha)
    return lower, upper


def sprt_llr_increment(score: float, cfg: SprtConfig) -> float:
    mu0 = elo_to_score(cfg.h0_elo)
    mu1 = elo_to_score(cfg.h1_elo)
    sigma2 = 0.25
    return ((score - mu0) ** 2 - (score - mu1) ** 2) / (2.0 * sigma2)


def sprt_decision(games_played: int, llr: float, cfg: SprtConfig) -> str | None:
    if games_played < cfg.min_games:
        return None
    lower, upper = sprt_bounds(cfg)
    if llr <= lower:
        return "accept_h0"
    if llr >= upper:
        return "accept_h1"
    return None


def build_run_metadata(
    args: argparse.Namespace,
    a_cfg: EngineCfg,
    b_cfg: EngineCfg,
    a_options: list[str],
    b_options: list[str],
    openings: list[str],
) -> dict:
    return {
        "event": "run_start",
        "run_id": uuid.uuid4().hex,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": os.path.abspath(__file__),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "games": args.games,
        "movetime_ms": args.movetime,
        "depth": args.depth,
        "max_plies": args.max_plies,
        "parallel": args.parallel,
        "openings_file": args.openings_file,
        "openings_count": len(openings),
        "engine_a": {
            "name": a_cfg.name,
            "cmd": a_cfg.cmd,
            "options": a_options,
        },
        "engine_b": {
            "name": b_cfg.name,
            "cmd": b_cfg.cmd,
            "options": b_options,
        },
        "sprt": {
            "enabled": args.sprt_h0_elo is not None,
            "h0_elo": args.sprt_h0_elo,
            "h1_elo": args.sprt_h1_elo,
            "alpha": args.sprt_alpha,
            "beta": args.sprt_beta,
            "min_games": args.sprt_min_games,
        },
    }


def build_game_record(
    run_id: str,
    result: GameResult,
    total_games: int,
    wins: int,
    draws: int,
    losses: int,
    sprt_llr: float | None,
    sprt_state: str | None,
) -> dict:
    record = {
        "event": "game_result",
        "run_id": run_id,
        "game_index": result.game_index,
        "game_number": result.game_index + 1,
        "total_games": total_games,
        "score_a": result.score_a,
        "result": score_to_result(result.score_a),
        "reason": result.reason,
        "plies": result.plies,
        "opening": result.opening,
        "opening_index": result.opening_index,
        "a_white": result.a_white,
        "base_position": result.base_position,
        "opening_moves": result.opening_moves,
        "wins": wins,
        "draws": draws,
        "losses": losses,
    }
    if sprt_llr is not None:
        record["sprt_llr"] = sprt_llr
    if sprt_state is not None:
        record["sprt_state"] = sprt_state
    return record


def build_run_end_record(
    run_id: str,
    status: str,
    error: str | None,
    args: argparse.Namespace,
    a_cfg: EngineCfg,
    b_cfg: EngineCfg,
    wins: int,
    draws: int,
    losses: int,
    plies_list: list[int],
    sprt_cfg: SprtConfig | None,
    sprt_llr: float | None,
    sprt_state: str | None,
) -> dict:
    total = wins + draws + losses
    score_frac = (wins + 0.5 * draws) / total if total else 0.0
    record: dict = {
        "event": "run_end",
        "run_id": run_id,
        "status": status,
        "error": error,
        "games_requested": args.games,
        "games_played": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_frac": score_frac,
        "score_pct": score_frac * 100.0,
        "approx_elo_diff": score_to_elo(score_frac),
        "a_name": a_cfg.name,
        "b_name": b_cfg.name,
    }
    if sprt_cfg is not None:
        lower, upper = sprt_bounds(sprt_cfg)
        record.update(
            {
                "sprt_h0_elo": sprt_cfg.h0_elo,
                "sprt_h1_elo": sprt_cfg.h1_elo,
                "sprt_alpha": sprt_cfg.alpha,
                "sprt_beta": sprt_cfg.beta,
                "sprt_min_games": sprt_cfg.min_games,
                "sprt_llr": sprt_llr,
                "sprt_lower_bound": lower,
                "sprt_upper_bound": upper,
                "sprt_state": sprt_state or "continue",
            }
        )

    if plies_list:
        record.update(
            {
                "plies_avg": sum(plies_list) / len(plies_list),
                "plies_median": statistics.median(plies_list),
                "plies_min": min(plies_list),
                "plies_max": max(plies_list),
            }
        )

    if total >= 2:
        sample_scores = [1.0] * wins + [0.5] * draws + [0.0] * losses
        score_stdev = statistics.stdev(sample_scores)
        score_se = score_stdev / math.sqrt(total)
        ci_low = max(0.0, score_frac - 1.96 * score_se)
        ci_high = min(1.0, score_frac + 1.96 * score_se)
        record.update(
            {
                "score_ci_low": ci_low,
                "score_ci_high": ci_high,
                "approx_elo_ci_low": score_to_elo(ci_low),
                "approx_elo_ci_high": score_to_elo(ci_high),
            }
        )

    return record


def main() -> int:
    ap = argparse.ArgumentParser(description="No-deps UCI gauntlet runner")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--movetime", type=int, default=100)
    ap.add_argument("--depth", type=int, default=None, help="If set, use fixed search depth instead of movetime")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--parallel", type=int, default=1, help="Number of independent game workers")
    ap.add_argument(
        "--openings-file",
        default=None,
        help="Optional file of opening position specs, one per line",
    )
    ap.add_argument(
        "--jsonl",
        default=None,
        help="Optional JSONL file for per-game results and run metadata",
    )
    ap.add_argument("--sprt-h0-elo", type=float, default=None, help="SPRT null hypothesis Elo, e.g. -5")
    ap.add_argument("--sprt-h1-elo", type=float, default=None, help="SPRT alternative hypothesis Elo, e.g. 5")
    ap.add_argument("--sprt-alpha", type=float, default=0.05, help="SPRT type-I error rate")
    ap.add_argument("--sprt-beta", type=float, default=0.05, help="SPRT type-II error rate")
    ap.add_argument("--sprt-min-games", type=int, default=24, help="Minimum games before an SPRT stop")

    ap.add_argument("--rustfish", default="./target/release/rustfish")
    ap.add_argument("--stockfish", default=None)
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

    ap.add_argument("--engine-a", default=None, help="Generic engine A path for A/B testing")
    ap.add_argument("--engine-b", default=None, help="Generic engine B path for A/B testing")
    ap.add_argument("--engine-a-name", default=None, help="Display name for engine A")
    ap.add_argument("--engine-b-name", default=None, help="Display name for engine B")
    ap.add_argument(
        "--engine-a-option",
        action="append",
        default=[],
        help="UCI option for engine A, e.g. Hash=256",
    )
    ap.add_argument(
        "--engine-b-option",
        action="append",
        default=[],
        help="UCI option for engine B, e.g. Hash=256",
    )
    args = ap.parse_args()

    if args.games <= 0:
        print("error: --games must be positive", file=sys.stderr)
        return 2
    if (args.sprt_h0_elo is None) != (args.sprt_h1_elo is None):
        print("error: pass both --sprt-h0-elo and --sprt-h1-elo together", file=sys.stderr)
        return 2
    if args.sprt_h0_elo is not None:
        if args.sprt_h0_elo >= args.sprt_h1_elo:
            print("error: SPRT requires h0 < h1", file=sys.stderr)
            return 2
        if not (0.0 < args.sprt_alpha < 1.0 and 0.0 < args.sprt_beta < 1.0):
            print("error: SPRT alpha/beta must be between 0 and 1", file=sys.stderr)
            return 2
        if args.parallel != 1:
            print("info string SPRT forcing parallel=1 for strict sequential stopping", flush=True)
            args.parallel = 1

    try:
        a_cfg, b_cfg, a_options, b_options = build_matchup(args)
        ensure_engine_exists(a_cfg)
        ensure_engine_exists(b_cfg)
        openings = load_openings(args.openings_file)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    workers = max(1, min(args.parallel, args.games))
    batches = split_game_indices(args.games, workers)
    wins = draws = losses = 0
    plies_list: list[int] = []
    pending: dict[int, GameResult] = {}
    next_to_print = 0
    sprt_cfg = None
    sprt_llr = 0.0
    sprt_state: str | None = None
    if args.sprt_h0_elo is not None:
        sprt_cfg = SprtConfig(
            h0_elo=args.sprt_h0_elo,
            h1_elo=args.sprt_h1_elo,
            alpha=args.sprt_alpha,
            beta=args.sprt_beta,
            min_games=args.sprt_min_games,
        )
    logger: JsonlLogger | None = None
    run_id = ""
    if args.jsonl:
        try:
            logger = JsonlLogger(args.jsonl)
        except OSError as e:
            print(f"error: unable to open JSONL log {args.jsonl}: {e}", file=sys.stderr)
            return 2
        run_meta = build_run_metadata(args, a_cfg, b_cfg, a_options, b_options, openings)
        run_id = run_meta["run_id"]
        logger.write(run_meta)

    exit_code = 0
    error_message: str | None = None

    def consume_result(current: GameResult) -> bool:
        nonlocal wins, draws, losses
        if current.score_a > 0.75:
            wins += 1
        elif current.score_a < 0.25:
            losses += 1
        else:
            draws += 1
        plies_list.append(current.plies)
        print_result_line(
            current.game_index,
            args.games,
            current.score_a,
            current.reason,
            current.plies,
            current.opening,
            wins,
            draws,
            losses,
        )
        if sprt_cfg is not None:
            sprt_llr_nonlocal = sprt_llr_increment(current.score_a, sprt_cfg)
            nonlocal_sprt_llr[0] += sprt_llr_nonlocal
            current_games = wins + draws + losses
            state = sprt_decision(current_games, nonlocal_sprt_llr[0], sprt_cfg)
            if state is not None:
                nonlocal_sprt_state[0] = state
        if logger:
            logger.write(
                build_game_record(
                    run_id,
                    current,
                    args.games,
                    wins,
                    draws,
                    losses,
                    nonlocal_sprt_llr[0] if sprt_cfg is not None else None,
                    nonlocal_sprt_state[0],
                )
            )
        return nonlocal_sprt_state[0] is not None

    nonlocal_sprt_llr = [sprt_llr]
    nonlocal_sprt_state = [sprt_state]

    try:
        if workers == 1:
            for result in run_game_batch(
                batches[0],
                a_cfg,
                b_cfg,
                a_options,
                b_options,
                openings,
                args.movetime,
                args.depth,
                args.max_plies,
            ):
                pending[result.game_index] = result
                while next_to_print in pending:
                    stop = consume_result(pending.pop(next_to_print))
                    next_to_print += 1
                    if stop:
                        break
                if nonlocal_sprt_state[0] is not None:
                    break
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        run_game_batch,
                        batch,
                        a_cfg,
                        b_cfg,
                        a_options,
                        b_options,
                        openings,
                        args.movetime,
                        args.depth,
                        args.max_plies,
                    )
                    for batch in batches
                ]
                for future in as_completed(futures):
                    for result in future.result():
                        pending[result.game_index] = result
                    while next_to_print in pending:
                        stop = consume_result(pending.pop(next_to_print))
                        next_to_print += 1
                        if stop:
                            break
                    if nonlocal_sprt_state[0] is not None:
                        break
    except RuntimeError as e:
        error_message = str(e)
        print(f"error: {e}", file=sys.stderr)
        exit_code = 2
    finally:
        sprt_llr = nonlocal_sprt_llr[0]
        sprt_state = nonlocal_sprt_state[0]
        if logger:
            try:
                logger.write(
                    build_run_end_record(
                        run_id,
                        "ok" if exit_code == 0 else "error",
                        error_message,
                        args,
                        a_cfg,
                        b_cfg,
                        wins,
                        draws,
                        losses,
                        plies_list,
                        sprt_cfg,
                        sprt_llr if sprt_cfg is not None else None,
                        sprt_state,
                    )
                )
            finally:
                logger.close()

    if exit_code != 0:
        return exit_code

    total = wins + draws + losses
    score_frac = (wins + 0.5 * draws) / total if total else 0.0
    pct = score_frac * 100.0
    print(f"\nFinal WDL: {wins}/{draws}/{losses}  score={pct:.1f}%")
    if sprt_cfg is not None:
        lower, upper = sprt_bounds(sprt_cfg)
        print(f"SPRT: llr={sprt_llr:.3f} bounds={lower:.3f}..{upper:.3f} state={sprt_state or 'continue'}")

    if plies_list:
        print(
            "Plies: "
            f"avg={sum(plies_list)/len(plies_list):.1f} "
            f"median={statistics.median(plies_list):.1f} "
            f"min={min(plies_list)} max={max(plies_list)}"
        )

    if total >= 2:
        sample_scores = [1.0] * wins + [0.5] * draws + [0.0] * losses
        score_stdev = statistics.stdev(sample_scores)
        score_se = score_stdev / math.sqrt(total)
        ci_low = max(0.0, score_frac - 1.96 * score_se)
        ci_high = min(1.0, score_frac + 1.96 * score_se)
        print(f"Score 95% CI: {ci_low * 100:.1f}% .. {ci_high * 100:.1f}%")
        print(
            f"Approx Elo 95% CI ({a_cfg.name} - {b_cfg.name}): "
            f"{score_to_elo(ci_low):.1f} .. {score_to_elo(ci_high):.1f}"
        )

    elo_diff = score_to_elo(score_frac)
    print(f"Approx Elo diff ({a_cfg.name} - {b_cfg.name}): {elo_diff:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
