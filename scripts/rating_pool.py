#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAUNTLET = os.path.join(ROOT, "scripts", "gauntlet.py")


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def parse_engine(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"invalid --engine {raw!r}; expected NAME=PATH")
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"invalid --engine {raw!r}; expected NAME=PATH")
    return name, path


def parse_engine_option(raw: str) -> tuple[str, str]:
    if ":" not in raw:
        raise ValueError(f"invalid --engine-option {raw!r}; expected NAME:Option=Value")
    name, option = raw.split(":", 1)
    name = name.strip()
    option = option.strip()
    if not name or not option:
        raise ValueError(f"invalid --engine-option {raw!r}; expected NAME:Option=Value")
    return name, option


def parse_fixed_rating(raw: str) -> tuple[str, float]:
    if "=" not in raw:
        raise ValueError(f"invalid --fixed-rating {raw!r}; expected NAME=ELO")
    name, value = raw.split("=", 1)
    name = name.strip()
    try:
        rating = float(value.strip())
    except ValueError as e:
        raise ValueError(f"invalid --fixed-rating {raw!r}; expected numeric ELO") from e
    if not name:
        raise ValueError(f"invalid --fixed-rating {raw!r}; expected NAME=ELO")
    return name, rating


def load_run_end(path: str) -> dict:
    last = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = json.loads(line)
    if not isinstance(last, dict) or last.get("event") != "run_end":
        raise RuntimeError(f"missing run_end record in {path}")
    return last


def gaussian_solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-9:
            raise RuntimeError("rating system is singular; check that the pool is connected")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_val = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_val

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def solve_pool_ratings(
    engine_names: list[str],
    matches: list[dict],
    fixed_ratings: dict[str, float],
) -> tuple[dict[str, float], float]:
    variable_names = [name for name in engine_names if name not in fixed_ratings]
    if not variable_names:
        return dict(fixed_ratings), 0.0

    idx = {name: i for i, name in enumerate(variable_names)}
    dim = len(variable_names)
    normal = [[0.0 for _ in range(dim)] for _ in range(dim)]
    rhs = [0.0 for _ in range(dim)]

    for match in matches:
        a_name = match["a_name"]
        b_name = match["b_name"]
        games = max(1, int(match.get("games_played", 0)))
        observed = float(match.get("approx_elo_diff", 0.0))

        row = [0.0 for _ in range(dim)]
        known = 0.0
        if a_name in fixed_ratings:
            known += fixed_ratings[a_name]
        else:
            row[idx[a_name]] += 1.0
        if b_name in fixed_ratings:
            known -= fixed_ratings[b_name]
        else:
            row[idx[b_name]] -= 1.0

        target = observed - known
        for i in range(dim):
            if row[i] == 0.0:
                continue
            rhs[i] += games * row[i] * target
            for j in range(dim):
                if row[j] != 0.0:
                    normal[i][j] += games * row[i] * row[j]

    solution = gaussian_solve(normal, rhs)
    ratings = dict(fixed_ratings)
    for name, value in zip(variable_names, solution):
        ratings[name] = value

    total_weight = 0.0
    weighted_sq = 0.0
    for match in matches:
        expected = ratings[match["a_name"]] - ratings[match["b_name"]]
        observed = float(match.get("approx_elo_diff", 0.0))
        weight = max(1, int(match.get("games_played", 0)))
        err = expected - observed
        weighted_sq += weight * err * err
        total_weight += weight
    rmse = (weighted_sq / total_weight) ** 0.5 if total_weight else 0.0
    return ratings, rmse


def main() -> int:
    ap = argparse.ArgumentParser(description="Round-robin local rating pool using gauntlet.py pairwise matches")
    ap.add_argument(
        "--engine",
        action="append",
        default=[],
        help="Engine spec NAME=PATH. Repeat for each engine in the pool.",
    )
    ap.add_argument(
        "--engine-option",
        action="append",
        default=[],
        help="Per-engine UCI option NAME:Option=Value. Repeat as needed.",
    )
    ap.add_argument(
        "--common-option",
        action="append",
        default=[],
        help="UCI option applied to every engine, e.g. OwnBook=false",
    )
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--movetime", type=int, default=50)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--max-plies", type=int, default=220)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--openings-file", default=os.path.join(ROOT, "scripts", "openings.txt"))
    ap.add_argument("--jsonl-dir", default=os.path.join(ROOT, "tmp", "rating-pool"))
    ap.add_argument(
        "--fixed-rating",
        action="append",
        default=[],
        help="Fix a pool member to a rating: NAME=ELO. Repeatable.",
    )
    ap.add_argument("--reuse-existing", action="store_true", help="Reuse existing JSONL logs if present")
    args = ap.parse_args()

    if len(args.engine) < 2:
        print("error: pass at least two --engine NAME=PATH entries", file=sys.stderr)
        return 2

    try:
        engines = [parse_engine(raw) for raw in args.engine]
        per_engine_options: dict[str, list[str]] = {name: list(args.common_option) for name, _ in engines}
        for raw in args.engine_option:
            name, option = parse_engine_option(raw)
            if name not in per_engine_options:
                raise ValueError(f"unknown engine {name!r} in --engine-option")
            per_engine_options[name].append(option)
        fixed_ratings = {}
        for raw in args.fixed_rating:
            name, rating = parse_fixed_rating(raw)
            fixed_ratings[name] = rating
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    engine_names = [name for name, _ in engines]
    engine_paths = {name: path for name, path in engines}
    for name in fixed_ratings:
        if name not in engine_paths:
            print(f"error: unknown engine {name!r} in --fixed-rating", file=sys.stderr)
            return 2
    if not fixed_ratings:
        fixed_ratings = {engine_names[0]: 0.0}

    os.makedirs(args.jsonl_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    matches: list[dict] = []

    for i in range(len(engine_names)):
        for j in range(i + 1, len(engine_names)):
            a_name = engine_names[i]
            b_name = engine_names[j]
            log_name = f"{stamp}_{sanitize_name(a_name)}_vs_{sanitize_name(b_name)}.jsonl"
            jsonl_path = os.path.join(args.jsonl_dir, log_name)

            if args.reuse_existing and os.path.exists(jsonl_path):
                result = load_run_end(jsonl_path)
            else:
                cmd = [
                    sys.executable,
                    GAUNTLET,
                    "--engine-a",
                    engine_paths[a_name],
                    "--engine-b",
                    engine_paths[b_name],
                    "--engine-a-name",
                    a_name,
                    "--engine-b-name",
                    b_name,
                    "--games",
                    str(args.games),
                    "--max-plies",
                    str(args.max_plies),
                    "--parallel",
                    str(args.parallel),
                    "--openings-file",
                    args.openings_file,
                    "--jsonl",
                    jsonl_path,
                ]
                if args.depth is not None:
                    cmd.extend(["--depth", str(args.depth)])
                else:
                    cmd.extend(["--movetime", str(args.movetime)])
                for option in per_engine_options[a_name]:
                    cmd.extend(["--engine-a-option", option])
                for option in per_engine_options[b_name]:
                    cmd.extend(["--engine-b-option", option])

                code = run(cmd)
                if code != 0:
                    return code
                result = load_run_end(jsonl_path)

            result["jsonl"] = jsonl_path
            matches.append(result)

    ratings, rmse = solve_pool_ratings(engine_names, matches, fixed_ratings)
    rows = [
        {
            "name": name,
            "rating": ratings[name],
            "games": sum(
                int(match.get("games_played", 0))
                for match in matches
                if match["a_name"] == name or match["b_name"] == name
            ),
        }
        for name in engine_names
    ]
    rows.sort(key=lambda row: row["rating"], reverse=True)

    print("\nPairwise Results")
    for match in matches:
        print(
            f"{match['a_name']} vs {match['b_name']}: "
            f"score={match.get('score_pct', 0.0):.1f}% "
            f"elo={match.get('approx_elo_diff', 0.0):.1f} "
            f"games={match.get('games_played', 0)} "
            f"log={match['jsonl']}"
        )

    print("\nLocal Rating Pool")
    fixed_text = ", ".join(f"{name}={rating:.1f}" for name, rating in fixed_ratings.items())
    print(f"fixed={fixed_text}  fit_rmse={rmse:.1f} elo")
    for row in rows:
        print(f"{row['name']}: {row['rating']:.1f}  games={row['games']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
