#!/usr/bin/env python3
"""Sample positions from PGN files for the policy-order prototype.

Output is JSONL, one row per sampled position:
  {"game_id": "...", "ply": 24, "fen": "..."}

No engine is involved. Positions are deduplicated by FEN.
"""
import argparse
import json
import os
import sys

import chess.pgn


def iter_positions(paths, every, min_ply, max_ply, max_positions):
    seen = set()
    emitted = 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                game_id = os.path.basename(path)
                site = game.headers.get("Site", "")
                if site:
                    game_id += ":" + site
                board = game.board()
                for ply, move in enumerate(game.mainline_moves(), start=1):
                    board.push(move)
                    if ply < min_ply or ply > max_ply or ply % every:
                        continue
                    fen = board.fen()
                    if fen in seen:
                        continue
                    seen.add(fen)
                    yield {"game_id": game_id, "ply": ply, "fen": fen}
                    emitted += 1
                    if emitted >= max_positions:
                        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--max-ply", type=int, default=80)
    ap.add_argument("--max", type=int, default=1000, dest="max_positions")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for row in iter_positions(
            args.pgn, args.every, args.min_ply, args.max_ply, args.max_positions
        ):
            out.write(json.dumps(row) + "\n")
            n += 1
    if n == 0:
        sys.exit("no positions sampled")
    print(f"wrote {n} positions to {args.out}")


if __name__ == "__main__":
    main()
