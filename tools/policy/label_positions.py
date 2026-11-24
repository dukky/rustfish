#!/usr/bin/env python3
"""Label sampled positions with rustfish and extract per-move features.

Input: JSONL from extract_positions.py (game_id, ply, fen).
Output: JSONL with best move, score, and all legal moves + features.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import chess

FEATURES = [
    "from_file",
    "from_rank",
    "to_file",
    "to_rank",
    "piece_type",
    "capture_value",
    "promotion_value",
    "is_capture",
    "is_check",
    "is_castle",
    "is_en_passant",
    "to_center",
    "from_center",
    "relative_rank",
    "to_attacked",
    "to_defended",
    "is_pawn_push",
    "is_quiet",
]

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def center(square):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    return 3.5 - max(abs(file - 3.5), abs(rank - 3.5))


def move_features(board, move):
    piece = board.piece_at(move.from_square)
    if piece is None:
        return None
    capture = board.piece_at(move.to_square)
    if board.is_en_passant(move):
        capture = chess.Piece(chess.PAWN, not piece.color)
    capture_value = PIECE_VALUE.get(capture.piece_type, 0) if capture else 0
    promo = PIECE_VALUE.get(move.promotion, 0) if move.promotion else 0
    check = board.gives_check(move)
    castle = board.is_castling(move)
    ep = board.is_en_passant(move)
    is_capture = board.is_capture(move)
    attacked = bool(board.attackers(not piece.color, move.to_square))
    defended = bool(board.attackers(piece.color, move.to_square))
    rank = chess.square_rank(move.to_square)
    rel_rank = rank if piece.color == chess.WHITE else 7 - rank
    return [
        chess.square_file(move.from_square),
        chess.square_rank(move.from_square),
        chess.square_file(move.to_square),
        rank,
        piece.piece_type,
        capture_value,
        promo,
        int(is_capture),
        int(check),
        int(castle),
        int(ep),
        center(move.to_square),
        center(move.from_square),
        rel_rank,
        int(attacked),
        int(defended),
        int(piece.piece_type == chess.PAWN and not is_capture),
        int(not is_capture and not promo and not check),
    ]


class Engine:
    def __init__(self, path):
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self.wait_for("uciok")
        self.send("isready")
        self.wait_for("readyok")

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def wait_for(self, token):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            if token in line:
                return line

    def search(self, fen, nodes):
        self.send("ucinewgame")
        self.send(f"position fen {fen}")
        self.send(f"go nodes {nodes}")
        score = None
        best = None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            if line.startswith("info") and " score " in line:
                toks = line.split()
                if "cp" in toks:
                    i = toks.index("cp")
                    try:
                        score = int(toks[i + 1])
                    except (IndexError, ValueError):
                        pass
                elif "mate" in toks:
                    i = toks.index("mate")
                    try:
                        score = 10000 - abs(int(toks[i + 1]))
                    except (IndexError, ValueError):
                        pass
            if line.startswith("bestmove"):
                toks = line.split()
                if len(toks) >= 2 and toks[1] not in ("(none)", "0000"):
                    best = toks[1]
                break
        return best, score

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nodes", type=int, default=50000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    engine = Engine(args.engine)
    rows = 0
    written = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    try:
        with open(args.positions, encoding="utf-8") as inp, open(
            args.out, "w", encoding="utf-8"
        ) as out:
            for line in inp:
                if not line.strip():
                    continue
                rows += 1
                if args.limit and rows > args.limit:
                    break
                pos = json.loads(line)
                board = chess.Board(pos["fen"])
                if board.is_game_over(claim_draw=False):
                    continue
                best, score = engine.search(pos["fen"], args.nodes)
                if best is None:
                    continue
                try:
                    best_move = chess.Move.from_uci(best)
                except ValueError:
                    continue
                if best_move not in board.legal_moves:
                    continue
                moves = []
                for move in board.legal_moves:
                    x = move_features(board, move)
                    if x is None:
                        continue
                    moves.append(
                        {
                            "uci": move.uci(),
                            "label": int(move == best_move),
                            "x": x,
                        }
                    )
                if len(moves) < 2 or sum(m["label"] for m in moves) != 1:
                    continue
                out.write(
                    json.dumps(
                        {
                            "game_id": pos["game_id"],
                            "ply": pos["ply"],
                            "fen": pos["fen"],
                            "best": best,
                            "score_cp": score,
                            "features": FEATURES,
                            "moves": moves,
                        }
                    )
                    + "\n"
                )
                written += 1
                if written % 25 == 0:
                    print(f"labeled {written} positions", file=sys.stderr)
    finally:
        engine.close()
    if written == 0:
        sys.exit("no positions labeled")
    print(f"wrote {written} labeled positions to {args.out}")


if __name__ == "__main__":
    main()
