mod book;
mod eval;
mod movegen;
mod search;
mod tt;
mod uci;

use std::cmp::Reverse;
use std::io::{self, BufRead, Write};
use std::time::{Duration, Instant};

use book::{build_opening_book, try_book_move, BookEntry};
use eval::eval_for_side;
use movegen::{generate_pseudo_moves, in_bounds, is_square_attacked, mobility_from, move_to_uci, parse_square, parse_uci_move};
use search::{handle_go, piece_attacks_square, piece_value};
use tt::{board_hash, score_from_tt, score_to_tt, Bound, TransTable, Zobrist};
use uci::{flush_stdout, handle_divide, handle_perft, handle_perft_test, handle_position, handle_selftest, handle_setoption};

const MATE_SCORE: i32 = 30_000;
const INF: i32 = 1_000_000;
const MAX_PLY: usize = 128;
const ASPIRATION_WINDOW: i32 = 35;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Color {
    White,
    Black,
}

impl Color {
    fn opposite(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

fn color_index(color: Color) -> usize {
    match color {
        Color::White => 0,
        Color::Black => 1,
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct Piece {
    color: Color,
    kind: PieceType,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct Move {
    from: u8,
    to: u8,
    promotion: Option<PieceType>,
    is_en_passant: bool,
    is_castle: bool,
}

#[derive(Copy, Clone, Debug)]
struct Undo {
    captured: Option<Piece>,
    captured_sq: Option<u8>,
    castling: u8,
    en_passant: Option<u8>,
    halfmove_clock: u32,
    fullmove_number: u32,
    moved_piece: Piece,
}

#[derive(Clone, Eq, PartialEq)]
struct Board {
    squares: [Option<Piece>; 64],
    side_to_move: Color,
    castling: u8,
    en_passant: Option<u8>,
    halfmove_clock: u32,
    fullmove_number: u32,
}

impl Board {
    fn new() -> Self {
        let mut b = Self {
            squares: [None; 64],
            side_to_move: Color::White,
            castling: 0,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
        };
        b.set_startpos();
        b
    }

    fn set_startpos(&mut self) {
        self.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    fn set_fen(&mut self, fen: &str) {
        self.squares = [None; 64];
        self.castling = 0;
        self.en_passant = None;
        self.halfmove_clock = 0;
        self.fullmove_number = 1;

        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            self.set_startpos();
            return;
        }

        let mut rank = 7_i32;
        let mut file = 0_i32;
        for ch in parts[0].chars() {
            if ch == '/' {
                rank -= 1;
                file = 0;
                continue;
            }
            if ch.is_ascii_digit() {
                file += ch.to_digit(10).unwrap_or(0) as i32;
                continue;
            }
            let color = if ch.is_ascii_uppercase() {
                Color::White
            } else {
                Color::Black
            };
            let kind = match ch.to_ascii_lowercase() {
                'p' => PieceType::Pawn,
                'n' => PieceType::Knight,
                'b' => PieceType::Bishop,
                'r' => PieceType::Rook,
                'q' => PieceType::Queen,
                'k' => PieceType::King,
                _ => continue,
            };
            if rank >= 0 && rank < 8 && file >= 0 && file < 8 {
                let sq = (rank * 8 + file) as usize;
                self.squares[sq] = Some(Piece { color, kind });
            }
            file += 1;
        }

        self.side_to_move = if parts[1] == "w" {
            Color::White
        } else {
            Color::Black
        };

        if parts[2].contains('K') {
            self.castling |= 1;
        }
        if parts[2].contains('Q') {
            self.castling |= 2;
        }
        if parts[2].contains('k') {
            self.castling |= 4;
        }
        if parts[2].contains('q') {
            self.castling |= 8;
        }

        if parts[3] != "-" {
            self.en_passant = parse_square(parts[3]);
        }

        if parts.len() >= 5 {
            self.halfmove_clock = parts[4].parse().unwrap_or(0);
        }
        if parts.len() >= 6 {
            self.fullmove_number = parts[5].parse().unwrap_or(1);
        }
    }

    fn king_square(&self, color: Color) -> Option<u8> {
        for i in 0..64 {
            if let Some(p) = self.squares[i] {
                if p.color == color && p.kind == PieceType::King {
                    return Some(i as u8);
                }
            }
        }
        None
    }

    fn is_in_check(&self, color: Color) -> bool {
        if let Some(ksq) = self.king_square(color) {
            is_square_attacked(self, ksq, color.opposite())
        } else {
            false
        }
    }

    fn make_move(&mut self, mv: Move) -> Undo {
        let moving_piece = self.squares[mv.from as usize].unwrap();
        let mut captured = self.squares[mv.to as usize];
        let mut captured_sq = if captured.is_some() { Some(mv.to) } else { None };

        let undo = Undo {
            captured,
            captured_sq,
            castling: self.castling,
            en_passant: self.en_passant,
            halfmove_clock: self.halfmove_clock,
            fullmove_number: self.fullmove_number,
            moved_piece: moving_piece,
        };

        if moving_piece.kind == PieceType::Pawn || captured.is_some() {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        self.squares[mv.from as usize] = None;

        if mv.is_en_passant {
            let cap_sq = if moving_piece.color == Color::White {
                mv.to - 8
            } else {
                mv.to + 8
            };
            captured = self.squares[cap_sq as usize];
            captured_sq = Some(cap_sq);
            self.squares[cap_sq as usize] = None;
        }

        let mut placed = moving_piece;
        if let Some(promo) = mv.promotion {
            placed.kind = promo;
        }
        self.squares[mv.to as usize] = Some(placed);

        if mv.is_castle {
            match mv.to {
                6 => {
                    self.squares[7] = None;
                    self.squares[5] = Some(Piece {
                        color: Color::White,
                        kind: PieceType::Rook,
                    });
                }
                2 => {
                    self.squares[0] = None;
                    self.squares[3] = Some(Piece {
                        color: Color::White,
                        kind: PieceType::Rook,
                    });
                }
                62 => {
                    self.squares[63] = None;
                    self.squares[61] = Some(Piece {
                        color: Color::Black,
                        kind: PieceType::Rook,
                    });
                }
                58 => {
                    self.squares[56] = None;
                    self.squares[59] = Some(Piece {
                        color: Color::Black,
                        kind: PieceType::Rook,
                    });
                }
                _ => {}
            }
        }

        self.en_passant = None;
        if moving_piece.kind == PieceType::Pawn {
            let from_rank = mv.from / 8;
            let to_rank = mv.to / 8;
            if moving_piece.color == Color::White && from_rank == 1 && to_rank == 3 {
                self.en_passant = Some(mv.from + 8);
            }
            if moving_piece.color == Color::Black && from_rank == 6 && to_rank == 4 {
                self.en_passant = Some(mv.from - 8);
            }
        }

        match moving_piece.color {
            Color::White => {
                if moving_piece.kind == PieceType::King {
                    self.castling &= !(1 | 2);
                }
                if moving_piece.kind == PieceType::Rook {
                    if mv.from == 0 {
                        self.castling &= !2;
                    }
                    if mv.from == 7 {
                        self.castling &= !1;
                    }
                }
                if let Some(c) = captured {
                    if c.kind == PieceType::Rook {
                        if mv.to == 56 {
                            self.castling &= !8;
                        }
                        if mv.to == 63 {
                            self.castling &= !4;
                        }
                    }
                }
            }
            Color::Black => {
                if moving_piece.kind == PieceType::King {
                    self.castling &= !(4 | 8);
                }
                if moving_piece.kind == PieceType::Rook {
                    if mv.from == 56 {
                        self.castling &= !8;
                    }
                    if mv.from == 63 {
                        self.castling &= !4;
                    }
                }
                if let Some(c) = captured {
                    if c.kind == PieceType::Rook {
                        if mv.to == 0 {
                            self.castling &= !2;
                        }
                        if mv.to == 7 {
                            self.castling &= !1;
                        }
                    }
                }
            }
        }

        if self.side_to_move == Color::Black {
            self.fullmove_number += 1;
        }
        self.side_to_move = self.side_to_move.opposite();

        Undo {
            captured,
            captured_sq,
            ..undo
        }
    }

    fn undo_move(&mut self, mv: Move, undo: Undo) {
        self.side_to_move = self.side_to_move.opposite();
        self.castling = undo.castling;
        self.en_passant = undo.en_passant;
        self.halfmove_clock = undo.halfmove_clock;
        self.fullmove_number = undo.fullmove_number;

        self.squares[mv.to as usize] = None;
        self.squares[mv.from as usize] = Some(undo.moved_piece);

        if mv.is_castle {
            match mv.to {
                6 => {
                    self.squares[5] = None;
                    self.squares[7] = Some(Piece {
                        color: Color::White,
                        kind: PieceType::Rook,
                    });
                }
                2 => {
                    self.squares[3] = None;
                    self.squares[0] = Some(Piece {
                        color: Color::White,
                        kind: PieceType::Rook,
                    });
                }
                62 => {
                    self.squares[61] = None;
                    self.squares[63] = Some(Piece {
                        color: Color::Black,
                        kind: PieceType::Rook,
                    });
                }
                58 => {
                    self.squares[59] = None;
                    self.squares[56] = Some(Piece {
                        color: Color::Black,
                        kind: PieceType::Rook,
                    });
                }
                _ => {}
            }
        }

        if let Some(cp) = undo.captured {
            if let Some(cs) = undo.captured_sq {
                self.squares[cs as usize] = Some(cp);
            }
        }
    }

    fn generate_legal_moves(&mut self, captures_only: bool) -> Vec<Move> {
        let side = self.side_to_move;
        let pseudo = generate_pseudo_moves(self, captures_only);
        let mut legal = Vec::with_capacity(pseudo.len());
        for mv in pseudo {
            let undo = self.make_move(mv);
            if !self.is_in_check(side) {
                legal.push(mv);
            }
            self.undo_move(mv, undo);
        }
        legal
    }
}

type HistoryTable = [[[i32; 64]; 64]; 2];
type CounterMoveTable = [[[Option<Move>; 64]; 64]; 2];

struct EngineState {
    zobrist: Zobrist,
    tt: TransTable,
    history: HistoryTable,
    counter_moves: CounterMoveTable,
    hash_mb: usize,
    opening_book: Vec<BookEntry>,
    own_book: bool,
}

impl EngineState {
    fn new() -> Self {
        let hash_mb = 64;
        let zobrist = Zobrist::new();
        let opening_book = build_opening_book(&zobrist);
        Self {
            zobrist,
            tt: TransTable::new(hash_mb),
            history: [[[0; 64]; 64]; 2],
            counter_moves: [[[None; 64]; 64]; 2],
            hash_mb,
            opening_book,
            own_book: true,
        }
    }

    fn set_hash(&mut self, mb: usize) {
        self.hash_mb = mb.clamp(1, 2048);
        self.tt = TransTable::new(self.hash_mb);
    }

    fn clear_for_new_game(&mut self) {
        self.tt.clear();
        self.history = [[[0; 64]; 64]; 2];
        self.counter_moves = [[[None; 64]; 64]; 2];
    }

    fn decay_history(&mut self) {
        for side in 0..2 {
            for from in 0..64 {
                for to in 0..64 {
                    self.history[side][from][to] /= 2;
                }
            }
        }
    }
}

fn main() {
    let stdin = io::stdin();
    let mut board = Board::new();
    board.set_startpos();
    let mut engine = EngineState::new();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(v) => v,
            Err(_) => break,
        };
        let cmd = line.trim();
        if cmd.is_empty() {
            continue;
        }

        if cmd == "uci" {
            println!("id name Rustfish ZeroDep");
            println!("id author pi+chatgpt");
            println!("option name Hash type spin default 64 min 1 max 2048");
            println!("option name Clear Hash type button");
            println!("option name OwnBook type check default true");
            println!("uciok");
            flush_stdout();
            continue;
        }
        if cmd == "isready" {
            println!("readyok");
            flush_stdout();
            continue;
        }
        if cmd == "ucinewgame" {
            board.set_startpos();
            engine.clear_for_new_game();
            continue;
        }
        if cmd.starts_with("setoption") {
            handle_setoption(cmd, &mut engine);
            continue;
        }
        if cmd.starts_with("position") {
            handle_position(cmd, &mut board);
            continue;
        }
        if cmd.starts_with("go") {
            let best = handle_go(cmd, &mut board, &mut engine);
            println!("bestmove {}", best.unwrap_or_else(|| "0000".to_string()));
            flush_stdout();
            continue;
        }
        if cmd == "perfttest" {
            handle_perft_test(&mut board);
            continue;
        }
        if cmd.starts_with("perft") {
            handle_perft(cmd, &mut board);
            continue;
        }
        if cmd.starts_with("divide") {
            handle_divide(cmd, &mut board);
            continue;
        }
        if cmd.starts_with("selftest") {
            handle_selftest(cmd, &mut board);
            continue;
        }
        if cmd == "quit" {
            break;
        }
        if cmd == "stop" {
            continue;
        }
    }
}
