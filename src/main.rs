mod book;
mod eval;
mod movegen;
mod search;
mod tt;
mod uci;

use std::cmp::Reverse;
use std::io::{self, BufRead, Write};
use std::thread;
use std::time::{Duration, Instant};

use book::{build_opening_book, try_book_move, BookEntry};
use eval::eval_for_side;
use movegen::{generate_pseudo_moves, in_bounds, is_square_attacked, mobility_count, move_to_uci, parse_square, parse_uci_move};
use search::{build_lmr_table, handle_go, piece_attacks_square, piece_value};
use tt::{board_hash, child_hash, null_move_hash, score_from_tt, score_to_tt, Bound, TransTable, Zobrist};
use uci::{flush_stdout, handle_divide, handle_perft, handle_perft_test, handle_position, handle_selftest, handle_setoption};

const MATE_SCORE: i32 = 30_000;
const INF: i32 = 1_000_000;
const MAX_PLY: usize = 128;
const ASPIRATION_WINDOW: i32 = 35;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum BookStyle {
    Normal,
    Grinder,
    Chaos,
}

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

fn square_mask(sq: u8) -> u64 {
    1u64 << sq
}

fn ray_mask_inclusive(from: u8, to: u8) -> u64 {
    let ff = (from % 8) as i8;
    let fr = (from / 8) as i8;
    let tf = (to % 8) as i8;
    let tr = (to / 8) as i8;
    let df = tf - ff;
    let dr = tr - fr;

    let (step_f, step_r) = if df == 0 {
        (0, dr.signum())
    } else if dr == 0 {
        (df.signum(), 0)
    } else if df.abs() == dr.abs() {
        (df.signum(), dr.signum())
    } else {
        return 0;
    };

    let mut mask = 0u64;
    let mut f = ff;
    let mut r = fr;
    loop {
        let sq = (r as u8) * 8 + (f as u8);
        mask |= square_mask(sq);
        if f == tf && r == tr {
            break;
        }
        f += step_f;
        r += step_r;
    }
    mask
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

#[derive(Clone, Eq)]
struct Board {
    squares: [Option<Piece>; 64],
    side_to_move: Color,
    castling: u8,
    en_passant: Option<u8>,
    halfmove_clock: u32,
    fullmove_number: u32,
    white_king_sq: Option<u8>,
    black_king_sq: Option<u8>,
    piece_sqs: [[u8; 16]; 2],
    piece_cnt: [usize; 2],
    sq_to_pl: [u8; 64],
}

impl PartialEq for Board {
    fn eq(&self, other: &Self) -> bool {
        self.squares == other.squares
            && self.side_to_move == other.side_to_move
            && self.castling == other.castling
            && self.en_passant == other.en_passant
            && self.halfmove_clock == other.halfmove_clock
            && self.fullmove_number == other.fullmove_number
            && self.white_king_sq == other.white_king_sq
            && self.black_king_sq == other.black_king_sq
    }
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
            white_king_sq: None,
            black_king_sq: None,
            piece_sqs: [[0; 16]; 2],
            piece_cnt: [0; 2],
            sq_to_pl: [255; 64],
        };
        b.set_startpos();
        b
    }

    fn pl_add(&mut self, sq: u8, color: Color) {
        let c = color_index(color);
        let idx = self.piece_cnt[c];
        self.piece_sqs[c][idx] = sq;
        self.sq_to_pl[sq as usize] = idx as u8;
        self.piece_cnt[c] += 1;
    }

    fn pl_remove(&mut self, sq: u8, color: Color) {
        let c = color_index(color);
        let idx = self.sq_to_pl[sq as usize] as usize;
        self.piece_cnt[c] -= 1;
        let last = self.piece_cnt[c];
        let last_sq = self.piece_sqs[c][last];
        self.piece_sqs[c][idx] = last_sq;
        self.sq_to_pl[last_sq as usize] = idx as u8;
        self.sq_to_pl[sq as usize] = 255;
    }

    fn pl_move(&mut self, from: u8, to: u8, color: Color) {
        let c = color_index(color);
        let idx = self.sq_to_pl[from as usize] as usize;
        self.piece_sqs[c][idx] = to;
        self.sq_to_pl[to as usize] = idx as u8;
        self.sq_to_pl[from as usize] = 255;
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
        self.white_king_sq = None;
        self.black_king_sq = None;
        self.piece_sqs = [[0; 16]; 2];
        self.piece_cnt = [0; 2];
        self.sq_to_pl = [255; 64];

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
                self.pl_add(sq as u8, color);
                if kind == PieceType::King {
                    if color == Color::White {
                        self.white_king_sq = Some(sq as u8);
                    } else {
                        self.black_king_sq = Some(sq as u8);
                    }
                }
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
        match color {
            Color::White => self.white_king_sq,
            Color::Black => self.black_king_sq,
        }
    }

    fn is_in_check(&self, color: Color) -> bool {
        if let Some(ksq) = self.king_square(color) {
            is_square_attacked(self, ksq, color.opposite())
        } else {
            false
        }
    }

    fn legal_filter_state(&self, color: Color) -> (usize, u64, [u64; 64]) {
        let mut pin_masks = [u64::MAX; 64];
        let mut check_count = 0usize;
        let mut evasion_mask = u64::MAX;
        let Some(king_sq) = self.king_square(color) else {
            return (check_count, evasion_mask, pin_masks);
        };
        let enemy = color.opposite();
        let kf = (king_sq % 8) as i8;
        let kr = (king_sq / 8) as i8;

        let mut register_checker = |mask: u64| {
            check_count += 1;
            if check_count == 1 {
                evasion_mask = mask;
            } else {
                evasion_mask = 0;
            }
        };

        let pawn_dirs: &[(i8, i8)] = match enemy {
            Color::White => &[(-1, -1), (1, -1)],
            Color::Black => &[(-1, 1), (1, 1)],
        };
        for &(df, dr) in pawn_dirs {
            let nf = kf + df;
            let nr = kr + dr;
            if !in_bounds(nf, nr) {
                continue;
            }
            let sq = (nr as u8) * 8 + (nf as u8);
            if self.squares[sq as usize]
                == Some(Piece {
                    color: enemy,
                    kind: PieceType::Pawn,
                })
            {
                register_checker(square_mask(sq));
            }
        }

        for (df, dr) in [
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ] {
            let nf = kf + df;
            let nr = kr + dr;
            if !in_bounds(nf, nr) {
                continue;
            }
            let sq = (nr as u8) * 8 + (nf as u8);
            if self.squares[sq as usize]
                == Some(Piece {
                    color: enemy,
                    kind: PieceType::Knight,
                })
            {
                register_checker(square_mask(sq));
            }
        }

        for df in -1..=1 {
            for dr in -1..=1 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let nf = kf + df;
                let nr = kr + dr;
                if !in_bounds(nf, nr) {
                    continue;
                }
                let sq = (nr as u8) * 8 + (nf as u8);
                if self.squares[sq as usize]
                    == Some(Piece {
                        color: enemy,
                        kind: PieceType::King,
                    })
                {
                    register_checker(square_mask(sq));
                }
            }
        }

        for (step_f, step_r, rook_like) in [
            (1, 0, true),
            (-1, 0, true),
            (0, 1, true),
            (0, -1, true),
            (1, 1, false),
            (1, -1, false),
            (-1, 1, false),
            (-1, -1, false),
        ] {
            let mut blocker = None;
            let mut f = kf + step_f;
            let mut r = kr + step_r;
            while in_bounds(f, r) {
                let sq = (r as u8) * 8 + (f as u8);
                if let Some(piece) = self.squares[sq as usize] {
                    if blocker.is_none() {
                        if piece.color == color {
                            blocker = Some(sq);
                        } else if matches!(piece.kind, PieceType::Queen)
                            || (rook_like && matches!(piece.kind, PieceType::Rook))
                            || (!rook_like && matches!(piece.kind, PieceType::Bishop))
                        {
                            register_checker(ray_mask_inclusive(king_sq, sq));
                        }
                        if piece.color != color {
                            break;
                        }
                    } else {
                        if piece.color != color
                            && (matches!(piece.kind, PieceType::Queen)
                                || (rook_like && matches!(piece.kind, PieceType::Rook))
                                || (!rook_like && matches!(piece.kind, PieceType::Bishop)))
                        {
                            let pinned_sq = blocker.unwrap();
                            pin_masks[pinned_sq as usize] = ray_mask_inclusive(king_sq, sq);
                        }
                        break;
                    }
                }
                f += step_f;
                r += step_r;
            }
        }

        (check_count, evasion_mask, pin_masks)
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
        let enemy = moving_piece.color.opposite();

        if mv.is_en_passant {
            let cap_sq = if moving_piece.color == Color::White {
                mv.to - 8
            } else {
                mv.to + 8
            };
            captured = self.squares[cap_sq as usize];
            captured_sq = Some(cap_sq);
            self.squares[cap_sq as usize] = None;
            self.pl_remove(cap_sq, enemy);
        } else if captured.is_some() {
            self.pl_remove(mv.to, enemy);
        }

        self.pl_move(mv.from, mv.to, moving_piece.color);

        let mut placed = moving_piece;
        if let Some(promo) = mv.promotion {
            placed.kind = promo;
        }
        self.squares[mv.to as usize] = Some(placed);
        if moving_piece.kind == PieceType::King {
            if moving_piece.color == Color::White {
                self.white_king_sq = Some(mv.to);
            } else {
                self.black_king_sq = Some(mv.to);
            }
        }

        if mv.is_castle {
            match mv.to {
                6 => {
                    self.squares[7] = None;
                    self.squares[5] = Some(Piece {
                        color: Color::White,
                        kind: PieceType::Rook,
                    });
                    self.pl_move(7, 5, Color::White);
                }
                2 => {
                    self.squares[0] = None;
                    self.squares[3] = Some(Piece {
                        color: Color::White,
                        kind: PieceType::Rook,
                    });
                    self.pl_move(0, 3, Color::White);
                }
                62 => {
                    self.squares[63] = None;
                    self.squares[61] = Some(Piece {
                        color: Color::Black,
                        kind: PieceType::Rook,
                    });
                    self.pl_move(63, 61, Color::Black);
                }
                58 => {
                    self.squares[56] = None;
                    self.squares[59] = Some(Piece {
                        color: Color::Black,
                        kind: PieceType::Rook,
                    });
                    self.pl_move(56, 59, Color::Black);
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

        self.pl_move(mv.to, mv.from, undo.moved_piece.color);
        self.squares[mv.to as usize] = None;
        self.squares[mv.from as usize] = Some(undo.moved_piece);
        if undo.moved_piece.kind == PieceType::King {
            if undo.moved_piece.color == Color::White {
                self.white_king_sq = Some(mv.from);
            } else {
                self.black_king_sq = Some(mv.from);
            }
        }

        if mv.is_castle {
            match mv.to {
                6 => {
                    self.squares[5] = None;
                    self.squares[7] = Some(Piece {
                        color: Color::White,
                        kind: PieceType::Rook,
                    });
                    self.pl_move(5, 7, Color::White);
                }
                2 => {
                    self.squares[3] = None;
                    self.squares[0] = Some(Piece {
                        color: Color::White,
                        kind: PieceType::Rook,
                    });
                    self.pl_move(3, 0, Color::White);
                }
                62 => {
                    self.squares[61] = None;
                    self.squares[63] = Some(Piece {
                        color: Color::Black,
                        kind: PieceType::Rook,
                    });
                    self.pl_move(61, 63, Color::Black);
                }
                58 => {
                    self.squares[59] = None;
                    self.squares[56] = Some(Piece {
                        color: Color::Black,
                        kind: PieceType::Rook,
                    });
                    self.pl_move(59, 56, Color::Black);
                }
                _ => {}
            }
        }

        if let Some(cp) = undo.captured {
            if let Some(cs) = undo.captured_sq {
                self.squares[cs as usize] = Some(cp);
                self.pl_add(cs, cp.color);
            }
        }
    }

    fn generate_legal_moves(&mut self, captures_only: bool, buf: &mut Vec<Move>) {
        let side = self.side_to_move;
        let (check_count, evasion_mask, pin_masks) = self.legal_filter_state(side);
        generate_pseudo_moves(self, captures_only, buf);
        // Filter in-place: write index tracks legal moves kept
        let mut write = 0usize;
        for read in 0..buf.len() {
            let mv = buf[read];
            let moving_piece = self.squares[mv.from as usize].unwrap();
            if moving_piece.kind == PieceType::King || mv.is_en_passant {
                let undo = self.make_move(mv);
                let legal = !self.is_in_check(side);
                self.undo_move(mv, undo);
                if !legal {
                    continue;
                }
            } else {
                if check_count >= 2 {
                    continue;
                }
                if check_count == 1 && (evasion_mask & square_mask(mv.to)) == 0 {
                    continue;
                }
                let pin_mask = pin_masks[mv.from as usize];
                if pin_mask != u64::MAX && (pin_mask & square_mask(mv.to)) == 0 {
                    continue;
                }
            }
            buf[write] = mv;
            write += 1;
        }
        buf.truncate(write);
    }
}

type HistoryTable = [[[i32; 64]; 64]; 2];
type CounterMoveTable = [[[Option<Move>; 64]; 64]; 2];
type ContinuationHistoryTable = [[[[i32; 64]; 64]; 64]; 2];

#[derive(Clone, Copy)]
struct SearchParams {
    futility_margin_d1: i32,
    futility_margin_d2: i32,
    reverse_futility_margin_d1: i32,
    reverse_futility_margin_d2: i32,
    reverse_futility_margin_d3: i32,
    null_margin_base: i32,
    null_margin_depth_scale: i32,
    null_reduction_base: i32,
    null_reduction_depth_div: i32,
    lmp_limit_d1: usize,
    lmp_limit_d2: usize,
    lmp_limit_d3: usize,
    lmr_min_depth: i32,
    lmr_min_move_index: usize,
    lmr_extra_depth: i32,
    lmr_extra_move_index: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            futility_margin_d1: 120,
            futility_margin_d2: 220,
            reverse_futility_margin_d1: 90,
            reverse_futility_margin_d2: 180,
            reverse_futility_margin_d3: 300,
            null_margin_base: 90,
            null_margin_depth_scale: 20,
            null_reduction_base: 2,
            null_reduction_depth_div: 4,
            lmp_limit_d1: 6,
            lmp_limit_d2: 10,
            lmp_limit_d3: 14,
            lmr_min_depth: 3,
            lmr_min_move_index: 3,
            lmr_extra_depth: 5,
            lmr_extra_move_index: 8,
        }
    }
}

struct EngineState {
    zobrist: Zobrist,
    tt: TransTable,
    history: HistoryTable,
    counter_moves: CounterMoveTable,
    continuation_history: ContinuationHistoryTable,
    search_params: SearchParams,
    lmr_table: [[i32; 64]; 64],
    searches_since_decay: u8,
    threads: usize,
    hash_mb: usize,
    opening_book: Vec<BookEntry>,
    own_book: bool,
    book_style: BookStyle,
}

impl EngineState {
    fn new() -> Self {
        let hash_mb = 64;
        let zobrist = Zobrist::new();
        let book_style = BookStyle::Normal;
        let opening_book = build_opening_book(&zobrist, book_style);
        Self {
            zobrist,
            tt: TransTable::new(hash_mb),
            history: [[[0; 64]; 64]; 2],
            counter_moves: [[[None; 64]; 64]; 2],
            continuation_history: [[[[0; 64]; 64]; 64]; 2],
            search_params: SearchParams::default(),
            lmr_table: build_lmr_table(),
            searches_since_decay: 0,
            threads: 1,
            hash_mb,
            opening_book,
            own_book: true,
            book_style,
        }
    }

    fn set_hash(&mut self, mb: usize) {
        self.hash_mb = mb.clamp(1, 2048);
        self.tt = TransTable::new(self.hash_mb);
    }

    fn set_threads(&mut self, threads: usize) {
        self.threads = threads.clamp(1, 32);
    }

    fn clear_for_new_game(&mut self) {
        self.tt.clear();
        self.history = [[[0; 64]; 64]; 2];
        self.counter_moves = [[[None; 64]; 64]; 2];
        self.continuation_history = [[[[0; 64]; 64]; 64]; 2];
        self.searches_since_decay = 0;
    }

    fn set_book_style(&mut self, style: BookStyle) {
        self.book_style = style;
        self.opening_book = build_opening_book(&self.zobrist, style);
    }

    fn decay_history(&mut self) {
        for side in 0..2 {
            for from in 0..64 {
                for to in 0..64 {
                    self.history[side][from][to] /= 2;
                    for next_to in 0..64 {
                        self.continuation_history[side][to][from][next_to] /= 2;
                    }
                }
            }
        }
    }

    fn prepare_for_search(&mut self) {
        self.tt.new_search();
        self.searches_since_decay = self.searches_since_decay.saturating_add(1);
        if self.searches_since_decay >= 4 {
            self.decay_history();
            self.searches_since_decay = 0;
        }
    }

    fn worker_clone(&self) -> Self {
        Self {
            zobrist: self.zobrist.clone(),
            tt: self.tt.clone(),
            history: self.history,
            counter_moves: self.counter_moves,
            continuation_history: self.continuation_history,
            search_params: self.search_params,
            lmr_table: self.lmr_table,
            searches_since_decay: 0,
            threads: 1,
            hash_mb: self.hash_mb,
            opening_book: self.opening_book.clone(),
            own_book: false,
            book_style: self.book_style,
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
            println!("option name Threads type spin default 1 min 1 max 32");
            println!("option name Clear Hash type button");
            println!("option name OwnBook type check default true");
            println!("option name BookStyle type combo default normal var normal var grinder var chaos");
            println!("option name FutilityMarginD1 type spin default 120 min 0 max 500");
            println!("option name FutilityMarginD2 type spin default 220 min 0 max 700");
            println!("option name ReverseFutilityMarginD1 type spin default 90 min 0 max 500");
            println!("option name ReverseFutilityMarginD2 type spin default 180 min 0 max 700");
            println!("option name ReverseFutilityMarginD3 type spin default 300 min 0 max 900");
            println!("option name NullMarginBase type spin default 90 min 0 max 300");
            println!("option name NullMarginDepthScale type spin default 20 min 0 max 80");
            println!("option name NullReductionBase type spin default 2 min 0 max 6");
            println!("option name NullReductionDepthDiv type spin default 4 min 1 max 12");
            println!("option name LmpLimitD1 type spin default 6 min 0 max 24");
            println!("option name LmpLimitD2 type spin default 10 min 0 max 32");
            println!("option name LmpLimitD3 type spin default 14 min 0 max 40");
            println!("option name LmrMinDepth type spin default 3 min 1 max 12");
            println!("option name LmrMinMoveIndex type spin default 3 min 1 max 24");
            println!("option name LmrExtraDepth type spin default 5 min 1 max 16");
            println!("option name LmrExtraMoveIndex type spin default 8 min 1 max 32");
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
