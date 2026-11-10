mod bitboard;
mod book;
mod datagen;
mod eval;
mod evalcache;
mod eval_params;
mod movegen;
mod nnue;
mod psqt;
mod search;
mod tt;
mod tune;
mod uci;

use std::cmp::Reverse;
use std::io::{self, BufRead, Write};
use std::thread;
use std::time::{Duration, Instant};

use book::{build_opening_book, try_book_move, BookEntry};
use eval::eval_for_side;
#[cfg(test)]
use movegen::{generate_pseudo_moves, is_square_attacked_occ};
use movegen::{generate_legal_moves_fused, in_bounds, is_square_attacked, piece_attacks, move_to_uci, parse_square, parse_uci_move};
use search::{build_lmr_table, handle_go};
use tt::{board_hash, child_hash, null_move_hash, score_from_tt, score_to_tt, Bound, TransTable, Zobrist};
use uci::{flush_stdout, handle_bench, handle_divide, handle_evalhash, spawn_input_reader, handle_perft, handle_perft_test, handle_position, handle_selftest, handle_setoption};

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

/// Reference oracle for the ray masks `legal_filter_state` builds from the
/// bitboard ray tables.
#[cfg(test)]
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

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct EvalAcc {
    mg: i32,
    eg: i32,
    phase: i32,
    npm: [i32; 2],
    queens: [i8; 2],
    bishops: [i8; 2],
}

/// Non-pawn material (`search::piece_value` for N/B/R/Q, 0 for P/K) and game
/// phase weight, indexed by `PieceType as usize`.
const ACC_NPM: [i32; 6] = [0, 320, 330, 500, 900, 0];
const ACC_PHASE: [i32; 6] = [0, 1, 1, 2, 4, 0];

impl EvalAcc {
    #[inline]
    fn apply(&mut self, sq: u8, p: Piece, sign: i32) {
        let k = p.kind as usize;
        let rel = if p.color == Color::White { sq } else { sq ^ 56 } as usize;
        let mg = psqt::PSQT_MG[k][rel];
        let eg = psqt::PSQT_EG[k][rel];
        let side_sign = if p.color == Color::White { sign } else { -sign };
        self.mg += side_sign * mg;
        self.eg += side_sign * eg;

        let c = color_index(p.color);
        self.npm[c] += sign * ACC_NPM[k];
        self.phase += sign * ACC_PHASE[k];
        self.queens[c] += sign as i8 * (p.kind == PieceType::Queen) as i8;
        self.bishops[c] += sign as i8 * (p.kind == PieceType::Bishop) as i8;
    }

    /// Same result as `apply(from, p, -1)` then `apply(to, p, 1)`: material
    /// and phase cancel, only the PSQT delta remains (exact integer sums).
    #[inline]
    fn shift(&mut self, from: u8, to: u8, p: Piece) {
        let k = p.kind as usize;
        let (f, t) = if p.color == Color::White {
            (from as usize, to as usize)
        } else {
            ((from ^ 56) as usize, (to ^ 56) as usize)
        };
        let dmg = psqt::PSQT_MG[k][t] - psqt::PSQT_MG[k][f];
        let deg = psqt::PSQT_EG[k][t] - psqt::PSQT_EG[k][f];
        if p.color == Color::White {
            self.mg += dmg;
            self.eg += deg;
        } else {
            self.mg -= dmg;
            self.eg -= deg;
        }
    }
}

#[cfg(test)]
mod eval_acc_tests {
    use super::*;

    const KINDS: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    #[test]
    fn tables_match_piece_value_and_phase() {
        for kind in KINDS {
            let minor_major = !matches!(kind, PieceType::Pawn | PieceType::King);
            let npm = if minor_major { search::piece_value(kind) } else { 0 };
            assert_eq!(ACC_NPM[kind as usize], npm);
            let phase = match kind {
                PieceType::Knight | PieceType::Bishop => 1,
                PieceType::Rook => 2,
                PieceType::Queen => 4,
                _ => 0,
            };
            assert_eq!(ACC_PHASE[kind as usize], phase);
        }
    }

    #[test]
    fn shift_equals_remove_then_add() {
        let base = EvalAcc { mg: 17, eg: -9, phase: 12, npm: [1000, 2000], queens: [1, 1], bishops: [2, 1] };
        for color in [Color::White, Color::Black] {
            for kind in KINDS {
                let p = Piece { color, kind };
                for from in 0..64u8 {
                    for to in 0..64u8 {
                        let mut a = base;
                        a.apply(from, p, -1);
                        a.apply(to, p, 1);
                        let mut b = base;
                        b.shift(from, to, p);
                        assert_eq!(a, b);
                    }
                }
            }
        }
    }
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
    eval_acc: EvalAcc,
    bb: bitboard::BoardBB,
    /// NNUE accumulator stack; `None` (the default) costs one branch per
    /// make/undo and leaves the HCE path untouched.
    nnue: Option<Box<nnue::NnueState>>,
}

#[derive(Copy, Clone, Debug)]
struct LegalState {
    check_count: usize,
    evasion_mask: u64,
    pinned: u64,
    pin_rays: [u64; 8],
}

impl LegalState {
    /// Legal destination mask for a non-king piece on `sq`.
    #[inline(always)]
    fn allowed(&self, sq: u8) -> u64 {
        let mut allowed = self.evasion_mask;
        let bit = square_mask(sq);
        if self.pinned & bit != 0 {
            for &r in &self.pin_rays {
                if r & bit != 0 {
                    allowed &= r;
                    break;
                }
            }
        }
        allowed
    }
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
            eval_acc: EvalAcc::default(),
            bb: bitboard::BoardBB::default(),
            nnue: None,
        };
        b.set_startpos();
        b
    }

    fn acc_add(&mut self, sq: u8, p: Piece) {
        self.eval_acc.apply(sq, p, 1);
        self.bb.toggle(sq, p);
    }

    fn acc_remove(&mut self, sq: u8, p: Piece) {
        self.eval_acc.apply(sq, p, -1);
        self.bb.toggle(sq, p);
    }

    /// `acc_remove(from, p)` + `acc_add(to, p)` in one step.
    #[inline]
    fn acc_shift(&mut self, from: u8, to: u8, p: Piece) {
        self.eval_acc.shift(from, to, p);
        let bits = square_mask(from) | square_mask(to);
        let c = color_index(p.color);
        self.bb.pieces[c][p.kind as usize] ^= bits;
        self.bb.occ[c] ^= bits;
    }

    fn recompute_eval_acc(&self) -> EvalAcc {
        let mut acc = EvalAcc::default();
        for sq in 0..64u8 {
            if let Some(p) = self.squares[sq as usize] {
                acc.apply(sq, p, 1);
            }
        }
        acc
    }

    fn recompute_bb(&self) -> bitboard::BoardBB {
        let mut bb = bitboard::BoardBB::default();
        for sq in 0..64u8 {
            if let Some(p) = self.squares[sq as usize] {
                bb.toggle(sq, p);
            }
        }
        bb
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

        self.eval_acc = self.recompute_eval_acc();
        self.bb = self.recompute_bb();
        if let Some(mut st) = self.nnue.take() {
            st.refresh(self);
            self.nnue = Some(st);
        }
    }

    /// Attach (refreshing from scratch) or detach the NNUE accumulators.
    fn attach_nnue(&mut self, net: Option<std::sync::Arc<nnue::Network>>) {
        self.nnue = net.map(|n| Box::new(nnue::NnueState::new(n, self)));
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

    /// Checkers and pins for `color`'s king, consumed by the fused legal
    /// generator. A pinned piece's legal targets are the one `pin_rays` entry
    /// containing it (king..pinner inclusive); `pinned` marks those pieces.
    #[inline(always)]
    fn legal_state(&self, color: Color) -> LegalState {
        let mut st = LegalState {
            check_count: 0,
            evasion_mask: u64::MAX,
            pinned: 0,
            pin_rays: [0; 8],
        };
        let Some(king_sq) = self.king_square(color) else {
            return st;
        };
        use crate::bitboard::{DIAG_RAYS, KING_ATTACKS, KNIGHT_ATTACKS, ORTHO_RAYS, PAWN_ATTACKERS};

        let c = color_index(color);
        let e = 1 - c;
        let bb = &self.bb;
        let k = king_sq as usize;
        let king_bit = square_mask(king_sq);
        let own = bb.occ[c];
        let occ = own | bb.occ[e];

        let mut leapers = (PAWN_ATTACKERS[e][k] & bb.pieces[e][0])
            | (KNIGHT_ATTACKS[k] & bb.pieces[e][1])
            | (KING_ATTACKS[k] & bb.pieces[e][5]);
        while leapers != 0 {
            let sq = leapers.trailing_zeros() as u8;
            leapers &= leapers - 1;
            st.check_count += 1;
            st.evasion_mask = if st.check_count == 1 { square_mask(sq) } else { 0 };
        }

        let diag = bb.pieces[e][2] | bb.pieces[e][4];
        let ortho = bb.pieces[e][3] | bb.pieces[e][4];
        for (slot_base, rays, sliders) in [(0usize, &DIAG_RAYS, diag), (4usize, &ORTHO_RAYS, ortho)] {
            if sliders == 0 {
                continue;
            }
            for d in 0..4 {
                let ray = rays[d][k];
                if ray & sliders == 0 {
                    continue;
                }
                let first_of = |b: u64| -> usize {
                    if d < 2 {
                        b.trailing_zeros() as usize
                    } else {
                        63 - b.leading_zeros() as usize
                    }
                };
                let blockers = ray & occ;
                let b1 = first_of(blockers);
                let b1_bit = 1u64 << b1;
                if b1_bit & own == 0 {
                    if b1_bit & sliders != 0 {
                        st.check_count += 1;
                        st.evasion_mask = if st.check_count == 1 {
                            (ray & !rays[d][b1]) | king_bit
                        } else {
                            0
                        };
                    }
                    continue;
                }
                let rest = blockers & rays[d][b1];
                if rest == 0 {
                    continue;
                }
                let b2 = first_of(rest);
                if (1u64 << b2) & sliders != 0 {
                    st.pinned |= b1_bit;
                    st.pin_rays[slot_base + d] = (ray & !rays[d][b2]) | king_bit;
                }
            }
        }

        st
    }

    #[cfg(test)]
    fn legal_filter_state(&self, color: Color) -> (usize, u64, [u64; 64]) {
        let mut pin_masks = [u64::MAX; 64];
        let mut check_count = 0usize;
        let mut evasion_mask = u64::MAX;
        let Some(king_sq) = self.king_square(color) else {
            return (check_count, evasion_mask, pin_masks);
        };
        use crate::bitboard::{DIAG_RAYS, KING_ATTACKS, KNIGHT_ATTACKS, ORTHO_RAYS, PAWN_ATTACKERS};

        let c = color_index(color);
        let e = 1 - c;
        let bb = &self.bb;
        let k = king_sq as usize;
        let king_bit = square_mask(king_sq);
        let own = bb.occ[c];
        let occ = own | bb.occ[e];

        // Leaper checkers: the evasion target is the checker's square.
        let mut leapers = (PAWN_ATTACKERS[e][k] & bb.pieces[e][0])
            | (KNIGHT_ATTACKS[k] & bb.pieces[e][1])
            | (KING_ATTACKS[k] & bb.pieces[e][5]);
        while leapers != 0 {
            let sq = leapers.trailing_zeros() as u8;
            leapers &= leapers - 1;
            check_count += 1;
            evasion_mask = if check_count == 1 { square_mask(sq) } else { 0 };
        }

        // Sliders: walk each of the 8 rays from the king. The first piece is
        // either an enemy checker or an own piece that may be pinned by the
        // second piece. Masks run king..slider inclusive (ray_mask_inclusive).
        let diag = bb.pieces[e][2] | bb.pieces[e][4];
        let ortho = bb.pieces[e][3] | bb.pieces[e][4];
        for (rays, sliders) in [(&DIAG_RAYS, diag), (&ORTHO_RAYS, ortho)] {
            if sliders == 0 {
                continue;
            }
            for d in 0..4 {
                let ray = rays[d][k];
                if ray & sliders == 0 {
                    continue;
                }
                let first_of = |b: u64| -> usize {
                    if d < 2 {
                        b.trailing_zeros() as usize
                    } else {
                        63 - b.leading_zeros() as usize
                    }
                };
                let blockers = ray & occ;
                let b1 = first_of(blockers);
                let b1_bit = 1u64 << b1;
                if b1_bit & own == 0 {
                    if b1_bit & sliders != 0 {
                        check_count += 1;
                        evasion_mask = if check_count == 1 {
                            (ray & !rays[d][b1]) | king_bit
                        } else {
                            0
                        };
                    }
                    continue;
                }
                let rest = blockers & rays[d][b1];
                if rest == 0 {
                    continue;
                }
                let b2 = first_of(rest);
                if (1u64 << b2) & sliders != 0 {
                    pin_masks[b1] = (ray & !rays[d][b2]) | king_bit;
                }
            }
        }

        (check_count, evasion_mask, pin_masks)
    }

    fn make_move(&mut self, mv: Move) -> Undo {
        if let Some(mut st) = self.nnue.take() {
            st.push_move(self, mv);
            self.nnue = Some(st);
        }
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
            if let Some(cp) = captured {
                self.acc_remove(cap_sq, cp);
            }
            self.squares[cap_sq as usize] = None;
        } else if let Some(cp) = captured {
            self.acc_remove(mv.to, cp);
        }

        let mut placed = moving_piece;
        if let Some(promo) = mv.promotion {
            placed.kind = promo;
            self.acc_remove(mv.from, moving_piece);
            self.acc_add(mv.to, placed);
        } else {
            self.acc_shift(mv.from, mv.to, moving_piece);
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
            let rook_shift = match mv.to {
                6 => Some((7u8, 5u8, Color::White)),
                2 => Some((0, 3, Color::White)),
                62 => Some((63, 61, Color::Black)),
                58 => Some((56, 59, Color::Black)),
                _ => None,
            };
            if let Some((rook_from, rook_to, rook_color)) = rook_shift {
                let rook = Piece {
                    color: rook_color,
                    kind: PieceType::Rook,
                };
                self.squares[rook_from as usize] = None;
                self.squares[rook_to as usize] = Some(rook);
                self.acc_shift(rook_from, rook_to, rook);
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
        if let Some(st) = self.nnue.as_mut() {
            st.pop();
        }
        self.side_to_move = self.side_to_move.opposite();
        self.castling = undo.castling;
        self.en_passant = undo.en_passant;
        self.halfmove_clock = undo.halfmove_clock;
        self.fullmove_number = undo.fullmove_number;

        if let Some(promo) = mv.promotion {
            let mut placed = undo.moved_piece;
            placed.kind = promo;
            self.acc_remove(mv.to, placed);
            self.acc_add(mv.from, undo.moved_piece);
        } else {
            self.acc_shift(mv.to, mv.from, undo.moved_piece);
        }

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
            let rook_shift = match mv.to {
                6 => Some((7u8, 5u8, Color::White)),
                2 => Some((0, 3, Color::White)),
                62 => Some((63, 61, Color::Black)),
                58 => Some((56, 59, Color::Black)),
                _ => None,
            };
            if let Some((rook_from, rook_to, rook_color)) = rook_shift {
                let rook = Piece {
                    color: rook_color,
                    kind: PieceType::Rook,
                };
                self.squares[rook_to as usize] = None;
                self.squares[rook_from as usize] = Some(rook);
                self.acc_shift(rook_to, rook_from, rook);
            }
        }

        if let Some(cp) = undo.captured {
            if let Some(cs) = undo.captured_sq {
                self.squares[cs as usize] = Some(cp);
                self.acc_add(cs, cp);
            }
        }
    }

    fn generate_legal_moves(&mut self, captures_only: bool, buf: &mut Vec<Move>) {
        let side = self.side_to_move;
        let st = self.legal_state(side);
        // En passant is the one case the pin/check masks cannot decide
        // (both pawns leave the rank), so it keeps the make/undo test, run
        // up front for the at most two candidate pawns.
        let mut ep_legal_from = 0u64;
        if let Some(ep) = self.en_passant {
            use crate::bitboard::PAWN_ATTACKERS;
            let c = color_index(side);
            let ep_bit = square_mask(ep);
            let mut pawns = self.bb.pieces[c][0];
            while pawns != 0 {
                let sq = pawns.trailing_zeros() as u8;
                pawns &= pawns - 1;
                if PAWN_ATTACKERS[1 - c][sq as usize] & ep_bit != 0 {
                    let mv = Move {
                        from: sq,
                        to: ep,
                        promotion: None,
                        is_en_passant: true,
                        is_castle: false,
                    };
                    let undo = self.make_move(mv);
                    let legal = !self.is_in_check(side);
                    self.undo_move(mv, undo);
                    if legal {
                        ep_legal_from |= square_mask(sq);
                    }
                }
            }
        }
        generate_legal_moves_fused(self, captures_only, &st, ep_legal_from, buf);
    }

    #[cfg(test)]
    fn generate_legal_moves_reference(&mut self, captures_only: bool, buf: &mut Vec<Move>) {
        let side = self.side_to_move;
        let (check_count, evasion_mask, pin_masks) = self.legal_filter_state(side);
        generate_pseudo_moves(self, captures_only, buf);
        let enemy = side.opposite();
        // King destinations are tested with the king lifted off the board so
        // sliders x-ray through its origin square: no make/undo needed.
        let king_sq = self.king_square(side);
        let occ_no_king = match king_sq {
            Some(k) => self.bb.occ_all() & !square_mask(k),
            None => self.bb.occ_all(),
        };
        // Filter in-place: write index tracks legal moves kept
        let mut write = 0usize;
        for read in 0..buf.len() {
            let mv = buf[read];
            if Some(mv.from) == king_sq {
                if is_square_attacked_occ(self, mv.to, enemy, occ_no_king) {
                    continue;
                }
            } else if mv.is_en_passant {
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

/// Butterfly history, [side][from attacked][to attacked][from][to]: the two
/// threat bits say whether the opponent attacks the from/to square.
type HistoryTable = [[[[[i32; 64]; 64]; 2]; 2]; 2];
type CounterMoveTable = [[[Option<Move>; 64]; 64]; 2];
type ContinuationHistoryTable = [[[[i32; 64]; 64]; 64]; 2];

#[derive(Clone, Copy)]
struct SearchParams {
    futility_margin_d1: i32,
    futility_margin_d2: i32,
    reverse_futility_margin_d1: i32,
    reverse_futility_margin_d2: i32,
    reverse_futility_margin_d3: i32,
    null_reduction_base: i32,
    // Index quiet history by whether the from/to squares are attacked.
    threat_history: bool,
    // Ordering bonus for a quiet that moves a piece attacked by a lesser
    // piece to a square no lesser piece attacks (0 = off).
    threat_escape_bonus: i32,
    null_reduction_depth_div: i32,
    lmp_limit_d1: usize,
    // LMR adjustments by node type (plies): + at expected cut nodes, - at PV nodes.
    lmr_cut_node: i32,
    lmr_pv_node: i32,
    lmp_limit_d2: usize,
    // Plies the TT move loses when not singular but its TT score is >= beta.
    se_negative_extension: i32,
    lmp_limit_d3: usize,
    lmr_min_depth: i32,
    // Extra LMR plies for quiets at nodes whose static eval is not improving.
    lmr_not_improving: i32,
    lmr_min_move_index: usize,
    lmr_extra_depth: i32,
    lmr_extra_move_index: usize,
    // Node-fraction time management, see NODE_TM_MIN_DEPTH in search.rs.
    node_tm_base: u64,
    node_tm_scale: u64,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            futility_margin_d1: 120,
            futility_margin_d2: 220,
            reverse_futility_margin_d1: 90,
            reverse_futility_margin_d2: 180,
            reverse_futility_margin_d3: 300,
            null_reduction_base: 2,
            threat_history: true,
            threat_escape_bonus: 8_000,
            null_reduction_depth_div: 4,
            lmp_limit_d1: 6,
            lmr_cut_node: 2,
            lmr_pv_node: 1,
            se_negative_extension: 1,
            lmp_limit_d2: 10,
            lmp_limit_d3: 14,
            lmr_min_depth: 3,
            lmr_not_improving: 1,
            lmr_min_move_index: 3,
            lmr_extra_depth: 5,
            lmr_extra_move_index: 8,
            node_tm_base: 175,
            node_tm_scale: 75,
        }
    }
}

struct EngineState {
    zobrist: Zobrist,
    tt: TransTable,
    eval_cache: evalcache::EvalCache,
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
    book_file: Option<String>,
    position_history: Vec<u64>,
    last_search_nodes: u64,
    move_overhead: u64,
    /// Net loaded from `EvalFile`; `UseNNUE` selects whether search uses it.
    nnue_net: Option<std::sync::Arc<nnue::Network>>,
    use_nnue: bool,
}

/// Fourth net trained on 77M positions of net3 self-play (seed 5 at 8000
/// nodes, 2 epochs) with `tools/nnue-trainer`; embedded so the engine is
/// NNUE by default.
const EMBEDDED_NET: &[u8] = include_bytes!("../net/nnue.bin");

impl EngineState {
    fn new() -> Self {
        let hash_mb = 64;
        let zobrist = Zobrist::new();
        let book_style = BookStyle::Normal;
        let opening_book = build_opening_book(&zobrist, book_style, None);
        Self {
            zobrist,
            tt: TransTable::new(hash_mb),
            eval_cache: evalcache::EvalCache::new(),
            history: [[[[[0; 64]; 64]; 2]; 2]; 2],
            counter_moves: [[[None; 64]; 64]; 2],
            continuation_history: [[[[0; 64]; 64]; 64]; 2],
            search_params: SearchParams::default(),
            lmr_table: build_lmr_table(),
            searches_since_decay: 0,
            threads: 4,
            hash_mb,
            opening_book,
            // CCRL and most testers use their own opening books; an engine
            // book left on by default can be measured instead of the engine.
            own_book: false,
            book_style,
            book_file: None,
            position_history: Vec::new(),
            last_search_nodes: 0,
            move_overhead: 10,
            nnue_net: Some(std::sync::Arc::new(
                nnue::Network::from_bytes(EMBEDDED_NET).expect("embedded NNUE net"),
            )),
            use_nnue: true,
        }
    }

    /// The net search should evaluate with, if NNUE is enabled and loaded.
    fn active_net(&self) -> Option<std::sync::Arc<nnue::Network>> {
        if self.use_nnue {
            self.nnue_net.clone()
        } else {
            None
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
        self.eval_cache.clear();
        self.history = [[[[[0; 64]; 64]; 2]; 2]; 2];
        self.counter_moves = [[[None; 64]; 64]; 2];
        self.continuation_history = [[[[0; 64]; 64]; 64]; 2];
        self.searches_since_decay = 0;
        self.position_history.clear();
    }

    fn set_book_style(&mut self, style: BookStyle) {
        self.book_style = style;
        self.reload_opening_book();
    }

    fn set_book_file(&mut self, path: Option<String>) {
        self.book_file = path;
        self.reload_opening_book();
    }

    fn reload_opening_book(&mut self) {
        self.opening_book = build_opening_book(
            &self.zobrist,
            self.book_style,
            self.book_file.as_deref(),
        );
    }

    // Ages history between moves. Gravity already bounds magnitude; this
    // only fades stale entries that are no longer being updated.
    // Halves every entry once; walked in memory order (the 2 MB
    // continuation table used to be strided, ~4% of a 5000-node search).
    fn decay_history(&mut self) {
        let history = self.history.as_flattened_mut().as_flattened_mut().as_flattened_mut().as_flattened_mut();
        let cont = self.continuation_history.as_flattened_mut().as_flattened_mut().as_flattened_mut();
        for e in history.iter_mut() {
            *e /= 2;
        }
        for e in cont.iter_mut() {
            *e /= 2;
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
            eval_cache: evalcache::EvalCache::new(),
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
            book_file: self.book_file.clone(),
            position_history: self.position_history.clone(),
            last_search_nodes: 0,
            move_overhead: self.move_overhead,
            nnue_net: self.nnue_net.clone(),
            use_nnue: self.use_nnue,
        }
    }
}

fn main() {
    let input = spawn_input_reader();
    let mut board = Board::new();
    board.set_startpos();
    let mut engine = EngineState::new();
    // Counts `go` commands in the same order the reader numbered them.
    let mut go_seq = 0u64;

    for line in input {
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
            println!("option name Move Overhead type spin default 10 min 0 max 5000");
            println!("option name OwnBook type check default false");
            println!("option name BookStyle type combo default normal var normal var grinder var chaos");
            println!("option name BookFile type string default");
            println!("option name EvalFile type string default");
            println!("option name UseNNUE type check default true");
            println!("option name FutilityMarginD1 type spin default 120 min 0 max 500");
            println!("option name FutilityMarginD2 type spin default 220 min 0 max 700");
            println!("option name ReverseFutilityMarginD1 type spin default 90 min 0 max 500");
            println!("option name ReverseFutilityMarginD2 type spin default 180 min 0 max 700");
            println!("option name ReverseFutilityMarginD3 type spin default 300 min 0 max 900");
            println!("option name NullReductionBase type spin default 2 min 0 max 6");
            println!("option name ThreatHistory type check default true");
            println!("option name ThreatEscapeBonus type spin default 8000 min 0 max 30000");
            println!("option name NullReductionDepthDiv type spin default 4 min 1 max 12");
            println!("option name LmpLimitD1 type spin default 6 min 0 max 24");
            println!("option name LmrCutNode type spin default 2 min 0 max 4");
            println!("option name LmrPvNode type spin default 1 min 0 max 4");
            println!("option name SeNegativeExtension type spin default 1 min 0 max 3");
            println!("option name LmpLimitD2 type spin default 10 min 0 max 32");
            println!("option name LmpLimitD3 type spin default 14 min 0 max 40");
            println!("option name LmrMinDepth type spin default 3 min 1 max 12");
            println!("option name LmrNotImproving type spin default 1 min 0 max 3");
            println!("option name LmrMinMoveIndex type spin default 3 min 1 max 24");
            println!("option name LmrExtraDepth type spin default 5 min 1 max 16");
            println!("option name LmrExtraMoveIndex type spin default 8 min 1 max 32");
            println!("option name NodeTmBase type spin default 175 min 110 max 300");
            println!("option name NodeTmScale type spin default 75 min 50 max 300");
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
            handle_position(cmd, &mut board, &mut engine);
            continue;
        }
        if cmd == "go" || cmd.starts_with("go ") {
            go_seq += 1;
            search::ACTIVE_GO.store(go_seq, std::sync::atomic::Ordering::SeqCst);
            let best = handle_go(cmd, &mut board, &mut engine);
            search::ACTIVE_GO.store(0, std::sync::atomic::Ordering::SeqCst);
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
        if cmd.starts_with("datagen ") {
            datagen::handle_datagen(cmd);
            continue;
        }
        if cmd.starts_with("extract ") {
            tune::handle_extract(cmd);
            continue;
        }
        if cmd.starts_with("tune ") {
            tune::handle_tune(cmd);
            continue;
        }
        if cmd.starts_with("evalhash") {
            handle_evalhash(cmd, &mut board, engine.active_net());
            continue;
        }
        if cmd.starts_with("bench") {
            handle_bench(cmd, &mut board, &mut engine);
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

#[cfg(test)]
mod legal_filter_tests {
    use super::*;
    use crate::bitboard::{DIAG_RAYS, ORTHO_RAYS};

    #[test]
    fn ray_table_segments_match_inclusive_ray_mask() {
        for k in 0..64usize {
            for rays in [&DIAG_RAYS, &ORTHO_RAYS] {
                for d in 0..4 {
                    let ray = rays[d][k];
                    let mut sqs = ray;
                    while sqs != 0 {
                        let b = sqs.trailing_zeros() as usize;
                        sqs &= sqs - 1;
                        let seg = (ray & !rays[d][b]) | square_mask(k as u8);
                        assert_eq!(seg, ray_mask_inclusive(k as u8, b as u8), "k={k} b={b} d={d}");
                    }
                }
            }
        }
    }

    #[test]
    fn legal_filter_state_pins_and_checks() {
        let mut board = Board::new();
        // Rook e8 pins knight e2, bishop a5 pins pawn d2, queen h4 checks e1.
        board.set_fen("4r1k1/8/8/b7/7q/8/3PN3/4K3 w - - 0 1");
        let (count, evasion, pins) = board.legal_filter_state(Color::White);
        assert_eq!(count, 1);
        assert_eq!(evasion, ray_mask_inclusive(4, 31));
        assert_eq!(pins[12], ray_mask_inclusive(4, 60));
        assert_eq!(pins[11], ray_mask_inclusive(4, 32));
        assert_eq!(pins.iter().filter(|&&m| m != u64::MAX).count(), 2);
    }

    fn assert_same_legal_moves(board: &mut Board, ctx: &str) {
        let mut fused = Vec::new();
        let mut reference = Vec::new();
        for captures_only in [false, true] {
            let before = board.clone();
            board.generate_legal_moves(captures_only, &mut fused);
            board.generate_legal_moves_reference(captures_only, &mut reference);
            assert!(*board == before, "board changed by movegen: {ctx}");
            assert_eq!(fused, reference, "captures_only={captures_only} {ctx}");
        }
    }

    fn walk(board: &mut Board, depth: u32, fen: &str) {
        assert_same_legal_moves(board, fen);
        if depth == 0 {
            return;
        }
        let mut moves = Vec::new();
        board.generate_legal_moves(false, &mut moves);
        for mv in moves {
            let undo = board.make_move(mv);
            walk(board, depth - 1, fen);
            board.undo_move(mv, undo);
        }
    }

    /// The fused legal generator must reproduce the pseudo-generate-then-
    /// filter reference element for element (move ordering ties depend on
    /// the order). Walks full trees from perft/tactical positions; set
    /// RUSTFISH_MOVEGEN_FENS to a `FEN;result` file to also check its first
    /// 20k positions and their children.
    #[test]
    fn fused_legal_movegen_matches_reference() {
        let positions: [(&str, u32); 10] = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3),
            ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3),
            ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 2),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 3),
            ("8/8/8/3pP3/8/8/8/8 w - d6 0 1", 1),
            ("8/P7/8/8/8/8/7p/7k w - - 0 1", 3),
            ("8/8/8/K2pP2q/8/8/8/7k w - d6 0 1", 3),
        ];
        let mut board = Board::new();
        for (fen, depth) in positions {
            board.set_fen(fen);
            walk(&mut board, depth, fen);
        }

        if let Ok(path) = std::env::var("RUSTFISH_MOVEGEN_FENS") {
            let text = std::fs::read_to_string(&path).expect("read RUSTFISH_MOVEGEN_FENS");
            let mut n = 0usize;
            for line in text.lines().take(20_000) {
                let fen = line.split(';').next().unwrap();
                board.set_fen(fen);
                walk(&mut board, 1, fen);
                n += 1;
            }
            eprintln!("fused movegen matched reference on {n} file positions");
        }
    }
}
