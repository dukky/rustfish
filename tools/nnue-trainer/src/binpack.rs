//! Stockfish `.binpack` reader (the format of the Lc0 "T80" training data)
//! and the `binpack` subcommand that converts it to our 32-byte records.
//!
//! Format (as written by Stockfish / nnue-pytorch, `binpack.h`,
//! `training_data_entry.h`, `chess.h`): a file is a sequence of chunks
//! `"BINP" <u32 LE size> <size bytes>`. Chunks are independent, so any
//! prefix of a file that ends on a chunk boundary is a valid binpack (a
//! truncated last chunk is dropped). A chunk is a sequence of
//!
//! ```text
//! stem (32 bytes, big-endian fields):
//!   occupied  u64          squares a1=bit 0 .. h8=bit 63
//!   nibbles   16 bytes     one per occupied square, LSB-first square order,
//!                          low nibble first: piece = 2*type + colour
//!                          (P N B R Q K = 0..5, white = 0), plus
//!                          12 = pawn with the ep square behind it (rank 4: white,
//!                          rank 5: black), 13/14 = white/black rook with its
//!                          castling right, 15 = black king and black to move
//!   move      u16          type<<14 | from<<8 | to<<2 | promo (N B R Q = 0..3);
//!                          type 0 normal, 1 promotion, 2 castle (king takes own
//!                          rook: e1h1, e1a1, ...), 3 en passant
//!   score     u16          zigzag-coded i16, side-to-move POV
//!   ply|res   u16          ply in the low 14 bits, zigzag result (-1/0/1,
//!                          side-to-move POV) in the top 2
//!   rule50    u16
//! count       u16 BE       number of positions chained after the stem
//! movetext    bits, MSB first, `count` (move, score) pairs, padded to a byte
//! ```
//!
//! Each chained move is coded relative to the position it is played in:
//! the index of the moving piece among the mover's pieces (LSB-first, in
//! `bits(popcount - 1)` bits), then the index of the destination among that
//! piece's pseudo-destinations (sliders/knights: attacks minus own pieces;
//! pawns: pushes, double push, captures incl. the ep square, x4 for
//! promotions N B R Q; king: attacks minus own pieces, then one slot per
//! remaining castling right, long first). The score is the side-to-move
//! score, coded as a VLE (4-bit blocks + continuation bit) of the zigzag
//! delta to minus the previous position's score. The result flips sign
//! every ply.
//!
//! The ep square follows `chess.h` exactly: after a double push it is kept
//! only if a pawn of the side to move could capture without exposing its
//! king to a slider, because the move coding depends on it.
//!
//! Scores are Stockfish internal units (T80: converted from Lc0's Q);
//! `32002` (`VALUE_NONE`) marks positions without a score.

use crate::{Rec, REC};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::time::Instant;

pub const VALUE_NONE: i16 = 32002;
pub const NO_SQ: u8 = 64;
/// Empty square in `Board::sq`; pieces are `black << 3 | kind` (our record nibbles).
pub const EMPTY: u8 = 0xFF;
pub const PAWN: u8 = 0;
pub const KNIGHT: u8 = 1;
pub const BISHOP: u8 = 2;
pub const ROOK: u8 = 3;
pub const QUEEN: u8 = 4;
pub const KING: u8 = 5;

// Castling rights bits (as chess.h).
pub const WK: u8 = 1;
pub const WQ: u8 = 2;
pub const BK: u8 = 4;
pub const BQ: u8 = 8;

// ---------------------------------------------------------------- board

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoveKind {
    Normal,
    /// Promoted piece kind (KNIGHT..QUEEN).
    Promotion(u8),
    /// King "captures" its own rook: `to` is the rook square.
    Castle,
    EnPassant,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub kind: MoveKind,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Board {
    pub sq: [u8; 64],
    /// Occupancy by colour (0 = White).
    pub color: [u64; 2],
    /// 0 = White, 1 = Black.
    pub stm: u8,
    /// ep square or `NO_SQ`.
    pub ep: u8,
    pub castling: u8,
    pub rule50: u16,
    pub ply: u16,
}

const fn leaper(deltas: [(i32, i32); 8]) -> [u64; 64] {
    let mut t = [0u64; 64];
    let mut s = 0;
    while s < 64 {
        let (f, r) = ((s % 8) as i32, (s / 8) as i32);
        let mut i = 0;
        while i < 8 {
            let (nf, nr) = (f + deltas[i].0, r + deltas[i].1);
            if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                t[s] |= 1u64 << (nr * 8 + nf);
            }
            i += 1;
        }
        s += 1;
    }
    t
}

const KNIGHT_ATT: [u64; 64] = leaper([(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]);
const KING_ATT: [u64; 64] = leaper([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]);
const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

#[inline]
fn slide(s: u8, occ: u64, dirs: &[(i32, i32); 4]) -> u64 {
    let mut a = 0u64;
    let (f0, r0) = ((s % 8) as i32, (s / 8) as i32);
    for &(df, dr) in dirs {
        let (mut f, mut r) = (f0 + df, r0 + dr);
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let b = 1u64 << (r * 8 + f);
            a |= b;
            if occ & b != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    a
}

#[inline]
pub fn bishop_att(s: u8, occ: u64) -> u64 {
    slide(s, occ, &BISHOP_DIRS)
}
#[inline]
pub fn rook_att(s: u8, occ: u64) -> u64 {
    slide(s, occ, &ROOK_DIRS)
}
/// Squares attacked by a pawn of `color` on `s`.
#[inline]
pub fn pawn_att(s: u8, color: u8) -> u64 {
    let b = 1u64 << s;
    let (w, e) = (b & !0x0101_0101_0101_0101, b & !0x8080_8080_8080_8080);
    if color == 0 {
        (w << 7) | (e << 9)
    } else {
        (w >> 9) | (e >> 7)
    }
}

/// Castling rights kept when a move touches `s` (from or to).
#[inline]
fn preserved(s: u8) -> u8 {
    match s {
        4 => !(WK | WQ),
        60 => !(BK | BQ),
        7 => !WK,
        0 => !WQ,
        63 => !BK,
        56 => !BQ,
        _ => 0xFF,
    }
}

/// `n`-th (0-based) set bit of `v`, LSB first.
#[inline]
fn nth_bit(mut v: u64, n: u32) -> Option<u8> {
    if n >= v.count_ones() {
        return None;
    }
    for _ in 0..n {
        v &= v - 1;
    }
    Some(v.trailing_zeros() as u8)
}

/// Bits needed to code an index below `n` (`usedBitsSafe` in binpack.h).
#[inline]
fn bits_for(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

pub fn zigzag_decode(r: u16) -> i16 {
    let r = r.rotate_right(1);
    (if r & 0x8000 != 0 { r ^ 0x7FFF } else { r }) as i16
}

pub fn zigzag_encode(a: i16) -> u16 {
    let r = a as u16;
    (if r & 0x8000 != 0 { r ^ 0x7FFF } else { r }).rotate_left(1)
}

impl Board {
    fn empty() -> Self {
        Board { sq: [EMPTY; 64], color: [0; 2], stm: 0, ep: NO_SQ, castling: 0, rule50: 0, ply: 0 }
    }

    #[inline]
    fn put(&mut self, s: u8, p: u8) {
        self.sq[s as usize] = p;
        self.color[(p >> 3) as usize] |= 1 << s;
    }

    #[inline]
    fn remove(&mut self, s: u8) {
        let p = self.sq[s as usize];
        if p != EMPTY {
            self.color[(p >> 3) as usize] &= !(1u64 << s);
            self.sq[s as usize] = EMPTY;
        }
    }

    #[inline]
    pub fn occ(&self) -> u64 {
        self.color[0] | self.color[1]
    }

    /// Bitboard of `color`'s pieces of `kind`.
    pub fn pieces(&self, color: u8, kind: u8) -> u64 {
        let (mut bb, mut out) = (self.color[color as usize], 0u64);
        while bb != 0 {
            let s = bb.trailing_zeros();
            bb &= bb - 1;
            if self.sq[s as usize] & 7 == kind {
                out |= 1 << s;
            }
        }
        out
    }

    pub fn king_sq(&self, color: u8) -> Option<u8> {
        let k = self.pieces(color, KING);
        (k != 0).then(|| k.trailing_zeros() as u8)
    }

    /// Is `s` attacked by `by`'s pieces (occupancy `occ`)?
    pub fn attacked(&self, s: u8, by: u8, occ: u64) -> bool {
        let them = self.color[by as usize];
        let mut bb = them;
        while bb != 0 {
            let f = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            let hit = match self.sq[f as usize] & 7 {
                PAWN => pawn_att(f, by),
                KNIGHT => KNIGHT_ATT[f as usize],
                BISHOP => bishop_att(f, occ),
                ROOK => rook_att(f, occ),
                QUEEN => bishop_att(f, occ) | rook_att(f, occ),
                _ => KING_ATT[f as usize],
            };
            if hit >> s & 1 != 0 {
                return true;
            }
        }
        false
    }

    pub fn in_check(&self) -> bool {
        match self.king_sq(self.stm) {
            Some(k) => self.attacked(k, self.stm ^ 1, self.occ()),
            None => false,
        }
    }

    /// `chess.h` `isEpPossible`: can the side to move capture on `ep`
    /// without exposing its king to a slider?
    fn ep_possible(&self, ep: u8) -> bool {
        let stm = self.stm;
        let attackers = pawn_att(ep, stm ^ 1) & self.pieces(stm, PAWN);
        if attackers == 0 {
            return false;
        }
        let fwd: i32 = if stm == 0 { 8 } else { -8 };
        let at = |d: i32| self.sq[(ep as i32 + d) as usize];
        if self.sq[ep as usize] != EMPTY || at(fwd) != EMPTY || at(-fwd) != ((stm ^ 1) << 3 | PAWN) {
            return false;
        }
        let Some(k) = self.king_sq(stm) else { return false };
        let them = stm ^ 1;
        let (b, r, q) = (self.pieces(them, BISHOP), self.pieces(them, ROOK), self.pieces(them, QUEEN));
        let sliders = b | r | q;
        let queen_rays = bishop_att(k, 0) | rook_att(k, 0);
        let mut a = attackers;
        while a != 0 {
            let s = a.trailing_zeros() as u8;
            a &= a - 1;
            if sliders & queen_rays == 0 {
                return true;
            }
            let captured = (s & 56) | (ep & 7);
            let occ = ((self.occ() ^ (1 << s)) | (1 << ep)) ^ (1 << captured);
            if bishop_att(k, occ) & (b | q) == 0 && rook_att(k, occ) & (r | q) == 0 {
                return true;
            }
        }
        false
    }

    /// `chess.h` `Position::doMove` (no legality checks).
    pub fn do_move(&mut self, m: Move) {
        let p = self.sq[m.from as usize];
        let kind = p & 7;
        self.ply = self.ply.wrapping_add(1);
        self.rule50 = self.rule50.wrapping_add(1);
        if m.kind != MoveKind::Castle && (kind == PAWN || self.sq[m.to as usize] != EMPTY) {
            self.rule50 = 0;
        }
        self.castling &= preserved(m.from) & preserved(m.to);
        self.ep = NO_SQ;
        if kind == PAWN && (m.from ^ m.to) == 16 {
            self.ep = (m.from + m.to) / 2;
        }
        match m.kind {
            MoveKind::Normal => {
                self.remove(m.to);
                self.remove(m.from);
                self.put(m.to, p);
            }
            MoveKind::Promotion(k) => {
                self.remove(m.to);
                self.remove(m.from);
                self.put(m.to, (p & 8) | k);
            }
            MoveKind::EnPassant => {
                self.remove((m.from & 56) | (m.to & 7));
                self.remove(m.from);
                self.put(m.to, p);
            }
            MoveKind::Castle => {
                let rook = self.sq[m.to as usize];
                let base = m.from & 56;
                let short = m.to & 7 == 7;
                self.remove(m.from);
                self.remove(m.to);
                self.put(base | if short { 6 } else { 2 }, p);
                self.put(base | if short { 5 } else { 3 }, rook);
            }
        }
        self.stm ^= 1;
        if self.ep != NO_SQ && !self.ep_possible(self.ep) {
            self.ep = NO_SQ;
        }
    }

    pub fn fen(&self) -> String {
        let mut s = String::with_capacity(90);
        for r in (0..8).rev() {
            let mut gap = 0;
            for f in 0..8 {
                let p = self.sq[r * 8 + f];
                if p == EMPTY {
                    gap += 1;
                    continue;
                }
                if gap > 0 {
                    s.push((b'0' + gap) as char);
                    gap = 0;
                }
                let c = b"pnbrqk"[(p & 7) as usize] as char;
                s.push(if p >> 3 == 0 { c.to_ascii_uppercase() } else { c });
            }
            if gap > 0 {
                s.push((b'0' + gap) as char);
            }
            if r > 0 {
                s.push('/');
            }
        }
        s.push_str(if self.stm == 0 { " w " } else { " b " });
        if self.castling == 0 {
            s.push('-');
        }
        for (bit, c) in [(WK, 'K'), (WQ, 'Q'), (BK, 'k'), (BQ, 'q')] {
            if self.castling & bit != 0 {
                s.push(c);
            }
        }
        s.push(' ');
        if self.ep == NO_SQ {
            s.push('-');
        } else {
            s.push_str(&sq_name(self.ep));
        }
        s.push_str(&format!(" {} {}", self.rule50, self.ply / 2 + 1));
        s
    }

    /// FEN -> board (ep square taken as given; rights as given).
    pub fn from_fen(fen: &str) -> Option<Board> {
        let mut b = Board::empty();
        let mut it = fen.split_whitespace();
        let (mut r, mut f) = (7i32, 0i32);
        for c in it.next()?.chars() {
            match c {
                '/' => {
                    r -= 1;
                    f = 0;
                }
                '1'..='8' => f += c as i32 - '0' as i32,
                _ => {
                    let kind = "pnbrqk".find(c.to_ascii_lowercase())? as u8;
                    if !(0..8).contains(&r) || !(0..8).contains(&f) {
                        return None;
                    }
                    b.put((r * 8 + f) as u8, (c.is_ascii_lowercase() as u8) << 3 | kind);
                    f += 1;
                }
            }
        }
        b.stm = match it.next()? {
            "w" => 0,
            "b" => 1,
            _ => return None,
        };
        for c in it.next()?.chars() {
            b.castling |= match c {
                'K' => WK,
                'Q' => WQ,
                'k' => BK,
                'q' => BQ,
                _ => 0,
            };
        }
        b.ep = match it.next()? {
            "-" => NO_SQ,
            s => parse_sq(s)?,
        };
        b.rule50 = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
        let full: u16 = it.next().and_then(|x| x.parse().ok()).unwrap_or(1);
        b.ply = (full.max(1) - 1) * 2 + b.stm as u16;
        Some(b)
    }
}

pub fn sq_name(s: u8) -> String {
    format!("{}{}", (b'a' + s % 8) as char, s / 8 + 1)
}

fn parse_sq(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
        return None;
    }
    Some((b[1] - b'1') * 8 + (b[0] - b'a'))
}

/// UCI string (castling as king to g/c file).
pub fn uci(m: Move) -> String {
    let to = match m.kind {
        MoveKind::Castle => (m.from & 56) | if m.to & 7 == 7 { 6 } else { 2 },
        _ => m.to,
    };
    let mut s = sq_name(m.from) + &sq_name(to);
    if let MoveKind::Promotion(k) = m.kind {
        s.push(b"pnbrqk"[k as usize] as char);
    }
    s
}

// ---------------------------------------------------------------- entries

#[derive(Clone, Debug)]
pub struct Entry {
    /// Position before `mv`; `board.ply` is the game ply.
    pub board: Board,
    /// Move played from `board` (the best move for Lc0 data).
    pub mv: Move,
    /// Side-to-move score, Stockfish units; `VALUE_NONE` = none.
    pub score: i16,
    /// Game result, side-to-move POV: -1 / 0 / 1.
    pub result: i8,
}

impl Entry {
    /// Captures a piece (castling and en passant excluded, as `isCapturingMove`).
    pub fn is_capture(&self) -> bool {
        let t = self.board.sq[self.mv.to as usize];
        self.mv.kind != MoveKind::Castle && t != EMPTY && t >> 3 != self.board.stm
    }
}

/// Decodes a 32-byte stem.
pub fn unpack_stem(b: &[u8]) -> Result<Entry, &'static str> {
    let be = |i: usize| u16::from_be_bytes([b[i], b[i + 1]]);
    let occ = u64::from_be_bytes(b[..8].try_into().unwrap());
    if occ.count_ones() > 32 {
        return Err("stem: more than 32 pieces");
    }
    let mut board = Board::empty();
    let (mut bb, mut i) = (occ, 0usize);
    while bb != 0 {
        let s = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        let nib = b[8 + i / 2] >> (4 * (i & 1)) & 0xF;
        i += 1;
        let (r, piece) = (s / 8, |kind: u8, black: u8| black << 3 | kind);
        let p = match nib {
            0..=11 => piece(nib / 2, nib & 1),
            12 => match r {
                3 => {
                    board.ep = s - 8;
                    piece(PAWN, 0)
                }
                4 => {
                    board.ep = s + 8;
                    piece(PAWN, 1)
                }
                _ => return Err("stem: ep pawn not on rank 4/5"),
            },
            13 => {
                board.castling |= match s {
                    0 => WQ,
                    7 => WK,
                    _ => return Err("stem: castling rook off a1/h1"),
                };
                piece(ROOK, 0)
            }
            14 => {
                board.castling |= match s {
                    56 => BQ,
                    63 => BK,
                    _ => return Err("stem: castling rook off a8/h8"),
                };
                piece(ROOK, 1)
            }
            _ => {
                board.stm = 1;
                piece(KING, 1)
            }
        };
        board.put(s, p);
    }
    if board.pieces(0, KING).count_ones() != 1 || board.pieces(1, KING).count_ones() != 1 {
        return Err("stem: not one king per side");
    }
    let m = be(24);
    let (from, to) = ((m >> 8 & 63) as u8, (m >> 2 & 63) as u8);
    let kind = match m >> 14 {
        0 => MoveKind::Normal,
        1 => MoveKind::Promotion(KNIGHT + (m & 3) as u8),
        2 => MoveKind::Castle,
        _ => MoveKind::EnPassant,
    };
    let pr = be(28);
    board.ply = pr & 0x3FFF;
    board.rule50 = be(30);
    let entry = Entry { board, mv: Move { from, to, kind }, score: zigzag_decode(be(26)), result: zigzag_decode(pr >> 14) as i8 };
    if entry.board.sq[from as usize] == EMPTY || entry.board.sq[from as usize] >> 3 != entry.board.stm {
        return Err("stem: move does not start on a side-to-move piece");
    }
    Ok(entry)
}

/// MSB-first bit reader over the movetext (`extractBitsLE8`).
struct Bits<'a> {
    data: &'a [u8],
    pos: usize, // in bits
}

impl Bits<'_> {
    #[inline]
    fn read(&mut self, n: u32) -> Result<u32, &'static str> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = *self.data.get(self.pos / 8).ok_or("movetext runs past the chunk")?;
            v = v << 1 | (byte >> (7 - self.pos % 8) & 1) as u32;
            self.pos += 1;
        }
        Ok(v)
    }

    fn vle16(&mut self) -> Result<u16, &'static str> {
        let (mut v, mut shift) = (0u32, 0u32);
        loop {
            let block = self.read(5)?;
            v |= (block & 15) << shift;
            if block >> 4 == 0 {
                return Ok(v as u16);
            }
            shift += 4;
            if shift > 16 {
                return Err("score VLE too long");
            }
        }
    }

    fn bytes(&self) -> usize {
        self.pos.div_ceil(8)
    }
}

/// Pseudo-destinations of the pawn on `from` as the move coding sees them.
fn pawn_dests(b: &Board, from: u8) -> u64 {
    let stm = b.stm;
    let mut targets = b.color[(stm ^ 1) as usize];
    if b.ep != NO_SQ {
        targets |= 1 << b.ep;
    }
    let mut d = pawn_att(from, stm) & targets;
    let occ = b.occ();
    let fwd = if stm == 0 { from + 8 } else { from - 8 };
    if occ >> fwd & 1 == 0 {
        d |= 1 << fwd;
        let start = if stm == 0 { 1 } else { 6 };
        let fwd2 = if stm == 0 { fwd + 8 } else { fwd - 8 };
        if from / 8 == start && occ >> fwd2 & 1 == 0 {
            d |= 1 << fwd2;
        }
    }
    d
}

fn piece_dests(b: &Board, from: u8, kind: u8) -> u64 {
    let (occ, own) = (b.occ(), b.color[b.stm as usize]);
    let a = match kind {
        KNIGHT => KNIGHT_ATT[from as usize],
        BISHOP => bishop_att(from, occ),
        ROOK => rook_att(from, occ),
        QUEEN => bishop_att(from, occ) | rook_att(from, occ),
        _ => KING_ATT[from as usize],
    };
    a & !own
}

fn our_rights(b: &Board) -> (bool, bool, u32) {
    let (long, short) = if b.stm == 0 { (WQ, WK) } else { (BQ, BK) };
    let (l, s) = (b.castling & long != 0, b.castling & short != 0);
    (l, s, l as u32 + s as u32)
}

/// Next chained move in position `b` (`nextMoveScore`, minus the score).
fn read_move(bits: &mut Bits, b: &Board) -> Result<Move, &'static str> {
    let own = b.color[b.stm as usize];
    let id = bits.read(bits_for(own.count_ones()))?;
    let from = nth_bit(own, id).ok_or("piece index out of range")?;
    match b.sq[from as usize] & 7 {
        PAWN => {
            let d = pawn_dests(b, from);
            let promo_rank = if b.stm == 0 { 6 } else { 1 };
            if from / 8 == promo_rank {
                let id = bits.read(bits_for(d.count_ones() * 4))?;
                let to = nth_bit(d, id / 4).ok_or("pawn move index out of range")?;
                Ok(Move { from, to, kind: MoveKind::Promotion(KNIGHT + (id % 4) as u8) })
            } else {
                let id = bits.read(bits_for(d.count_ones()))?;
                let to = nth_bit(d, id).ok_or("pawn move index out of range")?;
                let kind = if to == b.ep { MoveKind::EnPassant } else { MoveKind::Normal };
                Ok(Move { from, to, kind })
            }
        }
        KING => {
            let a = piece_dests(b, from, KING);
            let n = a.count_ones();
            let (long, _, nc) = our_rights(b);
            let id = bits.read(bits_for(n + nc))?;
            if id >= n {
                if id - n >= nc {
                    return Err("castling index out of range");
                }
                let base = if b.stm == 0 { 0 } else { 56 };
                let to = if id == n && long { base } else { base + 7 };
                Ok(Move { from, to, kind: MoveKind::Castle })
            } else {
                Ok(Move { from, to: nth_bit(a, id).unwrap(), kind: MoveKind::Normal })
            }
        }
        k => {
            let a = piece_dests(b, from, k);
            let id = bits.read(bits_for(a.count_ones()))?;
            Ok(Move { from, to: nth_bit(a, id).ok_or("move index out of range")?, kind: MoveKind::Normal })
        }
    }
}

/// Calls `f` on every entry of one chunk's payload, in file order. On a
/// decode error the entries already passed to `f` stand and the rest of the
/// chunk is abandoned.
pub fn for_each_entry(chunk: &[u8], mut f: impl FnMut(&Entry)) -> Result<(), &'static str> {
    let mut off = 0usize;
    while off + REC + 2 <= chunk.len() {
        let mut e = unpack_stem(&chunk[off..off + 32])?;
        off += 32;
        let n = u16::from_be_bytes([chunk[off], chunk[off + 1]]);
        off += 2;
        f(&e);
        if n == 0 {
            continue;
        }
        let mut bits = Bits { data: &chunk[off..], pos: 0 };
        let mut last = e.score.wrapping_neg();
        for _ in 0..n {
            e.board.do_move(e.mv);
            e.mv = read_move(&mut bits, &e.board)?;
            let delta = zigzag_decode(bits.vle16()? as u16);
            e.score = last.wrapping_add(delta);
            last = e.score.wrapping_neg();
            e.result = -e.result;
            f(&e);
        }
        off += bits.bytes();
    }
    Ok(())
}

/// Streams the chunks of a binpack. `Ok(None)` at a clean end; a truncated
/// last chunk (a byte-range download) also ends the stream, counted in
/// `truncated`.
pub struct ChunkReader<R: Read> {
    r: R,
    pub truncated: bool,
}

impl<R: Read> ChunkReader<R> {
    pub fn new(r: R) -> Self {
        ChunkReader { r, truncated: false }
    }

    pub fn next_into(&mut self, buf: &mut Vec<u8>) -> Result<bool, String> {
        let mut h = [0u8; 8];
        let got = read_full(&mut self.r, &mut h)?;
        if got == 0 {
            return Ok(false);
        }
        if got < 8 {
            self.truncated = true;
            return Ok(false);
        }
        if &h[..4] != b"BINP" {
            return Err(format!("bad chunk magic {:02x?}", &h[..4]));
        }
        let size = u32::from_le_bytes(h[4..].try_into().unwrap()) as usize;
        if size > 100 << 20 {
            return Err(format!("chunk size {size} > 100 MiB: malformed file?"));
        }
        buf.resize(size, 0);
        if read_full(&mut self.r, buf)? < size {
            self.truncated = true;
            return Ok(false);
        }
        Ok(true)
    }
}

fn read_full(r: &mut impl Read, buf: &mut [u8]) -> Result<usize, String> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(n)
}

// ---------------------------------------------------------------- conversion

/// Entry -> our record: White-POV score `round(score * scale)` (clamped to
/// +-32000), result 0/1/2 White POV, side to move in byte 27.
pub fn to_record(e: &Entry, scale: f32) -> Rec {
    let b = &e.board;
    let mut rec = [0u8; REC];
    let occ = b.occ();
    rec[..8].copy_from_slice(&occ.to_le_bytes());
    let (mut bb, mut n) = (occ, 0usize);
    while bb != 0 {
        let s = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        rec[8 + n / 2] |= b.sq[s] << (4 * (n & 1));
        n += 1;
    }
    let white = if b.stm == 1 { -(e.score as f32) } else { e.score as f32 };
    let s = (white * scale).round().clamp(-32000.0, 32000.0) as i16;
    rec[24..26].copy_from_slice(&s.to_le_bytes());
    let res_white = if b.stm == 1 { -e.result } else { e.result };
    rec[26] = (res_white + 1) as u8;
    rec[27] = b.stm;
    rec
}

/// Skip reasons, in the order they are tested (first match counts).
pub const REASONS: [&str; 8] =
    ["value_none", "placeholder_zero", "min_ply", "in_check", "capture", "en_passant", "promotion", "score_cap"];

pub struct Filter {
    /// Skip |score| > max_score (raw binpack units).
    pub max_score: i32,
    pub min_ply: u16,
    // Placeholder-zero heuristic state (nnue-pytorch `skip_placeholder_zero`).
    last_ply: i32,
    last_score: i32,
}

impl Filter {
    pub fn new(max_score: i32, min_ply: u16) -> Self {
        Filter { max_score, min_ply, last_ply: -1, last_score: VALUE_NONE as i32 }
    }

    /// `None` = keep, else the index into `REASONS`. Must see every entry in
    /// file order (the placeholder heuristic looks at the previous one).
    pub fn check(&mut self, e: &Entry) -> Option<usize> {
        let (ply, score) = (e.board.ply as i32, e.score as i32);
        // nnue-pytorch: a 0 score in a decisive game right after a sizeable
        // score was a placeholder for "no score".
        let placeholder = ply > self.last_ply
            && self.last_score != VALUE_NONE as i32
            && self.last_score.abs() > 100
            && e.result != 0
            && score == 0;
        self.last_ply = ply;
        if e.score == VALUE_NONE {
            return Some(0);
        }
        if placeholder {
            return Some(1);
        }
        self.last_score = score;
        if e.board.ply < self.min_ply {
            return Some(2);
        }
        if e.board.in_check() {
            return Some(3);
        }
        if e.is_capture() {
            return Some(4);
        }
        match e.mv.kind {
            MoveKind::EnPassant => return Some(5),
            MoveKind::Promotion(_) => return Some(6),
            _ => {}
        }
        if score.abs() > self.max_score {
            return Some(7);
        }
        None
    }
}

/// `nnue-trainer binpack <in.binpack|-> <out.bin> [options]`
pub fn cmd_binpack(args: &[String]) {
    let usage = "usage: binpack <in.binpack|-> <out.bin> [--score-scale F] [--max-score N] [--min-ply N] [--max N] [--fens <file.txt> [--fens-every K]] [--no-filter]";
    let (mut paths, mut scale, mut max_score, mut min_ply, mut max) = (Vec::new(), 1.0f32, 10_000i32, 0u16, u64::MAX);
    let (mut fens, mut every, mut no_filter) = (None, 1u64, false);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| panic!("{a} needs a value\n{usage}")).clone();
        match a.as_str() {
            "--score-scale" => scale = v().parse().unwrap(),
            "--max-score" => max_score = v().parse().unwrap(),
            "--min-ply" => min_ply = v().parse().unwrap(),
            "--max" => max = v().parse().unwrap(),
            "--fens" => fens = Some(v()),
            "--fens-every" => every = v().parse::<u64>().unwrap().max(1),
            "--no-filter" => no_filter = true,
            o if o.starts_with("--") => panic!("unknown option {o}\n{usage}"),
            _ => paths.push(a.clone()),
        }
    }
    assert!(paths.len() == 2, "{usage}");
    let input: Box<dyn Read> = if paths[0] == "-" {
        Box::new(BufReader::with_capacity(1 << 22, std::io::stdin()))
    } else {
        Box::new(BufReader::with_capacity(1 << 22, File::open(&paths[0]).unwrap_or_else(|e| panic!("{}: {e}", paths[0]))))
    };
    let mut out = BufWriter::with_capacity(1 << 22, File::create(&paths[1]).unwrap());
    let mut fen_out = fens.map(|p| BufWriter::new(File::create(p).unwrap()));

    let mut reader = ChunkReader::new(input);
    let mut filter = Filter::new(max_score, min_ply);
    let mut chunk = Vec::new();
    let (mut chunks, mut bad_chunks, mut total, mut kept) = (0u64, 0u64, 0u64, 0u64);
    let mut skipped = [0u64; REASONS.len()];
    // |raw score| histogram of kept positions, and results (White POV) of kept.
    const EDGES: [i32; 9] = [25, 50, 100, 200, 400, 800, 1600, 3200, 10000];
    let mut hist = [0u64; EDGES.len() + 1];
    let (mut sum_abs, mut results) = (0f64, [0u64; 3]);
    // Decisive kept positions with |score| > 200: does the score's sign match the result?
    let (mut agree, mut decided) = (0u64, 0u64);
    let t0 = Instant::now();
    'outer: loop {
        match reader.next_into(&mut chunk) {
            Ok(true) => {}
            Ok(false) => break,
            Err(e) => {
                eprintln!("stopping: {e}");
                break;
            }
        }
        chunks += 1;
        let mut done = false;
        let res = for_each_entry(&chunk, |e| {
            if done {
                return;
            }
            total += 1;
            if let Some(r) = filter.check(e) {
                skipped[r] += 1;
                if !no_filter {
                    return;
                }
            }
            let rec = to_record(e, scale);
            out.write_all(&rec).unwrap();
            kept += 1;
            let s = e.score as i32;
            hist[EDGES.iter().position(|&x| s.abs() < x).unwrap_or(EDGES.len())] += 1;
            sum_abs += s.abs() as f64;
            results[rec[26] as usize] += 1;
            if e.result != 0 && s.abs() > 200 {
                decided += 1;
                agree += ((s > 0) == (e.result > 0)) as u64;
            }
            if let Some(w) = fen_out.as_mut() {
                if kept % every == 0 {
                    let white = i16::from_le_bytes([rec[24], rec[25]]);
                    let res = ["0.0", "0.5", "1.0"][rec[26] as usize];
                    writeln!(w, "{} | {white} | {res} | {} | {} | {}", e.board.fen(), uci(e.mv), e.score, e.board.ply).unwrap();
                }
            }
            if kept >= max {
                done = true;
            }
        });
        if let Err(e) = res {
            bad_chunks += 1;
            eprintln!("chunk {chunks}: {e} (rest of chunk skipped)");
        }
        if done {
            break 'outer;
        }
    }
    out.flush().unwrap();
    if let Some(mut w) = fen_out {
        w.flush().unwrap();
    }
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "{}: {chunks} chunks{}{}, {total} positions in {secs:.1}s ({:.2}M pos/s), kept {kept} ({:.1}%)",
        paths[1],
        if bad_chunks > 0 { format!(", {bad_chunks} with decode errors") } else { String::new() },
        if reader.truncated { ", last chunk truncated (dropped)" } else { "" },
        total as f64 / secs / 1e6,
        100.0 * kept as f64 / total.max(1) as f64,
    );
    let skip: Vec<String> =
        REASONS.iter().zip(skipped).map(|(r, n)| format!("{r} {n} ({:.1}%)", 100.0 * n as f64 / total.max(1) as f64)).collect();
    println!("skipped: {}", skip.join(", "));
    let k = kept.max(1) as f64;
    let mut h = Vec::new();
    for (i, n) in hist.iter().enumerate() {
        let lo = if i == 0 { 0 } else { EDGES[i - 1] };
        let hi = EDGES.get(i).map(|x| x.to_string()).unwrap_or("inf".into());
        h.push(format!("[{lo},{hi}) {:.1}%", 100.0 * *n as f64 / k));
    }
    println!("kept |raw score|: mean {:.1}; {}", sum_abs / k, h.join(" "));
    println!(
        "kept results (White POV): win {:.1}% draw {:.1}% loss {:.1}%; score sign agrees with result in {:.1}% of decisive |score|>200",
        100.0 * results[2] as f64 / k,
        100.0 * results[1] as f64 / k,
        100.0 * results[0] as f64 / k,
        100.0 * agree as f64 / decided.max(1) as f64,
    );
    if scale != 1.0 {
        println!("scores written x{scale} (White POV, clamped to +-32000)");
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    // ---- a test-only writer (the inverse of the reader, from binpack.h's
    // `packEntry` / `addMoveScore`), to build chunks by hand.

    fn pack_stem(e: &Entry) -> [u8; 32] {
        let b = &e.board;
        let mut out = [0u8; 32];
        let occ = b.occ();
        out[..8].copy_from_slice(&occ.to_be_bytes());
        let (mut bb, mut i) = (occ, 0);
        while bb != 0 {
            let s = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            let p = b.sq[s as usize];
            let (kind, black) = (p & 7, p >> 3);
            let mut nib = 2 * kind + black;
            if kind == PAWN && b.ep != NO_SQ && s % 8 == b.ep % 8 && ((s / 8 == 3 && b.stm == 1) || (s / 8 == 4 && b.stm == 0)) {
                nib = 12;
            }
            if kind == ROOK {
                let right = match (s, black) {
                    (0, 0) => WQ,
                    (7, 0) => WK,
                    (56, 1) => BQ,
                    (63, 1) => BK,
                    _ => 0,
                };
                if b.castling & right != 0 {
                    nib = 13 + black;
                }
            }
            if kind == KING && black == 1 && b.stm == 1 {
                nib = 15;
            }
            out[8 + i / 2] |= nib << (4 * (i & 1));
            i += 1;
        }
        let m = &e.mv;
        let (ty, promo) = match m.kind {
            MoveKind::Normal => (0u16, 0u16),
            MoveKind::Promotion(k) => (1, (k - KNIGHT) as u16),
            MoveKind::Castle => (2, 0),
            MoveKind::EnPassant => (3, 0),
        };
        let mv = ty << 14 | (m.from as u16) << 8 | (m.to as u16) << 2 | promo;
        out[24..26].copy_from_slice(&mv.to_be_bytes());
        out[26..28].copy_from_slice(&zigzag_encode(e.score).to_be_bytes());
        let pr = b.ply | zigzag_encode(e.result as i16) << 14;
        out[28..30].copy_from_slice(&pr.to_be_bytes());
        out[30..32].copy_from_slice(&b.rule50.to_be_bytes());
        out
    }

    #[derive(Default)]
    struct BitWriter {
        bytes: Vec<u8>,
        n: usize,
    }

    impl BitWriter {
        fn put(&mut self, v: u32, n: u32) {
            for i in (0..n).rev() {
                if self.n % 8 == 0 {
                    self.bytes.push(0);
                }
                let bit = (v >> i & 1) as u8;
                *self.bytes.last_mut().unwrap() |= bit << (7 - self.n % 8);
                self.n += 1;
            }
        }
        fn vle16(&mut self, mut v: u16) {
            loop {
                let more = v > 15;
                self.put((v & 15) as u32 | (more as u32) << 4, 5);
                v >>= 4;
                if v == 0 {
                    break;
                }
            }
        }
    }

    fn index_of(bb: u64, s: u8) -> u32 {
        (bb & ((1u64 << s) - 1)).count_ones()
    }

    fn write_move(w: &mut BitWriter, b: &Board, m: Move) {
        let own = b.color[b.stm as usize];
        w.put(index_of(own, m.from), bits_for(own.count_ones()));
        match b.sq[m.from as usize] & 7 {
            PAWN => {
                let d = pawn_dests(b, m.from);
                let mut id = index_of(d, m.to);
                let mut n = d.count_ones();
                if let MoveKind::Promotion(k) = m.kind {
                    id = id * 4 + (k - KNIGHT) as u32;
                    n *= 4;
                }
                w.put(id, bits_for(n));
            }
            KING => {
                let a = piece_dests(b, m.from, KING);
                let (long, _, nc) = our_rights(b);
                let n = a.count_ones();
                let id = if m.kind == MoveKind::Castle {
                    // binpack.h: attacks - 1, +1 if the long right exists, +1 for short.
                    n - 1 + long as u32 + (m.to & 7 == 7) as u32
                } else {
                    index_of(a, m.to)
                };
                w.put(id, bits_for(n + nc));
            }
            k => {
                let a = piece_dests(b, m.from, k);
                w.put(index_of(a, m.to), bits_for(a.count_ones()));
            }
        }
    }

    /// One stem + chained (move, score) list as a chunk payload.
    fn build_chain(start: &Board, moves: &[(Move, i16)], result: i8) -> Vec<u8> {
        let (m0, s0) = moves[0];
        let e = Entry { board: start.clone(), mv: m0, score: s0, result };
        let mut out = pack_stem(&e).to_vec();
        out.extend_from_slice(&((moves.len() - 1) as u16).to_be_bytes());
        let mut w = BitWriter::default();
        let mut b = start.clone();
        let mut last = s0.wrapping_neg();
        let mut mv = m0;
        for &(m, s) in &moves[1..] {
            b.do_move(mv);
            write_move(&mut w, &b, m);
            w.vle16(zigzag_encode(s.wrapping_sub(last)));
            last = s.wrapping_neg();
            mv = m;
        }
        out.extend_from_slice(&w.bytes);
        out
    }

    fn mv(b: &Board, u: &str) -> Move {
        let from = parse_sq(&u[..2]).unwrap();
        let mut to = parse_sq(&u[2..4]).unwrap();
        let p = b.sq[from as usize];
        let kind = if p & 7 == KING && (from as i32 - to as i32).abs() == 2 {
            to = (from & 56) | if to & 7 == 6 { 7 } else { 0 };
            MoveKind::Castle
        } else if let Some(c) = u.chars().nth(4) {
            MoveKind::Promotion("pnbrqk".find(c).unwrap() as u8)
        } else if p & 7 == PAWN && to == b.ep {
            MoveKind::EnPassant
        } else {
            MoveKind::Normal
        };
        Move { from, to, kind }
    }

    /// Plays `ucis` from `fen` with scores 10, -11, 12, ... and checks every
    /// decoded entry against `expect` (the FEN each move is played from).
    fn check_game(fen: &str, ucis: &[&str], expect: &[&str], result: i8) {
        let start = Board::from_fen(fen).unwrap();
        let mut b = start.clone();
        let mut moves = Vec::new();
        for (i, u) in ucis.iter().enumerate() {
            let m = mv(&b, u);
            let s = if i % 2 == 0 { 10 + i as i16 } else { -(10 + i as i16) };
            moves.push((m, s));
            b.do_move(m);
        }
        let chunk = build_chain(&start, &moves, result);
        let mut got = Vec::new();
        for_each_entry(&chunk, |e| got.push(e.clone())).unwrap();
        assert_eq!(got.len(), ucis.len());
        for (i, e) in got.iter().enumerate() {
            assert_eq!(e.board.fen(), expect[i], "position before move {i} ({})", ucis[i]);
            assert_eq!(uci(e.mv), ucis[i]);
            assert_eq!(e.score, moves[i].1);
            assert_eq!(e.result, if i % 2 == 0 { result } else { -result });
        }
    }

    #[test]
    fn zigzag() {
        for a in [0i16, 1, -1, 2, -20, 32002, -32768, 32767] {
            assert_eq!(zigzag_decode(zigzag_encode(a)), a);
        }
        assert_eq!(zigzag_encode(-20), 39);
        assert_eq!(zigzag_decode(1), -1);
        assert_eq!(zigzag_decode(2), 1);
    }

    /// Hand-assembled bytes: startpos stem (as the first 32 bytes of the real
    /// test80-2024-01 file: d2d4, score VALUE_NONE) + one chained move whose
    /// bits were computed by hand from the spec.
    #[test]
    fn hand_built_chunk() {
        let mut c = vec![0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0x2d, 0x84, 0x4a, 0xd2, 0, 0, 0, 0, 0x11, 0x11, 0x11, 0x11, 0x3e, 0x95, 0x5b, 0xe3];
        c.extend_from_slice(&[0x0b, 0x6c, 0xfa, 0x04, 0, 0, 0, 0]);
        // 1 chained ply: Black e7e5 = piece 4 of 16 (4 bits 0100), dest 0 of
        // {e5, e6} (1 bit), score -20 as a delta to -VALUE_NONE (wrapping).
        let delta = zigzag_encode((-20i16).wrapping_sub(VALUE_NONE.wrapping_neg()));
        let mut w = BitWriter::default();
        w.put(0b0100, 4);
        w.put(0, 1);
        w.vle16(delta);
        c.extend_from_slice(&[0, 1]);
        c.extend_from_slice(&w.bytes);
        let mut got = Vec::new();
        for_each_entry(&c, |e| got.push(e.clone())).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].board.fen(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(uci(got[0].mv), "d2d4");
        assert_eq!(got[0].score, VALUE_NONE);
        assert_eq!(got[1].board.fen(), "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1");
        assert_eq!(uci(got[1].mv), "e7e5");
        assert_eq!(got[1].score, -20);

        // The same move bits with a stem score of 0 and score -20: bytes fully by hand.
        // 0100 | 0 | VLE(39) = 10111 00010 -> 0100 0101 1100 010x = 0x45 0xC4.
        let mut c2 = c[..32].to_vec();
        c2[26..28].copy_from_slice(&[0, 0]);
        c2.extend_from_slice(&[0, 1, 0x45, 0xC4]);
        let mut got = Vec::new();
        for_each_entry(&c2, |e| got.push(e.clone())).unwrap();
        assert_eq!(uci(got[1].mv), "e7e5");
        assert_eq!(got[1].score, -20);
    }

    #[test]
    fn opening_with_ep_and_both_short_castles() {
        let ucis = ["e2e4", "d7d5", "e4e5", "f7f5", "e5f6", "g7f6", "g1f3", "g8h6", "f1c4", "f8g7", "e1g1", "e8g8", "d2d3"];
        let expect = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "rnbqkbnr/ppp1p1pp/5P2/3p4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3",
            "rnbqkbnr/ppp1p2p/5p2/3p4/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 4",
            "rnbqkbnr/ppp1p2p/5p2/3p4/8/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 4",
            "rnbqkb1r/ppp1p2p/5p1n/3p4/8/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 5",
            "rnbqkb1r/ppp1p2p/5p1n/3p4/2B5/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 5",
            "rnbqk2r/ppp1p1bp/5p1n/3p4/2B5/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 6",
            "rnbqk2r/ppp1p1bp/5p1n/3p4/2B5/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 6",
            "rnbq1rk1/ppp1p1bp/5p1n/3p4/2B5/5N2/PPPP1PPP/RNBQ1RK1 w - - 6 7",
        ];
        check_game(expect[0], &ucis, &expect, 1);
    }

    #[test]
    fn long_castle_promotions_and_check() {
        let ucis = ["e1c1", "g2h1n", "b7a8r", "e8e7", "a8a1"];
        let expect = [
            "r3k3/1P6/8/8/8/8/6p1/R3K2R w KQq - 0 40",
            "r3k3/1P6/8/8/8/8/6p1/2KR3R b q - 1 40",
            "r3k3/1P6/8/8/8/8/8/2KR3n w q - 0 41",
            "R3k3/8/8/8/8/8/8/2KR3n b - - 0 41",
            "R7/4k3/8/8/8/8/8/2KR3n w - - 1 42",
        ];
        check_game(expect[0], &ucis, &expect, -1);
        // Black long castle; then a white queen promotion.
        let ucis = ["e8c8", "a7a8q"];
        let expect = ["r3k2r/P7/8/8/8/8/8/4K3 b kq - 0 30", "2kr3r/P7/8/8/8/8/8/4K3 w - - 1 31"];
        check_game(expect[0], &ucis, &expect, 0);
        // Black short castle with only the short right left.
        let ucis = ["e8g8", "e1d1"];
        let expect = ["4k2r/8/8/8/8/8/8/4K3 b k - 0 30", "5rk1/8/8/8/8/8/8/4K3 w - - 1 31"];
        check_game(expect[0], &ucis, &expect, 0);
    }

    #[test]
    fn ep_square_kept_only_when_capture_is_legal() {
        // d4 pawn can take e3, but a4 king / h4 rook pin it along the rank.
        let expect = ["8/8/8/8/k2p3R/8/4P3/4K3 w - - 0 50", "8/8/8/8/k2pP2R/8/8/4K3 b - - 0 50"];
        check_game(expect[0], &["e2e4", "d4d3"], &expect, 0);
        let expect = ["8/8/8/8/k2p4/8/4P3/4K3 w - - 0 50", "8/8/8/8/k2pP3/8/8/4K3 b - e3 0 50", "8/8/8/8/k7/4p3/8/4K3 w - - 0 51"];
        check_game(expect[0], &["e2e4", "d4e3", "e1d1"], &expect, 0);
        // Black double push with White ep capture, White to move.
        let expect = ["4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 20", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 21", "4k3/8/3P4/8/8/8/8/4K3 b - - 0 21"];
        check_game(expect[0], &["d7d5", "e5d6", "e8d7"], &expect, 0);
    }

    #[test]
    fn record_matches_text_convert() {
        let b = Board::from_fen("r3k3/1P6/8/8/8/8/6p1/R3K2R b KQq - 3 40").unwrap();
        let e = Entry { board: b, mv: Move { from: 60, to: 59, kind: MoveKind::Normal }, score: -150, result: 1 };
        let r = to_record(&e, 2.0);
        // Black to move, -150 stm -> +150 White, x2; result +1 for Black -> White lost.
        let txt = crate::encode("r3k3/1P6/8/8/8/8/6p1/R3K2R b KQq - 3 40 | 300 | 0.0").unwrap();
        assert_eq!(r, txt);
    }

    #[test]
    fn filter_reasons() {
        let mut f = Filter::new(10_000, 0);
        let b = Board::from_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 0 30").unwrap();
        let e = Entry { board: b.clone(), mv: Move { from: 4, to: 12, kind: MoveKind::Normal }, score: 0, result: 0 };
        assert_eq!(f.check(&e).map(|i| REASONS[i]), Some("in_check"));
        let b = Board::from_fen("4k3/8/8/8/8/8/3r4/4K3 w - - 0 30").unwrap();
        let e = Entry { board: b.clone(), mv: Move { from: 4, to: 11, kind: MoveKind::Normal }, score: 5, result: 0 };
        assert_eq!(f.check(&e).map(|i| REASONS[i]), Some("capture"));
        let e = Entry { board: b.clone(), mv: Move { from: 4, to: 5, kind: MoveKind::Normal }, score: 20000, result: 0 };
        assert_eq!(f.check(&e).map(|i| REASONS[i]), Some("score_cap"));
        let e = Entry { board: b.clone(), mv: Move { from: 4, to: 5, kind: MoveKind::Normal }, score: VALUE_NONE, result: 0 };
        assert_eq!(f.check(&e).map(|i| REASONS[i]), Some("value_none"));
        let e = Entry { board: b.clone(), mv: Move { from: 4, to: 5, kind: MoveKind::Normal }, score: 300, result: 1 };
        assert_eq!(f.check(&e), None);
        let mut b2 = b.clone();
        b2.ply += 1;
        let e = Entry { board: b2, mv: Move { from: 4, to: 5, kind: MoveKind::Normal }, score: 0, result: 1 };
        assert_eq!(f.check(&e).map(|i| REASONS[i]), Some("placeholder_zero"));
    }

    #[test]
    fn truncated_stream_ends_cleanly() {
        let start = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let payload = build_chain(&start, &[(mv(&start, "e2e4"), 0)], 0);
        let mut file = Vec::new();
        for _ in 0..2 {
            file.extend_from_slice(b"BINP");
            file.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            file.extend_from_slice(&payload);
        }
        file.truncate(file.len() - 5);
        let mut r = ChunkReader::new(&file[..]);
        let mut buf = Vec::new();
        assert!(r.next_into(&mut buf).unwrap());
        assert!(!r.next_into(&mut buf).unwrap());
        assert!(r.truncated);
    }
}
