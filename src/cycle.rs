//! Upcoming-repetition detection ("can the side to move repeat right now?").
//!
//! The idea is Marcel van Kervinck's published cycle-detection method; this
//! module is an independent implementation written from a description of the
//! idea only.
//!
//! A reversible move (a quiet knight, bishop, rook, queen or king move) changes
//! the position key by a fixed amount: the piece's zobrist value on the two
//! squares plus the side-to-move value. That amount depends only on the
//! colour, the piece type and the unordered square pair, so all of them can be
//! listed once. If the current key XOR the key of a position `i` plies back
//! (odd `i`, so the opponent was to move there, as after a move of ours) is one
//! of those amounts, then a single reversible move of ours would recreate that
//! earlier position, provided the move is actually playable on the current
//! board: the piece belongs to the side to move, it stands on one end of the
//! pair, the other end is empty and, for sliders, nothing stands in between.
//!
//! The amounts are kept in a small cuckoo hash table (two candidate slots per
//! key), so each history entry costs two probes.
//!
//! Legality is not checked: a hit may be a pinned piece's move or a king step
//! into check. Search uses the result only to raise alpha to the draw score,
//! and the same approximation applies to every node, so this is accepted.

use super::*;
use crate::bitboard::{bishop_attacks, rook_attacks, KING_ATTACKS, KNIGHT_ATTACKS};
use crate::tt::{piece_index, splitmix64};
use std::sync::OnceLock;

/// Slot count, a power of two. 3668 entries give a load factor of about 0.45,
/// below the ~0.5 limit where two-choice cuckoo insertion starts to fail.
const SLOT_BITS: u32 = 13;
const SLOTS: usize = 1 << SLOT_BITS;
/// Evictions allowed per insertion before the build gives up and retries with
/// new multipliers.
const MAX_KICKS: usize = 1000;
/// Packed move value for an empty slot (a real entry never has from == to).
const EMPTY: u16 = 0;

/// The reversible-move table: `deltas[s]` is a key difference, `moves[s]` the
/// move that produces it packed as `piece_index << 12 | sq_lo << 6 | sq_hi`.
pub(crate) struct CycleTable {
    deltas: Box<[u64]>,
    moves: Box<[u16]>,
    mul_a: u64,
    mul_b: u64,
}

#[inline]
fn slot(delta: u64, mul: u64) -> usize {
    (delta.wrapping_mul(mul) >> (64 - SLOT_BITS)) as usize
}

#[inline]
fn pack(piece: usize, lo: usize, hi: usize) -> u16 {
    ((piece << 12) | (lo << 6) | hi) as u16
}

#[inline]
fn unpack(m: u16) -> (usize, usize, usize) {
    ((m >> 12) as usize, ((m >> 6) & 63) as usize, (m & 63) as usize)
}

/// Every square reachable from `sq` in one move by `kind` on an empty board.
fn empty_board_targets(kind: PieceType, sq: usize) -> u64 {
    match kind {
        PieceType::Knight => KNIGHT_ATTACKS[sq],
        PieceType::Bishop => bishop_attacks(sq, 0),
        PieceType::Rook => rook_attacks(sq, 0),
        PieceType::Queen => bishop_attacks(sq, 0) | rook_attacks(sq, 0),
        PieceType::King => KING_ATTACKS[sq],
        PieceType::Pawn => 0,
    }
}

const KINDS: [PieceType; 5] =
    [PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen, PieceType::King];

/// All (delta, packed move) pairs: one per colour, non-pawn piece type and
/// unordered square pair joined by an empty-board move of that piece.
fn enumerate(zobrist: &Zobrist) -> Vec<(u64, u16)> {
    let mut out = Vec::new();
    for color in [Color::White, Color::Black] {
        for kind in KINDS {
            let p = piece_index(Piece { color, kind });
            for lo in 0..64usize {
                let mut targets = empty_board_targets(kind, lo) & !((2u64 << lo) - 1);
                while targets != 0 {
                    let hi = targets.trailing_zeros() as usize;
                    targets &= targets - 1;
                    let delta = zobrist.piece[p][lo] ^ zobrist.piece[p][hi] ^ zobrist.side;
                    out.push((delta, pack(p, lo, hi)));
                }
            }
        }
    }
    out
}

impl CycleTable {
    /// One attempt with the given multipliers; `None` if some insertion runs
    /// out of kicks.
    fn try_build(items: &[(u64, u16)], mul_a: u64, mul_b: u64) -> Option<Self> {
        let mut deltas = vec![0u64; SLOTS].into_boxed_slice();
        let mut moves = vec![EMPTY; SLOTS].into_boxed_slice();
        for &(delta, mv) in items {
            let (mut cur_d, mut cur_m) = (delta, mv);
            let mut pos = slot(cur_d, mul_a);
            let mut placed = false;
            for _ in 0..MAX_KICKS {
                if moves[pos] == EMPTY {
                    deltas[pos] = cur_d;
                    moves[pos] = cur_m;
                    placed = true;
                    break;
                }
                std::mem::swap(&mut deltas[pos], &mut cur_d);
                std::mem::swap(&mut moves[pos], &mut cur_m);
                // The evicted entry goes to its other candidate slot.
                let a = slot(cur_d, mul_a);
                pos = if a == pos { slot(cur_d, mul_b) } else { a };
            }
            if !placed {
                return None;
            }
        }
        Some(Self { deltas, moves, mul_a, mul_b })
    }

    fn build(zobrist: &Zobrist) -> Self {
        let items = enumerate(zobrist);
        // Distinct moves must have distinct deltas, or lookups would be ambiguous.
        let mut sorted: Vec<u64> = items.iter().map(|&(d, _)| d).collect();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), items.len(), "reversible-move deltas collide");

        // Deterministic multiplier sequence; retry until every entry fits.
        let mut seed = 0x5a17_c0de_2026_1126u64;
        for _ in 0..64 {
            let mul_a = splitmix64(&mut seed) | 1;
            let mul_b = splitmix64(&mut seed) | 1;
            if let Some(table) = Self::try_build(&items, mul_a, mul_b) {
                assert_eq!(table.len(), items.len(), "cycle table size");
                for &(d, m) in &items {
                    assert_eq!(table.lookup(d), Some(m), "cycle table lost an entry");
                }
                return table;
            }
        }
        panic!("cycle table: no multiplier pair placed every entry");
    }

    /// The shared table for the engine's (deterministic) zobrist keys.
    pub(crate) fn get() -> &'static CycleTable {
        static TABLE: OnceLock<CycleTable> = OnceLock::new();
        TABLE.get_or_init(|| CycleTable::build(&Zobrist::new()))
    }

    /// Number of reversible moves stored.
    pub(crate) fn len(&self) -> usize {
        self.moves.iter().filter(|&&m| m != EMPTY).count()
    }

    #[inline]
    fn lookup(&self, delta: u64) -> Option<u16> {
        let a = slot(delta, self.mul_a);
        if self.deltas[a] == delta && self.moves[a] != EMPTY {
            return Some(self.moves[a]);
        }
        let b = slot(delta, self.mul_b);
        if self.deltas[b] == delta && self.moves[b] != EMPTY {
            return Some(self.moves[b]);
        }
        None
    }
}

/// True when the side to move has a reversible move (legality not checked)
/// that recreates a position from `history`, which `search::is_repetition`
/// would then score as a draw.
///
/// `history` holds the keys of the game and search path, oldest first, with
/// `key` (the current position) last. Only positions an odd number of plies
/// back are candidates (the opponent was to move there), from 3 plies up to
/// the halfmove clock (nothing before the last irreversible move can recur)
/// and up to `max_back` plies (callers stop at the last null move). As in
/// `is_repetition`, a single earlier occurrence counts, whether it lies in the
/// search tree or in the game before the root.
pub(crate) fn upcoming_repetition(board: &Board, history: &[u64], key: u64, max_back: usize) -> bool {
    let len = history.len();
    if len == 0 {
        return false;
    }
    let limit = (board.halfmove_clock as usize).min(max_back).min(len - 1);
    if limit < 3 {
        return false;
    }
    let table = CycleTable::get();
    let occ = board.bb.occ_all();
    let us = color_index(board.side_to_move);
    let mut i = 3;
    while i <= limit {
        let delta = key ^ history[len - 1 - i];
        if let Some(m) = table.lookup(delta) {
            let (p, lo, hi) = unpack(m);
            let color = p / 6;
            if color == us {
                let kind = p % 6;
                let own = board.bb.pieces[color][kind];
                let (lo_bit, hi_bit) = (1u64 << lo, 1u64 << hi);
                let from = if own & lo_bit != 0 && occ & hi_bit == 0 {
                    Some((lo, hi_bit))
                } else if own & hi_bit != 0 && occ & lo_bit == 0 {
                    Some((hi, lo_bit))
                } else {
                    None
                };
                if let Some((from, to_bit)) = from {
                    // Leapers always reach; sliders need the squares between free,
                    // i.e. the destination among the attacks under current occupancy.
                    let reach = match kind {
                        2 => bishop_attacks(from, occ),
                        3 => rook_attacks(from, occ),
                        4 => bishop_attacks(from, occ) | rook_attacks(from, occ),
                        _ => to_bit,
                    };
                    if reach & to_bit != 0 {
                        return true;
                    }
                }
            }
        }
        i += 2;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_holds_every_reversible_move() {
        let zobrist = Zobrist::new();
        let items = enumerate(&zobrist);
        let table = CycleTable::get();
        assert_eq!(table.len(), items.len());
        eprintln!("cycle table: {} reversible moves", items.len());
        for &(d, m) in &items {
            assert_eq!(table.lookup(d), Some(m));
        }
        // Random keys are (overwhelmingly) absent.
        let mut seed = 12345u64;
        for _ in 0..10_000 {
            let d = splitmix64(&mut seed);
            if let Some(m) = table.lookup(d) {
                assert!(items.contains(&(d, m)));
            }
        }
    }

    fn is_reversible(board: &Board, mv: Move) -> bool {
        let Some(p) = board.squares[mv.from as usize] else { return false };
        p.kind != PieceType::Pawn
            && !mv.is_castle
            && mv.promotion.is_none()
            && !mv.is_en_passant
            && board.squares[mv.to as usize].is_none()
    }

    /// Does `child` (key after one move from the current position) occur at an
    /// odd distance `i` from the current position with 3 <= i <= limit?
    fn hits(history: &[u64], child: u64, limit: usize) -> bool {
        let len = history.len();
        let mut i = 3;
        while i <= limit {
            if history[len - 1 - i] == child {
                return true;
            }
            i += 2;
        }
        false
    }

    /// Brute force over moves: (legal-move answer, pseudo-legal-move answer).
    fn brute(board: &mut Board, history: &[u64], key: u64, zobrist: &Zobrist) -> (bool, bool) {
        let limit = (board.halfmove_clock as usize).min(history.len() - 1);
        let side = board.side_to_move;
        let mut moves = Vec::new();
        generate_pseudo_moves(board, false, &mut moves);
        let mut legal_hit = false;
        let mut pseudo_hit = false;
        for mv in moves {
            if !is_reversible(board, mv) {
                continue;
            }
            let undo = board.make_move(mv);
            let child = child_hash(key, board, mv, undo, zobrist);
            assert_eq!(child, board_hash(board, zobrist));
            let legal = !board.is_in_check(side);
            board.undo_move(mv, undo);
            // A reversible move that changes castling rights cannot recur; the
            // key comparison handles it.
            if hits(history, child, limit) {
                pseudo_hit = true;
                legal_hit |= legal;
            }
        }
        (legal_hit, pseudo_hit)
    }

    #[test]
    fn pinned_piece_counts_although_illegal() {
        // The e2 knight is pinned by the e7 rook; Ne2-c3 would recreate the
        // position three plies back, which the fast test accepts (legality is
        // not checked) and the legal brute force rejects.
        let zobrist = Zobrist::new();
        let mut board = Board::new();
        board.set_fen("4k3/4r3/8/8/8/8/4N3/4K3 w - - 3 10");
        let key = board_hash(&board, &zobrist);
        let n = piece_index(Piece { color: Color::White, kind: PieceType::Knight });
        let old = key ^ zobrist.piece[n][12] ^ zobrist.piece[n][18] ^ zobrist.side;
        let history = vec![old, 1, 2, key];
        assert!(upcoming_repetition(&board, &history, key, usize::MAX));
        assert_eq!(brute(&mut board, &history, key, &zobrist), (false, true));
        // Blocked slider: Re1-e5 with a piece on e3 is not playable.
        board.set_fen("4k3/8/8/8/8/4p3/8/4RK2 w - - 5 10");
        let key = board_hash(&board, &zobrist);
        let r = piece_index(Piece { color: Color::White, kind: PieceType::Rook });
        let old = key ^ zobrist.piece[r][4] ^ zobrist.piece[r][36] ^ zobrist.side;
        let history = vec![old, 1, 2, key];
        assert!(!upcoming_repetition(&board, &history, key, usize::MAX));
        // Halfmove clock 2: the position three plies back predates the last
        // irreversible move.
        board.set_fen("4k3/8/8/8/8/8/8/4RK2 w - - 2 10");
        let key = board_hash(&board, &zobrist);
        let old = key ^ zobrist.piece[r][4] ^ zobrist.piece[r][36] ^ zobrist.side;
        let history = vec![old, 1, 2, key];
        assert!(!upcoming_repetition(&board, &history, key, usize::MAX));
        board.set_fen("4k3/8/8/8/8/8/8/4RK2 w - - 3 10");
        assert!(upcoming_repetition(&board, &history, key, usize::MAX));
        assert!(!upcoming_repetition(&board, &history, key, 2));
    }

    fn rand(seed: &mut u64, n: usize) -> usize {
        (splitmix64(seed) % n as u64) as usize
    }

    #[test]
    fn matches_brute_force_on_random_walks() {
        let zobrist = Zobrist::new();
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "4k3/8/8/2q5/8/8/3R4/1B2K2N w - - 0 1",
            "3rk3/8/2b5/8/8/5N2/1Q6/R3K3 b - - 0 1",
            "8/2k5/8/1r6/8/8/6B1/3QK3 w - - 0 1",
            "r1b1k2r/ppq2ppp/2n1pn2/3p4/1bPP4/2N1PN2/PP1B1PPP/R2QKB1R w KQkq - 0 1",
        ];
        let mut seed = 0xfeed_beefu64;
        let mut checked = 0usize;
        let mut positive = 0usize;
        let mut illegal_only = 0usize;
        let mut buf = Vec::new();
        for fen in fens {
            for _walk in 0..60 {
                let mut board = Board::new();
                board.set_fen(fen);
                let mut history = vec![board_hash(&board, &zobrist)];
                for _ply in 0..120 {
                    let key = *history.last().unwrap();
                    let (legal_hit, pseudo_hit) = brute(&mut board, &history, key, &zobrist);
                    let fast = upcoming_repetition(&board, &history, key, usize::MAX);
                    // The fast test ignores legality: it must equal the
                    // pseudo-legal brute force and cover the legal one.
                    assert_eq!(fast, pseudo_hit, "fen {fen}, history {}", history.len());
                    assert!(!legal_hit || fast);
                    // max_back cuts the scan off.
                    assert!(!upcoming_repetition(&board, &history, key, 2));
                    checked += 1;
                    positive += fast as usize;
                    illegal_only += (fast && !legal_hit) as usize;

                    board.generate_legal_moves(false, &mut buf);
                    if buf.is_empty() {
                        break;
                    }
                    // Mostly reversible moves (shuffling recreates positions),
                    // sometimes anything.
                    let rev: Vec<Move> = buf.iter().copied().filter(|&m| is_reversible(&board, m)).collect();
                    let mv = if !rev.is_empty() && rand(&mut seed, 10) < 8 {
                        // Favour undoing a recent move to produce repetitions.
                        rev[rand(&mut seed, rev.len())]
                    } else {
                        buf[rand(&mut seed, buf.len())]
                    };
                    let undo = board.make_move(mv);
                    let child = child_hash(key, &board, mv, undo, &zobrist);
                    history.push(child);
                    if board.halfmove_clock >= 100 {
                        break;
                    }
                }
            }
        }
        assert!(checked > 10_000, "checked {checked}");
        assert!(positive > 500, "only {positive} positives");
        eprintln!("cycle brute force: {checked} positions, {positive} hits, {illegal_only} illegal-only");
    }
}
