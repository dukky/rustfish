use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

#[derive(Copy, Clone, Eq, PartialEq)]
pub(crate) enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Copy, Clone)]
pub(crate) struct TTEntry {
    pub(crate) depth: i16,
    pub(crate) score: i32,
    pub(crate) static_eval: i16,
    pub(crate) bound: Bound,
    pub(crate) best: Option<Move>,
}

impl Default for TTEntry {
    fn default() -> Self {
        Self {
            depth: -1,
            score: 0,
            static_eval: i16::MIN,
            bound: Bound::Exact,
            best: None,
        }
    }
}

impl TTEntry {
    pub(crate) fn static_eval(self) -> Option<i32> {
        if self.static_eval == i16::MIN {
            None
        } else {
            Some(self.static_eval as i32)
        }
    }
}

// Each slot is two atomics: `data` packs the whole entry into one u64 and
// `key` holds `zobrist_key ^ data` (Hyatt/Mann lockless hashing). A reader
// accepts an entry only if `key ^ data == zobrist_key`, so a torn read from
// concurrent writers in the lazy-SMP path is rejected instead of mixing one
// position's depth/bound/score with another position's key.
//
// data layout (LSB first):
//   0..17  move (pack_move; 0 = none)
//  17..33  score (i16)
//  33..49  static eval (i16; i16::MIN = none)
//  49..57  depth + 1 (u8; 0 = empty slot)
//  57..59  bound
//  59..64  age (5 bits)
#[derive(Default)]
struct TTSlot {
    key: AtomicU64,
    data: AtomicU64,
}

const MOVE_BITS: u64 = 0x1ffff;
const AGE_MASK: u8 = 0x1f;

const BUCKET_SLOTS: usize = 4;

// Four 16-byte slots per bucket, aligned so a probe touches one cache line.
#[derive(Default)]
#[repr(align(64))]
struct Bucket {
    slots: [TTSlot; BUCKET_SLOTS],
}

struct TransTableInner {
    buckets: crate::hugemem::HugeBuf<Bucket>,
    mask: usize,
    current_age: AtomicU8,
}

#[derive(Clone)]
pub(crate) struct TransTable {
    inner: Arc<TransTableInner>,
}

fn pack_move(mv: Move) -> u64 {
    let promo = match mv.promotion {
        None => 0,
        Some(PieceType::Knight) => 1,
        Some(PieceType::Bishop) => 2,
        Some(PieceType::Rook) => 3,
        Some(PieceType::Queen) => 4,
        Some(PieceType::Pawn | PieceType::King) => 0,
    };
    mv.from as u64
        | ((mv.to as u64) << 6)
        | ((promo as u64) << 12)
        | ((mv.is_en_passant as u64) << 15)
        | ((mv.is_castle as u64) << 16)
}

fn unpack_move(bits: u64) -> Option<Move> {
    let bits = bits & MOVE_BITS;
    // from == to == 0 never encodes a real move.
    if bits & 0xfff == 0 {
        return None;
    }
    let promotion = match (bits >> 12) & 0x7 {
        0 => None,
        1 => Some(PieceType::Knight),
        2 => Some(PieceType::Bishop),
        3 => Some(PieceType::Rook),
        4 => Some(PieceType::Queen),
        _ => return None,
    };
    Some(Move {
        from: (bits & 0x3f) as u8,
        to: ((bits >> 6) & 0x3f) as u8,
        promotion,
        is_en_passant: ((bits >> 15) & 1) != 0,
        is_castle: ((bits >> 16) & 1) != 0,
    })
}

fn bound_to_bits(bound: Bound) -> u64 {
    match bound {
        Bound::Exact => 0,
        Bound::Lower => 1,
        Bound::Upper => 2,
    }
}

fn bits_to_bound(raw: u64) -> Bound {
    match raw {
        1 => Bound::Lower,
        2 => Bound::Upper,
        _ => Bound::Exact,
    }
}

fn pack_data(
    depth: i32,
    score: i32,
    static_eval: Option<i32>,
    bound: Bound,
    best: Option<Move>,
    age: u8,
) -> u64 {
    let mv = best.map(pack_move).unwrap_or(0);
    let score = score.clamp(-(i16::MAX as i32), i16::MAX as i32) as i16 as u16 as u64;
    let se = static_eval
        .unwrap_or(i16::MIN as i32)
        .clamp(i16::MIN as i32, i16::MAX as i32) as i16 as u16 as u64;
    let depth1 = (depth.clamp(0, 254) + 1) as u64;
    mv | (score << 17)
        | (se << 33)
        | (depth1 << 49)
        | (bound_to_bits(bound) << 57)
        | (((age & AGE_MASK) as u64) << 59)
}

#[inline]
fn data_depth(data: u64) -> i32 {
    ((data >> 49) & 0xff) as i32 - 1
}

#[inline]
fn data_age(data: u64) -> u8 {
    ((data >> 59) as u8) & AGE_MASK
}

fn unpack_data(data: u64) -> TTEntry {
    TTEntry {
        depth: data_depth(data) as i16,
        score: ((data >> 17) & 0xffff) as u16 as i16 as i32,
        static_eval: ((data >> 33) & 0xffff) as u16 as i16,
        bound: bits_to_bound((data >> 57) & 0x3),
        best: unpack_move(data),
    }
}

impl TransTable {
    pub(crate) fn new(megabytes: usize) -> Self {
        let bytes = megabytes.max(1) * 1024 * 1024;
        let bucket_size = std::mem::size_of::<Bucket>();
        // Round down so the table never exceeds the requested size.
        let n = (bytes / bucket_size).max(1);
        let n = if n.is_power_of_two() { n } else { n.next_power_of_two() / 2 };
        Self {
            inner: Arc::new(TransTableInner {
                // SAFETY: a Bucket is atomics only; all-zero = Bucket::default().
                buckets: unsafe { crate::hugemem::HugeBuf::zeroed(n) },
                mask: n - 1,
                current_age: AtomicU8::new(0),
            }),
        }
    }

    pub(crate) fn clear(&mut self) {
        for bucket in self.inner.buckets.iter() {
            for slot in bucket.slots.iter() {
                slot.key.store(0, Ordering::Relaxed);
                slot.data.store(0, Ordering::Relaxed);
            }
        }
        self.inner.current_age.store(0, Ordering::Relaxed);
    }

    pub(crate) fn new_search(&mut self) {
        let age = self.inner.current_age.load(Ordering::Relaxed);
        self.inner
            .current_age
            .store(age.wrapping_add(1) & AGE_MASK, Ordering::Relaxed);
    }

    #[inline]
    fn bucket(&self, key: u64) -> &Bucket {
        &self.inner.buckets[key as usize & self.inner.mask]
    }

    /// Hint the CPU to start loading `key`'s bucket so the child node's
    /// probe does not stall on a cache miss. Pure performance hint: no
    /// architectural effect on memory or search results.
    #[inline(always)]
    pub(crate) fn prefetch(&self, key: u64) {
        let ptr = self.bucket(key) as *const Bucket;
        #[cfg(target_arch = "aarch64")]
        // SAFETY: PRFM is a hint; it never faults or writes memory, and
        // `ptr` points into the live bucket array anyway.
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr, options(nostack, readonly, preserves_flags));
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: PREFETCHT0 is a hint (SSE, part of the x86-64 baseline);
        // it never faults or writes memory.
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = ptr;
    }

}

// Replacement policy (clean-room design, docs/cleanroom-tt-history-se.md).
//
// Same position already stored: the new result wins, unless the stored entry
// is from this search, deeper by more than TT_SAME_KEY_SLACK plies, and the new
// result is only a bound. A store without a move keeps the stored move.
// New position: fill an empty slot, else evict the slot with the lowest keep
// value: any entry from an earlier search ranks below every entry from this
// one, then shallower below deeper, then bounds below exact scores.

/// Plies by which a same-search entry must be deeper than a new non-exact
/// result for the same position to be kept instead of overwritten.
const TT_SAME_KEY_SLACK: i32 = 2;

/// Eviction rank of an occupied slot (higher = keep longer; empty slots rank 0).
#[inline]
fn keep_value(data: u64, age: u8) -> u32 {
    let current = (data_age(data) == age) as u32;
    let depth1 = ((data >> 49) & 0xff) as u32;
    let exact = (bits_to_bound((data >> 57) & 0x3) == Bound::Exact) as u32;
    1 + ((current << 10) | (depth1 << 1) | exact)
}

#[inline]
fn write_slot(slot: &TTSlot, key: u64, data: u64) {
    slot.data.store(data, Ordering::Relaxed);
    slot.key.store(key ^ data, Ordering::Relaxed);
}

impl TransTable {
    pub(crate) fn probe(&self, key: u64) -> Option<TTEntry> {
        for slot in self.bucket(key).slots.iter() {
            let data = slot.data.load(Ordering::Relaxed);
            if data_depth(data) >= 0 && slot.key.load(Ordering::Relaxed) ^ data == key {
                return Some(unpack_data(data));
            }
        }
        None
    }

    pub(crate) fn store(
        &self,
        key: u64,
        depth: i32,
        score: i32,
        static_eval: Option<i32>,
        bound: Bound,
        best: Option<Move>,
    ) {
        let age = self.inner.current_age.load(Ordering::Relaxed) & AGE_MASK;
        let depth = depth.clamp(0, 254);
        let bucket = self.bucket(key);
        let mut victim = 0usize;
        let mut victim_value = u32::MAX;
        for (i, slot) in bucket.slots.iter().enumerate() {
            let data = slot.data.load(Ordering::Relaxed);
            let occupied = data_depth(data) >= 0;
            if occupied && slot.key.load(Ordering::Relaxed) ^ data == key {
                let keep_old = data_age(data) == age
                    && bound != Bound::Exact
                    && data_depth(data) > depth + TT_SAME_KEY_SLACK;
                if !keep_old {
                    let best = best.or_else(|| unpack_move(data));
                    write_slot(slot, key, pack_data(depth, score, static_eval, bound, best, age));
                }
                return;
            }
            let value = if occupied { keep_value(data, age) } else { 0 };
            if value < victim_value {
                victim = i;
                victim_value = value;
            }
        }
        write_slot(&bucket.slots[victim], key, pack_data(depth, score, static_eval, bound, best, age));
    }
}

#[derive(Clone)]
pub(crate) struct Zobrist {
    pub(crate) piece: [[u64; 64]; 12],
    pub(crate) castling: [u64; 16],
    pub(crate) ep_file: [u64; 8],
    pub(crate) side: u64,
}

impl Zobrist {
    pub(crate) fn new() -> Self {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut piece = [[0u64; 64]; 12];
        for p in 0..12 {
            for sq in 0..64 {
                piece[p][sq] = splitmix64(&mut seed);
            }
        }
        let mut castling = [0u64; 16];
        for i in 0..16 {
            castling[i] = splitmix64(&mut seed);
        }
        let mut ep_file = [0u64; 8];
        for i in 0..8 {
            ep_file[i] = splitmix64(&mut seed);
        }
        let side = splitmix64(&mut seed);

        Self {
            piece,
            castling,
            ep_file,
            side,
        }
    }
}

pub(crate) fn splitmix64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[inline]
pub(crate) fn piece_index(piece: Piece) -> usize {
    // Color::White = 0 / Black = 1 and PieceType Pawn..King = 0..5.
    piece.color as usize * 6 + piece.kind as usize
}

pub(crate) fn board_hash(board: &Board, zobrist: &Zobrist) -> u64 {
    let mut key = 0u64;
    for sq in 0..64 {
        if let Some(p) = board.squares[sq] {
            key ^= zobrist.piece[piece_index(p)][sq];
        }
    }
    key ^= zobrist.castling[board.castling as usize];
    if let Some(ep) = board.en_passant {
        key ^= zobrist.ep_file[(ep % 8) as usize];
    }
    if board.side_to_move == Color::Black {
        key ^= zobrist.side;
    }
    key
}

pub(crate) fn child_hash(key: u64, board_after: &Board, mv: Move, undo: Undo, zobrist: &Zobrist) -> u64 {
    let mut next = key;
    next ^= zobrist.side;
    next ^= zobrist.castling[undo.castling as usize];
    next ^= zobrist.castling[board_after.castling as usize];

    if let Some(ep) = undo.en_passant {
        next ^= zobrist.ep_file[(ep % 8) as usize];
    }
    if let Some(ep) = board_after.en_passant {
        next ^= zobrist.ep_file[(ep % 8) as usize];
    }

    next ^= zobrist.piece[piece_index(undo.moved_piece)][mv.from as usize];

    let mut placed = undo.moved_piece;
    if let Some(promo) = mv.promotion {
        placed.kind = promo;
    }
    next ^= zobrist.piece[piece_index(placed)][mv.to as usize];

    if let Some(captured) = undo.captured {
        let capture_sq = if mv.is_en_passant {
            if undo.moved_piece.color == Color::White {
                mv.to - 8
            } else {
                mv.to + 8
            }
        } else {
            undo.captured_sq.unwrap_or(mv.to)
        };
        next ^= zobrist.piece[piece_index(captured)][capture_sq as usize];
    }

    if mv.is_castle {
        let rook = Piece {
            color: undo.moved_piece.color,
            kind: PieceType::Rook,
        };
        let (rook_from, rook_to) = match mv.to {
            6 => (7usize, 5usize),
            2 => (0usize, 3usize),
            62 => (63usize, 61usize),
            58 => (56usize, 59usize),
            _ => (0usize, 0usize),
        };
        next ^= zobrist.piece[piece_index(rook)][rook_from];
        next ^= zobrist.piece[piece_index(rook)][rook_to];
    }

    next
}

/// The child key of `mv` from the position before it is made, for
/// prefetching only: exact for ordinary moves and captures, but castling,
/// en passant and promotions (and castling-right changes) are ignored, so
/// for those it is just some other key and the prefetch is wasted.
#[inline(always)]
pub(crate) fn approx_child_hash(key: u64, board_before: &Board, mv: Move, zobrist: &Zobrist) -> u64 {
    let mut next = key ^ zobrist.side;
    if let Some(ep) = board_before.en_passant {
        next ^= zobrist.ep_file[(ep % 8) as usize];
    }
    if let Some(p) = board_before.squares[mv.from as usize] {
        let pi = piece_index(p);
        next ^= zobrist.piece[pi][mv.from as usize] ^ zobrist.piece[pi][mv.to as usize];
    }
    if let Some(c) = board_before.squares[mv.to as usize] {
        next ^= zobrist.piece[piece_index(c)][mv.to as usize];
    }
    next
}

pub(crate) fn null_move_hash(key: u64, ep_before: Option<u8>, zobrist: &Zobrist) -> u64 {
    let mut next = key ^ zobrist.side;
    if let Some(ep) = ep_before {
        next ^= zobrist.ep_file[(ep % 8) as usize];
    }
    next
}

pub(crate) fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_SCORE - 1000 {
        score + ply as i32
    } else if score <= -MATE_SCORE + 1000 {
        score - ply as i32
    } else {
        score
    }
}

pub(crate) fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_SCORE - 1000 {
        score - ply as i32
    } else if score <= -MATE_SCORE + 1000 {
        score + ply as i32
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_after(moves: &[&str]) -> (u64, Option<u8>) {
        let zobrist = Zobrist::new();
        let mut board = Board::new();
        board.set_startpos();
        for m in moves {
            let mv = crate::movegen::parse_uci_move(&mut board, m).unwrap();
            board.make_move(mv);
        }
        (board_hash(&board, &zobrist), board.en_passant)
    }

    #[test]
    fn en_passant_square_only_kept_when_capturable() {
        // Same position by two move orders, each ending with a double push
        // nothing can capture: one key, no en passant square.
        let a = key_after(&["d2d4", "g8f6", "e2e4"]);
        let b = key_after(&["e2e4", "g8f6", "d2d4"]);
        assert_eq!(a, b);
        assert_eq!(a.1, None);
        // A black pawn beside the pushed pawn keeps it.
        let c = key_after(&["e2e4", "d7d5", "e4e5", "f7f5"]);
        assert_eq!(c.1, Some(45));
    }

    fn mv(from: u8, to: u8, promotion: Option<PieceType>, ep: bool, castle: bool) -> Move {
        Move { from, to, promotion, is_en_passant: ep, is_castle: castle }
    }

    #[test]
    fn packed_entry_round_trips() {
        let tt = TransTable::new(1);
        let cases = [
            (0x1234_5678_9abc_def0u64, 7, MATE_SCORE - 3, Some(-812), Bound::Lower,
             Some(mv(52, 60, Some(PieceType::Queen), false, false))),
            (0x0fed_cba9_8765_4321u64, 0, -MATE_SCORE + 40, None, Bound::Upper,
             Some(mv(4, 6, None, false, true))),
            (0x1111_2222_3333_4444u64, 99, -1, Some(i16::MAX as i32), Bound::Exact,
             Some(mv(35, 44, None, true, false))),
            (0x5555_6666_7777_8888u64, 3, 0, Some(0), Bound::Exact, None),
        ];
        for (key, depth, score, se, bound, best) in cases {
            tt.store(key, depth, score, se, bound, best);
            let e = tt.probe(key).expect("entry present");
            assert_eq!(e.depth as i32, depth);
            assert_eq!(e.score, score);
            assert_eq!(e.static_eval(), se);
            assert!(e.bound == bound);
            assert!(e.best == best);
        }
    }

    #[test]
    fn mismatched_key_and_data_is_rejected() {
        let tt = TransTable::new(1);
        let key = 0xdead_beef_0000_0001u64;
        assert!(tt.probe(key).is_none());
        tt.store(key, 5, 10, Some(3), Bound::Exact, None);
        let slot = &tt.bucket(key).slots[0];
        // Simulate a torn write: data from another position, key word unchanged.
        let other = pack_data(9, -500, None, Bound::Lower, None, 0);
        slot.data.store(other, Ordering::Relaxed);
        assert!(tt.probe(key).is_none());
    }

    /// Keys that share one bucket in a 1 MB table (bucket = key & mask).
    fn same_bucket_keys(tt: &TransTable, n: usize) -> Vec<u64> {
        let stride = tt.inner.mask as u64 + 1;
        (0..n as u64).map(|i| 0x1234_5678_9abc_0005 & tt.inner.mask as u64 | (i + 1) * stride * 0x9e37).collect()
    }

    fn quiet(from: u8, to: u8) -> Option<Move> {
        Some(mv(from, to, None, false, false))
    }

    #[test]
    fn same_key_new_result_wins_and_moveless_store_keeps_move() {
        let tt = TransTable::new(1);
        let k = same_bucket_keys(&tt, 1)[0];
        tt.store(k, 6, 40, Some(10), Bound::Lower, quiet(12, 28));
        tt.store(k, 5, -20, Some(10), Bound::Upper, None);
        let e = tt.probe(k).unwrap();
        assert_eq!((e.depth, e.score), (5, -20));
        assert!(e.bound == Bound::Upper);
        assert_eq!(e.best, quiet(12, 28));
    }

    #[test]
    fn same_key_keeps_much_deeper_bound_from_this_search() {
        let mut tt = TransTable::new(1);
        let k = same_bucket_keys(&tt, 1)[0];
        tt.store(k, 12, 40, None, Bound::Lower, quiet(1, 18));
        // Within the slack: overwritten.
        tt.store(k, 10, 30, None, Bound::Lower, quiet(6, 21));
        assert_eq!(tt.probe(k).unwrap().depth, 10);
        tt.store(k, 12, 40, None, Bound::Lower, quiet(1, 18));
        // Deeper by more than the slack: kept, unless the new result is exact.
        tt.store(k, 9, 30, None, Bound::Upper, None);
        assert_eq!(tt.probe(k).unwrap().depth, 12);
        tt.store(k, 3, 25, None, Bound::Exact, None);
        assert_eq!(tt.probe(k).unwrap().depth, 3);
        // From an earlier search: always overwritten.
        tt.store(k, 12, 40, None, Bound::Lower, quiet(1, 18));
        tt.new_search();
        tt.store(k, 1, 5, None, Bound::Upper, None);
        let e = tt.probe(k).unwrap();
        assert_eq!(e.depth, 1);
        assert_eq!(e.best, quiet(1, 18));
    }

    #[test]
    fn eviction_prefers_stale_then_shallow_then_bounds() {
        let mut tt = TransTable::new(1);
        let k = same_bucket_keys(&tt, 7);
        // Old search: one very deep entry.
        tt.store(k[0], 30, 1, None, Bound::Exact, None);
        tt.new_search();
        tt.store(k[1], 2, 1, None, Bound::Lower, None);
        tt.store(k[2], 8, 1, None, Bound::Lower, None);
        tt.store(k[3], 2, 1, None, Bound::Exact, None);
        // Bucket full: the stale entry goes first despite its depth.
        tt.store(k[4], 5, 1, None, Bound::Lower, None);
        assert!(tt.probe(k[0]).is_none());
        // Then the shallowest, a bound before an exact score at equal depth.
        tt.store(k[5], 5, 1, None, Bound::Lower, None);
        assert!(tt.probe(k[1]).is_none());
        assert!(tt.probe(k[3]).is_some());
        tt.store(k[6], 5, 1, None, Bound::Lower, None);
        assert!(tt.probe(k[3]).is_none());
        for &kk in &[k[2], k[4], k[5], k[6]] {
            assert!(tt.probe(kk).is_some());
        }
    }
}
