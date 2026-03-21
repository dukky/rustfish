use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, AtomicU8, Ordering};

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

struct TTSlot {
    key: AtomicU64,
    depth: AtomicI32,
    score: AtomicI32,
    static_eval: AtomicI32,
    best: AtomicU32,
    bound: AtomicU8,
    age: AtomicU8,
}

impl Default for TTSlot {
    fn default() -> Self {
        Self {
            key: AtomicU64::new(0),
            depth: AtomicI32::new(-1),
            score: AtomicI32::new(0),
            static_eval: AtomicI32::new(i16::MIN as i32),
            best: AtomicU32::new(u32::MAX),
            bound: AtomicU8::new(bound_to_u8(Bound::Exact)),
            age: AtomicU8::new(0),
        }
    }
}

struct TransTableInner {
    entries: Box<[TTSlot]>,
    mask: usize,
    current_age: AtomicU8,
}

#[derive(Clone)]
pub(crate) struct TransTable {
    inner: Arc<TransTableInner>,
}

fn pack_move(mv: Move) -> u32 {
    let promo = match mv.promotion {
        None => 0,
        Some(PieceType::Knight) => 1,
        Some(PieceType::Bishop) => 2,
        Some(PieceType::Rook) => 3,
        Some(PieceType::Queen) => 4,
        Some(PieceType::Pawn | PieceType::King) => 0,
    };
    mv.from as u32
        | ((mv.to as u32) << 6)
        | ((promo as u32) << 12)
        | ((mv.is_en_passant as u32) << 15)
        | ((mv.is_castle as u32) << 16)
}

fn unpack_move(bits: u32) -> Option<Move> {
    if bits == u32::MAX {
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

fn bound_to_u8(bound: Bound) -> u8 {
    match bound {
        Bound::Exact => 0,
        Bound::Lower => 1,
        Bound::Upper => 2,
    }
}

fn u8_to_bound(raw: u8) -> Bound {
    match raw {
        1 => Bound::Lower,
        2 => Bound::Upper,
        _ => Bound::Exact,
    }
}

impl TransTable {
    pub(crate) fn new(megabytes: usize) -> Self {
        let bytes = (megabytes.max(1) * 1024 * 1024) as usize;
        let entry_size = std::mem::size_of::<TTSlot>();
        let mut n = (bytes / entry_size).max(1).next_power_of_two();
        if n == 0 {
            n = 1;
        }
        Self {
            inner: Arc::new(TransTableInner {
                entries: (0..n).map(|_| TTSlot::default()).collect(),
                mask: n - 1,
                current_age: AtomicU8::new(0),
            }),
        }
    }

    pub(crate) fn clear(&mut self) {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            for slot in inner.entries.iter_mut() {
                *slot.key.get_mut() = 0;
                *slot.depth.get_mut() = -1;
                *slot.score.get_mut() = 0;
                *slot.static_eval.get_mut() = i16::MIN as i32;
                *slot.best.get_mut() = u32::MAX;
                *slot.bound.get_mut() = bound_to_u8(Bound::Exact);
                *slot.age.get_mut() = 0;
            }
            *inner.current_age.get_mut() = 0;
            return;
        }

        for slot in self.inner.entries.iter() {
            slot.key.store(0, Ordering::Relaxed);
            slot.depth.store(-1, Ordering::Relaxed);
            slot.score.store(0, Ordering::Relaxed);
            slot.static_eval.store(i16::MIN as i32, Ordering::Relaxed);
            slot.best.store(u32::MAX, Ordering::Relaxed);
            slot.bound.store(bound_to_u8(Bound::Exact), Ordering::Relaxed);
            slot.age.store(0, Ordering::Relaxed);
        }
        self.inner.current_age.store(0, Ordering::Relaxed);
    }

    pub(crate) fn new_search(&mut self) {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            *inner.current_age.get_mut() = inner.current_age.get_mut().wrapping_add(1);
            return;
        }
        self.inner.current_age.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn probe(&mut self, key: u64) -> Option<TTEntry> {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            return probe_unique(inner, key);
        }
        self.probe_shared(key)
    }

    fn probe_shared(&self, key: u64) -> Option<TTEntry> {
        let slot = &self.inner.entries[key as usize & self.inner.mask];
        let key1 = slot.key.load(Ordering::Acquire);
        if key1 != key {
            return None;
        }
        let depth = slot.depth.load(Ordering::Relaxed);
        if depth < 0 {
            return None;
        }
        let score = slot.score.load(Ordering::Relaxed);
        let static_eval = slot.static_eval.load(Ordering::Relaxed);
        let best = unpack_move(slot.best.load(Ordering::Relaxed));
        let bound = u8_to_bound(slot.bound.load(Ordering::Relaxed));
        let key2 = slot.key.load(Ordering::Acquire);
        if key2 != key1 || key2 != key {
            return None;
        }
        Some(tt_entry_from_parts(depth, score, static_eval, bound, best))
    }

    pub(crate) fn store(
        &mut self,
        key: u64,
        depth: i32,
        score: i32,
        static_eval: Option<i32>,
        bound: Bound,
        best: Option<Move>,
    ) {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            store_unique(inner, key, depth, score, static_eval, bound, best);
            return;
        }

        let idx = key as usize & self.inner.mask;
        let slot = &self.inner.entries[idx];
        let cur_key = slot.key.load(Ordering::Acquire);
        let cur_depth = slot.depth.load(Ordering::Relaxed);
        let cur_age = slot.age.load(Ordering::Relaxed);
        let new_depth = depth as i16;
        let current_age = self.inner.current_age.load(Ordering::Relaxed);

        let replace = if cur_depth < 0 || cur_key != key {
            true
        } else {
            let cur_age_bonus = if cur_age == current_age { 4 } else { 0 };
            new_depth + 1 >= cur_depth as i16 + cur_age_bonus
        };

        if replace {
            slot.depth.store(new_depth as i32, Ordering::Relaxed);
            slot.score.store(score, Ordering::Relaxed);
            slot.static_eval.store(
                static_eval
                    .unwrap_or(i16::MIN as i32)
                    .clamp(i16::MIN as i32, i16::MAX as i32),
                Ordering::Relaxed,
            );
            slot.best.store(best.map(pack_move).unwrap_or(u32::MAX), Ordering::Relaxed);
            slot.bound.store(bound_to_u8(bound), Ordering::Relaxed);
            slot.age.store(current_age, Ordering::Relaxed);
            slot.key.store(key, Ordering::Release);
        }
    }
}

fn tt_entry_from_parts(
    depth: i32,
    score: i32,
    static_eval: i32,
    bound: Bound,
    best: Option<Move>,
) -> TTEntry {
    TTEntry {
        depth: depth as i16,
        score,
        static_eval: static_eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        bound,
        best,
    }
}

fn probe_unique(inner: &mut TransTableInner, key: u64) -> Option<TTEntry> {
    let slot = &mut inner.entries[key as usize & inner.mask];
    if *slot.key.get_mut() != key {
        return None;
    }
    let depth = *slot.depth.get_mut();
    if depth < 0 {
        return None;
    }
    Some(tt_entry_from_parts(
        depth,
        *slot.score.get_mut(),
        *slot.static_eval.get_mut(),
        u8_to_bound(*slot.bound.get_mut()),
        unpack_move(*slot.best.get_mut()),
    ))
}

fn store_unique(
    inner: &mut TransTableInner,
    key: u64,
    depth: i32,
    score: i32,
    static_eval: Option<i32>,
    bound: Bound,
    best: Option<Move>,
) {
    let idx = key as usize & inner.mask;
    let slot = &mut inner.entries[idx];
    let cur_key = *slot.key.get_mut();
    let cur_depth = *slot.depth.get_mut();
    let cur_age = *slot.age.get_mut();
    let new_depth = depth as i16;
    let current_age = *inner.current_age.get_mut();

    let replace = if cur_depth < 0 || cur_key != key {
        true
    } else {
        let cur_age_bonus = if cur_age == current_age { 4 } else { 0 };
        new_depth + 1 >= cur_depth as i16 + cur_age_bonus
    };

    if replace {
        *slot.depth.get_mut() = new_depth as i32;
        *slot.score.get_mut() = score;
        *slot.static_eval.get_mut() = static_eval
            .unwrap_or(i16::MIN as i32)
            .clamp(i16::MIN as i32, i16::MAX as i32);
        *slot.best.get_mut() = best.map(pack_move).unwrap_or(u32::MAX);
        *slot.bound.get_mut() = bound_to_u8(bound);
        *slot.age.get_mut() = current_age;
        *slot.key.get_mut() = key;
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

pub(crate) fn piece_index(piece: Piece) -> usize {
    let color_idx = if piece.color == Color::White { 0 } else { 6 };
    let kind_idx = match piece.kind {
        PieceType::Pawn => 0,
        PieceType::Knight => 1,
        PieceType::Bishop => 2,
        PieceType::Rook => 3,
        PieceType::Queen => 4,
        PieceType::King => 5,
    };
    color_idx + kind_idx
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
