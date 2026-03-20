use super::*;

#[derive(Copy, Clone, Eq, PartialEq)]
pub(crate) enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Copy, Clone)]
pub(crate) struct TTEntry {
    pub(crate) key: u64,
    pub(crate) depth: i16,
    pub(crate) score: i32,
    pub(crate) bound: Bound,
    pub(crate) best: Option<Move>,
    pub(crate) age: u8,
}

impl Default for TTEntry {
    fn default() -> Self {
        Self {
            key: 0,
            depth: -1,
            score: 0,
            bound: Bound::Exact,
            best: None,
            age: 0,
        }
    }
}

pub(crate) struct TransTable {
    entries: Vec<TTEntry>,
    mask: usize,
    current_age: u8,
}

impl TransTable {
    pub(crate) fn new(megabytes: usize) -> Self {
        let bytes = (megabytes.max(1) * 1024 * 1024) as usize;
        let entry_size = std::mem::size_of::<TTEntry>();
        let mut n = (bytes / entry_size).max(1).next_power_of_two();
        if n == 0 {
            n = 1;
        }
        Self {
            entries: vec![TTEntry::default(); n],
            mask: n - 1,
            current_age: 0,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.entries.fill(TTEntry::default());
        self.current_age = 0;
    }

    pub(crate) fn new_search(&mut self) {
        self.current_age = self.current_age.wrapping_add(1);
    }

    pub(crate) fn get(&self, key: u64) -> Option<TTEntry> {
        let e = self.entries[key as usize & self.mask];
        if e.depth >= 0 && e.key == key {
            Some(e)
        } else {
            None
        }
    }

    pub(crate) fn store(
        &mut self,
        key: u64,
        depth: i32,
        score: i32,
        bound: Bound,
        best: Option<Move>,
    ) {
        let idx = key as usize & self.mask;
        let cur = self.entries[idx];
        let new_depth = depth as i16;

        let replace = if cur.depth < 0 || cur.key != key {
            true
        } else {
            let cur_age_bonus = if cur.age == self.current_age { 4 } else { 0 };
            new_depth + 1 >= cur.depth + cur_age_bonus
        };

        if replace {
            self.entries[idx] = TTEntry {
                key,
                depth: new_depth,
                score,
                bound,
                best,
                age: self.current_age,
            };
        }
    }
}

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
