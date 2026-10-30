//! Direct-mapped memo of `eval_for_side`, keyed by the full zobrist key.
//!
//! `eval_for_side` is a pure function of the position (pieces, side to move;
//! castling/ep are also in the key, which only makes it stricter), so a hit
//! returns exactly the value a fresh evaluation would. The full 64-bit key is
//! stored and compared, so this never changes search behavior beyond the
//! zobrist-collision risk the TT already carries.

use crate::eval::eval_for_side;
use crate::Board;

const EVAL_CACHE_BITS: u32 = 17;
const EVAL_CACHE_SIZE: usize = 1 << EVAL_CACHE_BITS;
const EVAL_CACHE_MASK: usize = EVAL_CACHE_SIZE - 1;
/// Marks an empty slot; a key equal to it simply bypasses the cache.
const EMPTY_KEY: u64 = 0;

#[derive(Clone, Copy)]
struct Entry {
    key: u64,
    eval: i32,
}

#[derive(Clone)]
pub(crate) struct EvalCache {
    entries: Vec<Entry>,
}

impl EvalCache {
    pub(crate) fn new() -> Self {
        Self { entries: vec![Entry { key: EMPTY_KEY, eval: 0 }; EVAL_CACHE_SIZE] }
    }

    pub(crate) fn clear(&mut self) {
        self.entries.fill(Entry { key: EMPTY_KEY, eval: 0 });
    }

    /// `key` must be the zobrist key of `board` (as maintained by search).
    #[inline]
    pub(crate) fn eval(&mut self, board: &Board, key: u64) -> i32 {
        let slot = &mut self.entries[key as usize & EVAL_CACHE_MASK];
        if slot.key == key && key != EMPTY_KEY {
            return slot.eval;
        }
        let eval = eval_for_side(board);
        *slot = Entry { key, eval };
        eval
    }
}
