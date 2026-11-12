//! Direct-mapped memo of `eval_for_side`, keyed by the full zobrist key.
//!
//! `eval_for_side` is a pure function of the position (pieces, side to move;
//! castling/ep are also in the key, which only makes it stricter), so a hit
//! returns exactly the value a fresh evaluation would. The full 64-bit key is
//! stored and compared, so this never changes search behavior beyond the
//! zobrist-collision risk the TT already carries.

use crate::eval::{eval_for_side, eval_for_side_cached, PawnTable};
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
    /// Pawn-structure memo consulted on eval-cache misses (exact; see eval.rs).
    pawns: PawnTable,
}

impl EvalCache {
    pub(crate) fn new() -> Self {
        Self { entries: vec![Entry { key: EMPTY_KEY, eval: 0 }; EVAL_CACHE_SIZE], pawns: PawnTable::new() }
    }

    pub(crate) fn clear(&mut self) {
        self.entries.fill(Entry { key: EMPTY_KEY, eval: 0 });
        self.pawns.clear();
    }

    /// Start loading `key`'s slot (issued with the TT prefetch when a child
    /// is made, so the child's probe does not stall). A pure hint.
    #[inline(always)]
    pub(crate) fn prefetch(&self, key: u64) {
        let ptr = &self.entries[key as usize & EVAL_CACHE_MASK] as *const Entry;
        #[cfg(target_arch = "aarch64")]
        // SAFETY: PRFM is a hint; it never faults or writes memory.
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr, options(nostack, readonly, preserves_flags));
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: PREFETCHT0 is a hint (SSE, part of the x86-64 baseline).
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let _ = ptr;
    }

    /// `key` must be the zobrist key of `board` (as maintained by search).
    #[inline]
    pub(crate) fn eval(&mut self, board: &Board, key: u64) -> i32 {
        // NNUE evals are a pure function of piece placement and side to move
        // too, so they are memoised the same way (the cache is cleared
        // whenever the evaluator changes). The search consumes side-to-move
        // evals: `eval_for_side` is that wrapper for both evaluators.
        let slot = &mut self.entries[key as usize & EVAL_CACHE_MASK];
        if slot.key == key && key != EMPTY_KEY {
            return slot.eval;
        }
        let eval = if board.nnue.is_some() {
            eval_for_side(board)
        } else {
            eval_for_side_cached(board, &mut self.pawns)
        };
        *slot = Entry { key, eval };
        eval
    }
}
