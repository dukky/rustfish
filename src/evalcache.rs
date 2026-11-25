//! Direct-mapped memo of `eval_for_side`, keyed by the full zobrist key.
//!
//! `eval_for_side` is a pure function of the position (pieces, side to move;
//! castling/ep are also in the key, which only makes it stricter), so a hit
//! returns exactly the value a fresh evaluation would. The full 64-bit key is
//! stored and compared, so this never changes search behavior beyond the
//! zobrist-collision risk the TT already carries.

use crate::eval::{eval_for_side, eval_for_side_cached, PawnTable};
use crate::Board;

const EVAL_CACHE_BITS: u32 = 21;
const EVAL_CACHE_SIZE: usize = 1 << EVAL_CACHE_BITS;
const EVAL_CACHE_MASK: usize = EVAL_CACHE_SIZE - 1;

/// One slot in 8 bytes: bit 63 = occupied, then the key's high
/// 64 - EVAL_CACHE_BITS bits (the low bits are the slot index, so index and
/// tag together are the whole key: a hit is still a full 64-bit key match),
/// then the eval as i16 in the low 16 bits. Evals outside i16 are not cached.
#[derive(Clone, Copy)]
struct Entry(u64);

const VALID: u64 = 1 << 63;
const _: () = assert!(64 - EVAL_CACHE_BITS + 16 + 1 <= 64);

impl Entry {
    const EMPTY: Self = Entry(0);

    #[inline(always)]
    fn tag(key: u64) -> u64 {
        VALID | ((key >> EVAL_CACHE_BITS) << 16)
    }

    #[inline(always)]
    fn get(self, key: u64) -> Option<i32> {
        (self.0 & !0xFFFF == Self::tag(key)).then_some(self.0 as u16 as i16 as i32)
    }
}

#[derive(Clone)]
pub(crate) struct EvalCache {
    entries: Vec<Entry>,
    /// Pawn-structure memo consulted on eval-cache misses (exact; see eval.rs).
    pawns: PawnTable,
}

impl EvalCache {
    pub(crate) fn new() -> Self {
        Self { entries: vec![Entry::EMPTY; EVAL_CACHE_SIZE], pawns: PawnTable::new() }
    }

    pub(crate) fn clear(&mut self) {
        self.entries.fill(Entry::EMPTY);
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
        if let Some(eval) = slot.get(key) {
            return eval;
        }
        let eval = if board.nnue.is_some() {
            eval_for_side(board)
        } else {
            eval_for_side_cached(board, &mut self.pawns)
        };
        if let Ok(e) = i16::try_from(eval) {
            *slot = Entry(Entry::tag(key) | e as u16 as u64);
        }
        eval
    }
}
