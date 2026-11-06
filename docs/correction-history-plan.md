# Correction history: minimal prototype plan

**Status:** proposal for Opus to audit/implement. Not queued, not merged.
**Date:** 2026-11-06
**Why:** NNUE static eval is smooth but still noisy; correction history learns a
position-class bias from search results. Main lacks it. Previous attempts were
rejected before the eval-scale fix, so the negative result is not conclusive.

## Scope

Do **pawn correction only** first. Do not port Stockfish's full set of pawn /
minor / major / continuation tables in one change.

- Table: `pawn_corr[side][PAWN_HASH_SIZE]`, `i32`, side = 2,
  `PAWN_HASH_SIZE = 16384` (64 KB per side).
- Key: a Zobrist hash of **pawns only**, maintained in `Board` and updated in
  `make_move`/`undo_move` (or recomputed once per position if profiling shows it
  is cheap enough; do not scan all 64 squares per node).
- Correction: `static_eval = raw_eval + pawn_corr[side][pawn_hash & 0x3FFF]`.
- Clamp the corrected eval to the normal eval limit.
- Reset tables on `ucinewgame`, clear hash, and when the net/evaluator changes.

## Search integration

1. Keep `raw_eval` separate from `corrected_eval`; never feed the correction back
   into its own update.
2. TT entries store the **corrected** static eval, like the existing
   `tt_static_eval` path. On a TT hit, use the stored value directly.
3. In `negamax`, after the search result is known and the node is not in check:
   - `diff = score - raw_eval` (not corrected), clamped to ±256;
   - `bonus = clamp(diff * depth / 8, -256, 256)`;
   - gravity update:
     `entry += bonus - entry * abs(bonus) / 512`.
4. Do not update at depth < 4, or on a node with no static eval (in check).
5. Apply the same corrected eval in qsearch; update only in the main search.
6. Threading: if SMP is enabled, use per-thread tables or atomic-safe storage;
   do not introduce a data race. Root-only SMP currently clones engines, so a
   per-engine table is the safe first step.

## Tests

- `pawn_hash` matches a recomputation from the board after make/undo sequences,
  including captures, promotions, en passant.
- Correction table index is deterministic; decay update bounds are correct.
- `bench` changes (expected); `evalhash` should be unchanged when the table is
  empty, so it remains a useful regression gate.
- `cargo test --release`, `perfttest`, `selftest 300`.
- No strength claim until a queue SPRT.

## Validation

1. Build, tests, perft, bench trailer.
2. Local fixed-depth A/B for a smoke signal only; never accept/reject locally.
3. Queue stc/net-gate; LTC if positive.
4. Record the result in `docs/nnue-experiments.md` / `net/experiments.jsonl` even
   if rejected.

## Risks

- Pawn hash maintenance is the main correctness risk; a stale hash silently
  corrupts eval.
- The correction can bias the eval toward deeper-search scores and overfit the
  current net; keep the update rate conservative.
- Inference/NPS cost is small (one table lookup), but the update path adds
  branches at every node. Profile before queueing.
- This is not a proven +Elo feature; it is a cheap, testable hypothesis.

## AVX-512 note

The CCRL reference machine (i7-4770K) is **Haswell: AVX2, no AVX-512**. Any
AVX-512 kernel is dead code on the rating box, and a runtime-dispatched AVX-512
path would only help other machines. Do not prioritize it for CCRL. The current
x86-64-v3 (AVX2) build target is correct. If local AVX-512 experiments are
wanted, keep them separate from the shipped path and report them as local-only.
