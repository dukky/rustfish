# Review Findings

This review covers the current `rustfish` codebase with emphasis on correctness risks, strength ceilings, and practical next steps toward a much stronger engine.

## Findings

### [P1] Castling rights can create a rook and panic on rookless FENs

- File: [src/movegen.rs](/Users/andreasholley/dev/stockfish-challenge/src/movegen.rs):200
- Related code: [src/main.rs](/Users/andreasholley/dev/stockfish-challenge/src/main.rs):500, [src/main.rs](/Users/andreasholley/dev/stockfish-challenge/src/main.rs):193
- Castling generation checks the castling-rights bit and intermediate-square conditions, but it does not verify that the rook is actually present on the rook square.
- From a FEN such as `4k3/8/8/8/8/8/8/4K3 w K - 0 1`, the engine generates `O-O`, then panics when rook piece-list state is updated for a rook that does not exist.
- This is both a correctness bug and a crash bug for arbitrary UCI positions.

### [P1] Repetition detection scans the wrong ancestor parity

- File: [src/search.rs](/Users/andreasholley/dev/stockfish-challenge/src/search.rs):417
- The repetition scan starts from `rep_stack.len() - 2`, but the current node is already in the stack.
- That means the first comparison is against the opposite-side position, and in common cases the root position is never checked.
- Simple repetition draws and perpetual-check resources can be missed.

### [P1] Search ignores repetition history before the current root

- Files: [src/search.rs](/Users/andreasholley/dev/stockfish-challenge/src/search.rs):510, [src/uci.rs](/Users/andreasholley/dev/stockfish-challenge/src/uci.rs):114
- `handle_position` replays moves onto the board but does not preserve position-history hashes.
- `handle_go` seeds repetition tracking with only the current root hash.
- Draws that depend on earlier game history are therefore invisible near the root.

### [P2] Quiescence loses mate-distance information

- File: [src/search.rs](/Users/andreasholley/dev/stockfish-challenge/src/search.rs):1273
- Qsearch TT probe/store uses mate-score normalization with `ply = 0` rather than the actual ply.
- Qsearch mate returns are also not ply-aware.
- This weakens mate ordering, mate-distance pruning, and TT consistency.

### [P3] Some exposed LMR tuning options are dead knobs

- Files: [src/uci.rs](/Users/andreasholley/dev/stockfish-challenge/src/uci.rs):96, [src/main.rs](/Users/andreasholley/dev/stockfish-challenge/src/main.rs):728, [src/search.rs](/Users/andreasholley/dev/stockfish-challenge/src/search.rs):1137
- `LmrMinDepth`, `LmrMinMoveIndex`, `LmrExtraDepth`, and `LmrExtraMoveIndex` are exposed as UCI options.
- The actual LMR condition still uses hard-coded thresholds.
- That makes sweep results less trustworthy and wastes tuning time.

## Observations

- The engine already has a solid handcrafted classical base: legal movegen, make/undo, TT, PVS, null move, LMR/LMP, SEE, tapered eval, SMP scaffolding, and useful local tooling.
- The included correctness path is valuable but does not cover arbitrary-FEN castling sanity, repetition correctness, or mate-distance consistency.
- Local validation during this review passed:
  - `cargo check`
  - `perfttest`
  - `selftest 100`
  - UCI smoke: `go depth 4`
  - release build
  - quick bench: about `0.72 Mnps` on the built-in depth-5 suite
- The current architecture is still far from a plausible near-world-class trajectory.
- The biggest medium-term ceiling is not one missing pruning margin; it is the combination of non-incremental eval, relatively expensive board/attack handling, and limited search/eval sophistication.

## Obvious Improvements

1. Fix the correctness debt first.
2. Add regression tests for rookless-castling FENs, repetition parity, repetition from pre-root history, and mate-distance TT handling.
3. Either wire up the exposed LMR options fully or remove them until they are real.
4. Improve speed architecture before more micro-tuning:
   - incremental eval groundwork
   - pawn/eval caching
   - cheaper attack generation
   - stronger SMP coordination and heuristic sharing
5. Keep using frozen binaries and fixed conditions for all serious acceptance decisions.

## Potential Next Steps

### Immediate

1. Fix castling generation so it verifies rook presence before generating a castle move.
2. Fix repetition scanning parity in [src/search.rs](/Users/andreasholley/dev/stockfish-challenge/src/search.rs).
3. Preserve game-history hashes across `position ... moves ...` so root repetition detection is correct.
4. Make qsearch mate handling ply-aware on probe, store, and terminal returns.

### Short Term

1. Expand built-in selftests with targeted tactical/check/pin/repetition regressions.
2. Audit threaded search heuristic sharing so `Threads > 1` does not silently diverge from the single-thread learning path.
3. Add more robust non-startpos correctness tests to the default local loop.

### Medium Term

1. Prioritize speed-enabling structural work over more margin sweeps.
2. Strengthen endgame knowledge and scaling.
3. Revisit evaluation architecture with a path that can scale much further in strength.

## Strategic Note

If the short-term target is roughly human-GM strength and the long-term target is near world class, this engine needs to treat current handcrafted classical search/eval as a foundation, not the destination. A serious path to that level almost certainly requires much stronger search infrastructure and, in practice, a learned evaluation approach such as NNUE, even if the project keeps its no-third-party-crate rule.
