# Rustfish Long-Term Plan (No Dependencies)

Goal: keep improving this engine until it can challenge very strong Stockfish settings.

## Phase 0 — Foundation (done)
- [x] Create a no-dependency Rust UCI engine skeleton
- [x] Implement board representation + FEN parsing
- [x] Implement legal move generation (castling, en passant, promotions)
- [x] Implement make/undo move
- [x] Implement alpha-beta negamax + iterative deepening
- [x] Implement quiescence search
- [x] Implement baseline static evaluation (material + PST + mobility)

## Phase 1 — First Strength Jump (in progress)
- [x] Add deterministic Zobrist hashing
- [x] Add transposition table (probe/store, bounds, best move)
- [x] Add principal variation style search (PVS at non-root and root)
- [x] Add aspiration windows in iterative deepening
- [x] Add killer move heuristic
- [x] Add history heuristic
- [x] Add repetition detection in search tree (line-based/perpetual handling)
- [x] Add 50-move rule draw detection in search
- [x] Add better TT replacement strategy (depth+age hybrid)

## Phase 2 — Search Quality
- [x] Null-move pruning
- [x] Late move reductions (LMR)
- [x] Basic late move pruning (LMP)
- [x] Basic futility pruning (shallow, non-PV)
- [x] Basic reverse futility pruning
- [x] Basic SEE for capture ordering + qsearch pruning
- [x] Quiescence check handling (evasion generation when in check)
- [x] Quiescence TT probe/store + simple delta pruning
- [ ] SEE follow-up: pin-aware/king-safety-aware exactness improvements
- [x] Basic check extension
- [x] Mate-distance pruning
- [ ] Singular/check extension tuning follow-up
- [x] Internal iterative deepening for move ordering when TT move is missing
- [x] History heuristic decay per search
- [x] Side-aware quiet history with counter-move ordering and fail-high maluses
- [ ] Tune pruning margins/reductions from gauntlet data

## Phase 3 — Evaluation Quality
 - [x] Tapered eval (true MG/EG interpolation for all major terms)
- [x] Basic pawn structure (isolated, doubled, passed)
- [x] Pawn structure follow-up (connected pawns, candidate passers)
- [x] Pawn structure follow-up (backward pawns, passer support)
- [ ] Pawn structure follow-up (backward pawns, lever detection, passer support)
- [x] Basic king safety (pawn shield, attacked king zone, open files near king)
- [x] King safety follow-up (attack units by piece type)
- [x] King safety follow-up (tropism, scaling by attacking material)
- [x] King safety scaling by game phase/material
- [x] Basic piece activity (bishop pair, rooks on open/semi-open files)
- [x] Piece activity follow-up (outposts, 7th-rank rooks)
- [x] Piece activity follow-up (space, bishop scope)
- [ ] Piece activity follow-up (trapped pieces)
- [x] Endgame terms (king activity, basic mop-up / king chase)
- [ ] Endgame terms follow-up (opposition-ish heuristics, rook endings, pawn races)
- [ ] Incremental evaluation updates (faster nodes/sec)

## Phase 4 — Engine Correctness + Performance
- [x] Add perft/divide commands
- [x] Add built-in perft smoke suite (startpos d1..d4)
- [x] Expand perft suite with additional canonical tricky positions (castling/en-passant/promotion basics)
- [x] Add move-generation/make-undo invariant selftest command
- [x] Expand invariants with targeted edge-case suites
- [ ] Expand invariants with tactical/pin/check-regression suites
- [ ] Optimize hot paths (copy avoidance, branch cleanup)
- [ ] Add optional bitboard backend for speed comparison
- [x] Add benchmark harness script (nodes/sec by depth/position set)

## Phase 4.5 — Refactor / Maintainability
- [x] Split engine into modules (`book`, `eval`, `movegen`, `search`, `tt`, `uci`)
- [x] Keep behavior intact through refactor (perft/selftest/UCI/gauntlet smoke verified)
- [ ] Consider further split of `main.rs` into `board` / `types` modules
- [ ] Add module-local unit tests as code stabilizes

## Phase 5 — Practical Strength Tooling
- [x] Add basic built-in opening book support
- [x] Add OwnBook UCI option
- [x] Expand built-in opening book depth
- [ ] Add opening book external plain-text support
- [ ] Add simple endgame tablebase probing (optional local files)
- [ ] Add self-play tuning loop (no external crate dependency)
- [x] Add basic gauntlet script vs fixed opponents (UCI, no deps)
- [ ] Add SPRT-style statistical decision logic to gauntlet
- [ ] Add automatic Elo confidence intervals / ladder summary to gauntlet
- [ ] Track Elo progression over frozen-binary versions

## Current Focus
1. Improve evaluation stability: true tapered eval, stronger king attack scaling, backward/support pawn logic.
2. Improve search efficiency: tune pruning margins/reductions, tighten SEE, speed hot paths.
3. Improve measurement quality: frozen-binary gauntlets, Elo ladder summaries, confidence bounds.
4. Keep refactor healthy: only small follow-up structure changes while strength work continues.

## Working Protocol
- Keep this file current.
- Check off completed tasks immediately.
- Add new tasks when needed.
- Prioritize search strength + correctness before fancy features.

## Current Blocker
- None right now.
