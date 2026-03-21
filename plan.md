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
- [x] Store static eval in TT and reuse it for pruning/qsearch eval fallback

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
- [x] One-ply continuation history for quiet move ordering
- [x] Replace full staged non-root/qsearch move sorting with staged pick-next move picker
- [x] Pruning stability follow-up (null-move eval gate, priority-quiet LMP/LMR guards)
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
- [x] Piece activity follow-up (opening development / castling incentives)
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
- [x] Optimize hot paths (SEE attacker lookup, bounded repetition scan, cheaper history decay cadence)
- [x] Reduce staged move-ordering allocation churn (single scored sort instead of bucket rebuilds)
- [x] Reduce eval rescanning overhead with piece-list reuse for king safety and mobility
- [x] Reduce eval allocation churn and pawn-structure helper rescans
- [x] Reduce eval duplicate mobility work (reuse bishop mobility in main mobility pass)
- [x] Reduce legal-move filtering cost with exact pin/check-based legality filtering
- [x] Reuse move buffers across legal move generation and search callers
- [x] Add experimental root-split `Threads` support with conservative gating for longer/deeper searches
- [x] Recover `Threads` with a shared-TT lazy-SMP style search path
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
- [x] Add book-style option for repertoire experiments
- [ ] Add opening book external plain-text support
- [ ] Add simple endgame tablebase probing (optional local files)
- [ ] Add self-play tuning loop (no external crate dependency)
- [x] Add basic gauntlet script vs fixed opponents (UCI, no deps)
- [x] Add frozen-binary A/B gauntlet mode with parallel workers
- [x] Add opening-suite support for paired A/B gauntlets
- [x] Add SPRT-style statistical decision logic to gauntlet
- [x] Add automatic Elo confidence intervals / ladder summary to gauntlet
- [x] Add machine-readable gauntlet result logging
- [x] Add sweep driver for candidate variants
- [x] Add two-stage sweep flow (parallel fixed-length screens, then SPRT confirmation)
- [x] Add single-command validation runner
- [x] Expose core pruning/reduction knobs as UCI options for sweep-based tuning
- [x] Track Elo progression over frozen-binary versions with a local round-robin rating pool

## Current Focus
1. Improve search efficiency: keep pushing post-2100 structural work, including the new shared-TT SMP path and the next round of single-thread eval/search cleanup.
2. Improve evaluation stability: keep accepted MG/EG terms and only add changes that survive A/B gauntlets.
3. Improve measurement quality: frozen-binary gauntlets, JSONL logs, confidence bounds, early-stop logic, sweepable UCI tuning knobs, and direct option-vs-option head-to-head checks.
4. Keep refactor healthy: only small follow-up structure changes while strength work continues.

## Post-2100 Review
- Accepted from the post-2100 structural path:
  - TT static eval reuse (`f4143cd`)
  - staged pick-next non-root/qsearch move picker (`9b0b346`)
  - move-buffer reuse for pseudo/legal generation and search-side move collection (bench-positive, strength-neutral)
- Long confirmation that the new structural baseline is real:
  - `9b0b346` vs `f4143cd`, `96g / 100ms / parallel 2`: `41/37/18`, score `62.0%`, about `+84.9 Elo`
- External confidence progress:
  - `9b0b346` vs Stockfish `2100`, `96g / 100ms / parallel 2`: `59/9/28`, score `66.1%`, about `+116.4 Elo`
  - `9b0b346` vs Stockfish `2200`, `96g / 100ms / parallel 2`: `44/12/40`, score `52.1%`, about `+14.5 Elo`
  - `9b0b346` vs Stockfish `2300`, `96g / 100ms / parallel 2`: `46/8/42`, score `52.1%`, about `+14.5 Elo`
  - those `2200` / `2300` results are suspiciously similar, so frozen A/B plus a local pool is a better anchor than over-interpreting `UCI_LimitStrength`
- Structural ideas tried and rejected so far:
  - clustered TT
  - pawn hash / pawn-eval cache groundwork
  - capture history
  - correction history
  - ProbCut
  - singular extension variants tried so far
- Best remaining post-2100 priorities:
  - incremental eval groundwork
  - better endgame knowledge / scaling
  - another genuinely different search-concept upgrade, not more local margin tuning

## Working Protocol
- Keep this file current.
- Check off completed tasks immediately.
- Add new tasks when needed.
- Prioritize search strength + correctness before fancy features.

## Current Blocker
- Recent pruning-parameter sweeps are mostly neutral or negative. SMP throughput is now meaningfully better with shared TT, but strength is still unvalidated and single-thread overhead remains a follow-up target.

## SMP Recovery
- Replaced the old static root-split worker assignment with a shared-TT lazy-SMP style path:
  - all workers search the same root
  - workers share TT and zobrist state
  - workers keep thread-local history / killers / continuation data
  - root move order is diversified slightly per worker instead of statically partitioned
- Fixed shared TT sizing so the atomic slot table uses its true slot size instead of the old compact entry size.
- Current throughput snapshot on the local depth-7 fixed bench (`Hash=256`):
  - shared-TT SMP candidate: `Threads=1` `~720k nps`, `Threads=2` `~1.12M nps`, `Threads=4` `~1.57M nps`, `Threads=8` `~2.47M nps`
  - old root-split baseline: `Threads=1` `~645k nps`, `Threads=2` `~672k nps`, `Threads=4` `~1.07M nps`
- Depth-8 bench also stayed monotonic (`~729k -> ~1.14M -> ~1.40M -> ~2.16M`), so the recovered SMP path is a real throughput improvement even though it is not close to linear scaling yet.
