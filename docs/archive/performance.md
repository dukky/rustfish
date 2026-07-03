# Performance Summary

## Overall impression
This repo looks structurally strong but not especially fast yet.

It already has many meaningful search features in place:
- TT + Zobrist hashing
- PVS + aspiration windows
- null-move pruning
- LMR / LMP / futility / reverse futility
- killers / history / continuation history
- qsearch + SEE
- benchmark / gauntlet / frozen binary workflow

So the engine is beyond the "missing major search ideas" stage. The likely bottleneck is now **cost per node**, not lack of core search features.

## Current performance signal
From the latest frozen metadata:
- `bin/rustfish_0e57d10_20260320_172622.json`
- bench depth 5 total: about **428k NPS**

Caveat:
- the bench output shows `startpos` as `nodes=0 time_ms=0`, so the benchmark script appears to undercount very short searches there
- treat the number as a rough ballpark, not a precise ceiling

Still, the general takeaway is:
- performance is reasonable for a simple no-dependency array/mailbox-style engine
- raw speed is likely well below what a more optimized architecture could achieve

## Search tuning status
`plan.md` says recent pruning-parameter sweeps are mostly neutral or negative, and the logs support that.

One potentially promising recent result:
- `tmp/sweeps/20260320_180413_lmr_light.jsonl`
- 32 games / 20 ms vs baseline
- score: **60.9%**
- approximate reported Elo: **+77**

That sample is too small to treat as accepted truth, but it does suggest there may still be some gains in reduction policy / search ordering.

By contrast, aggressive null tuning looked worse:
- `tmp/sweeps/20260320_180413_null_aggr.jsonl`
- score: **40.6%**
- approximate reported Elo: **-66**

So the repo seems near diminishing returns on local pruning-threshold tuning. Structural efficiency work is probably higher leverage now.

## Main hotspots / likely quick wins

### 1. SEE is likely expensive
Files:
- `src/search.rs`

Relevant functions:
- `see()`
- `least_valuable_attacker()`
- `piece_attacks_square()`

Why it stands out:
- `see()` copies `board.squares`
- then repeatedly finds attackers by scanning all 64 squares
- each attacker probe calls `piece_attacks_square()`
- SEE is used in move ordering and qsearch pruning, so this cost repeats often

Assessment:
- this is one of the clearest immediate performance targets
- a cheaper SEE implementation could improve NPS without changing engine behavior much

### 2. Move ordering does a lot of allocation and sorting
Files:
- `src/search.rs`

Relevant functions:
- `order_moves()`
- `staged_moves()`

Current pattern:
- generate full legal move list
- split into multiple temporary vectors
- sort multiple buckets with `sort_by_cached_key`
- rebuild a final ordered vector

Why it matters:
- this is clean and readable
- but it adds repeated allocation, copying, and sorting overhead deep in the search tree

Assessment:
- likely good constant-factor speedup available here
- especially by reducing full sorts and temporary vector churn

### 3. Legal move generation is fully make/undo filtered
Files:
- `src/main.rs`
- `src/movegen.rs`

Relevant function:
- `Board::generate_legal_moves()`

Current pattern:
- generate pseudo-legal moves
- for every move: `make_move()` -> `is_in_check()` -> `undo_move()`

Why it matters:
- very robust and simple
- but expensive in the search hot path

Assessment:
- this is probably one of the biggest architectural taxes on nodes/sec
- even partial reductions in how often full legal generation is needed could matter

### 4. Evaluation is full recomputation and scan-heavy
Files:
- `src/eval.rs`
- `src/movegen.rs`

Relevant patterns:
- full board scan every eval
- repeated pawn-list and pawn-file work
- repeated passed/candidate checks with nested pawn scans
- king safety recomputes attack-related information
- mobility rescans all pieces via `mobility_count()`

Assessment:
- evaluation is likely a major cost center
- incremental eval is already listed in `plan.md` and still looks like one of the best medium-term speed projects
- even before full incremental eval, reducing repeated scans would likely help

### 5. Repetition detection is linear each node
Files:
- `src/search.rs`

Relevant pattern:
- `rep_stack.iter().rev().skip(1).any(|k| *k == key)`

Assessment:
- simple and correct
- probably not the top hotspot, but still avoidable overhead
- a lightweight repetition counter or current-line hash count would be cheaper

### 6. History decay touches a very large table
Files:
- `src/main.rs`

Relevant items:
- `ContinuationHistoryTable`
- `EngineState::decay_history()`

Current pattern:
- full-table decay over history and continuation-history structures once per `go`

Assessment:
- not the hottest per-node issue
- but a very obvious bulk cost
- probably worth replacing with lazier decay or cheaper scaling logic eventually

## Prioritized recommendations

### Best short-term quick wins
1. **Cheapen SEE**
2. **Reduce move sorting / temporary vector churn**
3. **Trim search-path legal move overhead where possible**
4. **Reduce eval rescanning**

### Best medium-term structural win
- **Incremental evaluation updates**

That item is already on the roadmap and still looks justified.

## Bottom line
The repo is in a good place conceptually: the search has many modern ingredients already, and the tooling is solid.

The biggest remaining opportunity is probably **efficiency of the node pipeline**:
- move generation cost
- SEE cost
- ordering/sorting churn
- full recomputed evaluation

So the next meaningful performance jump is more likely to come from **structural speedups** than from another round of small pruning-parameter sweeps.
