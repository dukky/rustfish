# Performance Follow-up for Current Work

## Context
Recent small hot-path changes appear to have produced limited or no clear engine-strength gain overall.

That outcome is not surprising. Most of the changes completed so far are **constant-factor speedups**, not major changes to:
- move quality
- search structure
- evaluation quality
- parallel scaling

So they may improve nodes/sec somewhat without necessarily moving Elo much at short time controls.

---

## What appears to be done

These items look implemented in the current codebase.

### 1. SEE attacker lookup was improved
Files:
- `src/search.rs`

Observed change:
- the old full-board-scan style `least_valuable_attacker()` appears to have been replaced with a more targeted geometric lookup using helpers such as:
  - `piece_on_square()`
  - `first_slider_attacker()`

Why this matters:
- SEE was one of the clearer hot-path costs
- this should reduce some per-node overhead in:
  - move ordering
  - qsearch capture filtering

Assessment:
- good cleanup
- likely a real speed improvement
- but not necessarily large enough by itself to move Elo much

### 2. Repetition detection is now bounded
Files:
- `src/search.rs`

Observed change:
- there is now an `is_repetition(rep_stack, key, halfmove_clock)` helper
- this replaces the broader reverse scan approach

Why this matters:
- it avoids scanning the full repetition stack every time
- it uses parity and the halfmove bound more intelligently

Assessment:
- sensible hot-path optimization
- good to keep
- likely modest in overall impact

### 3. History decay cadence was made cheaper
Files:
- `src/main.rs`

Observed change:
- `searches_since_decay` was added
- `prepare_for_search()` now delays `decay_history()` so it only happens every few searches instead of every search

Why this matters:
- continuation history is a large table
- full decay every `go` is expensive and unnecessary

Assessment:
- another good constant-factor cleanup
- useful, but not likely a major strength lever by itself

### 4. `plan.md` reflects these as completed
`plan.md` now marks the hot-path item as done:
- `Optimize hot paths (SEE attacker lookup, bounded repetition scan, cheaper history decay cadence)`

That matches what is visible in the code.

---

## What still appears missing

The larger structural items from the earlier performance review still seem mostly untouched.

### 1. Move ordering / move-picking structure is still heavy
Files:
- `src/search.rs`

Current situation still appears to be:
- full legal move generation
- split into multiple temporary vectors
- multiple bucket sorts
- rebuild final ordered list

Relevant functions:
- `order_moves()`
- `staged_moves()`

Why this still matters:
- there is still allocation churn
- there is still full sorting work in hot search paths
- this remains a likely node-cost hotspot

Assessment:
- still a strong candidate for future structural improvement
- likely higher leverage than more tiny search-parameter tuning

### 2. Legal move generation is still fully make/undo filtered
Files:
- `src/main.rs`
- `src/movegen.rs`

Current situation:
- `Board::generate_legal_moves()` still:
  - generates pseudo-legal moves
  - makes each move
  - tests `is_in_check()`
  - undoes the move

Why this still matters:
- this is simple and correct
- but it is still one of the major structural taxes on nodes/sec

Assessment:
- still a large cost center
- not yet addressed

### 3. Evaluation still looks like full recomputation
Files:
- `src/eval.rs`

Current situation still appears to be:
- full board scan per eval
- rebuild pawn/piece lists per eval
- repeated pawn structure scans
- king safety recomputation
- mobility recomputation

Why this still matters:
- evaluation is probably still one of the biggest per-node costs
- especially once obvious search micro-costs have already been cleaned up

Assessment:
- still a major remaining performance opportunity

### 4. Incremental evaluation is still missing
`plan.md` still has unchecked:
- `Incremental evaluation updates (faster nodes/sec)`

Assessment:
- still one of the clearest medium-term speed projects
- probably more important than SIMD or GPU ideas

### 5. Structural move picker work still appears missing
Even with the SEE improvement, the overall move pipeline still does a lot of:
- staging into buckets
- sorting
- vector rebuilding

Assessment:
- a more incremental move-picker design still looks like one of the more promising remaining search-efficiency tasks

### 6. Multithreading has not yet landed
At the time of this note, there is no visible sign of:
- `Threads` UCI option
- Lazy SMP
- root-split SMP
- shared TT across workers

That is fine because multithreading is reportedly in progress now.

### 7. Bitboards / SIMD / GPU work is still not relevant yet
Still missing:
- optional bitboard backend
- architecture-specific SIMD fast paths
- any GPU work

Assessment:
- this is acceptable
- these are not the highest-priority missing items right now

---

## Why the recent changes may not have improved strength much
This is the expected behavior for the kind of optimizations that were made.

The completed work primarily reduces **node cost**, but does not directly improve:
- move ordering quality
- search selectivity quality
- tactical correctness
- evaluation accuracy

That means:
- bench numbers may improve a bit
- search may still reach roughly the same effective depth at short controls
- game results may remain flat or noisy in small gauntlets

So the weak Elo response does **not** mean those changes were bad. It more likely means:
- they were modest constant-factor improvements
- and the engine now needs either larger structural speedups or search-quality improvements to get a visible practical gain

---

## Recommended interpretation for the current multithreading work
Since basic multithreading is being implemented now, the current state is best viewed as:

### Already cleaned up enough to support SMP reasonably
The recent hot-path work was still worthwhile because it:
- trims wasted scalar work
- improves the base single-thread engine
- gives a better foundation before adding threads

### But the next visible gain is more likely from SMP or bigger structural work
The most likely next sources of measurable progress are:
1. **Basic multithreading / SMP**
2. **Structural move-picking improvements**
3. **Evaluation cost reduction / incremental eval**
4. **A larger search-structure improvement**

---

## Recommended priority after basic multithreading lands

### Short-term after SMP baseline exists
1. Verify `Threads=1` remains correct and competitive
2. Measure scaling on real hardware
3. Check whether SMP improves practical depth / score under fixed movetime
4. Run short A/B gauntlets of `Threads>1` vs `Threads=1`

### Best remaining single-thread structural opportunities
If SMP is in place or mostly done, the strongest remaining non-SMP candidates still look like:
1. **move-picking / ordering structure cleanup**
2. **eval rescanning reduction / incremental eval groundwork**
3. **legal move overhead reduction**

---

## Condensed handoff summary

### Implemented recently
- cheaper SEE attacker lookup
- bounded repetition scan
- cheaper history decay cadence

### Still missing
- move-picking / ordering structure rewrite
- legal move generation cost reduction
- full eval cost reduction / incremental eval
- multithreading integration (currently in progress)

### Likely reason strength stayed flat
- recent changes were mostly small constant-factor speedups
- not large search-quality or eval-quality changes
- therefore practical Elo gain may be small or hard to detect

### Best next opportunities
- get basic SMP working cleanly
- then target structural move-picking and eval-cost work
