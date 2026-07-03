# Post-2100 Strength Plan

## Context
The engine appears to have reached a point where small local improvements are no longer producing reliable strength gains.

That is normal.

At roughly this level, the problem is usually no longer:
- missing obvious first-wave heuristics
- missing basic pruning ideas
- missing basic handcrafted eval terms

Instead, the remaining gap to stronger engines comes from:
- better core data structures
- better quality per node
- better static-eval reliability
- better tactical selectivity
- better TT leverage
- eventually, a stronger eval framework than classical handcrafted terms alone

This note is meant to help re-center development priorities.

---

## Main diagnosis
The current engine is likely in a **plateau regime**.

That means:
- tiny pruning-margin sweeps are mostly noise
- small handcrafted eval ornaments rarely survive A/B
- hot-path speedups may improve NPS but not practical Elo
- root-split SMP will not rescue a single-thread architecture that still leaves too much strength on the table

In other words:
- the current issue is probably **not execution quality by the other agent**
- the current issue is that the engine now needs **structural upgrades** rather than more local tuning

---

## Where Stockfish is still far ahead
Stockfish is better not because of one trick, but because many layers are stronger at once.

The biggest practical gaps still appear to be:
- move ordering depth and learning quality
- pruning/selectivity sophistication
- evaluation quality and calibration
- extension/reduction discipline
- TT structure and reuse quality
- tactical selectivity and static-eval trustworthiness

The current engine already has decent first-generation versions of many search ideas. The next gains need to come from the second generation of engine structure.

---

## What to prioritize now

## Priority Tier 1: strongest near-term ROI
These are the most promising next steps for meaningful strength gain.

### 1. Clustered transposition table
Current TT appears to be effectively direct-mapped, one entry per index.

Why this matters:
- weak collision behavior
- weaker persistence of useful entries
- weaker move ordering support
- weaker future SMP support

Recommended upgrade:
- move to a **clustered TT** (e.g. 4-entry bucket/cluster)
- improve replacement policy inside the cluster
- preserve depth/age-based logic, but make it compete within a bucket instead of a single slot

Expected value:
- better TT hit quality
- better ordering
- better pruning support
- better basis for later SMP

This is one of the strongest remaining core-structure upgrades.

### 2. Store static eval in TT
Current TT stores score/bound/best move, but not static eval.

Why this matters:
- static eval gets recomputed repeatedly
- static eval in TT helps:
  - pruning
  - move ordering
  - null-move gating
  - reverse futility style logic

Recommended upgrade:
- store static eval in TT entries
- use it conservatively where current code calls `eval_for_side()` repeatedly in search

Expected value:
- reduces eval recomputation
- improves pruning quality
- improves node efficiency

### 3. Pawn hash table
This is one of the clearest missing classic engine components.

Why this matters:
- pawn structure is expensive relative to how slowly it changes
- many search nodes revisit the same pawn skeleton
- king safety and passer logic often depend heavily on pawn structure

Recommended upgrade:
- add a pawn hash keyed only by pawn placement (and maybe side-dependent derived info if needed)
- cache:
  - pawn files
  - passed pawn status / candidate information
  - isolated/doubled/backward summaries
  - maybe pawn shield / king shelter summaries later

Expected value:
- lower eval cost
- more room for richer pawn eval
- better quality per node

### 4. Capture history
Current move ordering is decent, but still lacks one of the major modern ordering ingredients.

Why this matters:
- SEE is useful but not enough
- good captures and bad captures are not fully separable from static exchange alone
- search should learn which captures actually cause cutoffs

Recommended upgrade:
- add capture history indexed by mover / target / captured piece or similar compact scheme
- integrate into capture ordering alongside SEE

Expected value:
- better move ordering
- stronger tactical search efficiency
- practical Elo upside without requiring huge eval changes

### 5. Singular extension
This is already acknowledged as unfinished in the plan and remains a high-value structural search feature.

Why this matters:
- improves trust in dominant TT moves
- helps tactical lines where one move is clearly superior
- can improve both tactical depth and move ordering confidence

Recommended upgrade:
- implement carefully and conservatively
- validate heavily with correctness and gauntlets

Expected value:
- one of the most realistic next search-concept gains still missing

### 6. ProbCut
This is another major search feature that is still absent.

Why this matters:
- can provide meaningful tactical selectivity gains when integrated carefully with qsearch/SEE
- often one of the major steps between a decent alpha-beta engine and a stronger one

Recommended upgrade:
- start conservatively
- validate with tactical regressions and A/B, not bench alone

Expected value:
- likely better than continued micro-tuning of current futility/null margins

---

## Priority Tier 2: eval architecture improvements
These are likely needed for the next band of strength.

### 7. Incremental evaluation groundwork
Still unchecked in `plan.md`, and still highly relevant.

Why this matters:
- lets the engine afford more eval at the same movetime
- reduces cost per node
- improves future headroom for richer terms

Recommended direction:
- do not try to incrementalize everything at once
- start with:
  - material / phase
  - PST
  - pawn-file counts or pawn summaries
  - king-square dependent pieces of eval where feasible

Expected value:
- large medium-term payoff
- also helps future SMP because cheaper nodes matter even more there

### 8. Correction history
This is not currently in the plan, but should be.

Why this matters:
- classical handcrafted eval often has systematic biases
- correction-style mechanisms let search outcomes inform eval trust
- modern engines use these ideas to improve pruning safety and search guidance

Recommended upgrade:
- add a simple correction-history mechanism tied to search outcomes and hash/material buckets
- start small and validate carefully

Expected value:
- can improve quality per node significantly
- especially useful once handcrafted eval reaches diminishing returns

### 9. Better endgame knowledge
The plan already includes more endgame follow-up, which still looks worthwhile.

Most promising handcrafted endgame areas:
- rook endings
- pawn races
- opposition-ish king/pawn themes
- cleaner endgame scaling

Why this matters:
- once obvious tactical mistakes decline, endgame conversion becomes a larger Elo source
- also improves practical results in drawn-ish positions

Expected value:
- probably more reliable than adding many more middlegame ornament terms

---

## Priority Tier 3: architecture ceiling raisers
These may not be the next patch, but they matter for longer-term progress.

### 10. Optional bitboard backend
This is still on the plan and still makes sense.

Why this matters:
- current mailbox/array design can be pushed only so far
- bitboards make attack generation, pinned logic, mobility, and many eval/search helpers faster and cleaner
- also make future SIMD and other low-level optimizations more realistic

Expected value:
- potentially large, but higher effort and higher risk
- probably worth doing only when you are ready for a structural refactor, not as a side quest

### 11. Revisit SMP only after stronger single-thread structure
Threading should remain experimental until the single-thread core is stronger.

Why this matters:
- a mediocre scalar search does not become strong just by adding threads
- current root-split SMP already showed limited payoff
- shared-TT Lazy SMP may be worth revisiting later, but not before single-thread architecture improves further

Recommended policy:
- freeze SMP work unless a very simple high-confidence improvement is available
- come back after TT/eval/search structure is upgraded

### 12. Long-term NNUE / neural-eval path
This is not necessary for the next few patches, but it should be on the strategic radar if the long-term goal really remains “challenge very strong Stockfish settings.”

Why this matters:
- handcrafted eval can still improve a lot
- but Stockfish-class strength today depends heavily on much stronger evaluation machinery
- no-dependency does not prevent a pure-Rust inference path with embedded/local weights

Recommended action now:
- do not implement immediately if the team is not ready
- but add a roadmap item so the project does not pretend handcrafted eval alone will eventually erase the whole gap

---

## What to deprioritize now
These are likely lower ROI than the items above.

### 1. More local pruning-margin sweeps
Recent results already suggest these are flattening out.

### 2. More small handcrafted eval ornaments
Unless very targeted and strongly justified, these are unlikely to move Elo much now.

Examples to deprioritize relative to structural work:
- tiny mg/eg piece bonuses
- another round of local king-safety weight nudges
- many small PST/space tweaks without broader architecture changes

### 3. SMP complexity before scalar architecture improves
Do not spend a lot more time trying to rescue root-split SMP right now.

### 4. SIMD / GPU / AMX
These remain lower priority than search/eval data-structure work.

---

## What should be added to the roadmap
The current roadmap is good, but it still misses some high-leverage classic/modern engine ideas.

Recommended additions near the top of active priorities:
- [ ] Add clustered transposition table
- [ ] Store static eval in TT
- [ ] Add pawn hash table
- [ ] Add capture history for capture ordering
- [ ] Add correction history / static-eval calibration support
- [ ] Add ProbCut
- [ ] Long-term: investigate NNUE-style eval path compatible with no-dependency goals

These are more important than several remaining local eval follow-ups.

---

## Proposed reordered priorities
If the project wants a clearer post-plateau focus, this would be a good order.

### Near-term
1. Clustered TT
2. TT static eval
3. Pawn hash
4. Capture history
5. Singular extension
6. ProbCut

### Medium-term
7. Incremental eval groundwork
8. Correction history
9. Better rook endings / pawn races / endgame scaling

### Later
10. Bitboard backend experiment
11. Revisit SMP only after stronger scalar core is frozen
12. Long-term NNUE path

---

## If only three things get chosen
If the team needs a very short list, the best three bets appear to be:

### 1. Clustered TT + static eval in TT
This strengthens:
- move ordering
- pruning
- eval reuse
- future SMP viability

### 2. Pawn hash + eval caching groundwork
This improves:
- eval speed
- room for richer eval
- quality per node

### 3. Singular extension or ProbCut
The search now likely needs one more serious concept, not another tuning sweep.

---

## Guidance for the current other-agent workflow
Since the other agent is still working off:
- `performance-followup.md`
- and has not yet started `smp-recovery.md`

Suggested message to them:
- the current plateau is expected
- do not interpret flat results as failure of effort
- prioritize **single-thread structural upgrades** over more SMP work
- treat root-split `Threads` as experimental and non-blocking
- if choosing next work, prefer TT / eval architecture / search-concept upgrades over local heuristic tuning

---

## Bottom line
The engine is probably not stuck because of bad implementation work.
It is stuck because the remaining gains now require **structural upgrades**.

The best path forward is probably:
- stop expecting tiny tuning wins to carry the project much further
- improve the core information architecture of the engine
- add at least one more major search idea
- and prepare for a stronger long-term eval path

That is the most plausible route out of the current ~2100 plateau.
