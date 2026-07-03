# SMP Recovery / Handoff Notes

## Context
I checked the current threading implementation and why it likely failed to produce a convincing gain.

The current version is not a Lazy SMP design. It is an **experimental root-split implementation** with conservative gating, and it is understandable that it did not pay off much yet.

This note is meant as a practical handoff for the current agent.

---

## What is already in the tree

### 1. `Threads` UCI option exists
Files:
- `src/uci.rs`
- `src/main.rs`

Observed:
- `setoption name Threads value N` is supported
- engine stores `threads: usize`
- default is `1`
- advertised in UCI options

### 2. Experimental root-split search exists
Files:
- `src/search.rs`

Observed:
- `handle_go()` enables threading only under conservative gating:
  - enough root moves
  - `Threads > 1`
  - `depth >= MIN_PARALLEL_DEPTH`
  - and effectively longer searches / longer movetime
- `search_root_parallel()` partitions root moves across workers
- `search_root_worker()` searches assigned root moves independently

### 3. Workers are cloned engine states
Files:
- `src/main.rs`

Observed:
- `worker_clone()` creates per-worker engine states
- each worker gets its own:
  - TT
  - history
  - counter-moves
  - continuation history
- own book is disabled in workers

### 4. Some useful single-thread speedups also landed
Separate from SMP, the codebase now also appears to have:
- bounded repetition detection
- cheaper SEE attacker lookup
- reduced history-decay cadence
- reduced eval rescanning via piece-list reuse
- a legal-move fast path using `move_may_expose_king()` to skip make/undo for some safe moves
- staged move ordering apparently moved toward a single scored sort rather than bucket rebuilds

So the engine is not standing still; the SMP work is happening on top of a somewhat improved scalar base.

---

## Why the current SMP version likely underperformed

## Main diagnosis
The current implementation has the **costs of threading** without the **main benefits of strong SMP information sharing**.

In short:
- workers are independent enough to duplicate work heavily
- but not coordinated enough to improve root search quality much
- and they do not share the most important state in the way practical SMP usually does

---

## Specific issues in the current design

### 1. There is no shared TT
Files:
- `src/main.rs`
- `src/search.rs`

Observed:
- `worker_clone()` creates a fresh per-thread `TransTable`
- hash is divided across workers with `per_thread_hash`
- workers do **not** share a common TT

Why this is a problem:
- one of the biggest practical benefits of SMP is TT sharing
- without shared TT, workers duplicate search effort much more often
- root splitting without shared TT is much less attractive than it looks on paper

Practical effect:
- more total nodes, but many are redundant
- weaker synergy across workers
- lower effective speedup and often weak Elo payoff

This is probably the single most important reason the current SMP attempt did not impress.

---

### 2. Root workers do not cooperate via alpha updates
Files:
- `src/search.rs`

Observed:
- `search_root_parallel()` assigns root moves statically to workers
- `search_root_worker()` searches each assigned move with the same initial `alpha`/`beta`
- workers do not get tighter bounds as another worker finds a better move

Why this is a problem:
- at root, a lot of pruning value comes from improved alpha as soon as a strong move is found
- static root-split with no bound sharing means each worker may search too much
- this especially hurts compared with strong single-thread PVS root behavior

Practical effect:
- duplicated or unnecessary effort
- weaker pruning than the single-thread root loop
- more nodes without proportional strength gain

---

### 3. TT results are not merged back into the main engine TT in a useful way
Files:
- `src/search.rs`

Observed:
- after parallel root search, the code stores the final root result only into `worker_engines[0].tt`
- that is not the main engine TT
- worker-local TT contents are effectively discarded after the iteration, aside from worker reuse inside `worker_engines`

Why this is a problem:
- the main engine’s TT is not benefiting from parallel root search in the normal way
- iterative deepening usually wants the TT to guide later iterations and future searches
- local worker TT data remains siloed

Practical effect:
- poorer reuse of search information
- weaker move ordering carryover than expected

---

### 4. Workers are too independent for real SMP, but still pay thread overhead
Observed:
- each worker has local TT + local heuristics + local search state
- there is no shared stop flag beyond each worker’s own deadline polling
- no shared best-so-far / aspiration coordination

Why this is a problem:
- this is close to running multiple small engines in parallel on disjoint root moves
- that is simple, but often not good enough to beat a tuned single-thread root loop

Practical effect:
- overhead increases
- quality of global root search coordination decreases
- gains can be flat or negative

---

### 5. Static root assignment is brittle
Files:
- `src/search.rs`

Observed:
- root moves are distributed round-robin once
- there is no dynamic load balancing

Why this is a problem:
- root move costs are highly uneven
- one worker may get the expensive moves while another finishes early
- the slower worker determines completion time

Practical effect:
- poor core utilization
- noisy scaling
- especially bad in tactical or highly asymmetric root positions

---

### 6. Threading may be harming root search quality compared with the scalar path
Observed:
- single-thread root search uses ordered/PVS-like probing through moves
- root-split search loses some of that sequential information flow

Why this is a problem:
- alpha-beta gains strength from searching the best moves first
- a static parallel root split can search many moves as if they were all early candidates

Practical effect:
- even if node count rises, search quality per node can drop
- Elo can stay flat or worsen

---

### 7. M3 MacBook Air is a weak platform for proving SMP payoff
Even if the code were decent, the M3 Air is not the ideal validation box:
- fanless
- thermal throttling risk
- limited sustained multicore scaling compared with desktop CPUs

So if the current implementation was mainly judged there, it may look worse than it really is.

That said, the architectural issues above are still real and should be fixed regardless.

---

## Recommendation: do not make this design more complicated
The current root-split idea is already enough to show the main issue.

The next step should **not** be:
- deeper split-point parallel search
- more shared mutable heuristic state
- more synchronization around alpha/history/killers
- trying to salvage this exact design with extra complexity

Instead, the best recovery path is to **simplify and re-center the SMP design**.

---

## Best recovery path: move toward Lazy SMP

## Why Lazy SMP is a better fit here
For this engine, a first practical SMP version should aim for:
- shared TT
- shared stop/deadline signal
- thread-local heuristics
- minimal synchronization
- same root position searched by all workers
- slight move-order diversity between workers

This usually works better than naïve root split because:
- the shared TT becomes useful immediately
- workers help each other indirectly through TT entries
- there is less fragile partitioning logic
- implementation is simpler than full tree splitting

---

## Proposed SMP v2 design

### Shared state
Shared across workers:
- TT
- stop flag / time-up signal
- perhaps a best root move / best score report structure

### Thread-local state
Per worker:
- `Search`
- killers
- history
- counter-moves
- continuation history
- local node count
- local move buffers

### Worker behavior
Each worker:
- searches the same root position
- uses same depth target
- uses shared TT
- may vary root ordering slightly or vary aspiration/order seed
- stops when shared stop flag is raised or deadline expires

### Main thread behavior
Main thread:
- launches workers
- collects best completed result
- prints best available completed iteration
- keeps `Threads=1` path unchanged

---

## If keeping root split temporarily, minimum fixes to make it less bad
If the current agent wants to keep the root-split implementation alive for one more iteration, the minimum fixes should be:

### 1. Share TT across workers
This is the biggest one.
Without this, root split is handicapped from the start.

### 2. Merge/use root alpha information across workers if possible
Even a coarse shared best score could help prune some wasted work.
This must be done carefully to avoid too much synchronization overhead.

### 3. Use dynamic work distribution instead of static round-robin assignment
A simple work queue for root moves would help load balance.

### 4. Preserve single-thread root search as the reference path
Do not try to make threaded root search replace the scalar path unconditionally.

### 5. Measure on the Ryzen desktop, not just the M3 Air
The desktop box is the right place to judge whether SMP is viable.

---

## Practical do / don’t list

### Do
- keep `Threads=1` pristine
- treat SMP as experimental until proven in gauntlets
- prioritize shared TT over clever partitioning
- keep history/killers/continuation thread-local
- test on the Ryzen machine for real scaling
- compare `Threads=2` vs `Threads=1` under the same movetime and openings

### Don’t
- don’t share history/killers/continuation at first
- don’t add coarse TT mutexes everywhere
- don’t build deeper split-point tree parallelism yet
- don’t assume higher NPS means stronger play
- don’t judge SMP quality only from the MacBook Air

---

## Recommended measurement plan

### Correctness
1. `cargo check`
2. `perfttest`
3. `selftest 100`
4. UCI smoke with `Threads=1`
5. UCI smoke with `Threads=2`

### Performance
1. bench on M3 Air
2. bench on Ryzen desktop
3. compare depth reached at fixed movetime
4. compare total nodes and time behavior

### Strength
1. short A/B: `Threads=2` vs `Threads=1`
2. identical openings file
3. identical hash/book settings
4. enough games to filter noise before keeping SMP enabled by default

---

## Bottom line
The current threading attempt did not fail because multithreading is a bad idea in general.
It likely underperformed because the current design is:
- **root-split**
- **non-shared-TT**
- **weakly coordinated**
- and therefore missing the main strengths of practical SMP

## Suggested next step
Best path forward:
1. keep this root-split version experimental
2. do not add more complexity to it
3. move toward a **shared-TT Lazy SMP** design
4. keep heuristics thread-local
5. validate primarily on the Ryzen desktop

If Lazy SMP is too large a step immediately, then at minimum:
- share TT
- improve load balancing
- and avoid over-interpreting current root-split results
