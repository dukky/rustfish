# Parallelism / SIMD / GPU Notes

## Executive summary
For this repo, the best upside among parallel hardware options is:

1. **Multithreading / SMP search**
2. **Possibly SIMD later, after structural cleanup**
3. **Not GPU search**
4. **Not AMX**

This engine is a classical alpha-beta chess engine with branch-heavy move generation, pruning, make/undo, and irregular tree search. That profile generally benefits much more from **CPU-side search parallelism** than from GPU acceleration, and more from **data-structure improvements** than from early SIMD micro-optimization.

---

## 1. Multithreading: strongest practical option

### Recommendation
If the goal is a meaningful performance jump without changing the engine style, the highest-value hardware feature to add is:
- **multi-threaded search**

The most practical first design is:
- **Lazy SMP** or a simple **root-split SMP**

### Why multithreading is promising here
The engine already has the main ingredients needed for a first SMP implementation:
- iterative deepening
- TT + Zobrist
- stable root search flow
- UCI option framework
- time/deadline logic

That means parallel search is an extension of the current architecture, not a full restart.

### Best first design: Lazy SMP
A good initial approach:
- spawn `N` worker threads
- each searches the same root position
- workers share:
  - transposition table
  - stop flag / deadline state
- workers keep local:
  - killers
  - node count
  - history tables
  - continuation history
  - counters / temporary buffers
- accept the best finished root result from the main thread or strongest completed iteration

Lazy SMP is attractive because:
- it is much simpler than deep split-point parallel search
- it usually gives decent practical scaling
- it avoids a lot of synchronization complexity
- it works well with shared TT

### Alternative first design: root splitting
Another simple option:
- split root moves across threads
- each thread searches assigned root children
- merge results at root

This is easier to reason about than full tree parallelism, but often less robust than Lazy SMP in irregular positions.

### Shared vs thread-local state
A sensible first split would be:

#### Shared
- TT
- stop flag
- deadline / time control signal
- root position input
- possibly best root move / best score reporting

#### Thread-local
- `Search`
- killers
- history
- counter-moves
- continuation history
- per-thread node count
- local move buffers / staging buffers

That is the safest way to avoid lock contention and correctness hazards early.

### TT synchronization strategy
The TT is the main shared structure worth keeping global.

Practical options:
- lock-free-ish overwrite with atomics / relaxed race tolerance
- striped locks
- coarse mutex around table access

For chess engines, the ideal is usually a race-tolerant shared TT. A coarse mutex would likely kill most of the benefit.

If implementation simplicity matters first, a minimal path is:
- preserve the single-thread TT design conceptually
- migrate TT entries / access to a thread-safe shared backing structure
- tolerate benign overwrite races if correctness is unaffected

### UCI option to add
Recommended new option:
- `Threads`

Behavior:
- default `Threads=1`
- only enable SMP when `Threads > 1`
- keep single-thread search path intact as the baseline

That will make testing and regression isolation much easier.

### Validation plan for SMP
Because another agent may be changing the engine, SMP should be validated incrementally:

1. `Threads=1` must preserve behavior
2. `cargo check`
3. `perfttest`
4. `selftest 100+`
5. UCI smoke with `Threads=1`
6. UCI smoke with `Threads=2`
7. benchmark on both machines
8. short A/B gauntlet versus single-thread baseline

### What to expect on your hardware

#### M3 MacBook Air
- good single-thread efficiency
- fanless, so sustained multi-core scaling may flatten from thermals
- still useful for development and sanity testing

#### Ryzen 9600X PC
- better platform for:
  - scaling tests
  - longer SMP searches
  - gauntlets
  - sustained benchmarks

If SMP is added, the Ryzen box is likely the best place to measure real gains.

---

## 2. SIMD: possible, but not the first thing to do

### Recommendation
SIMD is possible and may help eventually, but it should not be a top priority for the current codebase.

### Why SIMD is not the first lever here
Current hotspots appear to be dominated by:
- branch-heavy move generation
- make/undo filtering
- SEE cost
- move sorting / allocation churn
- full recomputed evaluation

Those are not the kinds of bottlenecks where dropping intrinsics into one or two helper functions usually gives large gains.

### Architectures available
On your machines:
- **Apple Silicon M3**: NEON on `aarch64`
- **Ryzen 9600X**: SSE2 and likely AVX2 on `x86_64`

In Rust, this would typically mean:
- `std::arch::aarch64` for NEON
- `std::arch::x86_64` for SSE/AVX intrinsics
- runtime feature detection where needed on x86

### Where SIMD might help later
Potential candidates:
- evaluation term accumulation
- PST accumulation
- maybe mobility/attack aggregation if data layout changes
- maybe piece-list scans if converted into denser arrays

### Where SIMD is unlikely to help much right now
- legal move generation
- branching search logic
- make/undo
- current SEE implementation
- pruning decisions

### Best precondition before SIMD work
SIMD becomes more interesting if the engine moves toward:
- bitboards
- denser piece lists
- data-oriented eval
- reduced branchiness in core hot loops

Until then, structural scalar improvements are probably higher ROI.

### Suggested SIMD strategy if pursued later
If SIMD work is attempted later:
- keep a scalar reference path
- add arch-specific fast paths behind `cfg`
- do not let correctness depend on SIMD-only code
- measure on both Apple Silicon and x86_64

---

## 3. AMX: not worth targeting

### Recommendation
Do not target AMX for this engine.

### Why
AMX-style hardware is most compelling for:
- matrix-heavy workloads
- neural inference
- dense linear algebra

This engine is not doing that. It is a classical alpha-beta engine with hand-written eval.

So AMX would be:
- hard to access cleanly
- platform-specific
- poor portability fit
- low practical payoff

If the engine ever evolves toward NNUE or another neural eval, that recommendation could change. For the current repo, AMX should be ignored.

---

## 4. GPU: technically possible, practically the wrong fit

### Recommendation
Do not try to offload the current search to GPU.

### Apple GPU on M3
The machine’s GPU can be accessed through Metal, so yes, in principle the hardware is available.

### Why GPU search is a bad fit here
Classical alpha-beta search is poorly matched to GPU execution:
- highly branchy
- irregular workload sizes
- deep sequential dependencies
- make/undo-heavy
- hard to batch efficiently
- synchronization / dispatch overhead dominates small tasks

GPUs are much better for:
- dense matrix math
- batched neural inference
- regular large parallel workloads

### What GPU *could* be useful for in chess generally
Possible future uses in a different engine architecture:
- NNUE / neural eval inference
- large batched analysis workloads
- training or self-play pipelines

But for this repo’s current design, GPU acceleration would likely be:
- high effort
- low speedup
- high portability cost
- likely slower overall in practice

So GPU work should not be a near-term priority.

---

## 5. Practical implementation guidance for the other agent

### Recommended priority order
1. **Single-thread structural speedups first**
   - cheaper SEE
   - less move sorting / allocation churn
   - reduced legal move overhead
   - reduced eval rescanning

2. **Then add multithreading**
   - `Threads` UCI option
   - Lazy SMP or root-split SMP
   - shared TT
   - thread-local heuristics

3. **Consider bitboard/backend improvements**
   - this could help both speed and later SIMD viability

4. **Only then revisit SIMD**

5. **Skip GPU and AMX for now**

### Concrete first SMP milestone
A good first milestone would be:
- add `Threads` option
- preserve existing search when `Threads=1`
- implement shared-TT multi-worker root search when `Threads>1`
- keep history/killers/continuation local per worker
- compare:
  - NPS scaling
  - depth reached at fixed movetime
  - short gauntlet vs `Threads=1`

### What success should look like
A good first SMP patch should aim for:
- no correctness regressions
- acceptable scaling on the Ryzen box
- neutral or positive Elo in short A/B
- minimal disturbance to the single-thread path

---

## 6. Bottom line
For this repo:
- **Multithreading** is the best hardware acceleration opportunity
- **SIMD** is secondary and probably premature right now
- **AMX** is irrelevant for the current engine style
- **GPU search** is not a good fit

If the goal is practical strength growth, the best path is:
- improve the scalar hot path first
- then add CPU multithreading
- only consider lower-level ISA-specific optimization once the engine is more data-oriented
