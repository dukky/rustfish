# Hermes note — 2026-11-06

Non-disruptive note for the next agent. No queue actions, no main changes.
Written after reading E1, the eval-scale work and the two-lane runner change.

## Local verification from the VPS (x86 AVX2)

These are scratch-copy builds of the named branches, not the queue's test.

| branch | tests | bench | evalhash | local NPS |
|---|---|---|---|---|
| main (`91d0311`/`ad53daa` era) | 40 pass | 610670 | 18049521fa7dc89a | 957k |
| `nnue-speed` | 40 pass | 610670 | 18049521fa7dc89a | **1.104M (+15.4%)** |
| `nnue-scale-58` | 40 pass | 531213 | 114786dfd486d474 | 1.024M |
| `nnue-e2` (512) | 40 pass | 384434 | 96e63799a1da4b0a | 1.071M |
| `tm-ccrl` | 40 pass | 610670 | 18049521fa7dc89a | 1.008M |

`nnue-speed`'s +15.4% on this box says the lazy-accumulator/memoised-eval gain
is not Mac-only; the search signature is unchanged. `tm-ccrl` also leaves the
search signature unchanged, so its 5+0.05 pre-gate is mostly a time-management
test, not a strength proxy.

A 200-game 5+0.05 pre-gate of `nnue-scale-58` vs main was running at the time
of writing; at 60 games it was ~53% candidate, which is noise, not evidence.

## Two-lane runner observation

The 16/4 split is directionally sensible — it stops datagen and tests
serialising — but it is not automatically more efficient:

- A 16-shard gate is **20/16 = 25% slower** than a 20-shard gate.
- A 4-shard datagen is **5× slower** than a 20-shard run.
- It only wins when the 4-runner background lane is continuously fed.
- At 15:20 UTC seed 14 had completed, no replacement datagen was running,
  and the ladder had only ~10 jobs in progress. Several runner slots were idle
  while the queue waited for the ladder.

Suggested: allocate dynamically. If a datagen job is pending, run 16 test +
4 datagen; if not, give the gate all 20. Measure generations/hour, not runner
occupancy. For a single ladder, 20 shards is simply faster.

## CI-lane guard suggestions

The current `ltc-gainer` is correctly at **16 shards**, so the 4-runner
background lane can run datagen concurrently (16 + 4 = 20). That should be
enforced, not left to each mode's config:

- Add a validation in `scripts/queue.py` (or a small CI check) that rejects any
  foreground mode with `shards > 16`. A mode silently set to 20 starves the
  background lane, which is what I initially assumed had happened.
- Keep `datagen.yml` `max_parallel <= 4` when the foreground lane is active;
  the background lane is only useful if the foreground cap really is 16.
- Update the dashboard's `MAX_PARALLEL`/runner estimate if the caps change; it
  is currently showing the correct 16 + 4.
- For docs-only branches, `check_bench.py` currently fails if their `Bench:`
  trailer is stale (my two proposal branches did). Either auto-rewrite the
  trailer from `main-bench.json` for docs-only commits, or add a `--docs-only`
  path that compares against main's measured bench. The queue already fixes
  stale trailers when it merges; CI on standalone docs branches does not.

## Free wins vs. the things that actually close a top-10 gap

Plan.md already has SPSA (`spsa-nnue`) and Syzygy listed, but it does not rank
the architecture pieces. My ranking, for what it is worth:

**No net training required (do these in parallel with training):**

1. **SPSA of the NNUE-era search margins** (`tune/spsa-nnue.json`,
   `spsa-nnue`). Highest-confidence non-net gain: the eval scale was 2x off
   for a day, so the pruning margins are still calibrated on HCE. Expect
   roughly +10..30.
2. **Correction history** — see `docs/correction-history-plan`
   (branch `docs/correction-history-plan`). Pawn correction only first.
   Cheap, standard in modern NNUE engines, expected +5..15.
3. **Time management at 120+1 / 40/15**: `tm-ccrl` is testing exactly this.
4. **AVX2 (not AVX-512) speed work.** The CCRL reference box is an i7-4770K
   (Haswell, AVX2 only), so AVX-512 kernels are dead code there.

**One-retrain architecture wins, ordered by expected value:**

5. **King-relative inputs / king buckets.** The single biggest standard
   upgrade over absolute piece-square features; one clean retrain on the full
   pool, validation loss first.
6. **Threat features** (attacked pieces, pawn attacks, hanging pieces). Also
   one retrain; proven in modern engines.
7. **Output buckets on the 512 net.** E3's +4.9 was on 256 and inconclusive;
   cheap to retest at the width that actually merged.
8. **Material/phase input buckets** — capacity without changing feature
   semantics.

**Do not spend weeks on:** more width-only runs (E2 gave +26, not a top-10
jump), more output buckets on 256 (E3 +4.9, inconclusive), or a ladder of
7M-position nets — that is how the nine v1 nets got rejected.

For top 10 specifically: king-relative/threat inputs are the architecture
pieces that matter. Width alone will not get there.

## Critique: is the plan's weight on net training right?

The plan's centre of gravity is "train the net properly, generation after
generation". That was right for the v1 collapse (nine rejected nets) and E1
proved the point: same shape, proper pool, +152. But the plan should not
assume that marginal training keeps returning +100.

The evidence so far, on this net shape:

| change | gate result |
|---|---|
| E1: full pool, 6 epochs, same 256 shape | **+151.8** |
| E2: 512 wide, same data/recipe | **+26.1** |
| E3: 8 output buckets on 256 | +4.9, inconclusive |
| Eval-scale 1/2 vs 3/8 / 5/8 | **+123** (1/2 best; 3/8 -9.3, 5/8 -22.2) |

E1 and the scale fix are **one-time corrections**, not a rate. E2 and E3 are
what a marginal generation looks like: **+26 and +5**, not +150. If g2a/g2b
land at +25..40, the data is telling us the 768->H->1 shape is approaching its
ceiling for this feature representation, and more generations of the *same
features* will decay towards zero.

That does not mean stop training. It means:

1. **Do not budget the plan around "N generations x +30".** Budget around a
   measured marginal rate that is falling. The dashboard projection models
   decay 0.85 deliberately.
2. **Spend the next training runs on a feature change, not on more width or
   more buckets of the same 768 absolute inputs.** King-relative inputs /
   king buckets and threat features are the two that change what the net can
   represent. That is a real architectural gain, not a scaling one.
3. **Run the free wins in parallel**, not after the net chain plateaus: SPSA
   of the NNUE-era margins, correction history, time management. Those cost
   no training and are independent of which net is embedded.
4. **Keep a control.** Every generation should be usable as a control against
   the previous one on the same validation set, or the +25s are not
   distinguishable from data noise.

The honest framing for the plan: the net work has taken us from ~3200 to
~3400. Getting to ~3500 is plausibly more of the same plus king/threat
features; getting to 3750 (top 10) is very unlikely from more generations of
this representation, and the plan should say so rather than imply that
scaling training time is the path.

## Branches from this session, for review

- `docs/neural-guided-alpha-beta` — policy-head proposal (novel path; a
  research bet, not a free win).
- `docs/correction-history-plan` — pawn correction plan above.
- `spike-book-off-default` — `OwnBook=false` by default (submission hygiene,
  bench-identical).
- `dashboard-scenarios` — saturating projection chart + table (illustrative).
- `notes/hermes-2026-11-06` — this note.

## Other things that may matter

- E1's gain came from the whole valid pool (350M) and 6 epochs with cosine LR,
  not from architecture. Do not regress to 2-epoch single-generation nets as a
  "quick test"; that is what produced the nine rejected v1 nets.
- The fixed validation set and one-variable protocol are the right gate before
  CI. The local 200-game matches are only smoke tests.
- The eval-scale search is real: 1/2 beat 3/8 by 9.31 Elo at stc, so 5/8 is a
  reasonable bracket, not a formality.

— Hermes (DeepSeek), 2026-11-06
