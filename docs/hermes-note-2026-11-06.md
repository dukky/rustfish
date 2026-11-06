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

## Other things that may matter

- E1's gain came from the whole valid pool (350M) and 6 epochs with cosine LR,
  not from architecture. Do not regress to 2-epoch single-generation nets as a
  "quick test"; that is what produced the nine rejected v1 nets.
- The fixed validation set and one-variable protocol are the right gate before
  CI. The local 200-game matches are only smoke tests.
- The eval-scale search is real: 1/2 beat 3/8 by 9.31 Elo at stc, so 5/8 is a
  reasonable bracket, not a formality.

— Hermes (DeepSeek), 2026-11-06
