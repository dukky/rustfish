# Development workflow

How to change rustfish so that every change is measured, and later changes
can't silently erase earlier gains. The laptop does correctness and quick
sanity; GitHub Actions (public repo, free) does every strength measurement.

## The rules

1. **`main` is the champion.** Every commit on `main` that changes search or
   eval has passed an SPRT against the `main` it was merged onto. Gains
   compound because each candidate must beat the *latest* champion, so a patch
   that undoes an earlier gain fails its own test.
2. **Engine changes go through a branch and an SPRT.** No change is accepted
   on a local gauntlet, a hunch, or a short match.
3. **The bench count is the signature of search behavior.** `bench` searches 13
   fixed positions at fixed depth. Same count means the engine plays identically;
   a different count means behavior changed and the commit needs a
   `Bench: <nodes>` trailer. CI enforces this on every push.
4. **Every result is logged**, pass or fail, in `tests/sprt-log.jsonl` and
   `tests/ladder-log.jsonl`, so nothing is retried by accident and nothing is
   forgotten.

## What each kind of change needs

| Change | Bench | Gate | Bounds |
|---|---|---|---|
| Refactor, cleanup, tooling, docs | must be unchanged | CI | none |
| Pure speedup (same nodes, faster) | must be unchanged, and `evalhash` must match `main`'s | CI + interleaved local bench NPS (or CPU cycles via `/usr/bin/time -l`) | none; anything that changes the eval goes through an SPRT instead |
| New eval or search idea, parameter change | new `Bench:` trailer | STC SPRT | `[0, 10]` (default; `--elo1 5` later) |
| Search changes, in batches | — | LTC non-regression of `main` vs `main` ~5 merges back | `[-5, 0]` with `--ltc --base <sha>` |
| Simplification (removes code/terms) | new `Bench:` trailer | STC non-regression | `[-5, 0]` via `--nonreg` |
| Time management change | usually unchanged | LTC gainer (STC barely exercises it) | `[0, 5]` with `--ltc --gainer` |

Compute is the bottleneck, so three choices keep tests fast without raising
the false-accept rate (still 5%): waves start at 512 STC games and double up
to 2,048, so clear results stop early; STC bounds are `[0, 10]` while big gains
are still available (borderline tests ~4x shorter, but real +2..5 patches
often fail — return to `[0, 5]` when typical gains shrink); and LTC is run on
batches of merged search changes rather than every patch.

The LTC step for search changes is a non-regression check: the STC gainer
test already established the gain, and LTC only has to show it doesn't turn
into a loss at longer time controls. A real gain clears `[-5, 0]` in a fraction
of the games a second `[0, 5]` test would need; compute is the bottleneck.

STC is 10+0.1 and LTC is 60+0.6, both one thread, with UHO openings and paired
colors. Bounds are logistic Elo; alpha = beta = 0.05.

## The loop

### 1. Branch and build (laptop)
```bash
git switch -c <idea-name> main
# ... edit ...
cargo build --release
```

### 2. Correctness (laptop, about a minute)
```bash
cargo test --release
printf 'perfttest\nselftest 300\nquit\n' | ./target/release/rustfish
python3 scripts/check_bench.py
```
`check_bench.py` compares the bench count with the latest `Bench:` trailer:
- **Matches**: no behavior change. Fine for refactors and speedups; for an
  intended functional change it means the change has no effect at bench depth,
  so check it actually does what you meant.
- **Differs**: behavior changed. That's expected for a functional change; put
  the printed count in the commit message as `Bench: <nodes>`.

For speedups, compare NPS against a `main` build with interleaved runs (the
fanless M3 drifts), never one before/after pair:
```bash
git worktree add ../rf-main main && cargo build --release --manifest-path ../rf-main/Cargo.toml
for i in 1 2 3; do
  printf 'bench\nquit\n' | ../rf-main/target/release/rustfish | grep Nodes/second
  printf 'bench\nquit\n' | ./target/release/rustfish | grep Nodes/second
done
```

### 3. Optional disaster check (laptop, a few minutes)
Only to catch crashes, illegal moves or an obviously broken idea before
spending CI time. It is **not** evidence of strength (24 games is about +-140 Elo):
```bash
python3 scripts/gauntlet.py --engine-a ./target/release/rustfish --engine-b ../rf-main/target/release/rustfish \
  --engine-a-name candidate --engine-b-name main --engine-a-option OwnBook=false --engine-b-option OwnBook=false \
  --games 24 --movetime 50 --parallel 4 --openings-file ./scripts/openings.txt
```

### 4. Commit, push, test (CI)
```bash
git commit          # include "Bench: <nodes>" if behavior changed
git push -u origin <idea-name>
python3 scripts/ci.py sprt <idea-name>            # STC [0, 5]; follows every wave
python3 scripts/ci.py sprt <idea-name> --ltc      # after an STC pass, for search changes: LTC [-5, 0]
python3 scripts/ci.py sprt <idea-name> --nonreg   # simplifications
```
- Pushing the branch also runs CI (build, tests, perft, selftest, bench).
- `--no-wait` dispatches and exits; `--resume <test_id>` re-attaches later.
- Each wave plays 2,048 STC games in about 15 minutes. Clear winners or
  losers usually stop in 1-2 waves; a real +3..+5 patch can take 5-10.
  `inconclusive` after 10 waves (about 20k games) means the effect is too
  small to matter: treat it as a fail.
- Results appear in the run's job summary and in `tests/sprt-log.jsonl`.

### 5. Merge or drop
- **PASSED** (and passed LTC if required): rebase on `main` if it moved. If
  `main` gained a functional change meanwhile, rerun the STC test against the
  new `main`; then fast-forward `main`, push, and delete the branch.
- **FAILED / inconclusive**: don't merge. Keep the branch if a variant is
  worth trying, otherwise delete it. The log keeps the result.
- Commit the updated `tests/*.jsonl` to `main` along with the merge, or in a
  periodic "Record test results" commit.

## Merge queue (unattended)

`.github/workflows/queue.yml` + `scripts/queue.py` run candidates through the
whole loop on GitHub with no laptop or LLM in the loop: rebase onto `main`,
build/test/perft/selftest/bench (refreshing the `Bench:` trailer), push, run
the SPRT, and fast-forward `main` on a pass. One candidate at a time, always
against the current `main`, so a stale base can't happen.

- State lives on the `queue-state` branch: `queue.json` (`pending` list of
  `{branch, mode}`, plus the in-flight `current`) and `log.jsonl` (every
  outcome: merged / rejected / needs-human / skipped).
- Modes: `stc` (gainer `[0, 10]`), `nonreg` (`[-5, 0]`), `ltc-gainer`
  (`[0, 10]` at 60+0.6), `speedup` (no SPRT; merges only if bench nodes equal
  `main`'s and the `evalhash` command gives the same hash as a `main` build:
  bench alone can't see an eval change that happens not to alter the bench
  tree). `evalhash` prints the HCE `Eval hash` and, when a net is active (the
  default), an `NNUE eval hash` computed through the incremental accumulator
  (make/undo along each walk), so it also catches NNUE update/SIMD changes.
- To add work: push the branch, then
  `python3 scripts/queue.py add <branch>:<mode>[,<branch>:<mode>]` (`--front`
  to jump the queue; `--expect <elo>` to replace the base-rate gain in the
  priority score, e.g. for a net with a big validation-loss lead; `--preempt`
  additionally cancels the
  in-flight SPRT, which goes back to the queue right behind the new item and
  is retested from scratch: use it for net generations on the critical path,
  never for ladders or LTC checks, which it does not interrupt). Re-adding a
  waiting branch moves it rather than queueing it twice. This drops a request file in
  `queue-state`'s `inbox/`, which the next queue run consumes. Don't pass work
  as workflow inputs or edit `queue.json` by hand: GitHub cancels all but one
  pending run per concurrency group, so inputs can silently vanish.
- **Speedups don't wait.** Every queue run merges all pending `speedup` items
  at once, even while an SPRT or ladder is in flight (they use no runners).
  Their commits are recorded in `queue.json`'s `identical_merges`.
- **Priority, not FIFO.** The next SPRT is the pending item with the highest
  expected Elo per runner-hour: `p_accept * gain / runner_hours`, with
  `runner_hours = shards * (p * hours_if_accepted + (1 - p) * hours_if_rejected)`.
  The per-mode base rates come from `log.jsonl` (accepted Elo clipped at 30 so
  bug fixes don't dominate; every rate is shrunk towards a prior with 3
  pseudo-samples, so a mode with few results gets sane defaults). Ties go to
  the older item. Overrides, in order: `--front` items (and requeued, retried
  or preempted ones) run first in queue order; a `ladder` or `ltc-check` keeps
  its place, so nothing queued behind it jumps ahead; an item jumped over 3
  times goes next, so a costly ltc-gainer can't starve behind cheap STCs.
  The `dispatched` log row carries a `priority` field with the score and its
  inputs. `python3 scripts/queue.py rates` prints the base rates and the pick
  order for the live queue without changing anything.
- A schedule runs the queue every 30 minutes as a heartbeat: it picks up inbox
  requests, and if the SPRT callback was lost it finds the current test's final
  result itself. `datagen.yml` also calls the queue back when a run ends, so
  the background lane starts its next seed at once.
- If `main` moved during a test only outside the engine sources (`src/`,
  `net/nnue.bin`, `Cargo.*`, `build.rs`) or by recorded speedup merges, a
  passing candidate is replayed onto `main` without a retest. After speedups
  the replayed tree differs from the tested one, so it is rebuilt, perft and
  selftest are rerun, and its bench must equal the tested candidate's;
  otherwise (or after any other engine change, or a replay conflict) it is
  requeued at the front.
- `scripts/queue-settings.json` is validated on load: every SPRT mode must use
  at most 16 shards (the 16 foreground runners; 3 are the background lane and
  1 is kept for the queue's own job).
- Tests: `python3 scripts/test_queue.py` (stdlib unittest; real git in a temp
  dir, cargo/gh/engine mocked).
- `needs-human` entries (rebase conflict, failed checks) are skipped; fix the
  branch and re-add it. Infrastructure failures (an SPRT run that ends with no
  report, a queue error) are retried once first.
- Every 5 merges the queue puts an `ltc-check` at the front: LTC `[-5, 0]` of
  `main` vs the `main` from 5 merges back. If it fails the queue pauses
  (`regression-alert` in the log) until `python3 scripts/queue.py unpause`.
- Every 5 merges (`ladder_every` in `scripts/queue-settings.json`) it also runs
  a CCRL ladder on `main` (elite anchors, 120+1, ~2 h). The ladder workflow
  appends the fitted rating to `tests/ladder-log.jsonl` on `main` itself, so the
  dashboard chart updates with no one involved. Run one sooner with
  `python3 scripts/queue.py add main:ladder --front`.
- Bot merges don't trigger `ci.yml`; the queue runs the same checks itself.

## Periodic checks (protection against slow erosion)

SPRTs at STC catch most regressions, but some things only show up at longer
time controls or across many patches. Every 5 accepted functional patches, or
monthly, whichever comes first:

1. **CCRL ladder** on `main` (about 1.5-2 hours on GitHub):
   ```bash
   python3 scripts/ci.py ladder
   ```
   Plot `tests/ladder-log.jsonl`. A flat or falling trend while SPRTs keep
   passing means STC gains aren't carrying over to 120+1. Investigate before
   adding more patches.
2. **Progression check**: current `main` vs the champion from 5 patches ago at
   LTC. Read the Elo, not the verdict:
   ```bash
   python3 scripts/ci.py sprt main --base <older-sha> --ltc --max-waves 2
   ```
3. **Anchor health**: the ladder summary flags anchors scoring outside
   15-85% and says when fewer than half are within +-150 of the estimate.
   Replace flagged anchors in `scripts/anchors.json` with engines from the
   [CCRL Blitz complete list](https://computerchess.org.uk/ccrl/404/rating_list_all.html)
   in the suggested range (open source, UCI, buildable on Linux; check
   `kind`/`pkg` against the repo). Run `ci.py ladder --games 4 --tc 10+0.1` once
   to confirm new anchors build and answer `go`.
4. **Re-tune** eval weights after any large search change, since tuned values
   drift as search changes.

## Tuning search parameters (SPSA)

The search margins and limits (`FutilityMarginD1`, `ReverseFutilityMarginD*`,
`NullReduction*`, `LmpLimitD*`, `Lmr*`) are UCI spin options, so
they can be tuned by SPSA on GitHub without rebuilding per point. The state
(theta, bounds, `c_end`, `r_end`, iteration counter, schedule length) lives in
`tune/spsa-params.json`; `scripts/spsa.py` implements fishtest-style SPSA
(alpha 0.602, gamma 0.101, A = 10% of `total_iterations`).

```bash
gh workflow run spsa.yml -f ref=main -f params=tune/spsa-params.json \
  -f settings='{"tc": "8+0.08", "hash": 16, "shards": 16, "iterations": 50, "pairs": 8, "rounds": 4, "combine": "mean"}'
gh run watch    # tuned values land in the merge job's summary + spsa-result artifact
```
- One iteration = `pairs` game pairs of theta+c_k*delta vs theta-c_k*delta.
  Each shard builds rustfish once and runs `iterations` iterations.
- **Parallel approximation:** shards can't share theta mid-run, so each runs
  an independent chain from the same theta with its own seed; the merge job
  takes the iteration-weighted mean of their theta deltas (`combine: mean`:
  averaged gradient, same step size; safe) or their sum (`combine: sum`:
  about `shards` times faster, but overshoots unless rounds are short). With
  `rounds > 1` the merge job dispatches the next round from the merged theta,
  so keep `iterations` per round small and use more rounds rather than one
  long round.
- `total_iterations` counts merged iterations: `mean` adds `iterations` per
  round, `sum` adds `shards * iterations`. Size it to the total you plan to
  run so the gains decay to `c_end` / `r_end` at the end.
- To continue a tune later, commit the `spsa-result.json` artifact over
  `tune/spsa-params.json` (it carries the iteration counter) and dispatch again.
- Sanity-check the maths offline: `python3 scripts/spsa.py toy` runs the same
  shard/merge scheme on a fake quadratic Elo function.

**Applying results:** round the tuned values (`python3 scripts/spsa.py show
spsa-result.json`), put them in `SearchParams::default()` *and* the matching
`option name ... default` lines of the `uci` handler in `src/main.rs`, commit
with the new `Bench:` trailer, and run a normal STC SPRT (`ci.py sprt`, `[0, 5]`)
plus the LTC non-regression check. A tune is just a candidate: it is merged
only if it passes the SPRT like any other change.

## Things to retest
Everything in `docs/archive/post-2100-revised.md`'s "Rejected for now" list was tested
before commit 80862d0, when every timed search was silently capped at depth 8,
and with 24-96 game samples. Those rejections are void; retest each through
this pipeline before ruling it out again.

## Reference
| Tool | Purpose |
|---|---|
| `bench [depth]` (UCI command) | deterministic node count + NPS |
| `evalhash` (UCI command) | hash of the static eval (HCE line, plus an NNUE line when a net is active) over ~190k fixed positions; must match `main` for a speedup |
| `scripts/check_bench.py` | bench vs last `Bench:` trailer (CI + local) |
| `scripts/ci.py sprt/ladder/status` | dispatch and follow GitHub tests, log results |
| `scripts/sprt.py` | LLR / Elo from pentanomial counts (used by the workflow) |
| `.github/workflows/ci.yml` | correctness gate on every push |
| `.github/workflows/sprt.yml` | wave-based SPRT on GitHub runners |
| `.github/workflows/ccrl-ladder.yml` | CCRL-scale rating vs anchors |
| `.github/workflows/release.yml` | release binaries on tag `v*` (docs/ccrl-submission.md) |
| `.github/workflows/spsa.yml`, `scripts/spsa.py`, `tune/spsa-params.json` | SPSA tuning of search options |
| `.github/workflows/datagen.yml` | sharded self-play training data (docs/nnue.md) |
| `tests/sprt-log.jsonl`, `tests/ladder-log.jsonl` | the record of every result |

Capacity: GitHub runs about 20 jobs at once for this account. An STC wave uses
19 (plus one build job) and a ladder uses about 20, so run them one at a time.
SPRT binaries are built once per wave with `-C target-cpu=x86-64-v3` and shared
with the shards.
