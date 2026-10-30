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
| Pure speedup (same nodes, faster) | must be unchanged | CI + interleaved local bench NPS | optional `--nonreg` if large |
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

## Things to retest
Everything in `post-2100-revised.md`'s "Rejected for now" list was tested
before commit 80862d0, when every timed search was silently capped at depth 8,
and with 24-96 game samples. Those rejections are void; retest each through
this pipeline before ruling it out again.

## Reference
| Tool | Purpose |
|---|---|
| `bench [depth]` (UCI command) | deterministic node count + NPS |
| `scripts/check_bench.py` | bench vs last `Bench:` trailer (CI + local) |
| `scripts/ci.py sprt/ladder/status` | dispatch and follow GitHub tests, log results |
| `scripts/sprt.py` | LLR / Elo from pentanomial counts (used by the workflow) |
| `.github/workflows/ci.yml` | correctness gate on every push |
| `.github/workflows/sprt.yml` | wave-based SPRT on GitHub runners |
| `.github/workflows/ccrl-ladder.yml` | CCRL-scale rating vs anchors |
| `.github/workflows/release.yml` | release binaries on tag `v*` (docs/ccrl-submission.md) |
| `tests/sprt-log.jsonl`, `tests/ladder-log.jsonl` | the record of every result |

Capacity: GitHub runs about 20 jobs at once for this account. An STC wave uses
16 and a ladder uses about 20, so run them one at a time.
