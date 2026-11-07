# Handoff guide for a new agent

For an agent (any model) continuing this project. It narrows the job to work
that the pipeline checks automatically, so mistakes cost CI time, not
strength. Read this, then `plan.md`, then `docs/development.md`.

## How the project works, in one paragraph

`main` is the champion. Every engine change is a branch. The merge queue on
GitHub Actions (`scripts/queue.py`, `.github/workflows/queue.yml`) rebases each
queued branch onto `main`, runs build / tests / perft / selftest / bench, runs
an SPRT against `main`, and fast-forwards `main` only if it passes. Every 5
merges it checks `main` against `main` from 5 merges back at a longer time
control and pauses itself on a regression. **You never merge anything
yourself.** Live state is on the dashboard: https://rustfish-dash.dev.duk.im.

## Your loop

1. Check the queue: `git fetch origin queue-state && git show origin/queue-state:queue.json`
   (or the dashboard). Read the recent outcomes in `log.jsonl` on that branch.
2. If fewer than ~8 items are pending, implement the next 2-3 ideas from
   "Idea backlog" below (or from `plan.md` "Plan", mid term). One idea per branch.
3. For each idea:
   ```bash
   git fetch origin && git switch -c <idea-name> origin/main
   # edit, keeping the diff small and local
   cargo build --release
   cargo test --release
   printf 'perfttest\nselftest 300\nquit\n' | ./target/release/rustfish   # all pass, "selftest ok"
   python3 scripts/check_bench.py        # prints the new bench count if behaviour changed
   git commit                            # message: what + why; add "Bench: <nodes>" if it changed
   git push -u origin <idea-name>
   python3 scripts/queue.py add <idea-name>:stc
   ```
4. Handle `needs-human` entries in the queue log that you can: a "rebase
   conflict" means re-apply the same idea on current `origin/main` as a new
   branch `<name>-r2` (or `-r3`) and queue that. If you can't tell what the idea
   was or how it fits the new code, leave it and note it in `plan.md`.
5. Record what you did in `plan.md` ("Plan", short term): what you queued, what
   you dropped and why. Commit that to `main` (docs-only commits are fine to push
   directly; see "Never" for what isn't).

## Choosing the queue mode

| Change | Mode |
|---|---|
| New search/eval idea, parameter change | `stc` |
| Removing code or a term (simplification) | `nonreg` |
| Time management | `ltc-gainer` |
| Same-shape net (H, buckets, king layout unchanged) | `net-gate` |
| Net with a new shape or king layout | `stc` |
| Speedup that changes nothing (bench AND `evalhash` identical to `main`) | `speedup`, but see "Never" |

Bounds and wave sizes are in `scripts/queue-settings.json`; `--expect <elo>`
sets the priority estimate, `python3 scripts/queue.py remove <branch[:mode]>`
drops a waiting item. `ladder` and `ltc-check` items are queued by the queue
itself (every 5 merges) or by a human.

## Nets (only if you are asked to work on them)

Nets are trained on the laptop (Metal GPU trainer `tools/nnue-trainer-metal`,
or the CPU trainer `tools/nnue-trainer`, ~2x faster since 2026-11-07 and with
the factoriser) or in CI with `train-sweep.yml` (ablations scored on
validation loss). `scripts/netloop.py` pulls datagen output, trains, screens on
best-scale validation loss (val.bin + val17x2), embeds the net on a
`nnue-<id>` branch and queues the gate; every run, failed or not, gets a row
in `net/experiments.jsonl`. `netloop.py auto` runs unattended but only
ingests data now (its short fine-tunes never won). The protocol and results
are in `docs/nnue-experiments.md`.

### Net naming

| Name | Meaning |
|---|---|
| `E1`-`E3` | first v2-trainer experiments (2026-11-06) |
| `g<N>` | generation N: from scratch on pool N (e.g. g4 = pool4, 730M positions) |
| suffix `a` / `b` | alternative shapes of the same generation |
| `f` | trained on fixed-scale (corrected) data, e.g. g3f |
| `w` | warm start from the previous champion, e.g. g4w |
| `s` | output layer rescaled (eval-scale control), e.g. g3fs, g4ws |
| `kb` / `kbf` | king buckets / king buckets + factoriser |
| `kb4`, `kb8m` | king layouts (4 buckets; 8 mirrored), e.g. g4kb8m |
| `l3` | lambda 0.3 instead of 0.5, e.g. g4l3 |
| `ft` | plain fine-tune control for a warm-started variant, e.g. g4ft |
| `m1` | mirrored one-bucket copy (start point for mirrored layouts), e.g. g4m1 |
| `h-` prefix | half-data learning-curve run |
| `auto-N` | net from the `netloop.py auto` loop |
| `local-<id>` | dashboard row for a local experiment that was never branched |

## Never

- Never push engine changes to `main`, force-push `main`, or delete branches.
- Never edit `queue.json` / `log.jsonl` on `queue-state` by hand; only use
  `scripts/queue.py add ...` / `scripts/queue.py unpause`.
- Never edit `.github/workflows/*` or `scripts/queue.py` / `scripts/sprt.py`.
  The pipeline is shared and a mistake stalls everything; leave a note in
  `plan.md` for a human instead. (Workflow files also need an SSH push.)
- Never queue something as `speedup` unless you verified: bench identical at
  default depth and `bench 15`, `evalhash` identical to a `main` build, and a
  real speed gain (≥2%, CPU cycles via `/usr/bin/time -l` over 7 interleaved
  pairs of `bench 13`). The queue re-checks bench and evalhash, but if in doubt
  use `stc`: the SPRT costs an hour of CI, a bad speedup costs strength.
- Never accept or reject a change on local games; only the queue's SPRT counts.
  Don't run long local matches at all.
- Never retry an idea that `log.jsonl` or `tests/sprt-log.jsonl` shows was
  rejected, unless you change it materially and say how in the commit message.
- Don't start `ccrl-ladder.yml`, `spsa.yml`, `train-sweep.yml` or `datagen.yml` runs: they take
  the runners the queue needs. Leave that to a human.
- Don't add dependencies (std only; Python scripts stdlib only).

## Idea backlog

The old precise-spec backlog (ttcapture-lmr, rfp-blend, nmp-tt-guard,
aspiration-failhigh-depth, cutnode-lmr, double-extension, ...) has all been
tested; see `log.jsonl`. The current list is the "Search gap backlog" in
`plan.md` (audit 2026-11-07, with Elo guesses): threat-aware quiet history,
ttPv flag, !improving in LMR, LMR for bad captures, continuation history
re-keyed to [piece][to][piece][to], cuckoo repetition, root LMR, qsearch
polish. Check the queue and `log.jsonl` first (several are queued as
`feat-*`), write each as a precise spec before coding, one branch each.

## Budget

The user caps LLM usage. Keep to a few branches per session and stop when asked.
The queue runs on GitHub with no LLM involvement, so leaving it full is the most
useful thing you can do before stopping.
