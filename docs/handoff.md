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
   "Idea backlog" below (or from `plan.md` "Medium term"). One idea per branch.
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
5. Record what you did in `plan.md` (short-term section): what you queued, what
   you dropped and why. Commit that to `main` (docs-only commits are fine to push
   directly; see "Never" for what isn't).

## Choosing the queue mode

| Change | Mode |
|---|---|
| New search/eval idea, parameter change | `stc` |
| Removing code or a term (simplification) | `nonreg` |
| Time management | `ltc-gainer` |
| Speedup that changes nothing (bench AND `evalhash` identical to `main`) | `speedup`, but see "Never" |

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
- Don't start `ccrl-ladder.yml`, `spsa.yml` or `datagen.yml` runs: they take
  the runners the queue needs. Leave that to a human.
- Don't add dependencies (std only; Python scripts stdlib only).

## Idea backlog (precise specs)

Check `src/search.rs` and the queue first: skip anything already present,
queued, or rejected. Values are conventional starting points; one branch each.

1. **ttcapture-lmr**: in negamax's LMR, if the TT move is a capture, reduce quiet
   moves one ply more (`r += 1`).
2. **check-lmr**: reduce one ply less for moves that give check (if not already).
3. **rfp-blend**: when reverse futility pruning fires, return `(eval + beta) / 2`
   instead of `eval` (non-mate scores only).
4. **nmp-tt-guard**: skip null-move pruning when the TT entry has an upper bound
   (or exact score) below beta.
5. **aspiration-failhigh-depth**: on an aspiration fail-high, re-search at depth
   `depth - 1` (not below 1) while widening, resetting to full depth when the
   window is re-centred (Stockfish style).
6. **cutnode-lmr**: if a `cut_node` flag exists in negamax (from `iir-cutnode`),
   reduce one ply more at expected cut-nodes for quiet moves.
7. **double-extension**: once a singular extension is on `main`, extend by 2 when
   the singular search score is below `singular_beta - 20` at non-PV nodes, with
   a per-path cap (e.g. at most 6 double extensions per line).
8. **qs-check-evasion-order**: in qsearch when in check, order evasions by
   captures (MVV-LVA) first, then killers, then history.

When the backlog is used up, add new ideas here (with a spec this precise) as
you think of them, or take them from `plan.md` "Medium term".

## Budget

The user caps LLM usage. Keep to a few branches per session and stop when asked.
The queue runs on GitHub with no LLM involvement, so leaving it full is the most
useful thing you can do before stopping.
