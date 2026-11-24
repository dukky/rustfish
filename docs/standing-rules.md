# Standing rules

Rules the owner has set over the project's life. They apply to every session
and agent on any machine: a laptop session, a desktop session, a cloud
routine or a subagent. `plan.md` says what to work on; this file says how.
Machine-specific details (hostnames, paths, how to reach a machine) are not
kept here: each machine has a gitignored `CLAUDE.local.md` for those.

## Engine identity
- **From scratch.** Train only on rustfish's own self-play data. No Lc0/T80 or
  other external training data (owner, 2026-11-12; revisit only at a real
  data wall).
- **Independent development.** Other engines are opponents and benchmarks
  (matches, NPS, nodes-to-depth) and a source of broad, publicly described
  ideas ("more aggressive LMR", "pawn history"). Never read, port or recall
  another engine's code, formulas, tables or tuned constants, and never ask an
  agent to. Find our weaknesses with our own instrumentation; design and tune
  our own fixes.

## Evidence
- **Measure, don't guess.** Back every choice, including infrastructure
  (threads, machine splits, power limits), with a controlled measurement:
  one variable, interleaved runs, noted spread, a no-op/dummy control.
- **Flag confounds** before drawing conclusions (different machine, trainer,
  load, base commit).
- **One confounded run is a hypothesis, not a verdict.** Label conclusions
  with their evidence strength; one-run negatives are "inconclusive", not
  "dead". Do not drop a line of work until a controlled test says so.
- **Acceptance comes only from the GitHub SPRT** (merge queue or
  `scripts/ci.py sprt`). Local games, loss numbers and node counts screen;
  they never accept.
- **Absolute strength comes only from the CCRL ladder**, with anchors on both
  sides of rustfish (none scored above ~75%). Ladders on a set mostly below us
  read high.
- **Before any hand-dispatched CI run,** compare its settings (anchors, games,
  TC, bounds, base ref) with the last comparable run, check the job list
  within minutes of starting it, and fix stale defaults at their source.
- **Validation sets:** each net generation gets a fresh held-out set from the
  newest datagen of the current main (never in any pool); see
  `docs/nnue-experiments.md`.

## Keeping state current
- `plan.md` is the source of truth for every session. Update it in the same
  turn a result lands or the direction changes (small edit, commit, push);
  retire stale rows at once.
- Update `README.md` (strength, targets, engine row) in the same commit as
  every ladder, architecture change or data-size change.
- Every deferred action needs an armed trigger (a watcher, a scheduled job, or
  a row in plan.md's deferred-actions table with its trigger).

## Compute
- GitHub Actions is for testing only (SPRTs, ladders, peer matches), never
  bulk datagen or training.
- **Never stop datagen** on the data machine without the owner's say-so, and
  restart it when it stops (power cut, end of a seed batch).
- **GPU power cap:** long training runs only at the owner-set cap (150 W
  from 2026-11-24; 125 W before). A reboot resets it to the default ~275 W;
  the owner sets it by hand. Never change the power limit
  yourself. At the default (uncapped) limit, short experiments only.
- Cancelling CI runs and killing owner-started GPU jobs need the owner's OK.
- The owner's laptop may be in use or asleep: no measurement runs there
  unless the owner says it is free.
- Don't redeploy the Lichess bot without being asked.

## Working safely
- Scratch work and git worktrees go in a scratch directory, never inside the
  repo checkout; never `rm -rf`, `git checkout --`, `git reset --hard`,
  `git stash`, or shell redirects into the repo.
- Delete only explicitly named files (`unlink <file>` or `rm <file>` with a
  full name), never globbed or variable-built `rm` patterns, and only after
  checking what they are (e.g. a packed copy verified position for position).
- Kill only your own processes, by PID, never `pkill -f <pattern>` on a
  shared machine.
- Subagents work on branches and report; the coordinating session decides
  what enters the merge queue.
- Usage: heavy agent fan-out only when the owner asks for it; prefer work that
  runs without tokens (the queue, datagen, training).
