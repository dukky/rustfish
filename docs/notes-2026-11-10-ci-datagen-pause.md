# CI datagen paused — 2026-11-10

**For:** Opus / the next engine agent
**From:** Hermes (on dukky's instruction)
**Main at pause:** `614ea1b` (ladder ~3508)
**Status:** Datagen workflow **disabled manually**; active/queued runs cancelled.

## Why

1. **GitHub Actions policy risk.** The background lane was running 26-shard
   datagen runs continuously (13 at a time), plus foreground datagen bursts
   whenever the SPRT queue was idle. GitHub's additional terms restrict hosted
   runners to production, testing, deployment and publication of the project,
   and prohibit workloads that place a disproportionate burden on their
   infrastructure. Bulk self-play data generation is compute-heavy, not a test;
   an always-on 26-shard lane is hard to defend. Public-repo free minutes do not
   exempt it. The safer position is no bulk datagen on GitHub-hosted runners.
2. **Data strategy is moving to T80/LC0.** Coda's own evidence: a 3B-position
   pure self-play replacement lost **−20 Elo** against LC0/T80 at similar size.
   Coda's production nets are trained on T80/LC0, not self-play. If rustfish
   pivots to T80/LC0, the 18–26 shard self-play farm is no longer the critical
   path; bulk datagen should move to owned hardware or stop entirely.
3. **Runner contention.** The datagen lane was holding 13 background runners and
   bursting into the foreground whenever the queue was idle, competing with
   SPRTs and ladders. CI is more valuable as a test farm.

## What was stopped

- Workflow: `Datagen` (`datagen.yml`, id `404561570`) — state
  `disabled_manually`.
- Cancelled runs:
  - `42916579070` — seed 168, 12×15,000 games (foreground burst,
    `q-burst-s168`)
  - `42919113219` — seed 169, 26×15,000 games (background lane)
- Completed seeds 160–167 were left in place; their artifacts are still on the
  workflow runs.
- Queue state may still show `current: {mode: datagen-burst, test_id:
  q-burst-s168}` until the next queue advance. `burst_done()` should clear it;
  verify after the next `queue.yml` heartbeat. Do not hand-edit `queue.json`.

## What is still allowed on CI

- Build/test (`ci.yml`)
- SPRT (`sprt.yml`) and the merge queue (`queue.yml`)
- CCRL ladder (`ccrl-ladder.yml`)
- SPSA (`spsa.yml`), trainer bench, NNUE ISA check
- Release builds
- Small, occasional datagen smoke shards if explicitly needed — not a
  continuous background lane

## How to resume (only if the owner decides)

```bash
gh workflow enable datagen.yml
```

Then either let the queue's background lane dispatch, or start a single
explicit run. Before re-enabling, decide whether the data source is still
self-play or T80/LC0. If T80/LC0 becomes primary, keep the lane disabled and
keep any local datagen on the 9600X/Mac/1080 Ti.

## Related

- One-off rustfish-vs-Coda 0.9.4 match: branch `ci-coda-match`, workflow
  `coda-match.yml`. It is an experiment only; never merge it.
- T80/LC0 data is ODbL-1.0/DBCL-1.0. Attribution required; do not redistribute
  the raw database. A trained net is a Produced Work under the interpretation
  Coda uses; get the notice right before any release.
