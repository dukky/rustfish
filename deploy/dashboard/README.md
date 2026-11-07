# rustfish progress dashboard (Dokku)

A live status page at https://rustfish-dash.dev.duk.im: measured CCRL strength
and the CCRL Blitz targets (3400 short term, top 100, top 10), CI runner usage
across both queue lanes, the foreground lane (the in-flight SPRT or CCRL ladder
and the pending merge queue), the background lane (datagen from main), the
ladder rating over time, NNUE experiments, recent queue outcomes, gate Elo of
accepted changes, recent commits and `plan.md` (the maintained roadmap) rendered.

`server.py` is Python stdlib only. It reads public data (no token):

- `raw.githubusercontent.com`: `tests/ladder-log.jsonl`, `tests/sprt-log.jsonl`,
  `plan.md`, `scripts/queue-settings.json` (wave sizes, time controls) and
  `scripts/ccrl-targets.json` (CCRL list thresholds) on `main`; `queue.json`
  and `log.jsonl` on `queue-state`; `net/experiments.jsonl` (main, else the
  `netloop` branch) and the v1 `net/chain.json`. The NNUE panel lists every net
  branch's latest gate status straight from the queue log joined by branch
  name to its `netloop record` row; a gated net with no row is flagged
  "no training record", and rows with branch `local-<id>` (laptop-only
  experiments, `netloop record-local`) show as "local only".
- `api.github.com` (unauthenticated, 60 req/h): recent Actions runs, the jobs
  of the in-flight foreground run (SPRT wave or ladder) and of the active
  datagen run, and recent `main` commits.

Each upstream URL is cached in memory (raw files 2 min, runs/jobs 3 min,
datagen jobs 5 min, commits 10 min) and refreshed only when a page asks, with
ETags, so the API use stays under ~60 requests/hour even with the page open
all day. The runner-usage panel reuses those fetches: exact job counts for the
runs whose jobs are fetched anyway, other in-progress runs estimated at their
workflow's max-parallel (`MAX_PARALLEL` in `server.py`). If GitHub is
unreachable the last good data is served with a warning banner. The page polls
`/api/status` every minute.

Ladder runs with far fewer games than planned (a cancelled or partial run) are
flagged `partial` and drawn hollow with their own CI bar.

## Run locally
```bash
PORT=8765 python3 deploy/dashboard/server.py   # http://localhost:8765
```

## One-time setup (done)
```bash
ssh dokku@dev.duk.im apps:create rustfish-dash
ssh dokku@dev.duk.im builder-dockerfile:set rustfish-dash dockerfile-path deploy/dashboard/Dockerfile
ssh dokku@dev.duk.im ps:set rustfish-dash procfile-path deploy/dashboard/Procfile
ssh dokku@dev.duk.im resource:limit rustfish-dash --memory 128m
ssh dokku@dev.duk.im ports:set rustfish-dash http:80:5000
git remote add dokku-dash dokku@dev.duk.im:rustfish-dash
git push dokku-dash main
ssh dokku@dev.duk.im ps:scale rustfish-dash web=1
ssh dokku@dev.duk.im letsencrypt:enable rustfish-dash
```

## Redeploy
```bash
git push dokku-dash main            # or <branch>:main
ssh dokku@dev.duk.im logs rustfish-dash -t
```
