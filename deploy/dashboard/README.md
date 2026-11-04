# rustfish progress dashboard (Dokku)

A live status page at https://rustfish-dash.dev.duk.im: measured CCRL Blitz
strength next to an explicitly unofficial estimate, the live datagen wave,
the NNUE net chain, the in-flight merge-queue SPRT, the pending queue, recent
SPRT/queue outcomes, the ladder rating over time with the changes that passed
in between, and a pointer to the current audit doc (`plan.md` is kept but
marked stale).

`server.py` is Python stdlib only. It reads public data (no token):

- `raw.githubusercontent.com`: `tests/ladder-log.jsonl`, `tests/sprt-log.jsonl`
  and `plan.md` on `main`; `queue.json` and `log.jsonl` on `queue-state`;
  `net/chain.json` on `main`, falling back to `net-chain` until it is merged.
- `api.github.com` (unauthenticated, 60 req/h): recent Actions runs, the jobs
  of the current SPRT wave and of the active Datagen run (to count finished
  shards), and recent `main` commits.

Each upstream URL is cached in memory (raw files 2 min, runs/jobs 3 min,
datagen jobs 5 min, commits 10 min) and refreshed only when a page asks, with
ETags, so the API use stays under ~55 requests/hour even with the page open all
day. If GitHub is unreachable the last good data is served with a warning
banner. The page polls `/api/status` every minute.

Games played in the in-flight SPRT are estimated from finished shard jobs and
the per-mode wave sizes mirrored from `scripts/queue.py` (`ROUNDS` in
`server.py`); update that table if the queue settings change. Datagen
positions are estimated as finished shards x games per shard x 79
positions/game (measured from the net3 set).

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
