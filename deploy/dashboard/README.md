# rustfish progress dashboard (Dokku)

A live status page at https://rustfish-dash.dev.duk.im: current CCRL estimate
vs the Stockfish 17 skill 17 target, the in-flight merge-queue SPRT, the
pending queue, recent SPRT/queue outcomes, the ladder rating over time with
the changes that passed in between, and `plan.md` rendered.

`server.py` is Python stdlib only. It reads public data (no token):

- `raw.githubusercontent.com`: `tests/ladder-log.jsonl`, `tests/sprt-log.jsonl`
  and `plan.md` on `main`; `queue.json` and `log.jsonl` on `queue-state`;
  `net/experiments.jsonl` (main, else the `netloop` branch) and the v1
  `net/chain.json`. The NNUE panel lists every net branch's latest gate
  status straight from the queue log (testing, queued, merged, rejected, ...)
  joined by branch name to its `netloop record` row; a gated net with no row
  is flagged "no training record".
- `api.github.com` (unauthenticated, 60 req/h): recent Actions runs, the jobs
  of the current SPRT wave (to count finished shards), and recent `main` commits.

Each upstream URL is cached in memory (raw files 2 min, runs/jobs 3 min,
commits 10 min) and refreshed only when a page asks, with ETags, so the API
use stays under ~50 requests/hour even with the page open all day. If GitHub
is unreachable the last good data is served with a warning banner. The page
polls `/api/status` every minute.

Games played in the in-flight SPRT are estimated from finished shard jobs and
the per-mode wave sizes mirrored from `scripts/queue.py` (`ROUNDS` in
`server.py`); update that table if the queue settings change.

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
