# rustfish progress dashboard (Dokku)

A live status page at https://rustfish-dash.dev.duk.im. Top to bottom:

1. Headline: measured CCRL strength, the estimate, the next CCRL Blitz goal
   and the gap, the goal strip (3400, top 100, top 10).
2. Strength: a one-line summary (last ladder, change vs the run before,
   estimate) and the ladder rating chart, with the ladder runs as a table.
3. In flight: the item the merge queue is testing (SPRT wave or CCRL ladder),
   the pending queue, CI runners (what is running on Actions right now) and,
   once a training box pushes its status, the training machines.
4. Nets: the NNUE generation table (every trainer-v2 net, grouped by
   generation) and the superseded v1 history.
5. Results: recent outcomes, gate Elo of accepted changes, recent commits.
6. Elsewhere: the Lichess bot card, then `plan.md` (the roadmap) rendered.

Long lists show their newest rows with a "Show all" button (the choice
survives the minute refresh); the plan is clamped behind "Show the full plan".
The layout is checked at 390 px wide (phone) as well as desktop.

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
- `api.github.com` (unauthenticated, 60 req/h): every in-progress Actions run
  (`actions/runs?status=in_progress`, so a long SPSA round is never missed),
  the latest 30 runs (queued runs and the last finished SPRT wave), the job
  list of each live multi-job run, and recent `main` commits.
- `lichess.org` (public API, no token) for the Lichess bot card, served
  separately as `/api/lichess`: `/api/user/{name}` (ratings and game counts per
  speed), `/api/users/status?ids={name}` (online / playing),
  `/api/games/user/{name}?max=10` (NDJSON, recent games) and
  `/api/user/{name}/rating-history` (sparklines, last 60 days played). Each is
  cached 5 minutes (at most 4 requests per 5 minutes) with a descriptive
  User-Agent; a failure is remembered for a minute before retrying. The page
  fetches it on its own, so if Lichess is down only that card says
  "unavailable". The account is bound to the bot's token, not named in
  `deploy/lichess-bot/config.yml`, so set it once:
  `ssh dokku@dev.duk.im config:set rustfish-dash LICHESS_USER=<bot account>`
  (unset: the card says "not configured" and makes no requests).
  Tests: `python3 deploy/dashboard/test_lichess.py` (no network).

Each upstream URL is cached in memory and refreshed only when a page asks,
with ETags: raw files 2 min, the in-progress runs list 3 min, the recent runs
list 6 min, commits 20 min, and the live runs' job lists share a budget of 20
fetches an hour (each run's TTL is `max(3 min, 3600 s x live runs / 20)`).
Worst case ~53 requests/hour, under the 60/hour limit even with the page open
all day; ETag 304s do not count. If GitHub is unreachable the last good data
is served with a warning banner. The page polls `/api/status` every minute.

### CI runners
The panel counts what is actually running, not the queue's lane settings: the
in-progress jobs (and jobs waiting for a runner) of every live run, from each
run's Actions job list, grouped by workflow file into SPRT / queue
(`sprt.yml`, `queue.yml`), SPSA (`spsa.yml`), CCRL ladder
(`ccrl-ladder.yml`) and other (CI, datagen, train-sweep, ...), out of the
account's 20 concurrent jobs. Single-job workflows (`queue.yml`, `ci.yml`)
count as one job without a fetch. A run whose job list can't be fetched is
listed as "jobs unknown" and the total shows as a lower bound (">= n"). The
stc shard count from `scripts/queue-settings.json` is shown as context only.
Logic: `build_runner_usage` in `server.py`.

### NNUE generations
One row per trainer-v2 net (`net/experiments.jsonl` joined by branch to the
queue log), grouped by generation (the `gN` in the id or branch; E1-E3 as
"E"), newest first: shape (hidden width H, output buckets, king-bucket layout,
threat inputs), pool (`positions_per_epoch`, plus `train_data` if recorded),
best-scale val loss on val.bin / val17x2 (record fields `best_scale_val` /
`best_scale_val17x2`, else parsed from the `notes` text; else the raw val
loss), and the gate (mode, Elo +- CI, games, status). The net merged last is
highlighted "on main". A gated net with no experiments row is still listed,
flagged "no training record", with its shape read from the branch name
(`nnue-g8h1024kb8m-ep`: gen 8, H1024, kb8m, threats ep; marked `*`).
Collapsed view: merged nets, anything testing or queued, the last 2 days, and
the newest net of a generation that never merged. Optional record fields the
table understands: `threats` (or `inputs`), `train_data`, `best_scale_val`,
`best_scale_val17x2`. Logic: `build_generations` in `server.py`.

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

## Machine status push
A training box (the desktop: GPU training, CPU datagen) reports through
`POST /api/machine`: a JSON object (<16 KB) with
`Authorization: Bearer $MACHINE_TOKEN` (constant-time compare; the legacy
`LAPTOP_TOKEN` is accepted too; rejected when neither is set). The server keeps
the latest payload per `machine` name (up to 8; also in
`$DATA_DIR/machines.json` if `DATA_DIR` is set) and `/api/status` returns them
as `machines: [{name, schema, payload, received, age_s}]`. The page shows a
"Training machines" card only for payloads with `"schema": "machine-status/1"`
(stale past 15 minutes) and hides it when there are none. The old
`POST /api/laptop` still works (stored as machine `laptop`), but the laptop
card is gone, so `scripts/laptop_status.py`'s payloads are no longer shown.

Payload (`machine-status/1`, what `scripts/machine_status.py` sends):
```json
{"schema": "machine-status/1", "machine": "desktop", "ts": 1794529069,
 "gpu": {"name": "g9h1024kb8m", "log": "~/rustfish-data/g9h1024kb8m.log",
         "hidden": 1024, "buckets": 8, "king_buckets": "kb8m", "threats": null,
         "epoch": 3, "epochs": 10, "positions": 3201732989, "val": 0.025601, "val_q": 0.025611,
         "pos_s": 2250000, "progress": 0.34, "eta_s": 9300, "done": false, "keep_best": null,
         "out": null, "updated_s": 400, "idle": false},
 "datagen": {"name": "D2005", "log": "~/rustfish-data/D2005.log", "seed": 2005, "nodes": 5000,
             "out": "D2005.txt", "games_done": 120000, "games_total": 200000,
             "positions": 9480000, "games_s": 9.8, "pos_s": 774, "eta_s": 8160,
             "done": false, "updated_s": 30, "idle": false},
 "disk": [{"path": "~/rustfish-data", "free_gb": 180.2, "total_gb": 1000.0}],
 "load": [9.1, 9.0, 8.8]}
```
`gpu` / `datagen` are `null` when no matching log is found. GPU progress is
epochs done plus the share of the current epoch implied by the time since the
log was last written (the trainer writes once per epoch) at the last pos/s.

`scripts/machine_status.py` (stdlib, read-only) builds it from log paths:
```bash
python3 scripts/machine_status.py --machine desktop \
  --gpu-log '~/rustfish-data/*.log' --datagen-log '~/rustfish-data/D*.log' \
  --disk ~/rustfish-data --print          # show the payload, post nothing
```
`--gpu-log` takes nnue-trainer logs (CPU, CUDA, Metal: same `train H=...` and
`epoch k ...` lines; a chain log's last stage wins), `--datagen-log` the
stderr of `rustfish datagen`; both take paths or globs, repeat, and use the
most recently modified matching log. A log untouched for `--gpu-idle`
(default 3 h) / `--datagen-idle` (30 min) is reported idle. Token: env
`MACHINE_TOKEN`, else `--token-file` (default `~/.config/rustfish/machine-token`);
`--url` overrides the endpoint. Not installed anywhere yet. To run it on the
desktop every 5 minutes:
```bash
TOKEN=$(openssl rand -hex 24)
ssh dokku@dev.duk.im config:set rustfish-dash MACHINE_TOKEN=$TOKEN   # restarts the app
mkdir -p ~/.config/rustfish && (umask 077; printf '%s\n' "$TOKEN" > ~/.config/rustfish/machine-token)
crontab -e   # */5 * * * * python3 ~/rustfish/scripts/machine_status.py --machine desktop --gpu-log '...' --datagen-log '...' --disk ~/rustfish-data
```

Tests (no network): `python3 deploy/dashboard/test_status.py` (CI runner
grouping, the generation table, `build_status` end to end),
`python3 deploy/dashboard/test_machine.py` (push endpoint, auth, parsers) and
`python3 deploy/dashboard/test_lichess.py`.
