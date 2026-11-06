# Rustfish roadmap

The single source of truth for what is running, what is next and why. Process
rules live in `docs/development.md`; net experiments in
`docs/nnue-experiments.md`; results in `tests/*.jsonl`, `net/chain.json` and
the `queue-state` branch. Update this file whenever work starts, lands or dies.

_Last updated: 2026-11-06 14:30._ Live state (current test, queue, results, strength
chart) is on the dashboard, **https://rustfish-dash.dev.duk.im**.

## Where we are

| | |
|---|---|
| Strength | **~3330 CCRL Blitz** (partial ladder, 95% CI 3223-3460) on `ad53daa` (E1 net + halved eval scale); rerun with 3273-3491 anchors in progress. Was 3168 (net4) this morning. |
| Goals | **Short term: 3400 CCRL Blitz** (~#130 on the single-CPU blitz list; we are ~3330). **Mid term: a top-100 engine on CCRL Blitz** (~3500). **Long term: top 10** (~3700+). Competing with Stockfish itself is out of scope for our resources. |
| Eval | NNUE E1, (768 -> 256) x 2 -> 1 SCReLU, i16, output halved to the HCE scale the search margins were tuned on (+123); HCE is the fallback. |
| Search | PVS, aspiration, TT buckets, NMP, RFP, futility, LMP, LMR, IIR, SEE, killers, counter-moves, history, 1-ply continuation history, ProbCut, singular extensions + multi-cut, check extension. |
| Speed | ~1.3M nps single thread with NNUE (AVX2/NEON kernels). |

## Status 2026-11-06 afternoon

- Merged today: E1 (+152 net gate), eval scale 1/2 (+123 stc). Rejected: E3 (8
  output buckets, +4.9 inconclusive), bigbang-r2 (-103).
- Testing: eval scale 3/8 and 5/8 (bracketing 1/2), nnue-speed (+4% NPS at
  256, +14% at 512, search-identical), ladder rerun, then g2a/g2b.
- Generation 2: g2a = 256x8 on pool2 (442M positions: pool1 + seeds 11, 12),
  15 epochs on the CPU; g2b = 512x8 on pool2, 10 epochs on the GPU.
- **Metal GPU trainer** (`tools/nnue-trainer-metal`): parity with trainer v2,
  2.7-3.3x faster. GPU does production training; the CPU now does local
  datagen. Generations are now **data-bound**, not training-bound.
- **CI lanes:** queue SPRTs and ladders on 16 runners (16 shards);
  background lane (4 runners) self-feeds datagen from main every heartbeat.
- In progress (subagents): king-bucketed inputs (engine + both trainers, off by
  default), time management for 120+1 and 40/15, search-identical speedups.
- SPSA of 22 search parameters is ready on `spsa-nnue` (8 shards, ~14 h); run
  overnight when the queue is empty.

## Strategy (2026-11-06): train the net properly

The net is the engine, and it has been held back for the wrong reason. The
audit in `docs/nnue-experiments.md` found that net5-net13 (nine rejected nets)
were confounded, undertrained (~2 epochs on ~80M positions on a 2-core VPS)
and judged by small game samples with no validation loss. The conclusions
drawn from them ("256 is saturated", "512 fails", "40M is below the data
knee") are withdrawn.

Every net step is judged by validation loss first, then an SPRT:

1. **E0** score net4 on the fixed validation set (baseline).
2. **E1** same shape (256, 1 bucket), trainer v2, whole valid pool (~350M
   positions), ~10 epochs, cosine LR. No engine change: tests "the trainer
   was the bottleneck".
3. **E2** 512 wide (`nnue-arch` supports it exactly). `stc` SPRT, since it
   costs ~15-20% NPS.
4. **E3** 8 output buckets on the best width (cheap at inference).
5. **Generation loop**: datagen from the merged net into the pool (keep every
   valid generation), retrain on the whole pool, longer each time.
6. Later: king/input buckets; a GPU trainer (bullet on the 1080 Ti desktop) if
   CPU training becomes the bottleneck.

Compute split: **training on the M3 laptop** (8 cores: ~1.1M positions/s at
H=256, ~0.5M at 512; ~12x the VPS), **datagen and SPRTs on CI** (20 runners;
they cannot overlap, so datagen runs when the queue is idle).

## Next 24 hours (from 2026-11-06 10:00 UTC)

Laptop (training, serial, one run at a time):

| when | run | cost | gate |
|---|---|---|---|
| now | E0: net4 on val | 1 min | baseline |
| now -> +1h | E1: 256x1, pool, 10 epochs | ~1h | val < net4, then net-gate |
| +1h -> +3h | E2: 512x1, same data and recipe | ~2h | val < E1, then stc |
| +3h -> +5h | E3: best width x 8 buckets | ~1-2h | val < best, then stc |
| overnight | gen-2: best shape, pool + seed 11 + seed 12, 20-30 epochs | ~4-6h | net-gate/stc vs merged best |

CI (one thing at a time, queue first):

| when | job |
|---|---|
| now -> ~+2h | datagen seed 11 (net4, 1M games, 8000 nodes) |
| as each net is ready | its SPRT (net gates take 10-40 min for real effects) |
| after the first net merge | datagen seed 12 from the new main (bootstrap rule) |
| idle gaps | rebased search fillers (`bigbang-structural`, `cont-hist-2ply`) |
| after the day's merges | one CCRL ladder (with new 3200-3400 anchors) |

Training efficiently:

- Convert once to 32-byte binary records; never re-parse text.
- Train on the whole pool every time; add new generations, don't replace.
- Screen by validation loss (minutes) before spending CI games (hours).
- One variable per run; record every run in `docs/nnue-experiments.md`.
- Keep the laptop busy with the next run while CI gates the last one.

## CI efficiency (measured 2026-11-06)

- STC waves: ~90% of wall time is games. Fine.
- Net-gate waves (5+0.05): ~170 s, of which ~60 s was rebuilding engines;
  fixed by caching the engines across waves (`34091d5`).
- The largest waste was *what* was tested (~10 doomed nets, ~24 rejected
  search patches, ~200k games in two days): screen by validation loss first.
- Datagen must leave runners free: dispatch with `max_parallel` <= 16 so the
  queue's own jobs and SPRT builds are never blocked behind it.
- Unused lever: runners can run *independent* training runs (ablation
  sweeps scored by validation loss) overnight, when no SPRT needs them.
- Owner decisions: GitHub Pro (20 -> 40 concurrent jobs, 2x throughput) and a
  self-hosted runner on the 9600X desktop (outside the job cap).
- Ladder cost: a 120+1 ladder is ~10 SPRTs of runner time; run it only after
  real gains. A 10+0.1 ladder is ~12x cheaper and comparable with other
  short-TC claims (e.g. the 3463 "24h" engine, which was rated at 10+0.1).

## Known problems

- **Anchors near the top.** Only stash-30/31 sit at 3150-3210 now; add
  3200-3400 anchors before the next ladder or the fit extrapolates, and check
  each new anchor's score against its neighbours (how black-marlin was caught).
- **`evalhash` covers the HCE only** (no net attached), so for NNUE code the
  speedup gate relies on bench identity. Make it hash NNUE evals too.
- **SPRT bounds:** `stc` is [0, 5]; `net-gate` (5+0.05, [0, 10], 4 waves) is
  for same-shape nets only.
- **Bot merges skip `ci.yml`**; the queue runs the same checks itself.
- **Time management** is only tested at 60+0.6; CCRL is 120+1 and 40/15.
- **Lazy SMP is untested for strength** (bot and 4-CPU lists).

## Medium term (after the net plan)

1. Search fillers (above), then **SPSA** of search parameters on the settled net.
2. **CCRL submission** (`ccrl-identity`, `ccrl-docs` ready; release/licence is
   the owner's decision).
3. **Lichess bot** and **Syzygy**.

## Process

What works: `main` only moves through an SPRT against the current `main`
(merge queue); pentanomial LLR; pinned SHAs; absolute strength from a CCRL
anchor ladder logged by CI.

Rules added from the 2026-11-04..06 history:

- Net experiments follow `docs/nnue-experiments.md`: one variable, validation
  loss before games, every run logged including failures.
- No strategic pivot on a sample that cannot support it (e.g. 200 games).
- Every ladder anchor's score is sanity-checked against its neighbours.

Division of labour: GitHub Actions plays every game and generates data; the
laptop trains nets and does correctness/speed work; the Dokku box runs the
dashboard (later the bot); the LLM writes candidates and keeps this plan
current, not polling and merging (the queue does that).

## History

- Jul 2026: pre-pipeline notes in `docs/archive/` (their rejections are void).
- 2026-10-29: depth-8 cap fixed, SPRT pipeline and CCRL ladder: ~2081.
- 2026-10-31: IIR, SEE fix, TT review, eval rework: ~2624 -> ~2810.
- 2026-11-01: unattended merge queue. 2026-11-02: 2889.
- 2026-11-03: first NNUE net (+51), singular extensions (+24), mobility (+36):
  ~3001 (refit). net2 +149.
- 2026-11-04: net3 +40, net4 +13, ProbCut +4: 3168 (refit) on net4.
- 2026-11-04..05: net5-net13 all rejected (see the audit); 512-wide -30.
- 2026-11-06: audit; trainer v2; black-marlin anchor removed.
