# Rustfish roadmap

The single source of truth for what is running, what is next and why. Process
rules live in `docs/development.md`; results live in `tests/*.jsonl` and the
`queue-state` branch. Update this file whenever work starts, lands or dies.

_Last updated: 2026-11-04._ Live state (current test, queue, results, strength
chart) is on the dashboard, **https://rustfish-dash.dev.duk.im**; this file holds
the plan, not the live state.

## Strategy update (2026-11-04)

NNUE is the engine and the net chain is the main line. The queue is no longer a
general search-patch pipeline: net generations go to the front, search work is a
filler behind them, and ladders are held until the chain plateaus.

- **Measured:** 3174.1 CCRL Blitz (CI 3151.1-3197.6), last ladder on `030df68`.
- **Estimated:** ~3230.8 = 3174.1 + probcut +4.35 + net3 +39.79 + net4 +12.58.
  The dashboard shows measured and estimated separately.
- **Main:** `c24588c`, NNUE net4 (768->256x2->1 SCReLU, i16).
- **Net results so far:** net1 +51.2, net2 +149.1, net3 +39.8, net4 +12.6,
  net5 **-66.9 rejected** (37.6M positions, below the data knee), net6 (trained
  on rejected net5 data) **void, do not queue**.
- **Next:** net6-r2 on seed 8 (1M games from net4, ~77M positions, 2 epochs,
  control). Then net7 on a **mix** of net3-era and net4-era data (80M total,
  3 epochs); if net7 does not beat net6-r2, the bottleneck is capacity, not data.
- **Capacity after that:** output buckets first (cheap inference), then 512-wide
  hidden, then king buckets.
- **Runner constraint:** 20 concurrent GitHub jobs, so datagen (20 shards) and
  SPRT (19 shards) cannot overlap; per-generation wall time is dominated by
  datagen plus the short gate.
- **Ladder hold:** `ladder_every` is 50 (branch `net-loop`) until the chain
  plateaus; restore 5 at the end. `net-gate` is 5+0.05, bounds [0,10]; search
  changes still use `stc` [0,5].

Full audit trail and open questions: `docs/opus-audit-2026-11-04.md` (once the
docs branch lands) and `net/chain.json`.

## Where we are

| | |
|---|---|
| Strength | **3174.1 CCRL Blitz** measured (95% CI 3151.1-3197.6), 1440 games, on `030df68`; **~3230.8 estimated** from merged SPRT gains since (probcut, net3, net4). |
| Target | 3400 (dashboard). Single-CPU list: 3400 is ~#131; top 100 is ~3500; top 50 ~3650. |
| Speed | ~1.3M nps single thread on the dev box after the wide-output AVX2 kernel (+21%) and runtime AVX2 dispatch (+15-20% on default builds). |
| Eval | NNUE net4 embedded and on by default; `EvalFile` overrides; HCE remains the fallback. |
| Search | PVS, aspiration, TT buckets, NMP, RFP, futility, LMP, LMR, IIR, SEE, killers, counter-moves, history, 1-ply continuation history, ProbCut, singular extensions, check extension. |
| Data | `datagen.yml` self-play, 20 shards, deterministic seeds; training local with `tools/nnue-trainer` (std only). |

## Known problems (not fixed yet)

- **Ladder anchors now reach 3208** (`viridithas-4.0.0`, `stash-28/30/31`,
  `black-marlin-4.0` added 2026-11-02; all built and smoke-tested locally).
  The ladder runs ~50% longer with 15 anchors, so watch `ladder_every`.
- **SPRT bounds `[0, 10]` reject real +2..5 patches.** Correct for now (big
  gains still available), switch to `[0, 5]` once typical passes drop below +10.
- **Ladder cost**: the queue now runs a CCRL ladder every 5 merges (logged
  automatically), which pauses SPRTs for ~2 h each time. Raise `ladder_every`
  if that overhead outweighs the value of the fresher number.
- **Bot merges skip `ci.yml`** (GitHub token pushes don't trigger workflows);
  the queue runs the same checks itself, so this is cosmetic.
- **Queue logs live on `queue-state`**, not `tests/sprt-log.jsonl`; both are
  records now, the queue's is authoritative for queue tests.
- **No NNUE training workflow yet.** `tools/nnue-trainer` is ready
  (checkpoint/resume, measured throughput), but `nnue-train.yml` has to be
  added by a human (workflow files need an SSH push). Until then, training
  runs locally on a downloaded datagen artifact.
- **Time management** is only tested at 60+0.6; CCRL is 120+1 / 40/15 and
  Lichess games are often longer. `tm-budget` is in the queue.
- **Lazy SMP is untested for strength** (CCRL blitz is 1 thread, so it only
  matters for the Lichess bot and 4-CPU lists).

## Medium term (next 2-6 weeks), in priority order

1. **Net chain (primary).** Keep 1M-game datasets; do not drop below ~75M
   positions per generation until a mixed-data experiment proves smaller sets
   hold. Train on mixed generations (latest + previous, 50/50) to avoid the
   self-play collapse that net5 exposed. Add mirror augmentation and 3-4 epochs
   with decay once the mixing baseline is measured.
2. **Capacity (engine + trainer).** Output buckets (8 material buckets):
   output layer only, negligible NPS cost, typically +10-20. Then a 512-wide
   hidden layer (2x accumulator, ~15% NPS cost, +20-30) and king buckets.
   Measure each with the same net-gate.
3. **Search fillers (behind the nets).** Rebase and queue `bigbang-structural`
   (ProbCut + razoring + qsearch futility + double extensions + NMP verification)
   and `cont-hist-2ply` when the net chain pauses or the queue is idle. Both hit
   rebase conflicts when main moved; they are parked, not rejected.
4. **SPSA** on search parameters once the search-filler wave lands; it consumes
   the same runners as SPRTs, so run it as a scheduled block, not in parallel.
5. **CCRL submission.** `ccrl-identity` and `ccrl-docs` are ready as speedups;
   the release/licence decision is the owner's. Until then all ratings are
   internal ladders.
6. **Lichess bot** and **Syzygy** as public/endgame work after the net chain.

### Data-size decision rule

For each generation measure `G(40M) / T(40M)` against `G(80M) / T(80M)`.
If the smaller set delivers at least ~55% of the larger set's gain, it wins on
Elo/hour. Net5 failed at 37.6M (-66.9), so the working answer is **80M minimum**
unless a mixed 40M set proves otherwise. Record the test in `net/chain.json`.

## Long term (the path to beating Stockfish 17 skill 17)

1. **NNUE, std only.** Inference is (768 -> N)x2 -> 1 with king buckets still to
   come; trainer is std-only Adam with WDL+score loss, checkpoint/resume. Data
   is self-play at fixed nodes, mixed across generations; acceptance is the same
   SPRT gate as everything else.
2. **Syzygy probing** (std only, read-only of user-supplied files): small
   blitz gain, matters more for CCRL 40/15 and correct endgame conversion.
## Long term (the path to beating Stockfish 17 skill 17)

1. **NNUE, std only.**
   - Inference: (768 → N)x2 → 1 with king buckets later, int16 quantized,
     incremental accumulator in make/undo, SIMD via `std::arch` (AVX2 on CI,
     NEON on M3) behind a scalar fallback. Net embedded with `include_bytes!`.
   - Trainer: our own Rust trainer (`tools/` crate or `src/bin`, still no
     dependencies): Adam, WDL+eval blended loss, multithreaded with std threads.
     Train on CI (6 h job limit, checkpoint + resume) or locally overnight.
   - Data: self-play at fixed nodes from the current engine (HCE first), then
     retrain on data from the first net (the usual bootstrap).
   - Acceptance: an STC SPRT like everything else; expect the first net to be
     the largest jump the project has had.
2. **Syzygy probing** (std only, read-only of user-supplied files): small
   blitz gain, matters more for CCRL 40/15 and correct endgame conversion.
3. **Lazy SMP strength** (for the bot and the 4-CPU lists).
4. **Search refinements that only pay off with NNUE** (more eval-dependent
   pruning, correction history on a better eval).

## Process audit (2026-11-01)

What works:
- `main` only moves through an SPRT against the current `main`, now enforced by
  the queue rather than by discipline. The stale-base mistake that voided 14
  tests is structurally impossible.
- Waves stop clear results at 512 games; pentanomial LLR; pinned SHAs.
- Absolute strength comes from a CCRL anchor ladder, not `UCI_Elo` guesses.

What to change (ordered by payoff / effort):

| # | Change | Why | Who decides |
|---|---|---|---|
| 1 | Build both engines once per wave (`target-cpu=x86-64-v3`) and pass binaries to shards as artifacts | ~15-20% more games per CI hour; `-C target-cpu=native` binaries can't be shared across runner CPUs, v3 (AVX2/BMI2) can | **done** |
| 2 | Use 19 shards for STC SPRTs | 4 idle runners = +20% throughput | **done** |
| 3 | Queue runs the LTC non-regression of `main` vs the `main` 5 merges back itself, and pauses on a regression | regression protection without anyone remembering | **done**; weekly CCRL ladder cron still to build (needs result logging on CI) |
| 4 | Queue retries infra failures once before `needs-human`; a heartbeat recovers lost callbacks | transient GitHub/network errors shouldn't drop a candidate or stall the queue | **done** |
| 5 | Keep the queue ≥ 24 h deep; refill in batches from the medium-term list | CI is the bottleneck, it should never idle; LLM time goes into writing candidates, not watching tests | process |
| 6 | Combine exact speedups into one `speedup` item | no SPRT, no conflicts with each other in the queue | process |
| 7 | Optional: a scheduled cloud agent that refills the queue when it drops below ~8 items | fully hands-off idea generation; costs LLM tokens on a schedule | user |
| 8 | Optional: more compute: a self-hosted runner (a rented many-core box, or this laptop overnight) | CI time is the binding constraint; a 16-core box roughly doubles test throughput | user (cost) |

Division of labour, explicitly:
- **GitHub Actions**: every game played for acceptance or rating, the queue,
  data generation, SPSA, later NNUE training.
- **Laptop**: writing candidates, correctness gates, deterministic bench
  checks, speed work (NPS comparison is valid locally; Elo is not).
- **Dokku box (2 shared cores)**: the Lichess bot only; too small for testing.
- **LLM**: ideas, implementation, triage of `needs-human` items, and this
  plan. Not polling, not merging, not logging: the queue does those.

Is it enough in parallel? CI throughput is compute-bound, so running several
SPRTs at once would not finish them sooner, only make each slower; serial is
right. The parallelism worth adding is on the *input* side (more candidates
ready, so the queue never idles) and in *non-game* work that doesn't compete
for runners: speedups, data generation tooling, the NNUE code.

## History

- Jul 2026: plan and strategy notes in `docs/archive/plan-2026-07.md` and
  `docs/archive/post-2100-revised.md` (pre-pipeline; their rejections are void).
- 2026-10-29: depth-8 cap fixed (+221), SPRT pipeline and CCRL ladder built: ~2081.
- 2026-10-31: IIR, SEE fix, TT review, eval rework (king safety, threats,
  passers, joint retune): ~2624 → ~2810.
- 2026-11-03: crossed 3000 CCRL Blitz: **3025.5** (CI 3004.8-3046.3) on the
  first NNUE net, after 2889.1 on the previous ladder (+136).
- 2026-11-03: queue wave: singular extensions +23.7 and per-piece mobility
  +35.8 merged; first NNUE net +51.2 merged (after fixing the eval-cache
  stm-POV sign); anchors extended to 3208. Rejected: probcut (+5.0,
  inconclusive), extensions, ttcapture-lmr, aspiration fail-high depth.
- 2026-11-01: unattended merge queue.
