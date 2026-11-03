# Rustfish roadmap

The single source of truth for what is running, what is next and why. Process
rules live in `docs/development.md`; results live in `tests/*.jsonl` and the
`queue-state` branch. Update this file whenever work starts, lands or dies.

_Last updated: 2026-11-02._ Live state (current test, queue, results, strength
chart) is on the dashboard, **https://rustfish-dash.dev.duk.im**; this file holds
the plan, not the live state.

## Where we are

| | |
|---|---|
| Strength | **3025.5 CCRL Blitz** (95% CI 3004.8-3046.3), 1440 games, 0 time losses, measured 2026-11-03 on the first NNUE net + singular extensions + mobility (15 anchors, 120+1). |
| Stockfish cross-check | ~2881 on Stockfish's UCI_Elo scale: 61% vs skill 10, 25% vs skill 15, 22% vs skill 17 (`tests/stockfish-log.jsonl`) |
| Target | Stockfish 17 skill 17 (UCI_Elo 3139, ~3140 CCRL Blitz) |
| Speed | ~3.0M nps single thread on M3 after the exact speedups (magics, fused movegen, make/undo, eval) |
| Eval | hand-crafted, Texel/jointly tuned (PSQT, king safety, threats, passers) |
| Search | PVS, aspiration, TT (4-slot buckets), NMP, RFP, futility, LMP, LMR, IIR, SEE, killers, counter-moves, history, 1-ply continuation history, check extension, mate-distance pruning |

Where the remaining ~330 Elo plausibly comes from (rough, overlapping estimates):

| Source | Expected | Confidence |
|---|---|---|
| Standard search features still missing (queue below) | +60..120 | medium: each is proven elsewhere, several failed here before for implementation reasons |
| NPS (eval is 60% of time) | +15..40 | high for the speed, Elo from speed is ~+50..70 per doubling at blitz |
| SPSA of search parameters | +15..40 | medium |
| Eval retune on far more (self-play) data | +10..30 | medium |
| **NNUE** (own trainer + inference, std only) | **+150..300** | high: the largest single jump every HCE engine at this level has made |

So: the queue and speed work keep us moving, but **reaching 3139 almost
certainly needs NNUE**. That is the long-term project and should start soon,
because its data generation can run on CI in the gaps.

## Short term (running now / this week)

**Merge queue (GitHub, unattended).** ~25 candidates: the new search ideas
(history pruning, IIR at cut-nodes, razoring, NMP tuning, TT cutoff guards),
KPK bitbase, and the 13 older branches re-applied on the new main (`-r2`), plus
infra that merges without an SPRT once bench is identical (`nnue-inference`,
`datagen-ci`). About a day of CI. Live order and results: the dashboard.
Two `needs-human` rebase conflicts were re-applied on 2026-11-02 and queued:
`endgame-scale-r3` (stc) and `nnue-inference-r2` (speedup, plus an eval-cache
bypass so an attached net actually drives search). Two backlog ideas were
added then: `ttcapture-lmr` and `aspiration-failhigh-depth`. `check-lmr` needs
no branch (LMR already skips checking moves) and `cutnode-lmr` has no
`cut_node` flag (`iir-cutnode` was rejected).

Five more rebase conflicts appeared after `singular-ext-r2` merged and were
re-applied on 2026-11-03: `nmp-tt-guard-r2`, `futility-moves-r3`,
`followup-history-r3`, `probcut-r3`, `extensions-r3` (the last reconciles the
pawn-push extension with the singular-extension `extension` variable instead
of shadowing it).

Done Nov 1-2: merged `lmr-priority-quiets` (+13), `history-gravity` (+22),
`improving-rfp` (+17), `see-quiets` (+9), `insufficient-material` (nonreg) and five
exact speedups; rejected `lmp-improving`, `tm-budget`, `qsearch-see`,
`capture-history`.

**Measurement.** Elite CCRL ladder on the new main (after the merges above).

**NNUE.** Inference, the trainer and the first net are merged. The first
datagen run produced 3,295,473 positions from 40k games at 5000 nodes in
~17 min per shard (794-1160 pos/s per runner); the net (10 epochs, lr 0.02,
33M samples) passed its first SPRT at **+51.2 Elo (95% CI +27.7..+75.3**,
608 games) once the stm-POV eval-cache sign bug was fixed (`nnue-eval-sign`).
The buggy first attempt lost all 608 games, so this is the real first result.
Next: a 65M-position run, retrain, and a second net (the usual bootstrap).

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

1. **CI throughput** (see process audit): build once per wave, use all runners,
   automate periodic checks. Cheap, and multiplies everything else.
2. **Land the speed branches.** Then the next structural speedups: incremental
   attack maps or per-piece attack caching shared by eval terms, staged move
   generation (captures first; skip quiet generation on TT/capture cutoffs),
   TT prefetch on x86 (std::arch).
3. **SPSA run on the search parameters** (infra exists: `spsa.yml`). Run once the
   queue's search features have landed, since they shift the optima. Then again
   after each large search change.
4. **Self-play data generation on CI** (`datagen.rs` exists). Gives (a) a much
   larger Texel retune set than the 445k positions we have, and (b) the NNUE
   training set. Target 100M+ positions. Run in the gaps when the queue is
   idle or with low priority. Built on branch `datagen-ci`: deterministic
   sharded `datagen.yml` (~65M positions per 20-shard run, estimated);
   see docs/nnue.md "Data generation".
5. **Search features not yet written** (each its own branch → queue):
   - 2-ply continuation history and history-based LMR (partly in `lmr-history-scale`)
   - NMP verification search at high depth, razoring
   - multi-cut / double extensions on top of a working singular extension
   - QS futility and better QS evasion ordering
   - TT cutoff refinements (no cutoffs at PV nodes, halfmove-clock guard)
   - correction history done properly (pawn + material keys), if `correction-history` fails again
6. **Draw and endgame knowledge**: insufficient material, KPK bitbase (generated
   at startup, std only), opposite-colour bishop scaling, fortress-ish scaling.
7. **Lichess bot live** on the Dokku box (needs the bot account and token from
   the user, set with `dokku config:set`). Gives a public, continuous rating.
8. **CCRL submission** (`docs/ccrl-submission.md`): needs a licence decision and
   a tagged release; their games are the most credible independent rating.

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
