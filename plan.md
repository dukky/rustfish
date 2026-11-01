# Rustfish roadmap

The single source of truth for what is running, what is next and why. Process
rules live in `docs/development.md`; results live in `tests/*.jsonl` and the
`queue-state` branch. Update this file whenever work starts, lands or dies.

_Last updated: 2026-11-01._

## Where we are

| | |
|---|---|
| Strength | **~2810 CCRL Blitz** (95% CI 2787-2833), main `859b255`, elite anchors, 120+1 |
| Target | Stockfish 17 skill 17 (UCI_Elo 3139, ~3140 CCRL Blitz): **~330 Elo short** |
| Speed | ~2.5M nps single thread on M3 (bench 650958 nodes) |
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

**Merge queue (GitHub, unattended).** 22 items, serial, each rebased onto the
latest `main` and SPRT'd; passes fast-forward `main` automatically. Order:

1. `lmr-priority-quiets` (in flight; passed twice before at +21), `history-gravity` (passed at +24)
2. `improving-rfp`, `lmp-improving`, `insufficient-material` (nonreg), `tm-budget` (LTC gainer)
3. `endgame-scale`, `see-quiets`, `correction-history`, `singular-ext`, `futility-moves`,
   `followup-history`, `probcut`, `see-pruning`, `qsearch-see`, `capture-history`,
   `aspiration-widen`, `extensions`, `lmr-pv-captures`, `lmr-history-scale`
4. `eval-mobility` (pre-dates the joint retune, likely conflicts), `pawn-hash` (speedup)

Most of group 3 showed big losses earlier, but those results were invalid (the
base had just gained +79 from the eval merge), so they get a clean retest.
At ~45-90 min per test this is about 1-2 days of CI.

**Speed work (laptop, 7 agents).** Exact speedups: same bench at two depths,
bit-identical eval on 50k positions, ≥2% median NPS in interleaved pairs.
Branches `speed-magics`, `speed-eval-king`, `speed-eval-body`, `speed-movegen`,
`speed-picker`, `speed-board`, `speed-eval-calls`. Survivors are combined and
queued as `speedup` at the front (no SPRT needed).

**Measurement.** Stockfish 17 skill 10/15/17 ladder (`anchors-stockfish.json`)
running, to cross-check the CCRL number against the actual target.

## Known problems (not fixed yet)

- **Stockfish 17 at skill 17 is measured only by formula**, not by games (the
  ladder above fixes this).
- **The anchors top out at 2956.** The next ladder needs 3000-3200 anchors
  (e.g. stronger Stash/Velvet/Reckless releases) or the estimate will drift.
- **SPRT bounds `[0, 10]` reject real +2..5 patches.** Correct for now (big
  gains still available), switch to `[0, 5]` once typical passes drop below +10.
- **Each SPRT shard builds both engines** (~2-3 min of LTO builds per shard per
  wave): ~15-20% of CI time is compiling, not playing. And 4 of the 20 runners
  sit idle during an SPRT (fixed next, see process).
- **Periodic checks are manual**: the LTC non-regression after ~5 merges and
  the CCRL ladder only happen when someone remembers.
- **Bot merges skip `ci.yml`** (GitHub token pushes don't trigger workflows);
  the queue runs the same checks itself, so this is cosmetic.
- **Queue logs live on `queue-state`**, not `tests/sprt-log.jsonl`; both are
  records now, the queue's is authoritative for queue tests.
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
   idle or with low priority.
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
| 1 | Build both engines once per test (plan job, `target-cpu=x86-64-v3`) and pass binaries to shards as artifacts | ~15-20% more games per CI hour; `-C target-cpu=native` binaries can't be shared across runner CPUs, v3 (AVX2/BMI2) can | done next |
| 2 | Use 19 shards for SPRTs while no ladder runs | 4 idle runners = +20% throughput | done next |
| 3 | Queue runs the periodic checks itself: LTC non-regression of `main` vs the `main` 5 merges back, and a weekly CCRL ladder (cron) | regression protection without anyone remembering | build |
| 4 | Queue retries infra failures once before `needs-human` | transient GitHub/network errors shouldn't drop a candidate | build |
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
- 2026-11-01: unattended merge queue.
