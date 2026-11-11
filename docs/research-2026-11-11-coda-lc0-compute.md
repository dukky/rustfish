# Coda, LC0/T80 and compute — research notes for Claude

**Date:** 2026-11-11
**Author:** Hermes, for the next engine agent (Claude/Opus)
**Status:** research note only; not a plan of record. Verify live state before acting.
**Main at writing:** `614ea1b`; measured ladder **3508** (point estimate; CI straddles the 3506 top-100 line).

## How to use this

Read this before planning the next data/net step. It records what we learned from
Coda's development, what the self-play-vs-LC0 evidence actually says, what hardware
we have, and the operational decisions already taken. It deliberately separates
**evidence** from **inference**. Where it says "estimate", do not quote it as a
measurement.

## Executive summary

1. **Coda's strength came mainly from LC0/T80 data + a much bigger threat-aware
   net, not from a secret search feature.** At Coda 0.4 their engine was already a
   modern alpha-beta engine with a 63 MB, FT=1024, threat-aware net; it was roughly
   our current strength or a little above.
2. **Coda's production nets were trained on LC0 T80 binpacks, not self-play.** A
   3B-position pure self-play replacement lost **−20 Elo** against T80 at similar
   size. Their docs say **no self-play training** in production.
3. **Self-play is an amplifier, not a substitute for a stronger teacher.** LC0/T80
   is Leela's self-play labelled by a much stronger MCTS teacher. Replacing it with
   our own weaker self-play labels loses; mixing small fresh self-play, re-scoring
   with a stronger net, or relabelling with a wider teacher can help.
4. **Compute:** Coda has roughly 5 modern GPUs and a CPU/OpenBench fleet; we have a
   9600X, an M3 MacBook, an old 1080 Ti and CI. They are ~10–20× ahead on effective
   *training* throughput, but our CI testing throughput is broadly comparable.
5. **Do not use GitHub-hosted CI for bulk datagen or training.** It is a policy and
   engineering dead end; CI is for builds, SPRTs, ladders, gates and small smoke
   shards. Datagen is currently disabled on CI.
6. **Current data:** ~2.7B usable positions on CI + a few hundred million on the
   desktop ⇒ roughly **2.8–3.0B usable**, ~5B raw generated. Pool8 is not built.
7. **T80/LC0 is the highest-value next experiment**, not more self-play volume. It
   also happens to remove the need for the CI datagen farm.

## Coda version/strength timeline

Coda is a Rust rewrite of GoChess, developed almost entirely by Claude Code with
Adam Twiss directing, testing and reviewing. Public CCRL Blitz numbers (single CPU):

| Version | Date | CCRL Blitz | Notes |
|---|---|---|---|
| 0.9.0 | 6 Jul 2026 | not rated | README estimated ~3500–3600 local; actual inferred ~3712 |
| 0.9.2 | 16 Jul 2026 | **3734** | 1,422 games; same net/strength as 0.9.1 |
| 0.9.3 | 26 Jul 2026 | **3755** | only 32 games; provisional |
| 0.9.4 | 22 Aug 2026 | **3769** | 1,545 games; single-CPU; 8CPU 3793 |
| 0.9.5 | 21 Oct 2026 | not yet listed | post-0.9.4 changes |

Historical README estimates before 0.9:

- `v0.1.0-nets` (1 Apr): ~3000 Elo, based on Crafty/ExChess/GreKo.
- `v0.4.0-nets` (25 Apr): ~2900 Lichess, competes ~3500 CCRL.
- `v0.7.0-nets` (11 Jun): 2900–3000 Lichess, 3500–3600 CCRL.
- `v0.9.0` (6 Jul): 3000–3080 Lichess, 3500–3600 CCRL.

**Learning:** these README numbers were consistently low. 0.9.2 is 3734, and 0.9.1
was +22 over 0.9.0, so 0.9.0 was probably ~3712. The "3500–3600" pre-0.9 estimate
was an unanchored local-pool guess, not a CCRL measurement. Do not use it as the
starting point for "Coda went from our strength to 3769".

## Code/feature comparison

Approximate source sizes from the tagged tarballs:

- **Coda 0.4:** 28.0k lines, 151 tests, 29 files; ~63 MB net.
- **Coda 0.7:** 37.8k lines, 194 tests, 52 files; ~63 MB net.
- **Coda 0.9.0:** 44.7k lines, 229 tests, 56 files; ~84 MB net.
- **rustfish:** 11.4k lines, 46 tests, 14 files; ~6.1 MB net.

Coda 0.4 already had the full modern search battery: correction history, singular
extensions, ProbCut, SEE pruning, LMR/LMP, cuckoo repetition detection,
threat-aware history, aspiration, Syzygy, Polyglot, Lazy SMP and self-play datagen.
0.7 adds more search/TM tuning, SIMD (`nnue_simd.rs`, `setwise.rs`), audits and
correctness fixes. 0.9 adds a thread pool, `threats_splat`, more search/TM polish and
the L1=32 / 84 MB net.

**Learning:** the code gap is real but narrower than the line count suggests. Our
search core is already close; the missing strength is concentrated in:

- eval representation: explicit threat/x-ray features, more king buckets, wider
  accumulator;
- data/training: LC0/T80 labels, Bullet GPU trainer, multistage/SWA/SPSA;
- a few subsystems: correction history, cuckoo, Syzygy/Polyglot.

## Peer engine data and the current 3500 band

The full peer deep dive is in `docs/peer-analysis-2026-11-08.md`; this is the part
that matters for the data/net decision. Ratings are CCRL Blitz single-CPU anchors
(7 Nov 2026), not necessarily each project's newest release.

- **Panda 2.0 — 3523.** NNUE `(768 -> 512) x 2 -> 1 x 8`, trained on self-play
  with Bullet; ~31 Elo above us with the **same net shape**. This is the strongest
  argument that our architecture is not the ceiling.
- **Carp 3.0.1 — 3525.** Simple Rust NNUE, didactic engine; proof that a clean,
  small net can sit above 3500.
- **Endspiel 2.0 — 3505.** Wider 1536 feature transformer, king buckets, Bullet,
  **Lc0 T80 score labels** plus self-play, Syzygy. AI-assisted, GPL, accepted
  normally by CCRL.
- **Lambergar 1.5 — 3506.** Threat-aware quiet history, self-play datagen, PyTorch;
  the search-side version of the threat idea.
- **Grail 3.0 — 3490.** ~500M self-play positions, 12 king regions; a fully
  integrated datagen/CUDA/Metal pipeline.
- **Leorik 3.2 — 3489.** `(768x5 -> 640) x 2 -> 1x8`, **6B labelled positions** with
  Bullet; bigger data does not automatically beat a well-tuned smaller net.
- **Koivisto 6/9 — 3487/3631.** Billions of positions, own trainer lineage; mature
  search + data.
- **Reckless 0.7 — 3491.** 384-wide net, Bullet; 0.8-dev moved to threat features
  and king buckets.
- **Altair 7 — 3579.** 3.1B FENs, bootstrapped from HCE then recycled; data volume
  with a mature search.
- **Viridithas 20 — current Rust leader.** Own self-play, continual retuning;
  structurally the closest long-running Rust analogue to rustfish.
- **Stash 37 — 3417.** HCE-only; shows the ceiling of a well-tuned classical eval.

### What the peers say

1. **The net shape is not the gap.** Panda uses our exact shape at 3523; Carp is
   simpler at 3525. Data quality, labels and tuning decide the last 20–40 Elo.
2. **Data volume is not linear.** Leorik has 6B at 3489; Grail has ~500M at 3490;
   we have ~1.7B at the last ladder. More positions help, but labels/recency/search
   decide.
3. **Threat features are the common next step.** Endspiel, Reckless 0.8 and
   Lambergar all move in that direction; our search-side threat history was +4.7,
   the net-side threat inputs are the larger version.
4. **Engineering polish is secondary at 1-CPU Blitz.** Syzygy/Ponder/MultiPV/SMP
   are product features, worth only a few Elo each on the CCRL list.
5. **AI assistance is not disqualifying; provenance is.** Endspiel is tested and
   discussed normally. Coda was the flashpoint because of alleged AGPL-derived
   code and licence mismatch, not because it used LC0/T80 data.

## Coda's development process, 0.4 → 0.7

Between the 0.4 and 0.7 net tags: **507 commits, 158 files changed**, six weeks.
Commit mix: ~51 search, 35 experiment logs, 31 docs, 12 fixes, 10 scripts, 9 TM,
6 audits, 4 NNUE.

The loop was:

1. Hypothesis + expected Elo in an experiment ledger.
2. SPRT on OpenBench at STC/LTC with a fixed opening book.
3. If it passes, SPSA-tune the affected parameters; retune after net changes.
4. For nets: train on GPU with Bullet, validate, net-vs-net SPRT, then promote.
5. Audit correctness/performance; document the result; repeat.

Multiple Claude instances worked in parallel on different hosts; Adam reviewed and
directed. This is a research-lab process, not a solo engine project.

Net promotions in the window:

- `1EF1C3E5` (v0.4.0-nets): +4.9 package.
- `E2773E50` (v0.5.0-nets): SB1200 + factoriser + extra data + SWA; net +12.3,
  retune +7.3, package +13.1.
- `E4B66CE4` (v0.6.0-nets): multi-v6 stage 4; +19.4 raw / +12.4 SWA.
- `549C20A5` (v0.7.0-nets): multi-v6 stage 5 extended terminal; +7.9 bundle.
- `E161C665` (0.9.0): multi-v9 stage 3, SWA, WDL 0.20; +4.8 net-vs-net.

**Learning:** each generation was a data/recipe change with a controlled net-vs-net
gate; search work was incremental. The largest jumps were not one clever search
patch.

## What Coda trained on

**LC0 T80 binpacks — the same data Stockfish uses.** From Coda's docs:

- `test80-2024-01-jan-2tb7p.min-v2.v6.binpack` and other monthly T80 files.
- LC0 MCTS scores stored in the binpack; ~800 nodes / depth-20-equivalent labels.
- Filtering: quiet-only, ply filters, score caps; roughly 37% of raw positions
  survive into the training pool.
- A 3B-position pure self-play replacement lost **−20 Elo** against T80 at similar
  size. Their conclusion: stay on T80; mix small fresh self-play or re-score the
  corpus with a stronger net, don't replace it.
- 0.9 later mined T80 for "blindspot" positions where Coda's static eval was far
  from LC0, and relabelled with LC0/SF. GPU3/GPU4 held the full SF training set.
- 0.9.4 switched to **BT4-scored LC0 data** — still LC0, better scoring.

Training recipe: customised **Bullet** CUDA trainer; SB200/SB800/SB1200 multistage;
WDL 0.15–0.20; factoriser; 30-superbatch warmup; low final LR; SWA averaging; SPSA
retune after each net.

## What the self-play-vs-LC0 evidence says

LC0/T80 is **self-play data too** — Leela's, labelled by a stronger MCTS teacher.
The useful question is "whose self-play, with how strong a teacher, at what compute?"

Evidence from Coda's own survey of 17 engines:

- **Pure replacement loses.** Coda 3B self-play vs T80: −20. Viridithas 450M fresh
  self-play replacing the corpus: −16.4 with shuffle, −68.4 without. Viridithas
  320K fresh mixed into the corpus: +7.6. 450M RL positions added to a 1.35B corpus:
  +6.95.
- **Re-scoring with a stronger net works.** Viridithas re-evaluated its corpus with
  a stronger net: +50, +25, +24 at different cycles; then +4–8 per cycle.
- **Bigger-teacher relabelling works.** Tarnished generated 22B self-play positions
  but relabelled scores with a 4096-FT teacher; that is the productive version.
- **Pure RL from random works only at scale.** Stormphrax/Halogen did 70+ iterations
  from random; not a quick replacement for a strong corpus.
- **Top T80 engines:** Stockfish, Reckless, Obsidian, Alexandria, Arasan, Coda.
  Viridithas is hybrid; PlentyChess self-play + self-distillation.

Mechanism: self-play is **policy improvement** only when the search/labels are
stronger than the raw net (AlphaZero/expert iteration). If you label with the same
weak net, you amplify its errors. Distillation from a stronger teacher is more
sample-efficient at our compute scale.

**Learning for rustfish:** at 3508, our self-play labels are weaker than LC0's
teacher. More self-play generations mostly teach the net to imitate itself. Use
T80/LC0 as the bulk teacher; mix small fresh self-play for coverage; if we generate
self-play, label it with high-node search or a wider teacher.

## Hardware: Coda vs us

Coda's known hardware:

- **Dev/fleet CPUs:** AMD EPYC 7351P, 16C/32T, Zen 1, AVX2 only (Atlas/Titan);
  AMD Ryzen 7 9700X, 8C/16T, Zen 5, AVX-512/VNNI (Zeus).
- **Primary dev laptop:** Apple M5 MacBook (ARM first-class).
- **GPUs:** RTX 5070, 5070 Ti, 5080, 4090, 4070S across multiple GPU hosts.
- Training costs reported: S200 ≈ 3–4 h on a 5070 Ti; SB800 ≈ 30–40 h.

Our hardware:

- **9600X:** 6C/12T Zen 5; trainer measured **4.08M pos/s plain / 1.98M
  factorised** at 12 threads; 6 threads gives ~96%/92% of that. CPU trainer is
  memory-bound (SMT adds only 4–8%).
- **M3 MacBook:** Metal trainer ~1.79M plain / 1.32M factorised; CPU ~1.6M.
- **1080 Ti:** ~11.3 TFLOPS FP32, 484 GB/s, no BF16/TF32. Useful as an extra lane,
  not a Coda-fleet replacement.
- **iGPU on 9600X:** 2 CUs, ~0.55 TFLOPS, shares DDR5 bandwidth; no ROCm support.
  Display only, not a training engine.
- **CI:** 20 × 4 vCPU = ~80 vCPU; AVX2 is the safe target; AVX-512 is not guaranteed.
  6-hour job cap, no GPU, ephemeral disk, no distributed training.

**Memory bandwidth:** 9600X DDR5-6000 dual channel ≈96 GB/s theoretical; base M3
≈100 GB/s. Both are in the same class; CPU+GPU concurrency on either shares the
same memory pipe. The 1080 Ti's 484 GB/s dedicated VRAM is why it can add a real
lane.

**Training-time estimates:**

- Current 6 MB net: ~2 h per 1.7B generation on the 9600X (measured throughput
  applied to the recipe; one real 1.17B desktop net was trained and validated).
- Coda 0.4/0.7-class 63 MB net: ~20–40 h per generation on the 9600X; ~12–30 h on
  one modern GPU.
- Coda 0.9-class 84 MB net: ~1.5–2.5 days per generation on the 9600X.
- CI cannot train these; MapReduce over GitHub-hosted runners is impractical
  (artifact latency, 6 h cap, no GPU) and likely a policy violation.

**Compute verdict:** Coda is ~10–20× ahead in effective training throughput; our CI
is broadly comparable for SPRT/testing. 3650 is realistic on our hardware *if* we
switch to T80/LC0 and build a Coda-0.4-class threat net; 3700 is a modern-GPU target,
not a 9600X+CI guarantee.

## GitHub Actions policy and datagen

**State:** `Datagen` workflow is **disabled manually**; seeds 168 and 169 were
cancelled. Queue `current: null`, `pending: []`. See
`origin/notes/hermes-ci-datagen-pause:docs/notes-2026-11-10-ci-datagen-pause.md`.

Why:

- GitHub's additional terms restrict hosted runners to production, testing,
  deployment and publication, and prohibit disproportionate compute.
- A continuous 26-shard self-play datagen lane is bulk compute, not testing.
- We are pivoting to T80/LC0, so bulk self-play is no longer the critical path.
- CI is more valuable as a test farm.

Allowed on CI: builds, SPRTs, ladders, SPSA, release checks, small occasional smoke
shards. Not allowed: bulk datagen or training.

## Current rustfish data

- Pool7 / g7 training pool: **1.72B usable positions**.
- CI datagen after pool7 through seed 167: ~2.72B raw, ~1.0B usable at the ~37%
  filter rate.
- Desktop: a few hundred million raw; ~75–150M usable if raw.
- **Total usable: ~2.8–3.0B**; raw generated ~5B.
- **Pool8 has not been built.** Seeds 168/169 were cancelled; their partial output
  should be ignored.
- Exact pool lives on the laptop/desktop under `~/rustfish-data`; this VPS only has
  CI artifacts/logs.

## Lichess bot (operational note)

- Lichess now caps **bot-vs-bot games at 100/day** (`modules/bot/src/main/BotLimit.scala`).
  The `/api/challenge` 429 is intentional, not a crash.
- The bot had ~85 bot-vs-bot games on 10 Nov plus rejected challenge attempts; it
  hit the limiter and stopped.
- Config deployed: `challenge_timeout` raised **5 → 30 minutes**; 3 concurrent games.
- A 30-minute cooldown yields roughly 40–50 games/day, not 100. To approach 100,
  use **2 concurrent games + 10–15 minute timeout**. Three concurrent just reaches
  the limiter faster and produces more rejected challenges.
- Human challenges are not subject to the bot-vs-bot cap.

## Recommended next steps (priority order)

1. **Keep CI datagen disabled.** Do not restart it for self-play volume; the policy
   risk is not worth it.
2. **Build a T80/LC0 ingest path.** Start with a 1–5B-position subset; quiet-only,
   score-capped, held-out validation split. Confirm disk/bandwidth before a full
   corpus.
3. **Train the current 6 MB net on T80** as the cheap proof. Gate it against main.
   Do not change architecture at the same time.
4. **If T80 helps, build the Coda-0.4-class net:** FT=1024, 10–16 king buckets,
   L1=16 (then 32), explicit threat features, material-bucketed output. That is the
   3650 path.
5. **Use the hardware as separate lanes:** 9600X trainer, Mac Metal trainer, 1080 Ti
   if the CUDA trainer supports Pascal; CI for gates. Do not try to combine CPU+iGPU
   on one net.
6. **Lichess:** keep 2 concurrent; choose 10–15 min for max games or 30 min for a
   quieter bot.
7. **Update the plan/docs** with the T80 decision and the CI datagen policy; record
   measured results, not forecasts.

## Open questions / risks

- **Licence:** T80 is ODbL-1.0/DBCL-1.0. Attribution required; share-alike attaches
  to database redistribution. Coda relies on the "Produced Work" reading for nets.
  Get the notice/legal position right before a release. This note is not legal advice.
- **Format:** T80 is SF binpack, often monthly multi-GB/TB shards. Our trainer wants
  32-byte records; need a converter/streaming reader and a filtered subset.
- **Architecture:** our current net is 768 PSQ inputs, no threats. T80 labels help,
  but the full gain needs threat features and more capacity.
- **Data coverage:** T80 may not cover positions our engine reaches; keep a small
  self-play/blindspot lane, not a bulk farm.
- **Compute:** a 63–84 MB net on the 9600X is 1.5–3 days per generation; a rented
  modern GPU is the difference between weeks and months.
- **Coda estimates:** do not quote Coda's old local-pool estimates as CCRL; use the
  measured 0.9.2/0.9.4 numbers and label inferences.

## Sources

- Coda repo: https://github.com/adamtwiss/coda
- Coda tags/tarballs used: `v0.4.0-nets` (`a89c0146`), `v0.7.0-nets`
  (`c0c0ed78`), `v0.9.0` (`95c79cd4`).
- Coda docs: `selfplay_data_strategy_2026-04-30.md`,
  `training_methodology_cross_engine_2026-05-01.md`, `t80_data_analysis.md`,
  `sf_recipe_experiments_2026-05-22.md`, `net_catalog.md`.
- Coda releases: 0.9.0–0.9.5; CCRL Blitz/40-15 lists.
- Lichess bot cap: https://github.com/lichess-org/lila/commit/eeefdcf48bc2b8b2a740b2254a8e4c589ff10802
- Local: `docs/desktop-9600x-setup.md`, `docs/handoff-2026-11-08.md`,
  `docs/notes-2026-11-10-ci-datagen-pause.md`, `net/chain.json`,
  `net/experiments.jsonl`.
