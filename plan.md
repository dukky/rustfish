# Rustfish roadmap

The single source of truth for what is running, what is next and why. Process
rules live in `docs/development.md`; net experiments in
`docs/nnue-experiments.md`; results in `tests/*.jsonl`, `net/chain.json` and
the `queue-state` branch. Update this file whenever work starts, lands or dies.

_Last updated: 2026-11-07 11:35._ Live state (current test, queue, results, strength
chart) is on the dashboard, **https://rustfish-dash.dev.duk.im**.

## Where we are

| | |
|---|---|
| Strength | **3463 CCRL Blitz** (95% CI 3434-3493) on `0aaf3b9` (g4), ladder 2026-11-07 11:23. Short-term goal (3400) passed on 2026-11-06 (3429). Was 3168 on the morning of 2026-11-06. |
| Goals | CCRL Blitz single-CPU list (best versions, 2026-10-31; `scripts/ccrl-targets.json`): ~~short term 3400 (#131)~~ reached; **mid term top 100 = 3504** (41 to go); **long term top 10 = 3751**. Stockfish (3792) is out of scope. |
| Eval | NNUE g4: (768 -> 512) x 2 -> 8 output buckets, SCReLU, i16; eval = net / 2. Trained from scratch on pool4 (730M positions) with the Metal GPU trainer. |
| Search | PVS, aspiration, TT buckets, NMP (eval >= beta), RFP, futility, LMP, LMR, IIR, SEE, killers, counter-moves, history, 1-ply continuation history, ProbCut, singular extensions + multi-cut, check extension; node-fraction time management. |
| Compute | M3 laptop: GPU training (~1.8M pos/s at 512x8), datagen when the GPU is idle. CI (20 runners): 13 datagen / 6 test (trial), queue job 1. |

## Plan

**Short term (next ~24 h): ship what is queued, prove data, next generation.**
- Queue: g4l3 (lambda 0.3) net gate, then feature tests (cut-node LMR,
  threat history, singular negative extension, ttPv, LMR improving, capture
  LMR, qsearch fail-soft) and simplification nonregs, ordered by expected Elo
  per runner-hour.
- Data-heavy runner split (13/6) until gen 5 proves it: gen 5 from scratch
  at ~1.0B positions (~15:00), gated ~17:00. Revert to 6/13 if gen 5 < +8.
- King buckets: exact factoriser warm starts from g4 beat the plain
  fine-tune control: kb8m (via a mirrored copy g4m1) val.bin 0.025943 vs
  0.026086, val17x2 0.025316 vs 0.025387; kb4 smaller. Both queued (stc,
  kb8m expect 8). Factoriser/keep-best being ported to the fast CPU trainer
  (`trainer-kbf-fast`). Lambda 0.3 rejected (-8.8, inconclusive).
- feat-lmr-cutnode: +15.6 [6.4, 24.9] at stc.
- Ladder after each 2-3 merges or a net merge.

**Mid term (days): top 100 (3504).**
- Net generations every ~1/3 more data (gen 6 tomorrow); drop the oldest,
  weakest data once the pool is large; king buckets when the data supports
  them (~3-4B positions if the gap only halves per doubling, sooner if the
  warm start works).
- SPSA of the ~30 search tunables after the feature batch lands (the new
  LMR/singular/threat terms interact).
- Search backlog: continuation history re-keyed to [piece][to][piece][to]
  then 2/4 ply, cuckoo upcoming-repetition, root LMR, quiet-check LMR,
  qsearch polish.
- Second trainer: 9600X. CPU trainer ~2x faster since dd12f29 (Zen 5
  runner 1.69M pos/s on 4 vCPU; M3 1.6M at 8 threads when quiet ~ the Metal
  trainer); projected ~4M pos/s on the 9600X (12 threads, unmeasured), so it
  may take generations while the M3 GPU does experiments
  while the M3 GPU does generations; 1080 Ti (CUDA port) only as a last
  resort (power draw).

**Long term: top 10 (3751).**
- Bigger nets (768-1024 wide, king buckets, possibly a second hidden layer)
  once data and trainer throughput allow; AVX-512 paths for them
  (`nnue-avx512`).
- Multi-billion-position pools with opening diversity; better data
  (deeper/softnode search, filtering).
- Novel ideas only offline and cheap until a clear wall: uncertainty head,
  hard-example weighting, two-net disagreement.
- CCRL submission (identity and docs ready; release and licence are the
  owner's call), Lichess bot, Syzygy.

## Status 2026-11-07 08:00

- **Ladder 3429 CCRL Blitz (CI 3401-3459)** on main after g3: past the 3400
  short-term target (~#122). Next: top 100 = 3504.
- Merged 2026-11-06: E1 (+152 net gate), eval scale 1/2 (+123 stc), E2 512x1
  (+26 stc), **g3 512x8 gen-3 (+81.5 net gate)**, speedups (nnue-speed,
  speed-r3, datagen-r2), king-bucket support (off), datagen net-scale fix,
  OwnBook off by default. Rejected: E3, scale 3/8 and 5/8, bigbang-r2.
  TM node-fraction (tm-ccrl): +5.9 [0.9, 10.9] at 60+0.6 but hit the wave
  cap; queued as nonreg at stc to merge on the LTC evidence.
- **Data bug fixed:** after the eval halving, datagen wrote scores at half
  the net scale (seeds 14-18, all local data). Fixed in code; corrected copies
  in ~/rustfish-data/fix and auto/. `nnue-trainer scalecheck` catches it.
- correction-history-r2 rejected under NNUE: +0.1 [-3.4, 3.7] over 13.8k stc
  games (was +4 under the HCE). Testing now: g4w (warm start from g3f on new
  data, best on the newest val sets), then g3f (g3 on corrected data, better on
  all val sets), then tm-ccrl nonreg.
- **g4w rejected at the net gate, -21.7 [-38, -5] (768 games).** Hypothesis
  (not yet a verdict): eval scale. Corrected data has labels ~1.13x the net's
  evals, so each net trained on it grows its output: mean|eval| on val17x2 is
  g3 444, g3f 482 (+9%), g4w 513 (+16%), and a +25% scale was -22 earlier.
  Validation loss rewards matching the labels' scale, so "better val" partly
  measured scale: rescaled to g3's mean|eval|, g3fs is worse than g3 on
  val17x2 (0.02666 vs 0.02649). Competing explanation for g4w: overfitting
  to 44M recent positions (its val.bin got worse). Discriminating gates
  queued: g3f (as is) vs g3fs (same net, output scaled x0.92), and g4ws
  (x0.865). **Result:** g3f +1.4 [-9.4, 12.2], g3fs -8.9 [-20.1, 2.2]
  (1792 games each). Shrinking the scale did not help: the scale hypothesis
  is not supported at +9% (weak evidence, overlapping CIs). Leading
  explanation for g4w is now overfitting of the short warm start; g4ws still
  queued. Validation loss (even at fixed scale) has not predicted gate Elo
  for g3f: games remain the gate.
- Scale review (subagent): with the scale removed (`nnue-trainer eval
  --best-scale`), g3f ~ g3 on val16x2/val17x2, slightly better on val.bin
  (matches +1.4); g4w best on recent sets, clearly worst on val.bin
  (specialised warm start). The x2 sets' labels reward ~1.2x wider nets, so
  raw loss there partly measures scale. **Screen on best-scale loss on
  val.bin plus a recent set.** Tooling merged (6b9b709): netloop auto,
  scalecheck, `rescale` (exact factor, slope or mean|eval| matching), `eval
  --best-scale`. Per-net NNUE_EVAL_DIV retuning not justified unless a net
  drifts 20%+.
- **Laptop:** Metal GPU trainer (~3x CPU; king buckets ~1.3x one bucket) does
  all training. `scripts/netloop.py auto` runs unattended: ingests CI and local
  datagen, scale-checks, warm-starts a candidate per ~30M new positions, screens
  on validation, embeds and queues a net gate. GPU training throttles CPU
  datagen 4-5x on this fanless M3; CI datagen carries data meanwhile.
- **CI:** smarter queue (af32031): speedups merge without waiting; priority by
  expected Elo per runner-hour (`--expect`); datagen calls the queue back.
  Foreground 16 runners, background datagen 3.
- **King buckets:** kb8m+factoriser from scratch with g3f's schedule loses to
  g3f by ~6e-4 on val.bin, val16x2 and val17x2 (0.02678 vs 0.02617); val flat
  from epoch 2 while train keeps falling (memorisation; note wd was 0). Not
  promoted. Running the learning curve (chain7: 50% of games, plain vs kbf):
  data-limited only if the gap is clearly larger at 50%. Also untried: wd > 0,
  lambda 0.3, fewer epochs.
- **Threat inputs** (branch `threat-inputs`, off by default, engine + both
  trainers, verified): `ep` (en-prise block) -1.0% val loss at H128 but -22%
  NPS; `ad` no gain, -41% NPS. Parked until the update path is cheaper.
- **Overnight (02:30-07:40 lost to the session limit):** simp-nmp-margin
  merged **+8.2 [1.1, 15.4]** (null move needs eval >= beta), simp-lmr-goodhist
  merged (neutral); simp-castle-order rejected (-1.4). g4ws (g4w at g3 scale)
  -22.7: g4w's loss was over-specialisation of the short warm start, not
  scale. King-bucket learning curve: gap 0.0011 at 50% of games vs 0.0006 at
  100%, overfits from epoch 1 -> data-limited; revisit at several times the
  pool. **Gen 4 from scratch** on pool4 (677M: pool3f + s17x2, L1003, CI
  s18-24) training (chain8), gate ~10:30. Short warm starts on new data
  (auto loop, g4w) don't win *with that recipe* (half new / half replay,
  lr 2e-4, 3 epochs, from a non-champion base, raw-loss screen): open
  question, not rejected. After g4: 2-3 x 30 min fine-tunes of the champion
  varying one thing each (80% replay; lower lr / fewer steps), screened on
  best-scale loss on val.bin + val17x2; gate any that win both.
- **g4 merged at the net gate: +26.7 [10.1, 43.5]** (from scratch on pool4,
  730M, +35% data vs g3, same recipe). Data pays: background datagen lane
  3 -> 6 runners (18 shards, ~25M/h), foreground 16 -> 13 shards (0fdb630);
  datagen bursts (1685c16) fill low-value gaps. Ladder next in the queue.
  Local datagen restarted on g4 (seed 1005). GPU: g4l3 (lambda 0.3), then
  chain9: g4kb4 (kb4 exact factoriser warm start from g4, keep-best) vs
  g4ft (same fine-tune, no kb) ~11:40. kb8m can't start exactly from an
  unmirrored net (needs an m1 copy first).
- **Runner split trial (10:45, f5379f8):** 13 datagen / 6 test runners
  (`lanes` in queue-settings.json), ~58M positions/h. Gen 5 when the pool is
  ~+35% (~1.0B, ~15:00). **Stop rule:** gen 5 < +8 at its gate -> back to
  6/13 (background_parallel 6, mode shards 13, burst 13/13, ladder
  max-parallel 13). Feature tests take ~2x longer meanwhile. AVX-512 NNUE
  paths (`nnue-avx512`, not merged): bit-identical everywhere, kernels +30-90%
  but whole-engine NPS / datagen games/s within +-3% on Zen 4/5 and Xeon
  runners (NNUE is not the bottleneck at H=512). Fleet: ~1/2-2/3 of runners
  have AVX-512 (EPYC 7763 Zen 3 no; 9V74, 9V45, Xeon 6973P/8573C yes).
  Re-run nnue-isa-check (branch nnue-avx512) when a net with H >= 768 or
  king buckets is about to embed; merge as a speedup if datagen games/s rises.
- **Local datagen** (09:30): `~/rustfish-data/datagen-gate.sh` runs seed 1004
  (main 7d1fc8a, net scale, 7 threads, ~1300 pos/s = ~4.7M/h) only while no
  GPU training runs (SIGSTOP/SIGCONT); the auto loop ingests local/L1004.txt.
- **King buckets next steps** (GPU idle slots, after plain generations):
  exact factoriser warm start from g4 (factor = g4, buckets = 0, low lr, best
  epoch), kb4, real wd > 0, 2-3 epochs. ~3-4B positions needed if the gap
  only halves per doubling.
- Queue (09:30): ltc-check, g4 net gate, ladder, then 7 feat + 4 simp tests,
  ~15 h. Estimates: ladder ~3440-3445 at ~11:45; ~3450-3460 when drained.
- Round 2: `simp2-*` (HCE-era margins) in progress (subagent).
  Queued: tm-ccrl-r2, simp-lmr-extra-r2, feat-lmr-cutnode (cut/PV node LMR,
  stc, expect 8), feat-se-negext (singular negative extension, stc, expect 5).
- Search gap backlog (audit 2026-11-07, Elo guesses at our level, next after
  the two feat branches): threat-aware quiet history + escape bonus (+8-15);
  ttPv flag in TT for LMR/pruning (+5-10); !improving in LMR (+3-6); LMR for
  late/bad captures (+3-6); re-key continuation history to
  [piece][to][piece][to] before retrying 2/4-ply (+3-8); LMR re-search depth
  adjust (+2-5); cuckoo upcoming-repetition (+2-4); reduce (not skip) LMR on
  quiet checks, root LMR, qsearch polish, improving ply-4 fallback (+1-3 each).
  New options still to add to docs/ccrl-submission.md after merges.
- Simplification pass: four `simp-*` branches queued as nonreg (castle
  ordering bonus, LMR good-history double count, NMP eval margin, LMR extra
  reduction).
- Next: SPSA of 22 search params (`spsa-nnue`) when the queue is quiet.

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

- **Anchors near the top.** The elite set tops out at reckless-0.7.0 (~3491);
  add 3500-3650 anchors before the ladder passes ~3480, or the fit
  extrapolates. Check each new anchor against its neighbours.
- **`evalhash` covers the HCE only** (no net attached), so for NNUE code the
  speedup gate relies on bench identity. Make it hash NNUE evals too.
- **SPRT bounds:** `stc` is [0, 5]; `net-gate` (5+0.05, [0, 10], 4 waves) is
  for same-shape nets only.
- **Bot merges skip `ci.yml`**; the queue runs the same checks itself.
- **Time management** is tested at 60+0.6 and 10+0.1; CCRL also plays 40/15.
- **Lazy SMP is untested for strength** (bot and 4-CPU lists).

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

- 2026-11-07: removed the 2026-11-06 "train the net properly" strategy and
  "next 24 hours" tables (done: E1-E3, generation loop, GPU trainer); see git history.

- Jul 2026: pre-pipeline notes in `docs/archive/` (their rejections are void).
- 2026-10-29: depth-8 cap fixed, SPRT pipeline and CCRL ladder: ~2081.
- 2026-10-31: IIR, SEE fix, TT review, eval rework: ~2624 -> ~2810.
- 2026-11-01: unattended merge queue. 2026-11-02: 2889.
- 2026-11-03: first NNUE net (+51), singular extensions (+24), mobility (+36):
  ~3001 (refit). net2 +149.
- 2026-11-04: net3 +40, net4 +13, ProbCut +4: 3168 (refit) on net4.
- 2026-11-04..05: net5-net13 all rejected (see the audit); 512-wide -30.
- 2026-11-06: audit; trainer v2; black-marlin anchor removed.
