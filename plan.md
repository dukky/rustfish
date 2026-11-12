# Rustfish roadmap

The single source of truth for what is running, what is next and why. Process
rules live in `docs/development.md`; net experiments in
`docs/nnue-experiments.md`; results in `tests/*.jsonl`, `net/chain.json` and
the `queue-state` branch. Update this file whenever work starts, lands or dies.

_Last updated: 2026-11-11 23:30 (UK)._ Live state (current test, queue,
results, strength chart) is on the dashboard, **https://rustfish-dash.dev.duk.im**.

## Where we are

| | |
|---|---|
| Strength | **~3508 CCRL Blitz**: three ladders in a row at or above the top-100 line: 3510 (CI 3492-3528, 2026-11-09 04:16, g7kb8m), 3508.5 (2026-11-09 22:11), 3507.6 (CI 3490-3525, 2026-11-10 12:24, `1b3d81d`). Each CI still overlaps 3506 alone; together they put us at the line, not clearly past it. Before: 3492 (2026-11-08), 3474, 3463 (g4). |
| Goals | CCRL Blitz single-CPU list (best versions, list of 2026-11-07; `scripts/ccrl-targets.json`): ~~short term 3400~~ (2026-11-06); **mid term top 100 = 3506**: at the line (~3508), needs a clear margin; **long term top 10 = 3751**. Stockfish (3792) is out of scope. |
| Eval | NNUE **g7kb8m** (merged +30.6, 2026-11-08): (768 x 8 mirrored king buckets -> 512) x 2 -> 8 output buckets, SCReLU, i16; eval = net / 2. g7 from scratch on pool7 (1.72B positions), m1 copy, kb8m factoriser warm start. |
| Search | PVS, aspiration, TT buckets, NMP, RFP, futility, LMP, LMR (cut-node, not-improving, quiet checks reduced one ply less, root moves), IIR, SEE, killers, counter-moves, history + threat-aware history with escape bonus, 1-ply continuation history, ProbCut, singular extensions + multi-cut + negative extension, check extension; node-fraction time management. |
| Compute | **Desktop 9600X** (Zen 5, AVX-512, DDR5-6000, 180 GB free on WSL): main training + datagen machine from 2026-11-11; CPU trainer ~3x the M3; CUDA trainer for its 1080 Ti in progress (desktop session). **M3 laptop:** GPU (Metal) trainer for net-shape experiments on the fixed pool7. **CI (20 runners): testing only** (18 test shards, queue job 1, 1 idle background slot); CI datagen stopped 2026-11-11. |

**Self-play vs ladder, 2026-11-09..10:** feat-lmr-quiet-check (+12.9) and
feat-root-lmr (+12.2) merged between the 3510 and 3508 ladders, and the
ladder did not move. Both passed fast (2880 games) so their true gains are
smaller than their point estimates, and the ladder CI (+-18) can hide +10;
but it reinforces the rule: discount self-play gains ~0.3-0.5x, and nets
(+20-30 per generation) are the lever that has moved the ladder.

## Plan

**Short term (next ~24-48 h): gen 8 on the desktop, 1024-wide test.**
- pool7 (54.9 GB) copying laptop -> desktop over Tailscale (LAN, ~12.5 MB/s
  powerline, done ~00:15 2026-11-12). Gen 8 = pool7 + the CI seeds generated since +
  the desktop's 3 x 1M-game runs; same recipe (g8 -> g8m1 -> g8kb8m),
  gate vs g7kb8m.
- **g7h1024** (Metal, laptop, pool7, 10 epochs, done ~05:45 2026-11-12):
  epoch 1 val 0.026793 vs g7's 0.027082 at 2.45x the training cost. Compare
  best-scale on val.bin + val17x2 at the end; if it wins clearly, measure
  NPS cost and gate a 1024 net in games (stc, shape change).
- Queue: **feat-cuckoo merged +5.5** [1.1, 9.9] (2026-11-12 01:54);
  feat-conthist-rekey rejected (-0.2, 10.9k games: the [piece][to] re-key
  alone is neutral, so the 2/4-ply retry loses its premise; deprioritised);
  feat-qs-evasion-prune (+2 exp; if merged add `QsQuietEvasions` to
  docs/ccrl-submission.md and tune/spsa-params.json).
- Ladder after the next net merge.

**Mid term (days-weeks): clear top 100, start on bigger nets.**
- Data decision (Hermes notes `docs/research-2026-11-11-coda-lc0-compute.md`,
  critique `docs/research-2026-11-11-critique.md`): own self-play still gains
  +17-31 per generation, so keep it as arm A; arm B = the same positions
  rescored by g7kb8m at higher nodes (cheap, best-evidenced: Viridithas
  +50/+25/+24); arm C = Lc0 T80 subset (binpack reader, Q -> cp calibration,
  ODbL check first). One variable each, current net shape, gated in games.
- Net-shape experiments on the laptop GPU against fixed pool7 (results
  compare with g7/g7m1/g7kb8m directly): 1024-wide (running), threat
  inputs (parked at -22% NPS; re-check with AVX-512 at 1024), other
  bucket layouts. Winners are retrained on pool8+ on the desktop. Pool7
  results stay useful until the data changes kind or the desktop has
  spare training capacity (~1-2 weeks).
- SPSA of the search tunables (`spsa-nnue`) once the feature batch drains:
  the LMR, singular and threat terms all changed in a week.
- Search backlog: continuation history 2/4 ply (low priority: re-key was
  neutral), quiet-check
  and root LMR polish, qsearch polish, improving ply-4 fallback.
- Streaming shuffle (trainers read many source files through a shuffle
  buffer, no pool copy): halves disk; needed before pools outgrow 180 GB.
- PGO on the desktop x86: +5% on the M3, noise on CI x86; measure on the
  9600X (datagen games/s) before using it there.

**Long term: top 10 (3751).**
- Bigger nets (1024+ wide, king buckets, possibly a second hidden layer);
  AVX-512 inference paths for them (`nnue-avx512`).
- Multi-billion-position pools with opening diversity; better data
  (deeper/softnode search, rescoring, filtering).
- Novel ideas only offline and cheap until a clear wall: uncertainty head,
  hard-example weighting, two-net disagreement.
- CCRL submission (identity and docs ready; release and licence are the
  owner's call), Lichess bot, Syzygy.

## Status log

Dated log, newest last; superseded detail is in the commit history and
`docs/nnue-experiments.md` ("Conclusions to date").

**2026-11-07**
- simp-nmp-margin +8.2; g4 merged +26.7; g4kb8m merged +11.0 (king
  buckets via exact factoriser warm start; from scratch is data-limited);
  feat-lmr-cutnode +15.6; feat-se-negext-r2 +8.9; tm-ccrl-r2 nonreg.
  Rejected: scale hypothesis (g3f/g3fs/g4ws), correction-history-r2,
  g4l3 (lambda 0.3), feat-ttpv (-3.9). Ladder 3463.
- Infra: queue hardening, datagen bursts, settings-driven lanes,
  `queue.py remove`, ladder anchors to 3631 with pentanomial CIs, fast CPU
  trainer (2x, factoriser), Metal `--keep-best`.

**2026-11-08..10 (unattended)**
- Merged: simp2-qs-delta (nonreg, +9.8), feat-threat-hist +4.7 (17.9k
  games), **nnue-g7kb8m +30.6** (net-gate), feat-lmr-improving +5.1,
  feat-lmr-quiet-check +12.9, feat-root-lmr +12.2; fix-ownbook-default
  (`OwnBook` now advertises `default false`).
- Rejected: feat-lmr-captures -3.3, simp2-razor-d1 -3.2, simp2-rfp-linear
  -6.6; simp2-qs-failsoft removed by hand (slow, ~0). simp-lmr-extra-r2
  still needs a redo from current main (expect ~0, -15% nodes).
- Ladders: 3492 -> 3510 -> 3508.5 -> 3507.6.
- Gen 7 disk fix (`shuffle --consume`, 6f71956). Desktop 9600X set up
  (docs/desktop-9600x-setup.md): CPU trainer ~3x the M3.

**2026-11-11**
- CI datagen stopped (GitHub-hosted runners are for testing): queue
  lanes -> 18 test shards, bursts off (0d3af12). PGO tooling merged
  (77f0d75; opt-in, CI unchanged).
- Laptop disk audit; pool7 -> desktop copy; g7h1024 started on the M3 GPU.
- Hermes/DeepSeek notes on Coda, Lc0 data and compute, and the peer
  analysis (docs/peer-analysis-2026-11-08.md) read and critiqued.

## CI efficiency

- STC waves: ~90% of wall time is games. Net-gate waves cache engines
  across waves (`34091d5`).
- The largest waste was *what* was tested (~10 doomed nets, ~24 rejected
  search patches in two days): screen nets by validation loss first.
- CI is for testing only from 2026-11-11 (no bulk datagen or training on
  GitHub-hosted runners). Foreground = 19 - background_parallel (>= 1);
  allowing 0 would give 19 shards (small queue.py change).
- Owner decisions: GitHub Pro (20 -> 40 concurrent jobs) and a self-hosted
  runner on the 9600X (outside the job cap).
- Ladder cost: a 120+1 ladder is ~10 SPRTs of runner time; run it only after
  real gains. A 10+0.1 ladder is ~12x cheaper.
- PGO (2026-11-11, `scripts/build-pgo.sh`): identical bench/evalhash
  everywhere. M3: +4.8% bench NPS, +6% datagen games/s. GitHub x86-64-v3
  runners: inside the A/A noise (+-2%). CI stays plain; integration parked
  on `pgo-ci-integration`. `freeze.py --pgo` is opt-in.

## Known problems

- **Disk / data size.** Data grows with every generation and we keep
  sources + a shuffled pool (+ a copy while shuffling without `--consume`).
  Laptop: 59 GB free; keep pool7 for shape experiments, `auto/` (52 GB)
  can go once the desktop copy is verified. Desktop: 180 GB free on the
  250 GB WSL volume (~1 TB more on other drives). Fixes in order:
  streaming shuffle; recent-data window (drop seeds from much weaker
  engines; test as one variable); binpack-style compression (~2-3
  B/position) if still needed.
- **Anchors near the top.** The elite set reaches 3631 (11 anchors,
  3318-3631); add stronger anchors before the ladder passes ~3600.
- **SPRT bounds:** `stc` is [0, 5]; `net-gate` (5+0.05, [0, 10]) is for
  same-shape nets only; a 1024 net goes through `stc`.
- **Queue backlog:** mark a needs-human row resolved when its `-r2`
  merges; a scheduled ladder should replace (not duplicate) a pending
  manual one; a scheduled ltc-check shouldn't jump a `--front` ladder.
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

Division of labour (from 2026-11-11): GitHub Actions plays every test game
and the ladders; the desktop generates data and trains production nets; the
laptop GPU runs net-shape experiments and the laptop does
correctness/speed work; the Dokku box runs the dashboard and the Lichess
bot; the LLM writes candidates and keeps this plan current, not polling and
merging (the queue does that).

## History

- Jul 2026: pre-pipeline notes in `docs/archive/` (their rejections are void).
- 2026-10-29: depth-8 cap fixed, SPRT pipeline and CCRL ladder: ~2081.
- 2026-10-31: IIR, SEE fix, TT review, eval rework: ~2624 -> ~2810.
- 2026-11-01: unattended merge queue. 2026-11-02: 2889.
- 2026-11-03: first NNUE net (+51), singular extensions (+24), mobility (+36):
  ~3001 (refit). net2 +149.
- 2026-11-04: net3 +40, net4 +13, ProbCut +4: 3168 (refit) on net4.
- 2026-11-04..05: net5-net13 all rejected (see the audit); 512-wide -30.
- 2026-11-06: audit; trainer v2; 3429 on g3.
- 2026-11-07: g4, king buckets, LMR features: 3463.
- 2026-11-08..10: g7kb8m, LMR batch: 3508-3510 (top-100 line).
