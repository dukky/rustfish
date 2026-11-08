# Rustfish roadmap

The single source of truth for what is running, what is next and why. Process
rules live in `docs/development.md`; net experiments in
`docs/nnue-experiments.md`; results in `tests/*.jsonl`, `net/chain.json` and
the `queue-state` branch. Update this file whenever work starts, lands or dies.

_Last updated: 2026-11-07 18:35 (UK time)._ Live state (current test, queue, results, strength
chart) is on the dashboard, **https://rustfish-dash.dev.duk.im**.

## Where we are

| | |
|---|---|
| Strength | **3474 CCRL Blitz** (pentanomial CI 3456-3493, 2026-11-08 01:38, new anchor set); was **3463** (95% CI 3434-3493) on `0aaf3b9` (g4), ladder 2026-11-07 11:23. Since then merged: feat-lmr-cutnode +15.6, g4kb8m +11.0, simp-nmp-margin +8.2, tm-ccrl-r2 (nonreg): **estimate ~3495**. Short-term goal (3400) passed on 2026-11-06 (3429). Was 3168 on the morning of 2026-11-06. |
| Goals | CCRL Blitz single-CPU list (best versions, list of 2026-11-07; `scripts/ccrl-targets.json`): ~~short term 3400~~ reached; **mid term top 100 = 3506** (#100 Endspiel 3505); **long term top 10 = 3751**. Stockfish (3792) is out of scope. |
| Eval | NNUE g4kb8m: (768 x 8 mirrored king buckets -> 512) x 2 -> 8 output buckets, SCReLU, i16; eval = net / 2. Exact factoriser warm start from g4 (from scratch on pool4, 730M positions). |
| Search | PVS, aspiration, TT buckets, NMP (eval >= beta), RFP, futility, LMP, LMR, IIR, SEE, killers, counter-moves, history, 1-ply continuation history, ProbCut, singular extensions + multi-cut, check extension, cut-node LMR; node-fraction time management. |
| Compute | M3 laptop: GPU training (~1.8M pos/s at 512x8), datagen when the GPU is idle. CI (20 runners): 13 datagen / 6 test (trial), queue job 1. |

**Ladder 2026-11-08 01:38 (e42fe7a, new anchor set 3318-3631, 40 games each):
3474 [3456, 3493] pentanomial** (binomial [3441, 3508]). Only +11 over 3463
although self-play merges since sum to ~+66 (cut-node 15.6, kb8m 11.0,
g5kb8m 30.4, se-negext 8.9). On the 6 anchors common to both ladders the
implied rating rose 3452 -> 3489 (+37, noisy): the stronger new anchors
(3525-3631, where we score 23-41%) pull the fit down, so part of the gap is
anchor-set change, part is self-play inflation. Lesson: discount self-play
gains ~0.3-0.6x for CCRL estimates; compare ladders on a fixed anchor set.
Top 100 needs 3506 (32 to go).

## Plan

**Short term (next ~24 h): gen 5 gate, feature batch, ladder.**
- Gen 5 (pool5, 976M, from scratch, chain11) is trained; its gate candidate
  is **g5kb8m** (g5 -> m1 copy -> kb8m exact warm start) vs main g4kb8m.
  Split stop rule: g5 < +8 at that gate -> back to 6/13 runners.
- Runner split trial: 13 datagen / 6 test (`lanes` in queue-settings.json).
- Queue: feature tests (feat-threat-hist requeued after +4.3 inconclusive,
  feat-se-negext-r2, ttPv, LMR improving, capture LMR, qsearch fail-soft) and
  simplification nonregs, ordered by expected Elo per runner-hour.
- Ladder after the next net merge or 2-3 search merges (pentanomial CI,
  ~+-20 at 40 games/anchor over 11 anchors 3318-3631).
- Queue backlog: mark a needs-human row resolved when its `-r2` is queued
  (feat-se-negext still shows needs-human); a test for the `queue.py remove`
  inbox path; a scheduled ladder should replace (not duplicate) a pending
  manual one, and a scheduled ltc-check shouldn't jump a `--front` ladder
  (2026-11-07 20:00: the front ladder arrived seconds after a dispatch, then
  5 merges queued a second ladder and an ltc-check ahead of it).

**Mid term (days): top 100 (3504).**
- Net generations every ~1/3 more data (gen 6 next), each followed by the
  m1 -> kb8m exact warm start that won for g4; drop the oldest, weakest data
  once the pool is large. King buckets from scratch need ~3-4B positions.
- SPSA of the ~30 search tunables after the feature batch lands (the new
  LMR/singular/threat terms interact).
- Search backlog: continuation history re-keyed to [piece][to][piece][to]
  then 2/4 ply, cuckoo upcoming-repetition, root LMR, quiet-check LMR,
  qsearch polish.
- Second trainer: 9600X. CPU trainer ~2x faster since dd12f29 (Zen 5
  runner 1.69M pos/s on 4 vCPU; M3 1.6M at 8 threads when quiet ~ the Metal
  trainer); projected ~4M pos/s on the 9600X (12 threads, unmeasured), so it
  may take generations while the M3 GPU does experiments; 1080 Ti (CUDA port) only as a last
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

## Status log 2026-11-07

Dated log, newest last; superseded detail is in the commit history and
`docs/nnue-experiments.md` ("Conclusions to date").

- **Overnight:** simp-nmp-margin merged +8.2 [1.1, 15.4]; simp-lmr-goodhist
  merged (neutral); simp-castle-order rejected. Ladder 3429 on g3.
- **Scale hypothesis rejected:** g3f +1.4, g3fs -8.9, g4ws -22.7; g4w's loss
  (-21.7) was over-specialisation of a short warm start. Screen on
  best-scale loss on val.bin + val17x2; games remain the gate.
  correction-history-r2 rejected under NNUE (+0.1 over 13.8k games).
- **g4 merged +26.7 [10.1, 43.5]** (from scratch, +35% data). Ladder
  **3463** (11:23). Background datagen lane raised; datagen bursts fill
  low-value gaps.
- **Runner split trial (10:45):** 13 datagen / 6 test, ~58M positions/h.
  Stop rule: gen 5 < +8 -> back to 6/13.
- **g4l3 (lambda 0.3) rejected** (-8.8, inconclusive).
- **King buckets:** from scratch data-limited (learning curve: gap 0.0011 at
  50% of games vs 0.0006 at 100%); exact factoriser warm starts from g4 win:
  kb8m (via g4m1) > kb4 > g4ft on val.bin and val17x2.
  **g4kb8m merged +11.0 [4.9, 17.1]** (17:31).
- **feat-lmr-cutnode merged +15.6 [6.4, 24.9]**; tm-ccrl-r2 merged as
  nonreg; feat-threat-hist +4.3 inconclusive, requeued.
- **Gen 5** (pool5, 976M) trained; smaller validation gain over g4 than g4
  had over g3. g5kb8m is the gate candidate.
- **Infra:** queue hardening (fd15da6), datagen bursts (1685c16),
  settings-driven lanes, `queue.py remove`, ladder anchors to 3631 with
  pentanomial CIs, fast CPU trainer (2x, factoriser, `--exact`), CI
  train-sweep v2, Metal trainer `--keep-best` / exact init. Auto loop is
  ingest-only (short fine-tunes don't win). Local datagen runs only while
  the GPU is idle (`~/rustfish-data/datagen-gate.sh`).
- **Parked:** AVX-512 NNUE paths (`nnue-avx512`: kernels +30-90% but
  whole-engine NPS within +-3%; re-check at H >= 768); threat inputs
  (`threat-inputs`: -1% val loss but -22% NPS).
- Search gap backlog (audit 2026-11-07, Elo guesses at our level):
  threat-aware quiet history + escape bonus (+8-15); ttPv flag in TT for
  LMR/pruning (+5-10); !improving in LMR (+3-6); LMR for late/bad captures
  (+3-6); re-key continuation history to [piece][to][piece][to] before
  retrying 2/4-ply (+3-8); LMR re-search depth adjust (+2-5); cuckoo
  upcoming-repetition (+2-4); reduce (not skip) LMR on quiet checks, root
  LMR, qsearch polish, improving ply-4 fallback (+1-3 each). New options
  still to add to docs/ccrl-submission.md after merges.
- Next: SPSA of the search tunables (`spsa-nnue`) when the queue is quiet.

## CI efficiency (measured 2026-11-06)

- STC waves: ~90% of wall time is games. Fine.
- Net-gate waves (5+0.05): ~170 s, of which ~60 s was rebuilding engines;
  fixed by caching the engines across waves (`34091d5`).
- The largest waste was *what* was tested (~10 doomed nets, ~24 rejected
  search patches, ~200k games in two days): screen by validation loss first.
- Datagen must leave runners free: the queue sizes it from `lanes` in
  `scripts/queue-settings.json`, so its own jobs and SPRTs are never blocked.
- Unused lever: runners can run *independent* training runs (ablation
  sweeps scored by validation loss) overnight, when no SPRT needs them.
  `train-sweep.yml` v2 (branch `train-sweep-v2`) does this: shared <= 300M
  pool, val.bin + val17x2 best-scale table, warm starts; it takes the 6
  foreground runners (docs/nnue-experiments.md, "CI training sweeps").
- Owner decisions: GitHub Pro (20 -> 40 concurrent jobs, 2x throughput) and a
  self-hosted runner on the 9600X desktop (outside the job cap).
- Ladder cost: a 120+1 ladder is ~10 SPRTs of runner time; run it only after
  real gains. A 10+0.1 ladder is ~12x cheaper and comparable with other
  short-TC claims (e.g. the 3463 "24h" engine, which was rated at 10+0.1).

## Known problems

- **Disk is now the gen-7 blocker (~14:00 2026-11-08):** 64 GB free with
  pool6 (42 GB) on disk; pool7 (~1.8B, ~58 GB) can't be shuffled next to
  pool6. Options: build pool7 straight from the source files (needs a clean
  manifest of pool6's sources, val tails excluded, x2 fixes) after deleting
  pool6, or do fix (1) below first. g6kb8m (val.bin 0.025832 / val17x2
  0.024920 vs g5kb8m 0.025886 / 0.025096) is gating now.
- **Disk (pick up before gen 7, ~3-4 days):** data grows ~1B positions/day
  (~32 GB/day) and we keep sources + a shuffled pool copy (+ a third copy
  while shuffling): ~64 GB/day against 88 GB free on a 460 GB disk. Fixes in
  order: (1) trainers read many source files with an on-the-fly shuffle
  buffer, no pool copy (~halves disk; subagent job); (2) recent-data window
  (drop seeds 4-13 from much weaker engines; test as a CI sweep, one
  variable) -> caps ~100-130 GB; (3) move data + generations to the 9600X if
  its disk is larger; (4) binpack-style compression (~2-3 B/position) only if
  needed. CI datagen artifacts (90-day retention) are a re-downloadable cold
  copy. Manual cleanup the owner can run: raw seed .gz dirs
  ~/rustfish-data/s{4,5,6,8,9,11} (~5 GB) and finished worktrees.

- **Anchors near the top.** The elite set now reaches 3631 (11 anchors,
  3318-3631); add stronger anchors before the ladder passes ~3600. Check
  each new anchor against its neighbours.
- **`OwnBook` advertises `default true`** in the `uci` output although the
  engine default is off (src/main.rs); wrong for GUIs that trust it. Fix in
  a small branch.
- **SPRT bounds:** `stc` is [0, 5]; `net-gate` (5+0.05, [0, 10], up to 5
  waves) is for same-shape nets only.
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
