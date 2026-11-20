# Rustfish roadmap

The single source of truth for what is running, what is next and why. Process
rules live in `docs/development.md`; net experiments in
`docs/nnue-experiments.md`; results in `tests/*.jsonl`, `net/chain.json` and
the `queue-state` branch. Update this file whenever work starts, lands or dies.

_Last updated: 2026-11-19 23:30 (UK)._ Live state (current test, queue,
results, strength chart) is on the dashboard, **https://rustfish-dash.dev.duk.im**.

## Where we are

| | |
|---|---|
| Strength | **3588 CCRL Blitz** (ladder 2026-11-19, CI 3570-3605, 400 games, `2d945ec`: gen 10, 10 anchors). Before: 3581 (ladder 2026-11-16, CI 3563-3599, `b1961e7`: g9 + corrhist + futility + speed work). Before: 3562 (11-13), **3547 CCRL Blitz** (ladder 2026-11-12 20:07, CI 3528-3567, `419d29b`, g8h1024kb8m): **top 100 cleared** (3506). Before: **~3508**: three ladders in a row at or above the top-100 line: 3510 (CI 3492-3528, 2026-11-09 04:16, g7kb8m), 3508.5 (2026-11-09 22:11), 3507.6 (CI 3490-3525, 2026-11-10 12:24, `1b3d81d`). Each CI still overlaps 3506 alone; together they put us at the line, not clearly past it. Before: 3492 (2026-11-08), 3474, 3463 (g4). |
| Goals | CCRL Blitz single-CPU list (best versions, list of 2026-11-07; `scripts/ccrl-targets.json`): ~~short term 3400~~ (2026-11-06); **mid term top 100 = 3506**: at the line (~3508), needs a clear margin; **long term top 10 = 3751**. Stockfish (3792) is out of scope. |
| Eval | NNUE **g10h1024kb8m** (merged +12.2, 2026-11-19): (768 x 8 mirrored king buckets -> 1024) x 2 -> 8 output buckets, SCReLU, i16; eval = net / 2. pool10 (6.08B), plain -> m1 -> kb8m factoriser, all `--half`. |
| Search | PVS, aspiration, TT buckets, NMP, RFP, futility, LMP, LMR (cut-node, not-improving, quiet checks reduced one ply less, root moves), IIR, SEE, killers, counter-moves, history + threat-aware history with escape bonus, 1-ply continuation history, ProbCut, singular extensions + multi-cut + negative extension, check extension; node-fraction time management. |
| Compute | **Desktop 9600X** (Zen 5, AVX-512, DDR5-6000, 180 GB free on WSL): main training + datagen machine from 2026-11-11; CPU trainer kb8m H512 2.2M pos/s (2.4M `--half`); **CUDA trainer for its 1080 Ti merged 2026-11-12** (`tools/nnue-trainer-cuda`, `netloop train --backend cuda`): kb8m H512 3.6M (4.1M `--half`), H1024 kb8m 1.9M (2.25M). `--half` = fp16 forward weights, val equal to 6 digits on 3 x pool7. **M3 laptop:** GPU (Metal) trainer for net-shape experiments on the fixed pool7. **CI (20 runners): testing only** (18 test shards, queue job 1, 1 idle background slot); CI datagen stopped 2026-11-11. |

**Self-play vs ladder, 2026-11-09..10:** feat-lmr-quiet-check (+12.9) and
feat-root-lmr (+12.2) merged between the 3510 and 3508 ladders, and the
ladder did not move. Both passed fast (2880 games) so their true gains are
smaller than their point estimates, and the ladder CI (+-18) can hide +10;
but it reinforces the rule: discount self-play gains ~0.3-0.5x, and nets
(+20-30 per generation) are the lever that has moved the ladder.

## Plan

**Short term (updated 2026-11-19 23:30): search parity is now the main lever; nets keep running.**
- **Search (biggest lever).** Clover 9.1 (3725, same net class) as a black-box gauge: we are
  +91 at depth 1 (eval fine) but -197 at 200k fixed nodes (was -213 before the TT fix),
  and we search ~3.1x its nodes to depth 14. Depth sweep on main: +564 d5, +32 d8,
  **-51 d14** (quality erodes with depth, not only node count). Losses are slow
  middlegame squeezes; our eval is ~1 pawn more optimistic than Clover's in lost positions.
  Owner rule: other engines are benchmarks only, never code sources (memory: independent-development).
  - **Merged: `tt-failow-nomove` +16.7** [7.9, 25.6] (`f7d7a51`): fail-low nodes store no TT
    move (it cut only 5-26% as first move); nodes -33%, EBF 1.73 -> 1.66.
  - Failed: history-bonus-linear +2.6, cut-relabel -0.6, iir-depth3 -6.9. Dropped before CI
    (stats did not move): move-count pruning d4-6, counter-move first, fail-low hint.
    Pruning error rates are 0-1.2% and reduced searches fail high only 0.3-1.5%: margins are
    conservative, single-knob tweaks don't pay, so a **long SPSA** (sum combine; the old
    400-iteration run used mean, which applied one chain's steps per round) is running.
  - **Overnight 19-20 Nov:** `prune-exempt-killers` **merged +11.1** [4.2, 18.1] (`0c46b02`). Bug fixes: `rep-null-guard` H0 -4.8 [-11.5, 1.9] (dropped); `ep-hash-capturable` H0 +0.7 [-2.0, 3.4] over 18.8k games, neutral but a real correctness fix, so queued `:nonreg`; `queen-promo-tactical` running. Since 19 Nov: TT fix +16.7, prune-exempt +11.1.
  - **Round 4 (20 Nov):** Clover gauge on 48d4541: **-205.8** [-221, -191] at 200k nodes, unchanged: our merges save nodes (time), not value per node. Depth-14 misses (696 positions vs our own depth-20 oracle): 26% differ, 64 significant; **no single mechanism dominates** (disabling LMR fixes 25/39 at 5.4x nodes; +2 plies fixes 24/39 at 3x). Misses are near-ties or deep tactics: the gap is value per node, not one switch-off bug. Real fixes queued: `qsearch-fail-soft` (qsearch stored alpha on fail-low; nodes -7.9%), `quiet-cont-neg`, `pv-no-tt-cut`, `lmr-index-1based`, `root-bad-captures-last`, `nmp-mate-clamp:nonreg`. Aspiration fails waste 8-13% of nodes (not linear widening); extensions don't explode. Tooling on `diag-r4` (DiagOff per mechanism, `go searchmoves`, scripts/diag/).
  - **Round 5 (20 Nov, `diag-r5`):** depth-14 total nodes on diag-mid300 are noise within ~±5% (a no-op change moved +5%); judge by cutoff/TT rates. Queued `:stc`: `killer-reset-per-parent` (killer-1 cut rate 42%->65%), `nmp-tt-store` (null cutoffs stored as lower bounds), `root-order-nodes` (root sorted by subtree nodes; late best moves -40%, more switching; LTC look if it passes). Capture history gave the best ordering gain (+0.9 pt first-move cut) but already failed twice in SPRT; not retested. Details on branch `docs-round5`.
  - **Bug audit (19-20 Nov):** three proven long-standing bugs, each queued `:stc`: repetition check looked past null moves (61/368 bench draws false); ep square hashed after every double push even when no capture possible (one position, several keys); quiet queen promotions pruned/reduced as quiets (624/2843 pruned). Node counts within noise, so SPRT decides. NPS 'drop' is only the first second (flat 1.41-1.45M from depth 15-27): not a bug.
  - In the queue: `prune-exempt-killers` (only killers/counter exempt from move-loop
    pruning; nodes -9.6%). Next: PV stability at depth (X3), reduction/extension balance at depth.
  - Tooling: `diag-search-stats` (feature `stats`, `stats-verify`), `scripts/search_stats.py`,
    `scripts/diag-mid300.txt`; Clover matches via `peer-match.yml`. New candidates go through
    the merge queue (`queue.py add <b>:stc`), not manual `ci.py sprt`.
- **Data: more beats newer.** Window test (same recipe/compute, val-g10 / valL1008):
  newest 1.27B only 0.023470 / 0.022888; whole 6.08B pool 0.023336 / 0.022723; whole pool
  with newest 2x **0.023299 / 0.022690**. Keep growing, don't drop old seeds, weight the
  newest 2x. **Running:** `gen10w.sh` = gen 10 recipe with newest 2x (GPU, 22:44 -> ~09:30
  20 Nov); gate it if it beats g10.
- **gen10w (newest 2x) does not beat g10 at full scale:** val-g10 0.022958 vs 0.022948, valL1008 0.022415 vs 0.022408, val.bin 0.025641 vs 0.025551. The window test's short-recipe weighting gain (~4e-5) vanishes with full training: no gate; **gen 11 = plain, unweighted pool11**. CUDA speed table (idle GPU): pairwise 2.07x (717k -> 1.49M), plain -0.9/-1.5% (being checked before merge).
- **Pairwise from scratch WORKS (pilot 20 Nov, shape recipe):** val-g10 **0.022927** / valL1008 **0.022299** vs plain 0.023470 / 0.022888 and dense head 0.022993; head alive (W1 97.5% nonzero). **Gen 11 started 13:03 as a full pairwise chain** on pool11 (`gen11.sh`, trainer a808d8a, ~17-19 h -> ~Sat 07:00). Pilot net under game test (`nnue-shape-h1024kb8m-pw` vs `nnue-shape-h1024kb8m`, fixed nodes + stc); rule: FN clearly positive and stc >= ~0 -> gen 11 continues pairwise, else abort and restart plain. **Fixed 200k nodes: +40.9 [+29.9, +52.0], 1152 games (H1). Stc 10+0.1: -1.7 [-7.3, +3.9], 4992 games (H0)**: the eval gain is real but the speed cost (-7.9% NPS x86, -20.6% M3; head alive -> ~37% chunks nonzero) eats it at stc. LTC test running (decides gen 11; CCRL Blitz is 2+1, longer than stc). Gen 11 continues pairwise meanwhile. Speed levers if needed: integer L2 (needs QAT L2), FT pair-order permutation. **Speed pass done (`nnue-shape-h1024kb8m-pw-fast` bd57ff7, eval bit-identical):** offline FT pair permutation (`head_permute` test in src/nnue_head.rs; nonzero chunks 95.4 -> 66.9), SIMD L2/L3, cheaper NEON pair product: only **+3.6-4% M3, +0.7% x86** (desktop loaded). Head overhead ~280 ns/eval, mostly the fused FT update (memory/port-bound). Bigger wins need the trainer: QAT integer L2/L3, an L1 penalty for sparser pairwise inputs, narrower pairs. Run gen-11 nets through head_permute before embedding.
- **Generation cadence (audit 20 Nov):** net-gate gains g5..g10 = +30, +17, +31, +37, +23, **+12**; pool11 is only +10% over pool10. After gen 11, train the next generation when the pool has grown ~30-50% or a new architecture is ready; GPU time in between goes to pairwise speed/recipe experiments (Lazy Adam, recipe length, head variants). Datagen keeps running (data is the long-run constraint, ~20B for 3700-class nets). **Lazy Adam (20 Nov, branch `trainer-lazy-adam`, off by default, byte-identical default path):** no speed win: ~84% of FT rows / 92% of features occur in every 16384 batch, so at most ~16% of adam_ft is skippable; GPU smoke <1% per batch. CPU loss sanity (H256, 3 x 1M) identical to ~1e-5. Low priority; GPU A/B only if a slot is idle (recipe in the agent report, docs on the branch).
- **Queue priority note:** pick_next scores by mode base rates, so all `:stc` items tie (FIFO); use `--front` or `--expect <elo>` for high-prior items (killer-reset-per-parent, capture-history-r5 moved to the front 20 Nov).
- **Pairwise head: the head never trained** (W1 = its random init after 3 epochs). From the
  g9 warm start only ~1% of pairwise inputs are nonzero, so the head gets no gradient. QAT
  and CUDA are not the cause. Likely needs an FT trained for pairwise (full chain), so it
  is parked until gen 11/12; init/pair-order experiments are on `trainer-head-pairwise`
  206880a. The CUDA pairwise-head trainer is 2.2x faster byte-identical (`perf-cuda-pairwise`, WIP).
- **Usage:** weekly budget tight (50% by Thu 19 Nov); at most 1-2 subagents, poll rarely.

**Earlier short-term notes (2026-11-19 daytime):**
- **Gen 10 merged +12.2** [2.9, 21.6] (net-gate, `2d945ec`, g10h1024kb8m):
  pool10 6.08B (pool8 + ci9 + D2004-D2022; D2000/D2001 out; D2021/D2022
  tails = val-g10), gen-9 chain all `--half` (plain 10 ep 1e-3 -> 1e-5, m1 1
  ep 2e-4 -> 1e-5 keep-best, kb8m factorise 4 ep 2e-4 -> 1e-6 keep-best),
  CUDA trainer on the 1080 Ti at 125 W, 19 Nov 00:28 -> 11:02. Best-scale
  val-g10 / valL1008 / val.bin 0.022948 / 0.022408 / 0.025551 (g9 0.023016 /
  0.022553 / 0.025538). **Ladder ~3588** [3570, 3605] (400 games; 10
  anchors: the Stash 35 anchor build fails in the ladder).
- **Net-shape test done** (val-g10 best-scale; plain 4 -> m1 1 -> kb8m 2 ->
  head 2 on the 1.27B newest positions):

  | width | no head | head |
  |---|---|---|
  | 256 | 0.024609 | 0.023984 |
  | 512 | 0.023924 | 0.023363 |
  | 1024 | 0.023470 | 0.022993 |

  Metal vs CUDA at 256: 0.024551 vs 0.024609 (trainers agree to 6e-5).
  Within-recipe games: H512 vs H1024 -17.3 stc / -36.9 fixed-nodes; head at
  512 -69.1 stc / +24.8 FN; head at 1024 -79.2 stc / +25.4 FN.
  **1024 stays the width.** The head's eval gain is real but its speed cost
  dominates; loss overstates the head in games, so **fixed-nodes games are
  part of the gate** from now on.
- **Head-fix** (exact W1): engine-vs-trainer agreement 0.47 cp; -46.7 stc,
  **+41.8 at fixed nodes (200k)**: the head is right, just too slow. New
  `ci.py sprt --nodes N` diagnostic mode (PR #1, labelled DIAG, never used
  by the queue).
- **Pairwise head** (branches `trainer-head-pairwise` 8c9ff85,
  `nnue-head-pairwise-wip` d75267d): QAT i8 W1, inputs 0..127, linear output
  kept. NPS -15% x86 / -19% M3 (head-fix: -45% / -41%). Pilot loss level
  with the dense head; the skip variant was dropped (much worse, not
  faster). **Running:** full 3-epoch fine-tune from g9, lands ~18:30 today.
  Then games vs pre-g10 main via `--base`; if it wins, **gen 11 = pairwise
  head on g10**. CUDA pairwise-head trainer speed work in progress
  (`perf-cuda-pairwise`).
- **Peer analysis** (`docs/peer-analysis-2026-11-19.md`): 15 engines in the
  3620-3800 band. Levers in order: cheap head; threats the peer way (after
  the head); data to 15-20B; search ideas; long SPSA. Step 0: a local Clover
  9.1 comparison.
- **Datagen:** D2024 done; D2025-D2027 queued on the desktop; L1011 done;
  L1012 running on the gen-10 engine.
- **Ops:** desktop GPU cap 125 W (set by hand, not persistent across
  reboots; ~16% trainer cost accepted). Datagen thread test: 10 threads fine
  at H1024; H512 is 21% CPU-bound. Tailscale SSH ~53 Mbit/s vs 75 plain TCP
  (Tailscale in WSL is the cap). Pack sources in parallel next time.
  Ladder: fix the Stash 35 anchor build.

**Mid term (days-weeks): clear top 100, start on bigger nets.**
- **Data decision (2026-11-12, owner): no T80 for now.** The goal is a
  stable release as a true from-scratch engine (own self-play only). Not
  closed: revisit at a real data-scale wall, or for a later version. Data
  grows from desktop + laptop datagen; get more Elo per position from net
  architecture (1024 done; threats after the speed fix; deeper head next).
  T80 reader parked on `t80-reader` (docs/t80-data.md).
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
- Bigger nets (1024+ wide, king buckets); AVX-512 inference paths for
  them (`nnue-avx512`).
- **Deeper head** (design: `docs/nnue-deeper-head.md`): FT -> 16 -> 32 -> 1
  per output bucket instead of width, cheaper in the engine (~9% NPS at H 512
  vs ~19% for H 1024 head). CPU+CUDA trainers work. Bulk plain FT stays
  CPU-friendly; head training is dense and stays on the 1080 Ti for
  production runs (CPU H512 head 84-104k pos/s vs GPU H1024 head ~960k after
  bucket-indexed backward). Optional future work: sparse/int8 CPU head L1
  and head-only freezing; neither is needed for the current pipeline.
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

**2026-11-12..18 (short-term notes, moved from the plan on 2026-11-19)**
- **pool8 built (desktop, 2026-11-12 08:44, `~/rustfish-data/pool8.sh`): 3,201,732,989
  positions** (102 GB): pool7 (1.72B, read not consumed: the desktop's 1024 kb8m run
  trains from it) + CI seeds **119-167** except s145 (no successful run; ~1.1B, newest engine) + CI seeds
  **52-55, 57, 58, 61** (~187M) and local L1007 chunks c2/3/5/7/9/10/14 (~20M),
  all wrongly quarantined from pool7 by the scale check at slope 1.250-1.255
  (drift, not the half-scale bug: on g6kb8m, which saw none of s56-s61, their
  loss interleaves with accepted neighbours'; band widened to 1.40, e45b9b7)
  + desktop **D2000 + D2001** (163M). D2002/D2003 were copies of D2001 (seed
  2001 reused; deleted; next desktop seed 2004). CI seeds **62-118** (~1.6B,
  minus those above) are held back for pool9 (artifacts kept to ~early Feb 2027).
  pool7 on the desktop goes once the desktop session's 1024 kb8m run ends. Gen 8
  (g8 -> g8m1 -> g8kb8m, or 1024 if g7h1024kb8m wins) on the 1080 Ti.
- **g7h1024 done (2026-11-12 04:55)**, 10 epochs on pool7, no king buckets.
  Best-scale val.bin / val17x2: g7 0.026073 / 0.025162; **g7h1024
  0.025733 / 0.024701**; main g7kb8m 0.025821 / 0.024812. So plain 1024
  beats main's king-bucket net by 9e-5 / 1.1e-4 (g7kb8m's gate win of
  +30.6 came with ~1e-5 / 1.1e-4 over g6kb8m). Engine cost: -20% NPS on
  the M3 (bench 13 interleaved, 4.27M -> 3.39M; LAYOUT_NONE, H=1024).
  **g7h1024kb8m** (chain16, done 09:20): best-scale **0.025637 / 0.024590**
  (-1.8e-4 / -2.2e-4 vs main; the kb8m step gave -1e-4 / -1.1e-4 over plain
  1024, half its 512 gain: train/val gap 0.0240 vs 0.0262, data-hungry).
  M3 NPS -28% vs main (2.89M vs 4.04M). **Queued `stc` at the front**
  (branch nnue-g7h1024kb8m). Gen 8 on the 1080 Ti is 1024 by the >= 1e-4
  rule; the desktop's CUDA g7h1024kb8m is the trainer cross-check.
- Queue: **feat-cuckoo merged +5.5** [1.1, 9.9] (2026-11-12 01:54);
  feat-conthist-rekey rejected (-0.2, 10.9k games: the [piece][to] re-key
  alone is neutral, so the 2/4-ply retry loses its premise; deprioritised);
  feat-qs-evasion-prune rejected (-3.5); feat-lmr-deeper (SF-style
  deeper/shallower LMR re-search, margins 43 + 2d / 9) rejected -7.6
  [-13.7, -1.4] (one run; the shallower branch also skips re-searches, a
  deeper-only variant is untested); feat-improving-ply4 queued.
- Ladder after the next net merge.
- **SPSA between gates** (2026-11-12 08:59, spsa.yml, 8 shards, 50 it/round,
  10 rounds; queue on 10 test shards meanwhile, e97a6dc). Pause it between
  rounds for the gen-8 gate (state = last `spsa-result` artifact), restore
  18 test shards, resume after.
- **Datagen: desktop datagen restarted by the user (laptop L1008/L1009 too) (D2004 on the gen-8
  engine, then D2005 overnight; 9 threads).** Data decision still pending. CI
  datagen stopped means new data is ~350M/day (desktop) vs ~1.4B/day
  before, so data, not training, is the bottleneck now; that raises T80's
  value. In progress (subagents, branches): `t80-reader` (binpack decoder,
  filters, score calibration, licence notes) and `packed-data` (compact
  on-disk format + streaming shuffle loader, parity-tested against a
  shuffled pool).

**2026-11-12 results:** **g7h1024kb8m merged +11.5** [5.5, 17.5] (stc,
4560 games, 14:53; the 1024 width pays for its -28% NPS on x86).
**g8h1024kb8m merged +37.2** [23.6, 50.9] (net-gate, 1060 games, 17:58;
pool8 3.2B, 1080 Ti CUDA: best-scale 0.025574 / 0.024364). Main is now a
1024-wide kb8m net. Rejected: feat-lmr-deeper-only +0.5 (11.5k games).
ltc-check then ladder (~21:00). SPSA restarts on the new main after
round 6 (its rounds were pinned to the 512 engine). Tonight on the
1080 Ti: threat control fine-tune (no threats, ~19:00), then kb8m from
scratch on pool8 (tests the "KB from scratch needs 3-4B" estimate), then
the ep-threat fine-tune once threat-inputs-v2's lib lands.
**Evening:** ladder **3547.4** [3528, 3567] (g8h1024kb8m, +40 over
3508). **Threat inputs (ep)**: g8h1024kb8m + ep rows, 3 epochs on pool8
(1080 Ti, branch threat-inputs-v2 incl. cuda-threats): best-scale
0.025346 / 0.024097, -2.3e-4 / -2.7e-4 vs g8 (its no-threat control did
not improve on g8). M3 NPS -24.9%. **Queued `stc` --front** (branch
nnue-g8h1024kb8m-ep, Bench 356168). Desktop deeper-head step 1 (branch
head-v1, FT -> 16 -> 32 -> 1, 3 x 200M pool8, raw val): H512+head
0.027596 beats plain H1024 0.027683; needs CUDA/speed work for full size.
SPSA on the 512 engine: 400 iterations moved nothing beyond ~6% of
range; restarted on the 1024 main (22:41).
**Desktop handed over (23:00)**: one Claude session (the laptop's) now
drives the desktop over ssh; the desktop session is idle. Its handoff:
`~/rustfish-data/HANDOFF-desktop.md`. Tonight, detached: GPU kb8m-from-
scratch (~00:45, `gpu-tonight.log`) -> packed-data A/B (~01:45,
`packed-ab.log`); CPU D2005 (~05:00) -> convert (`convert-d2005.sh`).
Desktop findings not recorded before: `--half` from scratch matches f32
(plain H512, 3 x 400M: 0.027702 vs 0.027704), so future gens can use it
throughout (~40 min/gen saved); batch 32768 is worse at equal positions
and no faster. Next desktop datagen seed: 2006. Gen 9 waits on three
results: from-scratch vs chain, the threat stc, the packed A/B.
**Threats ep rejected in games: -23.2** [-35.2, -11.2] (stc, 960 games,
23:03) despite the loss gain. Diagnosis so far: eval is fine (fixed depth 9
on the desktop, 240 games: **+58** [34, 84] vs main); same output scale as
g8 (best scale x0.7918 vs x0.7963, mean|eval| 468 vs 464); cost: **-21% NPS
on x86 AVX2** (1.70M -> 1.33M) and +18% nodes at bench 13 (one tree, not
conclusive). Plan (owner, 2026-11-12): fix and experiment during gen 9,
bake into gen 10: (1) time-odds stc (ep at ~1.3x time) to confirm cost is
the whole story; (2) profile and make threat inputs incremental (per-move
attack-map updates instead of recomputing); (3) ltc test (CCRL is 2+1).
Gen 9 trains without threats.
**Engine speed at 1024 (branch speed-1024, queued as `speedup`, bench
unchanged):** fused last accumulator update + output layer, dual-output
finny refresh, 64-byte alignment, runtime-selected AVX-512 kernels, wider
NEON loads, eval-cache prefetch. M3 **+13.6%**; 9600X (under datagen load)
**+3.3% AVX2, +7.7% AVX-512**. Profile at H1024: NNUE ~55% (update 39%,
SCReLU dot 14%), MovePicker::new 11%, movegen 7%, SEE 5%. Not merged
experiments: `speed-1024-rowprefetch`, `speed-1024-hugepages` (x86 only;
measure with the desktop idle). Threat-cost ideas: route threat rows
through the fused kernel in fixed chunks; reuse the move picker's attack
maps instead of rebuilding per ply; i8 threat weights (trainer + format).
**Clean x86 (idle 9600X, x86-64-v3 builds, bench 13, x6):** speed work
+8.9% on the AVX-512 path (3.20M -> 3.48M), only **+2.2% on AVX2** (3.24M ->
3.31M): the AVX2 fused path is the next speed target (most CCRL testers
are AVX2). **PGO -2.2% on x86** (both paths): releases stay plain.
Huge pages / row prefetch: -1.4% each, dropped. Datagen on 9 threads
halves single-thread NPS on the desktop: benchmark only when it's idle.
**Overnight 2026-11-13:** kb8m from scratch on pool8 (10 epochs) lost to
the chain: best-scale 0.025897 / 0.024827 vs g8 0.025574 / 0.024364 (train
0.0236 vs val 0.0270: still data-limited). **Packed-data A/B** (643M ci9
positions, H1024 kb8m, 2 epochs): streaming loader (64M buffer, stride 4)
**beat** the shuffled pool: 0.026742 / 0.025839 vs 0.026778 / 0.025897.
So pools are now manifests, no shuffle copies (merge `packed-data` to main
next). **Gen 9 started 01:17** (`~/rustfish-data/gen9.sh`, logs in
`gpu-jobs/`): pool9 by stream = pool8 + CI 62-118 + D2004 = **4.90B**, gen-8
chain recipe, --half, no threats; ETA ~10:30. Threat net LTC diagnostic (manual, vs 0f6fda6, 60+0.6): **-3.0**
[-11.6, +5.6] (1720 games) vs -23.2 at 10+0.1: the gap closes with time,
so it is cost. **threats-fast** (a8e778c; THREATS off = main exactly) and
**nnue-g8h1024kb8m-ep2** (8650d76, same ep net and search): threat NPS cost
on the M3 -32% -> **-20%** (fused path, threat-aware finny refresh, lazy
per-ply work, fixed-shape kernels); part of the rest is the ep net's 18%
larger tree. **stc -10.2** [-15.9, -4.6] (4560 games; was -23.2); **ltc (60+0.6) vs main
-7.0** [-19.3, +5.2] (840 games; the old ep was -3.0 at ltc). So at longer TC
threats are about break-even, not a clear win: the +58 at fixed depth
overstates them. Parked for merging; revisit with a fully trained threat
net in gen 10 and/or i8 threat rows. Next lever: i8 threat rows (0.16% of
weights out of range; exact split possible, no format change).
**head-v2** (cecaf14 / aff868a / c80674f): engine head inference
(`src/nnue_head.rs`, HEAD off = main), agrees with the trainer within
~0.5-0.6 cp mean; H512 kb8m + head runs 9% slower than main on the M3;
CPU head trainer +29%; CUDA head port untested. After gen 9 the desktop
runs `head-after-gen9.sh`: CUDA vs CPU head check (<= 1e-5, deterministic),
then g9h1024kb8m + head fine-tune (3 epochs pool9). packed-data merged (4c75112). feat-history-prune rejected
(-1.3). speed-1024 merged (00:09).
**Validation sets are stale (2026-11-13):** gen 9's plain stage looked
3.2e-4 *worse* than gen 8's on raw val.bin, but best-scale it is within
2-5e-5 on val.bin / val17x2 (old seed-9/17 engines) and **better by
9.6e-5 best-scale (1.9e-4 raw) on valL1008** = 1M held-out positions from
the gen-8 engine (laptop L1008 tail, in no pool): the old sets measure an
outdated distribution (score scale drifted 1.19 -> 1.29). **Rule (docs/nnue-experiments.md "Validation sets"):** a new held-out set
per generation (~1M positions, whole games, >= 2 seeds, newest main
engine, never in a pool), screened first via `val-current.bin`
(netloop's default; now -> valL1008 on both machines); old sets kept as
references. Next: val-g10 from the D2006 + L1009 tails.
**Today (2026-11-13), owner-agreed direction:** data growth is slow
(pool10 ~ +4%) and pool9's extra 1.6B was older-engine data, so the next
jumps come from more Elo per position: (1) deeper head (1080 Ti after
gen 9: CUDA check, then g9 + head fine-tune); (2) **overfitting sweep** on
the M3 GPU (`~/rustfish-data/overfit-sweep.sh`: pool7, H512 plain, 4 epochs
each: base, wd 0.01, mirror, lambda 0.3, lambda 0.7, wd 0.001; scored on
val-current at lambda 0.5; laptop datagen stopped at L1009 54M); (3) a
recent-data window test (drop the oldest seeds; relabelling old data is
parked by the owner: not on CI (bulk compute, same as datagen), and only a
small desktop test arm if the window test shows old data hurts); (4) gen 10 as an
**architecture generation** (head / threats trained from the start /
winning training settings) rather than a data one; (5) **AVX2 speed**
(subagent, branch `speed-avx2`, measured on the 9600X with datagen paused).
**2026-11-13 10:00: gen 9 merged +23.4** [8.8, 38.0] (net-gate, 864
games; g9h1024kb8m, pool9 4.90B: best-scale valL1008 0.022553 vs 0.022610,
small but consistent). **speed-avx2 merged** (speedup, bench identical):
early TT/eval-cache prefetch at move pick + null-move prefetch, and real
transparent huge pages for the TT on Linux (allocate, madvise, then
zero): **+7% on both AVX2 and AVX-512** on the 9600X. NNUE is only ~25%
of x86 time (55% on the M3) and the AVX2 kernels are load-bound, not
spilling; next x86 costs: MovePicker::new 13%, movegen 9%, SEE 8%.
SPSA cancelled (nothing in nine rounds); lanes back to 18 shards.
Ladder queued.
**Audit 2026-11-18 (back after the weekly limit):**
- **Head gate -49.8 was an engine bug, not the head:** the gated build's
  eval disagrees with the trainer by **8.2 cp mean, p95 24.5, max 88**
  (2000 valL1008 positions; worst: engine -250 vs trainer -162), against
  ~0.5 cp on head-v2's small test heads: the i8 W1 quantisation / clamps
  don't survive a fully trained head. The head net itself is the biggest
  loss step yet (valL1008 0.021993 vs g9 0.022553). Subagent fixing the
  engine path (`head-fix`); rule: no head net is gated without passing the
  engine-vs-trainer check (< 1 cp mean).
- **GPU idle 13 Nov 16:52 -> 18 Nov 20:00** (5 days) while datagen ran:
  the chain had nothing queued after the head fine-tune and the desktop
  rebooted (~14 Nov; also resets the 1080 Ti power cap and killed the
  dashboard status loop, restarted). Need a standing GPU job queue.
- **Desktop disk 95% full** (13 GB free). Fix running: pack game-ordered
  sources to .rfp (ci9, local D20xx; lossless, verified), then gen 10.
- **Gen 10 starting tonight** (`~/rustfish-data/gen10-prep.sh` -> `gen10.sh`):
  pool10 = pool8 + ci9 + D2004-D2022 (D2021/D2022 tails = **val-g10**, 1M,
  held out), gen-9 chain, --half; screened on val-g10 / valL1008 / val.bin.
- **Desktop seeds D2016-D2019 are burned** (runlong-2016.sh, e3390ac engine,
  12 threads): no output files survive, most likely lost in the ~14 Nov
  reboot. Never reuse 2016-2019; next free desktop seed is **2024**. Datagen
  and the GPU queue should resume automatically after a reboot (needs a
  Windows startup task, with the 180 W power cap).
- **Net-shape test before gen 10** (owner, 2026-11-18): H256 / H512 / H1024
  kb8m, each with and without the head, on the same 1.27B newest positions
  (D2004-D2022), short recipe (plain 4, m1 1, kb8m 2, head 2), new CUDA
  trainer (`perf-head-speed-cuda`, verified byte-identical, +10% plain /
  +20% kb8m). Loss first, then games (speed counts; head arms after the
  engine head fix). Gen 10 follows automatically. Laptop datagen: L1010 on
  main 738abec, 6 threads.
- Overfitting sweep final (M3, pool7 H512, best-scale valL1008 / val.bin):
  base 0.024591 / 0.026185; wd 0.01 0.024590 / 0.026116; mirror 0.024671 /
  0.026248 (worse); **lambda 0.7 0.024469 / 0.026595** (better on current
  data, worse on old; needs a game test before use).
- 13-17 Nov (other session): merged perf-see-ge (speedup), **per-move
  futility +8.8**; rejected formula LMP -6.9, capture-history-SEE +1.6
  (20k games, H0), cont2-ply x2, countermove history, root LMR quiets,
  rfp-improving simp, quiet-check ordering, capture ordering +2.1 (H0),
  pawn history, razoring, ttcapture-lmr, LMR bad captures, recapture ext,
  correction-LMR, improving-ply4 smoothed, 50-move damping. Cheap search
  ideas look exhausted; nets/architecture are the lever.

**Midday 2026-11-13:** ladder **3562.4** [3545, 3580] (gen 9 + speed work).
**feat-corrhist-v2 merged +12.8** [6.9, 18.6] (multi-table correction
history, used in qsearch too, ~1% cycles). simp2-qs-failsoft merged
(nonreg +3.5); double-extension rejected (-3.6). **--half from scratch at
H1024** (3 x 400M pool8, same seed): valL1008 0.024375 vs f32 0.024350
(+2.5e-5), val.bin equal: fine. **Head fine-tune on g9** (1080 Ti, 628k
pos/s, 3 epochs, done ~16:50): epoch 1 raw valL1008 0.022437 vs g9
0.022681 (-2.4e-4). The CUDA head check was 8e-5 off the CPU trainer at
epoch 1, 3e-6 by epoch 3, deterministic. **Overfitting sweep** (M3,
pool7, H512, 4 epochs, best-scale valL1008 / val.bin): base 0.024591 /
0.026185; wd 0.01 0.024590 / 0.026116 (neutral / slightly better);
**mirror worse** 0.024671 / 0.026248; lambda 0.7 running. Next search
features (subagent): formula LMP with improving, per-move futility,
capture history.
**Rejected-work sweep (2026-11-13):** verdicts from the HCE era and the
mis-scaled NNUE window (11-03 17:00 .. 11-06 13:15) are weak evidence.
Queued now (quick): `double-extension` (stc), `simp2-qs-failsoft` (nonreg).
Subagent: **correction history v2** (multi-table: pawn, non-pawn per
colour, minor/major, continuation; used in qsearch too), branch
`feat-corrhist-v2`. Later rewrites (0 to +5 each): formula LMP with
improving to depth ~8 (after the 1024 SPSA, which tunes LMP), per-move
futility with a cheap gives-check test, capture history as a SEE
threshold, 2-ply continuation history kept out of the priority test.
Cheap but low-power (likely H0 at [0, 5]): ttcapture-lmr, razoring d3-4,
50-move NNUE damping. Not now: NNUE_EVAL_DIV retest (a rational divisor
would corrupt datagen labels). Correctly dead: net5-13, bigbang bundles,
HCE-only work, recent decisive losses.
**Pool9** (~5B = pool8 + CI 62-118 + D2004/D2005): phase 1 downloading
on the desktop; the shuffle doesn't fit on the 246 GB WSL volume
(~262 GB peak), so it goes to /mnt/d or /mnt/f (owner's choice) or waits
for packed-data.


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
  B/position) if still needed. Branch `packed-data` (2026-11-12): packed
  `.rfp` sources at ~3.4 B/position plus a streaming shuffle (`train
  --data <dir|list> --shuffle-buffer N --stream-stride k`); laptop parity
  within ~5e-5 of a shuffled pool; desktop A/B on pool8 pending
  (`docs/data-format.md`).
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
- 2026-11-17: head gate replayed cleanly onto main as
  `nnue-g9h1024kb8m-head-clean` (bench 366942, queued stc expect +5) after
  the original shape gate hit a rebase conflict; net/docs records on
  `docs-net-records-1117`.   CUDA head perf (on `head-v2`): `dacc` 670k -> 766k
  pos/s, `rowwarp` -> 878k, `bucket-backward` -> 960k, then
  `perf-head-speed-cuda` cherry-picks `speed-cuda` into the head stack:
  **1.05M head / 2.48M plain**, all byte-identical to their baselines;
  `backward_ft` 3.35ms -> 1.67ms/batch. fuse/warp-for-all-rows experiments
  were slower and are parked. The combined branch supersedes standalone
  `speed-cuda` for the head stack; both are trainer-only owner merge
  decisions. Datagen3 runs g9 with 10 threads on D2020-D2023 (1M games
  each, 5000 nodes); D2020 at ~42 games/s, ~27h for the whole sweep.
  Head inference follow-up `perf-head-fuse-screlu` (2841118) fuses SCReLU
  into the last accumulator pass: bench 366942 and evalhash unchanged,
  ~2% faster on M3 depth 12; parked behind the shape gate. CPU trainer
  `perf-trainer-head-w1-batch` (66efbba) batches W1 gradient adds by output
  bucket: 31 tests pass, byte-identical nets, +24% H256 / ~1.8x H512 pos/s
  on M3; trainer-only owner merge decision. M3 only while datagen runs.

## Deferred actions (each has a trigger; re-check this list whenever a job finishes)
| Action | Trigger (armed) |
|---|---|
| gen10w (newest 2x): evals vs g10, net-gate if better | watcher on `gpu-jobs/gen10w.log` (~09:30 20 Nov) |
| CUDA pairwise-head trainer speed (`perf-cuda-pairwise`): full-speed table on a free GPU, README, squash | next free GPU window (after gen10w); paused subagent |
| Pairwise head revival: init / pair-order experiments, else an FT trained for pairwise in a full chain | paused (budget); decide at gen 11/12 |
| Merge the verified faster CUDA trainer (perf-head-speed-cuda) into main | next free agent slot (independent of the pairwise decision) |
| GPU order 20 Nov: gen10w evals -> CUDA speed table -> **pairwise-from-scratch pilot** (shape recipe, ~2-3 h; if the head trains and beats plain, gen 11 = full pairwise chain) -> gen 11 | chained/agents; pairwise agent owns the pilot |
| **Pool11 ready (20 Nov 09:40):** 6,715,454,861 positions = pool10 + D2023-D2026 + D2027.train + L1009 + L1011 + L1012.train (.rfp, unweighted); **val-g11.bin** 1,000,000 = D2027 + L1012 tails (500k each, TT-fixed f7d7a51 / gen-10 2d945ec engines), sha df6e70abea4d0fb5; add to the val table in docs/nnue-experiments.md | done; gen 11 waits for the pairwise pilot |
| Recipe-length test: does a shorter chain (e.g. plain 6-8 epochs) match the 10+1+4 recipe? Cuts every generation's ~11 h | next free GPU window after gen 11 starts? (needs its own GPU slot) |
| CUDA trainer speedups **merged to main 20 Nov (cc99ebe)**: perf-head-speed-cuda, perf-cuda-pairwise (pairwise 2.07x), trainer-head-pairwise (CPU+CUDA pairwise heads; engine `src/nnue_head.rs` with HEAD=false, bench unchanged 283739), perf-cuda-plain (+6% plain, byte-identical; idle-GPU A/B pending `~/pwperf/speed-plain-free.txt`). Next: lazy Adam as a flagged option (adam_ft ~11% of a batch) | build the gen-11 CUDA trainer from main |
| Cut val-g11 (~1M, whole games, >= 2 seeds, gen-10+ engine: D2027 + L1012 tails), then build pool11 (~6.7B, newest seeds 2x) | D2027 done (~Sat 21 Nov early) |
| Bug audit fixes in the merge queue (`rep-null-guard`, `ep-hash-capturable`, `queen-promo-tactical`, all `:stc`); unverified suspects listed in docs/search-diagnosis (Bug audit) | queue results (auto-merge on pass) |
| Search: next experiments from round 3 (X3 PV stability; reduction/extension balance at depth); gauge = 200k-node Clover match after each batch | after prune-exempt-killers' verdict |
| D2026 (gen-9 engine) -> D2027-D2030 on the TT-fixed engine (`rustfish-6daf892-native` is a symlink to `rustfish-f7d7a51-native`) | automatic: `runlong-2027.sh` in tmux `datagen5` on the desktop |
| Long SPSA (run 44631902925, 605b560, 6 shards x 40 rounds, sum combine, ~2 days): restore queue to 18 shards and lanes 1/1 (scripts/queue-settings.json), then queue the tuned values `:stc`, then `:ltc-check` | last round's merge job finishes (round 40/40); theta = its `spsa-result` artifact |
| Search diagnosis round 3 (docs/search-diagnosis-2026-11-19.md): `prune-exempt-killers:stc` queued (nodes -9.6%); next X3 = PV-stability measurement | queue result |
