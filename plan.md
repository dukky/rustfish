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
| Strength | **3547 CCRL Blitz** (ladder 2026-11-12 20:07, CI 3528-3567, `419d29b`, g8h1024kb8m): **top 100 cleared** (3506). Before: **~3508**: three ladders in a row at or above the top-100 line: 3510 (CI 3492-3528, 2026-11-09 04:16, g7kb8m), 3508.5 (2026-11-09 22:11), 3507.6 (CI 3490-3525, 2026-11-10 12:24, `1b3d81d`). Each CI still overlaps 3506 alone; together they put us at the line, not clearly past it. Before: 3492 (2026-11-08), 3474, 3463 (g4). |
| Goals | CCRL Blitz single-CPU list (best versions, list of 2026-11-07; `scripts/ccrl-targets.json`): ~~short term 3400~~ (2026-11-06); **mid term top 100 = 3506**: at the line (~3508), needs a clear margin; **long term top 10 = 3751**. Stockfish (3792) is out of scope. |
| Eval | NNUE **g7kb8m** (merged +30.6, 2026-11-08): (768 x 8 mirrored king buckets -> 512) x 2 -> 8 output buckets, SCReLU, i16; eval = net / 2. g7 from scratch on pool7 (1.72B positions), m1 copy, kb8m factoriser warm start. |
| Search | PVS, aspiration, TT buckets, NMP, RFP, futility, LMP, LMR (cut-node, not-improving, quiet checks reduced one ply less, root moves), IIR, SEE, killers, counter-moves, history + threat-aware history with escape bonus, 1-ply continuation history, ProbCut, singular extensions + multi-cut + negative extension, check extension; node-fraction time management. |
| Compute | **Desktop 9600X** (Zen 5, AVX-512, DDR5-6000, 180 GB free on WSL): main training + datagen machine from 2026-11-11; CPU trainer kb8m H512 2.2M pos/s (2.4M `--half`); **CUDA trainer for its 1080 Ti merged 2026-11-12** (`tools/nnue-trainer-cuda`, `netloop train --backend cuda`): kb8m H512 3.6M (4.1M `--half`), H1024 kb8m 1.9M (2.25M). `--half` = fp16 forward weights, val equal to 6 digits on 3 x pool7. **M3 laptop:** GPU (Metal) trainer for net-shape experiments on the fixed pool7. **CI (20 runners): testing only** (18 test shards, queue job 1, 1 idle background slot); CI datagen stopped 2026-11-11. |

**Self-play vs ladder, 2026-11-09..10:** feat-lmr-quiet-check (+12.9) and
feat-root-lmr (+12.2) merged between the 3510 and 3508 ladders, and the
ladder did not move. Both passed fast (2880 games) so their true gains are
smaller than their point estimates, and the ladder CI (+-18) can hide +10;
but it reinforces the rule: discount self-play gains ~0.3-0.5x, and nets
(+20-30 per generation) are the lever that has moved the ladder.

## Plan

**Short term (next ~24-48 h): gen 8 on the desktop, 1024-wide test.**
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
**Pool9** (~5B = pool8 + CI 62-118 + D2004/D2005): phase 1 downloading
on the desktop; the shuffle doesn't fit on the 246 GB WSL volume
(~262 GB peak), so it goes to /mnt/d or /mnt/f (owner's choice) or waits
for packed-data.

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
  per output bucket instead of width, cheaper in the engine (~5-10% NPS vs
  H 1024's 20-28%) and keeps the FT CPU-trainable. Step 1: CPU-trainer
  `--head` + small CPU signal tests (H 256/512, ~300M positions, head vs
  none); step 2: full-size H 512 kb8m + head vs main's H 1024 on the 1080
  Ti (needs the CUDA port); engine work only if step 2 reaches H 1024's loss.
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
