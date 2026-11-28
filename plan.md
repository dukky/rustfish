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
| Strength | **3620 CCRL Blitz** (ladder 2026-11-26 08:00, CI 3605-3635, 440 games, `f7a45bc`: + gen 12 net, small-fixes-stack, SPSA r14 interim; same 11 anchors; +2, inside the CI). Before: **3618** (queue ladder 2026-11-25 03:58, CI 3602-3633, 440 games, `eb3b4a4`: + combo-small-positives-r2, caphist SEE prune, child-eval LMR, speed-stack-r1, bug fixes; same 11 anchors; only +4 on the ladder although the LTC check vs the 3613 main read +31 (648 games, early stop): gains show against the weaker anchors (koivisto-6 75% vs 67.5%, carp 61% vs 65%) but not the 3692-3773 ones (27.5-40%)). Before: **3613** (queue ladder 2026-11-22 20:05, CI 3599-3627, 440 games, `e8a3495`: gen 11 net + bug-iir-tt-depth, history-bonus-linear-r2, tm-full-inc, nmp-mate-clamp, negamax-rep-floor, fifty-move-mate; same 11 bracketing anchors; +13 vs 3600, LTC check of main +9.4). Before: **3600** (calibration ladder 2026-11-21 ~19:30, CI 3587-3614, 440 games, `f9c51e2`, run 44920906324: the new 11-anchor set 3487-3773 that brackets us, now the queue/workflow default; 7 shared mid anchors alone 3608, the 4 strong ones 3692-3773 alone 3585: we score 24-31% vs halogen-16/obsidian-16.0/plentychess-8.0.0). **The 3640/3618 figures below were inflated** by four weak anchors (3318-3417) we scored 84-95% against; read them as ~3600 and use only bracketed-anchor ladders from now on. Clover 9.1 (3725) is then ~125 ahead on CCRL vs -178 per node at 200k nodes (implies ~50 Elo from our NPS; unmeasured, check with a real-TC Clover match). Previously quoted: 3640 (ladder 2026-11-21 ~15:00, CI 3623-3657, 440 games, `013d1c2`: gen 11 pairwise net + pairwise speed stack on 58b2a6a; same 11 elite anchors; 44% vs koivisto-9.0 3631, the only anchor above us: stronger anchors needed before ~3660). Before: 3618 (ladder 2026-11-21 04:59, CI 3602-3633, 440 games, `58b2a6a`: gen 10 + tt-failow-nomove, lmp-killer-exempt, pv-no-tt-cut, qsearch-fail-soft, spsa-r14-interim + 3 speedups; 11 elite anchors, 45% vs koivisto-9.0 3631). Before: 3588 (ladder 2026-11-19, CI 3570-3605, 400 games, `2d945ec`: gen 10, 10 anchors). Before: 3581 (ladder 2026-11-16, CI 3563-3599, `b1961e7`: g9 + corrhist + futility + speed work). Before: 3562 (11-13), **3547 CCRL Blitz** (ladder 2026-11-12 20:07, CI 3528-3567, `419d29b`, g8h1024kb8m): **top 100 cleared** (3506). Before: **~3508**: three ladders in a row at or above the top-100 line: 3510 (CI 3492-3528, 2026-11-09 04:16, g7kb8m), 3508.5 (2026-11-09 22:11), 3507.6 (CI 3490-3525, 2026-11-10 12:24, `1b3d81d`). Each CI still overlaps 3506 alone; together they put us at the line, not clearly past it. Before: 3492 (2026-11-08), 3474, 3463 (g4). |
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
  - **Round 4 (20 Nov):** Clover gauge on 48d4541: **-205.8** [-221, -191] at 200k nodes, unchanged: our merges save nodes (time), not value per node. Depth-14 misses (696 positions vs our own depth-20 oracle): 26% differ, 64 significant; **no single mechanism dominates** (disabling LMR fixes 25/39 at 5.4x nodes; +2 plies fixes 24/39 at 3x). Misses are near-ties or deep tactics: the gap is value per node, not one switch-off bug. Real fixes queued: `qsearch-fail-soft` (qsearch stored alpha on fail-low; nodes -7.9%), `quiet-cont-neg`, `pv-no-tt-cut`, `lmr-index-1based`, `root-bad-captures-last`, `nmp-mate-clamp:nonreg`. **Round 6 (20 Nov eve):** history tables barely leave zero (95% of quiet history sums in [-4096, 4096); the -14000 bad-history LMR term almost never fires): sign informative, magnitude not. `lmr-history-sign` (reduce 1 more if quiet+cont history < 0; nodes -16.6%, EBF 1.59 -> 1.51, cutoff rates unchanged) queued `--front`. Held until `capture-history-r5` resolves (they build on it): `caphist-capture-lmr` (reduce captures with negative capture history, reach 16%), `caphist-see-prune` (bad-capture SEE threshold shifted by capture history). **History scale (20 Nov, game-like replay without clearing):** round 6's clustering partly an artifact (diag clears per position: ~80% of quiets in +-4k in games vs 95% fresh) but holds; positive magnitude IS informative (cut rate 6% at 0-2k -> 39% above 8k), negative side flat. Decay variants and the threat split: no ordering effect. Only a bigger shallow bonus improves ordering (AUC d<4 0.742 -> 0.788 at 160*d). Queued `history-bonus-linear-r2` (160*d cap 1536; the old test was +2.58 [-2.0, 7.2] rejected only by [0, 10] bounds); next value 224*d if it passes. Held: `history-malus-150` (mostly a reduction effect, overlaps lmr-history-sign). **Results (20 Nov): `pv-no-tt-cut` +24.8 merged, `qsearch-fail-soft` +14.9 merged, `quiet-cont-neg` -1.1 rejected.** Aspiration fails waste 8-13% of nodes (not linear widening); extensions don't explode. Tooling on `diag-r4` (DiagOff per mechanism, `go searchmoves`, scripts/diag/).
  - **Round 5 (20 Nov, `diag-r5`):** depth-14 total nodes on diag-mid300 are noise within ~±5% (a no-op change moved +5%); judge by cutoff/TT rates. Queued `:stc`: `killer-reset-per-parent` (killer-1 cut rate 42%->65%), `nmp-tt-store` (null cutoffs stored as lower bounds), `root-order-nodes` (root sorted by subtree nodes; late best moves -40%, more switching; LTC look if it passes). Capture history gave the best ordering gain (+0.9 pt first-move cut) but already failed twice in SPRT; not retested. Details on branch `docs-round5`.
  - **Bug audit (19-20 Nov):** three proven long-standing bugs, each queued `:stc`: repetition check looked past null moves (61/368 bench draws false); ep square hashed after every double push even when no capture possible (one position, several keys); quiet queen promotions pruned/reduced as quiets (624/2843 pruned). Node counts within noise, so SPRT decides. NPS 'drop' is only the first second (flat 1.41-1.45M from depth 15-27): not a bug.
  - In the queue: `prune-exempt-killers` (only killers/counter exempt from move-loop
    pruning; nodes -9.6%). Next: PV stability at depth (X3), reduction/extension balance at depth.
  - **Gauge cadence (owner, 21 Nov):** run the 200k-node Clover gauge (`peer-match.yml -f limit=nodes=200000 -f games=800`) whenever per-node value may have moved: after a new net merges, and after every ~3 merged search changes; it is the measure of the search gap, not the ladder. **Gauge 21 Nov on 1051d74 (gen 11 + speed stack): -178.2 [-192.3, -164.6]** (792 games, W53 D312 L427) vs -191.4 on 47ebb44: +13 per node from the net, search per node unchanged. Search agents started 21 Nov ~15:30 (branches only): history-table quality (`hist-*`) and depth erosion / reduction-extension balance (`depth-*`).
  - **History agent (21 Nov):** negative history carries no information (65-80% of searched quiets are negative from the malus; all negative buckets cut 0.1-0.2% at d4+); the sign is the signal, and h+ & ch+ cut 5.7% vs 1.5-1.8% one-positive at the same R; continuation history predicts better than main history at equal magnitude yet ordering halves it. **Method fix: single diag-mid300 runs are too noisy for ordering changes** (no-op moved first-move cut +0.3-0.5 pt); use the 962-position set (mid300 + diag-suite FENs), d13, 3 no-op seeds (sd ~0.08 pt; agent scratch `hist/ens.py`). Queued: `hist-lmr-both-pos:stc` (LMR -1 when h>0 & ch>0, +0-4), `hist-cont-order:stc` (h + ch, +0-3, weak), `hist-lmr-drop-bad:nonreg` (dead term). Dropped: malus scaling, depth-1 prune exemption, magnitude-graded LMR.
  - **Depth agent (21 Nov, tooling `diag-r8-depth`: `traceline`, DiagOff 1<<20 verify counters, scripts/diag/depth/):** d14/d15/d16 regret 5.78/4.68/3.86 cp (58/47/37 significant misses) vs d20. **Reductions/extensions do NOT compound with depth:** a reduced quiet failing low would beat alpha at full depth only 0.1-0.6% at every depth/node type; root LMR errors 0.1-0.5% (root-LMR-off control = noise). **Outlier: the negative singular extension** at expected-ALL d8-11 cuts falsely 10.9% (vs 0.5-1.1% for ordinary first moves); on 19 traced refutations. Queued `depth-negext-verify:stc` (ALL/PV nodes re-search a cutting neg-extended TT move at full depth; nodes x0.985, regret 5.78 -> 5.12, inside noise; bench unchanged 193878). Six other variants dropped (noise or worse). **Method: a 696-position self-oracle resolves ~1 ply, not +5 Elo; per-mechanism verify-rate tables are the clearer signal.** Implication: the Clover gap is not a broken reduction rule; it is diffuse value per node (ordering quality, eval at depth, what we search rather than how deep).
  - Tooling: `diag-search-stats` (feature `stats`, `stats-verify`), `scripts/search_stats.py`,
    `scripts/diag-mid300.txt`; Clover matches via `peer-match.yml`. New candidates go through
    the merge queue (`queue.py add <b>:stc`), not manual `ci.py sprt`.
- **Data: more beats newer.** Window test (same recipe/compute, val-g10 / valL1008):
  newest 1.27B only 0.023470 / 0.022888; whole 6.08B pool 0.023336 / 0.022723; whole pool
  with newest 2x **0.023299 / 0.022690**. Keep growing, don't drop old seeds, weight the
  newest 2x. **Running:** `gen10w.sh` = gen 10 recipe with newest 2x (GPU, 22:44 -> ~09:30
  20 Nov); gate it if it beats g10.
- **gen10w (newest 2x) does not beat g10 at full scale:** val-g10 0.022958 vs 0.022948, valL1008 0.022415 vs 0.022408, val.bin 0.025641 vs 0.025551. The window test's short-recipe weighting gain (~4e-5) vanishes with full training: no gate; **gen 11 = plain, unweighted pool11**. CUDA speed table (idle GPU): pairwise 2.07x (717k -> 1.49M), plain -0.9/-1.5% (being checked before merge).
- **Pairwise from scratch WORKS (pilot 20 Nov, shape recipe):** val-g10 **0.022927** / valL1008 **0.022299** vs plain 0.023470 / 0.022888 and dense head 0.022993; head alive (W1 97.5% nonzero). **Gen 11 started 13:03 as a full pairwise chain** on pool11 (`gen11.sh`, trainer a808d8a, ~17-19 h -> ~Sat 07:00). Pilot net under game test (`nnue-shape-h1024kb8m-pw` vs `nnue-shape-h1024kb8m`, fixed nodes + stc); rule: FN clearly positive and stc >= ~0 -> gen 11 continues pairwise, else abort and restart plain. **Fixed 200k nodes: +40.9 [+29.9, +52.0], 1152 games (H1). Stc 10+0.1: -1.7 [-7.3, +3.9], 4992 games (H0)**: the eval gain is real but the speed cost (-7.9% NPS x86, -20.6% M3; head alive -> ~37% chunks nonzero) eats it at stc. **LTC: +3.3 [-2.6, +9.2], 3600 games, PASSED (H1)**: the eval gain outweighs the speed cost at longer TC. **Gen 11 stays pairwise**: at gen11 `done` -> head_permute (pw-fast branch), embed with fixture, net-gate (fixed nodes + stc + LTC check). Desktop chain armed (`gpu-jobs/after-gen11.sh`): CUDA speed table, then recipe-length test with the pairwise head. (Commit 62c51e6 "Plan: ..." only holds the LTC line of tests/sprt-log.jsonl.) Speed levers if needed: integer L2 (needs QAT L2), FT pair-order permutation. **Speed pass done (`nnue-shape-h1024kb8m-pw-fast` bd57ff7, eval bit-identical):** offline FT pair permutation (`head_permute` test in src/nnue_head.rs; nonzero chunks 95.4 -> 66.9), SIMD L2/L3, cheaper NEON pair product: only **+3.6-4% M3, +0.7% x86** (desktop loaded). Head overhead ~280 ns/eval, mostly the fused FT update (memory/port-bound). Bigger wins need the trainer: QAT integer L2/L3, an L1 penalty for sparser pairwise inputs, narrower pairs. Run gen-11 nets through head_permute before embedding. **Trainer levers (20 Nov, `trainer-pw-levers` + engine `nnue-pw-int-l2`, default off, byte-identical):** QAT integer L2 (`--qat-l2`, net trailer "RFNT" gates integer L2 in the engine) is free in loss (+1e-5) and saves ~21 ns/eval on M3 but ~0 on x86 (AVX2 f32 L2 already cheap); activation sparsity (`--ft-l1`, new `--ft-l1-group`) costs ~3-4e-5 loss per % of chunks zeroed (pairwise gain over plain was ~5e-4), only <= 3e-5 plausible, ~10 ns per 10% chunks. Neither touches the dominant cost (fused FT pairwise update). So x86 speed (CI, CCRL) can't be rescued by these; pairwise lives or dies on the LTC test. GPU recipe staged: `~/pwlever-build/tools/nnue-trainer-cuda/pw-levers.sh ql2 sp1 sp2` (~2.5-3 h each), run only if pairwise survives LTC.
- **Generation cadence (audit 20 Nov):** net-gate gains g5..g10 = +30, +17, +31, +37, +23, **+12**; pool11 is only +10% over pool10. After gen 11, train the next generation when the pool has grown ~30-50% or a new architecture is ready; GPU time in between goes to pairwise speed/recipe experiments (Lazy Adam, recipe length, head variants). Datagen keeps running (data is the long-run constraint, ~20B for 3700-class nets). **Lazy Adam (20 Nov, branch `trainer-lazy-adam`, off by default, byte-identical default path):** no speed win: ~84% of FT rows / 92% of features occur in every 16384 batch, so at most ~16% of adam_ft is skippable; GPU smoke <1% per batch. CPU loss sanity (H256, 3 x 1M) identical to ~1e-5. Low priority; GPU A/B only if a slot is idle (recipe in the agent report, docs on the branch).
- **Recipe-length test RESULT (21 Nov, 5 of 8 arms; best-scale val-g11 / val-g10 / valL1008):** R (15 ep) 0.022501 / 0.022574 / 0.022021; **P4 (9 ep) 0.022500 / 0.022601 / 0.022028** (d -0.1e-5 / +2.7e-5 / +0.7e-5); P4s2 seed noise ~2e-5; K2 +13e-5, P4K2 +18e-5 (the kb8m 4-epoch stage matters). **P4 passes the pre-set rule: gen 12 uses plain4-m1-kb8m4 (~40% less GPU per generation)** unless S9/NM2 (still running, then P6, then sp1/sp2 sparsity levers) beat it. Reduced-scale match is conservative.
- **Recipe-length test (staged 20 Nov, started 21 Nov 04:36):** desktop `gpu-jobs/recipe-test.sh` (start by hand after gen 11: `cd ~/rustfish-data/gpu-jobs && nohup ./recipe-test.sh > recipe-test.out 2>&1 &`; it waits for `gen11.log` `done` + 3 idle GPU min). Gen-11 commands (a808d8a, pairwise, --half, H1024) on all of pool11 at epoch size N/6 (a uniform sample; the newest-1.27B subset repeats data, train < val from epoch 2). Arms (epochs): R plain10-m1-kb8m4 (15), K2 kb8m2 (13), P4 (9), P4K2 (7), P4s2 = P4 with seed 2 (noise), S9 kb8m factorised from scratch (9, compute-matched to P4), NM2 no m1 (vs K2), P6 (11); 59 epochs ~10-11 GPU h. Prior: window-B (4-1-2 at 1.27B) vs gen 10 (15 x 6.08B) = +3.9e-4 val-g10 for 1/10 compute, so compute is not free. Rule (best-scale val-g11, d = arm - R): adopt the cheapest arm with d <= max(3e-5, 2x seed noise) and no set worse by > 5e-5; 3e-5..1e-4 = check with a fixed-nodes match before using it; > 1e-4 = keep the stage. A reduced-scale match is conservative (fewer steps, steeper curve); a reduced-scale loss for the short arm may overstate the full-scale gap. Summary in `gpu-jobs/recipe-test/summary.log`.
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
  **Parked (owner, 20 Nov): no lever for the 1-CPU CCRL Blitz target, and SMP games cost ~4x CI per game.**
  - *When to pick it up:* single-thread rating within ~50 of the target (~3700 / 3751), or the owner rates the CCRL 4-CPU list or the Lichess bot, or the merge queue sits idle (< 2 waiting) with free shards.
  - *First step:* one scaling diagnostic on CI's 4-vCPU runners: Threads=4 vs Threads=1 at stc (then 2 vs 1). Sensible scaling: park again. Weak scaling (4T under ~+50): bug hunt. Check shared-TT races and torn entries, helper depth staggering and divergence, stop/ponder handling and node counting across threads, and time management with threads.
  - *Then:* SMP-specific SPRTs only for changes that touch threading; single-thread search work carries over automatically.

## Process

What works: `main` only moves through an SPRT against the current `main`
(merge queue); pentanomial LLR; pinned SHAs; absolute strength from a CCRL
anchor ladder logged by CI.

Rules added from the 2026-11-04..06 history:

- Net experiments follow `docs/nnue-experiments.md`: one variable, validation
  loss before games, every run logged including failures.
- No strategic pivot on a sample that cannot support it (e.g. 200 games).
- Every ladder anchor's score is sanity-checked against its neighbours.
- **README.md is part of every ladder update (owner, 2026-11-24):** when a
  ladder lands (or the targets, net architecture or data size change), update
  README.md's strength line, targets table and engine row in the same commit
  as plan.md's Strength row. It had sat at ~3508 (10 Nov) for two weeks.

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

## Provenance audit (26 Nov, external: two blind audits + one trajectory review, owner-supplied)

Verdict: independent implementation, no verbatim copying (0 shared constants, comments or code n-grams vs 9-21 peers). Flagged as close adaptations of GPL code: **(1) cuckoo upcoming-repetition table** = translation of an older GPL engine's implementation (same table size, hash functions, insertion loop, entry-count assert): **clean-room rewrite DONE** (`cleanroom-cycle` 118ce0c, src/cycle.rs: own multiplicative hash pair from a fixed splitmix seed, 2^13 slots, deterministic rebuild on failure, entry count enumerated not hard-coded; exact vs a brute-force reference on 40,769 positions; old tt.rs/search.rs code deleted unread; bench identical 202863; queued `:speedup` 26 Nov ~13:50); (2) TT replacement constants (age weight 8, 4-ply slack) match a GPL engine; (3) threat-indexed history layout + 16384 gravity divisor and the singular/multicut shape resemble another GPL engine. Also raised: no LICENSE/credits file, placeholder engine identity in the audited snapshot, quiet SEE prune margin 20*d^2 looks like a noisy-margin swap (A/B it). **Owner decisions pending:** licence (GPL-3 is the simple route), credits file, whether (2)/(3) get clean-room rewrites or are kept as published heuristics with credit.

**Release prep (owner, 27 Nov):** licence **Apache-2.0** (LICENSE, Cargo `license`), `CREDITS.md`, README licence + AI-disclosure sections (8e3461a); `id author dukky` (branch `release-id-author`, queued `:speedup`). **Clean-room rewrites of items (2) TT replacement and (3) history layout / SE shape: agent started 27 Nov ~11:20** (spec doc first, branches `cleanroom-tt`, `cleanroom-history-se`, non-regression at LTC). **Rename needed:** "Rustfish" is taken (Ronald de Man's 2018 Rust port of Stockfish, syzygy1/Rustfish); candidates with no engine found by a quick search: Rustduck (matches the Lichess account), Patina, Verdigris (Ferrite is taken). **Clean-room (27 Nov, interim; spec docs/cleanroom-tt-history-se.md):** `cleanroom-tt` 86c751d (bench 200459) STC nonreg running (-1.2 after 1152); `cleanroom-history` v1 **H0 -15.1** (2688 g), v2 7d555a5 (cap split 12000/3000) untested; `cleanroom-se` 5670e46 (extends 17% vs main 7.3%; riskiest) STC nonreg dispatched 27 Nov ~18:00. Exposure note: two indexing lines of the flagged history bodies appeared in a compiler error (layout order only, already in the permitted type), plus the first line of TT `store` and a doc comment; old functions deleted unread before tests passed. Next: history v2 STC after SE; LTC nonreg for each that passes STC. **Release builds (27 Nov, agent):** `release.yml` on main (9860399): Windows/Linux x86-64-v2 + v3, linux-aarch64, macos-arm64, macos-x86-64; name/version from Cargo.toml; a `v*` tag makes a DRAFT release only (owner publishes); no v4 tier (+0.75% vs v3 with runtime AVX-512, not worth a binary that crashes without AVX-512); README "Download and run". `portability` (8 MB UCI thread stack: plain Windows builds crashed with STATUS_STACK_OVERFLOW; v2/default x86 builds ~7x slow from non-inlined AVX2 closures, fixed; EvalFile empty keeps the embedded net) queued `:speedup --front`; merge before the first tag. **H1536 NPS on Windows = Linux, ~0.85x** (no Windows-specific penalty); Windows large pages sized (~70 lines in src/hugemem.rs, opt-in, needs Lock-pages privilege), not built. **Owner 27 Nov: ship unsigned** (README covers SmartScreen/Gatekeeper first run). **Owner chose Rustduck (27 Nov)**: `id name Rustduck <version>` queued (`release-id-name`); AI disclosure text approved. Full rename (crate/binary `rustduck`, README, scripts, CI, bot; ~460 mentions in 78 files) **on hold (owner, 27 Nov)**; repo, Dokku app and dashboard domain renames are separate owner decisions.

## Priorities (owner, 26 Nov: work highest expected Elo per time/compute first; re-rank when a result lands)

**Evening 27 Nov checklist (owner: low usage, machines + queue only, no new agents):**
- Watchers: H1536 SPRTs (abl4-h15 nodes + stc), SPSA final LTC (queue log), gen 12b LTC continuation ([0,5] wave 6+), clean-room TT + SE STCs, kb16 arm (ablate7; its output also shows ablate6/h15b), D2051 end (node-count test data), laptop stop at 19:00.
- **27 Nov 19:15: gen 12b MERGED by hand** (cbe686c, bench 217138): LTC [0, 5] H1 **+7.08, LLR 3.56, 4464 games**. **H1536 STC 10+0.1 vs abl4-ctlA: H0** (+1.2, LLR -3.06, 2688 g): the fixed-node gain does not survive its 0.85x NPS, so gen 13 = H1024 (warm start) unless h15b/speed work changes that. Lazy SMP merged 17:39. Clean-room TT nonreg -4.1 after 4992 (LLR -1.23, running). Laptop datagen stopped 19:00 (R3003 packed, 12.3M).
- **27 Nov 19:45 (owner: spare CPU short-term for tests/training, not datagen; no runner setup; CI at its 20-job cap):** the 19:30 extra datagen was stopped (L1017, D2070 partials packed). Opening-variety fine-tune `rv100m-s1` (= p1gpu q100m-s1 recipe on 50M R3002 + 50M R3001) trained on the GPU beside ablate7 (~/agent-runs/labels/ovft.sh; vhi loss 0.021381 vs q100m 0.019801, expected: vhi is standard-opening data, games decide). Local fixed-node SPRTs on the laptop: scratchpad `localtest/fixednodes.sh` (fastchess 1.8.2, UHO, sprt.yml adjudication, 200k nodes, [0, 10], 8 games at once): rv100m vs q100m first; then kb16 (arm binary from embed-cap.sh vs `engine-cap16` a274cfa + EvalFile ctlA.perm.bin), then l1w, node-count. Diagnostics only: they replace the planned CI 200k-node runs, freeing CI for merges. **19:40 owner: desktop datagen paused (SIGSTOP pid 119771, logged in runlong-2049.log; resume with kill -CONT) and its threads given to fastchess:** `~/agent-runs/localtest/run-stc.sh` (tmux `ftest`, 10 games at once): clean-room history v2 7d555a5 vs base e0b5a4b6 STC 10+0.1 [-5, 0] screen; only a pass goes to the CI nonreg. **Opening variety (D1) verdict, 27 Nov 19:50: H0, rv100m vs q100m -98.9 +/- 31.4 at 200k nodes (220 games, ptnml [14,48,34,13,1], local laptop SPRT [0,10])**; A/A q100m vs q100m exactly 0 (fixed nodes are deterministic: kit symmetric). Cause: randplies=4 games are drawish (W/D/L 26.7/46.6/26.6% vs D2041 31.9/40.2/32.2%), so a net tuned only on them loses; keep 8-9 random plies for datagen; R3001-R3003 can go in the pool at their ~3% share (not weighted up). Batch 2 fine-tunes running (`~/agent-runs/labels/ft2.sh`): g12c (gen 12b + 100M new D2049, lr 5e-5) vs g12b; lam03 / lam07 vs q100m; laptop chain `localtest/chain2.sh`. **Results 21:10 (local, 200k nodes, [0,10]): g12c vs g12b H0 -11.2 +/- 14.4 (716 g; polishing on new data at lambda 0.5 does not help); lam03 vs q100m H1 +18.8 +/- 13.4 (834 g, ptnml [5,84,203,111,14]): less game-result weight wins; lam07 vs q100m -14.7 after 494 (losing).** Lambda (result weight) is a gen 13 recipe lever. Batch 3 (`ft3.sh`): lam01, lam02 vs lam03; g12d = gen 12b + lam03 recipe vs g12b (merge candidate -> CI net-gate if it wins). h15b done 20:25 (H1536 shelved). **History v2 STC screen (desktop): H0 -10.8 +/- 7.4 (2278 g) -> no CI run; clean-room history needs a new design (v1 -15, v2 -11).** lam07 vs q100m H0 -14.9 +/- 15.8 (608 g). **kb16 (ablate7) best-scale val-g12 0.018845 / val-g11 0.022309 / valL1008 0.021876 vs ctlA 0.018919 / 0.022371 / 0.021966**; fixed-node test kb16 vs ctlA on the desktop (`run-nodes.sh`, engines from a274cfa: rf-cap-kb16m with kb16 embedded, rf-cap-kb8m + EvalFile ctlA) started 21:50. **Lambda probe (fine-tunes from ctlAm1, same 100M data, 200k nodes local): lam03 vs q100m(0.5) +18.8; lam07 vs q100m -14.9; lam02 vs lam03 -1.7 +/- 9.3 (H0); lam01 vs lam03 H1 +13.7 +/- 10.5 (1448 g) -> lower result weight keeps winning; lam00 vs lam01 running. g12d (gen 12b + lambda 0.3, lr 2e-4, 100M) vs g12b H0 -21.9: a polish of the finished net hurts; lambda belongs in gen 13's final stage(s). kb16 vs ctlA +6.1 +/- 5.2 at 5007 g (LLR 1.6, running).** **28 Nov ~01:00: clean-room STC nonregs both PASSED: TT -0.85 (18816 g, LLR 4.80), SE +1.92 (6336 g, LLR 4.28)**; LTC nonregs (`ci.py sprt <b> --ltc --nonreg`) armed to dispatch when tm-mtg-18 frees its CI slot (queue has no branch LTC-nonreg mode). lam00 vs lam01 H0 (-2.8, 1376 g): lambda optimum ~0.1. Gen 13 recipe candidate: kb16 + low lambda final stage + 12-epoch final + grown pool.
- On verdicts: merges via queue (`eval-scale-rule50:ltc`, `lazy-smp:speedup`, `spsa-r40-final:ltc` queued); gen 12b by hand if H1 at [0,5] (rebase, bench, ff); `tm-mtg-18` is the TM agent's (it reports).
- To do as slots free: `cleanroom-history` v2 STC (`ci.py sprt cleanroom-history --nonreg`); embed h15b (ablate6) and kb16/l1w (scratchpad `embed-cap.sh`) and run 200k-node SPRTs vs `abl4-ctlA`; README Threads note + AGENTS.md after lazy-smp merges; agents' SPRT results into tests/sprt-log.jsonl.
- After tonight's merges: CCRL ladder (TB on) + README update; then decide gen 13 (H1536 if its STC recovers; final stage 12 epochs; + kb16/l1w if they win) and datagen Rule50Div.


**Owner target (27 Nov): 3700 within a week if possible; no rented compute for now.** Own-hardware sprint: (1) tonight's merges (speed stack merged, TB ladder on, gen 12b + SPSA final + TM at LTC) +10-15; (2) **drawish-endgame / 50-move eval scaling** (agent started 27 Nov ~10:45, from the anchor-games conversion leak) +5-15; (3) datagen on main at max throughput (2.5k-node test decides Sat); (4) gen 13 = warm start from gen 12b on the grown pool with the recipe winners (+ H1536 if its test passes) Sat-Tue; (5) CI kept full (next SPSA on the new main, bug stacks). Realistic in a week: ~3650-3680.

**Dropped-item audit (27 Nov, owner: "bake them into the plan, and pick them up this time").** Things we planned and never did, ranked by expected Elo per compute; each has a trigger. Check this list at every milestone.

| Rank | Item | Origin | Why it slipped | Resource | Trigger / state |
|---|---|---|---|---|---|
| A1 | **More king buckets (kb16 mirrored)** reduced-scale arm vs ctlA | listed 26 Nov as a cheap capacity alternative to H1536 | never set up | agent (new layout in trainer + CUDA + engine) then GPU ~3 h | **built 27 Nov** (`nnue-capacity-kb16m-l1w` a274cfa, defaults bench-identical; kb16m: per-file buckets on ranks 1-2, edge/centre on 3-5, rank 6, ranks 7-8; H1=32 cargo feature `h1-32`, NPS ~0.97-0.98x). `ablate7.sh` after ablate6 (~21:20): kb16 ~22:30, l1w ~02:00-02:30; embed with scratchpad `embed-cap.sh <arm>`, then 200k-node SPRT vs `abl4-ctlA` |
| A2 | **Wider head L1 (16 -> 32)** reduced-scale arm | same | never set up | same agent (H1 is a compile-time const in trainer/CUDA/engine) + GPU ~3 h | with A1 (arm `l1w` in ablate7) |
| A3 | **Threat inputs, trained from the start + incremental + i8 threat rows** | owner 12 Nov: "fix and experiment during gen 9, bake into gen 10"; eval +58 at fixed depth, LTC -7 [-19, +5] after the speed pass (-20% NPS) | gen 10-12 went to king buckets, the pairwise head and data; never re-scheduled | agent (~1-2 days: rebase threat-inputs-v2 + cuda-threats onto the current recipe, incremental attack maps, i8 rows) then a reduced-scale arm | **gen 14 architecture candidate**; agent starts once usage allows after A1/A2 |
| B1 | **Clover per-node gauge** (owner cadence 21 Nov: after every net merge and every ~3 search merges) | last on main 21 Nov (-178.2) | not re-run after gen 12 and ~10 search merges | CI only | **done 27 Nov: -135.9 [-149.7, -122.6]** (792 games, main 4470e9a) vs -178.2 on 21 Nov: **+42 per node** (gen 12 net, SPSA interim, search merges) |
| B2 | `history-bonus-linear` next step **224*d** | 20 Nov: "next value 224*d if it passes"; 160*d merged +5.9 | never queued | CI (small branch) | next free CI lane |
| B3 | **Quiet SEE prune margin 20*d^2 A/B** | provenance audit 26 Nov | never queued | CI (small branch) | with the clean-room work |
| B4 | `lmr-protect-linear-r2` LTC | 25 Nov; chain script exited early | never ran | CI ~4 h | when CI is idle |
| B5 | `perf-head-fuse-screlu` (2841118, ~2% M3, bench-identical) | parked "behind the shape gate" 17 Nov | never merged | agent rebase + `:speedup` | next speed pass |
| A- | **Endgame eval scaling (27 Nov agent)** | anchor-game conversion leak | measurement: the leak is shuffling (draw share of +1.5 positions 15% at clock 0-9 -> 89% at clock 40-59), not pawn count | `eval-scale-rule50` cf811d6 (eval x (180 - clock)/180, UCI `Rule50Div`) **STC H1 +8.8 [+2.1, +15.5]**, queued `:ltc` 27 Nov ~15:00; `eval-scale-material` H0 -2.2, dropped. **Before the next datagen binary: decide Rule50Div for datagen (2000 = off keeps labels clock-free; the net cannot see the clock).** Later: add Rule50Div to SPSA | ltc queued |
| A0 | **Lazy SMP rewrite** (release blocker) | parked 20 Nov; trigger 'within ~50 of target or release' met 27 Nov | current Threads>1 = root splitting with a private TT clone per worker (no sharing, copy cost per move) | **DONE 27 Nov:** `lazy-smp` 51ceb57 (1T bench-identical 202863; helpers run their own ID on the shared TT with depth-offset divergence, persistent per-thread tables, summed node counts, Hash = total) **4T new vs 4T main +56.6 [+42.6, +70.8]** (576 g), **4T vs 1T +79.8 [+66.5, +93.3]**; stress 400 searches at 8T all legal; queued `:speedup --front`. CI `--threads` landed (f81cd81). Correction: main's TT was already shared (Arc + atomic slots); the waste was per-iteration respawns, rotated root splits, history copies. **Follow-ups:** (a) `src/nnue_head.rs:176` thread-local SCRATCH not 64-byte aligned on macOS debug = UB, check x86/Windows; (b) `src/cycle.rs:87` shift overflow at lo=63 (debug only; fix `!(u64::MAX >> (63 - lo))`); (c) README Threads note + AGENTS.md 'untested' line after merge; (d) tune helper divergence later | merged pending |
| C1 | Gen 12 passed only a non-regression LTC (bounds bug) | 27 Nov | ci.py bug, fixed 778a9e8 | none | covered by the next ladder |



| # | Work | Resource | Cost | Why this rank | State |
|---|---|---|---|---|---|
| 1 | **Tree-gap follow-ups**: `tree-lmr-all-node`, `tree-se-margin-4` at LTC; then aspiration 20, SEE-losing-check extension; ALL-node reduction depth scaling | CI + agent | ~4 h each | diagnosis done: gap = SE x1.53 (pays) x ALL-node reduction/check ext x1.27 x counting x1.26; the ALL-node saving grows with depth | round 1 H0 at LTC (lmr-all-node -6.4, se-margin-4 -18.5); **round 2 (26 Nov 23:00, nodes to d16 vs main on the 350 treegap positions):** `tree-check-ext-see` cb1594d (x0.96; STC run 45907418855), `tree-allnode-deep` 7a2ef20 (extra LMR ply at ALL nodes only at depth >= 8; x1.00 d8, x0.89 d16; STC run 45908874273; most likely to show at LTC), `tree-asp-20` f76b2fb (x0.98, aspiration is no tree lever on current main; low priority). **STC results 27 Nov:** `tree-allnode-deep` **H0 -7.2 [-14.3, -0.2]** (2688 games) dropped; `tree-check-ext-see` H0 +1.2 [-5.5, +7.8] at [0, 10] (2688), held (low prior; LTC only as filler when CI idles); `tree-asp-20` not tested. ALL-node reduction now failed 3 ways: closed. |
| 2 | **Queue at LTC**: `small-fixes-stack-3` (+ future bug stacks) | CI | ~4 h each | stacks were the most reliable gains (+15.1, +9.4); LTC so gains reach the ladder | queued |
| 3 | **Label quality** phase 0 (relabel tool + labelling speed) -> phase 1 (5k vs 20k/50k labels, same positions) | desktop CPU (datagen paused) + GPU + CI | 0.5 d + ~1-2 d | likely the biggest net lever (more 5k data now ~+3 per +17%); phase 0 decides compute/rental | **phase 1 losses done 27 Nov 01:00 (2 seeds agree to ~2e-5): deeper labels barely help, quantity dominates.** 100M-sample fine-tunes from ctlAm1 (vhi = 100k-node labels, lambda 0 / val-g12 lambda 0.5 / lambda 1): orig 5k 0.009429 / 0.021203 / 0.062299; r5k (main relabel) 0.009270 / 0.021154 / 0.062281; **r20k 0.009211 / 0.021215 / 0.062384** (5x CPU: -6e-5 on vhi vs r5k, worse on val-g12 and results); **q100m (equal CPU, 5x more positions) 0.007749 / 0.019530 / 0.060463** (-1.5e-3 everywhere; S arms repeat 20M 5x, train loss 0.013 = memorising). Verdict: spend CPU (and any rented compute) on more 5k games, not deeper labels; relabelling pools not worth it (r5k vs orig -5e-5 at 23 days for pool12). Games (200k nodes) optional, only if CI is idle. Tool on `datagen-relabel` (keep, not merged) |
| 4 | **SPSA final theta** -> `:ltc` | CI (6 shards, running) | to Fri ~14:00 | interim already +5.7 STC; final tuned on the new terms | **SPSA ended 27 Nov 11:18 (round 40; missed for ~2 h: no watcher was armed).** `spsa-r40-final` (8 defaults change vs main: FutMD1 120, FutMD2 219, RFP 33/66/219, ThreatEscape 8071, NullRedDepthDiv 3, LmpD2 8; bench 176508) queued `:ltc --front` 13:30; queue back to 18 shards, lanes 1/1 (2f527bd). GPU: `ablate6.sh` (h15b = H1536 with the 8-epoch final stage, ~2.7 h) armed after batch 4 |
| 5 | **H1536 shape test** (reduced scale, H1024 vs H1536, same recipe; needs >= +20 at 200k nodes) | GPU ~4 h + CI | after batch 3 | measured cost 0.84x NPS (~10-15 Elo); only worth a full gen if the eval gain clears it | **ablate4 done 27 Nov 12:37: H1536 val-g12 0.018755 / val-g11 0.022124 / valL1008 0.021683 vs ctlA 0.018919 / 0.022371 / 0.021966 (best scale; -1.6e-4, 2x the 8-epoch arm's gain, which was +16.9 at 200k nodes).** Embedded on main: `abl4-h15` 99c167e (H=1536, bench 158353), `abl4-ctlA` 8387cbf; SPRTs vs ctlA at 200k nodes and STC 10+0.1 started 12:53 (the STC includes the speed cost) |
| 5b | **NNUE inference speed, H1024 + H1536** (owner, 26 Nov 22:30): AVX2 / AVX-512 (Zen 5: VNNI, VBMI) kernels, FT update memory traffic (prefetch, accumulator footprint, lazy updates), eval-cache row cost, sparse L1; exact only, `:speedup` | agent + desktop CPU | ~1 day | H1536 costs 0.84x NPS (FT update 1.58x, memory-bound); every point lowers the bar for the bigger net; H1024 gains count now | running (agent, doc docs/nnue-speed-2026-11-27.md) |
| 6 | **gen 12b** (final stage 8 epochs: arm5 +16.9 at fixed nodes) -> net-gate + LTC; then batch 4 (final stage 12, pool8 out combined) | GPU ~9.5 h + CI | tonight | first recipe winner; direct merge candidate | GPU from ~15:00 |
| 7 | **Opening variety D1** (randplies 4 vs 8) | laptop 4 threads | ~2.5 days data + 1 GPU arm | free compute; unknown sign | laptop compute ends 27 Nov 19:00 (owner): armed stop at 19:00 packs the partial R3002 (R3001 complete, 335 MB .rfp); then train the D1 arm on R3001+R3002 vs an equal-size slice of D2041+ |
| 7b | **Hardware use (owner, 27 Nov):** (1) GPU never idle: `gpu-jobs/ablate5.sh` (batch 4: arm9 final stage 12 ep from ctlAm1; arm10 pool8 out + final 8 ep from arm8m1; ~4.5 h) armed behind ablate4; (2) desktop datagen on main 96d4748 (+8% NPS) from 27 Nov 10:21, `runlong-2049.sh` in tmux datagen7, 8 threads (D2045 on 44bbe8b ended early at 208.7k games when the old loop was stopped: packed, kept); (3) **node-count test**: D2049 5k vs D2050 + D2051 2.5k nodes (equal CPU, wall times in local/runlong-2049.log), then 100M-sample fine-tunes from ctlAm1 as in the label test (vhi / val-g12 / results-only) | desktop | ~16 h data + 1 h GPU | label test says quantity >> label depth; 2.5k could double data per CPU | running |
| 8 | **Gen 13** = winners of 3/6/7 (+ H1536 if 5 says yes) | GPU ~11 h+ | after 3, 6, 7 | combines the net wins | waiting |
| 9 | `lmr-protect-linear-r2` LTC | CI | ~4 h | last check of a lever whose STC was -2.2; lowest EV, runs only when CI is free | chained |
| 10 | **Syzygy: DONE (26 Nov, `syzygy-cleanroom` bfb25f6, queued `:speedup`, bench-identical with SyzygyPath empty).** Format fully recovered from prose + experiments (docs/syzygy-format.md, source log); WDL 4999/4999 vs the lichess oracle at clock 0, 51,942/52,592 with the 50-move clock (all diffs = DTZ full-move rounding at clock+DTZ = 101); DTZ exact or 1 ply off (format granularity); WDL probe ~0.9 us, DTZ 10-14 us; in-search WDL after zeroing moves, root DTZ filter. **CI wiring done (ea13a3f, default off):** cached 3-4-5 files (key `syzygy-345-v1`, md5-checked), `ci.py sprt --tb cand|both`, `ci.py ladder --tb`, queue mode `ltc-gainer-tb`; anchors without SyzygyPath (play without TBs): reckless-0.7.0, carp-3.0.1, altair-7.0.0. **LTC main+TB vs main (bounds [-5, 0]): H1 +1.28 [-2.39, +4.95], 7056 games (27 Nov 08:45): TBs do not regress. Queue ladder now runs with `tb: true` (queue-settings, 27 Nov); first TB ladder = the next one, note the switch in ladder-log.** Then switch the ladder to `--tb` (CCRL plays with TBs) and keep it on from then (note the switch in ladder-log). Was: **Syzygy 3-4-5 probing, strict clean room from prose** (owner, 26 Nov; agent, `syzygy-cleanroom`: format spec doc from prose + reverse-engineering, WDL validated vs the lichess tablebase API, root DTZ, `SyzygyPath` off by default) | agent + laptop | ~2 days | external review: +3..15 at CCRL (CCRL plays with TBs); needs TB files in the CI ladder/SPRT workflows to measure. **Fallback (owner, 26 Nov): if the prose route stalls, use Fathom (MIT) as the format reference** — own Rust code, credit in CREDITS, keep Fathom's MIT notice beside anything close to a translation; never Stockfish's tbprobe (GPL) | running |
| - | Rented compute | money | ~$35-100 | only if phase 0/1 says higher-node labels pay (owner decision) | waiting on 3 |

## Deferred actions (each has a trigger; re-check this list whenever a job finishes)

- **27 Nov: gen 12b net-gate +9.85 [0.71, 19.01] capped (1728 games), same as gen 12; rebased on main (with speed stack) 594d56e. Bounds bug found:** since 39e8d6a (30 Oct) `ci.py sprt --ltc --elo0 0 --elo1 5` silently ran **[-5, 0]** (fixed 778a9e8). So the gen 12 LTC (+2.87, merged) and main+TB LTC were non-regression tests, and the first gen 12b LTC `nnue-g12bh1024kb8mpw-ltc-1127-0925` (run 45973613458) is [-5, 0] too; owner: no second run, re-evaluate the same games. It stopped H1 at [-5, 0] after wave 5: **+5.71 [-0.65, +12.08], 2736 games**, LLR at [0, 5] = 1.53 (continue), so wave 6 was dispatched by hand with [0, 5] and the same SHAs + prior (run 46012512862).
- **Gen 12 MERGED 26 Nov 00:00 (main 5ac58f2, bench 207737): LTC H1 +2.87 [-1.9, +7.64]** (5328 games, LLR 4.54; wave 4 report rerun after a transient dispatch error). +17% data bought ~+3 at LTC: data scaling still positive but small, so more of the same data is weak value; recipe ablations first, rented datagen only for new kinds of data (D1/D2). Was: net-gate capped at **+9.65 [0.21, 19.11]** (1728 games, 13:34): LTC SPRT `ci.py sprt nnue-g12h1024kb8mpw --ltc --elo0 0 --elo1 5` started 13:40 (same settings as gen 11's passing LTC). H1 -> merge by hand (rebase, bench, ff, as gen 11) and data scaling holds (then: rented compute for datagen is worth it); H0 -> keep gen 11.

- **LMR-protection recall test: done (runs 25 Nov 01:00-08:00 on the desktop, `~/agent-runs/policy/out/D1`, `U1`-`U4`; summarised 25 Nov 20:00).** Labels (D1, 962 positions, 151k positive test nodes, oracle best = 0.96% of reduced moves, 4.7 reduced moves per positive node): the fitted linear score (split by position) ranks the oracle best first in **81.7%** (top-2 90.5%) vs history 58.7%, child-eval-alpha 78.4%, random 60.6%; threshold curve: 50% recall protects 7.2% of reduced moves, 80% recall 19.3%. **So the label gate passes on paper.** In search (depth 13, 962 positions, se ~1.9 cp): LMR +2 unprotected (arm 19) -31% nodes, +5.6 cp; oracle-protected (18) -30%, -1.2; **real linear scorer (22, t=690) -21%, +3.2; child-eval margin 3 (23) -21%, +2.9**; LMR +1 unprotected (14) -19%, +4.2. The real scorers sit close to the plain "reduce a bit more" trade-off (edge ~1 cp, inside noise): the oracle bound does not survive a cheap scorer. One engine branch is still worth an SPRT (linear protection + LMR +1/+2 for the rest, measured at time, not nodes); after that the reductions lever is closed unless it shows Elo.
- **SPSA final theta tested (~Fri 27 Nov) -> tree-size program (owner, 25 Nov), in this order:** (1) **LMR-protection recall test** (no engine change; tooling on `policy-oracle`, desktop runs): dump the nodes where the depth-17 oracle's best move is among the LMR-reduced moves (~0.5% of them) with the features available there (history, cont-history, SEE, checks, static and child eval vs alpha, TT bound, killer/counter, threats) plus matched non-best reduced moves; measure top-1/top-2 precision/recall of (a) existing signals, (b) a fitted linear score (split by position), (c) a tiny MLP, against the random-0.5% control. Gate: ~50-80% recall at useful precision. (2) If it passes: engine branches pairing LMR +1/+2 with the protection rule (oracle bound: LMR +2 with the best move protected = -30% nodes at equal quality; 50% recall still -31% / +2.8 cp), measured at fixed time and by SPRT/LTC, not nodes. (3) If it fails: smaller aggression steps with the measured safety rules only. Context: rustfish needs 2.1x Clover's nodes to depth 12; ordering near-optimal, SEE/ProbCut margins calibrated, exact speed work exhausted.

- **Gen 12 done (~25 Nov 13:10; laptop trigger runs ~/rustfish-data/gen12-embed.sh and queues the net-gate).** The GPU then runs `gpu-jobs/ablate1.sh` automatically (armed 25 Nov 10:36): ctlA, ctlB (recipe R, seeds 1/2) and arm0 (single-stage kb8m factorised from scratch, 15 epochs), reduced scale = pool12 stream with 1.5B-position epochs, val-g12; ~3 h each, done ~23:00; log `gpu-jobs/ablate1/summary.log`, nets + permuted copies in `~/rustfish-data/ablate1/`. **Then (needs a session):** embed each permuted net on a branch (gen12-embed.sh pattern, no record row needed for diagnostics) and run fixed-node SPRTs: ctlB vs ctlA (seed noise), arm0 vs ctlA (`ci.py sprt <branch> --nodes 200000 --base <ctlA branch>`); then batch 2 (plain 10 vs 4, lambda 0.7, m1 dropped, lr).

- **Ablation batch 1, controls done (25 Nov 20:03):** best-scale val-g12 / val-g11 / valL1008: **ctlA 0.018919 / 0.022371 / 0.021966, ctlB (seed 2) 0.018968 / 0.022410 / 0.021918** (seed noise ~5e-5, sign flips on valL1008; full-scale gen 12: 0.018820). Embedded (scratchpad `embed-abl.sh`): `abl1-ctlA` 6ca6bd1 (bench 187268), `abl1-ctlB` da29b44 (bench 212977). arm0 (single stage, 15 epochs) due ~midnight (watcher armed). **When the gen 12 LTC frees the runners:** `ci.py sprt abl1-ctlB --base abl1-ctlA --nodes 200000` (seed-noise yardstick in Elo), then arm0 vs ctlA the same way. Data arm D1: `datagen-randplies` (option `randplies=N`, default byte-identical) queued `:speedup` 21:15.
- **26 Nov evening: `tree-se-margin-4` LTC H0 -18.52 [-25.82, -11.22]** (2160 games): fewer singular extensions lose badly with time too, consistent with SE paying (-47 without SE at fixed nodes). Both tree-gap candidates failed at LTC: the tree-size gap is not recovered by cutting SE or adding ALL-node reduction; next tree ideas must keep the SE signal and target ALL nodes more selectively (or the check extension, neutral at fixed nodes). `syzygy-cleanroom` MERGED 19:29 (bench-identical; TB off by default).
- **26 Nov afternoon:** `cleanroom-cycle` MERGED 13:43 (bench-identical; provenance item 1 closed). **`tree-lmr-all-node` LTC H0 -6.43 [-13.78, +0.91]** (2160 games; +10 at fixed nodes did not survive time); `tree-se-margin-4` on LTC since 15:46. **arm4 (plain peak lr 2e-3) val-g12 0.019200 vs 0.018919: clearly worse (~3e-4, arm0-like), dropped without games.** gen12b training since 15:17 (~01:00). SPSA round 29/40.
- **Ablation batch 4 (ablate5, 27 Nov 17:27), best-scale val-g12 (ctlA 0.018919, arm5 8 ep 0.018845):** **arm9 final stage 12 epochs 0.018800** (another -4.5e-5: 8 is not the plateau); arm10 pool8 out + 8 ep 0.018848 (= arm5: no additive gain on val; val does not rank arms at this resolution, games decide). Gen 13 recipe candidate: final stage 12 epochs. Games (200k nodes vs ctlA) when CI frees up.
- **Ablation batch 3 games (200k nodes vs ctlA, 26 Nov ~13:50): arm5 final kb8m stage 8 epochs (vs 4) +16.9 [+5.6, +28.2] H1 (1152 games); arm8 pool8 out +6.7 [-0.7, +14.1] H1 (2688 games)** — the first recipe winners (val-g12 0.018845 / 0.018876 vs 0.018919). arm4 (plain lr 2e-3) due ~15:00. **Acting on arm5 now: `gpu-jobs/gen12b.sh`** = gen 12 exactly with an 8-epoch final stage from gen12/m1.bin (~9.5 h after ablate3, ~01:00) -> embed, net-gate, LTC. pool8-out and further final-stage length (12?) go to gen 13 / batch 4.
- **26 Nov 11:42: `small-fixes-stack-3` MERGED at LTC +10.94 [3.81, 18.09]** (2160 games, first `:ltc` merge); `speed-stack-r3` merged 11:33 (+1.4-1.8% x86 NPS, exact).
- **Tree-gap diagnosis (26 Nov, `diag-tree-gap` 8eb572b, docs/tree-gap-2026-11-26.md; tooling only):** our nodes / Clover's to equal depth = **2.0 (d8-10), 2.44 (d12), 3.06 (d16), 3.80 (d20)**; worst in piece-heavy positions (2.5x opening, 1.4x endgame); best-move convergence per ply is similar (a Clover ply is not shallower); at 1 s Clover reaches d23.8, we d18.9. The d12 2.44x = **x1.26 node counting** (each frontier leaf counted twice: negamax depth 0 + qsearch; per visited position we are ~20-25% slower than Clover, not at parity; the 200k-node gauge gives us ~20% less work, -178 ~ -155..-160 in equal work) **x 1.53 singular extensions** (x1.75 at d16; 42% of tries singular; all our seldepth excess; but removing them = -47 at 200k nodes, -28 at 10+0.1: they pay) **x 1.27 rest** (less reduction at expected-ALL nodes x1.11 d12 -> x1.29 d20; check extension x1.07; NMP/LMP/RFP/futility/ProbCut/qsearch each <= 5%). Fixed-node games: +1 LMR ply at ALL nodes +10.1 [-3, +23] (nodes x0.81 d16); SE margin 4*depth -5.7 [-19, +8] (x0.75); no check extension +1.8 (x0.87). **Queued `:ltc` 26 Nov ~13:00: `tree-lmr-all-node` 03946db (bench 170305), `tree-se-margin-4` b3f6eb4 (bench 184664).** Next proposals: aspiration 20, check extension skipped for SEE-losing checks; never cut SE outright. Gauge note: compare peers in equal node units (or ~250k vs 200k).
- **H1536 inference cost (26 Nov, `nnue-h1536-speed` 2bbfdbe, docs/h1536-cost.md; synthetic widened gen-12 net, sparsity matched):** **NPS 0.83-0.85x H1024** (fixed depth -16.6% AVX-512 / -17.2% AVX2; fixed time 100 ms-3 s 0.85, flat in length and hash). NNUE cost per node 1.43x (FT update+activation 1.58x, memory-bound; eval-cache probe 2x from wider rows). Speed cost ~10-15 Elo STC / 8-12 LTC (assumed 40-60 Elo per doubling). **Gate: a reduced-scale H1024 vs H1536 shape test (same data/recipe) must show >= +20 at 200k nodes before any full 1536 generation**; real net gated at LTC. Cheaper capacity alternatives: wider head L1 (16 -> 32) or more king buckets.
- **`speed-stack-r3` 2a008e1 queued `:speedup` 26 Nov ~12:00** (bench 216602 + eval hashes identical to main, all kernel paths): x86 pairwise head: no permute after packus (W1 in packus order) +0.6-0.8%, VNNI dot for the output sum +0.4-0.6%, integer-rounded finish +0.4%, popcnt chunk count +0.45%; stack **+1.4..+1.8% x86 NPS** (3 CI runs, 6 runners each), arm flat. Dropped: L1 4-in-flight, sparse L2 (-0.6%). Profile: FT update+activation 18.5%, head 6.5%; what remains in the head is thin. Doc: speed-audit round 5 on `speed-diag`.
- **Ablation batch 2 games (200k nodes, 26 Nov ~10:30), each vs ctlA, 2688 games, all H0 at [-5, 5]: arm3 m1 dropped -6.0 [-13.1, +1.2]; arm1 plain 4 epochs -5.6 [-12.8, +1.7]; arm2 lambda 0.7 -5.2 [-12.7, +2.4]** (seed noise ctlB -0.1). Every part of recipe R earns ~5 Elo: keep m1, plain 10, lambda 0.5. Val losses were within 3e-5 of ctlA for all three, so val loss does not rank recipe arms at this resolution: games only. `lmr-protect-linear-r2` LTC did not start (chain script exited early); lowest priority, run when CI is idle.
- **Owner direction 26 Nov ~08:30 -> started:** (1) **search changes accepted at LTC** (new queue mode `ltc` 60+0.6 [0, 5]; standing rule); (2) **tree-gap diagnosis** (agent, branch `diag-tree-gap`, doc docs/tree-gap-2026-11-26.md: why we need 2.1x Clover's nodes to depth 12, measured black-box + own instrumentation, sized causes, LTC proposals); (3) **label quality** (agent, `datagen-relabel`: relabel command + labelling throughput on the desktop, then same positions relabelled at 20k/50k nodes vs 5k, identical fine-tunes from ctlAm1, fixed-node SPRTs, Elo per CPU-hour vs more 5k data; owner OK to pause desktop datagen with STOP/CONT for it); (4) **opening variety D1 on the laptop** (`~/rustfish-data/runlong-R3001.sh`: R3001-R3005, randplies=4, engine 44bbe8b + randplies, output cmp-identical to the desktop binary, 4 threads nice 20, ~2300 pos/s ~0.2B/day; control = desktop D2041-D2048, same engine, 8-9 plies); (5) **H1536 inference cost** (agent, `nnue-h1536-speed`, CI speed-compare: NPS ratio, break-even eval gain) before training a bigger net. **`small-fixes-stack-3` 9e5847e queued `:ltc`** (bench 202863): LMR clamp/LMP low-limit crash fixes (exact at defaults), root static eval recorded (improving at ply 2), futility-pruned moves fold their futility value into the fail-soft best, LMP mate guard; desktop 400-game screen -8.7 [-29, +12]. SPSA unaffected by the LmrMinDepth=1 panic (theta 2.9, c 0.5).
- **Ladder 26 Nov 08:00 (main f7a45bc: gen 12 + small-fixes-stack + SPSA r14 interim): 3620 [3605, 3635]** vs 3618 on eb3b4a4 (25 Nov). The SPRT gains since (+9.4 STC, +2.9 LTC, +5.7 STC) do not show at 120+1 within the ladder CI (+-15): short-TC self-play gains shrink against the anchors; ~80 to 3700.
- **26 Nov 03:03: `spsa-r14b-interim` MERGED +5.72 [2.1, 9.34]** (9600 games; round-14 theta on gen 12 main). The SPSA run continues on f6beb3e to round 40 (Fri); final theta then tested against the new main. **`lmr-protect-linear` STC H0 -2.16 [-7.17, 2.86]** (4992 games); rebased on the new main as `lmr-protect-linear-r2` c8eb771 (bench 166880) for the one planned LTC [0, 5] (chained after the batch 2 arm SPRTs); H0 there -> reductions lever closed. **CCRL ladder dispatched 06:45** on main (gen 12 + SPSA interim): README + ladder-log with the result.
- **Ablation batch 2 (26 Nov 06:04), best-scale val-g12 / val-g11 / valL1008 (ctlA 0.018919 / 0.022371 / 0.021966):** arm3 m1 dropped 0.018939 / 0.022395 / 0.021974; arm1 plain 4 epochs 0.018945 / 0.022390 / 0.021971 (denser: nonzero chunks 24.7% vs 18.3%, slower inference); arm2 lambda 0.7 0.018948 / 0.022541 / 0.022079. All within ~3e-5 on val-g12: games decide (`abl2-arm3/arm1/arm2` vs `abl1-ctlA`, 200k nodes, running). **Batch 3 started 06:36** (`gpu-jobs/ablate3.sh`, ~8.5 h): arm5 final stage 8 epochs (from ctlAm1), arm8 pool8 out, arm4 plain peak lr 2e-3.
- **Ablation batch 1 games (200k nodes, 26 Nov 01:25): seed noise ctlB vs ctlA -0.1 [-5.4, +5.3]** (4992 games: two seeds of the same recipe are equal in games, so a ~5e-5 val difference is noise); **single stage arm0 vs ctlA -47.3 [-68.3, -26.7] H0** (384 games). The staged chain is right; arm0 is closed (its 2x sparsity cannot buy back ~47 Elo). Calibration: ~4e-4 val-g12 within one recipe family ~ -47 Elo at fixed nodes.
- **Ablation batch 1 done (25 Nov 23:35): arm0 (single stage, kb8m factorise from scratch, 15 epochs) val-g12 0.019310 / val-g11 0.022824 / valL1008 0.022333: worse than the chain by ~4e-4 (~8x seed noise) on every set**, but much sparser (active pairwise 3.4% vs 6.6%; nonzero chunks 9.8% vs 18%: faster inference). Embedded `abl1-arm0` 15898a7 (bench 225604). Games decide: fixed-node SPRTs ctlB vs ctlA, arm0 vs ctlA after the gen 12 LTC. **Batch 2 started 23:37** (`gpu-jobs/ablate2.sh`, ~6.5 h): arm3 m1 dropped (reuses ctlA's plain stage), arm1 plain 4 epochs, arm2 lambda 0.7.
- **`lmr-protect-linear` 0abaa71 queued stc 23:37** (bench 179274; linear protection scorer from the recall test, unprotected reduced quiets +1; desktop pilot: x0.907 nodes at depth 14, regret -0.4 cp (se 0.9), 400 games at 40 ms +5 [-19, +30]; t=6900 and unprotected +1 trended -18). If STC is flat: one LTC [0, 5]; then close the reductions lever.
- **SPSA 25 Nov (run 45558507472):** round-14 theta queued 26 Nov ~00:45 as `spsa-r14b-interim:stc --front` (bench 216602; 12 of 19 change: FutMD1 94->114, FutMD2 228->220, RFP 51/126/268 -> 43/82/257, NullReductionBase 3->4, NullReductionDepthDiv 3->4, ThreatEscape 7563->8085, LmrPvNode 1->0, LmrNotImproving 1->0, LmpD2 10->9, LmpD3 14->13); when the run ends (~Fri 27 Nov afternoon) or is stopped, restore 18 shards and lanes 1/1 (revert the 25 Nov queue-settings commit's hunk) and queue the final theta.
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
| Search: next experiments from round 3 (X3 PV stability; reduction/extension balance at depth); gauge = 200k-node Clover match after each batch. **Gauge 21 Nov on 47ebb44: -191.4 [-206.8, -176.7]** (792 games, 200k nodes; after pv-no-tt-cut + qsearch-fail-soft) vs -205.8 [-221, -191] on 48d4541: +14, CIs overlap; the value-per-node gap is still ~190 | next gauge after the next batch of merges |
| D2026 (gen-9 engine) -> D2027-D2030 on the TT-fixed engine (`rustfish-6daf892-native` is a symlink to `rustfish-f7d7a51-native`) | automatic: `runlong-2027.sh` in tmux `datagen5` on the desktop |
| **In progress (20 Nov eve, 4 agents, branches only, nothing queued yet):** (1) engine NPS speedups on the main plain net (bench-identical, M3 + x86 interleaved; land via queue `speedup` mode, no runners); (2) CUDA trainer hot-path kernels (plain + pairwise; byte-identical or flagged; real speed table on the idle GPU after gen 11); (3) pairwise trainer levers `trainer-pw-levers`: activation-sparsity penalty + QAT integer L2 with engine integer L2 (GPU A/B after gen 11); (4) search round 6: capture history in pruning/LMR (on `capture-history-r5`) + 2-3 counter-driven candidates | agent reports; queue only the best 1-2 of round 6 while >4 items wait (CI drains ~10 tests/day until the SPSA ends) |
| Pairwise LTC test (`nnue-shape-h1024kb8m-pw` vs `-main`, `ci.py sprt --ltc`) | result: >= 0 -> gen 11 stays pairwise (run head_permute, embed, net-gate); < 0 -> stop gen 11, no plain gen 11, GPU to experiments |
| CCRL ladder | auto after the 5th merge since 19 Nov (4/5 after qsearch-fail-soft +14.9) |
| CUDA hot-path pass (`perf-cuda-hot` a5e3c86: forward 8 cols/thread, backward_ft word-wide offsets, backward_rest shared-mem chunk data; byte-identical across 9 configs). Idle-GPU speed table armed on the desktop (`~/trainerperf-build/idle-table.sh --wait`, starts after gen11.log `^done`, ~8 min, builds base/cand/fwd/bft/rest) | `~/trainerperf-build/speed-idle.txt`: merge only the changes that are faster (the shared-GPU smoke had cand slower: suspect forward occupancy) |
| Desktop datagen continues: `runlong-2031.sh` (tmux datagen6) waits for runlong-2027 (pid 57946, D2030 ~36% at 19:00) and runs D2031-D2036 on current main `bin/rustfish-aeae3de-native` (bench 268722; today's search fixes, g10 net), 10 threads | after D2036 (~3 days): new script on the then-current main |
| Engine NPS pass (20 Nov, bench 268722 + evalhash identical; queued `:speedup`): `speed-nnue-row-prefetch` (+2.6% M3 / +2.0% x86), `speed-picker-branchless` (+1.7% / +1.2%), `speed-movegen-sink` (+0.3% / +1.5%); x86 measured under datagen load (±3% per run, 10 interleaved rounds). M3 profile: NNUE 43% (0.56 evals/node, fused update+dot ~480 ns vs ~300 ns floor), picker 17%, movegen 7%, SEE 7%. Next: evals with 2+ pending plies (8% of time), picker scoring on an idle x86 with a profiler | confirm the combined NPS on main after the merges (interleaved bench, idle desktop) |
| **Time management at 120+1 (20 Nov, ladder PGNs, 1680 games):** budget = clock/20 + 3/4 inc at every stage, so we front-load (6.3 s/move early vs anchors 5.7) and drain to ~11 s by move 50 (anchors 27 s); median end clock 8.6 s vs 18.1 s; no forfeits; time trouble follows lost positions (120/125) rather than causing them. Branches: `tm-horizon-by-move` (horizon max(20, 40 - 2*move/5)) under a manual `ci.py sprt --ltc` [0, 5] (60+0.6); `tm-inc-reserve` (hold back min(10 inc, clock/4)) held. Note: queue `ltc-gainer` bounds [0, 10] are too wide for single-digit gains (same issue as history-bonus-linear); consider [0, 5] | horizon LTC result: pass -> merge by hand (rebase, bench, ff), then a ladder; then test inc-reserve on top |
| **Clover games (20 Nov, 4 peer-match runs, 2376 games; docs/search-diagnosis, scripts/diag/clover-misses.epd + epd_depth.py):** losses are slow squeezes out of unbalanced openings (2.3% from the worse side at 60+0.6, edge fades 3-4 cp/move when better), decided early (58-69% in opening/early middlegame, 95% without any single 1.5-pawn blunder); our eval stays too optimistic (agrees the game is lost ~3 moves after Clover, 82-95 cp more optimistic just before). Endgames/draw rules not the leak. Critical blunders are ~2 plies past our horizon; **we print exact 0.00 on 5.5-6.4% of moves vs Clover 0.1-0.3%, 8/24 blunders at 0.00; 7/23 blunders never chosen by a fresh search (in-game state)**. **H2/H3 result (replays reproduce the games exactly):** the 0.00 rate is cosmetic (Clover prints +-0.01 for draws: |s|<=0.01 is 20% of its moves vs 7.4% ours; blunders-at-0 a selection effect); carry-over HELPS (game state searches ~1.7 plies deeper; differing moves +5.5 cp better with game state), no TT/history fix justified. Two real bugs queued `:nonreg`: `negamax-rep-floor` (after the cuckoo raise to 0, fail-soft returns below 0 could cause false cutoffs; bench 266587) and `fifty-move-mate` (50-move draw returned before checking mate). `rep-preroot-3fold` (rule choice) not queued. **H1 (20 Nov eve): null on the EPD.** Dummy controls (LMR shifted on a meaningless hash condition) move the 23-blunder score as much as any real variant (noise ~+-2 solved, +-1 ply), so 23 positions can't judge small changes; build a set of hundreds or rely on SPRT. Blunder classes: exchange/sacrifice sequences 5/11 (no prune change helped: likely eval/qsearch on exchange resolution), threat-with-tempo 2, passed pawns 2, quiet maneuvers 3. Reserve branches (low prior, not queued while the queue is long): `h1-see-quiet-checks` (quiet SEE prune keeps checks; #06 solved at depth 7 vs 14 robustly; nodes +3%, bench 275940) and `h1-adv-pawn-push` (pushes to rank 6+ exempt from pruning, 1 ply less LMR; bench 264842). Eval optimism in squeezes -> net pipeline question | agent report; fixes via queue (correctness bugs first) |
| **Anchor games (21 Nov, ladder 58b2a6a, branch `diag-anchor-games`, scripts/diag/anchor_games.py; reproduces the Clover round-6 numbers):** the Clover patterns do NOT generalise. Top-5 anchors (200 games, 52%, +21 =166 -13): losses slow (13/13 no 1.5-pawn blunder, 12/13 as Black) but late (median move 53, 0 in the opening); not too optimistic when losing (agree lost a median 1 move after the anchor; calibrated at -2). **New leak: optimism when BETTER and failure to convert.** At our +2 we score 0.74 vs 0.95 at the anchor's +2; our eval reached >=+1.5 in 26/166 top-5 draws (anchor 5/166), 9 stayed >+1.5 for 10-53 moves; drawish material (KRP v KR x6, R v B, RB v R, bishop endings). Clear wins convert (111/1/0). Time: we finish ~10 s richer (25 vs 15 s in draws; 2.80 vs 3.00 s/move), which CONFLICTS with the 20 Nov TM analysis (move-50 clock 11.5 vs 27.5 s, older anchor set): `tm-inc-reserve` saves more time, the opposite of this finding; its LTC decides. Small samples (13 losses, 26 draws). **Next:** (a) net: eval scale on drawish material advantages (WDL weight lambda > 0.5 arm after the recipe test; draw-heavy endgame data); (b) engine: our own drawishness scaling of the eval (material class / 50-move counter), screened on these draw positions then SPRT; (c) TM: spend the spare clock if tm-inc-reserve fails | agent report |
| **Gen 11 gate (21 Nov 05:15): net-gate H0, -3.4 [-12.5, +5.6]** at 5+0.05 (1728 games), despite val -0.0005; same pattern as the pilot (stc -1.7, LTC +3.3): the pairwise head's NPS cost dominates at short TC. My watcher read sprt-log on main, but the queue logs to queue-state log.jsonl: missed for 3 h (fixed: watcher on queue-state log). **Fixed nodes 200k: +21.4 [+9.9, +33.0] H1** (1152 games): the net is clearly better per node; the 5+0.05 loss is the pairwise speed cost (x86 ~8%, structural). 08:25: started `ci.py sprt nnue-g11h1024kb8mpw --nodes 200000` (eval quality without the speed cost) and `--ltc --elo0 0 --elo1 5` (decider). **LTC 60+0.6 [0,5] H1: +9.8 [+1.9, +17.8]** (1872 games): merged by hand 21 Nov ~12:20 as 9e160dc (bench 193878), then the bench/evalhash-identical speed stack `nnue-pw-speed-r2` + `nnue-pw-speed-x86` on top (8366484; +5% M3, +1-2% x86); net record verdict accepted. **Lesson: architecture changes that cost NPS must be gated at LTC; queue a net-gate-ltc mode.** Ladder on 8366484 started. **CUDA speed table (idle GPU, 21 Nov 04:27):** perf-cuda-hot builds all slower or equal (plain: base 2.10M, fwd 2.08M, rest 2.09M, bft 1.88M, cand 1.75M; pw: base 1.55M, fwd 1.54M, rest 1.53M, bft 1.41M, cand 1.41M pos/s): nothing to merge, branch dropped. Recipe test running since ~04:30 (arms R, K2 done) | watchers |
| **GPU plan to gen 12 (21 Nov):** cadence rule = gen 12 when the pool is +30-50% (~25-28 Nov; now +8%). Meanwhile: (1) sp2 (--ft-l1-group 3e-5) to ~17:00; **sp1 (1e-5): nonzero L1 chunks 28.8% -> 16.8%, val-g10 0.022745 vs pilot 0.022927, valL1008 0.022164 vs 0.022299** (same recipe/data, from scratch; one run). **sp2 (3e-5): chunks 9.9%, val-g10 0.022783, valL1008 0.022234** (still below the pilot). (2) **x86 bench-13 NPS via EvalFile, 20 interleaved rounds, core 11 under datagen load: sp1 +1.6% [IQR +0.2, +6.8], sp2 +5.0% [+1.4, +11.6] vs the pilot** (the head costs ~8% x86): sp2 chosen (+5% NPS ~ +5 Elo vs ~1-2 Elo of loss). (3) **gen11s started 21 Nov 17:13 with SP=3e-5** on 87 files, N=7,238,768,472 (pool11 + D2028-D2032 packed + L1013), ~10.5 h -> ~04:00 Sun; then embed (head_permute, record row), gate stc + LTC. D2028-D2032 .bin now redundant (~14 GB, delete later) = P4 recipe + sparsity on pool11 + D2028-D2032 + L1013 (~7.3B, ~10-11 h; `gpu-jobs/gen11s.sh`, SP=..., staged not started); gate stc + LTC. (4) WDL lambda arms for drawish conversion (judge by fixed-node games, not loss: targets differ). (5) reduced-scale H1536 shape test + NPS. (6) gen 12 with the winners. Only gen11s and the best lambda arm go to the queue | staged |
| **Queue 21 Nov 17:07:** `spsa-r28-interim` **H0 -0.08 [-3.92, 3.76]** (8640 games): rounds 15-28 on the stale base moved nothing; stopping it was right. **Next SPSA deferred** until the queue backlog (15 branches) drains (~Sun): it costs 6 of 18 shards, and a fresh run should tune the new search terms (history sign rules, neg-ext verify, capture-history divisor) on current main, which needs those UCI tunables first. `capture-history-r5` dispatched (front) | queue |
| **Queue 21 Nov 14:49:** `lmr-index-1based` **H0 -0.72 [-4.41, 2.97]** (9600 games), dropped. `spsa-r28-interim` dispatched 14:50 (front, 18 shards) | queue |
| **21 Nov ~14:30 reallocation:** (1) **SPSA stopped at round 28** (owner OK; base 605b560 stale after gen 11 + search merges): round-28 theta (iteration 6720/9600) as `spsa-r28-interim` (8 defaults change vs r14: FutMD1 82, FutMD2 235, RFP 43/109/259, ThreatEscape 7318, LmpD2 9, LmpD3 15; bench 195624) queued stc `--front`; queue back to **18 shards, lanes 1/1** (08754bb). Next SPSA on current main after the interim verdict, adding new tunables. (2) **Recipe test stopped after S9 started** (owner OK: P4 already passes; S9 is compute-matched, NM2/P6 can't save much). Sparsity levers sp1/sp2 chained, but the wrapper's own command line matched `pgrep -f nnue-trainer-cuda` (GPU idle ~14:25-14:50): wrapper killed, pw-levers proceeds. (3) **Ladder mistake:** a bare `ci.py ladder` ran the old weak anchors.json set (861/864 wins, '3351' logged invalid); defaults fixed everywhere (e4cd703); elite rerun run 44910372339 on 013d1c2 (= gen 11 + speed stack). Prediction ~3628 [3615-3645] | watchers |
| **Disk 21 Nov 13:45:** laptop cleaned (owner OK): deleted pool7.bin, old seed .bin/dirs, L1000-L1007 and the raw L1009/L1011/L1012 txt/bin (all inside desktop pool8 or present there as .rfp); L1013.rfp copied to the desktop (size verified). Laptop 17 -> 175 GB free; kept every .rfp, nets, logs. Desktop pool8.bin is now the only copy of pool7-era data. Desktop: WSL 67 GB free but Windows C: 30 GB free (97%): watch it, pack D2028-D2032 .bin to .rfp before gen 12 | watch C: |
| **Resources 21 Nov 09:50:** laptop datagen had stopped (L1013 finished 20 Nov 22:24, nothing chained; ~11 h idle). L1013 packed to .rfp (86.3M positions; txt deleted: laptop disk 13 GB free); `~/rustfish-data/runlong-L1014.sh` runs L1014-L1017 (6 threads, 5000 nodes, main 2613426 binary, packs + deletes each txt). Desktop GPU: after recipe-test (pid 91414) -> `pw-levers.sh sp1 sp2` (FT-L1 chunk group-lasso: fewer nonzero chunks = cheaper pairwise head, ~6 h; `gpu-jobs/after-recipe.out`). Caffeinate extended to ~Mon 09:40 | chained |
| **Queue 21 Nov 09:11:** `lmr-history-sign` **H0 -0.07 [-3.7, 3.6]** (9600 games): sign-based extra reduction does nothing; released `history-malus-150` (queued stc). `capture-history-r5` hit a rebase conflict with the branchless picker (push_either); resolved by hand (score = SEE stage score + caphist/8 into push_either), bench 199044, tests pass, re-queued `--front`. Fixed: queue results are in queue-state `log.jsonl` (not main's sprt-log); watchers read that | queue |
| **Pairwise inference speed (21 Nov, `nnue-pw-speed-r2` on the gen-11 branch 6a3473a, bench 193878 + evalhash unchanged):** profile on gen 11: fused FT update 36.6% of the search thread, evaluate 12%; most head cost is bookkeeping/codegen (nonzero-chunk masks, paired-load addressing), not the pair arithmetic. ldp/stp loads (+1.3..1.7%) + cheaper chunk masks (+3.0%): **+5.0% M3 NPS**. NEON-only: x86 (the CI runners, where the gate/ladder are played) unchanged and unmeasured. **x86 (`nnue-pw-speed-x86` on r2, exact):** cheaper AVX2 chunk list (+0.7%) + fused L1/L2/L3 kernel (+0.2%, near noise): **+0.9% median / +1.75% paired** in the CI-like config (x86-64-v3, no VNNI/AVX-512; desktop under load 13, no perf available); native AVX-512/VNNI build unchanged. Fused update near its load/store limit; 16-bit L1 sums ruled out (gen-11 W1 pair max 254). The x86 pairwise cost is structural: further gains need a cheaper head design, not tuning. Both branches land as a speedup after gen 11 merges | after gen 11 verdict |
| **Drawish-edge eval scaling (21 Nov, agent):** the engine had no NNUE scaling (only insufficient material, 50-move, repetition). Fit on 25.4k root scores from the 440 ladder games (>=+1.5: 428 drawn vs 4952 won; dummy = permuted factor): `eval-drawish-pawns` (eval x min(128, 64 + 16 x pawns of the favoured side)/128; UCI DrawScaleBase/DrawScalePawn; bench 211070) log-loss -0.0235 [-0.042, -0.009] vs dummy +0.007, AUC 0.866 -> 0.914; `eval-drawish-hmc` (x (200 - hmc)/200; DrawHmcHorizon; bench 217413) -0.0094 [-0.014, -0.005]. Applied before corrhist in negamax+qsearch, TT keeps raw eval, datagen off. Caveats: root-score proxy; signal is 'endgame scores inflated' generally (total pawns/phase fit as well); top-5 subset alone not significant. **Queued `eval-drawish-pawns:stc`** (prior ~30%); hmc held (prior ~20%). If stc rejects narrowly, retry at ltc: the leak shows at 120+1 vs other engines. Params to SPSA if it passes | queue |
| **TM (21 Nov):** `tm-horizon-by-move` LTC [0,5] **inconclusive at the cap, -2.7 [-6.7, +1.3]** over 7056 games: not a gain, dropped. `tm-inc-reserve` (keep 10 increments out of the sudden-death budget; independent of the horizon change) rebased on eb1bce0 (bench 204182 unchanged), LTC [0,5] **H0, -10.2 [-16.5, -3.8]** over 2736 games: holding time back hurts, consistent with the anchor-game finding that we already finish ~10 s richer; dropped. Next TM idea is the opposite direction (spend the spare clock: a slightly larger base fraction), one variable, LTC | dropped |
| **TM (21 Nov eve):** `tm-full-inc` (soft target = clock/mtg + the whole increment, was 3/4; bench 193878 unchanged) **LTC [0,5] H1 +2.9 [-2.5, +8.3]**, 3600 games; merged `160aecb` by hand (rebased on f9c51e2) | merged |
| **Capture history (21 Nov 21:10):** `capture-history-r5` stc [0,5] **inconclusive at the cap, +1.64 [-0.87, +4.15]**, 20160 games: rejected alone. Its two users rebuilt on r5 + main b6a552f and queued as stacks vs main: `caphist-capture-lmr-r5` (bench 187799), `caphist-see-prune-r5` (bench 187135); old held branches on 2099851 superseded | queued |
| **Queue 21 Nov 22:40:** `root-bad-captures-last` **H0 +0.6 [-2.84, +4.04]**, 10944 games: rejected. `nmp-tt-store` **H0 -4.22 [-9.84, +1.39]**, 4032 games. | rejected |
| **Search strategy reset (21 Nov 23:20, owner):** of 30 search tests since 16 Nov, the 3 big wins were bug fixes (pv-no-tt-cut, qsearch-fail-soft, prune-exempt-killers); ~6 landed +0.6..+2.1 (too small for [0,5] alone); ~20 bolt-on features were 0 or negative. Changes: (1) **`combo-small-positives`** (capture-history-r5 + order-captures-r2 + queen-promo-tactical on 8ffaefd, bench 183334) queued `--front`; `queen-promo-tactical:nonreg` and `history-malus-150` removed from the queue (subsumed / overlaps the H0 lmr-history-sign). (2) **Two agents (branches only, nothing queued without review):** a coherent move-ordering + pruning program (measure first: cutoff stage/index, EBF, verify rates, quality-controlled node savings; waste map in docs/tree-audit-2026-11-22.md; `tree-*` branches) and a systematic correctness bug hunt by invariant (TT bounds/mates, windows, draws, extensions, picker, qsearch, abort, NNUE incremental vs refresh; `bug-*` branches). (3) Queue is ~20 h deep: prefer bug fixes and stacks over single features; SPSA after feature batches | running |
| **Bug-hunt agent (22 Nov 00:10, report in session):** no severe bugs; differential tests clean (zobrist child/null hash, NNUE incremental vs refresh over 77k positions + 18k null moves, gives_check over 6M moves, picker yields each legal move exactly once over 311k pickers, mate-distance invariant, no TT stores after stop). Fixes: `bug-iir-tt-depth` (fail-low nodes store no move, so IIR re-searches at D-1 and stores D-1, which a depth-D probe can never use; probe asks D-1 when IIR will apply; bench 188159) and `bug-se-no-tt-refine` (singular verification refined its static eval with the excluded move's own Lower bound, 53% of verifications; bench 193737) queued stc `--front`; `bug-root-failow-nomove` (root aspiration fail-low stored a failed move as TT move; bench 192700) queued nonreg. Merged as bench-identical no-ops: `fefb81a` (no ProbCut in singular verification; latent) and `2f73d42` (`go infinite` waits for `stop`). Unbranched leads: razoring at d2 returns qsearch fail-highs (10.6%); check extension doubled in in-check verification searches; single legal move spends normal time; movestogo clamp 20; `go movetime` ignores Move Overhead; qsearch skips quiet queen promotions; 5-bit TT age. Built-in `perfttest` is shallow (8 positions, 1 ms): deepen it | queued |
| **Queue 22 Nov 00:45:** `root-order-nodes` **H0 +0.63 [-2.78, +4.05]**, 10944 games: rejected. `bug-iir-tt-depth` dispatched | rejected |
| **Tree agent interim (22 Nov 01:15, tooling `tree-diag` 74333d7):** 962 pos d13: 139M nodes (qsearch 44%), EBF 1.72, first-move cut 80.4%, 1.51 moves per fail-high; waste is 22% of fail-high-loop nodes at CUT d1-3, 45% of d4+ waste under a TT move that failed first. **Ordering is saturated**: killer/counter/ch-weight knobs inside noise (sd 1.2% nodes); equal captures before killers is right. Only lever that moves nodes is bad-capture placement (BADPOS: -4.3..-10.4% nodes) but regret rises in step (5.78 -> 6.43..7.02; fixed-node -6.9/-2.9 +-12): it buys extra LMR/LMP, not better order (`tree-bad-before-quiets` = evidence, not a candidate). **Pruning verify rates:** futility d>=3 0.0-0.1% (too conservative), RFP CUT d>=3 <0.2%, but **razoring wrong 13.5% at the CUT d1 threshold** and RFP ALL d1 7.1%. Screens running: bad1q (bad captures excluded from the LMP/LMR index), razcut100, futd3-75, rfpcut80, rfpall125, razor-d2 q<=alpha; then fixed-node games, then `tree-*` branches | running |
| **Queue 22 Nov 01:38: `bug-iir-tt-depth` H1 +6.53 [+1.95, +11.11]**, 6336 games, merged `dd5698b` (moveless fail-low TT entry one ply shallower now cuts at an IIR node). Next: `bug-se-no-tt-refine`. Tree screens: futd3-75 (-3.0% nodes, verify error ~0) and razcut100 (nodes flat, cut rate +0.2 pt) are candidates pending regret + fixed-node games; bad1q small | merged |
| **Tree audit final (22 Nov 02:10; docs/tree-audit-2026-11-22.md on `tree-diag` ef19135):** no bugs; quiet ordering saturated (all knobs inside noise; equal captures before killers is right); node savings from reordering cost quality in step. Two margins keep quality (regret vs depth-20 oracle, main 5.78, no-op 5.56-5.86): `tree-futility-d3` (futility x3/4 at d>=3; 5.51, nodes -3.0%) and `tree-rfp-cut-margin` (RFP x0.8 at cut nodes d>=3; 5.81, nodes -1.8%). **`tree-pruning-stack` (c2f16aa, both, bench 184622) queued stc `--front`**; singles held as fallback. Dropped: razor-no-failhigh-return (regret 6.82), bad-capture-index (6.53), razor-cut-margin (5.89 borderline, not stacked), RFP x1.25 at ALL, razret d2. Next if the stack passes: re-derive futility d1-2 and SEE quiet prune d>=3 (0.0-0.2% error) the same way; then the failing TT move at d4+ (45% of d4+ waste; deep lower-bound TT moves below beta cut only 51%) | queued |
| **22 Nov (session logged out ~02:15-19:40; queue ran unattended):** `tree-pruning-stack` **H0 -2.76** (regret-clean margin tightening still lost: pruning harder needs safety first); `bug-se-no-tt-refine` H0 -0.36; `combo-small-positives` cap +2.27 [-0.25, 4.8]; `hist-cont-order` cap +2.34 [-0.15, 4.84]; `hist-lmr-both-pos` +0.76, `depth-negext-verify` +0.87, `caphist-capture-lmr-r5` -2.69, `eval-drawish-pawns` -9.25 rejected. **Merged: `history-bonus-linear-r2` +5.92 [1.44, 10.41]**, nonreg `nmp-mate-clamp` +4.57, `negamax-rep-floor` +2.76, `fifty-move-mate` -0.97. **Scheduled LTC check of main e8a3495 +9.42 [0.84, 18.02]**; ladder q-ladder-1122-1908 running (partial 3616 on 9 anchors vs 3607 for the 21 Nov run on the same 9). **Gen 11s** done 03:15 (GPU idle since, unwatched): best-scale val-g11 0.022346 / val-g10 0.022426 / valL1008 0.021896 = parity with gen 11; nonzero L1 chunks 6.4% vs 34.5%; embedded `nnue-g11sh1024kb8mpw` df16c31 (bench 194049, fixture mean 0.256 cp), row recorded, **net-gate queued --front** (LTC after if close). **Peer analysis (owner's separate agent, no peer source, directions only):** NPS 519k vs Clover 842k locally (~1.6x, fits the ~50 Elo speed share); eval not ahead at depth (fixed nodes -133/-213/-196 at 20k/200k/1M); gap is node economy: missing safety mechanisms that make aggressive reduction safe, and expensive qsearch (44% of our nodes). **Two agents started 22 Nov ~20:10 (branches only):** qsearch economy (`qs-*`: verify-rate-derived qsearch prunes, time-to-depth) and extension/reduction safety (`ext-*`: miss analysis, signal precision/recall, safety+aggression pairs) | running |
| **Datagen + GPU (22 Nov 20:00):** desktop datagen had stopped at 14:53 (runlong-2031 finished D2036, no follow-on: 5 h lost); restarted `runlong-2037.sh` D2037-D2048 on main 44bbe8b (gen 11 net), 10 threads, .rfp packing; ends ~Wed 25 Nov (re-arm then). Laptop: owner using it, L1016 stopped at 60.5M positions and packed; **laptop datagen stopped altogether (owner, 22 Nov 20:10;** restart when the laptop is free). **Pool now ~7.84B** (gen 11s 7.24B + D2033-D2036 360M + L1014-L1016 245M); desktop only ~0.34B/day: 9B ~Thu 26 Nov morning, +30% (9.4B) ~Thu 26 evening, 10B ~Sat 28. **GPU: `g11sl7`** = gen 11s's final kb8m stage retrained from the same p4m1 with WDL lambda 0.7 (only change; 13 Nov H512 sweep lead, never game-tested), started 19:55, ~4.5 h; then embed + net-gate vs gen 11s | running |
| **22-24 Nov:** queue (unattended): **`caphist-see-prune-r5` merged +4.64 [1.22, 8.05]** (capture history + capture-history-shifted SEE prune now on main); nonreg merged `hist-lmr-drop-bad` -1.08, `bug-root-failow-nomove` -0.76; **gen 11s nonreg REJECTED -6.09 [-10.78, -1.40]** (6336 games): the sparse net loses despite loss parity, so the FT-L1 group-sparsity lever costs strength in games; do not use it for gen 12 without a game-tested fix (g11sl7 lambda-0.7 arm, built on gen 11s, abandoned). The queue then sat empty ~42 h. **Power cut 22 Nov 22:31 -> 24 Nov ~20:20:** desktop datagen (D2038 at 16.4M, packed), the g11sl7 training and all agent runs died; session also restarted. GPU now at its default power limit (owner will restore the 125 W cap): short experiments only, no long training. Datagen restarted `runlong-2039.sh` D2039-D2048, 8 threads. **Qsearch agent (22 Nov) final:** only a calibrated per-victim deficit prune keeps quality (nodes -2.3%, regret inside the no-op spread 5.34-6.59, sd 0.48); `qs-deficit-prune` c88d0da queued stc --front (`qs-deficit-pawn` fallback); qsearch is 36.5% of nodes on current main; finding (WITHDRAWN 24 Nov, see below): "the NNUE eval scale is not the piece-value scale". **Extension agent round 1 (cut off by the session limit):** `ext-child-eval-lmr` (LMR one less when child static eval > alpha-25; regret 23.39 vs 28.36 no-op, nodes x1.13) and `ext-lmr-see-loss` (one more for SEE<0 late quiets; 25.78, nodes x0.979) queued stc with their stack `ext-lmr-pair` (bench 179684). **New agents 24 Nov 20:40 (desktop, nice 19, branches only):** extension safety round 2 (`ext-*`) and an eval-scale audit of ProbCut and the negamax SEE prune margins (`see-*`/`probcut-*`) | running |
| **GPU 24 Nov evening (owner set the cap to 150 W):** power benchmark `gpu-jobs/pwbench.sh <label>` (gen 11s stages, 400M-position epochs): **275 W: plain 2.21M pos/s, kb8m 1.96M; draws a median 262-267 W, 1.92 GHz, 77-80 C** **150 W: plain 1.86M, kb8m 1.74M; 149 W, 1.52 GHz, 63-64 C** = 275 W is +19% / +12% for 1.8x the power. (The gen 11s 125 W figures 1.90M / 1.72M are not comparable: that run had the FT-L1 sparsity term and full epochs.) **Gen 12 launched** (`gpu-jobs/gen12.sh`): gen 11's exact recipe R and trainer a808d8a, no sparsity, ONE change = data: pool12 = pool11 + D2028-D2036 + L1013-L1016 (~7.85B, +17%); **val-g12** = first 500k positions of D2037 + D2038 (main 44bbe8b; sha 89409054902da379; gen 11 scores 0.018727), both seeds held out. ~16-17 h at 150 W, done ~Wed afternoon; then embed + net-gate (LTC if close). **Disk:** D2028-D2036 raw .bin removed after position-exact .rfp checks (53 -> ~78 GB free); pool8.bin (102 GB, 3.20B shuffled positions = 40% of positions, 60% of bytes): on 2 GB / 512 MB slices **.rfp 1.64x, zstd -3 1.80x (decompress 609 MB/s = ~19M pos/s, fine to stream into training), xz -6 2.09x (34 min per 2 GB: archive only)**; sorting the records helps xz (2.08 -> 2.78x) but barely .rfp (1.64 -> 1.74x); zstd -19 pending. Plan: zstd for pool8 (~45 GB saved) once the trainer can read a source through `zstd -dc` (pool8 is pre-shuffled, so a sequential pipe works); not urgent (68 GB free, perpetual .rfp datagen ~1.2 GB/day). Later experiment: a net trained without pool8 (old gen 7-8 data) vs with, in games | running |
| **Ext agent round 2 interim (24 Nov ~21:30):** the clip300 oracle metric is noisier than thought (no-op sd ~1.6 cp, range 3.6; base itself biased by the searchmoves judge: compare only with TNOISE no-ops) so every arm needs 3-4 replicates. Round-1 SEE rule (`ext-lmr-see-loss`) did NOT replicate on current main: it and `ext-lmr-pair` removed from the queue; `ext-child-eval-lmr` weakly replicated, stays queued. **Best lead: singular-extension margin 3*depth instead of 2 (`semg3`): clip300 23.81 vs no-op mean 27.28 at x0.89 nodes (~2.7 sd), replicates running.** SE min depth 10: neutral quality at -17% nodes (replicate pending). Null-move threat variants: noise (sampler: threats are not a warning sign for NMP cutoffs). Fact found: the neg-ext-above-beta+M branch is unreachable (multi-cut returns first) | running |
| **Gen 12 training (24 Nov 21:05):** 94 files, N=7,844,440,110, 150 W, 99% GPU; 15 epochs, done ~Wed 25 Nov 15:00-16:00 (watcher armed). **Speed agent started 24 Nov ~21:30** (branches `speed-*`, tooling `speed-diag`): profile on x86, exact speedups only (bench nodes + evalhash = main), NPS measured on GitHub runners (interleaved, pgo-check style), never on the shared desktop; target: Clover runs 1.6x our NPS on the same machine (~40-50 Elo) | running |
| **SEE-scale audit (24 Nov, `see-diag` 420dcf0, docs/see-scale-audit-2026-11-24.md): no scale mismatch, no miscalibration, no candidates.** Removing each piece from 962 positions (18k pieces) drops the static eval by a median P 98, N 342, B 379, R 511, Q 1024: the same scale as the SEE table (Q/B ~12-15% above it); the qsearch audit's 185/500/625/855 does not reproduce (withdrawn). Eval-scale SEE ranks moves with the same error as piece SEE (+-0.03 pt). ProbCut cuts are wrong 0.0-0.4%, loosening saves <=0.7% nodes. Quiet SEE prune (-20 d^2) is safe and does nothing at d7-8; a cap at 300 cp: -1.3% nodes (borderline). Capture SEE prune: error flat ~0.5% at d3-6, capture history strongly predictive (supports the 22 Nov merge); only unexplained error = exchange sacrifices at d1 (3.3-3.7%, 0.2% of the tree). Regret runs for the cap/ProbCut arms still running on the desktop (`~/agent-runs/qs/see/p3.sh`, p4.sh); branch `see-quiet-cap300` only if they hold. Conclusion: SEE/ProbCut margins are not where the gap is | done |
| **Queue 24 Nov 21:39:** `qs-deficit-prune` **H0 -0.44 [-4.29, 3.41]** (8640 games): rejected; qsearch closed as a lever. `ext-child-eval-lmr` hit a rebase conflict (needs-human); re-queued `--front` as `ext-child-eval-lmr-r2` (the conflict-resolved commit from the pair build, bench 182825) | queued |
| **Ext agent round 2 final (24 Nov ~23:00, read by the main session; 6 pooled no-ops: clip300 27.09, sd 1.32; z = node-adjusted excess / sd):** **`semg3` (SE margin 3*depth) did NOT replicate**: first run z -3.0, replicates -0.1 / +0.8 / +0.4 (it saves 11% nodes, but only as much quality as searching 11% less). `cevs` (child static eval > alpha-25 reduces one less) replicates weakly: z -1.2 / -1.2 / -0.7 (`ext-child-eval-lmr-r2` queued). Everything else inside noise or worse (IIR, null-move threat/near-beta, SE min depth, SEE LMR, CTTS/WTA pairs; nmall25 and iirmd3 clearly worse). No new branch; the extension/reduction program found no lever beyond cevs. Draft doc in the session scratchpad (wt-doc), not pushed | done |
| **Queue 25 Nov 00:52: `combo-small-positives-r2` MERGED +15.09 [6.62, 23.57]** (1728 games, early stop: likely optimistic) = hist-cont-order + order-captures-r2 + queen-promo-tactical stacked on main (each had capped at ~+2 alone): stacking capped small positives works. `ext-child-eval-lmr-r2` capped at **+2.86 [0.36, 5.37]** over 20160 games (CI excludes 0) -> re-queued as nonreg (merge on no-regression, the gain is already evidenced). **Next ladder due** (README with it) | merged |
| **25 Nov 01:30 (after the session-usage reset):** speed agent resumed (AVX-512 fused accumulator update next; branches from the new main). **Policy move-ordering v0 agent started** (owner, from an external review: small learned move scorer in the MovePicker, trained on deeper-search rankings; phases gated): A) upper bound first: how many nodes would oracle/perfect ordering save at equal quality (stop if < ~5%: the tree audit found quiet ordering near-saturated); B) inference cost of a target-size MLP on CI runners (stop if > ~3% NPS); C) only then labels (deeper search, held out by seed) + model + offline ranking metrics. Branches `policy-*`, doc docs/policy-order-v0.md; nothing queued | running |
| **Speed agent report (25 Nov, `speed-diag` b3e71a5, docs/speed-audit-2026-11-24.md; CI tool `speed-compare.yml`: interleaved bench NPS on 6 runners + arm, refuses bench/evalhash changes; A/A control -0.04..+0.26%):** time map x86: NNUE ~40% (fused FT update + pairwise activation 16.5%, FT row prefetch 9.7%, head 6.2%, refresh 3.1%), move ordering ~18% (MovePicker::new 8.9% incl. SEE 3.5-3.9%), movegen 6.5%, make/undo 4.8%, TT 3.2%; IPC 2.44, ~2000 cycles/node. **`speed-stack-r1` 7adfb33 (prefetch 8 of 32 FT-row lines + i16 history tables) +2.51% x86 NPS [1.21, 4.05], arm +0.94%, bench/evalhash identical: queued `speedup`.** Dropped: AVX-512 kernels (Zen 5 only), SoA picker -5.6%, corr prefetch. Node-changing idea for SPRT: yield the TT move before generating/scoring the rest (~10% of time on TT-move cutoffs) | queued |
| **25 Nov ~02:30:** `speed-stack-r1` **merged as speedup** (+2.5% NPS). Speed round 2: **`speed-tt-first` 5cb399f** (picker yields a legality-checked TT move before generating anything; bench 183218; new test checks tt_move_legal against generation) NPS -2.4% but **time to depth 13 -6.4% on every runner**, nodes to depth -8% at d13/15: queued stc --front. Dropped: `speed-eval2` (one-pass 2-ply accumulator update, -2.0% x86 / -2.6% arm). Exact SEE early exits impossible (capture ordering uses the exact SEE value). Post-stack profile: fused FT update 21.9%, picker 8.7%, head L1 6.7%, movegen 6.4%: remaining exact headroom is small (memory-bound FT update). Speed agent stopped | queued |
| **25 Nov 02:23: `ext-child-eval-lmr-r2` merged (nonreg +2.19 [-2.33, 6.71]; stc had +2.86 [0.36, 5.37]).** Scheduled LTC check of main 2d13152 vs e8a3495 (the 3613 ladder's main) dispatched; the ladder follows it | merged |
| **Policy move-ordering v0: NO-GO for phase 2 (25 Nov, `policy-oracle`, docs/policy-order-v0.md).** Gate A (upper bound, depth-17 TT oracle, 962 pos d13, noise ~2%): perfect ordering of everything -27.5% nodes; a perfect quiet-move bonus (what a scorer could do) -12.8%, -9.3% at d>=3; break-even precision ~50% on the nodes the heuristics miss (80% precise: -4.1% at d>=3; always wrong: +9.5%); 38-45% of late cutters at cut nodes are killers/counters that cut only after the TT move or a capture failed. Gate B FAILS: the cheapest useful MLP (width 32-64) costs 4.3-5.4% NPS on CI (plumbing alone 3.1%). Net: a perfect scorer ~+5-8 Elo, a realistic one 0-+3. Untested: a scorer for reductions (phase 3) or only where the first move failed | dead |
| **25 Nov 03:01: LTC check of main 2d13152 vs e8a3495 (the 3613 ladder's main): +31.18 [18.26, 44.19]** at 60+0.6, 648 games (early stop: optimistic). Ladder q-ladder-1125-0301 on eb3b4a4: **3617.5 [3601.8, 3633.2]** (+4 vs 3613; ladder noise +-15, the LTC check's +31 was an early stop). `speed-tt-first` dispatched | done |
| **25 Nov 06:05:** `speed-tt-first` **H0 +0.56 [-2.30, 3.42]** (15552 games): time-to-depth -6.4% did not turn into Elo; dropped. Queue drained -> **long SPSA restarted on current main f6beb3e** (run 45558507472; 19 params, values = main's defaults; 6 shards x 40 it x 40 rounds = 9600 iterations, sum combine, 8+0.08; last time round-14 theta merged +21.7). Queue on 12 test shards (lanes 7/7) meanwhile | running |
| **Net recipe ablation program (owner, 25 Nov: "we need to experiment lots here"; GPU work after gen 12):** the recipe (plain 10 -> m1 1 -> kb8m factorise 4, lr 1e-3->1e-5 / 2e-4->1e-6, lambda 0.5, --half, wd 0, stride 4, unweighted) grew by accretion; most parts were chosen on validation loss, which gen 11s showed can mislead (loss parity, -6 Elo). **Method:** one variable per arm at reduced scale (fixed ~1.5B-position file subset of pool12, same val sets, same seed), ~2-3 h GPU per arm at 150 W; every arm embedded on a branch (gen12-embed.sh pattern) and judged by **fixed-node games between arm nets** (`ci.py sprt <arm> --nodes 200000` with the control arm as base: eval quality without speed), plus losses recorded; **two control seeds first** to measure seed noise in games. Arms, in order of value: (0) **single stage: kb8m --factorise from scratch, same total epochs, vs the chain** (owner asked why not one stage; 13 Nov kb8m from scratch on pool8 3.2B lost to the chain on loss, 0.025897 vs 0.025574, data-limited; pool12 is 2.4x larger and factorise may not have been on); (1) plain epochs 10 vs 4, no sparsity (settles the gen 11s confound; 4 = ~40% faster generations); (2) lambda 0.5 vs 0.7 (better loss on 13 Nov, never game-tested); (3) m1 stage kept vs dropped; (4) peak lr x0.5 / x2; then (5) final-stage epochs, (6) shuffle stride/buffer, (7) newest-data weighting, (8) pool8 (old gen 7-8 data) in vs out. **Data arms (owner, 25 Nov):** (D1) opening variety: datagen starts are startpos + 8-9 uniformly random plies (rejected if a 1000-node probe scores beyond +-400 cp), which gives variety but many unnatural early positions; generate ~0.5-1B positions each with 4-6 random plies vs the current 8-9 (same engine, nodes, seeds disjoint), train identical reduced-scale nets, compare in fixed-node games; needs a datagen option `randplies=N` (today a const, RANDOM_PLIES_MIN = 8; options are parsed as key=value like `softnodes=`). (D2, later) nodes per move 5000 vs more (label quality vs quantity). A curated opening set would be external input: owner's call under the from-scratch rule. Desktop at 8 threads makes ~0.34B/day, so D1 costs ~1.5-3 days per arm there; a rented hourly box (e.g. 128-core EPYC 9754 at ~$2.20/h, ~30-40k pos/s est.) does it in hours (~$35): decide after the gen 12 gate shows whether data scaling still pays. Winners combine into the gen 13 recipe | planned |
| **Policy, 25 Nov:** an external agent queued its handover branch `feat-policy-order-v0` as a `speedup`; the queue merged it (bench-identical docs/tools) at 06:32. It added non-stdlib Python (numpy, python-chess) to tools/policy: **reverted on main (owner)**, branch kept; docs/policy-order-v0.md on main is now the measured gate A/B doc from `policy-oracle` plus a short note on the prototype (top-1 5.5-10.9% offline). Rule added to docs/standing-rules.md: only the coordinating session or the owner queues. **Upper-bound agent started** for the remaining shapes (perfect LMR policy, ordering only at depth >= 6/8 with an off-hot-path hook, root-only ordering) | running |
| **25 Nov ~07:30: `small-fixes-stack` 1aae1fb queued stc --front** (bench 187769): (1) check extension no longer counted twice in in-check singular verification (confirmed in code: (depth-1)/2+1 -> (depth-1)/2); (2) a single legal root move is played after a 1-ply search under clock control; (3) qsearch generates quiet queen promotions; (4) rook-takes-minor exempt from the capture SEE prune at depth 1. Singles for bisecting: `fix-singular-check-ext`, `fix-single-legal-move`, `fix-qs-queen-promo`, `fix-exchange-sac-d1`. **Clover at a real TC** (peer-match 45561961963, 10+0.1, 800 games, NPS on one runner) running: decides whether the 1.6x NPS / -178 fixed-node gap is node-counting artifact (CCRL gap is only ~107) | queued |
| **25 Nov 08:06: `small-fixes-stack` MERGED +9.40 [4.29, 14.51]** (4992 games): singular check-extension fix, single legal move, qsearch quiet queen promotions, exchange-sac exemption. Third stack of small items to pass this week (combo-small-positives-r2 +15.1): stacking bug fixes and capped positives is the most reliable source of Elo right now | merged |
| **Speed round 4 started 25 Nov ~09:30 (owner: target the long-search NPS drop):** (1) exact per-king-bucket accumulator cache for refreshes (refreshes 3-3.5% of time, +18% on long searches); (2) SPRT: fewer unneeded NNUE evals (TT-decided stand-pat, in-check nodes); (3) exact lazy legality at pick time (legal movegen 6.5%). Measured at 100 ms / 1 s / 3 s with nps-length | done |
| **Gen 12 trained (25 Nov 13:08):** best-scale vs gen 11: val-g11 0.022256 vs 0.022322, val-g10 0.022343 vs 0.022405, valL1008 0.021813 vs 0.021901 (better by 6-9e-5), but **val-g12 0.018820 vs 0.018727 (worse)**; val-g12 was generated by the gen 11 engine, whose own search labels may favour it, so games decide. Embedded `nnue-g12h1024kb8mpw` 4ebe33f (bench 207737, fixture mean 0.256 cp, nonzero chunks 15.6% after permute), row recorded, **net-gate queued --front**. (The auto-embed's record step failed on a stray % in a format string; done by hand, script fixed.) | queued |
| **Speed round 4 result (25 Nov, `speed-diag` a3be665): nothing recovered, nothing queued.** The per-king-bucket refresh cache already exists on main (ffc7802; a refresh applies ~6 rows); no NNUE eval goes unused (in-check skipped, TT eval used first; the rise 0.435 -> 0.476 evals/node is the LMR child-eval check and the static eval, both reused); fused refresh -0.4..-1.3%, lazy king/castle legality ~0, fewer-evals ~0, stack -0.6..-1.1% (A/A +-0.13%); full lazy legality ceiling ~1%. **The long-search NPS drop is the tree reaching more new positions, not a defect.** Exact speed work is exhausted at this level; remaining levers: move ordering (~18% of time), FT memory traffic, and search changes that skip work (SPRT) | done |
| **Clover 9.1 at 10+0.1 (25 Nov, run 45561961963, 800 games): -166.5 [-178.8, -154.7]** (close to the fixed-node -178; CCRL gap at 120+1 ~107: short TCs stretch gaps). Same CI runner (EPYC 9V74, Hash 16): NPS on depth-12 runs **1.47M vs 1.53M (near parity)**, but on 1-second searches **1.21M (-18%) vs 1.63M**: our NPS degrades with search length (memory: TT, eval cache, history tables?), Clover's does not. **Nodes to depth 12: 4.72M vs 2.25M (2.1x), time 3.21 s vs 1.47 s.** So: (a) a memory-behaviour speed lever the exact-speedup agent never saw (it measured short bench runs); (b) tree size 2.1x = reduction/pruning aggressiveness (ordering is near-optimal per the tree audit) | measured |
| **Speed round 3 started 25 Nov ~08:00 (owner):** NPS vs search length and hash size on CI runners, perf of long vs short searches, then fixes (exact -> speedup; TT/eval-cache layout changes -> SPRT) | running |
| **Speed round 3 result (25 Nov, `speed-diag` 3a0598d, CI tool nps-length.yml):** NPS falls ~9% from 100 ms to 3 s searches (not 18%: the peer job's baseline was 65 ms depth runs), same slope at Hash 16/64/256: **not the TT** (hit rate flat 0.21). Cause: longer searches reach more new positions, eval-cache hit rate 0.43 -> 0.35, NNUE evals per node +17%, king-bucket refreshes +18% (NNUE ~40% of time). Fix: **`speed-ec-packed` 093aa5f** (eval cache 2^20 entries packed into 8 bytes, 8 MB; exact by full-key match; +1.7-1.9% at 1-3 s, -0.5..-1.1% at 100 ms) queued `speedup`. Search-changing ideas for later: skip the stand-pat eval when the TT bound decides; cheaper king-bucket refreshes | queued | `speed-ec-packed` merged (speedup) 07:36.
| **Policy upper bounds (25 Nov, `policy-oracle` fbbb23d, doc section):** ordering only at depth >= 6/8: -5.2%/-2.4% perfect, ~-2% realistic: no-go; root-only ordering -3.6%: no-go. **Reductions: LMR +2 everywhere with the oracle-best move reduced one ply less than main: -30% nodes at depth 13 with root-move quality unchanged (dloss -1.2 +- 1.7 cp; 80% recall -29% / +1.4 cp, 50% recall -31% / +2.8 cp) vs unprotected LMR +2 +5.5 cp; a random 0.5% protected does nothing (+7.9).** The oracle-best move is among reduced moves only ~0.5% of the time: those cases are what make aggressive LMR unsafe. Optimistic (protected move = depth-17 best, the quality measure rewards it), but it matches the 2.1x tree-size gap to Clover. Cost of a scorer called once per node out of line: 1.9% (H32) at depth >= 3, layout cost removed. **Next (running): recall test** (dump nodes where the oracle best is reduced; can existing signals / a linear score / a tiny MLP pick it top-1/2 at 50-80% recall?) | running |
| **Gen 11 done 21 Nov 04:21** (`g11h1024kb8mpw`, pairwise chain on pool11 6.72B): best-scale val-g11 **0.022322** vs g10 0.022818, val-g10 0.022405 vs 0.022948, valL1008 0.021901 vs 0.022408 (-0.0005 on all three, bigger than g9 -> g10). Embedded on main eb1bce0 with the pw-fast inference as branch `nnue-g11h1024kb8mpw` (95697ea, Bench 193878; fixture mean 0.246 cp; head_permute nonzero chunks 50.3 -> 34.5); queued `:net-gate --front` behind the running ladder (fired by the spsa-r14-interim merge, 02:35). If the 5+0.05 gate fails on the NPS cost: run `ci.py sprt nnue-g11h1024kb8mpw --ltc --elo0 0 --elo1 5` before rejecting (pilot: stc -1.7, LTC +3.3). `nnue-pw-speed-r2` lands after as a `speedup` | watcher (ladder + gate) |
| **Qsearch / exchanges (21 Nov, round 8, branch `diag-qsearch`, doc + `qsinfo`/QsDiag tooling):** qsearch is NOT the exchange leak. 2957 positions (2261 right after a capture) vs fresh 4M-node references: static eval is -109 cp after a capture, qsearch brings it to +0.4 cp; error after captures 60 cp vs 62 elsewhere. Stand-pat under threat slightly optimistic (+11..15 mean, +5 median) but no penalty helps (in-search hanging/2: depth-10 error 30.2 -> 34.9). Big misses (>200 cp, 4.2%) shrink only with depth. SEE-losing captures in qsearch: worse (agreement 75.9% vs dummy band 77.4-78.6%). No qsearch/SEE bugs (stand pat in check, TT bounds, ep/promo/x-ray/king SEE checked; SEE ignores pins by design). Reserve: `qs-quiet-checks` (<=16 SEE-safe quiet checks at qsearch ply 1; bench 292044; EPD 11/18 blunders >=2 plies earlier vs dummies +0..6; 200k set inside noise; NPS -8%, nodes -10%; prior ~25%), not queued while the queue is long. Exchange-class blunders -> depth/eval, not qsearch | reserve branch |
| **Diagnostic suite (21 Nov, branch `diag-suite` 3953948, README there):** 662 positions from 3660 Clover/ladder games, labelled by our own 6M-node search; ~3 min/build. **Not usable as a pre-queue screen:** every build that differs from main, dummies included, loses 14-33 solved (self-label bias ~-22, larger than real effects); rank correlation with SPRT Elo +0.20 (solved) / +0.09 (cp loss) over 6 known changes. Kept only to see which positions a change fixes. A fix would need deeper/ensemble labels, a multi-dummy baseline, ~10x positions; not worth it now. Queue throughput has to come from elsewhere (shards after SPSA) | dropped as a screen |
| **Agents 21 Nov 00:55 (3, branches only, nothing queued):** (1) pairwise head inference speed, exact (`nnue-pw-speed-r2` on pw-fast; target before the gen-11 gate ~07:00: head costs -7.9% x86 / -20.6% M3 NPS); (2) qsearch exchange resolution (H1's largest blunder class 5/11; `qs-*` + `diag-qsearch`); (3) offline diagnostic suite of 500-1000 game positions with dummy-control noise floor and SPRT calibration (`diag-suite`), as a pre-queue screen. Next ladder: check the Clover findings (early squeezes, eval optimism) against all 9 anchors' PGNs | agent reports |
| **Agents 20 Nov 19:50 (weekly reset):** engine NPS (resumed after the session limit); time-management analysis of the 120+1 ladder PGNs (TM never examined at the rated TC; branches only); Clover-game loss analysis (where/why we lose in the existing peer-match PGNs; critical-position EPD in scripts/diag/; branch `diag-clover-games`) | agent reports; TM changes need an LTC/60+0.6 test, not stc |
| SPSA interim (`spsa-r14-interim`, round-14 theta on current main, bench 204182, queued `--front` 20 Nov 19:00): pass -> merge, consider restarting the SPSA from it on the new main (605b560 base is stale after the TT fix, killers, pv-no-tt-cut, qsearch-fail-soft); fail -> SPSA continues, re-judge at round 40 | its queue result |
| Long SPSA (run 44631902925, 605b560, 6 shards x 40 rounds, sum combine; ~1.4 h/round under CI contention -> ends ~Sun 22 Nov morning-midday): restore queue to 18 shards and lanes 1/1 (scripts/queue-settings.json), then queue the tuned values `:stc`, then `:ltc-check` | last round's merge job finishes (round 40/40); theta = its `spsa-result` artifact |
| Search diagnosis round 3 (docs/search-diagnosis-2026-11-19.md): `prune-exempt-killers:stc` queued (nodes -9.6%); next X3 = PV-stability measurement | queue result |
