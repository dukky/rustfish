# SEE-scale audit, 2026-11-24: ProbCut's SEE filter and the negamax SEE prunes

Question: do the rules that compare SEE (100/320/330/500/900) with eval
differences assume the wrong scale? We measured them with our own counters
only. No other engine's code, formulas or constants were used.

## Method
- Branch `see-diag` from origin/main 8ec43aa (engine = 836b44c, bench 186105;
  the plain and `knobs` builds reproduce it exactly).
  - `--features stats`: sampled decision logs (env `SEELOG`). ProbCut-filtered
    moves 1/4 (we run the attempt anyway: qsearch, then the reduced search),
    every tried ProbCut move, 1/4 of ProbCut cuts checked by a full-depth
    re-search with ProbCut off at that node, the outcome of every node whose
    ProbCut failed, 1/8 of quiets with SEE < 0 at the quiet-SEE-prune site and
    1/4 of bad captures at the capture-prune site. Pruned and kept moves both
    get the same check, a depth-1 search with a null window on alpha; a move
    counts as an error when it beats alpha. Verification nodes are not
    counted, but they still touch the TT and histories.
  - `pscale` UCI command: the static-eval (and qsearch) drop for the owner
    when one piece is removed.
  - `--features knobs`: PCSH/PCEV (ProbCut filter shift / eval-scale SEE),
    QPM/QPCAP/QPLIN/QPEV (quiet prune), CPM/CPCAP/CPCH/CPEV (capture prune),
    EVP..EVQ (eval-scale table), TNOISE (no-op control).
- Positions: pos962, `go depth 13`, Hash 64, 1 thread, desktop, nice 19.
  Scripts are in scripts/diag/see/.

## Finding 1 (correction): the static eval already uses the piece-table scale
Median static-eval drop when the piece is removed, from 962 positions
(18k pieces):

| piece | all | >= 10 minor/major pieces | 6-9 | <= 5 | table |
|---|---|---|---|---|---|
| P | 98 (IQR 79-120) | 96 | 99 | 107 | 100 |
| N | 342 | 343 | 339 | 323 | 320 |
| B | 379 | 380 | 373 | 359 | 330 |
| R | 511 | 501 | 541 | 563 | 500 |
| Q | 1024 | 1021 | 1030 | 1045 | 900 |

The start-position numbers in the qsearch audit (pawn ~185, knight ~500, rook
~625, queen ~855, at depth 1) did not reproduce. At the start position the
static drops are pawn 50-114, N 311, B 352, R 476 and Q 1003. The net's
scale matches the piece table to within 3-15%: queens and bishops are worth
more, by about 12% and 15%. The qsearch audit's excess over stand pat +
victim reflects positional swings after captures, not a scale mismatch.

**Scale test.** At every site and depth, pruning the lowest X% by eval-scale
SEE (table 100/342/379/512/1024) gives the same error rate as pruning by
piece SEE, to 0.01-0.03 pt. The scale itself does not matter.

## Noise
Verify rates reproduce between seeds 100 and 99 within 0.01-0.1 pt for every
row shown below. Depth-13 node totals of the no-op arms (TNOISE 100/99/101)
are 162.35M / 159.17M / 159.40M, a spread of 2% (sd ~1.1%).

## ProbCut (seed 100; seed 99 matches)
Filtered good-stage moves, bucketed by slack s - (probcut_beta - static eval):

| slack | -300 | -250 | -200 | -150 | -100 | -50 | tried 0..+50 | tried +100 | tried +300 | tried +600+ |
|---|---|---|---|---|---|---|---|---|---|---|
| success if tried | 1.9% | 2.7% | 3.6% | 5.6% | 7.4% | 10.2% (s99 11.1%) | 20.9% | 51% | 74% | 96% |
| attempt cost (nodes) | 2 | 2 | 2 | 3 | 3 | 4 | 5 | 9 | 10 | 5 |

- Cuts checked at full depth fail below beta 0.0-0.4% of the time (d5 ALL
  0.40%, all others <= 0.24%, 0 at d10+). Error does not rise near the
  filter edge (slack 0..100: 0.14-0.22%).
- Nodes whose ProbCut failed still fail high 62-77% of the time. Their
  remaining search costs 60 (d5) to 950 (d12) nodes.
- Replayed direct effect of loosening the filter by 100 cp: 135k extra
  tries, 9-10% succeed, -0.7% of the tree (upper bound). The tree's reaction
  is not modelled.
- Bad-stage captures are skipped even when SEE >= threshold (12k moves,
  11-57% success). That is a small population.
- Verdict: the filter is conservative but cheap. Loosening it gains at most
  ~0.7-1% of nodes. Measured: PCSH=-100 -0.40%, -200 +0.34%, PCEV +0.21%
  (3 seeds each, all inside the no-op spread).

## Quiet SEE prune (threshold -20 d^2)
Error rate of pruned / kept moves, by depth:

| depth | share pruned (SEE<0) | err pruned | err kept | err of SEE <= -400, kept |
|---|---|---|---|---|
| 1 | 98% | 0.56% | 1.32% | - |
| 2 | 98% | 0.37% | 1.07% | - |
| 3 | 84% | 0.13% | 0.86% | - |
| 4 | 62% | 0.07% | 0.42% | 0.15% (bucket -400) |
| 5 | 29% | 0.02% | 0.20% | 0.07-0.12% |
| 6 | 24% | 0.02% | 0.19% | 0.06-0.11% |
| 7 | 0% | - | 0.17% | 0.02-0.10% |
| 8 | 0% | - | 0.16% | 0.00-0.09% |

- At d7-8 the margin (-980 / -1280) prunes nothing. Quiets that hang a minor
  or more are wrong only 0.05-0.1% of the time there.
- At d1-2 hanging-pawn quiets (SEE -100) are wrong 2.3-3.1% of the time, and
  checks are wrong more often than non-checks.
- Replay: capping the margin at 400 prunes 0.9M more moves at 0.05% error,
  3.3% of the tree measured as unreduced verify cost; a cap at 300 prunes
  2.2M more at 0.10% (8.7%). In the real search those quiets are mostly
  LMR-reduced, so the measured node effect is much smaller: QPCAP=400 -0.30%,
  QPCAP=300 -1.30% (seed deltas -1.7/-1.7/-0.5%), borderline against the
  no-op spread.

## Capture SEE prune (threshold -100 d - caphist/64)

| depth | pruned | err pruned | err kept |
|---|---|---|---|
| 1 | 80% | 0.97% | 8.1% |
| 2 | 63% | 0.49% | 3.7% |
| 3 | 43% | 0.27% | 1.5% |
| 4-6 | 19-31% | 0.12-0.22% | 0.95-1.24% |

- Capture history is strongly predictive (scripts/diag/see/capch.py). At a
  fixed SEE the error rate rises 2-5x from ch < -4k to ch >= 4k. Example: d1
  SEE -220, error 0.5% / 0.7% / 1.3% / 3.2% across the four bands. The shift
  that was merged on 22 Nov is justified.
- Exchange sacrifices (RxN/RxB, SEE -170/-180) are wrong 3.3-3.7% of the time
  at d1 with ch < 4k, and they are pruned. That is worse than pawn-losing
  captures (-120/-130: 2.4%) and minor-for-pawn captures (-220/-230: 1.1-1.3%).
  The rate is not monotone in SEE, and an eval-scale table cannot fix it
  (R-N = -170 on both scales). Keeping them at d1 would affect 87k moves
  (0.2% of the tree).
- At d3-6 the error is flat at ~0.5% for every SEE from -220 to -580, which
  makes this the least safe prune per move at those depths. Do not tighten
  it there. CPCAP=400 is +0.51% nodes.
- Measured: CPM=80 -0.88% (2 seeds), CPM=125 +1.70% (fewer prunes, in the
  expected direction), CPEV -1.04% (s100 -2.9, others -0.1).

## Verdict
No threshold is miscalibrated by scale. The eval and SEE scales agree, and
the eval-scale SEE separates no better than piece SEE at any site. No bug
was found. The remaining margin questions are depth shape, not scale:
quiet prune at d7-8 (prunes nothing, about 0.05% error available) and
exchange sacrifices at d1 in the capture prune. Their node effects
(<= 1.3%) are inside the depth-13 node noise.

## Oracle regret
The 696-position d14 regret runs (main and no-ops at TNOISE 99/100/101/98/102,
plus all the arms above at seed 100) were queued in
~/agent-runs/qs/see/p3.sh, writing cand-see-*.jsonl in ~/agent-runs/qs/orc.
They were still running at handback. More node seeds (102/103) for qcap300,
qcap400, pc100 and cpev follow in p4.sh.
