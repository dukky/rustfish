# Qsearch audit, 2026-11-22: where qsearch spends nodes, and which captures can go

Goal: make quiescence cheaper without losing tactical accuracy. Every rule
and threshold below comes from our own counters. No other engine's code,
formulas or constants were used.

## Method

- **Build:** branch `qs-diag` (origin/main 44bbe8b + counters + knobs; bench
  193456 in every build, so the default search is unchanged).
  - `--features stats`: qsearch counters (`src/search.rs`, `qst` helpers).
  - `--features knobs`: env knobs for candidate prunes (QW*/QE* per-victim
    deficit thresholds, QDFS fail-soft bound, TNOISE no-op control).
- **Positions:** the 962-position ensemble (`pos962`), `go depth 13` after
  `ucinewgame`, Hash 64, 1 thread (`scripts/diag/qs/qrun.py`).
- **Determinant:** a searched move "determined" the node when it caused the
  cutoff or was the final best move of an exact node. A prune's verify rate
  is the share of pruned moves that were determinant in the unpruned
  search. All moves are searched in the base run, so no re-search sampling
  is needed: these rates are the full population.
- **Noise:** 3 seeds (TNOISE 99/100/101, history bonus scaled +-1%).
  Total nodes 161.6M / 162.5M / 163.1M (sd 0.45%); every table rate below
  moves by 0.05 pt or less between seeds.
- **Quality:** oracle regret (696 positions, depth 14, chosen move scored
  at depth 20 by the oracle engine; `scripts/diag/depth/evalcand.py`) and
  the paired per-position regret difference against main (`pair.py`).

## Profile (main, mean of 3 seeds)

Qsearch is **36.5%** of all nodes on current main (59.4M of 162.5M; the 44%
in the tree audit was measured on older main).

| qsearch ply | share of qsearch nodes | in check |
|---|---|---|
| 0 | 55.1% | 0.1% |
| 1 | 21.1% | 7.5% |
| 2 | 11.3% | 5.6% |
| 3 | 5.9% | 7.1% |
| 4+ | 6.6% | 7-10% |

How qsearch nodes end: stand-pat cutoff 50.2%, fail low 25.4%, cutoff by a
move 13.4%, TT cutoff 6.2%, no captures 4.5%, exact 0.3%.

Searched moves by class (all qsearch plies):

| class | searched | determinant | subtree share of qsearch |
|---|---|---|---|
| winning capture (SEE > 0) | 18.3M | 35.2% | **64.5%** |
| equal capture (SEE = 0) | 4.4M | 11.2% | 18.9% |
| quiet evasion | 2.8M | 12.2% | 7.0% |
| capture evasion | 1.0M | 65.1% | 3.7% |
| TT move | 0.13M | 38.4% | 0.6% |
| queen promotion | 40k | 74.0% | 0.2% |
| under-promotion | 31k | **0.6%** | 0.1% |

The determinant rate does not depend on the qsearch ply (winning captures
33-40%, equal captures 10-13% at every ply), so a ply limit on equal
captures has nothing to cut. A move-count limit does not work either: when
stand pat trails alpha by 500-600 cp, the 3rd-5th captures are still
determinant 2-3% of the time.

### Finding 1: the eval scale is not the piece-value scale
The old delta rule (stand pat + victim + 150 < alpha, victim from the
100/320/330/500/900 table; removed in `simp2-qs-delta`, +9.8) assumed the
net scores material at those values. It does not. Excess of a capture's
result over stand pat + victim: P(excess >= +200) = 10%, >= +300 = 5.7%
(winning captures, qsearch ply 0). A winning capture with stand pat +
victim 300 cp below alpha still reaches alpha 5.8% of the time. Material
deltas of the net at the start position (depth 1): pawn ~185, knight ~500,
rook ~625, queen ~855.
The same mismatch sits in ProbCut, which compares `see_ge(probcut_beta -
static_eval)` (piece-table units) with an eval difference; it is
conservative, not wrong, but worth a measurement of its own.

### Finding 2: the right axis is the deficit per victim, and checks are different
Table: prune "alpha - stand pat >= D", non-checking, non-promoting
captures, all qsearch plies. Cells: share of that class pruned / verify
rate of the pruned set / qsearch subtree nodes removed.

| victim | D=300 | D=400 | D=500 | D=600 | D=800 | D=1000 |
|---|---|---|---|---|---|---|
| P, SEE>0 | 33.8% / 0.66% / 4.5% | **25.2% / 0.24% / 3.3%** | 18.0% / 0.13% / 2.3% | 12.4% / 0.12% / 1.6% | 5.8% / 0.11% / 0.8% | 2.3% / 0.07% / 0.3% |
| N, SEE>0 | 44.5% / 17.7% | 30.9% / 11.7% | 20.2% / 6.0% | 13.0% / 2.2% | **5.4% / 0.25% / 0.24%** | 1.8% / 0.05% |
| B, SEE>0 | 43.1% / 20.6% | 30.7% / 15.6% | 20.7% / 9.9% | 13.5% / 4.6% | 5.4% / 0.53% | 1.8% / 0.11% (D=900: **0.23%**) |
| R, SEE>0 | | | 30.2% / 28.7% | | 9.0% / 15.0% | 2.9% / 5.4% |
| Q, SEE>0 | | | 31.9% / 52.1% | | 12.5% / 41.4% | 5.5% / 34.0% |
| P, SEE=0 | **36.2% / 0.08% / 1.3%** | 27.2% / 0.02% | 19.2% / 0.01% | | | |
| N, SEE=0 | 31.8% / 1.1% | 24.5% / 0.49% | **17.8% / 0.24% / 0.13%** | 12.1% / 0.12% | | |
| B, SEE=0 | 35.5% / 1.3% | 27.2% / 0.73% | 19.4% / 0.32% | **13.5% / 0.10% / 0.11%** | | |
| R, SEE=0 | 38.6% / 3.0% | 30.7% / 2.1% | 23.4% / 1.3% | 16.5% / 0.60% | **7.7% / 0.09% / 0.08%** | |
| Q, SEE=0 | | 23.5% / 5.2% | | 8.6% / 2.9% | 2.1% / 1.7% | 0.4% / 1.1% |

- Rook and queen winning captures can never be pruned by deficit: even a
  1000 cp deficit is recovered 5% (R) and 34% (Q) of the time.
- **Checking captures must be exempt:** at D=400 a checking pawn capture
  is determinant 15% of the time (non-checking: 0.24%).
- Slack measured with SEE instead of the victim mixes the victims and is
  worse at every cut (SEE>0, slack <= -400: 1.6% verify).
- Under-promotions (0.6% determinant, 0.1% of qsearch nodes) are not worth
  a change.

## Interventions (knobs; depth 13 nodes, 1 run each; regret vs depth-20 oracle)

All prunes skip non-promoting, non-checking captures only. "Bound" is what
the node's fail-soft best is raised to for a pruned move: stand pat + D (on)
or nothing (off).

| arm | rule | bound | nodes d13 | regret | paired diff vs main (se) |
|---|---|---|---|---|---|
| main | - | - | 162.45M | 5.79 | - |
| no-op TNOISE=99 | - | - | -0.5% | 5.72 | +0.02 (0.82) |
| no-op TNOISE=101 | - | - | +0.4% (stats run) | 6.59 | +0.80 (0.82) |
| **P** | P: 400 (SEE>0) / 300 (SEE=0) | on | **-2.7%** | 5.50 | -0.29 (0.67) |
| **A** | P 400/300, N 800/500, B 900/600, R -/800 | on | **-4.1%** | 5.92 | +0.13 (0.71) |
| C | P 300/200, N 600/400, B 700/500, R -/600 | on | -2.1% | 5.96 | +0.17 (0.69) |
| Pw | P 400, SEE ignored | on | -1.8% | 6.46 | +0.67 (0.75) |
| Pe | P 300 for SEE=0 only | on | -1.4% | 6.76 | +1.01 (0.62) |
| P0 | as P | off | -0.1% | 5.41 | -0.38 (0.58) |
| A0 | as A | off | -1.6% | 5.98 | +0.19 (0.67) |

Readings:
- **Regret cannot separate these arms.** The paired standard error is
  0.6-0.8 cp, and the no-op seeds span 5.34-6.59 (below). Nothing here is
  outside the no-op spread in either direction.
- The node saving needs the fail-soft bound. Without it (P0, A0) the
  pruned subtrees are gone but the total barely moves: the lower upper
  bounds a pruned node returns make the parents work harder. With it, the
  saving exceeds the pruned subtrees alone (A removes ~5% of qsearch
  subtree nodes directly = ~1.9% of the tree; the total drops 4.1%).
- More aggressive thresholds (C) save fewer nodes than A: the tree reacts
  to the changed values, so node savings are not additive in the pruned
  subtrees.
- Laptop time-to-depth (plain builds, interleaved, 962 positions, 1
  thread, owner using the machine, indicative only): main 58.3 / 60.3 /
  61.6 s, P 55.9 / 57.5 / 57.2 s (-5.3%), A 56.0 / 58.1 s (-5.1%, 2
  rounds). To be re-measured on an idle machine.

## Branches (from origin/main 4c34c91; nothing queued)

| branch | sha | bench | change |
|---|---|---|---|
| `qs-deficit-pawn` | 8dfa140 | 167895 | arm P |
| `qs-deficit-prune` | c88d0da | 167960 | arm A |

## More seeds (desktop, same binaries' source, results machine-independent)

The oracle judge must support `go searchmoves`: main (08c62f7) ignores it
and scores the best move instead. The judge is `tree-diag` ef19135 built
on the desktop; it reproduces the cached depth-20 scores and the oracle's
depth-14 node counts exactly. Depth-13 node counts also match the laptop
exactly (noop99 161,624,897 on both).

Oracle regret, 7 no-op runs (main + TNOISE 97/98/99/101/102/103): 5.34,
5.45, 5.72, 5.79, 5.97, 6.46, 6.59; mean 5.90, sd 0.48.

| arm | regret by seed (100 / 99 / 101) | mean | paired diff vs main by seed |
|---|---|---|---|
| A | 5.92 / 5.48 / 6.10 | 5.83 | +0.13 / -0.22 / +0.31 |
| P | 5.50 / 6.40 / 6.01 | 5.97 | -0.29 / +0.61 / +0.22 |

Both are inside the no-op spread (the se of a 3-seed mean is ~0.28 cp).
This is weak evidence of no harm, not evidence of a gain.

Depth-13 nodes against the no-op of the same seed:

| arm | s100 | s99 | s101 | s102 | mean |
|---|---|---|---|---|---|
| A | -4.1% | -2.1% | -2.1% | -0.8% | **-2.3%** |
| P | -2.7% | -0.6% | -0.65% | -1.3% | **-1.3%** |

The single-seed figures in the interventions table overstate the saving:
a changed search varies more between seeds than the no-op controls do with
each other (0.45%). At the 696-position depth-14 oracle the saving is
smaller still (A x0.926-0.945, P x0.923-0.935 against no-ops x0.926-0.974).

Fixed-node games were started and stopped (about 3 h per 600 games at 1
thread; +-18 Elo would not resolve a 2% node saving). Tests: `cargo test
--release` on `qs-deficit-prune` passes (61 passed). Timing is left to an
idle machine.

## Reading and next steps

- The rule is calibrated where the removed delta rule was not: per victim,
  on the eval's own scale, checks exempt, verify rate ~0.1-0.25% per pruned
  move. Its saving is small (-1.3% / -2.3% of nodes at depth 13, more
  in time if the laptop's indicative -5% holds), so expect 0..+3 Elo and
  treat it as an SPRT question.
- The rest of qsearch has no cheap cut: equal captures are determinant at
  every qsearch ply, move-count limits fail at 2-3%, rook/queen captures
  are never safe to skip, under-promotions are 0.1% of qsearch nodes.
- The eval-scale mismatch (Finding 1) is the more general lesson: every
  rule that compares SEE or piece-table values with eval differences
  (ProbCut's `see_ge(probcut_beta - static_eval)`, the SEE prune margins
  in negamax) assumes the wrong scale. Measure those the same way.
