# Tree audit, 2026-11-22: where the tree is wasted, and what to change

Goal: cut rustfish's search tree through move ordering and pruning, judged
only at equal search quality. No other engine's code, formulas or constants
were used; every change below comes from our own counters.

## Method

- **Build:** branch `tree-diag` (diag-r8-depth merged with main 08c62f7). Its
  search is identical to main at fixed depth: bench 193878, and the depth-20
  oracle reproduces regret 5.78 exactly.
  - `--features stats`: tree-shape counters (see below).
  - `--features stats-verify`: re-searches 1/16 of the prunes.
  - `--features knobs`: env variables that switch each variant on, with no
    counters, for fixed-node games.
- **Positions:** the 962-position ensemble (diag-mid300 plus the diag-suite
  FENs), `go depth 13` after `ucinewgame`, Hash 64, 3 processes.
- **No-op control:** every arm runs 3 seeds, TNOISE = 100 / 99 / 101, which
  scales the history bonus by +-1%. The sd across the 3 base seeds is the
  noise floor:

  | metric | noise sd |
  |---|---|
  | total nodes | 1.2% |
  | first-move cut rate | 0.07-0.12 pt |
  | moves searched at a fail high | 0.003 |

- **New counters:**
  - subtree nodes of every searched move, by class and by index;
  - pre-cutter waste: at a fail-high node, the subtree nodes of the moves
    searched before the cutter, by the class of each failed move and of the
    cutter;
  - quiet "kinds" (sign of h, sign of ch, sign of the picker score);
  - rank of the cutting quiet;
  - best-move index at exact nodes;
  - verify rates bucketed by node type, depth and slack past the margin, in
    25 cp steps (or moves past the LMP limit).
- **Quality control:**
  1. Regret against the depth-20 self-oracle: 696 positions, depth 14, chosen
     move scored at depth 20 by the base engine (`scripts/diag/depth/evalcand.py`).
     Base 5.78 cp; no-op controls 5.56 and 5.86, so the noise band is about
     +-0.3 cp.
  2. Fixed-node self-play at 30k nodes per move: UHO openings, both colours,
     score adjudication. 1200 games give a CI of about +-12.5 Elo.
- **Scripts:** in the agent scratchpad: `run.sh`, `waste.py`, `vk.py`,
  `fn.py`.

## Headline numbers (main, depth 13, mean of 3 seeds)

| metric | value |
|---|---|
| total nodes | 138.5M (qsearch 44%) |
| EBF, iterations 9-13 | 1.72 |
| first-move cut rate, all / CUT / d1-3 / d4+ | 80.4 / 83.2 / 79.1 / 87.8% |
| moves searched at a fail high | 1.51 |

## Waste map

### Pre-cutter waste at fail-high nodes
Columns: subtree nodes of the moves searched before the cutter, as a share of
the node's move-loop nodes; and the same nodes as a share of the whole tree.
The tree-share column overlaps across levels, because waste at a deep node is
also inside its ancestors.

| type | depth | waste % of FH loop | waste % of tree |
|---|---|---|---|
| CUT | 1 / 2 / 3 | 23.2 / 20.7 / 23.4% | 3.6 / 4.1 / 4.4% |
| CUT | 4-5 / 6-8 / 9+ | 19.3 / 14.8 / 8.9% | 4.6 / 6.0 / 3.6% |
| ALL | 1 / 2 / 3 | 28.0 / 25.2 / 27.2% | 1.9 / 2.6 / 2.9% |
| ALL | 4-5 / 6-8 / 9+ | 25.2 / 21.3 / 18.6% | 3.0 / 3.2 / 2.2% |

### Who wastes and who arrives late (CUT + ALL nodes, share of waste)

| class | waste by failed move, d1-3 | d4+ | waste by late cutter, d1-3 | d4+ |
|---|---|---|---|---|
| TT move | 6.6% | **45.0%** | - | - |
| winning capture | 23.0% | 9.4% | 9.1% | 7.3% |
| killer 1 | 17.8% | 11.3% | 8.9% | 12.5% |
| counter | 6.8% | 4.8% | 7.6% | 8.7% |
| killer 2 | 5.9% | 4.1% | 5.2% | 6.2% |
| quiet, h>0 or ch>0 | 20.5% | 13.8% | **31.5%** | **31.4%** |
| plain quiet | 9.8% | 7.0% | **24.4%** | **22.8%** |
| losing capture | 0.3% | 0.3% | 8.3% | 7.9% |
| equal capture | 9.4% | 4.3% | 5.0% | 3.3% |

### Findings, ranked by tree share

1. **At depth 4 and above, half the waste is under a TT move that failed.**
   A lower-bound TT move whose entry is deep enough but scores below beta cuts
   only 51% as the first move (482k cases). Shallower lower-bound entries cut
   82-88%.
2. **Late cutters are mostly quiets** (54% of waste). The cutting quiet is
   the first non-exempt quiet tried 66% of the time, and among the first
   three 89% of the time.
3. **Continuation history is the better signal.** Cut rate of searched quiets
   at CUT nodes, depth 4+:

   | quiet kind | cut rate |
   |---|---|
   | h>0 and ch>0 | 39.0% |
   | ch>0 only | 15.8% |
   | h>0 only | 6.7% |
   | neither | 0.5-1.1% |

   `hist-cont-order` is already queued on this. In this audit, changing the
   weight of ch did not move fixed-depth nodes (see the ordering
   interventions).
4. **Losing captures cut 2.9% when searched, against 1.1% for plain quiets**,
   although they are searched last, and they supply 8% of the waste as late
   cutters.
5. **Best move at exact (PV) nodes:** index 0 for 55% (d1) to 85% (d9+);
   index 6 or later for 22% (d1) to 4% (d9+).

## Ordering interventions (fixed depth 13, 3 seeds each; deltas against main)

| variant | nodes | first-move cut d1-3 / d4+ | moves searched at FH | regret (cp) | 30k-node games |
|---|---|---|---|---|---|
| equal captures after killer 1 | +3.1% | -0.52 / -0.10 | +0.008 | | |
| equal captures after counter | +2.9% | -0.69 / ... | | | |
| equal captures after killer 2 | +6.0% | -0.97 / -0.33 | +0.026 | | |
| killer-2 bonus 0 | -0.6% | +0.01 / +0.13 | -0.004 | | |
| counter bonus 0 | -0.3% | -0.35 / -0.07 | +0.011 | | |
| ch weight x2 (h + ch) | +0.2% | +0.11 / +0.06 | -0.001 | | |
| ch weight x4 | -0.3% | +0.15 / +0.15 | -0.004 | | |
| plain quiets scored with ch | -0.5% | +0.04 / +0.06 | 0.000 | | |
| losing captures before plain quiets | -4.3% | +0.13 / -0.07 | **-0.037** | **6.43** | -6.9 [-19.4, +5.5] |
| losing captures after killers/counter | -10.4% | +0.06 / +0.26 | -0.034 | **7.02** | -2.9 [-15.5, +9.7] |
| losing captures with SEE > -200 early | -2.6% | | -0.032 | | |
| before plain quiets, left out of the LMP/LMR index (`bad1q`) | -1.5% | +0.13 / +0.07 | **-0.037** | see Branches | |

**Conclusions:**
- Quiet ordering is saturated: no intervention is outside noise.
- The current "equal captures before killers" order is right. The
  first-move cut-rate table had suggested the opposite. Conditional cut rates
  are selection-biased; only interventions are evidence.
- Moving losing captures earlier saves nodes but raises regret in step with
  the saving. The saving comes from the index shift: quiets land at higher
  indices, so LMR (average R 2.54 -> 2.62) and LMP cut more. Excluding the
  losing captures from the index (`bad1q`) keeps the ordering gain (moves
  searched at a fail high -0.037) without the extra pruning.

## Pruning verify rates (stats-verify, 1/16 sample)
Each cell is the share of pruned results that a full search would change.
"Threshold" means the bucket within 25 cp of the margin.

| rule | CUT nodes | ALL nodes | reading |
|---|---|---|---|
| move futility, d1-2 | 0.1-0.4% | 0.1-0.3% | fine |
| move futility, d3-8 | **0.0-0.1%, also at the threshold** | 0.0-0.1% | conservative: tighten |
| RFP, d1 | 2.5% at threshold, 0.7% overall | **7.1% at threshold, 2.9% overall** | ALL nodes error-prone |
| RFP, d2 | 0.9% at threshold, 0.3% overall | 3.2% at threshold, 1.2% overall | |
| RFP, d3-7 | **0.0-0.2%** | 0.2-1.8% | CUT nodes conservative: tighten |
| razoring, d1 | **13.5% at threshold, 5.2% overall** | 6.0% at threshold, 2.3% overall | largest error source |
| razoring, d2 | 8.1% at threshold, 2.7% overall | 4.0% at threshold, 1.4% overall | |
| LMP, d1 / d2 / d3 | 1.3 / 0.9 / 0.6% | 1.0 / 0.5 / 0.3% | |
| SEE quiet prune, d1-6 | 0.7% falling to 0.0% | 0.5% falling to 0.0% | |
| SEE capture prune, d1-6 | 2.3% falling to 0.0% | 1.1% falling to 0.0% | |

### Pruning interventions (knobs)

| variant | nodes | ordering effect | regret (cp) | verdict |
|---|---|---|---|---|
| futility margin x0.75 at d>=3 (`FUTD3=75`) | -3.0% | flat | **5.51** | clean |
| RFP margin x0.8 at CUT nodes, d>=3 (`RFPCUT=80`) | -1.8% | first-move cut d4+ -0.14 (2 sd) | **5.81** | clean |
| RFP margin x1.25 at ALL nodes, d>=3 (`RFPALL=125`) | -0.5% | flat | - | noise, dropped |
| razor margin +100 at CUT nodes (`RAZCUT=100`) | -0.8% | first-move cut +0.20 / +0.17 | **5.89** | borderline, see games |
| razoring keeps searching when qsearch beats alpha (`RAZRET=1`, all depths) | +1.3% | first-move cut d1-3 +0.63, moves searched at FH -0.021 | **6.82** | worse than noise |
| same, depth 2 only (`RAZRET=2`) | +1.0% | +0.16 / +0.08 | - | weaker, dropped |

## Branches (from origin/main 516a2e1, bench 188159; nothing queued)

| branch | sha | bench | change | regret (cp) | expected Elo |
|---|---|---|---|---|---|
| tree-pruning-stack | see git | 184622 | the two below together | not run | +2..+5 |
| tree-futility-d3 | b5e8f95 | 188218 | futility margin x3/4 at d>=3 | 5.51 | +1..+3 |
| tree-rfp-cut-margin | 63383e5 | 183504 | RFP margin x0.8 at cut nodes, d>=3 | 5.81 | +0..+2 |
| tree-razor-cut-margin | see git | 185038 | razor margin +100 at cut nodes | 5.89 (borderline) | 0..+2, games pending |

Not pushed:
- `tree-razor-no-failhigh-return` (regret 6.82).
- `tree-bad-capture-index` (`bad1q`, regret 6.53): moving losing captures
  earlier hurts quality even when they are left out of the LMP/LMR index.
- `tree-bad-before-quiets` (bd80569) is pushed as evidence only. Do not queue.
