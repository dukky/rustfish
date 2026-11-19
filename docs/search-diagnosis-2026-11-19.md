# Search diagnosis, 2026-11-19

Why does rustfish need 3.8x Clover 9.1's nodes to reach depth 12 (8.51M vs
2.25M on the 50 OpenBench FENs, `docs/peer-analysis-2026-11-19.md`)? The
working hypothesis was that move ordering is too weak to support more
aggressive pruning. This note measures our own tree. No other engine's code
was read. Where a "rule of thumb" is quoted, it is general published
knowledge (Chess Programming Wiki and similar), marked **[general]**.

## Method

- Instrumentation is on branch `diag-search-stats` (not for merging):
  `src/stats.rs`, `st!{}` blocks in `src/search.rs`, all behind the cargo
  feature `stats`, so the default build is unchanged (bench 398177 with and
  without the feature). `stats-verify` also re-searches 1/16 of pruned
  nodes/moves to estimate error rates; that perturbs the TT and histories, so
  its own counts are only used for the error-rate table.
- `scripts/search_stats.py` runs bench or a position list at fixed depth (one
  process per slice, `ucinewgame` before each position, Hash 64), merges the
  counters and prints the tables; `compare` mode prints key metrics side by side.
- Positions: 300 middlegames, each a random UHO_Lichess_4852_v1 start plus
  8-30 plies of rustfish `go depth 7` self-play (seed 20261119). Depth 14
  unless stated. Also the 50 OpenBench FENs at depth 12 / Hash 16, which
  reproduce the peer-analysis count exactly (8,509,009).
- Node types: PV (open window), CUT (null window, `cut_node` true), ALL (null
  window, `cut_node` false), as the search itself predicts them. "Depth" is
  remaining depth at the move loop (after check extension and IIR).
- Move classes: TT move, winning capture (SEE > 0), equal capture (SEE = 0),
  killer 1, counter-move, killer 2, quiet with history > 0 (or cont. history
  > 0), plain quiet, bad capture (SEE < 0).

## Headline numbers (300 positions, depth 14)

| metric | value |
|---|---|
| total nodes (negamax + qsearch) | 160.1M (qsearch 52.7%) |
| mean nodes to depth 12 / 14 | 242k / 663k |
| EBF, iterations 10-14 (geo. mean) | 1.73 |
| first-move cutoff rate, all / CUT / depth 1-3 | 83.9% / 86.1% / 83.3% |
| avg moves searched at a fail-high node | 1.38 |
| predicted CUT nodes that fail low | 11.2% (10-17% by depth) |
| LMR: reduced searches that fail high | 0.7% |
| pruning error rates (all rules) | 0.0-1.2% |

**[general]** First-move cutoff rates around 90% or more are usually quoted
for well-ordered engines, and strong engines report EBFs in the
1.5-1.6 range. We are a few points short on the first and well above on the
second; (1.73 / 1.62)^12 is about 2.2x, so EBF alone explains most of the
node gap.

## Where the tree goes

Negamax nodes by remaining depth (65% of negamax nodes are depth 1; 68% of
depth-1 tries end in RFP):

| depth | PV | CUT | ALL | share |
|---|---|---|---|---|
| 1 | 96k | 38.7M | 10.4M | 64.8% |
| 2 | 64k | 9.6M | 4.4M | 18.5% |
| 3 | 43k | 4.0M | 1.8M | 7.8% |
| 4-5 | 55k | 3.3M | 1.4M | 6.2% |
| 6-8 | 38k | 1.2M | 0.47M | 2.3% |
| 9+ | 16k | 0.22M | 0.07M | 0.4% |

Move-loop nodes (CUT and ALL; "searched" = moves actually searched):

| type | depth | fail high | searched @FH | searched @FL | legal | TT move available |
|---|---|---|---|---|---|---|
| CUT | 1 | 90.3% | 1.28 | 6.6 | 35.8 | 16% |
| CUT | 3 | 83.2% | 1.30 | 10.5 | 30.8 | 39% |
| CUT | 4-5 | 87.1% | 1.35 | 25.8 | 32.7 | 82% |
| CUT | 6-8 | 85.0% | 1.34 | 30.5 | 30.6 | 93% |
| ALL | 1 | 58.6% | 1.48 | 6.2 | 33.4 | 10% |
| ALL | 3 | 39.3% | 1.67 | 9.6 | 32.6 | 30% |
| ALL | 4-5 | 41.5% | 1.91 | 22.2 | 33.5 | 79% |
| ALL | 6-8 | 34.0% | 2.05 | 26.8 | 32.7 | 92% |

- From depth 4 up, a node that fails low searches 70-100% of its legal moves:
  LMP stops at depth 3, so only futility (when eval is far below alpha) and
  SEE prune there. Fail-low nodes at depth 4-8 produce about 3x as many child
  searches as fail-high nodes. This is the main bloat in absolute terms.
- Predicted ALL nodes that reach the move loop fail high 34-59% of the time:
  these are the children of mispredicted CUT parents (a CUT node that fails
  low makes every child fail high).
- Horizon: 61% of the negamax -> qsearch transitions are above the
  iteration depth (reductions), 13.5% exactly at it, 25% beyond it (+1: 8.2%,
  +2: 5.4%, +3 or more: 11.5%). 4.0% of negamax nodes are check-extended;
  singular extension fires on 33% of 141k tries.

## Ordering

Index of the cutoff move:

| depth | idx 0 | 1 | 2 | 3 | 4-7 | 8+ |
|---|---|---|---|---|---|---|
| 1 | 82.3% | 10.7% | 3.4% | 1.5% | 1.9% | 0.3% |
| 2 | 84.3% | 8.2% | 3.0% | 1.4% | 2.0% | 1.0% |
| 3 | 85.6% | 7.3% | 2.9% | 1.2% | 1.6% | 1.4% |
| 4-5 | 88.7% | 5.2% | 2.5% | 0.9% | 1.2% | 1.5% |
| 6-8 | 88.4% | 5.0% | 2.6% | 1.1% | 1.4% | 1.4% |
| 9+ | 91.9% | 3.3% | 1.9% | 1.0% | 1.1% | 0.9% |

By node type: PV 65.7%, CUT 86.1%, ALL 77.4%. The shortfall against
~90% is at depth 1-3, where 82% of all fail highs happen and where the TT
move is usually missing.

Which class cuts, and how often a searched move of that class cuts:

| class | share of cutoffs | cut rate when searched |
|---|---|---|
| TT move | 26.5% | 69.1% |
| winning capture | 38.0% | 65.4% |
| equal capture | 6.9% | 38.0% |
| killer 1 | 16.5% | 42.8% |
| counter-move | 3.6% | 23.2% |
| killer 2 | 1.6% | 9.7% |
| quiet, history > 0 | 5.4% | 4.1% |
| plain quiet | 1.0% | 0.6% |
| bad capture | 0.5% | 2.8% |

At CUT nodes, the class of the first move and its cut rate: TT 84.4%, winning
capture 83.5%, killer 1 69.5%, equal capture 55.9%, counter 50.2%. Of the
late cutoffs at CUT nodes (idx >= 1, 2.26M), the failed first move was a
winning capture in 31.5%, killer 1 in 24.7%, an equal capture in 23.4%, the
TT move in 13.6%.

**The clearest ordering defect: TT moves from fail-low entries.** TT move
searched first, cut rate by the entry's bound and depth deficit
(node depth - entry depth):

| node | bound | deficit <= 0 | 1 | 2-3 | >= 4 |
|---|---|---|---|---|---|
| CUT | lower | 89.0% | 93.1% | 90.0% | 82.8% |
| CUT | upper | 24.6% | 11.4% | 17.6% | 26.4% |
| ALL | lower | 75.9% | 90.4% | 85.1% | 71.4% |
| ALL | upper | 13.2% | 5.5% | 7.8% | 10.6% |

A fail-low node stores the move with the highest of its fail-low scores. As a
TT move it cuts 5-26% of the time, yet it is searched before winning captures
and killers (which cut 70-84% when first). It was first in 434k CUT and 1.44M
ALL move loops.

Quiet history is a weak signal: only 20% of visited quiets have positive
history (those are exempt from LMP, futility and SEE pruning), and quiets
outside killers/counter cut 0.6-4.1% of the time. The history bonus is
16 * d^2: 16 at depth 1 and 144 at depth 3 (0.1-0.9% of HISTORY_MAX), so the
depths where most cutoffs happen barely train the table.

## Reductions

| depth | LMR applied (of null-window quiets) | avg R | reduced fails high | re-search confirms |
|---|---|---|---|---|
| 3 | 71.9% | 1.00 | 1.3% | 60.6% |
| 4-5 | 88.9% | 2.23 | 0.6% | 50.4% |
| 6-8 | 91.4% | 3.90 | 0.5% | 48.9% |
| 9+ | 93.2% | 4.83 | 0.3% | 57.6% |

By node type the reduced-fails-high rate is PV 1.1%, CUT 1.5%, ALL 0.4%.
Captures are never reduced (5-13% of null-window moves). Re-searches are rare
and about half are confirmed.

## Pruning

Fires (and rate: of attempts for node rules, of visited plain quiets for LMP
/ futility / SEE quiet, of visited bad captures for SEE capture):

| depth | fut -> qsearch | RFP | NMP | ProbCut | LMP | move futility | SEE quiet | SEE capture |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.7M (8.3%) | 28.1M (67.9%) | - | - | 56.3M (79%) | 9.8M (14%) | 1.3M (2%) | 4.4M (88%) |
| 2 | 1.6M (12.9%) | 4.9M (44.4%) | - | - | 22.3M (59%) | 7.4M (19%) | 2.9M (8%) | 1.9M (76%) |
| 3 | - | 1.1M (22.9%) | 1.4M (48.3%) | - | 11.2M (29%) | 18.3M (48%) | 2.8M (7%) | 0.9M (39%) |
| 4-5 | - | 0.71M (17.1%) | 0.68M (39.9%) | 94k (35%) | - | 4.4M (24%) | 1.8M (10%) | 0.31M (29%) |
| 6-8 | - | 0.15M (12.1%) | 0.29M (38.4%) | 169k (41%) | - | 1.1M (15%) | 0.22M (3%) | 54k (11%) |
| 9+ | - | - | 55k (33.4%) | 44k (57%) | - | - | - | - |

NMP has no verification search. There is no razoring beyond the depth 1-2
futility drop to qsearch.

Error rates (`stats-verify`, depth 13, 1/16 sample; a pruned node is
re-searched at full depth with its node-level prunes off, a pruned move is
searched at depth - 1 on a null window at alpha; error = the full search
disagrees):

| rule | d1 | d2 | d3 | d4-5 | d6-8 | all |
|---|---|---|---|---|---|---|
| fut -> qsearch | 0.8% | 1.2% | - | - | - | 0.9% (176k) |
| RFP | 0.3% | 0.1% | 0.1% | 0.0% | 0.0% | 0.2% (1.31M) |
| NMP | - | - | 0.4% | 0.3% | 0.1% | 0.3% (89k) |
| ProbCut | - | - | - | 0.1% | 0.0% | 0.0% (11k) |
| LMP | 0.7% | 0.4% | 0.4% | - | - | 0.6% (3.54M) |
| move futility | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% (1.58M) |
| SEE quiet | 0.2% | 0.1% | 0.1% | 0.0% | 0.0% | 0.1% (356k) |
| SEE capture | 1.0% | 0.6% | 0.1% | 0.1% | 0.0% | 0.7% (302k) |

Every rule is below ~1%. Pruning is not making many local mistakes. The
recent losses of more selective patches (`feat-lmr-deeper`,
`feat-lmp-formula`, `retest-razoring`) are therefore not explained by high
local error rates. They more likely come from rare but decisive misses, or
from extra pruning removing the shallow searches that train the TT and
histories. This metric cannot tell which.

## TT

| node | hit | hit with move | hit with depth >= d | TT cutoff |
|---|---|---|---|---|
| PV | 95.2% | 88.8% | 68.7% | 0.5% |
| CUT | 18.3% | 17.2% | 9.0% | 5.7% |
| ALL | 32.6% | 19.0% | 6.8% | 3.9% |

qsearch TT hit rate 10.3%. With 64 MB and ~0.5M nodes per position this
is not a capacity problem: most CUT nodes are new positions.

## Where the tree is bloated, plainly

1. **Bad first moves from fail-low TT entries.** They hurt ordering in 2.7% of
   CUT and ~8% of ALL move loops, and they also skip IIR (the node "has" a TT
   move). Fixing this alone cuts the tree by a third (below).
2. **Fail-low nodes at depth 4-8 search almost every move.** There is no
   move-count pruning above depth 3, and CUT mispredictions (11%) add more
   such nodes.
3. **Shallow ordering without a TT move** (depth 1-3, TT move present in
   only 10-39% of nodes): idx-0 cut rate 82-86%. Equal captures go first too
   often (56% cut rate), and history ordering is weak.
4. qsearch is 53% of nodes. Its TT hit rate is 10%. (Delta pruning was
   removed with a gain on 2026-11-08, so this is not an obvious lever.)

Not bloated: LMR re-searches (rare), the pruning error rates, TT capacity.

## Experiments (our own designs)

Each was measured with the stats build before any game test. "Nodes" = 300
positions at depth 14.

| # | experiment | measured effect | status |
|---|---|---|---|
| E3a | **Fail-low nodes store no TT move** (`tt.store` keeps an older move for the same key; otherwise the next visit orders afresh and IIR applies) | nodes -33%, depth-14 nodes 663k -> 450k, EBF 1.729 -> 1.662; upper-bound TT-move-first loops 1.87M -> 185k, and those left (older proven moves) cut 20-57% instead of 5-26%. OB50 depth 12: 8.51M -> 6.40M (Clover 2.25M; the ratio goes from 3.8x to 2.8x). Bench 398177 -> 316654 | **SPRT running**: branch `tt-failow-nomove`, `tt-failow-nomove-stc-1119-1752` |
| E2 | **Linear history bonus** 160*d (cap 1536 unchanged) | first-move cut 83.9 -> 84.0%, nodes +0.4%, EBF 1.729 -> 1.717 (96*d: similar) | **SPRT running** as a low-prior test: `history-bonus-linear`, `history-bonus-linear-stc-1119-1753` |
| E3b | Order fail-low TT moves like normal moves, keep them in the TT | nodes +14%, first-move cut -0.5 pt | dropped before CI: the opposite of E3a, so skipping IIR is not what costs; keeping the move in the TT does |
| E1 | Equal captures (SEE = 0) after killer 1, before the counter-move | nodes +3%, first-move cut -0.4 pt (killer 1 promoted into those nodes cuts 64.6%, not 69.5%) | dropped before CI: prediction failed |

Proposed next steps, in order. Each depends on E3a, so measure each on top of
it.

1. **E4: fail-low TT move as a hint, not the first move.** After E3a, check
   whether a separate "last best of a fail-low" field helps at all. Order
   such a move just after killer 1, and only when the entry's depth deficit
   is <= 1. Prediction: small. Do it only if E3a passes and stats show
   upper-bound loops still matter.
2. **E5: move-count pruning at depth 4-6 for plain quiets, ALL nodes only**
   (predicted ALL, not improving). Use a conservative count of about
   2x today's depth-3 limit, growing with depth. The data say these nodes
   search 70-100% of moves while plain quiets cut 0.1% there. Prediction:
   nodes -15-25% and LMP error rate at depth 4-6 at most 0.5% (measure with
   `stats-verify` first). The earlier `feat-lmp-formula` changed every node
   type and the low depths too; this version leaves CUT/PV and depth 1-3 alone.
3. **E6: shallow-node ordering without a TT move.** Before the move loop at
   depth 1-3 CUT nodes with no TT move, try the "refutation of the parent's
   move" first: the counter-move when its history is positive, ahead of
   equal captures. Prediction: CUT depth 1-3 first-move cut 83 -> 85%. Measure
   first (E1 shows these per-class cut rates do not transfer between nodes).
4. **E7: CUT-prediction repair.** 11% of predicted CUT nodes fail low and
   then search ~all moves with the +2 cut-node LMR bonus applied to their
   children's labels. Once a CUT node has searched N (e.g. 4) moves without
   a cutoff, relabel it ALL: drop the `cut_node` LMR bonus for its remaining
   moves and pass the ALL-node child flag. Prediction: CUT fail-low cost
   unchanged in nodes, but fewer reduced-move fail highs at those nodes (1.5%
   at CUT now). This is a quality change, not a node saving.

Re-run `scripts/search_stats.py` (and `compare`) on each candidate: the
metric to watch is depth-14 nodes and EBF at an equal or better first-move
cutoff rate, plus the `stats-verify` error table for any new pruning.

## Reproduce

```bash
git switch diag-search-stats
cargo build --release --features stats --target-dir target-st
python3 scripts/search_stats.py positions --engine target-st/release/rustfish \
    --positions scripts/diag-mid300.txt --depth 14 --parallel 4 --out tmp/diag/base14.json
python3 scripts/search_stats.py compare tmp/diag/base14.json tmp/diag/<candidate>.json
cargo build --release --features stats-verify --target-dir target-sv   # error rates
```
Positions: `scripts/diag-mid300.txt` (this branch); OB50 FENs are in
`scripts/peer_nps.py`.
