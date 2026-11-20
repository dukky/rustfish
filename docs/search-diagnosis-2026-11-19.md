# Search diagnosis, 2026-11-19

Why does rustfish need 3.8x Clover 9.1's nodes to reach depth 12 (8.51M vs
2.25M on the 50 OpenBench FENs, `docs/peer-analysis-2026-11-19.md`)? The
working hypothesis was that move ordering is too weak to support more
aggressive pruning. This note measures our own tree. No other engine's code
was read. Where a "rule of thumb" is quoted, it is general published
knowledge (Chess Programming Wiki and similar), marked **[general]**.

## Method

- Instrumentation is on branch `diag-search-stats` (not for merging; this doc was copied to main for Round 3):
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
| E3a | **Fail-low nodes store no TT move** (`tt.store` keeps an older move for the same key; otherwise the next visit orders afresh and IIR applies) | nodes -33%, depth-14 nodes 663k -> 450k, EBF 1.729 -> 1.662; upper-bound TT-move-first loops 1.87M -> 185k, and those left (older proven moves) cut 20-57% instead of 5-26%. OB50 depth 12: 8.51M -> 6.40M (Clover 2.25M; the ratio goes from 3.8x to 2.8x). Bench 398177 -> 316654 | **PASSED +16.7 [7.9, 25.6]**, 1728 games; merged as f7d7a51 |
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

## Round 2 (on top of tt-failow-nomove)

Baseline "F" = main + tt-failow-nomove (stats build on branch `diag-failow`).
Same 300 positions, depth 14; error rates from `stats-verify` at depth 13.

| metric | F | E5 LMP d4-6 ALL | E7 cut relabel | IIR from d3 | E4 near-miss hint |
|---|---|---|---|---|---|
| total nodes | 107.9M | 108.0M | 105.5M | **96.1M** | 156.8M |
| mean nodes to depth 14 | 450k | 451k | 440k | **401k** | 651k |
| EBF d10-14 | 1.662 | 1.674 | 1.660 | 1.668 | 1.728 |
| first-move cutoff all / CUT / d1-3 | 83.8 / 86.1 / 83.2% | 83.6 / 85.9 / 83.0% | 84.2 / 86.6 / 83.6% | 84.2 / 86.5 / 83.7% | 83.7 / 85.9 / 83.2% |
| CUT-node fail-low rate | 10.9% | 11.0% | 9.6% | 10.2% | 11.1% |

- **E5, move-count pruning at depth 4-6, expected ALL nodes, not improving,
  plain quiets** (limits continue the 6/10/14 progression by 4 per ply:
  18/22/26). It fires on 1.85M moves at depth 4-5 (18% of plain quiets) with
  an error rate of 0.2% (d6-8: 0.1%), so it is locally safe. But total nodes
  are unchanged and EBF is slightly worse. Those moves were already reduced
  by LMR to near-zero cost. The prediction (-15-25% nodes) failed: **dropped**.
- **E7, after 4 searched moves without a cutoff an expected cut node is
  treated as ALL for the rest of its moves** (no +2 cut-node LMR term,
  ALL-node child flags). Nodes -2.2%, first-move cutoff +0.4 pt, CUT
  fail-low rate 10.9 -> 9.6%, CUT reduced fails-high 1.4 -> 1.3%. Moved as
  predicted: **SPRT** `cut-relabel-stc-1119-1825` (vs tt-failow-nomove).
- **E6, counter-move first in shallow no-TT cut nodes**: not built. The
  counter-move as first move cuts 50%, below equal captures (56%), and E1
  showed these per-class rates do not carry over to other nodes.
  The measurement does not support it.
- **E4, fail-low move as a hint** (a cheap variant with no TT format change:
  keep the fail-low move only when best >= alpha - 16): those moves still
  cut only 6-24% as first moves, and nodes rose 45%. **Dropped.** A
  separate hint field would need a TT format change, and this data gives
  no reason to expect it to pay.
- **New: IIR from depth 3** (was 4). After tt-failow more shallow nodes
  arrive without a TT move. Nodes -11%, first-move cutoff +0.4 pt, CUT
  fail-low 10.2%. At depth 3 the reduced node also skips null move. **SPRT**
  `iir-depth3` (vs main = tt-failow).
- **qsearch is not bloated.** Of 58.4M qsearch nodes, 49.6% end in a
  stand-pat cutoff and only 2.3% are in check (evasions). They search 28M
  children (0.43 per node), and the first move makes 87.7% of the cutoffs
  (90.7% in check). The 53% share is just leaf count: half the leaves are
  one-eval stand-pat nodes. No check, evasion or bad-capture excess shows up.

## Round 3: black-box comparison with Clover 9.1 (on main 285288a, after tt-failow)

Clover is used only as a black box (its UCI `info` output and game results).
All runs are `peer-match.yml` on GitHub runners, UHO book, Hash 16, 1 thread.

### Node curves (50 OpenBench FENs, `go depth 14`, `scripts/peer_nps.py`)

Run 44632063690 (`nps` artifact; `peer_nps.py` now stores per-FEN nodes at
every completed depth). Cumulative nodes summed over the 50 FENs:

| depth | rustfish | Clover | ratio | growth rf | growth Clover |
|---|---:|---:|---:|---:|---:|
| 2 | 17.8k | 4.8k | 3.7 | 3.65 | 1.82 |
| 3 | 50.5k | 9.3k | 5.4 | 2.84 | 1.96 |
| 4 | 96k | 19k | 5.1 | 1.91 | 2.03 |
| 6 | 255k | 76k | 3.4 | 1.69 | 2.42 |
| 8 | 734k | 331k | 2.2 | 1.73 | 1.93 |
| 10 | 2.17M | 0.95M | 2.3 | 1.62 | 1.74 |
| 12 | 6.40M | 2.25M | 2.8 | 1.62 | 1.46 |
| 14 | 15.36M | 4.91M | 3.1 | 1.49 | 1.49 |

Time to depth 14: 6.72 s vs 2.48 s. NPS at 1 s/move: 1.83M vs 2.08M.

- Our growth is **above Clover's at depths 2-3** (the ratio peaks at 5x by
  depth 3-4: the shallow iterations are much bushier) and again at **depths
  9 and 11-13** (1.6-1.8 vs 1.45-1.6), where the ratio climbs from 2.2x
  (depth 8) back to 3.1x. Clover grows faster at depths 6-8, so the
  difference is in the deep iterations, not a constant factor.
- Per position at depth 14 the ratio ranges from 0.8x to 10.7x (median
  2.95x). Worst: OB50 #6 (10.7x, closed middlegame), #46 (10.0x, move 2),
  #28, #20, #5, #24 (7.5-8.4x, piece middlegames). Best (1.1x or less):
  pawn and rook endgames (#4, #36) and one quiet middlegame (#39, 0.78x).

**Stats build on the worst 12 vs the best 12 (depth 14, Hash 64):**

| metric | best 12 | worst 12 |
|---|---:|---:|
| EBF d10-14 | 1.53 | 1.77 |
| PV nodes per position | 478 | 888 |
| first-move cutoff all / CUT | 86.6 / 89.0% | 84.5 / 86.9% |
| CUT fail-low rate (all / depth 9+) | 11.3 / 6.6% | 12.8 / 10.3% |
| NMP success, depth 3 / 4-5 | 54 / 47% | 42 / 38% |
| moves searched at a fail-low PV node, depth 9+ | 15 | 24 |
| priority share of visited quiets | 19.2% | 23.9% |

In the bad positions the PV is unstable (twice the PV nodes, deep PV nodes
that fail low search 24 moves), more predicted cut nodes fail low, null move
works less, and a quarter of the visited quiets are "priority" quiets.
Priority quiets (killers, counter-move, **and any quiet with history > 0 or
continuation history > 0**) are exempt from LMP, move futility and quiet
SEE pruning at every depth.

### Depth sweep and progress gauge (main 285288a)

| limit | games | W/D/L | Elo (95% CI) | before tt-failow |
|---|---:|---|---|---|
| depth 5 | 400 | 377/16/7 | **+564** (+501 .. +658) | - |
| depth 8 | 396 | 143/146/107 | **+32** (+8 .. +56) | - |
| depth 11 | - | - | - | +26 |
| depth 14 | 400 | 75/192/133 | **-51** (-69 .. -33) | - |
| 200k nodes | 800 | 37/316/447 | **-197** (-211 .. -183) | -213 |

Runs 44632067435, 44632072019, 44632077379, 44632063690.
At equal depth our lead disappears between depth 8 and 14: Clover's
depth-14 search, with a third of our nodes, beats ours. Its depth means
more as depth grows, so the gap is not only "we waste nodes" but also "our
deep iterations do not buy as much", which matches the deep-growth excess
above. The 200k-node gauge moved +16 since the TT fix (inside the CI of
the old run).

### Loss breakdown (10+0.1, 792 games, run 44543352392)

- **By opening side** (side = Clover's first eval >= +0.5 for the side we
  play): with the worse UHO side we score **3.3%** (0/26/368); with the
  better side **44.8%** (18/318/59). We neither hold the bad side nor
  convert the good one: in the advantaged draws both engines' evals peak
  at about +1.4 for us and then fade.
- **When it is lost** (first point from which Clover's eval stays at
  <= -2.5 for us): moves 16-25: 131, 26-35: 141, 36-50: 98, >50: 46,
  <= 15: 13. At that point 200 of 429 losses still have > 20 non-king pieces
  and 164 have 13-20: lost in the **middlegame**, not the endgame (10).
- **We are not blind:** just before that point our own eval is already
  <= -1.0 in 384 of 429 losses, but it is about a pawn more optimistic than
  Clover's (median gap 102 cp). Only 158 losses have one of our moves with a
  > 1 pawn eval drop, and it falls on the decisive point itself (median lag
  0 moves). These are slow squeezes, not one-move tactical blunders.
- **Time is not it:** 0.25 s/move each over all games, 0.35 vs 0.33 s in
  the losses. Reported depth 16.7 vs 24.5 (not comparable across engines).

### Experiments from this round (our own designs)

| # | experiment | stats (300 positions, depth 14) | status |
|---|---|---|---|
| X1 | **Only killers and the counter-move are exempt from LMP, move futility and quiet SEE pruning** (a positive history still gives the -1 LMR ply) | nodes -9.6%, depth-14 nodes 450k -> 402k, EBF 1.662 -> 1.647, first-move cut 83.8% unchanged, cutoffs at idx >= 4 2.5 -> 2.3% | **queued** `prune-exempt-killers:stc` (Bench 283739) |
| X2 | LMR history term continuous: `r -= (history + cont. history) / 8192`, replacing the +1 at history < -14000 | nodes -0.7%, EBF 1.664, first-move cut +0.2 pt | not queued: a quality change the stats cannot judge, and its node and ordering effects are within noise |
| X3 | (proposed, not built) PV stability: in the worst positions PV nodes double. Measure aspiration re-search counts and root best-move changes per iteration against Clover's `info` lines before designing anything | - | next |

The long SPSA (below) tunes the margins together; X1 is the structural
change it cannot make (which quiets the prunes may touch).

**SPSA** (run 44631902925, started 2026-11-19 22:45 on 605b560): 19
pruning/reduction/extension tunables, 8+0.08, 6 shards x 40 iterations x 8
pairs per round, 40 rounds (9600 iterations, ~154k games, about 2 days),
`combine: sum`. The earlier runs used `mean`, which applies one chain's
worth of steps per round (8 shards x 50 iterations moved theta like 50
iterations), which is why 400 "iterations" moved nothing. Read the
merged theta from the last round's `spsa-result` artifact
(`gh run download <id> -n spsa-result`; the job summary has the table).

## Bug audit (2026-11-19, late)

Read through `search.rs`, `tt.rs`, the picker and the history code looking
for code that does not do what its comments/design say. Proofs used
throwaway counters on `bench 14` (13 positions) and depth-19 searches;
node effects are total nodes to depth 13 over `diag-mid300.txt` vs `main`.

**Proven and fixed** (one branch each, queued `:stc`):

| branch | bug | proof | bench | nodes d13 |
|---|---|---|---|---|
| `rep-null-guard` | `is_repetition` scanned past the last null move on the path, so null / x / null / x-back scored a false draw (`upcoming_repetition` already stopped at `null_idx`) | 61 of 368 repetition draws in bench 14 crossed a null move | unchanged (316654; first hit is deeper than bench depth) | -0.1% |
| `ep-hash-capturable` | the ep square was set (and hashed) after every double push, so one position had several keys (1.d4 Nf6 2.e4 vs 1.e4 Nf6 2.d4) and transpositions/repetitions were missed; FEN parsing likewise | unit test (fails on `main`: two keys); 92.5% of double pushes had no capturer beside them | 304996 | +0.1% |
| `queen-promo-tactical` | non-capture queen promotions were staged as quiets: LMP- and futility-pruned (futility assumes the move cannot gain ~800 cp), LMR-reduced, fed into quiet history | bench 14: 624 of 2843 queen promotions reaching quiet pruning were pruned unsearched, 491 more reduced | 319206 | -0.7% |

Node effects are within noise; these are correctness fixes, so the SPRT is
the judge.

**Checked and clean / harmless as is:** TT bounds and mate-ply adjustment
both ways; lockless key check; replacement; qsearch TT; side/castling
hashing; cut-node labelling; LMR re-search chain; singular exclusion (TT
not stored, null move off); correction-history conditions; history
gravity and maluses; killers quiet-only; counter-move indexing.
Root fail-low also stores its "best" move as the TT move (the same pattern
as the fixed bug), but in 19 fail-low re-searches the stored move was the
previous best every time, so it is harmless today.

**Unverified / design questions (not bugs as shown):**
- ProbCut is not guarded against singular-verification searches (it could
  search the excluded move and store into the node's TT slot), but
  `tt_says_no` blocks it whenever `2 * depth < PROBCUT_MARGIN`; 0 hits at
  depth 19. Worth an explicit `excluded.is_none()` for safety.
- Null move returns unproven mate scores (no clamp to beta / below mate).
- PV nodes take TT cutoffs (truncated PVs; a design choice).
- Root ordering puts every capture, SEE-losing ones included, ahead of
  killers and quiets (the picker puts bad captures last).
- Aspiration widening is linear (+-70 cp per fail), not exponential; fail
  re-searches are rare (8-19 per 10 depth-19 searches).
- The LMR table is built for 1-based move numbers but indexed with the
  0-based searched-move count (an offset SPSA can absorb).
- Non-priority quiets are scored `max(h + ch/2, h)`, so negative
  continuation history is ignored for them.
- Qsearch has no quiet queen promotions; pre-root positions count as
  draws on a 2-fold repetition.
- **NPS vs search length:** one 20 s search: 1.73M nps at depth 12, then flat
  at 1.41-1.45M from depth 15 (0.7 s) to 27 (17 s). The drop happens in the
  first second (tree outgrows the caches); nothing found that grows with
  search length (pools, history, TT age, root counters are all bounded).

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

## Round 4 (2026-11-20, on main 48d4541: tt-failow + prune-exempt-killers)

Stats build: branch `diag-r4` (the `diag-search-stats` counters rebased on
main, plus root counters `it_nodes`, `asp_fl`/`asp_fh`/`asp_multi`/`asp_waste`,
`bm_change`, `bm_idx`, `root_raise`, `score_delta`, `net_ext`,
`pv_net_ext_by_ply`, `nmp_mate`). Bench 283739 with and without the feature.
300 middlegames, depth 14, Hash 64: 401.8k nodes to depth 14, EBF 1.636.

### Gauge

200k nodes per move vs Clover 9.1, 792 games (run 44698430657, main 48d4541):
W/D/L 31/309/452, **-205.8 (-221 .. -191)**. Before the TT fix -213, after
it -197: prune-exempt-killers (+11 at STC) did not move the fixed-node gauge
(it saves nodes, which a node budget does not reward).

### Depth-14 misses against our own depth-20 oracle

696 positions (diag-mid300 plus 396 middlegames grown by depth-7 self-play
from the start FENs of the Clover 10+0.1 match, run 44543352392). Each:
fresh `go depth 14` (move A) and fresh `go depth 20` (move B). Where they
differ, B and A were each scored at depth 20 with `searchmoves` (new on
`diag-r4`), and a miss is "significant" when B beats A by >= 25 cp there.
Classification used `searchmoves` probes at depth 14 and the `DiagOff`
switch bits on `diag-r4` (each turns off one of LMR, LMP, move futility,
SEE prunes, null move, RFP, razoring, ProbCut, singular, IIR, non-root TT
cutoffs, the cut-node LMR term, cut-node labels on root children,
aspiration). Scripts: `misses.py`, `rootpass.py`, `deppass.py`.

- d14 and d20 differ in 180 of 696 positions (26%); 64 are significant
  (median gap 39 cp).
- **Root class, 39 of 64 (61%)**: searched alone at depth 14 (PV window), B
  scores higher than A (median +34 cp), yet the full search keeps A. B's
  null-window root search fails low, in 24 of 39 at exactly alpha. B had
  been the best move at an earlier iteration in 20 of 39.
- **Subtree class, 25 of 64**: B scores lower than A at depth 14 even alone.
- **No mechanism dominates.** The unchanged search reproduces A in 39 of 39,
  but almost any switch makes the full depth-14 search pick B in 14-25 of 39
  root-class misses, including cheap ones (cut-node LMR term off: 18/39 at
  1.08x nodes, cut-node labels off: 14/39 at 1.08x, aspiration off: 18/39 at
  1.32x). LMR off is the highest at 25/39, but costs 5.4x nodes. Plain depth
  does better per node: depth 15 picks B in 11/39 (1.8x nodes), depth 16 in
  24/39 (3.0x), depth 18 in 34/39 (6.3x). In the subtree class, single
  switches flip 7-17 of 25 (only 3 have a single sole cause), and depth 18
  picks B in 11/25.
- **Agreement check:** with the oracle fixed, a perturbed search reshuffles
  rather than improves: cut-node LMR term off agrees 72.1% (fixes 22 of the
  64 misses, breaks 76 agreements), fail-soft qsearch (below) 73.0% at 0.9x
  nodes, base 74.1% (biased upward: the oracle is the base engine).

Reading: the depth-14 misses are knife-edge root decisions that one or two
more plies resolve, plus deeper content that only depth 18+ sees. There is no
single reduction, prune, extension or TT rule hiding the better move. So the
fixed-depth deficit against Clover is not a broken mechanism we can switch
off; it is "how much each node is worth", and the levers stay the known ones
(eval, node efficiency at the shallow end, SPSA of the reduction terms).

**One real inconsistency found on the way:** qsearch was fail-hard on a fail
low (it returned and stored `alpha`) inside a fail-soft negamax, which is
why so many root fails land exactly on alpha. Fail-soft qsearch (returns and
stores the best score, stand pat included; floor 0 when the repetition check
raised alpha): nodes -7.9%, EBF 1.636 -> 1.621, aspiration fails 457 -> 553,
bench 275954. Queued as `qsearch-fail-soft:stc`.

### Quality at depth: what our own tree shows

Per iteration (300 searches each):

| depth | asp. fails (low/high) | 2nd+ fails | nodes in failed searches | best move changes | best move was root move #2-4 / #5+ |
|---:|---|---:|---:|---:|---|
| 6 | 19 / 20 | 3 | 7.9% | 21.0% | 20 / 36 |
| 8 | 20 / 23 | 2 | 7.9% | 15.7% | 13 / 28 |
| 10 | 17 / 32 | 3 | 10.4% | 9.7% | 9 / 16 |
| 12 | 18 / 31 | 2 | 12.7% | 11.3% | 9 / 18 |
| 14 | 12 / 19 | 1 | 8.5% | 11.3% | 11 / 20 |

- **Aspiration:** 10-17% of iterations fail (fail high more often than fail
  low), and the failed searches cost 8-13% of the iteration's nodes. Only 33
  of 457 fails were a second fail, so the linear widening (+-70 per fail) is
  not what costs; the single re-search is. **Suspect ruled out.**
- **PV stability:** the best move changes in 10-21% of iterations, more at
  even depths 4-9 than at odd ones. The root re-sorts moves 2..N statically
  every iteration (TT move, then all captures, then killers and history), and
  the iteration's best move sat at root index 5 or later in 5-15% of
  iterations.
- **Extensions do not explode.** Check extensions are 3.7% of negamax nodes
  and fall with depth. Singular tries run at 22-35% of depth 8-12 nodes and
  extend 28-35% of them (PV: 45-52%); multi-cut 12%, negative 20%. The net
  extension at PV nodes (depth + ply - iteration depth) averages +0.4 at
  ply 4, +1.0 at ply 10 and +1.6 at ply 12.
- **Nominal depth:** at qsearch entry 65% of horizon leaves are shallower
  than the iteration depth, 13% exactly at it and 21% beyond it. The net depth
  at CUT/ALL nodes spreads over -8..0: most of the tree is reduced.
- **Reductions, PV vs non-PV** (avg R of reduced quiets, PV / CUT / ALL):
  depth 6: 3.2 / 3.9 / 3.6; depth 9: 3.7 / 6.2 / 4.7; depth 12: 4.2 / 6.9 /
  5.0. PV late quiets lose 4 plies at depth 9+ (`lmr_pv_node` is only 1 ply
  less than ALL), and CUT ones end near the `depth - 2` clamp. Reduced
  searches that fail high: PV 0.4-1.2%, CUT 0.3-1.3%, ALL 0.1-0.3%. Both node
  terms are tunables in the running SPSA, so no hand change was queued.

Reading: no single broken mechanism shows up. The deep tree is heavily
reduced (PV nodes included) and still 3x Clover's nodes, which points at
where the nodes go (shallow bushiness, Round 1) rather than at extensions or
aspiration. Fixed-depth Elo against another engine mostly compares reduction
schemes, so on its own it is a weak per-depth diagnostic.

### Unverified suspects from the bug audit

| suspect | verdict | evidence | action |
|---|---|---|---|
| Null move returns unproven mate scores | **real, rare** | 4 of 1.45M null-move fail highs returned a mate score; clamping to beta leaves bench and the 300-position tree identical | branch `nmp-mate-clamp` ready (bench unchanged, `:nonreg`); taken off the queue for `qsearch-fail-soft` |
| PV nodes take TT cutoffs | **real (design)** | 1.9% of PV nodes cut on the TT. Without it: nodes +7%, PV first-move cut 60.3 -> 61.9%, bench 285432 | `pv-no-tt-cut:stc` |
| Root orders losing captures before killers | **real** | root scoring gives every capture 1,000,000 + 12*SEE, so a -800 SEE capture still beats killers (900,000). Bad captures last: nodes +1.8%, EBF 1.636 -> 1.621, bench 284951 | branch `root-bad-captures-last` pushed, not queued (queue cap) |
| Aspiration widens linearly | ruled out | 7% of fails are repeats (above) | none |
| LMR table index off by one | **real** | the table is built for move numbers m >= 1 (column 0 = 0) and indexed with the 0-based searched-move count, so searched move k gets the ln(k-1) entry. With idx + 1: nodes -2.0%, EBF 1.634, bench 293434. The table shape is not an SPSA tunable | branch `lmr-index-1based` pushed, not queued (queue cap) |
| Quiet scoring ignores negative cont. history | **proven** | plain quiets have h <= 0 and ch <= 0, so `max(h + ch/2, h)` is always h: continuation history never ordered them. Fixed: nodes -0.6%, EBF 1.636 -> 1.626, bench 307623 | `quiet-cont-neg:stc` |

Candidate numbers: 300 positions, depth 14, stats build on `diag-r4` with
each patch applied. Every queued branch passed cargo test --release,
perfttest, selftest 100 and a UCI smoke.

## Round 5 (2026-11-20, on main c8e3ff7, before the queued round-4 fixes)

Stats build: branch `diag-r5` (`diag-r4` rebased on c8e3ff7). Same 300
middlegames, depth 14, Hash 64. Baseline: 95.0M nodes, EBF(10-14) 1.639,
first-move cutoff 83.6% (PV 60.1 / CUT 86.0 / ALL 76.8).

**Noise first.** A one-unit change to the history bonus (`16d^2 + 1`, no
real effect) moves total nodes **+5.0%** and EBF 1.639 -> 1.647, while the
first-move cutoff rates move <= 0.2 pt. So on this set, node totals within
+-5% are noise (several earlier "-1..-2% nodes" readings were too); cutoff
rates, per-class cut rates, TT hit rates and root counters are the stable
signals. Hash 48 vs 64 changes nodes by only 0.1%: the noise is the
chaos of the tree under any ordering change, not TT capacity.

| candidate | ordering / TT / root signal | nodes | status |
|---|---|---|---|
| capture history (`[side][mover][to][victim]`, ordering only, negamax; /8) | first-move cut 83.6 -> 84.5% (CUT 86.8, ALL 77.8), idx-1 cuts 9.4 -> 8.7%: the biggest ordering gain measured so far | +3.5% | **not pushed**: the same idea was H0 twice in the queue (+0.2 over 6.7k, -2.4 over 9.1k games). Better ordering by this metric did not buy Elo |
| continuation history keyed by piece-to on both sides | +0.1 pt | +1.8% | dropped (matches the rejected `feat-conthist-rekey`) |
| 2-ply follow-up history in quiet ordering | +0.1 pt | +6.8% | dropped (matches the rejected cont2-ply tests) |
| **killers per parent** (clear `killers[ply+1]` at node entry) | killer 1 cut rate when searched 42.2 -> 64.7% (5.61M -> 3.11M searched), counter 22.9 -> 27.0%, first-move cut +0.3 pt (ALL +0.5) | +4.5% | pushed `killer-reset-per-parent`, Bench 293830 |
| **null-move cutoffs stored in the TT** (lower bound, no move) | CUT TT hit 23.2 -> 24.5%, hit with depth >= d 12.4 -> 13.4%, TT cutoff 8.1 -> 8.5% | +3.4% | pushed `nmp-tt-store`, Bench 278392 |
| **root moves 2..N ordered by subtree nodes so far** | best move from root index >= 2 in iterations 6-14: 286/305 -> 182/172 of 2700 (two runs each); aspiration fails 382/403 -> 360/364; best-move changes 353/377 -> 429/407 | +2.8% | pushed `root-order-nodes`, Bench 351039 |

Reading: the three ordering-structure ideas that the queue already rejected
(capture history, piece-to re-key, follow-up history) are also the ones this
round measures as either neutral or "better ordering, no node saving". The
first-move cutoff rate is a necessary signal, not a sufficient one. The three
pushed branches target things the earlier rounds had not touched: stale
killers from unrelated subtrees, a TT that forgets null-move cuts, and a root
order that ignores the previous iteration. None is queued yet (queue full).

## Round 6 (2026-11-20): where the Clover games are lost (game records only)

Source: the PGNs of four existing peer-match runs (no new CI): 60+0.6
(44545275669, 396 games), nodes=200000 (44698430657, 792), 10+0.1
(44575415699, 792), nodes=1000000 (44547040291, 396). fastchess records
`score/depth time` per move (no node counts). Only the games were analysed;
Clover's code was not looked at. Both engines' scores are put into our POV;
Clover's scale is about 1.12x ours (median ratio on same-sign middlegame
pairs), which the gaps below allow for. Phase = non-pawn material
(N/B 1, R 2, Q 4; 24 = full): opening/early mg >= 20, mg 9-19, eg <= 8.
"Clover loss of our move" = Clover's score before our move minus its score
after it (a judge that is about 200 Elo better than us, not an oracle).

**1. The deficit is the UHO squeeze, on both sides.** Side by Clover's
first score: worse side we score 2.3% (60+0.6: 0/9/189), better side 47.7%
(11/167/20). Median scores by moves after the book, better side, Clover |
us: m0 +118|+91, m10 +89|+103, m20 +42|+85, m30 +6|+72. Worse side:
m0 -123|-93, m10 -166|-122, m20 -249|-204, m30 -376|-319. Clover gains 6-8
cp per move when it is better; our edge fades 3-4 cp per move when
we are; we keep believing in our edge (when Clover's score drops to
<= 0.40 for good in our better-side draws, ours is still +0.91 median at
60+0.6, +0.66 at 200k nodes). The same at all four limits.

**2. Losses are decided early, slowly.** Clover's lost point (its score
<= -2.0 for us for the rest of the game) is in the opening/early middlegame
in 58-69% (144/209 at 60+0.6, 264/452 at 200k), middlegame 24-31%,
endgame 7-10%; median 15-16 moves after the book, 25-26 pieces on the
board. 95% of losses (198/209 at 60+0.6, 396/452 at 200k) have **no single
move of ours that Clover rates as costing >= 1.5 pawns**. Of the Clover
loss of our moves before the lost point, 30-34% comes in < 25 cp steps,
30% in 25-50 cp, 21-23% in 50-100 cp, only 13-21% in >= 100 cp steps. We
agree the game is lost a median 2.5-3.5 moves (mean 4.5) after Clover,
and just before Clover's lost point our score is 82-95 cp (median) more
optimistic.

**3. What costs.** Mean Clover loss per move of ours (|score| < 3,
200k nodes): pawn moves 6.1 cp (4.9% >= 50 cp), piece moves 4.7, king 2.9,
captures 2.9, checks 0.4; by phase open 6.7 / mg 5.9 / eg 1.1. Our own
score drops most after **their captures** (8.7 cp mean, 6.5% >= 50 cp)
and pawn moves (6.6), vs 4.1 after their piece moves: we misjudge how
exchanges and pawn breaks resolve. By our reported depth at 200k nodes:
d9-11 13.9 cp, d12-14 6.4, d15-17 2.1 (depth tracks complexity, so this
is partly phase).

**4. Endgames and draw rules are not the leak.** Endgames where both
scores say >= +2 for us: 11 won / 5 drawn / 0 lost (60+0.6), 29/10/1
(200k). Endgames both near 0 that we then lose: 0 in every run. In
better-side draws the last three moves have both scores within 0.5 in
96-98% of games (final material RvR 54, OCB 19, of 167 at 60+0.6).
Draws: threefold 67, 50-move 24, adjudication 75 of 176.

**5. Depth at equal budget** (reported depths, not comparable across
engines): 200k nodes us 13.2 / 14.5 / 20.7 vs Clover 16.9 / 18.6 / 29.2
(open / mg / eg); 60+0.6 us 17.7 / 18.1 / 27.5 vs 24.7 / 26.4 / 43.5 at the
same time per move (1.43 vs 1.36 s).

**6. Critical positions** (`scripts/diag/clover-misses.epd`, 31 positions;
`scripts/diag/epd_depth.py` measures them). Searched fresh (ucinewgame,
1 thread, Hash 64, fcbf8e7, up to 20M nodes; plus the position after the
game move up to 10M nodes to score the game move itself):

- 24 *blunders* (Clover loss >= 1.5 pawns, Clover said ~0 before, we said
  >= -0.5): our own search rates the game move >= 50 cp below its best,
  stably, in 23/24, from a median depth **15** vs median game depth **13**
  (+2 plies; 620k vs 240k nodes at depth 15/13 on these positions, ~2.6x);
  its score of the game move reaches Clover's verdict in 22/24 at median
  depth 15. So the blunders sit just past our horizon. Range -12..+10 plies:
  in 7/23 a fresh search never picks the game move at all, i.e. the move
  came from the in-game search state (TT/histories from earlier moves,
  history of the game for repetitions), not from depth alone.
- 8/24 blunders were played at an **exact 0.00**, and in the deep runs
  the game move often sits at 0 for several iterations before collapsing
  (Kc5: d16 0, d20 -12.97; Rhf8: d12-16 0, d20 -1.88; n1m g3: d20 0,
  d24 -10.23; Nc6: d16 0, d20 -3.96). Over all games we print exact 0.00 on
  5.5-6.4% of our moves with phase >= 4, Clover on 0.1-0.3% (it prints
  +-0.01, so that ratio is partly cosmetic); "our 0.00, then Clover >= +1
  for itself" occurs in 14% of our 200k losses (62/452) and 11% at 60+0.6.
- 30 *squeeze* moves (Clover loss 0.6-1.5 pawns, not captures, not yet
  lost): our deep search rejects the game move in only 19/30, from median
  depth **19** (game 13; ~2.8M nodes, ~11x the budget); 11/30 are still the
  choice at 10-20M nodes. Those are evaluation (or Clover being wrong),
  not search.

**Reading.** Roughly: 5-10% of losses are short-horizon blunders that two
more plies fix; ~90% are a slow positional squeeze in the first 15-25 moves
out of the book, in which half the per-move loss is visible to our own
search 4-6 plies deeper and the rest not at all. The squeeze favours
whoever spends nodes better on the quiet, pawn-structure decisions of an
unbalanced middlegame; our score is a pawn too optimistic through it.

**Hypotheses for our own designs** (each measurable on the EPD at fixed
nodes before any SPRT):

- **H1, threat/exchange-aware reductions.** Their captures and pawn moves
  are what surprise us; blunders are +2 plies away. Reduce less (or
  extend by a fraction) for moves that answer a threat created by the
  opponent's last move (a piece newly attacked by a lower-valued piece, or
  a recapture square), and do not LMR/prune captures at CUT nodes whose
  SEE is only mildly negative when the last move was a capture.
  Metric: blunder subset solved at lower depth/nodes, no node blow-up on
  the 300-position set.
- **H2, repetition draws masking threats.** Score repetition draws in the
  tree without letting them enter the TT as exact (or as depth-0 bounds),
  and check whether a position repeated only before the root should count
  as a draw at the first repetition. Metric: frequency of exact 0.00 root
  scores on the 200k games' positions, and the 8 zero-score blunders.
- **H3, game-state carry-over.** 7/23 blunders are not chosen by a fresh
  search at any depth. Replay those games (`position startpos moves ...`
  with the real move sequence and node budget) to see whether TT entries
  or histories from previous moves cause them; candidate fixes are
  history decay at each new `go` and TT age preference. The squeeze side
  (point 1, 11/30 unsolved by depth) is an eval/data question for the net
  pipeline, not for search.

Tools (scratch, not committed): python-chess in a scratch venv for PGN
replay; parse/analyse scripts. To rerun the EPD:
`python3 scripts/diag/epd_depth.py --engine ./target/release/rustfish --nodes 20000000`
(baseline fcbf8e7: best move leaves `am` stably at median depth 13.5,
213k nodes; 7 never pick it). The EPD drops the move history and the
halfmove clock, so repetition-dependent cases (H2, H3) need the game replay.

## Round 7 (2026-11-20): exact-0.00 draws (H2) and game-state carry-over (H3)

Method: the nodes=200000 Clover games replayed move by move (`ucinewgame`,
Hash 16, 1 thread, `position fen <book> moves ...` + `go nodes 200000` for
every one of our moves, the opponent's moves taken from the PGN). The run's
build 48d4541 (bench 283739) reproduces the games exactly: 2452/2452 of our
moves in 48 games identical, same score and depth. Variants are compared on
96 games (every 2nd of 0-191, 4863 of our moves with phase >= 4) on main
debf84f. "Deep judge" = where two variants choose different moves, both
moves are scored by a fresh 2M-node search of the position after the move
(real move history kept), clipped to +-4 pawns. Scripts were scratch
(replay/zrate/judge/repset), not committed.

**H2: the 0.00 rate is cosmetic, not a leak.** Clover prints +-0.01 for its
draws: |score| <= 0.01 on 20.0% of its moves (phase >= 4, 200k run) vs our
7.4% (exact 0: Clover 0.26%, us 7.2%). After a near-0 score of ours, Clover
claims >= +1 for itself on its next move 3.7% of the time (2.9% at 60+0.6);
after Clover's near-0, our next score is >= +1 for us 6.1% (16.1%). The
round-6 "8/24 blunders at 0.00" is selection: blunders were chosen where
Clover said ~0 before the move, i.e. drawish positions.

Where our exact 0s come from (main, 473 zeros = 9.7% of replayed moves; a
build that returns a distinct tiny score per draw source):

| source | zeros |
|---|---|
| cuckoo upcoming-repetition raise (alpha -> 0) | 163 (34%) |
| repetition (`is_repetition`) | 146 (31%) |
| insufficient material | 39 |
| 50-move rule | 16 |
| stalemate | 3 |
| eval exactly 0, or the tagged build chose differently | 106 |

| candidate | exact 0 | moves changed | deep judge (candidate - main) |
|---|---|---|---|
| main, fresh state every move (`ucinewgame` before each `go`) | 7.4% | - | see H3 |
| pre-root positions draw only at the 3rd occurrence (+ cuckoo targets in-tree only) | 9.8% (all moves); 42.1% vs 55.1% on the 837 of 39437 moves (2.1%) whose halfmove window holds a repeat | 272/837 | -3.8 cp, 16 vs 31 better by >= 30 (main judge); -2.2 cp, 22 vs 31 (judged by the variant) |
| TT cutoffs ignore 0-score entries from earlier searches (GHI test) | 10.4% | - | not run: zeros went up, so carried-over TT draws are not the source |
| negamax floors its result at 0 after the cuckoo raise | 10.5% | 400 judged | +1.6 cp, 30 vs 31 |

Findings:

- Counting a single pre-root occurrence as a draw is a rule choice, not a
  bug: changing it moves a third of the decisions where it applies and the
  deep judge mildly prefers the current rule. Branch `rep-preroot-3fold`
  (Bench 268722 unchanged: bench positions have no history) is pushed for
  an ordinary stc test if wanted, low priority.
- **Bug (soundness):** after the cuckoo test raises alpha to 0, negamax kept
  fail-soft returns below 0 (TT upper bounds, razoring/qsearch drops, the
  final fail-low `best`), telling the parent its opponent cannot draw here,
  which can cause a false fail-high. Quiescence already floors at 0. Branch
  `negamax-rep-floor` (Bench 266587): the node runs with alpha 0 and clamps
  its result; the TT keeps the unclamped, path-independent bound. Decision
  effect neutral on the replays: **nonreg**.
- **Bug:** a node with halfmove clock >= 100 returned 0 before checking for
  mate, so a mate delivered by the 100th reversible half-move scored as a
  draw (`6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 99 80`: main plays h3 at +16.6,
  the fix plays Ra8#). Branch `fifty-move-mate` (Bench 268722 unchanged,
  unit test added). Rare; **nonreg**.
- Not found: draw scores leaking through the TT from earlier searches, a
  wrong insufficient-material rule, or a contempt problem (there is no
  contempt; draws are exactly 0).

**H3: carry-over helps; the 7 "fresh never picks it" blunders are near-ties.**
Five of the seven are in the 200k run and replay exactly. At the blunder ply,
with the real game state, clear one thing before the `go`:

| blunder | game state | -TT | -history | -correction | -TT-hist-corr | -move history | all |
|---|---|---|---|---|---|---|---|
| 10 Rf2 | Rf2 0 | Rf2 -37 | Rf2 -50 | Rxb7 -48 | Rxb7 -85 | Rf2 0 | Rxb7 -85 |
| 13 Nc5 | Nc5 +107 | Nb6 +68 | Nc5 +99 | Nb6 +83 | Nf8 +74 | Nc5 +107 | Nf8 +74 |
| 14 Ke5 | Ke5 -18 | Kc7 -116 | Ke5 -33 | Ke5 0 | Kc7 -30 | Ke5 -26 | Kc7 -45 |
| 16 Bf1 | Bf1 -15 | Kb2 -32 | Kb2 -44 | Bf1 -18 | Bf1 -17 | Bf1 -15 | Bf1 -17 |
| 18 Nc1 | Nc1 -44 | Nc1 -48 | Nd2 -78 | Nd2 -48 | Nb2 -58 | Nd2 -54 | Nd2 -67 |

The game move reproduces with the game state in 5/5; clearing everything
avoids it in 4/5 (Bf1 is chosen even fully fresh at 200k nodes, so the
fcbf8e7 Hash 64 EPD verdict does not transfer exactly). No single component
is responsible: clearing the TT flips 3/5, history 2/5, correction history
3/5, the move history 1/5. These are near-equal choices that any change of
state flips. Systematically, over the 96 games the game state reaches depth
15.2 vs 13.5 fresh at the same node count, and where the two choose
differently (500 judged moves) the game-state move is better by +5.5 cp mean,
81 vs 36 better by >= 30 cp. The carry-over is a net gain: no TT-aging or
history-decay fix is warranted by this data. (Blunders 02, 60+0.6, and 23,
1M nodes, were not replayed: timed / other build.)
