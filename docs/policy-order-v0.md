# policy-order v0: gates A and B (2026-11-25)

Question: can a small learned move scorer in the MovePicker improve
rustfish's move ordering enough, and cheaply enough, to gain Elo? The
planned phases: an upper bound first (A), then the inference cost (B), and
labels plus a model (C) only if A and B both pass.

**Result: gate A passes on paper, gate B fails. Gate C was not started.
No-go for phase 2 in this form.** A perfect quiet scorer would save ~13% of
fixed-depth nodes; a realistic one (right 60-80% of the time where it acts)
saves 2-6%. A 32-64 wide scorer costs 4-7% NPS as integrated, and 1-4% for
the arithmetic alone. Even with free inference, perfect quiet ordering is
worth roughly +10 Elo, not +20.

## Tooling

- Branch `policy-oracle`, cargo feature `policy-oracle`. The default build
  is unchanged: bench 198243 = main, both with and without the feature.
  - `oracletest <deep> <shallow> <arm>... | position ...` searches `deep`
    plies first. That search's TT (256 MB) becomes a read-only oracle. Each
    arm then clears all state (TT 64 MB, histories, killers, correction
    history) and searches `shallow` plies. At every move-loop node it looks
    up the oracle's best move. The move counts only if the entry's draft is
    at least the node depth plus `margin`. The arm then places the move as
    its `mode` says, and prints the tree counters. The root is never touched.
  - Modes:
    - 1: oracle move before the TT move.
    - 2: oracle move right after the TT move.
    - 3: like 2, but only when the oracle move is a quiet.
    - 4: the oracle quiet goes first among the quiets: after the good
      captures, ahead of killers and counter. This is what a bounded quiet
      bonus can do.
    - 5: the oracle quiet goes first among the non-priority quiets only.
    - 6: wrong-oracle control: a different quiet takes mode 4's slot.
    - 7: imperfect scorer: mode 4 at pct% of the nodes, mode 6 at the rest.
  - Other arm fields: `pct` applies the arm at that share of nodes (by key
    hash); `noise` scales the history bonus by x/100 (no-op seeds 99, 100,
    101); an optional `mindepth`.
- `scripts/policy/run_oracle.sh` (2 shards, nice 19) and
  `scripts/policy/oracle_sum.py`. The summary gives, per arm:
  - total nodes against the base arm (sum ratio, and the mean per-position
    log ratio with its standard error);
  - first-move cut rate by depth, and moves searched per fail high;
  - oracle coverage;
  - root-move agreement with the deep search.
  For the base arm it also gives the class of the cutting move when it is
  not searched first.
- Positions: the 962-position set (`~/agent-runs/qs/pos962.txt` on the
  desktop: diag-mid300 plus the diag-suite FENs). Oracle depth 17 (857k
  nodes on average); arms at depth 13. Desktop, 2 processes, nice 19.
  Raw output is in `~/agent-runs/policy/out/A{1,2,3}`.

## Gate A: upper bound (962 positions, depth 13, oracle = depth 17)

Noise floor:
- The no-op arms (history bonus x0.99 and x1.01) give log ratios of +0.005
  and -0.009, with a standard error of 0.010.
- Their sum ratios are 1.019 and 1.003.
- Reruns of the base arm are bit-identical.

So effects below ~2% are noise.

| arm | what it models | nodes (sum) | log ratio | first-move cut | moves per fail high |
|---|---|---|---|---|---|
| base | main | 1.000 | 0 | 81.9% | 1.485 |
| no-op x0.99 / x1.01 | noise | 1.019 / 1.003 | +0.005 / -0.009 (se 0.010) | 81.8% | 1.485 |
| 1, margin 0 | perfect ordering of every move, ahead of the TT move | **0.725** | -0.292 | 89.2% | 1.275 |
| 2, margin 0 | perfect, after the TT move | 0.814 | -0.193 | 87.9% | 1.314 |
| 3, margin 0 | perfect quiet, right after the TT move (ahead of captures) | 0.846 | -0.152 | 86.6% | 1.359 |
| **4, margin 0** | **perfect quiet bonus (after good captures, ahead of killers)** | **0.872** | **-0.129** | 84.5% | 1.389 |
| 4, margin 2 | same, oracle at least 2 plies deeper than the node | 0.891 | -0.109 | 83.4% | 1.421 |
| 5, margin 2 | perfect, but only within the plain quiets | 0.968 | -0.034 | 82.0% | 1.466 |
| 4, margin 0, at 50% of nodes | right half the time, heuristic otherwise | 0.948 | -0.059 | 82.9% | 1.443 |
| 7, 80% right | right 80%, a wrong quiet first 20% | 0.936 | -0.067 | 82.8% | 1.429 |
| 7, 60% right | right 60%, wrong 40% | 0.977 | -0.036 | 81.4% | 1.462 |
| 6 | always wrong (a different quiet first) | **1.095** | +0.087 | 78.4% | 1.541 |
| 4, depth >= 2 / 3 / 4 | perfect quiet bonus at deeper nodes only | 0.896 / 0.907 / 0.929 | -0.111 / -0.097 / -0.068 | | |
| 7, 80% right, depth >= 3 | realistic, deeper nodes only | 0.959 | -0.052 | | |

Readings:
- **A perfect quiet scorer saves 12.8% of nodes** (13-15% if it may also
  jump ahead of the good captures). This clears the pre-set 5% bar.
- **Accuracy dominates.** A wrong quiet placed first costs +9.5%. At 80%
  precision the saving is 6.4%; at 60% it is 2.3%; break-even is near 50%.
  These precisions are measured against a depth-17 search where it disagrees
  with the TT move. The existing heuristics already put the cutting move
  first 82% of the time, and the misses are the hard cases. So 80% precision
  on exactly those nodes is optimistic for a <1M-parameter model.
- Late cutters (the cutting move was not searched first, base arm; 16.6% of
  fail highs at d1-3 CUT nodes, 10.5% at d4-7, 5.3% at d8+): about 80% are
  quiets and 20% captures.
  - At CUT nodes 38-45% of late cutters are killers or the counter-move:
    they cut after a TT move or capture that failed first. Reordering among
    the quiets cannot fix those.
  - Moves with h>0 or ch>0 make up 29-36%; plain quiets 8-20% (61% at
    d1-3 PV nodes).
  - Pre-cutter waste is 19% of the tree at d1-3, 20% at d4-7 and 29% at
    d8+ (the levels overlap).
- **Quality caveat:** the oracle arms cannot be checked for equal quality.
  The oracle knows the depth-17 answer, so root agreement rises (74% vs 70%
  for base) and the score gap to depth 17 shrinks (33 vs 41 cp). This
  contamination flatters the oracle, so the numbers above are upper bounds.
- Consistent with the 22 Nov tree audit (quiet ordering knobs inside
  noise). The upper bound is real, but it needs a much better predictor of
  the late cutter than history, not a reweighting.

## Gate B: inference cost (GitHub runners, `policy-speed` workflow)

Prototype on branches `policy-mlp-*` (from main 30f83d4; tree = main, bench
198243 checked by the workflow on every runner):
- **Architecture:** a random-weight scorer, (768 + 2 x 384 + 2) x H
  parameters (H = 64: ~98k).
  - Per node: a context vector = bias + one row per piece on the board
    (side-relative).
  - Per quiet move: from-row + to-row for the moving piece; clamp to
    [0, 255]; dot with the output weights. i16 lanes, autovectorised for
    x86-64-v3.
- **How the cost is measured:** the score reaches the move as
  `black_box(s) & 0`, so the tree is unchanged and the NPS difference is
  pure cost.
- **Eager** (`e*`): every quiet is scored when the picker is built.
- **Lazy** (`l*`): quiets are scored only when the picker first reaches a
  quiet stage.
- **Variants:** `d2` and `d3` score only at depth >= 2 or 3; `noctx` skips
  the context sum; `never` has the lazy hook but never scores (a plumbing
  control).
- **Method:** `speed-compare` copied to `.github/workflows/policy-speed-compare.yml`
  (branch `policy-speed`, settings in `scripts/policy-speed.env`). Three
  rounds, 6 runners each, 10 interleaved rounds per runner, bench depth 13,
  median paired NPS change. A/A control (main twice): -0.7..+0.2%.

| variant | round 1 | round 2 | round 3 | read |
|---|---|---|---|---|
| main A/A | -0.2% | -0.7% | -0.1% | noise |
| `l64never` (plumbing only, never scores) | | | **-3.1%** (-1.0..-3.8) | code-layout cost of the hook alone |
| `l32` lazy H32 | **-3.0%** | -5.3% | -5.4% | |
| `l32d3` lazy H32, depth >= 3 | | -4.2% | | |
| `l64` lazy H64 | -4.5% | -6.8% | -7.2% | |
| `l64d2` | | -5.8% | | |
| `l64d3` lazy H64, depth >= 3 | | -4.9% | -4.3% | |
| `l64noctx` (per-move part only) | | -5.9% | | the per-move dot dominates, not the context |
| `l128` lazy H128 | -7.5% | | | |
| `e32` / `e64` eager | -4.5% / -7.2% | | e64 -9.5% | |

Spread across runner CPUs: round 1 drew mostly Intel runners; rounds 2-3
drew mostly AMD EPYC 9V74 / 7763 runners, which pay more.

Readings:
- As integrated, the cheapest useful sizes cost **4.3% (H64, depth >= 3)
  to 5.4% (H32, all depths)**. That is over the 3% limit, so **gate B
  fails**.
- The arithmetic alone (variant minus the plumbing control) is about 1.2%
  (H64, depth >= 3), 2.3% (H32) and 4.1% (H64). The 3.1% plumbing cost is a
  codegen/layout effect of putting any hook into the hot picker/negamax
  path. A careful integration might shrink it, but it cannot be assumed
  away: the same layout sensitivity shows up in the speed agent's work.
- A model with real features (threat squares, attacked-by, captured piece,
  history as input) would cost more than this dummy, not less.

## Net estimate

| scenario | node saving at fixed depth | NPS cost | net |
|---|---|---|---|
| perfect quiet scorer, all depths, H32 | 12.8% | 5.4% (2.3% if the plumbing were free) | +7% (+10%) |
| perfect, depth >= 3, H64 | 9.3% | 4.3% (1.2%) | +5% (+8%) |
| 80% precise, depth >= 3, H64 | 4.1% | 4.3% (1.2%) | ~0 (+3%) |
| 60% precise, all depths | 2.3% | 5.4% | -3% |

At a rough 7-10 Elo per 10% of speed at this level (an assumption, not measured here), even the perfect case
is +5..+8 Elo before quality effects. A realistic scorer lands at 0..+3 Elo,
which no SPRT can resolve. The +20 Elo target is out of reach for
quiet-ordering-only use. The full oracle, which reorders everything
including the TT move, bounds all ordering work at -27.5% nodes, about +20
Elo with zero cost and perfect knowledge.

## Recommendation

**No-go for phase 2** (ordering-only integration) in this form. Evidence
strength: gate A is one controlled run per arm with 3 no-op seeds, 962
positions and tight standard errors. Gate B is 3 CI rounds x 6 runners with
A/A and plumbing controls.

Not tested, and the only directions that would change the verdict:
1. A scorer used for **reductions** (phase 3 in the original plan), where
   one score per move could replace several LMR terms. The ordering bound
   here says nothing about that.
2. A far cheaper scorer: incremental per-node context, H <= 16, and only
   at nodes where the first move has already failed (about 18% of fail-high
   nodes). The arithmetic cost would fall below 1%, but the precision
   requirement stays: it must beat 50% on exactly the cases the heuristics
   miss.

Gate C (labels and model) was not started, per the gate rule. If it is ever
revisited, the cheapest label source is the oracle tooling itself: one
depth-17 search yields deep-draft best moves for thousands of in-tree nodes,
which matches the in-search distribution better than game positions do.

The older branch `feat-policy-order-v0` (24 Nov: Python ranker prototype
with a requirements.txt, top-1 13.5% vs 12.2% on 74 positions) is
superseded by this note. Its tools also need non-stdlib Python packages, which the
project's stdlib-only tooling rule excludes.
