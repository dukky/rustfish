# Critical-position suite (offline search screen)

662 positions from real games where our own deep search clearly prefers one move,
plus a runner that scores a binary on them at fixed nodes, and the calibration run that
measures how well that score tracks SPRT.

**Verdict (21 Nov 2026): not good enough to use as a pre-queue screen.** Its noise floor
is about the size of a 15-25 Elo change, and it is biased toward the binary that made the
labels. Details are under Calibration.

## Files

| file | what |
|---|---|
| `suite.jsonl` | the positions with their labels (one JSON per line) |
| `suite-extra.jsonl` | cache of reference scores for moves the labels did not cover (filled by `run.py --ref-engine`) |
| `build.py` | `extract` (PGN -> candidates), `label` (deep searches), `select`, `peek` |
| `run.py` | `run` (score one binary), `compare` (paired diff with bootstrap CIs) |
| `calib.py` | runs a directory of binaries and prints the calibration table |
| `chesslite.py`, `uci.py` | stdlib move generator and SAN parser (perft-verified), UCI driver |

## Sources
3660 games, downloaded read-only with `gh run download`:
- Clover 9.1 peer matches: run 44545275669 (60+0.6, 400 games), 44575415699 (10+0.1, 800
  games) and 44543687348 (depth 11, 800 games).
- CCRL ladder runs (120+1, 1660 games against 10 anchors).

A position becomes a candidate when at least two of three PGN-eval signals show a swing
of 100 cp or more: the mover's eval against the opponent's next eval, the opponent's
before/after evals, and the mover's eval two plies later. The eval must also be at most
2.5 pawns, and each game gives at most 3 candidates. That gives 2610 candidates. The peer
evals only propose positions; they are never used as labels. The 3658 "random" control
candidates were extracted but not labelled, because there was not enough time.

## Labels (own engine only)
Reference binary: commit `12a53ae` (main engine, bench 268722), built for M3, 1 thread,
Hash 64, a fresh process for every search, `nice -n 19`. For each candidate:
1. A 6M-node root search gives the best move B and its trace, plus a depth-3 search.
2. The candidate is dropped if B is already found from depth 4 or less (trivial), if B
   only became best in the last 2 iterations (unstable), or if |score| > 400 (decided).
3. B and up to 3 alternatives (the game move, the depth-3 move, and earlier PV moves) are
   each searched as a child at 3M nodes. `score` is the child score of B, and
   `alts[m]` is the child score of move m.
4. `select` keeps positions with gap = score - max(alts) >= 40 cp.

662 pass: 203 opening, 292 middlegame, 167 endgame. 1212 were skipped as
trivial, unstable or decided, and 736 had a gap below 40.
Labelling took about 80 min of wall time at 3 parallel processes, on a shared, loaded laptop.

## Usage
```bash
python3 scripts/diag/suite/run.py run --engine ./target/release/rustfish --nodes 1000000 \
    --parallel 3 --ref-engine <label binary built from 12a53ae> --out /tmp/x/cand.jsonl
python3 scripts/diag/suite/run.py compare /tmp/x/base.jsonl /tmp/x/cand.jsonl
```
Each position gets one search at N nodes. The moves the engine would play at N/2, N/4
and N/8 are read off the same deterministic trace (the "ladder"). Metrics are:
- solve rate at the top budget and averaged over the ladder;
- loss (reference cp loss of the chosen move, capped at 500) and mwpl (milli win-prob loss);
- geometric mean nodes-to-solve.

`compare` bootstraps the paired differences over positions.
A run takes about 3 min per binary for 662 positions at 1M nodes with 3 processes
(168-240 s measured under shared load).

## Calibration (1M nodes, ladder 1M/500k/250k/125k; every build compared with main)
Builds:
- `main2`: a byte-identical copy of main.
- `dummy1-3`: LMR +1 / -1 / +1 when 3 key bits match (1/8 of reduced moves).
- `R-*`: a merged change reverted, so the expected effect is -SPRT.
- `F-*`: a rejected change applied.

Positive = better than main.

| build | SPRT | moves changed | d solved (ladder) [95%] | d loss cp [95%] | d log nts |
|---|---|---|---|---|---|
| main2 | 0 | 0 | 0 | 0 | 0 |
| dummy1 | ~0 | 165 | -14.0 [-29, +1] | +1.2 [-4.2, +6.5] | -0.02 |
| dummy2 | ~0 | 174 | -27.0 [-43, -11] | -2.1 [-6.9, +2.7] | -0.07 |
| dummy3 | ~0 | 186 | -27.5 [-43, -12] | -3.0 [-7.6, +1.6] | -0.04 |
| R-pv-no-tt-cut | -24.8 | 170 | -20.8 [-37, -6] | -0.9 [-6.4, +4.5] | -0.00 |
| R-tt-failow-nomove | -16.7 | 169 | -33.3 [-50, -17] | -4.1 [-9.3, +1.0] | +0.02 |
| R-qsearch-fail-soft | -14.9 | 180 | -26.3 [-43, -10] | -0.1 [-5.8, +5.5] | +0.03 |
| R-prune-exempt-killers | -11.1 | 185 | -23.3 [-41, -7] | -3.9 [-8.9, +0.9] | -0.07 |
| F-iir-depth3 | -6.9 | 167 | -17.5 [-34, -2] | -0.6 [-5.5, +4.2] | +0.02 |
| F-history-bonus-linear | +2.6 | 159 | -26.0 [-42, -10] | -1.9 [-7.1, +3.4] | +0.08 |

Findings:
- **Runs are deterministic.** The same binary gives exactly the same result twice.
- **Every change looks worse than main, the dummies included** (14-28 positions lost).
  The labels come from main's own deeper search, so any perturbation that changes the
  tree drifts away from that search's choices. This self-reference bias (about -22
  solved) is larger than any real effect. Comparing against a dummy instead of main
  cancels it, but then the noise is the full spread between dummies.
- **Noise floor.** The standard deviation between dummies is about 8 positions solved
  and 2.2 cp of loss. About 25% of moves change under any perturbation. The SPRT changes
  (|Elo| 7-25) all fall inside the dummy range.
- **Sign agreement looks good but means little.** It is 5/6 on solved and loss, but only
  because almost everything comes out negative. Spearman rho against SPRT is +0.20
  (solved), +0.09 (loss) and +0.54 (nts, with nts deltas near 0). The suite cannot rank
  these changes, and it would not detect a 25 Elo change from one build.

## If someone picks this up
- Labels from a different, much deeper search (for example 100M+ nodes, or a consensus
  of several perturbed builds) would reduce the self-bias. Baselines should be an average
  of 3+ dummy builds, not main.
- Power grows as the square root of the number of positions. Resolving ~10 Elo would
  need roughly 10x more positions than the dummy spread allows now, which costs about
  the same CPU as a short SPRT.
- Use the suite only to investigate specific mechanisms (for example, which positions a
  change fixes), not to accept or reject changes.
