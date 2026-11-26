# Label quality: nodes per move for training labels (Nov 2026)

Question (owner, 26 Nov): our labels are 5000-node searches (depth ~7.9 in
datagen). What node count gives the most Elo per CPU-hour, and can we prove
it?

Status: phase 0 done (tool + throughput + a label-noise proxy); phase 1 (same
positions, different labels, identical fine-tunes, fixed-node games) running.
Results are filled in below as they land.

## Phase 0: the `relabel` command

`relabel <in> <out> [nodes] [threads] [softnodes=N] [clear=game|pos] [hash=MB] [skip=N] [limit=N]`
(`src/relabel.rs`, branch `datagen-relabel`). It reads the text training
format (`.txt`) or the trainer's 32-byte records (anything else; an `.rfp`
goes through `nnue-trainer unpack <in.rfp> /dev/stdout`) and writes the same
format with each score replaced by a fresh `go nodes N` search (White's view,
net units, exactly as datagen records it). The position and game result are
kept. Records carry no castling rights, en passant or move counters:
castling is inferred from kings and rooks on their home squares (on a 486-position
test the record path gave labels identical to the text path with real FENs).
Output is byte-identical for any thread count (blocks of whole games, each
from a cleared engine, written in input order). Mate scores are saturated to
+-32000. Bench unchanged (216602); perft, selftest, cargo test pass.

### State along a game (decided: `clear=game`, the default)

Datagen's labels come from searches that inherit the TT and history tables
of the earlier moves of their game. Relabelling a position cold loses that:

| 5000 nodes, same positions | completed depth | label MSE vs 1M-node ref (sigmoid, x1e-4) | pos/s, 10 threads |
|---|---|---|---|
| datagen in-game label (gen 11 engine, every ply searched) | 7.88 | 57.9 | (2975 at 8 threads, incl. game play) |
| relabel `clear=game` (TT/history reused along the game) | 6.99-7.07 | 56.2 | 3819-4147 |
| relabel `clear=pos` (cleared before every position), 16 MB | 6.10 | 67.9 | 1154 |
| relabel `clear=pos`, 1 MB hash | 6.10 | | 2627 |

Reusing the TT along the game is worth about one ply at 5k nodes and about a
doubling of nodes in label error (n5000pos ~ n2500), and per-position clears
cost a 16 MB memset each (3.6x slower at 10 threads: memory bandwidth). So
relabel works on game-ordered input (datagen text, game-ordered `.rfp`
sources, not shuffled pools) and clears at game starts (result change,
piece count rising, or fullmove number falling). Relabelling sees only the
kept positions (datagen also searched the skipped capture/check plies), so it
reaches ~0.8 ply less than datagen at the same nodes, but its label error is
the same as datagen's own (56.2 vs 57.9; the engine also differs: main = gen 12
net vs datagen's 44bbe8b = gen 11).

### Throughput (desktop Ryzen 5 9600X, 6c/12t, `-C target-cpu=native`, AVX2 build of 49ed436)

Input: 400k game-ordered records of D2042; the GPU trainer was running
alongside (1-2 cores), datagen paused.

| nodes/pos | 1 thread pos/s | 6 t | 8 t | 10 t | 12 t | depth | label MSE x1e-4 (proxy) |
|---|---|---|---|---|---|---|---|
| 100 (per-position overhead) | 23 800 | | | 84 300 | | 0.8 | |
| 1 000 | | | | 7 430 | | 3.4 | 79.5 |
| 2 500 | | | | 3 720 | | 5.4 | 66.7 |
| 5 000 | 749 | | | 3 820-4 150 | | 7.0 | 56.2 |
| 10 000 | | | | 833 (*) | | 8.5 | 48.1 |
| 20 000 | 149 | 558 | 656 | 745-799 | 817 | 9.9 | 35.7 |
| 50 000 | 66 | | | 146-180 | | 11.6-11.9 | 25.2 |
| 100 000 | | | | 75 | | 13.6 | 19.2 |
| 200 000 | | | | 38 | | 15.5 | 12.1 |
| 1 000 000 (reference) | | | | 7 | | 20.6 | (ref) |

(*) The 10k/50k-200k rows at 10 threads come from the proxy runs on 10k
positions (1-4 min each), the rest from 25k-100k-position runs.

- Cost per label grows faster than nodes: 20k costs 5.0x a 5k label, 50k
  ~24x, at 10 threads (1 thread: 5.0x and 11.3x). Larger searches scale worse
  across threads (10 threads give 5.1-5.5x one thread at 5k, 5.0-5.4x at
  20k, but only 2.2-2.7x at 50k: shared L3/memory), so high-node labelling
  wants many physical cores with large caches, not SMT threads. (Measured
  with the GPU trainer alongside; one run per cell, spread between repeats
  ~5-10%.)
- Per-position overhead is ~40 us total at 100 nodes (most of it the 100
  nodes): negligible at >= 1k nodes. Output formatting and I/O are binary
  copies (32-byte records). There was no cheap exact win in the labelling path
  beyond `clear=game`: TT size 2/4/16/64 MB moves 20k throughput by +4/+3/0/-4%
  (16 MB kept, as datagen).
- `softnodes=N` (stop after the first iteration past N, hard cap 4N): s5000
  spends 8.7k nodes for MSE 46.8 (n10000: 10k for 48.1); s10000 17.4k for 36.2
  (n20000: 20k for 35.7); s20000 32.9k for 26.9 (n50000: 50k for 25.2). About
  10-15% fewer nodes for the same label error, and wall-time slightly worse
  per node (deeper iterations): a small win, not a different regime.

### Label-noise proxy (10k positions of D2042, 109 games; reference = 1M nodes, depth 20.6)

MSE in the trainer's sigmoid space (score / 400 in net units) against the
1M-node label; positions where the reference is a mate excluded (174), same
numbers within 1%. Median |label - ref| in cp: 1k 56, 2.5k 50.5, **5k 46**,
10k 41, 20k 38, 50k 32, 100k 27, 200k 22; datagen's own 5k label 47.

Label error falls about linearly in log(nodes): roughly -9e-4 per doubling,
from 80 (1k) to 12 (200k). The reference has its own error, correlated with
the deeper labels (same engine), so this overstates the gain of the deepest
labels somewhat; it says nothing yet about Elo, which phase 1 measures.

## Phase 1: same positions, different labels

Design (one variable per arm):

- **S** = the first 20M records of D2042 (desktop seed, engine 44bbe8b,
  ~210k games, game order; never in any pool used by the ablation nets, not in
  any validation set).
- Arms, all on the same positions S: **orig** (datagen's 5k labels, gen 11
  engine), **r5k** (relabelled by main at 5k nodes, `clear=game`), **r20k**
  (main at 20k nodes). orig vs r5k = engine/procedure; r5k vs r20k = nodes only.
- **Quantity arm q100m**: the first 50M records of D2042 and of D2041,
  original 5k labels (100M positions, contains S). CPU cost of 20M 20k labels
  (~7.2 h at 10 threads) buys ~95M new 5k positions from datagen (3400-3700
  pos/s at 10 threads), so q100m is the equal-CPU alternative to r20k.
- Training: identical fine-tunes of the final kb8m stage from
  `ablate1/ctlAm1.bin` (batch-1 control's m1 net) with the ctlA final-stage
  flags (cuda a808d8a, `--king-buckets kb8m --factorise --lr 2e-4 --lr-end
  1e-6 --half --head pairwise`, shuffle buffer 64M, stride 4), the same 100M
  samples for every arm (5 epochs of S, or 1 of q100m), seed 1 (and seed 2
  for noise). No `--keep-best`: picking epochs on a 5k-label validation set
  would bias towards 5k labels.
- Validation: val-g12 at lambda 0.5 (as usual) and **lambda 1 (game results
  only: blind to label quality)**, plus **vhi** = 100k positions of val-g12
  (50k each from D2037 and D2038) relabelled at 100k nodes (depth ~13.6),
  scored at lambda 0 (eval vs deep label) and 0.5, and vq5k (the same
  positions at main 5k) at lambda 0.
- Games decide: each arm net is permuted, embedded on a branch from
  origin/main (`lbl-<arm>`), and arms play fixed-node SPRTs (200k nodes,
  [-5, 5], 4 waves): r20k vs r5k, r20k vs q100m, r5k vs orig.

Results: pending.
