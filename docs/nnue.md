# NNUE

## Data generation

Training data comes from fixed-node self-play by the engine itself
(`src/datagen.rs`), run at scale by the `Datagen` workflow
(`.github/workflows/datagen.yml`). The same files feed the NNUE trainer and the
Texel tuner (`tune` reads this format too and ignores the score).

### Format

One position per line, everything from White's point of view:

```
<FEN> | <score_cp> | <result>
r1bqkb1r/p1pp1ppp/1p5n/2n1p3/4P3/1PN3PP/P1PP1P2/R1BQKBNR w KQkq e6 0 6 | 53 | 0.5
```

`score_cp` is the search score of the position (the last completed iteration
of the fixed-node search), `result` is the game result: `1.0`, `0.5` or `0.0`.
The FEN's en-passant square is set after every double push, even when no
capture is possible.

### What is generated

- **Openings:** startpos plus 8 or 9 uniformly random legal plies (both sides
  get to move first). A start is rejected and redrawn if the game is already
  over or a 1000-node search scores it beyond ±400 cp. The random plies
  themselves are not recorded.
- **Moves:** `go nodes N` (default 5000), single thread, 16 MB hash; the TT and
  all history tables are cleared before each game.
- **Recorded positions:** every position played after the opening except when
  the side to move is in check, the chosen move is a capture or promotion, or
  the score is a mate score. That is about 80 positions per game (of ~118
  plies).
- **Game end:** mate, stalemate, 50-move rule, threefold repetition,
  insufficient material (no pawns/rooks/queens and at most one minor), or
  adjudication: win once |score| >= 1000 cp for 4 plies in a row with the same
  sign or a mate score is found, draw once |score| <= 10 cp for 10 plies in a
  row after ply 80, draw at 400 plies. About 85% of games end by adjudication;
  results are roughly 38% / 23% / 39% White / draw / Black.

### Determinism

`datagen <out.txt> <games> [nodes] [threads] [seed] [first_game] [softnodes=N]` plays games
`first_game .. first_game + games`. Game `g` depends only on `(seed, g, nodes)`
and the engine version, and games are written in index order, so the output is
byte-identical for any thread count, and shards that split one seed into
consecutive game ranges concatenate into exactly the single-run file (both
checked: 1 vs 4 threads, and 2 shards vs one run). Use a new seed for each
batch you want to add to the pool; batches with different seeds share no
openings except by chance.

Locally:

```bash
cargo build --release
echo 'datagen data.txt 1000 5000 4 1' | ./target/release/rustfish
```

Progress goes to stderr every 100 games (positions/game, games/s, pos/s, W/D/L,
mean nodes and completed depth per game move).

### Soft node limit (`softnodes=N`, opt-in)

By default every move's search runs into the `nodes` cap and throws the
unfinished iteration away: the label comes from the last *completed*
iteration. `softnodes=N` (any position after the positional arguments) stops
iterative deepening after the first iteration that completes past N nodes;
`nodes` stays as the hard cap. This changes the data (depth per label), so it
is off unless asked for, and a dataset made with it should be judged on
validation loss before it joins the pool.

Measured 2026-11-06 (E2 net, seed 78, 150 games, 1 thread, M3 under load):

| setting | nodes/move | mean depth | pos/CPU-s |
|---|---|---|---|
| `5000` (default) | 4997 | 6.22 | 227 |
| `10000 softnodes=2000` | 4548 | 6.35 | 248 |
| `10000 softnodes=1200` | 2907 | 5.59 | 403 |
| `25000 softnodes=5000` | 11545 | 7.84 | 98 |

`10000 softnodes=2000` is ~9% cheaper per move *and* slightly deeper than the
default; interpolating, the default's mean depth costs ~15% fewer nodes with
a soft limit (~1850). W/D/L and mean |score| differences between these runs
are within 150-game noise.

```bash
echo 'datagen data.txt 1000 10000 4 1 0 softnodes=2000' | ./target/release/rustfish
```

### On GitHub Actions

```bash
gh workflow run datagen.yml -f ref=main -f seed=1 -f shards=20 \
  -f games_per_shard=40000 -f nodes=5000 -f max_parallel=20
gh run watch   # summary lists positions per shard; data in the datagen-s<seed>-n<nodes> artifact
gh run download <run-id> -n datagen-s1-n5000
```

- `build` compiles once with `RUSTFLAGS=-C target-cpu=x86-64-v3` and prints
  bench; the merge job's summary records the engine SHA.
- `generate` runs one job per shard with `nproc` (4) threads, playing games
  `[shard * games_per_shard, (shard + 1) * games_per_shard)`. Each job is hard
  stopped at 320 min (inside the 6 h limit); a stopped shard keeps every complete
  line written so far, but then the merged file is no longer the exact
  single-run file. Shard artifacts are kept 3 days.
- `merge` concatenates the gzipped shards in shard order (gzip members
  concatenate into a valid gzip stream) into one artifact
  `datagen-s<seed>-n<nodes>` kept 90 days, and writes a summary. It also runs
  when some shards failed, merging whatever finished.
- The workflow competes with the SPRT queue for runners: lower `max_parallel`
  when the queue is busy.

### Throughput and sizes

Measured on the M3 laptop (shared, loaded), 5000 nodes/move: ~6 games/s and
~440-480 positions/s per thread (300 games with 4 threads: 1930 pos/s total).

GitHub runners (4 vCPU) are estimated, not yet measured: CI's plain-x86-64
bench runs at ~1.56 Mnps vs ~3.2 Mnps on the M3, x86-64-v3 and 4 threads on 2
cores with SMT should give roughly 5-6 Mnps per runner, i.e. **~850 positions/s,
~3M positions/hour, ~38k games/hour per runner**. The default 40,000 games per
shard is then ~1-1.2 h and ~3.2M positions; 20 shards give ~65M positions per
run. Check the shard logs of the first run and resize `games_per_shard` so a
shard takes 1-2 h (the hard stop leaves room for a 2-3x misestimate).

Sizes: ~65 bytes per position raw, ~12.5 bytes gzipped (5.2x), so ~40 MB per
shard and ~0.8 GB per 65M-position run artifact. The repo is public, so
artifact storage is not billed; the per-shard artifacts expire after 3 days.

Node count trades quality for quantity roughly linearly: 5000 nodes is the
usual starting point for a first net; later runs can use the first net's
engine and more nodes.

## Trainer

Two std-only trainers produce the net in `src/nnue.rs` (today `(768 x kb8m
king buckets -> 512) x 2 -> 8` output buckets, SCReLU, i16; see the header of
`src/nnue.rs` for the file layout):

- `tools/nnue-trainer` (Rust, CPU): reference trainer, bit-identical across
  ISAs and thread counts; also `eval [--best-scale]`, `scalecheck`,
  `rescale`. ~2x faster since 2026-11-07 (~1.7M pos/s on a 4-vCPU Zen 5
  runner, ~1.6M on the M3 at 8 threads), with `--factorise`, `--keep-best`
  and exact warm starts.
- `tools/nnue-trainer-metal` (Swift + Metal, M3 GPU): ~1.8M pos/s at 512x8;
  trains the generations. `--keep-best` and exact factoriser init too.

Loss is `lambda * (sigmoid(eval/400) - result)^2 + (1-lambda) *
((eval - score)/400)^2` with lambda 0.5 (0.3 was tried and rejected), AdamW,
cosine lr. Flags, layouts and formats are in each trainer's README;
`scripts/netloop.py` drives them (`docs/nnue-experiments.md`). Acceptance is a
`net-gate` (same shape) or `stc` (new shape) SPRT like every other change.

## Threat inputs (`--threats ep`): layout spec

Branch `threat-inputs-v2`. The engine has `THREATS` in `src/nnue.rs`
(default `None`: net format, bench and evalhash unchanged). The CPU
trainer lib and the Metal trainer take `--threats <none|ep>`; `ad`
(attacked / defended, 2 blocks) is engine-only (up to 92 features per
perspective) and gave no gain on a small test.

**Definition (`ep`, 1 block).** A non-king piece is "en prise" when it is
attacked by an enemy piece of lower value (P < N = B < R < Q; the king
counts as no lower attacker of anything), or attacked by any enemy piece
(king included) and not defended by any own piece (king included). Plain
attack sets, no x-rays, no pins, no side-to-move dependence. It is a
property of the position, so both perspectives activate the same pieces,
each in its own orientation.

**(a) Feature count.** `Sample.f` is `[[u16; MAXF]; 2]` with `MAXF = 64`
(was 32). `Sample.n` = piece features + threat features (the same for both
perspectives), new `Sample.np` = piece features. Worst case with `ep`:
32 pieces + 30 threat rows (kings never) = 62 <= 64, so overflow cannot
happen; no truncation code. The output bucket still comes from the piece
count (`np`).

**(b) Indexing.** Per perspective, `f[side][..np]` are the piece features
exactly as before (square order of the record), then `f[side][np..n]` the
threat features, in the same square order (the order of the en-prise
pieces in the record). Threat row of a piece = its piece feature with the
king bucket's offset replaced by the threat block's:

    threat = (KB + t) * 768 + side * 384 + kind * 64 + (sq_rel ^ flip)

with t = 0 for `ep`, so rows `KB*768 .. KB*768 + 768` (kb8m: 6144..6912).
`side` = 0 own / 1 enemy (perspective-relative), `sq_rel` = sq (White
perspective) or sq ^ 56 (Black), `flip` = the perspective's kb8m mirror
xor (7 when its own king is on files e-h, else 0) -- the mirror **is**
applied to threat rows, the bucket is **not** (one shared 768-row block).
`--mirror` augmentation flips files first, as for piece rows. In code:
`f[side][np + j] = f[side][i] - bucket_base[side] + KB*768`.
`king_bucket(s, side) = f[side][0] / 768` stays correct (f[side][0] is a
piece feature).

**(c) Shapes.** `Shape` gains `th` (threat id: 0 none, 1 ep) and
`tb()` = blocks (0/1). `ft_rows() = (kb + tb) * 768`, `ft_kb() = kb*768*H`
(end of bucketed rows), `ft_b() = ft_rows() * H`. Param vector / net file
(i16 LE, unchanged order): FT weights `(kb + tb) * 768 x H` (feature-major:
bucket rows, then threat rows), FT bias H, output weights `BUCKETS x 2H`,
output biases `BUCKETS`. So a threat net file is `tb*768*H*2` bytes larger
and the threat rows sit between the bucketed rows and the FT bias.
Factoriser: `F` stays 768 x H, stored after the net params, and merges into
the bucket rows only (`r < kb*768`); threat rows are plain parameters
(plain AdamW update, no merge, never touched by `F`). Checkpoint header
word 1 gains `threat id << 48` (0 = old checkpoints unchanged).
Warm start: `train --init <net without threat rows> --threats ep` copies
everything and zero-fills the threat rows (also with `--factorise`: W = the
net's bucket rows, F = 0, threat rows = 0), so the start evaluates exactly
like the source.

**(d) Weight decay.** `--wd-active-only` flags cover all FT rows,
threat rows included: a threat row decays only in batches where it occurs.
(Threat rows of en-prise pieces occur in almost every batch, so in practice
they decay like any hot row.)

**CUDA notes (for the desktop session).** GSample needs 64 features per
perspective (or a separate 32-wide threat list, `f[side][np..n]`);
`backward_ft` / `plan_masks` can treat the threat block as one extra
768-row "bucket" (block index KB) that every pair touches; `adam_ft` must
apply the factoriser merge to bucket rows only and a plain update to the
`tb*768` threat rows. Until then the CUDA trainer must refuse `--threats`.
