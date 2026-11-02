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

`datagen <out.txt> <games> [nodes] [threads] [seed] [first_game]` plays games
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

Progress goes to stderr every 100 games (positions/game, games/s, pos/s, W/D/L).

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
