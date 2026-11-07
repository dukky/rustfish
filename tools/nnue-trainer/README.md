# nnue-trainer (v2)

Std-only trainer for the `(768 -> H) x 2 -> B` SCReLU nets in `src/nnue.rs`
(`B` = output buckets by piece count; the engine currently runs H = 256, B = 1).
See the module docs in `src/main.rs` for the maths and quantisation.

```bash
cargo build --release
T=target/release/nnue-trainer
gunzip -c datagen-s9-n8000.gz | $T convert - s9.bin   # text -> 32-byte records
$T shuffle --out pool.bin s4.bin s5.bin s8.bin s9.bin  # optional global shuffle
$T train --data pool.bin --val val.bin --out net.bin --hidden 256 --epochs 20
$T eval --net ../../net/nnue.bin --val val.bin          # same metric for any i16 net
```

Rules that the v1 history taught us (see `docs/nnue-experiments.md`):

- **Always train with `--val`** on positions never trained on, and compare
  candidates by validation loss *before* spending CI games. The quantised
  validation loss is printed too; it should be within ~1% of the float one.
- Change **one variable per experiment** and record it in the log.
- Train on the **whole valid pool**, many epochs, cosine LR decay
  (`--lr 1e-3 --lr-end 1e-5`, batch 16384). Two epochs on one generation's
  data (what v1 did on a 2-core box) leaves most of the net's strength unused.
- Every weight is clipped to +-1.98 while training, so output weights export
  to |w| <= 127 (the engine's fast path) with no post-hoc clipping.

Throughput: ~1.1M positions/s at H = 256 on the 8-core M3 (8 threads).

## King buckets (input buckets)

`--king-buckets <layout>` (train and eval; default `none`) gives each
perspective its own block of 768 feature rows per bucket of its own king
square: `feature = bucket*768 + side*384 + kind*64 + sq`, with squares in
the perspective's orientation. Mirrored layouts also flip every file
(`sq ^ 7`) for a perspective whose king is on files e-h.

| layout | KB | mirror | buckets (perspective-relative king square) |
|---|---|---|---|
| `none` | 1 | no | today's net, bit-for-bit |
| `m1` | 1 | yes | mirroring only |
| `kb4` | 4 | no | rank 1 a-d / rank 1 e-h / rank 2 / ranks 3-8 |
| `kb4m` | 4 | yes | rank 1 a-b / rank 1 c-d / rank 2 / ranks 3-8 |
| `kb8m` | 8 | yes | rank 1: a, b, c, d each; rank 2: a-b, c-d; ranks 3-4; ranks 5-8 |

The net file grows to `(KB*768*H + H + B*(2H+1))` i16; nothing in the file
names the layout, so pass the same `--king-buckets` to `eval` and set the
engine's `KING_LAYOUT` (`src/nnue.rs`) to the matching `LAYOUT_*` constant
when embedding (`scripts/netloop.py embed --king-buckets <layout>` does that). `--init` from a one-bucket net copies its 768 rows into
every bucket (start loss equals the one-bucket net's for unmirrored
layouts). Checkpoints store the layout id in the high half of the buckets
header word, so one-bucket checkpoints are unchanged. The `--mirror` data
augmentation still applies (to the board, before buckets are chosen).

## Score-scale check

```bash
$T scalecheck --net ../../net/nnue.bin --data s18.bin [--buckets 1] [--n 200000] [--threads 2]
```

Over the first `--n` records: least-squares slope (through 0) of the
record's score on the net's eval (both cp, White POV), Pearson r, mean
|score| and mean |eval|; mates (`|score| >= --max-score`, 10000) are
skipped. Data written at the net's scale gives ~1.0-1.15 against the main
net (search scores run a bit wider than static evals); data written at the
engine's halved scale (`eval = net/2` since ad53daa) gives ~0.57. Data from
much older, weaker nets reads lower (s4: 0.65, s9/s11: 0.81). Pass
`--king-buckets` for a king-bucketed reference net (a one-bucket reference
net measures scale just as well). `scripts/netloop.py auto` uses it to
triage new data.
