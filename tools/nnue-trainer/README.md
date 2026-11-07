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

## Speed (batch kernels, `src/batch.rs`)

The forward pass runs by samples over 64-float column tiles of a tile-major
FT copy; the backward pass runs by 16-float column tiles over one tile-major
gradient (no per-thread gradients, no reduction); AdamW is vectorised and
clears the gradient. AVX-512 / AVX2 are picked at runtime (one binary runs on
any x86-64; `NNUE_ISA=portable|avx2|avx512` overrides), NEON on aarch64.

Results do not depend on `--threads`, scheduling or ISA (bit-identical nets).
The only change from the old trainer is the order of the output dot product
(16 interleaved partial sums); `--exact` restores the old sequential sum and
is then **bit-identical to the old trainer run with `--threads 1`** (checked
by the unit tests and by the `Trainer bench` workflow). `--sub-batch N`
(default ~2 MB of accumulators, 512 at H = 512) and `--max-batches N`
(benchmarking) are tuning knobs.

H = 512, B = 8, batch 16384, `--mirror`, positions/s (GitHub 4-vCPU runners,
`trainer-bench.yml`; old = the previous trainer at 4 threads):

| CPU | old | new | new 1 thread | new `--exact` |
|---|---|---|---|---|
| AMD EPYC 9V45 (Zen 5) | 0.71-0.82M | 1.69M | 0.65M | 1.43M |
| AMD EPYC 9V74 (Zen 4) | 0.46M | 0.82M | 0.39M | 0.70M |
| Intel Xeon 8573C | 0.43M | 0.90M | 0.40M | 0.77M |
| AMD EPYC 7763 (Zen 3, AVX2) | 0.41M | 0.69M | | |

M3 (8 threads, quiet): ~1.6M (was ~0.5M). The kernels synchronise a few
dozen times per batch, so they lose more than the old loop when the cores are
oversubscribed (e.g. next to datagen): use fewer threads then.

Build for a dedicated x86 box (runtime dispatch already uses AVX-512; native
only helps the decode/validation code):

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

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

## Output-scale rescale

```bash
$T rescale --net g4w.bin --out g4ws.bin --buckets 8 --factor 0.865 [--val val.bin]
$T rescale --net g4w.bin --out g4ws.bin --buckets 8 --match g3.bin --val val.bin \
   [--norm slope|mean-abs] [--max-eval 1500] [--ref-buckets B] [--ref-king-buckets L] [--plain]
$T eval --net g4w.bin --val val.bin --buckets 8 --best-scale   # loss at the loss-optimal scale
```

Multiplies the output layer (weights and bias) by a factor; the FT is
untouched, so every eval scales by that factor. `--match` picks the factor
that brings the net to the reference net's scale on `--val`. Output weights
are clipped to +-127 (the engine's fast output path); the count is printed.

**Normaliser.** `slope` (default): `1/k`, `k` the least-squares slope through
0 of candidate eval on reference eval, over positions with |reference| <=
`--max-eval`. `mean-abs`: `mean|ref| / mean|cand|`. Slope is the better
choice: if the candidate is the reference at a different scale plus signal
the reference lacks (`c = k*r + u`, `u` uncorrelated with `r`, which is what
a better net looks like), the slope recovers `k` exactly, while mean|eval|
also counts `u` and shrinks the new signal along with the scale (unit test
`slope_ignores_signal_the_reference_lacks`). Filtering on |reference| (the
regressor) leaves the slope unbiased. On g3f vs g3 the two differ by ~1%
(0.916 vs 0.926 on fix/val17x2, 0.954 vs 0.953 on val.bin); the factor also
depends on the position set (0.916 on val17x2 vs 0.954 on val.bin), so name
the val set when quoting a factor.

**Rounding.** Output weights are small (g3f: mean |w| 11.5, mode 6), so plain
rounding of `f*w` is biased: `6 * 0.92 = 5.52` rounds back to 6. Plain x0.92
(`~/rustfish-data/rescale.py`, i.e. g3fs) gives g3f an effective **x0.934**
on val17x2. With `--val`, `rescale` therefore re-rounds the cheapest
parameters (those closest to the rounding midpoint, per unit of effect) the
other way until the least-squares factor of new vs old evals equals `f`
(exact: the output is linear in its parameters); no parameter moves more
than one unit from `f*w`. `--plain` disables this.

**Caveat for screening.** Val loss is very sensitive to scale (about 1e-4
per 1% on val17x2, the size of typical candidate gains), so a fixed-scale
comparison is only as good as the normaliser. `eval --best-scale` profiles
the scale out instead (loss at the loss-optimal factor).
