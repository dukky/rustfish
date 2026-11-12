# nnue-trainer (v2)

Std-only trainer for the `(768 -> H) x 2 -> B` SCReLU nets in `src/nnue.rs`
(`B` = output buckets by piece count; the engine currently runs H = 256, B = 1).
See the module docs in `src/main.rs` for the maths and quantisation.

```bash
cargo build --release
T=target/release/nnue-trainer
gunzip -c datagen-s9-n8000.gz | $T convert - s9.bin   # text -> 32-byte records
zstd -dc t80.binpack.zst | $T binpack - t80.bin       # Stockfish binpack (Lc0 T80) -> records, docs/t80-data.md
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

The batch kernels also run `--factorise` (forward reads the merged rows
`clamp(W + F)` from the tile-major copy; AdamW updates `F` from every
bucket's gradient, then each bucket's `W`, then re-merges), `--wd-active-only`
(rows flagged during decode) and `--keep-best`; with `--exact` these runs are
bit-identical too (nets, epoch nets and checkpoints of kb8m `--factorise
--init g4.bin --keep-best --wd 0.01`, kb4 `--factorise` from scratch and kb4m
plain against the per-sample trainer at `--threads 1`). M3, 8 threads, kb8m
`--factorise` H 512 B 8: 0.62M positions/s (per-sample trainer 0.16M).

H = 512, B = 8, batch 16384, `--mirror`, positions/s (GitHub 4-vCPU runners,
`trainer-bench.yml`; old = the previous trainer at 4 threads):

| CPU | old | new | new 1 thread | new `--exact` |
|---|---|---|---|---|
| AMD EPYC 9V45 (Zen 5) | 0.71-0.82M | 1.69M | 0.65M | 1.43M |
| AMD EPYC 9V74 (Zen 4) | 0.46M | 0.82M | 0.39M | 0.70M |
| Intel Xeon 8573C | 0.43M | 0.90M | 0.40M | 0.77M |
| AMD EPYC 7763 (Zen 3, AVX2) | 0.41M | 0.69M | | |

Ryzen 5 9600X (Zen 5, 6 cores / 12 threads, DDR5-6000, WSL2, native
build, 12 threads), H 512 B 8 `--mirror`: plain 3.9M -> 4.1M, kb4 2.9M ->
3.3M, kb8m `--factorise` 1.8M -> 2.1-2.2M (trainer-kb-speed). King buckets
are bound by AdamW, which updates the whole 8x larger FT every batch:
it writes 12 B per FT param (params, m, v), and a single-CCD Zen 5 writes
only ~31 GB/s to DRAM (reads ~57 GB/s; measured), ~30% of a kb8m batch.
The rest of the gap to plain is cache footprint. What helps, all exact:
AdamW streams p/m/v with non-temporal stores (keeps the next batch's
weights and gradient in AMD's victim-cache L3), runs per column tile
inside the last backward phase on a tile-major copy of the FT state
(`step`/`export`), and skips stores for lines it does not change (rows of
impossible features); the forward reads bucket-sorted pairs.
`--half` (off by default; also in the CUDA trainer): the forward reads an
fp16 copy of the FT rows (F16C / AVX-512 `vcvtph2ps`, the same format and
round-to-nearest-even as the GPU; master weights, gradients and AdamW stay
f32). It halves the copy's cache footprint: kb8m H 512 +12% (1.54M ->
1.73M, measured under load), plain nets ~+1% (their copy already fits).
kb8m H 512, 3 x 2M positions: val 0.037560 (quantized 0.037757) vs f32
0.037563 (0.037791). Deterministic across ISAs and threads (unit tests check
the software conversion against the hardware one); not bit-identical to
f32 runs.

The batch's units run as one dependency-ordered stream (no barriers;
+2-4%, 3 sub-batches in flight: 4 was 5% slower).

Approaches that lost on the 9600X (kb8m H 512, exact variants):
- Backward over bucket-sorted pairs (dacc staged, then scattered one
  bucket's rows at a time): -10%; the gradient tile is fine in L2.
- CSR backward as in the Metal trainer (entries grouped by feature, each
  gradient line loaded and stored once): -10% at sub-batch 512, -12% at
  256, -27% at 128. Keeping the sum order exact means one dependent add
  chain per feature, and the staged dacc reloads cost as much as the
  scattered read-modify-writes they replace.
- Sub-batch 256 / 1024 / 2048 instead of 512: within noise or slower.

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

### Factoriser (`--factorise`, KB > 1 only)

Training also keeps a shared 768 x H block `F`; every bucket's effective
row is `clamp(W[kb][f] + F[f], +-1.98)` and `F` gets the sum of all
buckets' gradients for its feature, so rarely hit buckets start from (and
keep learning) what every bucket shares. Every exported net (`<out>.eN.bin`
and `<out>`) holds the merged rows: the file format and the engine are
unchanged, and `eval` needs no flag. Checkpoints store `W` and `F` (header
word 1 bit 40 = factorised; `--resume` needs the same flag).

- From scratch: `W = 0`, `F` = the one-bucket init (the same RNG draws as a
  KB=1 net with the same seed), so step 0 equals that KB=1 init.
- `--init` from a one-bucket net: `F` = its rows, `W = 0`. From a net that
  already has all KB buckets: `W` = the net, `F = 0`.

Warm starts are exact for what the layout can express: an unmirrored
layout (kb4) from a `none` net reproduces it exactly, and a mirrored
layout (kb4m, kb8m) from an `m1` net reproduces it exactly. A mirrored
layout cannot reproduce a `none` net: every mirrored bucket serves both a
king on file x (unflipped) and on file 7-x (flipped), so one row would have
to equal the source rows of square s and s ^ 7 at once. It starts as that
net read with mirroring (`eval --king-buckets m1` gives its loss).

### Warm start from a plain net (`--init`), and `--keep-best`

`--init plain.bin` (a one-bucket net) starts a king-bucket run exactly at
that net in both modes; they differ in what is trained afterwards:

| | start | trained |
|---|---|---|
| no `--factorise` | its 768 rows copied into every bucket's `W` | each bucket's copy on its own |
| `--factorise` | `F` = its 768 rows, every bucket `W = 0` | `F` (shared, gets every bucket's gradient) + `W` |

The FT bias and the output layer are the net's in both. `--init` always
starts AdamW fresh (`m = v = 0`, step 0; only `--resume` restores them);
`--mirror` is a data augmentation and validation never mirrors, so neither
changes the start. The trainer prints `start val loss F (quantized Q)`:
`Q` is the plain net's `eval` loss, and the epoch-0 export of a kb4 run from
g4 is g4's bytes in every bucket. `F` can differ from it in the 6th digit:
weights the source net holds at the quantised limit (|FT| = 505/255, |out|
= 127/64, g4: 408 + 19 of them) sit just above the +-1.98 training clip and
start clipped, which rounds back to the same i16. A mirrored layout
(kb4m, kb8m) is exact only from an `m1` net (see above; the trainer prints a
note); from a `none` net it starts at `eval --king-buckets m1` of that net.

Measured (first 100k of `val.bin`): g4 `eval` 0.026673, kb4 from g4 start
0.026672 (quantized 0.026673) with and without `--factorise`, CPU and GPU;
kb8m from g4 0.028718 = g4 read as m1. H 64 parity (plain H 64 net, kb4 and
kb8m `--factorise`, 3 x 65536, lr 2e-4 -> 1e-5): CPU and GPU print the
same start and epoch losses to 6 digits; an m1 net into kb8m starts at
exactly its own loss (0.077995).

`--keep-best` (needs `--val`): `<out>` is the epoch with the lowest
quantised val loss, the start net (epoch 0) included, instead of the last
epoch; every `<out>.eN.bin` and the checkpoint are still written. The log
ends with `keep-best: epoch N (quantized val Q) is the output`. With
`--resume`, epochs before the resume point are not considered.

### Active-only weight decay (`--wd-active-only`, default on for KB > 1)

AdamW's decay on the bucketed FT rows `W` is applied only to rows whose
feature occurs in the batch, so buckets that are rarely hit are not shrunk
toward zero between visits. `F`, the FT bias and the output layer decay as
before; `--no-wd-active-only` restores plain decay. It only matters with
`--wd > 0` (the default is 0). KB=1 nets keep plain decay and train
bit-for-bit as before.

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
