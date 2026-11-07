# nnue-trainer-metal

GPU (Apple Metal) version of `tools/nnue-trainer` `train`: same nets, same
data, same maths, same output, about 3x faster on the M3. Swift + Metal with
no dependencies. The MSL kernels are compiled at runtime (Command Line
Tools ships no offline `metal` compiler).

## Build

```bash
tools/nnue-trainer-metal/build.sh      # swiftc -O ... -> tools/nnue-trainer-metal/nnue-trainer-metal
```

## Usage

`train` takes the same flags as the CPU trainer and writes the same log
lines, so existing tooling (e.g. the `epoch ... val V (quantized Q) ... pos/s`
regex in `train-sweep.yml`) parses it unchanged:

```bash
M=tools/nnue-trainer-metal/nnue-trainer-metal
$M train --data pool.bin --val val.bin --out net.bin --hidden 512 --buckets 8 --epochs 20
$M train ... --init old.bin           # warm start from an exported net
$M train ... --resume net.ckpt        # resume (same checkpoint format as the CPU trainer, both ways)
$M eval --net net.bin --val val.bin --buckets 8   # GPU twin of `nnue-trainer eval`
$M gradcheck --data recs.bin --hidden 256 --buckets 8 --batch 2048
$M bench --data recs.bin --hidden 1024 --buckets 8   # per-kernel GPU time for one batch
```

Flags: `--data` (repeatable) `--val --out --hidden --buckets --king-buckets --epochs
--epoch-size --batch --lr --lr-end --wd --lambda --seed --mirror --init
--resume --checkpoint --factorise --wd-active-only --no-wd-active-only`.
`--threads` is accepted and ignored. Writes
`<out>.eN.bin` per epoch, `<out>.ckpt`, and `<out>`. At exit it prints one
line on stderr: time spent waiting for the loader, waiting for the GPU,
and GPU busy time.

## What is identical to the CPU trainer

Not just the formulas: the same random streams, so with the same seed both
trainers see the same batches in the same order.

- 32-byte records, feature index `side*384 + kind*64 + sq` (sq ^ 56 for a
  Black perspective), perspective 0 = side to move, `--mirror` file flip (sq ^ 7).
- The same `Feed` (files in order, 1M-record chunks, the same Fisher-Yates
  with the same splitmix RNG and seed `seed ^ 0xF00D ^ seen`), the same
  per-batch `seed_b` and per-256-chunk mirror RNG.
- The same init (`init_params` RNG sequence), `--init` clamping, and
  checkpoint layout.
- `(KB*768 -> H) x 2 -> B` (KB = king buckets, see below), SCReLU, bucket `(n-2)*B/32`, target
  `lambda*wdl + (1-lambda)*sigmoid(cp/400)`, MSE on sigmoid(out).
- AdamW (b1 .9, b2 .999, eps 1e-8, the same `powi` bias correction), cosine
  LR on positions seen, clip +-1.98, batch 16384.
- Quantised export (QA 255, QB 64, round half away from zero) and the
  quantised validation pass.

The only difference is float summation order. GPU training is deterministic:
a rerun with the same flags writes a bit-identical net.

## Validation

**Gradients** (`./gradcheck.sh <records>`): GPU gradient of a batch compared
with a CPU reference (a port of `train_sample`) over every parameter. Also
checks double-precision finite differences on FT weights of both
perspectives, FT bias, output weights of both halves and output bias, and
one AdamW step. Shapes H=16/B=1 (mirror), 32/8, 256/8, 1024/1 (mirror),
including batches that are not a multiple of 128. Max relative gradient
error is 1e-7 to 8e-7 and every finite-difference check agrees to about 6
digits.

**Parity runs** (6M records from `pool1.bin`, `--val` = first 100k of
`val.bin`, seed 5, `--mirror`, 2 epochs, default LR schedule; CPU trainer at
`--threads 3`):

| shape | trainer | start val | epoch 1 train / val | epoch 2 train / val (quantised) |
|---|---|---|---|---|
| 256x1 | CPU | 0.081055 | 0.044154 / 0.035888 | 0.032540 / 0.034714 (0.034794) |
| 256x1 | GPU | 0.081055 | 0.044155 / 0.035889 | 0.032541 / 0.034717 (0.034831) |
| 128x8 | CPU | 0.081256 | 0.048110 / 0.036801 | 0.033098 / 0.035280 (0.035347) |
| 128x8 | GPU | 0.081256 | 0.048110 / 0.036803 | 0.033098 / 0.035280 (0.035338) |

`nnue-trainer eval` (the CPU evaluator) on the GPU's exported nets gives
exactly the GPU's reported quantised losses (0.034831, 0.035338), and the
GPU `eval` on the CPU nets gives exactly the CPU's (0.034794, 0.035347). The
quantised-loss differences between the trainers (up to 0.1%) are rounding
noise: float val agrees to 3e-6. The 256x1 GPU net loads in the engine
(`net/nnue.bin`, `bench` runs, 598k nodes).

## Speed

Apple M3 (10-core GPU), batch 16384. Measured with the CPU training job
paused. Numbers measured with it running were within noise of these, so
the GPU trainer is not bandwidth-starved by a concurrent CPU job.

| H | B | GPU kernel step | GPU end-to-end | CPU trainer v2 (8 threads) | speed-up |
|---|---|---|---|---|---|
| 256 | 1 | 4.7 ms (3.5M pos/s) | 2.9M pos/s | 1.1M | 2.7x |
| 256 | 8 | 4.8 ms (3.4M) | 3.05M | ~1.1M | ~2.8x |
| 512 | 1 | 10.5 ms (1.57M) | 1.57M | 0.48M | 3.3x |
| 512 | 8 | 12.0 ms (1.36M) | 1.52M | ~0.48M | ~3.2x |
| 1024 | 1 | 21.9 ms (0.75M) | 0.79M | (~0.24M extrapolated) | ~3.3x |
| 1024 | 8 | 21.7 ms (0.75M) | 0.78M | | |

End-to-end = 8M-position epoch with no `--val`. Validation of 1M positions
adds about 0.1 s per epoch. Contended kernel benches of the same build gave
3.7M / 1.8M / 0.87M at H = 256 / 512 / 1024. Run-to-run variance on this
fanless machine is about ±15%.

### Where the time goes (H = 1024, per batch)

- `grad_fused` ~70% (15 ms). Backward of the feature transformer: for every
  active (feature, row) entry, add a 2H-wide dacc row into the feature's
  gradient row, i.e. ~850k entries x H adds per batch, from threadgroup
  memory. Cost is linear in H.
- `forward` ~23% (5 ms): gather-sum of ~52 FT rows per position.
- `adam` 4%, `sort_entries` + `decode` <1%. The CPU loader (feed, shuffle,
  mirror flags) runs in a thread 16 batches ahead and never stalls the GPU.

## Architecture

Per batch, one command buffer:

1. **decode**: raw records -> 2x32 feature indices, bucket and target per
   position. The mirror flag is set by the CPU loader in the record's
   padding byte so it follows the CPU trainer's RNG stream.
2. **forward_train**: one threadgroup per position, float4 columns;
   stores the pre-activation accumulators, the output gradient
   `2 err p (1-p)` and the loss.
3. **plan_rows** (one threadgroup) orders the rows (position x
   perspective) by king bucket, stably, and cuts them into ranges of 256
   rows and work items of up to 1/16 of the batch (identity order with one
   king bucket). **sort_entries**: per range, buckets the 256 rows' active
   features by feature, deterministically (a 256-bit row mask per feature,
   then rows emitted in order) and writes CSR offsets.
4. **grad_fused**: threadgroup = (32-column tile, work item),
   1024 threads = 32 simdgroups x 32 lanes (lane = column). For each range
   it stages `dacc[256 rows][32]` in threadgroup memory, collecting the
   output-weight and FT-bias gradients from the same accumulator reads.
   Simdgroup k then sums the staged rows of features k, k+32, ... into
   registers. No atomics, uniform control flow, conflict-free loads. Each
   threadgroup writes its own partial-gradient slice.
5. **grad_ob** (output bias) and **adam**: sum the partials in fixed
   order (an FT row: the items of its king bucket), AdamW + clip, all
   parameters dense.

Validation uses the same forward kernel on a GPU-resident decoded val set.
For the quantised loss, the CPU quantises and dequantises the weights and
uploads them to a second buffer.

Approaches that lost (kept here so nobody retries them blind):

- Per-entry threadgroup float atomics: these don't exist in MSL, and a CAS
  loop ran at 1.2M pos/s at H = 256.
- Lane-per-chunk segmented reduction over the sorted list with CAS or owner
  flushes: SIMT divergence made almost every iteration pay a flush.
- Per-range device read-modify-write of the gradient instead of registers:
  about 30% slower at H = 1024.

## King buckets

`--king-buckets <none|m1|kb4|kb4m|kb8m>` (train, eval, gradcheck, bench)
works as in the CPU trainer (same tables, same feature index, same
`--init` expansion from a one-bucket net, same checkpoint header). On the
GPU:

- `decode` finds both kings first and writes `bucket*768 + ...` indices
  (the layout table travels in `Cfg`). `meta` also carries each row's king
  bucket.
- `plan_rows` sorts the batch's rows by king bucket (stable, so with one
  bucket nothing moves). Every row's features are in its king's 768-row
  block, so each bucket's rows are a dense 768-feature problem: ranges of
  256 rows of one bucket get the same CSR as before (24 KB of masks), and a
  work item (a run of up to `per` ranges of one bucket, `per` = the
  one-bucket super-range length) sums into the same register layout. Every
  row is staged exactly once, so the work is that of one bucket plus at
  most one partial range per bucket. Items write 768 FT rows + FT bias +
  output weights; `adam` sums an FT row over its bucket's items, the rest
  over all items, in fixed order: training stays deterministic.
- KB = 1 uses `grad_fused1`, the same code with the row lookup and the
  per-side output-gradient registers compiled out; its summation order is
  the original one, so one-bucket nets (`none`, `m1`) are bit-identical to
  the trainer before king buckets.

Validation: `gradcheck.sh` covers kb8m (H 32/B 8, H 256/B 8, H 64/B 8
with `--mirror` and a batch that is not a multiple of 128), kb4m with
`--mirror`, kb4, m1, all PASS with errors the same size as one-bucket nets.
Parity on 4M records of `pool2.bin`, 1 epoch, `--val` = first 100k of
`val.bin`:

| config | trainer | start val | train / val (quantised) |
|---|---|---|---|
| H 128, B 1, kb8m, --mirror, seed 5 | CPU (2 threads) | 0.081495 | 0.061364 / 0.057181 (0.057218) |
| | GPU | 0.081495 | 0.061364 / 0.057181 (0.057219) |
| H 64, B 8, kb4, 2M records, seed 3 | CPU | 0.081524 | 0.074645 / 0.073476 (0.073482) |
| | GPU | 0.081524 | 0.074645 / 0.073476 (0.073482) |

Each evaluator gives the other trainer's nets the same quantised loss as
that trainer reported. Warm start from `net/nnue.bin` into kb4 at lr ~0
gives quantised val 0.027584, the one-bucket net's exact loss.

Bucket-sorted backward (plan_rows): 2M records of `pool3.bin`, H 128, B 8,
`--mirror`, seed 5, 1 epoch, `--val` = first 100k of `val.bin`: kb8m CPU,
GPU before and GPU after all give start val 0.081269, train 0.073039, val
0.070774 (quantised 0.070792); the CPU evaluator gives the new GPU net
0.070792. `none` and `m1` nets and checkpoints are byte-identical to the
previous build. End-to-end kb8m went 0.45M -> 0.96M pos/s at H 128.

Cost (`bench`, B 8, kernel step per 16384 batch). Uncontended moment of a
shared GPU:

| H | KB | before | after (grad / adam) |
|---|---|---|---|
| 256 | 1 | 5.25 ms | 4.81 ms (3.52 / 0.21) |
| 256 | 4 (kb4m) | 13.32 ms | 6.44 ms (4.86 / 0.41) |
| 256 | 8 (kb8m) | 26.70 ms | 6.71 ms (4.84 / 0.61) |
| 512 | 1 | 10.92 ms | 10.19 ms (8.28 / 0.42) |
| 512 | 4 (kb4m) | 29.17 ms | 12.86 ms (10.11 / 0.77) |
| 512 | 8 (kb8m) | 52.43 ms | 14.37 ms (10.56 / 1.22) |

kb8m costs ~1.3-1.4x one bucket (contended runs agree): the rest is the
row-id indirection and two output-gradient register sets in `grad_fused`
(~10%), dense AdamW over 8x the FT parameters, and a forward that gathers
from an 8x larger table. Tried without gain: equal-sized items, and
prefetching row id/meta/gradient per lane with simd shuffles.

## Factoriser and active-only weight decay

`--factorise` and `--wd-active-only` / `--no-wd-active-only` (train,
gradcheck; bench takes the first two) work as in the CPU trainer (see its
README: same init, same `--init` mapping, same checkpoint layout and header
bit, merged rows in every exported net). On the GPU:

- `Trainer.t` is the training vector (net params, then F); `Trainer.p` is
  the merged net that forward, grad, validation and export read. Without
  the factoriser they are the same buffer and nothing else changes.
- `adam` runs over `t`; F's gradient is the sum over every work item of
  its 768 local feature rows (each item belongs to one king bucket), in item
  order. `merge_factor` then writes `p = clamp(W + F)` (other params copied).
- `row_active` marks each FT row that has an entry in one of its bucket's
  ranges; `adam` skips the decay of unmarked bucketed rows.

Validation (`pool3f.bin` head, `--val` = first 100k of `val.bin`):

- `gradcheck.sh`: the factoriser cases (kb4, kb4m --mirror, kb8m at H 32 /
  64 --mirror / 256 --mirror, and kb8m with `--no-wd-active-only`) check
  W and F gradients against the CPU reference (rel. error 1e-7 to 4e-7),
  finite differences that perturb W or F through the merge, an AdamW step
  in which every FT row absent from the batch is exact (unchanged with
  active-only decay, plainly decayed without), and the merged net after the
  step. All 16 cases PASS.
- Parity, 2M records: H 64, B 8, kb8m, --factorise, --mirror, --wd 0.01,
  seed 5, 2 x 1M: CPU and GPU both print start 0.081513, epoch 1 0.077781 /
  0.069376 (q 0.069442), epoch 2 0.067674 / 0.065755 (q 0.065794 CPU,
  0.065793 GPU); H 128, B 1, kb4, --factorise, 1 x 1M: both 0.074950 /
  0.068991 (q 0.069007). Each evaluator gives the other trainer's nets that
  trainer's quantised loss, and checkpoints resume across trainers to the
  same epoch-3 numbers.
- Warm start at lr 0: `--factorise --init g3f.bin` (KB 1, H 512, B 8) into
  kb4 gives g3f's quantised loss (0.026746) and into kb8m the loss of g3f read
  as m1 (0.028761, see the CPU README for why a mirrored layout cannot equal a
  `none` net); an m1 net into kb8m gives exactly its own loss (0.078022). The
  CPU and GPU export byte-identical nets.
- Without `--factorise` (and, for KB > 1, with `--no-wd-active-only`) nets
  and checkpoints are byte-identical to the previous build (none, m1, kb8m).

Cost: H 512, B 8, kb8m kernel step 14.0-14.7 ms plain vs 15.2-16.5 ms
factorised (adam + merge 1.3 -> 2.1 ms), about 5%. `row_active` is within
noise. End-to-end ~1.0M pos/s in both cases.

## Known gaps

- Only `train` and `eval` are implemented. Keep using the CPU tool for
  `convert` and `shuffle`.
- `--hidden` must be a multiple of 4, and `--buckets` at most 16.
- Results are deterministic per build but not bit-identical to the CPU
  trainer (float summation order), and not across changes to
  `NSG`/`RS`/`Rf`.
- The rest of the speed is in `grad_fused`. Ideas not tried: 64-column
  tiles with half-precision staging (would change numerics), or splitting
  forward and backward over two batches to overlap.
