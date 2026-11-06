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

Flags: `--data` (repeatable) `--val --out --hidden --buckets --epochs
--epoch-size --batch --lr --lr-end --wd --lambda --seed --mirror --init
--resume --checkpoint`. `--threads` is accepted and ignored. Writes
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
- `(768 -> H) x 2 -> B`, SCReLU, bucket `(n-2)*B/32`, target
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
3. **sort_entries**: per 128-position range, buckets the 256 rows' active
   features by feature, deterministically (a 256-bit row mask per feature,
   then rows emitted in order) and writes CSR offsets.
4. **grad_fused**: threadgroup = (32-column tile, 1/16 of the batch),
   1024 threads = 32 simdgroups x 32 lanes (lane = column). For each range
   it stages `dacc[256 rows][32]` in threadgroup memory, collecting the
   output-weight and FT-bias gradients from the same accumulator reads.
   Simdgroup k then sums the staged rows of features k, k+32, ... into
   registers. No atomics, uniform control flow, conflict-free loads. Each
   threadgroup writes its own partial-gradient slice.
5. **grad_ob** (output bias) and **adam**: sum the 16 partials in fixed
   order, AdamW + clip, all parameters dense.

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
