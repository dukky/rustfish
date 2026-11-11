# nnue-trainer-cuda

CUDA backend for `tools/nnue-trainer` `train`: it *is* the CPU trainer's
command line (`nnue_trainer::cli` with a GPU `Backend`), so the flags, log
lines, data feed and RNG streams (same batches in the same order), init,
`--init`/`--resume`, validation, quantised export, checkpoints (both ways
with the CPU trainer) and `--keep-best` are the CPU trainer's code.
Forward, backward and AdamW run on the GPU; decode and the king-bucket
sort on the CPU pool (`--threads`). Rust std only on the host (driver API
through a small FFI); the kernels (`src/kernels.cu`) need nvcc to build.

```bash
# toolkit without root: the CUDA 12.x runfile with --toolkit --toolkitpath=$HOME/cuda-12.8
# (CUDA 13 dropped Pascal / the 1080 Ti); build.rs looks for $NVCC, $CUDA_HOME, ~/cuda-12.8, PATH
RUSTFLAGS="-C target-cpu=native" cargo build --release --manifest-path tools/nnue-trainer-cuda/Cargo.toml
G=tools/nnue-trainer-cuda/target/release/nnue-trainer-cuda
$G train --data pool.bin --val val.bin --out net.bin --hidden 512 --buckets 8 \
   --king-buckets kb8m --factorise --mirror --epochs 20
```

`eval`, `shuffle`, `convert`, ... are the CPU trainer's (they run on the
CPU). `--exact` and `--sub-batch` are ignored.

`--half` (GPU only, off by default): the forward reads an fp16 copy of the
merged FT rows (master weights, gradients and AdamW stay f32), which halves
its ~2 GB of row reads per batch: +14-18% (table below). fp16 keeps 11
mantissa bits, finer than the engine's 1/255 FT quantisation; kb8m H 512,
3 x 2M positions: val 0.037563 (quantized 0.037727) vs f32 0.037562
(0.037754). Not bit-identical to f32 runs, so A/B it before relying on it
for a generation.

## Deterministic, not bit-identical to the CPU

No float atomics: every sum has a fixed order, so reruns (any `--threads`)
write byte-identical nets and checkpoints. The order differs from the CPU
kernels', so nets agree with the CPU trainer's closely, not bit for bit:
kb8m H 512 `--factorise --mirror`, 3 x 2M positions, val (last 200k of
s45): CPU 0.049903 / 0.039111 / 0.037563, GPU 0.049905 / 0.039107 /
0.037562 (train loss equal to 6 digits).

## Kernels (per batch)

1. `forward`: a block per sample (bucket-pair order), H/2 threads, float4
   columns; accumulators, output gradient, loss.
2. `plan_masks` (once per batch, for every column tile): per range of 128
   pairs, per feature of its bucket, the bitmask of the pairs containing it.
   `backward_ft`: the Metal trainer's `grad_fused`. Block = (32-column tile,
   work item = up to 2048 pairs of one king bucket), 1024 threads (lane =
   column, warp w owns features w + 32i of the bucket in registers). Per
   128 pairs: dacc staged in shared memory (double-buffered, one barrier
   per range), each warp sums its features' rows in pair order.
   Writes per-item partial gradients.
3. `backward_rest` + `reduce_rest`: output-weight, FT-bias and output-bias
   gradients over 256-sample chunks, summed in chunk order.
4. `adam_ft`: per FT element, the bucket's item partials in item order;
   factoriser F (bucket order), W, merged rows; `--wd-active-only` flags
   from the CPU decode. `adam_flat` for the rest.

## Pipeline

Batch k is decoded and bucket-sorted on the CPU (pinned, double-buffered
inputs, async copies on one stream) while the GPU runs batch k-1; `step`
returns the previous batch's loss and `flush` (epoch ends) the last one,
so epoch losses are unchanged. 2 CPU threads are enough at kb8m H 512
(`--threads 2`: same speed as 12), which leaves the rest of the CPU free
for a CPU trainer run next to it.

## Speed (GTX 1080 Ti, batch 16384, B 8, `--mirror`, steady state)

| net | 1080 Ti | 1080 Ti `--half` | 9600X CPU trainer |
|---|---|---|---|
| H 256 plain | 7.0M | 8.0M | 7.7M |
| H 512 plain | 4.3M | 4.9M | 4.1M |
| H 512 kb8m `--factorise` | 3.6M | 4.1M | 2.2M |
| H 1024 kb8m `--factorise` | 1.9M | 2.25M | 0.73M |

Per batch at kb8m H 512 (in the trainer): forward 1.75 ms (~2 GB of FT row
reads, about the 1080 Ti's L2 bandwidth), FT backward ~1.9 ms, output-layer
gradients 0.34 ms, AdamW 0.47 ms. Uploads run on a second stream.

Kernels alone (`bench.cu`, kb8m H 512): forward 1.6 ms, FT backward 2.2
ms, AdamW 0.4 ms per batch (3.9M pos/s). Checked: CPU checkpoint resumed on
the GPU and back, `--init` warm start, `--keep-best`, `--wd`.
