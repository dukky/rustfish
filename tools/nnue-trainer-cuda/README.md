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
2. `backward_ft`: the Metal trainer's `grad_fused`. Block = (32-column tile,
   work item = up to 2048 pairs of one king bucket), 1024 threads (lane =
   column, warp w owns features w + 32i of the bucket in registers). Per
   128 pairs: dacc staged in shared memory, per-feature bitmasks of the
   pairs containing it, each warp sums its features' rows in pair order.
   Writes per-item partial gradients.
3. `backward_rest` + `reduce_rest`: output-weight, FT-bias and output-bias
   gradients over 256-sample chunks, summed in chunk order.
4. `adam_ft`: per FT element, the bucket's item partials in item order;
   factoriser F (bucket order), W, merged rows; `--wd-active-only` flags
   from the CPU decode. `adam_flat` for the rest.

## Speed (GTX 1080 Ti, batch 16384, B 8)

kb8m H 512 `--factorise --mirror`: 2.5M pos/s end to end (kernels alone
3.9M: forward 1.6 ms, backward 2.2 ms, AdamW 0.4 ms); 9600X CPU trainer
~2.2M. `bench.cu` (standalone kernel bench) for the other shapes.
