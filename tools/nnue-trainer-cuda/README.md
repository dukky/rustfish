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

## GPU + CPU on one net (`NNUE_PAIR`, experiment)

`NNUE_PAIR=K,cpu_threads,gpu_share nnue-trainer-cuda train ...` trains one
net on both: the GPU and a CPU worker (its own pool of `cpu_threads`, the
CPU batch kernels) each keep a copy of the net and the Adam moments; in
every round of K batches a fixed `gpu_share` of them (by batch index) go to
the GPU, the rest to the CPU; after each round and at every epoch end both
wait and average params and moments, and go on from the average (local SGD).
Deterministic (reruns byte-identical). Give the GPU decode `--threads 2`
and set `gpu_share` near the speed ratio (kb8m H 512 `--half` ~0.63).

Result (2026-11-12, plain H 512 B 8 from scratch, pool7, 3 epochs, `--half`
`--mirror`, val.bin): **it loses, also at equal wall time.** GPU only 3 x 300M:
0.027242; GPU only 3 x 210M (the pair runs' wall time): 0.027386; pair
3 x 300M with averaging, K = 16 / 64 / 256: 0.027490 / 0.027474 / 0.027473
(6.0-6.6M pos/s vs 4.55M); with `sum` merging (`NNUE_PAIR=...,sum`: add both
devices' changes since the last merge, which keeps the update count) K = 64
/ 256: 0.027596 / 0.027499, unstable early (epoch 1 val 0.032-0.035: the two
sets of Adam steps overshoot when added). Averaging two models is ~one step
at twice the batch, so the pair makes about half the updates per position;
the extra 43% throughput does not make up for it. Kept for reference; use the
two devices for separate runs instead.

## Deterministic, not bit-identical to the CPU

No float atomics: every sum has a fixed order, so reruns (any `--threads`)
write byte-identical nets and checkpoints. The order differs from the CPU
kernels', so nets agree with the CPU trainer's closely, not bit for bit:
kb8m H 512 `--factorise --mirror`, 3 x 2M positions, val (last 200k of
s45): CPU 0.049903 / 0.039111 / 0.037563, GPU 0.049905 / 0.039107 /
0.037562 (train loss equal to 6 digits).

## Head (`--head`): UNTESTED ON CUDA

Written on the laptop without a GPU or nvcc (2026-11-13, branch
`head-v2`): the host code type-checks (`NNUE_CUDA_NO_NVCC=1 cargo check`)
and `src/kernels.cu` passes a clang `-x cuda -fsyntax-only` device pass
with stub headers, but **no kernel has run**. It mirrors the CPU trainer's
math (`head_forward`, `head_backward`, `head_small_grads`, the `e` term of
`train_sample`; docs/nnue-deeper-head.md):

- `forward` + `head_forward` (since `perf-cuda-pairwise`, see "Head
  speed" below; before, all of this ran at the end of `forward` and in a
  `compute_dacc` kernel): the 16 z1 sums (a warp per row: lane partials,
  then the shuffle tree); warp 0 adds b1, does L2 (lane = z2 row), L3
  (shuffle tree), the loss and gradient, the head backward (dz2, dz1), and
  writes the sample's scratch `hs[sample][96]` = z1 | z2 | dz1 | dz2 (the
  CPU `HS` layout); then dacc for every lane.
- `backward_ft` staging and `backward_rest` (FT bias): dacc = `(g w + e)
  2 a`, `e = sum_q dz1[q] W1[q][i]` (q ascending), as `train_sample`.
- New: `backward_head_w1` (thread = lane of 2H, grid.y = chunk x bucket;
  walks its chunk's samples of that bucket in order, gW1[q] += dz1[q] a^2),
  `backward_head_small` (thread = one of the 593 b1/W2/b2/W3/b3 params per
  bucket), `reduce_head` (chunk partials in chunk order -> `gr`). AdamW of
  the head is `adam_flat` (the head is part of `rest`). Extra memory at
  batch 16384: hs 6 MB, partials 64 chunks x 8 x head_len floats (34 MB at
  H 512, 67 MB at H 1024).
- Without `--head` the kernels take the old code paths (`hd = 0`): check
  that plain runs are byte-identical to the previous build (step 1 below).

### Validating on the GPU (desktop checklist)

```bash
cd ~/dev/rustfish && git fetch && git checkout head-v2
RUSTFLAGS="-C target-cpu=native" cargo build --release --manifest-path tools/nnue-trainer/Cargo.toml
RUSTFLAGS="-C target-cpu=native" cargo build --release --manifest-path tools/nnue-trainer-cuda/Cargo.toml
C=tools/nnue-trainer/target/release/nnue-trainer
G=tools/nnue-trainer-cuda/target/release/nnue-trainer-cuda
D=~/rustfish-data; mkdir -p $D/headcheck
head -c $((32 * 6000000)) $D/pool8.bin > $D/headcheck/slice.bin     # any shuffled 32-byte pool
V="--val $D/val.bin"
S="--data $D/headcheck/slice.bin --hidden 512 --buckets 8 --king-buckets kb8m --factorise --mirror --epochs 3 --epoch-size 2000000 --seed 1"
# 1. no-head regression: same net bytes as main's nnue-trainer-cuda
git worktree add /tmp/rf-main origin/main
RUSTFLAGS="-C target-cpu=native" cargo build --release --manifest-path /tmp/rf-main/tools/nnue-trainer-cuda/Cargo.toml
G_MAIN=/tmp/rf-main/tools/nnue-trainer-cuda/target/release/nnue-trainer-cuda
$G train $S $V --out $D/headcheck/plain-new.bin --threads 2 && $G_MAIN train $S $V --out $D/headcheck/plain-old.bin --threads 2
cmp $D/headcheck/plain-new.bin $D/headcheck/plain-old.bin
# 2. CPU vs GPU with the head: epoch train / val losses within ~1e-5 (plain nets agree to ~1e-6)
$C train $S $V --head --out $D/headcheck/cpu.bin --threads 12 | tee $D/headcheck/cpu.log
$G train $S $V --head --out $D/headcheck/gpu.bin --threads 2 | tee $D/headcheck/gpu.log
grep epoch $D/headcheck/cpu.log $D/headcheck/gpu.log
# 3. deterministic: a rerun writes the same bytes
$G train $S $V --head --out $D/headcheck/gpu2.bin --threads 2 && cmp $D/headcheck/gpu.bin $D/headcheck/gpu2.bin
# 4. warm start from a plain net is exact (start val loss = the plain net's, both trainers)
$G train $S $V --head --init $D/headcheck/plain-new.bin --epochs 1 --max-batches 1 --out $D/headcheck/warm.bin --threads 2 | grep "start val"
$C eval --net $D/headcheck/plain-new.bin --buckets 8 --king-buckets kb8m --val $D/val.bin
# 5. checkpoints cross over: CPU checkpoint resumed on the GPU continues the CPU loss curve
$C train $S $V --head --epochs 1 --out $D/headcheck/x.bin --threads 12
$G train $S $V --head --resume $D/headcheck/x.ckpt --out $D/headcheck/y.bin --threads 2
# 6. engine agreement on the GPU net (builds a HEAD engine in the workdir)
python3 scripts/head_check.py --net $D/headcheck/gpu.bin --hidden 512 --king-buckets kb8m \
    --records $D/val.bin --trainer $C --workdir $D/headcheck/engine
```

If 2 disagrees: compare one batch (`--epochs 1 --epoch-size 16384`,
`--lr 1e-3`) with `eval` on both nets; a wrong head gradient shows as a
diverging train loss within the first epoch, a wrong forward as a
different start val loss (the start loss is computed by the shared CPU
code, so it must agree exactly; the first epoch's train loss comes from
the GPU forward).

## Kernels (per batch)

1. `forward`: a block per sample (bucket-pair order), H/2 threads, float4
   columns; accumulators, output gradient, loss.
2. `plan_csr` (once per batch, for every column tile): per range of 256
   pairs, its (pair, feature) incidences sorted by feature (pairs ascending
   within a feature): per-feature start/end and a byte list of pair offsets.
   `backward_ft`: block = (32-column tile, work item = up to 2048 pairs of
   one king bucket), 1024 threads (lane = column, warp w owns features
   w + 32i of the bucket in registers). Per 256 pairs: dacc and the range's
   lists staged in shared memory, each warp walks its features' lists and
   sums the staged rows in pair order. Writes per-item partial gradients.
3. `backward_rest` + `reduce_rest`: output-weight, FT-bias and output-bias
   gradients over 256-sample chunks, summed in chunk order.
4. `adam_ft`: per FT element, the bucket's item partials in item order;
   factoriser F (bucket order), W, merged rows; `--wd-active-only` flags
   from the CPU decode. `adam_flat` for the rest. `--lazy-adam` (CPU
   README, "Lazy Adam"): threads of rows without the flag return before
   reading partials or moments (F rows only if no bucket row occurred).
   Smoke (2026-11-20, GPU shared with a training run, so ~2.7x slow):
   kb8m H 1024 `--half` `--factorise`, 98 batches x 2 runs, `adam_ft`
   dense 2.28 / 2.23 ms, lazy 2.08 / 2.22 ms of ~17.9 ms: within noise,
   under 1% of the batch (84% of W rows and 92% of F rows occur in a
   16384 batch, and a skipped W row still reads p for its merged row).
   Default-off runs write the same bytes as main; lazy runs are
   deterministic and match the CPU trainer's lazy runs to 6 digits.

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

`NNUE_CUDA_PROF=1` records timing events between the kernels and prints
per-stage ms/batch (and the host's prep / wait time) at every epoch end.

Per batch at kb8m H 1024 `--half` (2026-11-13, 180 W cap): forward 2.1 ms,
`backward_ft` 2.2 ms (3.8 ms with the per-range bitmasks it replaced),
output-layer gradients 0.6 ms, AdamW 0.8 ms; host prep (decode, sort) 1.1
ms runs under the GPU's batch, so the feed is not the limit.

`backward_ft` variants measured (kb8m H 1024 `--half`, ms/batch; all
byte-identical): bitmasks R 128 3.82; incidence lists R 64 with 64-column
tiles and feature halves 3.0 (unrolled 3.4, register prefetch of the next
range 3.0, quarters at 512 threads 3.9-4.1); R 128 32 columns 2.67; R 128
64 columns, halves 2.26; **R 256 32 columns 2.22** (single-buffered: R 256
fills the 48 KB); same with 768 / 512 threads 3.1 / 2.9. Fewer, larger
ranges win: the per-feature loop and the barrier per range cost more than
the adds.

Head training on the same net/shape (H 1024 kb8m `--half`, batch 16384,
2 CPU threads, 300 batches, byte-identical nets at every step): cached
`dacc` 766k pos/s, one warp per z1 row 878k, per-bucket sample lists 960k,
plus the `speed-cuda` incidence-list FT backward **1.05M pos/s**. Plain
training on the final branch is **2.48M pos/s** (from 2.09M with the older
bitmask path). Head profile at batch 16384: `forward_half` 6.38 ms,
`compute_dacc` 3.34 ms, `backward_ft` 1.67 ms (3.35 ms before incidence
lists), `backward_head_w1` 1.32 ms, `adam_ft` 0.87 ms. The CPU head trainer
is 84-104k pos/s even at H 512, so bulk head runs stay on the GPU.

Approaches that lost (1080 Ti, kb8m):
- Smaller output-gradient chunks (`CH` 64 / 32 instead of 256): no gain.
- A per-pair record (accumulator row, output-weight offset, gradient)
  written by `plan_csr` so `backward_ft`'s staging skips the sample load:
  -8% at H 1024 `--half` (byte-identical, just slower).
- Larger work items (`WI` 1024 / 4096): no change.
- Bigger redesigns of `backward_ft` (64-column tiles, register prefetch of
  the next range) run out of registers: the bucket's 768 x 32 gradient
  slice already takes 24 of the 64 registers per thread at 1024 threads.

## Head speed (`--head pairwise`, branch `perf-cuda-pairwise`)

Byte-identical to `trainer-head-pairwise` (8c9ff85): every net (`.e1`,
`.e2`, final) and the checkpoint `cmp`-equal, 2 x 1.5M positions from
`pwcheck/slice.bin`, for H 1024 kb8m `--factorise --init g9h1024kb8m`
pairwise (`--half` and f32), pairwise-skip, dense `--head`, plain, and
pairwise H 512 / H 256, dense H 256.

Changes: the head moved out of `forward` into `head_forward` (block of 256
per sample, 4 blocks/SM instead of 2 of 512); `compute_dacc` folded into it
(e summed once per pairwise input, shared by both partner lanes; the dacc
rounding pinned with `__fmul_rn` / `__fmaf_rn` to nvcc's old contraction);
head inputs (pw_in / sq01) formed once into shared memory and `hin` (read by
`backward_head_w1` instead of recomputing pw_in from acc); W1 copied once
per batch (`quant_w1`): int8 q_w1 * 64 for pairwise (exact), f32 aligned for
dense, natural + lane-major layouts; small head params, ob, target in
shared; `backward_head_w1` / `backward_head_small` stage their sample rows
in shared memory.

Speed, H 1024 kb8m `--factorise --half`, batch 16384, `--threads 2`, 600
batches from `pwcheck/slice.bin`, `--init g9h1024kb8m`, 125 W cap; base and
candidate interleaved, two rounds:

| run | base 8c9ff85 | candidate | |
|---|---|---|---|
| `--head pairwise`, idle GPU (2026-11-20) | 717k / 709k | 1.49M / 1.46M | 2.07x |
| plain, idle GPU | 1.995M / 1.977M | 1.976M / 1.947M | -1.2% (noise level) |
| `--head pairwise`, GPU shared with a training run | 331k / 331k | 731k / 733k | 2.21x |
| plain, shared | 872k / 879k | 877k / 872k | same |

Profile at idle (ms/batch), pairwise base: forward (incl. the head) 16.6,
reduce_rest + head kernels 1.5, backward_ft 2.3, adam_ft 0.93; total 22.4.
Candidate: forward 3.1, head_forward 2.1, backward_ft 2.6, backward_rest
0.92, head_w1 0.35, head_small 0.12, reduce_head 0.13, adam_ft 1.0; total
10.6 (plain 7.9). Shared-GPU base profile: forward 26.6 + compute_dacc 12.0
of 52.6 ms. Lost (all byte-identical, just slower): fp16 W1 copies (L2
traffic beats the int8 decode), 3 forward blocks/SM (spills), head blocks of
128 / 512 threads. Keeping the head fused into `forward` (2 blocks/SM)
was as fast for pairwise but slower for the dense head, so it is split.

## Plain-path speed (branch `perf-cuda-plain`)

Profile, H 1024 kb8m `--factorise --half`, idle 1080 Ti at 125 W (ms/batch
of 16384, `NNUE_CUDA_PROF`): forward 2.87, backward_ft 3.14, backward_rest
0.73, adam_ft 0.89, plan_csr 0.10; total 7.84 (~2.0M pos/s in 600-batch
runs; host prep 1.9 ms runs under the GPU and is not the limit).

- **Plain `forward` caches dacc** (g w 2 c per lane, rounded exactly as
  nvcc contracted it inside backward_ft), so `backward_ft`'s staging reads
  one float per pair instead of the sample, acc, w and g (the head modes
  already did this). Byte-identical (plain kb8m `--factorise` H 1024
  `--half` and f32, H 512 `--mirror`, H 256 without `--factorise`; pairwise
  and dense heads unchanged). Shared GPU, 300 batches, interleaved:
  828k / 833k -> 879k / 879k pos/s (+6%), backward_ft 8.55 -> 5.8 ms,
  forward +0.35 ms. `backward_rest` keeps its own sum: there nvcc fuses the
  last multiply into the bias accumulation (fma(c, 2gw, gb)), so a cached
  rounded dacc would change it.
- Lost: `adam_ft` with the bucket gradients in registers (templated on
  kb <= 8) went to 108 registers and got slower (byte-identical).
- Not tried (changes the maths, would need its own flag and a loss A/B):
  lazy Adam on the FT rows a batch touches; adam_ft is ~11% of the batch.


