# Design note: a deeper NNUE head (FT -> 16 -> 32 -> 1)

Status (2026-11-13, branch `head-v2`): step 1 (CPU trainer `--head`) and the
engine inference (step 3) are implemented; see **Implementation** at the end.
Step 1 loss results (CPU, from scratch, pool8 3 x 200M, val.bin): H512
0.027971 vs H512+head 0.027596; H1024 0.027683 vs H1024+head 0.027352.

## Why

Today's net is `(KB*768 -> H) x 2 -> B`: a wide feature transformer (FT)
and a single linear output per output bucket, `B` = 8 by piece count. Every
gain so far came from the FT: more data, king buckets, and now width
(main = g7h1024kb8m, gen 8 = g8h1024kb8m). Width is expensive twice over:

- **Engine:** H 1024 costs 20-28% NPS against H 512 (M3 bench, CI
  in games). Accumulator updates and the output dot product are linear in H.
- **Training:** a kb8m H 1024 FT (6.3M weights) no longer fits the 9600X's
  32 MB L3 alongside its gradient: the CPU trainer drops to 0.64-0.77M pos/s
  (2.2M at H 512), so H 1024 generations need the 1080 Ti.

The usual next step in strong NNUE engines is depth after the FT: a few
small dense layers that combine the 2H accumulator features non-linearly.
They are cheap in both places because they are tiny (thousands of weights,
not millions) and act on one vector per evaluation, not per move.

## Proposed architecture

```
FT:  (KB*768 [+ threat rows] -> H) x 2         i16, as now (SCReLU input clamp)
L1:  2H -> 16       per output bucket            i8 weights, i32 sums
     activation: [clamp(x,0,1)^2 ; clamp(x,0,1)]  -> 32 (SCReLU + CReLU copy)
L2:  32 -> 32       per output bucket            i8 or i16 weights
     activation: clamp(x,0,1)^2
L3:  32 -> 1        per output bucket            + skip: one L1 output added
                                                   linearly to the result
```

- **Per output bucket** (8 "layer stacks"), as the output layer is now:
  piece-count buckets select the head. 8 x (2H*16 + 32*32 + 32) weights at
  H 512: ~140k, ~280 kB as i16, so it lives in L2 at eval time.
- **H stays 512 (or 768)** for the first comparison, so the FT keeps fitting
  the CPU caches for training and the engine keeps H 512 update costs.
- The L1 skip term (as Stockfish's 15 + 1 split) gives the head a direct
  linear path, so it starts no worse than today's single linear output.

## Engine cost (estimate, to be measured)

Per evaluation at H 512: L1 = 2*512*16 = 16k multiply-adds, i.e. 256 AVX2
`vpmaddubsw`/`vpmaddwd` steps (u8 activations x i8 weights), plus <2k for
L2/L3. Today's output dot is 1k. The accumulator update per move (~3 rows x
H adds per perspective) is unchanged. Expected NPS cost **~5-10% at H 512**,
against H 1024's 20-28%. The standard follow-up is sparse L1: post-SCReLU
activations are mostly zero, so L1 can loop over the nonzero 4-byte chunks
only (Stockfish's `find_nnz`), which roughly halves L1 again.

Exactness rule (`src/nnue.rs`): every SIMD kernel has a scalar twin and
results never depend on the target. That stays possible with integer L1/L2/L3
(i32 sums, fixed shifts); float layers would need a fixed operation order and
no FMA contraction, so integers throughout are the safer choice.

## Quantisation

- L1 input: `clamp(acc, 0, QA)` with QA = 255 fits u8, so SCReLU's square
  moves to after L1 (`(x * x) >> s`) or L1 takes the CReLU value and the
  square is applied to L1's outputs; pick the variant whose quantised loss
  stays within ~1% of the float loss (the trainer prints both).
- L1 weights: i8 with a fixed scale (e.g. 64), clipped while training (the
  trainer already clips every weight to +-1.98); L2/L3: i8 or i16.
- Export order: FT as now, then per bucket L1 w/b, L2 w/b, L3 w/b. A net
  file header (shape + layer sizes) would end the "nothing in the file
  names the layout" problem at the same time.

## Training

- **Trainers:** CPU (`tools/nnue-trainer`, lib + batch kernels), CUDA
  (`tools/nnue-trainer-cuda`), Metal (`tools/nnue-trainer-metal`). Start with
  CPU + CUDA. The `Backend` trait already separates the data / checkpoint /
  export code from the kernels.
- **Cost:** per sample, the head adds ~2 x 16k multiply-adds (forward +
  backward) at H 512, the same order as the FT's sparse work (~60 rows x H,
  forward and backward). GPU: small. CPU: maybe +30-60% per sample at H 512
  until the dense L1 kernels are vectorised like the FT ones. That is still
  far faster than H 1024 kb8m on the CPU.
- **Gradients:** dense layers per output bucket need deterministic sums over
  the batch's samples of that bucket (sort samples by output bucket, fixed
  order). The CUDA trainer's chunked `backward_rest` already does this for
  today's output layer and generalises.
- **Warm start:** FT from an existing net (g8: copy the FT), head initialised
  so its skip path reproduces the old output layer and the non-linear part
  starts small; then fine-tune. Start loss close to the source net, the
  same idea as the factoriser's exact warm starts.

## Experiments, in order

1. **Trainer support behind a flag** (`--head 16,32`), finite-difference
   gradient test (`gradient_matches_finite_difference` exists for today's
   net), CPU vs CUDA losses within ~1e-5, deterministic.
2. **Loss check, no engine work yet:** H 512 kb8m + head vs main's shape
   (H 1024 kb8m, no head) on pool8, best-scale on val.bin + val17x2. The
   head is worth the engine work if H 512 + head reaches H 1024's loss.
3. **Engine:** inference + SIMD (AVX2, AVX-512, NEON) + scalar twin + tests,
   NPS on the M3 and the 9600X.
4. **Gate:** `stc` (shape change) against main; then generations at the new
   shape.

## Risks and open questions

- Quantisation error in L1 (i8): measure the quantised val loss gap first.
- Three trainers to keep in sync: maybe CPU + CUDA only, Metal later.
- Sparse L1 is a second engine project; dense L1 first.
- Threat inputs (branch `threat-inputs-v2`) touch the FT only and combine
  with a head without conflict.
- How much of a head's gain shows at our data size: the H 1024 nets'
  train/val gap (0.0234 vs 0.0266) says we are data-limited; a head adds
  capacity too, so expect smaller gains until pool9+.

## Implementation (head-v2)

What the trainer actually trains (differs from the sketch above): no CReLU
copy and no skip neuron; the head is added to the existing linear output,
which plays the skip role.

```
z1 = b1 + W1 . sq(a)     (16, a = both accumulators, stm first, sq = clamp(.,0,1)^2)
z2 = b2 + W2 . sq(z1)    (32)
out = linear output + b3 + W3 . sq(z2)
```

**Net file:** the plain net, then per output bucket W1 [16][2H], b1, W2
[32][16], b2, W3 [32], b3 as i16 at 8192 (the trainer's export; the engine
re-quantises at load). `nnue-trainer` tools recognise the size
(`eval`, `fen-eval`, `--init`).

**Engine** (`src/nnue_head.rs`, `HEAD` in `src/nnue.rs`, default false =
main unchanged, bench identical): L1 input `(clamp(acc,0,255)^2 + 256) >> 9`
in u8 0..127; W1 i8 with a per-net power-of-two scale chosen at load (trained
W1 are small: mean ~0.02, max ~0.07, so a fixed scale of 64 would leave 1-2
levels); sparse L1 over nonzero 4-byte chunks (NEON `sdot` via asm, AVX2
`maddubs`/`madd`, scalar twin; the nonzero list is built in the activation
pass); L2/L3 i32 at 2^12 activations. Integer throughout: SIMD == scalar
bit for bit (tests).

- Agreement with the trainer's float forward (`scripts/head_check.py`,
  2000 val.bin positions): a from-scratch 2M-position H512 kb8m head net:
  mean |engine - trainer| 0.64 cp, p95 1.6, max 3.6; g7kb8m warm-started
  head: mean 0.51, p95 0.96, max 1.02 (the engine truncates to whole cp).
  Rounding the L1 input (not truncating) was worth 4x here.
- NPS (M3, bench 13, interleaved x3, busy laptop): main (H1024 plain)
  1.80M, H512 kb8m plain 2.39M, H512 kb8m + head 1.63M: the head costs
  ~32% against plain H512 and the head net runs ~9% slower than main.
  Per eval: activation ~35%, L1 ~45%, L2/L3 ~20% of the head.
- Sparsity: main's net has 10% nonzero L1 inputs but 34% nonzero 4-byte
  chunks. `nnue-trainer permute` reorders the FT neurons, most active
  first (exact: the integer eval is unchanged, bench nodes identical):
  g7kb8m+head 52% -> 42% nonzero chunks, +2% NPS. A greedy co-activation
  grouping did worse (51%).

**Gating a head net:** `scripts/netloop.py embed --head ...` sets `HEAD =
true` (no `--match-scale`: it rescales only the linear output).
`scripts/head_check.py --net N --records val.bin --workdir D` builds a head
engine in `D` and checks it against `nnue-trainer fen-eval`.

**Trainer speed (CPU, M3, 2 threads, H512 kb8m, interleaved):** plain
~150k pos/s; `--head` 52-60k (head-v1) -> 70-76k after moving the 16-term
loop outermost in the FT-tile backward (bit-identical). The rest of the
head cost is the per-sample W1 gradient read-modify-write and the `e` sum
(~7 vector ops per W1 element per sample); keeping a tile's W1 gradient in
registers over runs of same-bucket samples (stable bucket order keeps
every element's summation order) is the next step, est. 1.3x more.

**CUDA:** ported, untested (tools/nnue-trainer-cuda/README.md, "Head").

## Step 2 on the desktop (H512 kb8m + head vs main)

1. Validate the CUDA head (tools/nnue-trainer-cuda/README.md checklist).
2. Train with the gen-8/9 chain recipe at H 512 with `--head` on every
   stage (copy `~/rustfish-data/gen9.sh`, set `--hidden 512`, add
   `--head`): plain H512 -> m1 -> kb8m factoriser. `--init` carries the
   head through the chain (head nets load as head nets); `--init <plain
   net> --head` instead adds a zero-output head (W3 = 0, exact start) if
   only the last stage should have one. Data: `--data pool9.list
   --shuffle-buffer 67108864 --stream-stride 4 --epoch-size N`.
3. `nnue-trainer permute --net net.bin --buckets 8 --king-buckets kb8m
   --data val.bin --out net-p.bin` (exact; fewer nonzero L1 chunks).
4. `python3 scripts/head_check.py --net net-p.bin --records val.bin
   --workdir /tmp/headeng` (agreement + a HEAD engine for bench/NPS),
   then `scripts/netloop.py embed --head --hidden 512 --buckets 8
   --king-buckets kb8m ...` and an `stc` gate against main.
