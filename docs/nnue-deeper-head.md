# Design note: a deeper NNUE head (FT -> 16 -> 32 -> 1)

Status: proposal (2026-11-12, desktop session). Nothing implemented.

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
