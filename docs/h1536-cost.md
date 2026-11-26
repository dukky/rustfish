# H1536 inference cost (26 Nov 2026)

Question (owner): a 1536-wide feature transformer (same kb8m inputs, pairwise
head, 8 output buckets) instead of 1024: what does it cost at inference, and
how much better must its eval be to pay for that?

**Answer: -15% NPS (fixed time) to -17% (bench), the same on AVX2 and
AVX-512, flat from 100 ms to 3 s searches. That is roughly 10-15 Elo at
10+0.1 and 8-12 at 60+0.6. The wider net must win by about +12 (STC) /
+10 (LTC) at fixed nodes just to break even, and by +20 or more to be
worth a generation.** The arithmetic is in "Elo translation" below.

## Setup

- Branch `nnue-h1536-speed` (4574fd8, on main 1061d5a): `H = 1536`, the rest
  unchanged. All SIMD kernels are width-generic. The tests pass, except the
  trainer fixture test, which is ignored here because it belongs to the
  H1024 net.
- **Synthetic net.** The embedded gen-12 H1024 net is widened at load
  (`nnue::widen`):
  - The first 512 pairs are the source pairs.
  - The 256 extra pairs copy every other 4-pair chunk of the source, with
    ±2 noise on their FT weights. Copying whole chunks keeps the chunk
    grouping from `head_permute`, and with it the sparsity.
  - The linear-output and W1 weights of a duplicated pair are halved.
  - Result: the eval stays near the source net's. Depth-1 scores on the
    bench positions are within ~25 cp, and against the trainer fixture the
    mean difference is 25 cp with bias -0.7 cp. Search behaves like it does
    with a real net. The net is for speed only, not strength.
- **Sparsity matches main**, measured with the new ignored test
  `head_sparsity` and with the fixture:

  | set | net | nonzero pairwise inputs | nonzero 4-byte L1 chunks |
  |---|---|---|---|
  | fixture (2000 val positions) | H1024 main | — | 15.8% (40.4 of 256) |
  | fixture | H1536 synthetic | — | 15.8% (60.7 of 384) |
  | bench random walks (9962) | H1024 main | 5.09% (52.1 of 1024) | 15.09% (38.6 of 256) |
  | bench random walks | H1536 synthetic | 5.15% (79.2 of 1536) | 15.11% (58.0 of 384) |

  These are the plan's "~6.6% / ~18%" quantities, measured on different
  positions. Both nets are measured on the same positions, which is the
  comparison that matters. A real H1536 net might be sparser or denser. The
  sparse L1 cost scales with the nonzero count, but it is only ~3-4% of the
  time (see below).
- **Measurement.** `speed-diag` tooling on GitHub x86 runners, x86-64-v3
  builds, interleaved rotation, base listed twice as an A/A control.
  One fix was needed on `speed-diag`: the AVX2-only pass now also gets
  `--allow-diff`. Without it, a candidate with different bench nodes was
  dropped from that pass.

## Results

### Fixed-depth bench NPS (bench 13, 12 interleaved rounds per runner)

Runs 45811434063 (6 runners) and 45813100762 (8 runners). Each value is the
per-runner median of the paired ratio.

| kernels | runners | median | range | ratio H1536/H1024 |
|---|---|---|---|---|
| AVX-512 (Zen 4/5 EPYC 9V74/9V45, Xeon 8573/6975P) | 11 | **-16.6%** | -18.1 .. -15.4 | 0.835 |
| AVX2 (forced with RUSTFISH_NO_AVX512, plus EPYC 7763 and one 9V74 without AVX-512) | 9 | **-17.2%** | -18.1 .. -15.4 | 0.829 |
| A/A control (main vs main) | 25 | -0.06% | -1.9 .. +1.5 | 0.999 |

The arm runner built the branch and passed its tests in both runs.

### Fixed-time NPS (nps-length: 25 FENs, Hash 16-256, 1 thread)

Default kernels. Runs 45811434113 and 45813100422, 7 runners. Ratio =
h1536 / mean(main, A/A copy). The A/A spread is within ±1.3%.

| movetime | median ratio | range |
|---|---|---|
| 100 ms | **0.851** | 0.840 .. 0.865 |
| 1 s | **0.851** | 0.848 .. 0.870 |
| 3 s | **0.849** | 0.838 .. 0.859 |

The loss does not grow with search length (3 s is 0.2 points worse than
100 ms, inside the noise), and it does not depend on Hash size. The
fixed-time loss is ~1.5 points smaller than the fixed-depth loss. The
synthetic net's tree is also 7.9% bigger at bench depth 13, but that is a
property of this net, not of the width, so time-to-depth is not quoted.

### Where the time goes (perf, bench 16)

Same CPU model for both refs (EPYC 9V74), from run 45813100762. The "per
node" column is the share ratio divided by the NPS ratio (0.83).

| part | H1024 share | H1536 share | cost per node |
|---|---|---|---|
| fused FT update + pairwise activation + nonzero list (`update_pw_avx2`, `push8z`) | 19.8% | 25.9% | **x1.58** |
| FT refresh, plain updates, row prefetch (nnue.rs) | 7.6% | 8.3% | x1.33 |
| sparse L1 (VNNI/maddubs) | 2.9% | 3.8% | x1.57 |
| L2/L3 (f32) | 1.5% | 1.4% | x1.1 (width-independent) |
| eval-cache probe/prefetch (evalcache.rs) | 2.7% | 5.3% | **x2.0-2.3** |
| search, TT, movegen | ~46% | ~38% | x0.93-1.08 (unchanged) |
| NNUE in all (nnue.rs + nnue_head.rs) | 40.3% | 47.6% | x1.43 |

- The NNUE cost scales almost linearly with H (x1.43 for x1.5 width). The
  fused FT update is the bulk of it, and it is memory- and port-bound,
  which is why AVX-512 does not shrink the ratio.
- The eval cache doubles its per-node cost. Its code is unchanged; this is
  memory-level-parallelism pressure. Each eval now streams 50% more cache
  lines (48 per FT row instead of 32), so the eval-cache/TT prefetch lines
  wait for fill buffers.

### Memory

| item | H1024 | H1536 |
|---|---|---|
| net file | 12.9 MB | 19.3 MB |
| FT weights (8 x 768 x H i16) | 12.6 MB | 18.9 MB |
| one FT row | 2 KB (32 lines) | 3 KB (48 lines) |
| accumulator per ply (2 perspectives) | 4 KB | 6 KB |
| quiet-move update traffic per perspective (parent + 2 rows in, 1 out) | 8 KB | 12 KB |
| finny table (2 x 16 slots) | 64 KB | 96 KB |
| W1, i8 (8 buckets) | 128 KB | 192 KB |
| L1 input (u8) | 1 KB | 1.5 KB |

The FT weights never fit in L2 at either width; what matters is the
per-eval traffic, which grows by 1.5x.

## Elo translation

**Speed in doublings:** log2(1/0.84) = **0.25 doublings slower**.

**Elo per doubling of speed.** One own estimate exists, and it is weak.
H512 ran 1.39x as fast as H1024 (-28% NPS for 1024 on x86, plan.md
12 Nov). In the 18-19 Nov shape test, H512 vs H1024 scored -36.9 at fixed
nodes and -17.3 at STC. The 19.6 Elo difference is the speed share, over
0.47 doublings, which gives **~41 Elo per doubling at STC**. Caveats:
- the speed figure is from an earlier engine;
- the two CIs together leave a range of roughly 10-75.

No NPS-only SPRT exists. *Assumption* (labelled): 40-60 Elo per doubling at
10+0.1 and 30-50 at 60+0.6, since returns diminish with depth.

| TC | speed cost of H1536 | break-even fixed-node gain (200k nodes) |
|---|---|---|
| 10+0.1 | **-10 to -15 Elo** | **≈ +12** (+10..+15) |
| 60+0.6 | **-8 to -12 Elo** | **≈ +10** (+8..+12) |

The break-even figures assume fixed-node Elo carries over 1:1 to timed
play. That held for width (the H512 numbers are consistent with it). It did
not hold for the pairwise head:
- pilot: +40.9 at fixed nodes, -1.7 at STC and +3.3 at LTC, at -7.9% NPS;
- gen 11: +21.4 at fixed nodes, +9.8 at LTC.

So treat the table as a lower bound, and ask for **≥ +20 at fixed nodes
before expecting an LTC pass**.

**What the width could buy.** H512 -> H1024 was worth +36.9 at fixed nodes
(short recipe, 1.27B positions). 1024 -> 1536 is 0.58 of a width doubling,
so log-linear scaling would give ~+21. A bigger net also needs more data,
and we are data-limited (~7.8B positions). A plausible guess is +10-20 at
fixed nodes, which is marginal at STC and break-even to slightly positive
at LTC. CCRL plays 2+1, close to LTC.

## Inference optimisations specific to the wider net

1. **Fused FT update** (x1.58, ~26% of time). It is bandwidth-bound: 48
   lines per row. The levers are fewer updates, not faster ones:
   - more eval-cache hits;
   - skipping the update for nodes that never evaluate (already lazy);
   - the `speed-fewer-evals` / `speed-refresh-cache` lines on `speed-diag`.

   Every eval saved is worth 1.5x as much at 1536.
2. **Prefetch tuning.** `PF_LINES = 8` was tuned for 32-line rows. At 48
   lines, the right count, and the order of the row/eval-cache/TT
   prefetches, should be re-tuned. The doubled eval-cache stall
   (2.7% -> 5.3%) is the symptom. Possible gain: a few %.
3. **Sparsity.** It is free to keep, but the FT-L1 group penalty that buys
   more of it lost -6.1 in games at 1024 (gen 11s), so it is not a
   lever. L1 is only ~4% of the time anyway.
4. **AVX-512 does not help the ratio** (-16.6% vs -17.2% on AVX2): no
   kernel-width fix.
5. **Alternatives for the same capacity budget.**
   - Spend it in the head (a wider L1 such as 16 -> 32, at ~+3-4% of the
     time) or in more king buckets (no per-eval FT cost, only memory and
     refreshes) rather than in FT width.
   - The H1536 FT tax is ~15% of NPS for every eval.

## Recommendation

H1536 can still play fast enough (0.85x NPS), but it starts ~10-15 Elo in
the hole.
- Do not make it the next generation's default.
- Run a reduced-scale shape test first: H1024 vs H1536, same data and
  recipe, pairwise. Proceed only if the fixed-node SPRT at 200k nodes shows
  **≥ +20**, and gate the real net at LTC (owner rule).
- Do the wide-net speed items 1-2 above before or alongside, since they
  shrink the tax at both widths.
