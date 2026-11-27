# NNUE inference speed, 26-27 Nov 2026 (exact speedups only)

The goal was more NPS for the H1024 net and for a future H1536 net, with bench
nodes and `evalhash` identical to `main`. **Result: `nnue-speed-stack` is
+8.5% NPS at fixed depth on the desktop (Ryzen 5 9600X, Zen 5). The A/A
control read +0.7%.** Most of the gain is memory paging, not SIMD.

## Method

- Desktop, WSL2 Linux. THP is set to `madvise` (the common default).
  Datagen (9 threads) and GPU training kept running the whole time, so the
  machine was loaded at ~10 of 12 threads.
- Every run is pinned with `taskset -c 11` and interleaved: each round runs
  every binary once, in rotated order.
- The reported number is the median of the per-round ratios to the base,
  with the min/max over rounds. The base is always listed twice: the copy
  is the A/A control.
- Builds use `-C target-cpu=native`. The AVX2 build uses
  `-C target-cpu=x86-64-v3`, and "AVX2 kernels" means
  `RUSTFISH_NO_AVX512=1`.
- Fixed depth is `bench 15` (3.94M nodes). Fixed time is `go movetime`
  1000 / 3000 ms on 8 bench FENs, Hash 64.
- The harness is a stdlib Python script: interleave, rotate, median.
- Identity check, for every binary and every kernel env
  (`RUSTFISH_PW512=0/1`, `RUSTFISH_NO_AVX512`, `RUSTFISH_NO_VNNI`):
  - H1024: bench 202863, `NNUE eval hash 20bb9f040780000c`, the same as main.
  - H1536: bench 176046, NNUE eval hash `e8e9bf03a72b38a1`, the same as its
    base.
- `cargo test --release` passes on the desktop. A new test runs every
  fused-update flavour the CPU supports (avx2, avxvnni, avx512vnni-256, and
  the new 512-bit BW and VNNI kernels) against the scalar twin, for all
  (adds, subs) shapes, with and without the output dot product.

## Results, H1024 (fixed depth, bench 15)

| branch / change | NPS vs main | rounds | min..max |
|---|---|---|---|
| A/A (main copy), 3 runs | +0.16 / +0.73 / +0.80 | 20 each | -3.1..+3.9 |
| `nnue-speed-thp` (61232d0): eval cache (8 MB) on 2 MB THP pages | **+5.69** | 20 | +2.1..+8.9 |
| `nnue-speed-thp-ft` (1014f62): + FT weights on THP | **+7.00** | 20 | +1.4..+8.7 |
| `nnue-speed-zmm` (51fb054 alone on main): 512-bit fused update_pw, Zen 5+ only | +2.00 | 20 | -0.6..+5.7 |
| **`nnue-speed-stack`** (thp + thp-ft + zmm) | **+8.58 / +8.51** (2 runs) | 20 each | +4.4..+13.0 |
| stack with `RUSTFISH_PW512=0` (no zmm) | +6.14 | 20 | +2.6..+9.7 |
| `nnue-speed-thp-hist` (stack + continuation/correction history on THP) | +9.01 (+0.5 over stack) | 20 | unproven |
| `nnue-speed-thp-slider` (hist + magic table copied to THP) | +8.37 (~0) | 20 | no gain |

The FT THP result revises the old `hugemem.rs` note ("no gain at 12.6 MB").
That note held when FT pages were the only big 4 KB-paged random table. Once
the eval cache is off 4 KB pages, the FT pages add +1.3%.

### AVX2-only build (x86-64-v3, `RUSTFISH_NO_AVX512=1`), bench 15, 16 rounds

| | vs main (same build and kernels) |
|---|---|
| A/A | +1.45 (-0.6..+4.5) |
| stack, AVX2 kernels | **+6.77** (+5.0..+9.4) |
| main v3, runtime AVX-512 | +0.98 |
| stack v3, runtime AVX-512 (zmm on) | +9.27 |

The generic AVX2 build gains the full THP share. The zmm kernel is
runtime-detected, so a v3 release binary still uses it on Zen 5.

### Fixed time (8 FENs, Hash 64)

| | 1 s (10 rounds) | 3 s (8 rounds) |
|---|---|---|
| A/A | +1.66 | +1.44 |
| stack vs main | **+9.60** (+5.6..+11.6) | **+3.63** (+2.5..+4.4) |
| H1536 base vs H1024 main | -14.10 | -11.23 |
| H1536 stack vs H1024 main | -9.52 | -8.08 |

The 3 s gain is smaller. A likely cause is that the TT grows into its
64 MB, so the TT TLB/cache misses (already on THP) start to dominate. This
was not checked further; worth a re-measure with more rounds.

## Results, H1536 (synthetic widened net, branch base `nnue-speed-h1536-base` = 4574fd8 on current main)

| H1536 branch | vs H1536 base, bench 15, 16 rounds |
|---|---|
| A/A | -1.01 (-6.3..+5.8) (noisier: fewer rounds) |
| + eval cache THP | +2.46 |
| + zmm | +3.22 |
| + eval cache + FT THP | +3.52 |
| **stack** | **+6.56** (-1.0..+14.5) |
| (H1024 main vs H1536 base) | +19.6 |

H1536 now costs:
- **0.87x** at 1 s (was 0.86x on this machine, 0.85x on CI);
- **0.91x** at 3 s.

These ratios compare H1536 stack against H1024 main, so the same
optimisations applied to H1024 widen the gap again. **Against H1024 stack,
the H1536 ratio stays about the same (~0.83-0.89).** The speed tax on the
wider net is relative: these changes help both widths, and they do not
change the break-even in docs/h1536-cost.md much.

## What each change is

1. **Eval cache on huge pages** (`src/evalcache.rs`: `Vec` -> `HugeBuf`).
   - The cache is 8 MB of 8-byte slots probed at random.
   - With 4 KB pages, nearly every probe and prefetch also misses the TLB,
     and the FT rows plus the other tables crowd the L2 TLB.
   - This is the "eval-cache probe at 2x cost" from h1536-cost.md. The
     probe itself is already row-size independent (8 bytes per entry); the
     cost was paging.
2. **FT weights on huge pages** (`AlignedI16` -> `HugeBuf<Line>`).
   - Uses 12.6 MB at H1024 and 18.9 MB at H1536.
   - `HugeBuf` gained `Clone`.
3. **512-bit fused FT update + pairwise activation** (`nnue_head.rs`,
   `update_pw512[_vnni]`).
   - zmm adds and subs, `vpdpwssd` zmm for the linear output sum, and
     `packus` plus one `vpermq` that restores the 256-bit kernels' `slot`
     order. W1, the nonzero list and the sums are bit-identical.
   - **On only for AMD family >= 0x1A (Zen 5)**:
     - Zen 4 double-pumps 512-bit and measured -2.9% (speed audit, 24 Nov).
     - Intel measured +0.4%, and older AVX-512 Xeons downclock.
   - `RUSTFISH_PW512=0/1` overrides the choice.
   - The 24 Nov "AVX-512 does not help" verdict came from runners whose hot
     fused kernel was still 256-bit. On Zen 5 the 512-bit kernel is worth
     +2.0% alone and +2.4% on top of THP.

## Caveats

- **THP is Linux-only.** `madvise` is a no-op elsewhere.
  - Windows (common for CCRL testers) and macOS get none of the THP gain.
  - Windows large pages need `VirtualAlloc(MEM_LARGE_PAGES)` plus the "Lock
    pages in memory" privilege, which testers rarely grant. Not attempted.
  - GitHub runners are Linux, so SPRTs and ladders will see the gain.
- With THP set to `never`, the THP changes do nothing. Memory per search
  thread rises by at most 2 MB of rounding.
- The desktop was loaded throughout, which likely magnifies TLB/L2
  effects. **The CI speed-compare on idle-ish runners is the better
  absolute number.** A/A spread here is +-3% per round, and medians are
  within +-1.7%.

## Left / not done

- `thp-hist` +0.5% and `thp-slider` ~0: under the 1.5% bar, unproven, not
  recommended.
- The zmm L1 (24 Nov: neutral) and `PF_LINES` re-tuning for 48-line H1536
  rows were not re-measured. The prior pf4/8/16 results were flat, so this
  is probably under 1%.
- Not measured: the 3 s gain shrinking (TT-bound?).
- Not built: a Windows large-page path.
- NEON: nothing new. Apple Silicon already uses 16 KB pages, and the macOS
  superpage API is x86-only.
