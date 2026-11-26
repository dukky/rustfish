# Speed audit, 24-25 Nov 2026 (exact speedups only)

Goal: more nodes per second without changing the search (bench nodes and
`evalhash` identical to `main`). Reference gap: Clover 9.1 ~842k nps vs our
~519k on the same machine.

## Method

- **Profiles** on GitHub x86 runners (`.github/workflows/speed-compare.yml`,
  `profile` job): x86-64-v3, non-PIE, line tables, `perf record -F 4999` on
  `bench 16` (6.87M nodes), attributed by **source line through the inline
  chain** with `addr2line -i` (`scripts/speed-profile.py`; lto inlines nearly
  everything into `negamax_body`, so real symbols say little). Two runs: AMD
  EPYC 9V74 (Zen 4, `bench 14`, 8.5k samples, main aef19c5) and Intel Xeon
  6975P-C (Granite Rapids, `bench 16`, 19.7k samples, main fa393f0), each with
  the default (AVX-512) kernels and with `RUSTFISH_NO_AVX512=1` (AVX2).
- **NPS** only on GitHub runners (desktop shared, laptop in use):
  `scripts/speed-compare.py` builds base + candidates, refuses any candidate
  whose bench nodes or evalhash differ, then runs `bench 13` interleaved with
  plain rotation (each binary in each position equally often), 10-12 rounds
  per runner, 6 runners (a mix of EPYC 7763 / 9V74 / 9V45, Xeon 8370C / 85xx /
  6973P / 6975P), on AVX-512 runners once per kernel set. Reported number: the
  per-runner median of the paired per-round ratios, then the median and
  spread across runners. Every run includes an A/A control (main vs a copy).
  An `arm` job (ubuntu-24.04-arm) builds, tests and identity-checks the NEON
  path and gives a short NPS look.
- **A/A noise** (6 runners): median of runner medians -0.04..+0.26%, single
  runners -1.1..+2.0%. Gains below ~0.5% need several runs.
- First run had a rotation bug (rotation + reversal kept the base first with
  two binaries; A/A read +0.35%); fixed before the numbers below.

## Time map (x86, self time by source function / file, % of samples)

Intel 6975P-C, AVX-512 kernels (the AMD Zen 4 profile is within ~1-2 points
on every item; the AVX2 run is the same picture):

| area | % | main items |
|---|---|---|
| NNUE FT fused update + pairwise activation (`apply_delta_with` -> `update_pw_avx2`) | 16.5 incl. | row loads (L2/L3 misses on the 12.6 MB FT), stores, `mulhrs`/pack, nonzero-chunk list (`zero_mask`, `push8z`) |
| **NNUE row prefetch (`prefetch_pending` -> `prefetch_row`)** | **9.7** incl. (8.3 on the one `_mm_prefetch` line) | 32 prefetches per 2 KB row, 4-8 rows per move |
| MovePicker::new (scoring, history reads, SEE calls) | 8.9 incl. | cont-history row reads 1.8-4.4 (cache misses), SEE 3.5-3.9, `push_either` 1.1 |
| negamax_body self | 6.2 self / 11.5 incl. | spread over many lines |
| L1 + L2/L3 of the pairwise head (`l1_rest_v1/v2`) | 6.2 incl. | dpbusd 1.3, `rest_avx2_v` 3.1 |
| legal move generation | 6.5 incl. | `generate_legal_moves_fused` 3.2, magic lookups 1.2 |
| MovePicker::next / `take_best_scored` | 4.6 incl. / 3.9 | the max scan |
| static eval wrapper (`evaluate`, `finish`, scratch) | 3.7 | `head::finish` f64 math 1.4 |
| make/undo | 4.8 | NNUE `push_move` 1.4 |
| TT probe | 3.2 (6.1 on Zen 4) | the bucket load right after the prefetch |
| SEE / see_ge (other callers) | 2.8 | |
| accumulator refresh (`update_dual`, `refresh_cached`) | 3.1 | king-bucket changes |
| correction history read (`correction`) | 1.8-2.9 | 4 random i16 loads |
| child-key prefetch (`approx_child_hash`) | 2.5 | |
| qsearch self | 2.5 | |

By file: search.rs 30%, nnue_head.rs 24%, nnue.rs 18.5%, tt.rs 8.4%,
movegen.rs 5.8%, main.rs 5.3%, bitboard.rs 2%. IPC 2.44 (13.7G cycles,
33.4G instructions for 6.87M nodes: ~2000 cycles and ~4900 instructions per
node), 2.5% branch misses, 6.8% L1D load misses.

The NNUE (update + head + prefetch + refresh + eval wrapper) is ~40% of the
time; move ordering (picker, SEE, next) ~18%; movegen ~6.5%.

## Results (paired bench-13 NPS vs main, median of 6 runner medians [min, max])

On main 30f83d4 (bench 198243) unless noted.

| branch | change | x86 default kernels | x86 AVX2 kernels | arm (1 runner) | verdict |
|---|---|---|---|---|---|
| `speed-pf8` | prefetch only the first 8 of 32 lines of each pending FT row | **+1.68%** [+0.07, +3.37]; earlier runs on aef19c5/fa393f0: +2.13, +1.40, +0.98 | +1.47 [+0.28, +3.60] | +0.33 | keep |
| `speed-pf1` / `speed-pf2` | first 1 / 2 lines | +1.57 [+0.64, +3.86] / +1.56 [-0.28, +3.68] (same run: pf8 +1.96 [+0.68, +3.61]) | +1.84 / +1.73 (pf8 +2.07) | +0.53 / +0.83 (pf8 +1.25) | pf8 best |
| `speed-pf4` | first 4 lines | +1.30 [-1.13, +3.29]; earlier +2.10 | +1.80 | +1.25 | pf8 more consistent |
| `speed-pf16` | first 16 lines (aef19c5) | +2.16 [+0.31, +3.14] | +1.98 | - | similar to pf8 |
| `speed-pf0` | no prefetch at all (control, aef19c5) | +0.90 [-0.11, +4.62] | +2.35 [-1.03, +3.10] | - | full prefetch costs more than it saves on Intel; partial is best |
| `speed-hist-i16` | butterfly, continuation and capture history in i16 (gravity bounds them to +-16384) | **+0.81%** [+0.35, +2.43]; earlier +0.57, +0.38 | +0.35 [0.00, +1.51] | +1.03 | keep |
| **`speed-stack-r1`** | hist-i16 + pf8 | **+2.51%** [+1.21, +4.05] | +2.62 [+2.20, +3.72] | +0.94 | **merge this** |
| `speed-pw-avx512` | 512-bit fused FT update | +0.15 [-2.86, +2.34]: Zen 5 +2.1/+2.3, Zen 4 -2.9, Intel +0.4 | -0.95 (kernel unused) | - | drop (Zen 4 double-pumps 512-bit; mixed) |
| `speed-l1-avx512` | 512-bit L1 (BW / VNNI) | -0.26 [-0.85, +1.14] | +0.19 | - | drop (neutral) |
| `speed-corr-prefetch` | prefetch the child's correction entries after make_move | -0.78 [-1.59, +2.49] | -0.91 | - | drop (the extra CorrIdx per child costs more) |
| `speed-pick-soa` | picker lists as parallel move/score arrays, `max()` + `position()` | **-5.61%** [-6.58, -5.00] | -5.65 | - | drop (the existing scan is already good) |

All branches listed were bench- and evalhash-identical (desktop check with
default, `RUSTFISH_NO_AVX512=1` and `RUSTFISH_NO_VNNI=1`; perfttest,
`selftest 200`, `cargo test --release`; plain `cargo check` for the
runtime-dispatch path; arm job: build, tests, bench, evalhash).

## Why the prefetch change works

`prefetch_pending` issued one `prefetcht0` per 64-byte line of every row the
pending update reads: 32 per row, 128-256 per move. On Intel that line alone
was 8.3% of all samples (fill-buffer / load-port pressure; the instructions
retire behind earlier misses). The update reads each row front to back, so
the hardware stream prefetcher covers the tail once the first lines arrive:
prefetching the first 8 lines keeps the latency benefit at a quarter of the
instructions. No prefetch at all is still better than prefetching everything
on most runners.

## Ideas that would change nodes or eval (for an SPRT, not exact)

- Lazy move generation / scoring (yield the TT move before generating):
  quiet scores read history that the TT move's subtree updates, so the move
  order changes; a cutoff on the TT move would skip ~10% of the time
  (picker + movegen + SEE). Worth an SPRT as a search change.
- Staged capture/quiet generation for the same reason.
- Integer `head::finish` (f64 now): would need the trainer's rounding to
  match; small (1.4%).

## Next steps

- Merge `speed-stack-r1` (= `speed-hist-i16` + `speed-pf8`, +2.5% x86, +0.9%
  arm) via the `speedup` queue mode; the other branches are dropped.
- Bigger items left: the FT update itself (memory bound: 12.6 MB FT, rows
  missing L2) and move ordering (~18%). Exact options: fewer pending-update
  passes (evals with 2+ pending plies), cheaper SEE on the obvious cases,
  profile `negamax_body` line by line on the merged stack.
- AVX-512 kernels pay on Zen 5 only; revisit with a per-CPU choice only if
  the rating machines are known to be Zen 5 / Intel.

## Round 2 (25 Nov, main 30f83d4 / 1a453b8)

| branch | change | x86 NPS | x86 time-to-depth (bench 13) | arm | verdict |
|---|---|---|---|---|---|
| `speed-tt-first` (search change, `Bench: 183218`) | yield a verified TT move (`Board::tt_move_legal`, exact except castling / ep, which fall back) before generating and scoring the rest; the rest is scored after the TT move's subtree, so order and nodes change | -2.41% [-2.61, -0.88] | **-6.44%** [-7.89, -6.25] (faster) | NPS -0.02%, time -8.68% | SPRT |
| `speed-eval2` (exact) | two pending plies updated in one AVX2 pass (`update_pw2`) | -1.96% [-3.55, -0.43] | +2.00% | -2.55% (two-step fallback: the refactor alone costs) | drop |

Nodes to depth, main -> tt-first (deterministic, desktop): d9 198243 ->
183218 (-7.6%), d11 610971 -> 673470 (+10.2%), d13 1624905 -> 1483487
(-8.7%), d15 4226748 -> 3887979 (-8.0%). The NPS drop is the changed node
mix (fewer generate-and-score nodes are skipped than the tree shrinks);
time to depth is what counts, and the SPRT decides.

Cheaper SEE: not pursued. Capture ordering uses the exact SEE value
(`capture_stage_score_from_see`, 1024 * (16 * victim + SEE)), so no early
exit keeps the order; SEE is ~3.5% in all.

Line profile of `negamax_body` on `speed-stack-r1` (EPYC 7763, AVX2): 8.5%
self, no line above 1.1% (eval-cache probe 0.8%, `rep_stack.push` 0.7%,
move-loop overhead): nothing worth a branch. On the stack the profile is now
FT fused update 21.9% incl., picker 8.7%, head L1 6.7%, movegen 6.4%,
refresh 3.5%; the row prefetch has dropped out of the top items.

## Round 3 (25 Nov): NPS against search length (main f33163d, bench 185766)

Tool: `.github/workflows/nps-length.yml` + `scripts/nps-length.py` (25 of the
peer_nps FENs, `ucinewgame` + `go movetime T` per search, binaries
interleaved per search, main twice as A/A), counters from the diag-only
branch `speed-diag-stats` (`info string stats ...`).

NPS by movetime (total nodes / total time; A/A within ~0.5%):

| runner, Hash | 100 ms | 300 ms | 1000 ms | 3000 ms | 100 -> 3000 |
|---|---|---|---|---|---|
| EPYC 7763, 16 MB | 1.69M | 1.64M | 1.56M | 1.53M | -9.8% |
| EPYC 9V45, 64 MB | 2.61M | 2.50M | 2.42M | 2.37M | -9.3% |
| EPYC 9V45, 256 MB | 2.88M | 2.76M | 2.69M | 2.62M | -9.1% |

Same slope for every Hash. Counters per node, 100 ms -> 3000 ms (all three
Hash sizes agree to ~0.01):

| | 100 ms | 300 ms | 1000 ms | 3000 ms |
|---|---|---|---|---|
| TT hit / probe | 0.211 | 0.216 | 0.208 | 0.211 |
| eval-cache hit / probe | 0.432 | 0.414 | 0.378 | 0.352 |
| NNUE evals / node | 0.472 | 0.493 | 0.528 | 0.553 (+17%) |
| king-bucket refreshes / node | 0.067 | 0.070 | 0.075 | 0.079 (+18%) |
| qsearch share | 0.331 | 0.327 | 0.325 | 0.319 |

Node-limited runs on the desktop (load-independent) show the same:
100k -> 4M nodes, eval-cache hit 0.458 -> 0.365, NNUE evals per node 0.440 ->
0.515. perf, 100 ms vs 3 s searches (EPYC 9V45): the fused FT update grows
16.5 -> 17.9% and TT probe 7.4 -> 8.5% of samples; TT/eval-cache clearing
(0.7%) only appears in the short runs.

Reading: the TT is not the cause (hit rate flat over Hash 16-256 and over
length; NPS slope independent of Hash). The decline is the NNUE work per
node: longer searches reach more distinct positions, the fixed 2^17-entry
eval cache (2 MB) hits less, and refreshes rise. With NNUE ~40% of the time,
+17% evals per node is ~-7% NPS, which is most of the measured -9%.

Fix (exact): bigger eval cache. Results (one runner per cell; A/A within
0.5%):

| branch | entries / memory | 100 ms | 1000 ms | 3000 ms |
|---|---|---|---|---|
| `speed-ec18` | 2^18, 4 MB | +0.3 / -0.4 | +0.4 / +0.3 | +0.3 / +1.0 |
| `speed-ec19` | 2^19, 8 MB | -0.2 / -0.7 / -1.1 / -0.5 | +1.2 / +1.3 / -0.3 / +1.2 | +1.1 / +2.2 / +1.0 / +1.0 |
| `speed-ec20` | 2^20, 16 MB | -0.1 / -1.5 | +0.8 / +0.1 | +1.7 / +1.5 |
| **`speed-ec-packed`** | 2^20 in 8-byte entries (valid bit, key tag, i16 eval; still a full-key match), 8 MB | -0.5 / -1.1 | **+1.9 / +0.8** | **+1.7 / +1.8** |
| `speed-ec-packed21` | 2^21 packed, 16 MB | -1.1 / -1.9 | +1.2 / -0.3 | +1.6 / +1.5 |

(pairs: Xeon 6975P-C Hash 64 / Hash 16; ec18-ec20 first run: EPYC 7763 H64
/ Xeon H16.) `speed-ec-packed` is the pick: ~+1-2% from 1 s up, ~-1% at
100 ms. It recovers about a quarter of the decline; the rest is the tree
itself (more distinct positions per node), not a cache defect.

## Round 4 (25 Nov, main 04d6cc9, bench 187769): recovering the long-search NPS

Counters from the diag-only branch `speed-r4-diag` (do not merge): eval
sources and refresh diffs, 8 peer_nps FENs, node-limited on the desktop
(load-independent counts), Hash 64:

| per node | 100k nodes | 3M nodes |
|---|---|---|
| NNUE evals | 0.435 | 0.476 |
| from qsearch stand pat | 0.229 | 0.217 |
| from negamax static eval | 0.127 | 0.147 |
| from the LMR child-eval check | 0.078 | 0.112 |
| LMR child eval: eval-cache hit rate | 0.60 | 0.45 |
| LMR child eval: cache miss but child TT eval present | 0.1% of calls | 1.7% of calls |
| king-bucket refreshes | 0.066 | 0.074 |
| rows per refresh from the finny entry | 6.15 | 6.07 |
| nearest stack ancestor in the same bucket available | 21% | 24% |
| rows per refresh, best of entry / ancestor | 5.91 | 5.86 (-3.5%) |

Readings:
- **The per-king-bucket refresh cache already exists** (finny table, since
  king buckets landed in ffc7802: per perspective x bucket x mirror, last
  accumulator + piece bitboards, refresh applies only the difference). Its
  diffs are ~6 rows; refreshing from the nearest same-bucket ancestor on the
  stack instead would save only 3.5% of the rows. Nothing left there.
- **No NNUE eval is unused.** In-check nodes already skip the eval; the TT
  eval is already preferred over the eval cache. The LMR child-eval check
  (the evals/node growth with length is mostly this site and negamax) is
  consumed by the child itself (it hits the cache next), except the 1.7%
  where the child's TT entry already holds the eval. The qsearch lever
  (skipping the child's make + eval) is delta pruning, in SPRT separately
  (`qs-delta`).

Branches (all exact: bench 187769, `bench 15` 4168952 and evalhash
identical on the desktop with default, `RUSTFISH_NO_AVX512=1`,
`RUSTFISH_NO_VNNI=1`; perfttest, selftest 200, `cargo test --release`,
plain `cargo check`; arm job identical). Bench-13 paired NPS, median of 6
runner medians [min, max], two runs; nps-length vs the mean of main and its
A/A copy (Xeon 6975P-C Hash 16 / EPYC 7763 Hash 64):

| branch | change | bench NPS default | bench NPS AVX2 | arm | 100 ms | 1 s | 3 s | verdict |
|---|---|---|---|---|---|---|---|---|
| `speed-lazy-legal` | out of check, king moves and castling decided by the node's enemy attack mask (`ThreatMaps::all`, already built for quiet ordering, now before generation) instead of an attack test per square | +0.28 [-0.12, +1.17]; +0.15 [-0.27, +0.58] | -0.05; -0.03 | -0.33; -0.58 | -0.2 / -0.7 | +0.6 / -0.3 | +0.4 / -0.3 | neutral, not worth queueing |
| `speed-refresh-cache` | refresh at an eval ends in the fused `update_pw` kernel (runtime row count) + entry copy, instead of update passes + dot + activation passes | -0.44 [-1.51, +0.53]; -0.58 [-1.12, +0.46] | -1.34; -0.74 | -2.16; -0.10 | -1.1 / -1.3 | -0.6 / -0.6 | -1.2 / -1.0 | drop |
| `speed-refresh-cache2` | the same, kernel call out of line | -0.98 [-1.14, -0.04] | -0.64 | -0.54 | - | - | - | drop |
| `speed-fewer-evals` | LMR child eval: eval cache, then the child's TT eval, then the net | +0.15 [-0.49, +1.63] | -0.13 | +0.66 | -0.5 / 0.0 | -0.1 / -0.5 | +0.5 / -0.2 | neutral (1.7% of one site's evals), drop |
| `speed-r4-stack` | all three | -0.62 [-2.96, -0.38] | -1.13 | -1.79 | -1.3 / -0.9 | -1.1 / -0.9 | -0.4 / -0.3 | drop |

A/A this round: bench medians +0.11 / +0.13 (default), -0.11 / -0.10
(AVX2); nps-length main vs aa within 0.7%.

Why the fused refresh loses: the separate dot and activation passes read an
accumulator that is already in L1, so they are cheap, while the
runtime-length AVX2 `update_pw` is slower than the fixed 4+4 `update_dual`
(AVX-512 on the default kernels) and adds a 2 KB copy.

Full lazy legality (verify at pick time) was not built: in the stack profile
the remaining legality work is `legal_state` 0.6% and the leftover attack
tests ~0.8% (qsearch king captures, in-check nodes), so its ceiling is ~1%,
and keeping the move order exact needs the picker's `swap_remove`
tie-breaking emulated (verify a picked best move and every trailing element
before it is swapped forward, delete illegal ones order-preserving). Not
worth the risk for that ceiling.

Conclusion: the ~9% NPS fall from 100 ms to 3 s is the tree itself (more
distinct positions per node: more eval-cache misses and refreshes), and the
machinery that serves it (finny table, eval cache, TT eval reuse, lazy
accumulators) is already near what the counters allow. Remaining NPS levers
are elsewhere: move ordering (~18%), the FT update's memory traffic, and
search changes that skip work (`speed-tt-first`, `qs-delta`) through the
SPRT. Nothing from round 4 is for the queue.

## Round 5 (26 Nov, main 90b9223 / 2b679d9, gen 12, bench 216602): pairwise head path

Profile of main (run 45775246188; Xeon 6975P-C AVX-VNNI / EPYC 7763
AVX2): fused FT update + activation 18.5% incl., head L1 + L2/L3
(`l1_rest_v1`) 6.5% incl. (L2 loop 1.6%, VNNI L1 chain 1.2%),
`head::finish` 1.2%, chunk list (`zero_mask` + `push8z`) 2.3-2.6%.

All branches bench 216602 and evalhash identical (desktop: default,
`RUSTFISH_NO_AVX512=1`, `RUSTFISH_NO_VNNI=1`, both; `cargo test --release`
x86-64-v3; CI compare refuses mismatches on every runner; arm job tests).
Bench-13 paired NPS, median of 6 runner medians [min, max]:

| branch | change | default kernels | AVX2 kernels | verdict |
|---|---|---|---|---|
| `speed3-nopermute` | x86 L1 input left in `packus` chunk order (no `permute4x64`); W1 laid out to match (`head::slot`) | +0.60 [+0.12, +1.26]; +0.68 [+0.10, +0.99] | +0.80; +0.83 | keep |
| `speed3-dotvnni` | output-sum `madd`+`add` -> one `vpdpwssd` (AVX-VNNI / AVX512-VNNI), 4 accumulators | +0.57 [+0.05, +1.28]; +0.42 [+0.10, +1.19] | +0.31; +0.63 | keep |
| `speed3-finish` | `finish`: round by `trunc(y + copysign(0.5 - 2^-54, y))` in the conversion, clamp in integers (test vs the old one incl. exact ties) | +0.42 [-1.05, +0.89] | +0.37 | keep (stack) |
| `speed3-popcnt` | chunk count by `popcnt`, not a table load | +0.45 [-0.19, +0.57] | +0.30 | keep (stack) |
| **`speed-stack-r3`** | the four above | **+1.48 [+0.99, +1.83]; +1.39 [+0.18, +1.74]** | +1.50 [+1.06, +2.53]; +1.44 [+0.40, +2.33] | **speedup** |
| `speed3-l1x4` | L1 with 4 chunks in flight (8 accumulators) | -0.22 [-0.35, +2.41]; on the stack +-0 | +0.07 | drop (not latency-bound) |
| `speed3-l2sparse` | L2 over the active L1 neurons only (bit-identical: skipped terms add +-0) | -0.60 [-2.36, +0.72] | -0.80 | drop (data-dependent loop) |

A/A in the same runs: -0.30..+0.46 (default), -0.25..+0.20 (AVX2). arm:
only `finish` changes NEON-side code; arm medians are within the arm A/A
noise (+-1%). Runs: 45776044034, 45776965798, 45780090641, 45781018872.
