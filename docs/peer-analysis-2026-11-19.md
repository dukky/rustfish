# Peer analysis: the 3620-3800 band, 2026-11-19

**Purpose:** find what the engines between rustfish (~3581) and the top 10
(3751) share that rustfish lacks, and how they made heads and threat inputs
cheap enough to gain Elo at real time controls.
**Status:** research notes, not an audit. Architectures come from each
project's source or README at the commit or tag named below. Elo figures
are the projects' own self-play results (OpenBench-style SPRTs, UHO books
mostly) unless marked CCRL; self-play Elo overstates gains against other
engines. **Unverified** marks anything I could not check against a
primary source.

## The band (CCRL Blitz, single-CPU list, 14 Nov 2026)

Source: `computerchess.org.uk/ccrl/404/cgi/compare_engines.cgi?class=Single-CPU+engines`
(fetched 2026-11-19). The top-10 line is **3751** (#10 Integral 8).

| # | Engine | CCRL | Lang | Threat inputs in rated version | Head | Data |
|---|---|---|---|---|---|---|
| 4 | PlentyChess 8.0.0 | 3771 | C++ | yes (+ pawn pairs) | 16-32-1 x8 | 15B+ self-play, part self-distilled |
| 5 | Coda 0.9.4 | 3768 | Rust | yes | 32-32 x buckets | mostly Lc0 + own |
| 6 | Reckless 0.9.0 | 3765 | Rust | yes | 16-32-1 x8 | billions; source unpublished |
| 9 | pawnocchio 2.0 | 3754 | Zig | probably (master has TI + pawn pairs) | 16-32-1 x8 | self-play + Vine (MCTS) games |
| 10 | Integral 8 | 3751 | C++ | yes (+ pawn pairs, 50mr) | 16-32-1 x8 | 16.7B self-play |
| 11 | Alexandria 9.1.0 | 3750 | C++ | **no** (added 18 Oct, after 9.1) | 16-32-1 x8 | Lc0 data |
| 12 | Caissa 2.0 | 3747 | C++ | **no** | 16-32-1 x8 | 20.5B self-play |
| 12 | Viridithas 20.0.0 | 3747 | Rust | yes (+ pawn pairs) | 32-32x2-1 x8 | self-play only |
| 14 | Obsidian 16.0 | 3746 | C++ | **no** | 16-32-1 x8 | Lc0 data |
| 15 | Stormphrax 8.0.0 | 3744 | C++ | yes (+ pawn pairs) | 32x2-32-1 x8 | self-play only |
| 17 | Berserk 14 | 3737 | C | **no** (added 31 Oct, after 14) | 16-32-1, 1 bucket | own FenGen (volume unverified) |
| 23 | Clover 9.1 | 3725 | C++ | **no** | **none** (single layer) | self-play, 5k/20k nodes |
| 24 | Halogen 16.0.0 | 3720 | C++ | **no** (added 24 Feb, after 16) | 16-32-1 x8 | TD learning, then self-play |
| 40 | Starzix 6.0 | 3689 | C++ | **no** | **none** (single layer) | Lc0 data |
| 43 | Ethereal 14.25 | 3682 | C | no | (not checked) | own |
| 58 | Seer 2.8.0 | 3639 | C++ | no | WDL net | retrograde from 6-man EGTB |
| 60 | Koivisto 9.0 | 3630 | C++ | no | (not checked) | billions, own |

Rustfish's measured ~3581 sits between #70 Altair 7.0.0 (3578) and
#66 Thrawn 3.2 (3591).

**The table's main lesson:** threat inputs separate the top 7 from the rest
of the top 20. They do **not** separate 3580 from 3700. Alexandria 9.1,
Caissa 2.0, Obsidian 16, Berserk 14 and Halogen 16 reach 3720-3750
without them, and Clover 9.1 (3725) and Starzix 6.0 (3689) do it without
even a hidden layer. So most of rustfish's ~120-170 Elo gap to the
3700-3750 group is not net shape. It is search maturity, data, and speed,
plus a multilayer head that costs almost nothing.

## Engines

Architecture notation: `(inputs -> FT) x2 -> L1 -> L2 -> 1 x output buckets`.
"hm" means horizontally mirrored king buckets. "TI" means threat inputs
(piece-attacks-piece pairs; 60144 features in the Stockfish/Monty set,
59808 when combined with pawn-pair features, ~79856 in PlentyChess's
first version). "PP" means pawn-pair inputs (96*95/2 = 4560).

### PlentyChess 8.0.0 — 3771 (Yoshie2000, C++)
- **Net (master, `src/nnue.h`):** `(768x12 KB + TI + PP -> 1024) x2`, pairwise,
  `-> 16 -> 32 -> 1 x8`. i16 PSQ weights, **i8 threat weights**, i8 L1,
  float L2/L3, sparse L1 (NNZ tracking), Finny tables per king bucket.
  Before TI: `(768x12 -> 1792) x2 -> (16 -> 32 -> 1) x8`.
- **Training:** 15B+ self-generated positions (standard + DFRC), bullet,
  "self-distillation techniques for part of the training" (README).
- **How TI landed (PR #400, Oct 2025):** the FT was cut from **1792 to 640**
  to pay for the threats. Result: **+45.5 fixed nodes, -3.1 STC, +1.8 LTC**,
  so about neutral at real TCs. It was merged anyway and then a run of
  speed PRs followed: i8 threat weights (#411, +9.9 STC / +3.3 LTC), no
  incremental threat tables (#406, +3.4), Stockfish-style split lookup
  tables (#421, +3.5), AVX-512 threat updates (#433, +6.6), smaller
  DirtyThreat struct (#437, +1.4), `swapPiece` to skip redundant updates
  (#438, +1.4), DirtyPiece layout (#440, +7.7), prefetching threat weights
  (#463, +5.1), PP/TI speedups (#481, +7.3 / +4.7), VBMI2 NNZ + threats
  (#483), fused threat + PSQ accumulators (#488, +4.3), accumulator update
  refactor (#490, +2.4). That is roughly **+50 Elo of pure speed** on top
  of a net that started neutral, and the FT then grew back to 1024.
- **Search:** fractional depth pushed into many heuristics (8.0 notes),
  threat dimensions in continuation correction history (#428, +1.7),
  quiet-history aging per search (#451, +1.1), fail-firm in qsearch PV
  nodes, faster MovePicker constructor (#423, +3.7 VSTC), and tuning with
  7.5k-10k SPSA iterations (#409, #416, #347).
- **History:** 7.0 vs 6.0 was +24 STC; 8.0 vs 7.0 was **+42 STC / +47 LTC**
  (TI + speedups + search).
- Sources: github.com/Yoshie2000/PlentyChess (README, releases b-v7.0.2 to
  b-v8.0.0, PRs above).

### Coda 0.9.4 — 3768 (adamtwiss, Rust, written entirely by Claude Code)
- **Net (README):** HalfKA-style 16 KB + threat features `-> 1024` pairwise,
  `-> 32 -> 32` int8 hidden layers, material-bucketed outputs. Lazy
  accumulator, Finny tables, incremental threat deltas, runtime
  AVX2 / AVX-512-VNNI / NEON.
- **Training:** **mostly Lc0 self-play data** ("hundreds of billions"
  available) plus own data, customised bullet.
- **Search (README):** threat-aware 4D main history, 4-ply continuation
  history, pawn history, multi-source correction history (pawn, non-pawn,
  continuation, transition), cuckoo upcoming-repetition detection, about
  130 SPSA-tuned parameters.
- Started Jan 2026 (as GoChess) and was rewritten in Rust in March: it went
  from zero to 3768 in about 10 months. This is the closest methodological
  analogue to rustfish (an AI-written Rust engine), and the difference that
  stands out is the data source.
- Sources: github.com/adamtwiss/coda README; CCRL list.

### Reckless 0.9.0 — 3765 (codedeliveryservice et al., Rust)
- **Net (master, `src/nnue.rs`):** `(768x10 KB + 66864 TI -> 768) x2`,
  pairwise CReLU, `-> 16 -> 32 -> 1 x8`. i8 threat weights, i16 PSQ, i8 L1
  with a 256-entry NNZ table (sparse L1), float L2/L3.
- **Steps:** single layer `(768x4 -> 1024) x2 -> 1` SCReLU was replaced by a
  multilayer net (PR #402, Jun 2025): **+27.8 fixed nodes, +8.5 STC, +13.5
  LTC**. The cost was contained by 4-chunk sparse L1, a fused `dpbusd`
  emulation, the packus/mulhi trick, and separate quantisation for the
  pairwise inputs. TI came in PR #608 (Dec 2025) with the **FT cut from 1024
  to 384**: **+53.4 fixed nodes, +4.2 STC, +6.9 LTC**. The FT later grew back
  to 768 as the threat code got faster (AVX2 / AVX-512 delta splatting,
  setwise slider threats, no redundant add/sub pairs, sign-extension
  indexing: PRs #634-#728). v53 dropped piece-to-king threats (-6.4 MB, Elo
  neutral).
- **Training:** bullet, 60 numbered net generations, each SPRT-gated (each
  usually +2 to +7 Elo). Data source **unpublished (unverified)**.
- **History:** 0.7.0 was 3500 (Aug 2024). 0.8.0 (Aug 2025) was **+335 STC /
  +301 LTC** self-play, after the search was "rebuilt from the ground up"
  by three co-authors following the Kaggle challenge. 0.9.0 added +62 LTC
  (TI, NUMA, SMP). Its CCRL rating went 3500 -> 3765 in one release, and
  that release was search, not net.
- **Search:** threat-factorised quiet and noisy histories (#96, #222, #223).
- Sources: Reckless README, releases v0.8.0 / v0.9.0, PRs #402, #608, #703.

### Integral 8 — 3751 (aronpetko, C++)
- **Net (README):** `[768x16hm (factorised) + 1x11 (HMC) + 59808hm TI + 4560hm PP -> 768] x2 -> (16 -> 32 -> 1) x8`.
  It includes **fifty-move-rule inputs** (11 halfmove-clock features).
- **Training:** 16.7B unfiltered self-play positions, mostly at 20k soft
  nodes per side, 5-man Syzygy-guided datagen, openings from UHO_Lichess
  plus 3-4 random moves. The first net was bootstrapped from its HCE
  (v4); that data was later discarded. Datasets are public on Hugging Face.
- **History:** v5 (NNUE, +315 over HCE v4), v6 +53 LTC, v7 multilayer
  +57 LTC, **v8 +99 STC / +110 LTC** (data growth to 16.7B, TI, 50mr
  inputs, DFRC, SPSA for SMP).
- Sources: Integral README and releases v3 to v8.

### Alexandria 9.1.0 — 3750 (PGG106, C++)
- **Net at tag v9.1.0:** `(768x16 -> 1536) x2 -> 16 -> 32 -> 1 x8`, dual
  activation (CReLU and SCReLU concatenated into L2, so `EFFECTIVE_L2 = 32`),
  sparse L1 with a neuron **permutation step to maximise NNZ sparsity**.
  **No threats** (master added them, with 32 KB and FT 1024, on 18 Oct).
- **Training:** Lc0 data (README credits it), bullet, piece-count filtering,
  WDL filtering, a scheduled WDL weight, two-step training.
- **History:** 9.0 (Feb 2026) went multilayer with dual activation, NNZ
  and Finny AddSub: **+28.8 LTC**. 9.1 added the permutation, faster Finny
  tables and search work (6-ply conthist, pawn history, QS SEE pruning,
  "fail questionable RFP", score-stability TM): +18.3 LTC.
- Sources: Alexandria README, releases v8.1.12 / v9.0.0 / v9.1.0, `src/nnue.h`.

### Caissa 2.0 — 3747 (Witek902, C++)
- **Net:** `(32x768 hm -> 1536) x2`, pairwise, `-> 16 -> 32 -> 1` with 16
  output variants by piece count and queen presence (README) or 8 by piece
  count (2.0 release notes; they disagree). i16 FT, i8 hidden layers, sparse
  inference with `permuteNet` reordering. **No threats.**
- **Training:** own CUDA trainer (AdamW, cosine LR, input factoriser,
  **quantisation-aware training**). **20.5B+ self-play positions**,
  "progressive training: older games are purged", 312B positions seen for
  the 2.0 net, and the last layer was SPSA-tuned afterwards.
- **History:** 1.26 moved the single layer to SCReLU. 2.0's multilayer was
  **+40 LTC / +29 STC**. So Caissa was ~3700 class with a single-layer net.
- Sources: Caissa README, release 2.0.

### Viridithas 20.0.0 — 3747 (cosmobobak, Rust)
- **Net (master `src/nnue/network.rs`):** `(704x16hm + 60144 TI (+PP) -> 1024) x2 -> (32 -> 32x2 -> 1) x8`,
  with merged king planes (704 = 11 x 64). Before TI (PR #425, Apr 2026) it
  was `(704x16hm -> 2560) x2 -> (16 -> 32x2 -> 1) x8`. Moving to TI meant
  **2560 -> 1024 FT**, at **87% of the old speed** on AVX-512: **+27.4
  fixed nodes, +3.0 STC, +5.4 LTC**. Follow-ups: no intermediate buffering
  (+3.2), threat-weight prefetch (+2.0), branchless threat indexing (+2.5).
- **Training:** self-play only (never another engine's output), bullet.
  Net history file `notes/networkhistory.txt` (1850 lines) documents each
  experiment. Useful entries: **L1-norm penalty on FT activations to
  encourage sparsity, +5.9 LTC** ("sibilant"); HardSwish activations
  ruined block sparsity (70% -> 50% zero blocks) and lost; WDL 0.4 -> 1.0
  plus eval rescaling, +9.3 VLTC ("eleison"; the v19 "evaluation scaling
  breakthrough"); 2560 FT was +19 fixed nodes but 0 at VLTC ("sentinel":
  width alone stops paying); 3072-wide WDL-output net was -33 LTC
  ("mnestic").
- **History:** v16 +35 LTC, v17 +33 (net, eval-difference history updates),
  v18 +48, v19 +46 (eval scaling), v20 **+61 at 60+0.6** (new activations,
  TI).
- **Search:** threat-indexed main history since 2023 (+9.5), threat-factored
  capture history, MovePicker bonuses for moves into or out of threats.
- Sources: Viridithas README, releases v15-v20, PRs #425, #426, #433,
  #442, `notes/networkhistory.txt`.

### Obsidian 16.0 — 3746 (gab8192, C++)
- **Net (master `src/nnue.h`, Oct 2025):** `(768x13 KB -> 1536) x2 -> 16 -> 32 -> 1 x8`,
  `dpbusd` preprocessing of L1 weights, NNZ table, `L2Weights[L2*2]` (dual
  activation). **No threats.**
- **Training:** Lc0 data, bullet.
- **History:** most releases were +25 to +57 (bigger nets and search). v15
  added the first layered net (+27 LTC); v16 was +35 / +39.
- Sources: Obsidian README, releases v8-v16.

### Stormphrax 8.0.0 — 3744 (Ciekce, C++)
- **Net (master `src/eval/arch.h`):** `(704x16hm merged-mirrored + TI + PP -> 1024) x2`,
  pairwise CReLU, `-> 32` dual CReLU/SCReLU `-> 64` CReLU `-> 1`, with an
  L2 skip connection, x8 material buckets. FT quant 8 bits, L1 7 bits.
- **TI (PR #293, Feb 2026):** `(704x16hm + 60144 -> 640) x2 -> (32x2 -> 32 -> 1) x8`.
  SFNNv12-style threats (**no threats to or from kings**); FT **cut to 640
  and L2 raised to 32**, "because L1 being smaller makes increasing L2 much
  cheaper". **+33.5 fixed nodes, +2.7 STC, +9.9 LTC, +13.0 at 4 threads.**
- **Training:** "trained from zero knowledge starting with random weights",
  self-generated data only, bullet. 6.0's jump (+77-87) came from fixing a
  datagen bug and regenerating the whole dataset.
- **History:** 7.0 +88 LTC (bigger and more complex net), **8.0 +82 LTC**
  (TI, PP, grinding). Note that its LTC gains exceed its STC gains each time.
- Sources: Stormphrax README, releases v4.1-v8.0, PR #293.

### Berserk 14 — 3737 (jhonnold, C)
- **Net at tag 14 (`src/types.h`, `src/nn/evaluate.c`):** `(768x16hm -> 1024) x2 -> 16 -> 32 -> 1`
  with **one** output bucket. The accumulator is CReLU'd straight to int8
  (shift, pack, max 0) and fed to the i8 L1 dense, with **no pairwise**. No
  threats. The README at that tag still describes an older `-> 512 -> 1`
  net (stale). Master (Oct 2026) moved to `2x(12288 + 60144 TI -> 512) -> pairwise -> 16 -> 32 -> 32 -> 1 x8`.
- **Training:** bullet, own FenGen. Volume **unverified**.
- **History:** 12 +81 LTC, 13 +53 LTC, 14 (no published progression).
- Sources: Berserk README (master and tag 14), source at tag 14.

### Clover 9.1 — 3725 (lucametehau, C++)
- **Net (`src/net.h`):** `(768x7 KB, mirrored -> 1280) x2 -> 1 x8`, **single
  layer**, input-bucket factoriser (9.0). No threats.
- **Training:** bullet. 9.0's dataset mixed 5k and 20k minimum-node
  self-play, about 50% DFRC. Volume not stated.
- **Point:** this is rustfish's own net class (single layer, KB, 8 output
  buckets) and it is rated **~144 Elo higher**. The gap is search, data and
  speed, not shape.
- Sources: Clover README, releases v7.0-v9.1.

### Halogen 16.0.0 — 3720 (KierenP, C++)
- **Net at tag v16:** `(768 x KB -> 1536) x2 -> 16 -> 32 -> 1 x8`. TI came
  after v16, and master is now `768 FT + TI`.
- **Training:** TD learning, then supervised fine-tuning on self-play with
  bullet.
- **v16 (+58 LTC)** is a pure search and SMP list worth borrowing ideas
  from: hindsight extensions, triple extensions, fail-high TT cutoffs at
  insufficient depth when the score is far enough above beta, NMP only at
  expected cut nodes, a prior-countermove bonus applied to all quiet
  histories, fail-medium, negative LMR for PV nodes, LMR may not drop into
  qsearch, zero-width LMR re-search at depth 1, history updates capped at
  1/4 of max, and repeated SPSA (LTC, VLTC, SMP).
- Sources: Halogen README, release v16 notes, `src/network/arch.hpp` at v16.

### pawnocchio 2.0 — 3754 (JonathanHallstrom, Zig)
- **Net (master `src/nnue/arch.zig`):** `(768x16hm + 59808 TI + 4560 PP -> 768) x2 -> 16 -> 32 -> 1 x8`.
  This author introduced the **pawn-pair inputs** that Stormphrax, Integral,
  PlentyChess and Viridithas adopted. Whether 2.0 (Jun 2026) shipped TI is
  **unverified**.
- **Training:** self-play from zero plus games of Vine (an MCTS engine), bullet.
- **History:** 1.8 +119-121, 1.9 about +55-70, **2.0 about +110**, "most of
  the gain is from large improvements to the network".

### Starzix 6.0 — 3689 (zzzzz151, C++)
- **Net:** `(768x2x6 -> 1536) x2 -> 1`, single layer, mirrored; input
  buckets also keyed on enemy-queen presence. Trained on Lc0 data with the
  author's own trainer (Starway). 5.0 and 6.0 were each +105-133 in
  self-play.

### Older engines in the band (brief)
- **Ethereal 14.25 (3682):** mature C search, own NNUE. Development stopped
  in Aug 2024.
- **Seer 2.8.0 (3639):** WDL-output net trained by retrograde backup from
  6-man EGTB to 32 pieces. PyTorch trainer.
- **Koivisto 9.0 (3630):** CudAD/Grapheus lineage, billions of positions.
  Inactive.

### NPS
None of these projects publishes NPS that can be compared (Reckless and
PlentyChess ship a `speedtest` command; Coda says only that its v3 build is
about 12% faster than v2). **Measure it locally instead:** run their
`bench` and rustfish's on the idle 9600X with the same binary class (see
the route, item 0).

## Cross-cutting patterns

1. **The 3700+ net template is the same everywhere:** king-bucketed
   (12-32 buckets, mirrored), FT 768-1536, **pairwise multiplication**,
   `-> 16 (or 32) -> 32 -> 1`, 8 material output buckets, i8 L1 weights,
   float L2/L3, sparse L1 via NNZ lists, Finny tables. Threat engines add
   TI (+ PP) and **shrink the FT** to pay for them. Only Clover and Starzix
   still run single-layer nets.
2. **Every TI adoption landed at about 0 to +10 at real TCs, not at the
   fixed-node number:** PlentyChess +45 FN / -3 STC / +2 LTC; Reckless +53 /
   +4 / +7; Stormphrax +34 / +3 / +10; Viridithas +27 / +3 / +5. Each first
   cut the FT (1792->640, 1024->384, ->640, 2560->1024). Rustfish's ep net
   (+58 fixed depth, -10 STC, -7 LTC) is the same pattern, except that it
   kept the 1024 FT. The Elo came afterwards from months of threat-update
   speed PRs and net retrains.
3. **The multilayer head is cheap in every peer:** Reckless's switch cost
   little enough to turn +28 FN into +8.5 STC / +13.5 LTC. Caissa got +40
   LTC and Alexandria +29 LTC. Rustfish's head is +42 at equal nodes but
   runs **44% slower**. Peers do not pay that, because of the techniques
   listed under the route below.
4. **Self-play-only engines in the band use 15-20B positions:** PlentyChess
   15B+, Integral 16.7B, Caissa 20.5B. Rustfish has about 6B. The other
   group (Obsidian, Alexandria, Starzix, Coda) uses Lc0 data, which the
   owner's policy excludes for now.
5. **Big release jumps come from search rewrites as often as from nets:**
   Reckless 0.8 (+300, search rebuilt), Halogen 16 (+58, a list of search
   patches), Obsidian 10 (+57, same net, search and speed only).
6. **SPSA at scale:** PlentyChess tunes with 7.5k-10k iterations and
   re-tunes for 4 threads. Halogen re-tunes at LTC, VLTC and SMP every few
   weeks. Rustfish's SPSA ran 400 iterations and "moved nothing"; that is
   one to two orders of magnitude below the peers' budgets (inference:
   **too short to conclude that the tunables are optimal**).
7. **Gains grow with TC:** Stormphrax 8 (+62 STC, +82 LTC), Integral 8
   (+99 / +110), PlentyChess 8 (+42 / +47), and every TI PR. CCRL Blitz is
   2+1. Rustfish gates net and arch changes at 10+0.1, which systematically
   under-rates TI and heads.

## Route from 3580 to 3700

Ranked by expected Elo per unit of work for rustfish. The ranges are rough
and explicitly uncertain: they come from transferring peer self-play
numbers, which never transfer one to one. They overlap and do not add up.

**0. Diagnostic first (a day, no Elo by itself): measure rustfish against
Clover 9.1 and Caissa 2.0 / Alexandria 9.1 locally.** Run fixed-nodes
games (which isolate eval) and fixed-time games (which add speed and search
efficiency), plus `bench` NPS on the idle 9600X. Clover has rustfish's net
class and is rated 144 above it; if rustfish holds Clover at equal nodes
but loses at equal time, the gap is speed and search. This tells us which
of the levers below is real before months go into them.

**1. Make the head cheap: +15 to +35 (most confident).** Rustfish already
has the eval gain (+42 at equal nodes). Peers keep the slowdown small with
five techniques, which rustfish lacks or only partly has:
- **Pairwise multiplication of the FT output** (every multilayer peer except
  Berserk 14): split each side's accumulator in two halves and feed
  `clamp(a,0,1) * clamp(b,0,1)` to L1. This **halves the L1 input**
  (1024 FT -> 1024 L1 inputs for both sides, against rustfish's 2048) and
  replaces SCReLU's square. Implementation: the `packus/mulhi` trick, i.e.
  `mulhi_epi16(clamp(a) << s, clamp(b))` then `packus` to u8, done once per
  eval. Needs a retrain (the head's input changes).
- **Plain i8 W1 with u8 inputs and `maddubs` / `dpbusd`** (VNNI where
  available). Rustfish moved to "exact W1 (i16 / hi-lo i8)" to fix an 8 cp
  quantisation error, which roughly doubles L1 cost on AVX2 (madd on i16).
  Peers all run L1 at **i8 with scale 64** (step 1/64), the same grid that
  failed for rustfish. The difference is in training: **quantisation-aware
  training** (Caissa states it) or bullet's weight clipping keeps W1 in a
  range where 1/64 is fine. Rustfish's W1 had a median of 0.04 against a
  clip of 1.98. Hypothesis (**unverified**): train the head with W1 clipped
  to about +-0.5 or with fake-quantisation in the forward pass, then go
  back to i8.
- **Sparse L1 via NNZ lists** of 4-byte input groups (a 256-entry lookup
  table from a byte mask to indices), skipping zero groups. Rustfish has an
  nnz list; the multiplier comes from the next two items.
- **Permuting FT neurons** offline so that often-zero neurons share 4-byte
  blocks (Alexandria 9.1, Caissa `permuteNet`, Reckless via Viridithas's
  permuter tool). Free at runtime and needs no retrain.
- **An L1 penalty on FT activations during training** (Viridithas "sibilant",
  +5.9 LTC): it pushes block sparsity up (Viridithas reports about 70% of
  blocks zero). Costs a retrain.
- If it is still too slow, **cut the FT width** (all peers pair a head with
  a 768-1536 FT) or test the head at LTC rather than 10+0.1.
- Target: the head build within 10-15% of main's NPS. Peers' numbers
  suggest that turns +42 equal-nodes into roughly +10 to +25 at STC and a
  bit more at LTC.

**2. Threat inputs, done the peer way: +5 to +25 after the speed work,
about 0 at landing.** Every TI adopter that reached 3740-3770 did these
things:
- **Shrink the FT** to make room (to 384-1024, then grow it back as the
  code speeds up), and re-check L2: Stormphrax raised it to 32 because a
  smaller FT makes a bigger L2 cheap.
- **i8 threat weights** with i16 accumulation (Monty #116, PlentyChess
  #411, +9.9 STC). Rustfish's note says only 0.16% of rows fall out of
  range, so this is cheap.
- **The SFNNv12 feature set: no threats to or from kings** (Stormphrax,
  Viridithas; Reckless v53 dropped piece-to-king). Fewer features, and a
  king move does not re-touch threat features.
- **Incremental threat deltas computed in make-move** from the moved
  piece's attacks and the sliders' rays through the from/to squares, as a
  small `DirtyThreat` list (add/sub pairs, with redundant pairs cancelled;
  PlentyChess #437, #438; Reckless #653). Do not use persistent incremental
  threat tables (PlentyChess #406 removed them for +3.4).
- **Feature-index lookup split into small tables** (Stockfish style,
  PlentyChess #421) and **branchless indexing** (Viridithas #442).
- **A separate threat accumulator** that is not king-bucketed, so a
  king-bucket Finny refresh only rebuilds the PSQ part. Later **fuse the PSQ
  and threat adds into one pass** (PlentyChess #488) and **prefetch threat
  weight rows** (PlentyChess #463 +5.1, Viridithas #433 +2.0).
- AVX-512 / VBMI2 setwise delta generation (Reckless #650-#688) is x86-only
  polish. Most CCRL testers run AVX2.
- Gate at LTC (or at 2+1-equivalent), never only at 10+0.1.
- Order relative to item 1: do the head first. TI's main payoff in peers
  came alongside a head, and both compete for the same speed budget.

**3. Data: from 6B to 15-20B self-play, with recency and label hygiene:
+10 to +30 over several generations (slow, compute-bound).** The
self-play-only peers above 3740 sit at 15-20B. Concrete techniques:
- **Purge old-engine data** (Caissa: "older games are purged"). This
  matches rustfish's open "recent-data window" question.
- **Piece-count and WDL filtering** (Alexandria).
- **A WDL schedule towards pure game result late in training** (Viridithas
  eleison, 0.4 -> 1.0, +9 VLTC; Alexandria "better WDL scheduling").
- **5-man TB-guided datagen** (Integral) and 5k-20k-node games (Clover,
  Integral).
- Lc0 data would be the fastest data lever (Obsidian, Alexandria, Starzix,
  Coda) but is **excluded by owner policy**. It is noted here only so the
  trade-off stays visible.

**4. Search: a borrow-list from Halogen 16 / PlentyChess 8 / Alexandria 9:
+10 to +30 in total, each item +1 to +5 (uncertain; rustfish may already
have some).** Candidates not in rustfish's listed feature set: fractional
depth for reductions and extensions (PlentyChess); hindsight extensions and
triple extensions (Halogen); fail-high TT cutoffs at insufficient depth
when the score clears beta by a margin (Halogen #745); fail-firm /
fail-medium; threat dimensions in continuation correction history
(PlentyChess #428); 6-ply continuation history and pawn history
(Alexandria 9.1); quiet-history aging per search (PlentyChess #451);
eval-difference history updates (Viridithas 17); NMP only at cut nodes;
cuckoo upcoming-repetition detection (Caissa, Coda); bonuses for pawn
moves that create threats and for moves into or out of threats
(Viridithas #305, #306). Each needs its own SPRT; expect most to fail.

**5. Big SPSA runs: +5 to +15 (uncertain).** The peers run 5k-10k
iterations, at LTC, and re-tune after each net change. The 400-iteration
run on rustfish cannot rule this out. Run one long SPSA (several thousand
iterations, 2+1-like TC) after the head and TI land, not before (peers
re-tune after net changes).

**6. Non-NNUE speed: +3 to +10.** MovePicker 13%, movegen 9% and SEE 8% of
x86 time are larger than NNUE's 25%. Peer examples: a faster MovePicker
constructor (PlentyChess #423, +3.7 VSTC); "use mailbox in more places"
(Halogen); threat bitboards shared between the move picker and threat
features (rustfish's own idea, which also feeds item 2).

**Not levers at 1-CPU CCRL Blitz:** NUMA, SMP tuning, Syzygy and DFRC.
Peers ship them, but none of their 1-thread SPRTs credits them.

## Step 0 results: rustfish vs Clover 9.1 (2026-11-19)

Measured on GitHub runners with `.github/workflows/peer-match.yml`: rustfish
main at `8aa1c9c` (x86-64-v3) against Clover 9.1 built from its tag with its
own net (avx2), each with 1 thread and the same hash, using sprt.yml's
fastchess setup and adjudication. Elo is pentanomial from rustfish's view,
with 95% CIs. Clover is used only as a benchmark here, and everything below
is our own measurement.

| Limit | Book | Hash | Games | W/D/L | Elo (95% CI) |
|---|---|---:|---:|---|---|
| 10+0.1 | UHO | 16 | 792 | 18/345/429 | **-200** (-212 .. -188) |
| 10+0.1 | 8moves_v3 | 16 | 792 | 12/348/432 | **-205** (-222 .. -189) |
| 10+0.1 | UHO | 64 | 792 | 21/335/436 | **-202** (-215 .. -190) |
| 60+0.6 | UHO | 64 | 396 | 11/176/209 | **-191** (-205 .. -177) |
| 200k nodes | UHO | 16 | 792 | 39/282/471 | **-213** (-228 .. -198) |
| 1M nodes | UHO | 16 | 396 | 14/166/216 | **-196** (-212 .. -180) |
| depth 11 | UHO | 16 | 792 | 244/364/184 | **+26** (+11 .. +42) |
| depth 1 | UHO | 16 | 792 | 461/72/259 | **+91** (+68 .. +114) |

Speed on the 50-position OpenBench bench set (`scripts/peer_nps.py`, 1 thread,
Hash 16, ucinewgame before each position, engine order alternated over 2
repeats):

| Machine | Engine | Nodes to depth 12 | Time to depth 12 | NPS at `go movetime 1000` |
|---|---|---:|---:|---:|
| CI (EPYC 7763) | rustfish | 8,509,009 | 4.41 s | 1,525,515 |
| CI (EPYC 7763) | Clover 9.1 | 2,251,203 | 1.36 s | 1,800,411 |
| 9600X (datagen paused, GPU trainer running) | rustfish | 8,509,009 | 2.25 s | 2,628,172 |
| 9600X | Clover 9.1 | 2,251,203 | 0.75 s | 3,315,752 |

**Interpretation.**
- **Eval is not the gap.** At depth 1 (essentially eval plus qsearch)
  rustfish wins by +91, and at depth 11 by +26. Fixed depth is not an equal
  measure, since each engine defines and prunes "depth" differently, but
  nothing here points to the net being weaker.
- **The gap is per-node search efficiency.** At equal nodes rustfish loses
  by -213 at 200k and -196 at 1M, about the same as at 10+0.1 (-200) and
  60+0.6 (-191). Rustfish needs **3.8x as many nodes** to reach depth 12,
  and its 4x larger tree at equal depth buys only +26. The other engine
  searches far more selectively and gets more out of each node. That means
  pruning, reductions and move ordering, not eval.
- **Raw speed is a small part.** Clover is 18% faster at 1 s per move on
  the runner and 26% faster on the 9600X, worth about 20-30 Elo at these
  time controls. The time-control results match the fixed-nodes results, so
  time management is not a visible factor either. Nor is hash: Hash 64 (as
  in the ladder and the LTC SPRTs) gives -202 at 10+0.1, against -200 at 16.
- **Rustfish's NPS falls in longer searches:** about 1.9M in the
  depth-12 runs (around 90 ms per position) but 1.53M at 1 s per move on the
  runner, and 3.8M against 2.6M on the 9600X. Hash size does not explain
  it: 16 and 256 MB give the same movetime NPS on the M3. Worth profiling.
- **Head-to-head is larger than the rating gap.** All time controls,
  books and hash sizes give -190 to -205, against the ~137 implied by
  CCRL 3725 vs the ~3588 ladder estimate. Engine pairs are not transitive,
  and the ladder estimate has its own error, so read this as "about 140-200",
  not as a correction to the ladder.

**Caveats.** Engines count nodes in their own way (both count one node per
negamax and per qsearch entry, so the counts are roughly comparable, though
not exactly), so fixed-node Elo is indicative only. Fixed-depth Elo depends
on each engine's definition of depth. All results are head-to-head against
one engine.

**What next.** The levers are search selectivity, not the net: (1) find
where rustfish's extra nodes go (a node-count breakdown by node type and
move number at fixed depth on the bench set, and how many nodes LMR, LMP,
futility and NMP save); (2) SPSA the existing pruning and reduction
tunables at fixed nodes and at STC; (3) test more aggressive reduction and
pruning ideas from the route list as normal SPRTs; (4) profile the NPS drop
in longer searches. Repeat this match (`gh workflow run peer-match.yml -f
limit=nodes=200000`) after search changes as a cheap progress gauge.

## Caveats

- Architectures were read from master unless a tag is named. For Berserk
  14, Alexandria 9.1 and Halogen 16 I checked the rated tag, and **none of
  them has TI**, even though their masters do now. For pawnocchio 2.0 and
  Obsidian 16 I did not check the tag.
- All Elo numbers in the engine sections are self-play, mostly on UHO
  books, which inflate differences. How much of a self-play gain shows up on
  CCRL varies: Reckless 0.8 gained +265 on CCRL against +301 LTC self-play,
  but smaller patches usually shrink more (rule of thumb, **unverified**).
- "Fixed nodes" (FN) in peers and rustfish's "fixed depth" are not the same
  measure, but both isolate eval quality from speed.
- Coda's, Berserk's and Reckless's data volumes or sources are
  self-described or unpublished.
- I found no peer that published a head slowdown as large as rustfish's
  44%. The inference that QAT, pairwise and i8 explain the difference is
  mine, from reading their code, and has not been measured.
