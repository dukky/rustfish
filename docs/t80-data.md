# Lc0 T80 data (Stockfish `.binpack`) for the trainer

Preparation for arm C of the three-arm data test in
`docs/research-2026-11-11-critique.md` (own self-play vs rescored self-play
vs T80, same net shape and position count, gated in games). **No decision
to use T80 has been made**; this page is how to fetch, convert and calibrate
it, and what it would cost. Measurements below are from 2026-11-12 on the
M3 laptop.

## Source

linrock's Hugging Face datasets, the same files Stockfish and Coda train on:

- `https://huggingface.co/datasets/linrock/test80-2024` (also `test80-2023`,
  `test80-2022`, `test77`-`test79`, `test60`). List the files with
  `curl -s https://huggingface.co/api/datasets/linrock/test80-2024/tree/main`.
- Files are zstd-compressed binpacks, one per month. Coda used
  `test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst` (7.69 GB). The 2024 files
  are 6.9-12.3 GB each (`.min-v2.v6`); the `.tar.zst` files are the
  unminimised originals (15-35 GB) and are not needed.
- Download URL: `https://huggingface.co/datasets/linrock/test80-2024/resolve/main/<file>`.
  HTTP byte ranges work (`curl -r 0-209715199`), and a prefix of a zstd
  stream decompresses up to where it was cut, so a partial download gives a
  usable binpack: chunks are independent and the converter drops the
  truncated last one.

## Convert

```bash
cd tools/nnue-trainer && cargo build --release
T=$PWD/target/release/nnue-trainer
URL=https://huggingface.co/datasets/linrock/test80-2024/resolve/main/test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst
# whole file, streamed: no binpack on disk, only the 32-byte records
curl -sL "$URL" | zstd -dc | $T binpack - t80-2024-01.bin --score-scale 1.0
# or a quick look at the first 200 MB
curl -sL -r 0-209715199 "$URL" | zstd -dc 2>/dev/null | $T binpack - head.bin --fens head.txt --fens-every 1000
```

`zstd` is a system tool (Homebrew/apt); the trainer itself only reads
uncompressed binpacks and stays std-only. Python 3.14+ also ships
`compression.zstd` if the CLI is missing.

`nnue-trainer binpack <in.binpack|-> <out.bin>` options:

| option | default | meaning |
|---|---|---|
| `--score-scale F` | 1.0 | record score = round(F x binpack score), White POV, clamped to +-32000 |
| `--max-score N` | 10000 | skip \|score\| > N (raw binpack units, before scaling): mate/TB-win scores |
| `--min-ply N` | 0 | skip game ply < N |
| `--max N` | all | stop after N kept positions |
| `--fens F [--fens-every K]` | off | also write every K-th kept position as `FEN \| score \| result \| move \| raw stm score \| ply` (the first three fields are our datagen text format, so `convert` reproduces the records byte for byte) |
| `--no-filter` | off | keep everything (diagnostics only; still counts reasons) |

Filters, first match counts, all reported: `value_none` (score 32002 = no
score), `placeholder_zero` (nnue-pytorch's heuristic: score 0 in a decisive
game right after \|score\| > 100), `min_ply`, `in_check` (side to move),
`capture`, `en_passant`, `promotion` (of the stored move), `score_cap`.
These match what `src/datagen.rs` never writes. Note: in T80 the stored move
is the move *played* in the Lc0 training game (sampled with temperature),
not always the search's best move, so "best move is a capture" is really
"played move is a capture".

Output is in game order (consecutive positions of one game are adjacent):
always `shuffle` it, or mix it into a pool with `shuffle`, before training.

## Measured on the first 200 MB of test80-2024-01 (min-v2.v6)

- 200 MB zst -> 250 MB binpack (zstd gains little; binpacks are already
  ~2.6 bytes/position) -> 238 complete chunks, 95.4M positions, **0 decode
  errors**.
- Speed: **6.1M positions/s** on one M3 core (decode + filter + write),
  15.6 s for the 95M.
- Keep-rate **61.0%**: value_none 18.3%, capture 10.9%, score_cap 6.4%,
  in_check 3.2%, promotion 0.2%, en_passant 0.04%, placeholder_zero 0.001%.
  (Coda quotes ~37% with its stricter ply filters.)
- Kept scores (raw, side to move): mean \|score\| 241; 33% below 25, median
  59, p90 511, p99 3297 (our pool7: median 260, p90 1136, p99 2138). The cap
  matters: the skipped tail piles up at 24000-26624 (TB/mate-like scores).
- Kept results, White POV: 21.0% win, 60.2% draw, 18.8% loss (our data:
  ~34/31/35). In decisive games, the score's sign agrees with the result in
  96.0% of positions with \|score\| > 200.
- Correctness: 120,000 unfiltered sampled positions (1,925 castles, 279
  promotions, 130 with an ep square) and 58,150 kept ones were replayed in
  rustfish: the stored move is legal in every one (`divide 1`). The 58,150
  kept positions' records are byte-identical to `convert` of their FEN text.

## Score calibration

Our records are in net units (`src/datagen.rs`); T80 scores are Stockfish
internal units converted from Lc0's Q. Four estimates of the factor that puts
raw T80 scores on our scale (1M-position strided sample of the converted
head, 1M strided from pool7, `auto/s61.bin` as current CI data):

| method | T80 raw | pool7 | s61 | factor to pool7 | factor to s61 |
|---|---|---|---|---|---|
| `scalecheck` vs scale-ref, cap 10000 | slope 1.948 (r 0.60) | 1.030 (r 0.93) | 1.252 (r 0.94) | 0.53 | 0.64 |
| same, cap 3000 | 1.217 | 1.014 | 1.230 | 0.83 | 1.01 |
| same, cap 1500 | 0.946 | 0.930 | 1.139 | 0.98 | 1.20 |
| `eval --best-scale --lambda 0`, g7kb8m | x0.894 | x0.815 | x0.978 | 0.91 | 1.09 |
| outcome fit: k in sigmoid(k*score/400) ~ wdl | 1.359 | 1.465 | 1.338 | 0.93 | 1.02 |

(val.bin: slope 0.73, best-scale x0.62; val17x2: 1.01, x0.84. The "~1.25"
slope is the current CI data, s61.)

What to match, and why: the loss sees scores only through
`sigmoid(score / 400)`, so the scale that matters is win probability per
score unit, not a straight-line slope. `scalecheck`'s through-origin slope
is the wrong tool for T80: the net predicts T80 scores much worse (r 0.60 vs
0.93) and T80 has a heavy tail, so the slope swings from 0.95 to 1.95 with
the cap. The two probability-space estimates (net best-scale with lambda 0,
and the outcome fit, which is net-free but confounded by T80's 60% draw rate)
agree: **raw T80 scores are already within ~10% of our scale**, between
pool7 (x0.91-0.93) and current CI data (x1.02-1.09).

**Proposal: `--score-scale 1.0`**, with the cap at 10000 raw. Screening
uses best-scale loss (scale profiled out), so a residual 5-10% offset costs
little; if a T80 net looks promising, a second run at 0.9 or 1.1 is a cheap
follow-up. Recheck on each new month's file (the converter prints the
distribution).

Result/WDL convention: binpack results are side-to-move POV (-1/0/1, flipped
every ply); the converter writes ours (White POV, 0/1/2 = loss/draw/win,
like the 0.0/0.5/1.0 text). The 96% sign agreement above confirms the
mapping.

## Cost of a 1-5B-position T80 arm

From the measured 0.455 raw positions per compressed byte and the 61%
keep-rate (extrapolated from one file's head; months differ):

| kept positions | raw | download (zst) | ~monthly files | records on disk | conversion (1 core) |
|---|---|---|---|---|---|
| 1B | 1.6B | ~3.6 GB | 0.5 | 32 GB | ~5 min |
| 3.2B (= pool8) | 5.2B | ~11.6 GB | 1.5 | 102 GB | ~15 min |
| 5B | 8.2B | ~18 GB | 2.5 | 160 GB | ~22 min |

One whole Jan-2024 file is ~3.5B raw, ~2.1B kept. Streaming
`curl | zstd -dc | nnue-trainer binpack -` needs no space for the binpack.
`shuffle` needs room for its input and output at once (2x), so a 3.2B arm
needs ~200 GB free: the desktop, not the laptop (61 GB free today). The
200 MB test range downloaded in under a minute here. Also hold out a T80
validation slice (e.g. the last 1M of a different month) in addition to
val.bin + val17x2, to see in-distribution and our-distribution loss
separately.

## Licence (not legal advice)

The Lc0 project states its training data "is made available under the Open
Database License" (ODbL-1.0) and "any rights in individual contents of the
database are licensed under the Database Contents License" (DbCL-1.0)
(lczero.org blog, 2021-06, "The importance of open data"). The linrock
files are rescored/minimised derivatives of that data; the dataset pages
carry no licence statement of their own.

What that implies, as we read ODbL:

- **Nets trained on it**: most likely a *Produced Work* (ODbL 4.3). Then a
  release containing the net must carry a notice that it was produced from
  the Lc0 training data under the ODbL, e.g. in the README and release notes:
  "This network was trained on data from the Leela Chess Zero project
  (test80 training data, via linrock/test80-* on Hugging Face), made
  available under the Open Database License (ODbL-1.0)." The engine's own
  licence is unaffected under that reading. Coda and Stockfish rely on it.
- **Converted data** (`t80*.bin`, shuffled pools containing it) is a
  *Derivative Database*. Publishing it (including as GitHub Actions
  artifacts of a public repo, or in a train-sweep pool) makes it ODbL
  share-alike: it must be offered under ODbL. Keeping it on our own disks is
  fine.

Needs the owner's decision before any CCRL release with a T80-trained net:
whether to accept the Produced Work reading (the common one, not tested in
court), the exact attribution text and where it goes (README, release notes,
the net's embedded metadata?), and whether T80 data may enter CI artifacts.
