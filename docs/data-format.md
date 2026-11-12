# Training-data format: packed `.rfp` files and the streaming shuffle

Status (2026-11-12): implemented on branch `packed-data`
(`tools/nnue-trainer/src/packed.rs`, `src/stream.rs`). Parity was checked
on the laptop at a small scale (section 5). Not yet used by netloop or on the
desktop.

## 1. Problem

Today every position costs 32 bytes twice: the game-ordered per-seed source
(`s<seed>.bin`, from `convert`) and a fully shuffled pool (`shuffle`),
plus a third copy while shuffling without `--consume`. pool8 is 3.2B
positions = 102 GB, data grows by about 1B positions per generation, and the
training box has a 250 GB volume.

What the trainer needs from a position is exactly the 32-byte record
(`Rec`): the pieces (occupancy + 4-bit piece codes), the score (i16, White's
view), the result and the side to move; 4 bytes are padding. Castling
rights, en passant, the move counters and the move played are not stored and
not used.

## 2. Measurements on our data

Data: game-ordered sources `s4.bin` (net2, 79M positions), `s12`/`s13`
(17M each), `fix/s14x2..s17x2` (18-19M each), local text `L1005..L1007`
(28-41M each). Games keep ~81 of ~118 plies (checks, captures, promotions
and mate scores are dropped), average 17.8 pieces per position. All
options are lossless for `Rec`.

| option | game-ordered sources | shuffled pool (pool7) | decode, 1 thread | implementation risk |
|---|---|---|---|---|
| 32-byte `Rec` (today) | 32 B | 32 B | memcpy | none |
| (a) compact record, fixed (8 occ + 16 nibbles + 2 score + 1 flags) | 27 B | 27 B | memcpy-like | very low |
| (a) compact record, variable (nibbles for the actual pieces only) | 20.4 B | 20.6 B | >50M/s | low |
| (c) zlib -9 over 32-byte records | 9.7 B | 18.3 B | (std impl: ~300-line inflate) | medium |
| (c) bz2 -9 | 12.6 B | 16.9 B | | high |
| (c) lzma -6 | 6.7 B | 15.5 B | (std impl: ~1000+ lines) | high |
| **(b) chained board diffs (implemented)** | **3.38 B** (net-scale data), 3.54 B (s12/s13) | 19.6 B (all keyframes) | **18.4M pos/s** | low-medium |
| (b') binpack-style legal-move index (estimate) | ~2.3 B | n/a | ~5-10M/s | high |

Notes:

- (c) was measured with Python's zlib/bz2/lzma on 1M records (32 MB) of
  `s17.bin` and `pool7.bin`. Shuffling destroys most of the redundancy a
  general-purpose compressor finds; on sources it reaches 6.7 B at best and
  we would have to write the codec in std Rust. lzma over the packed `.rfp`
  stream saves only 5% more (zlib 3%), so (b) is close to the entropy of
  its own model.
- (b') was not built. It would need a full legal move generator in the
  trainer (the engine's lives in another crate), the full FEN (castling
  rights and en passant decide which moves are legal, and `Rec` drops
  them), and an encoder that finds the moves of the skipped plies between
  two kept positions. That only works from text or from a changed datagen
  that writes whole games. It would save about 1 B/position over (b).

## 3. The format: chained board diffs (`.rfp`)

Board diffs, not moves: each record is stored as the edits that turn the
previous record's board into its board. No chess rules are involved, so:

- skipped plies (captures, checks, the recapture after a dropped capture)
  are just more edits (1.19 MOVEs + 0.155 CLEARs per position on average);
- the encoder works from any record stream: datagen text, game-ordered
  `.bin` sources, or shuffled pools (these round-trip too, just as
  keyframes);
- game boundaries need no marker: when a delta would cost more than a full
  record (a new game) or the result changes, the encoder writes a keyframe.

Layout (all little-endian; bits LSB-first within u64 words):

```text
file   = "RFPACK1\0" chunk*
chunk  = "RFCK" n:u32 payload_bytes:u32 crc32(payload):u32 score_unit:u16 k:u8 0:u8  payload
record = "11" occ:64 piece:4 x popcount(occ) stm:1 result:2 score:16      keyframe
       | "0"  edits score_delta                                              delta, side to move flips
       | "10" edits score_delta                                              delta, same side to move
edits  = ( "0" from_index to_sq:6          MOVE (a piece on to_sq is replaced)
         | "110" sq:6 piece:4              SET
         | "111" from_index                CLEAR )* "10" (END)
from_index  = index of the square among the occupied squares, ceil(log2(pieces)) bits
score_delta = signed Exp-Golomb (order k) of (score - previous score) / score_unit
```

A chunk starts with a keyframe and decodes on its own; it holds ~65K
positions (~800 games, ~220 KB) and ends at a game boundary. `score_unit`
is the gcd of the chunk's scores (2 for net-scale data, which is why the
old x1 seeds cost 0.9 bits more per score); `k` is the best Exp-Golomb order
for the chunk. A CRC-32 per chunk catches corruption. The result is
inherited by deltas.

Where the bits go (net-scale data, per position): delta header 1.1 bits,
edits 16.7 bits, score 6.9 bits, keyframes 0.33 B (1.24% of records).

Measured: 3.367-3.398 B/position on every net-scale source (s14x2..s17x2,
L1005..L1007, s4), 3.54 on s12/s13. pool8 (3.2B) would be ~11 GB instead
of 102 GB; a generation of 1B positions is ~3.4 GB.

Possible v2 refinements (not needed now): coding a MOVE's target as an
index into the squares that piece type can reach on an empty board (about
-2.4 bits per move, ~3.0 B/position); starting a game's first keyframe from
the start position (-0.1 B).

## 4. Streaming shuffle

A chained format cannot be shuffled per position on disk, and position
order matters (game-ordered data trains much worse). The loader
(`stream.rs`) instead:

1. builds, for every pass, one random order of all chunks of all files
   (files are interleaved at chunk granularity, 65K positions);
2. decodes chunks in that order (in decoder threads read round-robin, or
   inline; the stream is identical for any thread count and a given seed);
3. pushes every record into a shuffle buffer of `N` records and emits a
   uniformly random resident (each record is emitted exactly once, after a
   delay with mean `N`).

How big `N` must be. A batch of `b` = 16384 from a buffer of `N` contains
other positions of the same game with probability ~ `b/N` per position
pair; positions of a game share the result label. The design effect of
batch means (`var(batch mean) * b / var(sample)`, 1 for i.i.d.;
`nnue-trainer shuffle-check`) on s4.rfp:

| buffer `N` | RAM | design effect, White-POV result | same, trainer target |
|---|---|---|---|
| 0 (chunk order only) | - | 80.2 | 0.27 |
| 64K | 2 MB | 9.79 | 0.93 |
| 256K | 8 MB | 3.23 | 0.96 |
| 1M | 32 MB | 1.54 | 0.99 |
| 4M | 128 MB | 1.10 | 1.01 |
| 16M | 512 MB | 0.99 | 0.98 |
| 64M | 2 GB | 1.04 | 0.98 |
| shuffled 32-byte pool (reference) | | 0.99 | 1.03 |

(+- 0.022 noise.) It follows `1 + ~40 * b / N`. At 16M the batches are
indistinguishable from a fully shuffled pool. The training sweep below
decides.

Throughput (M3, release build, `shuffle-check`, s4.rfp):

| | 1 decoder thread | 2 decoder threads |
|---|---|---|
| decode only (buffer 0) | 18.4M pos/s | 32.1M pos/s |
| decode + 16M shuffle buffer | 12.6M pos/s | 23.0M pos/s |
| 32-byte `.bin` via the stream, 16M buffer | 33.5M pos/s | |
| shuffled pool, old feed (reference) | 38.7M pos/s | |

The CUDA trainer needs ~4M pos/s: one decoder thread (or `--loader-threads
0`, decode inside the loader thread) is 3x faster than that.

PARITY_PLACEHOLDER

## 6. Usage

```bash
T=tools/nnue-trainer/target/release/nnue-trainer
gunzip -c datagen-s119-n5000.txt.gz | $T pack s119.rfp -   # text -> packed (verified round trip)
$T pack s4.rfp s4.bin                                        # game-ordered 32-byte source -> packed
$T unpack s4.rfp s4-back.bin                                 # packed -> 32-byte records, same order
$T train --data packed/ --shuffle-buffer 67108864 ...        # a directory of .rfp/.bin
$T train --data pool9.list ...                               # manifest: one path per line, # comments
$T shuffle-check --data packed/ --shuffle-buffer 16777216    # loader speed + batch i.i.d. check
mkfifo f; $T stream --data packed/ > f &                     # shuffled 32-byte stream for other trainers
```

`train` uses the streaming loader when `--shuffle-buffer` is given or any
`--data` is a directory, a manifest (`.txt`/`.list`) or a packed file;
plain `.bin` files without `--shuffle-buffer` take the old pool feed
unchanged. `--epoch-size` defaults to the total positions. The default
buffer is 16M records (512 MB). `--loader-threads N` (default 1; 0 =
decode in the loader thread).

`pack` verifies every chunk by decoding it before writing (`--no-verify`
skips it); it accepts `.bin`, text, `-` and `.rfp` inputs, concatenated.
Invalid records (nonzero padding, bad piece codes) are skipped and
counted.

## 7. What is and is not recoverable

- Datagen text and game-ordered `s<seed>.bin` sources pack exactly (byte
  round trip checked against `convert`: 8M positions of L1007 text, text
  and `.bin` inputs give byte-identical `.rfp` files, `unpack` gives back
  `convert`'s output byte for byte).
- Shuffled pools pack exactly but only to ~19.6 B/position: the game
  order cannot be recovered from a shuffled pool (we would have to
  rediscover which positions are consecutive). Keep or re-pack the
  sources instead; a pool is a derived file.
- The x2 score fixes (`fix/*x2.bin`) are per-record transforms; pack the
  fixed copies.

## 8. Migration

1. Desktop: `pack` every game-ordered source `s<seed>.bin` (and
   `fix/*x2.bin`) to `.rfp` next to it, check the reported count, delete the
   `.bin`. ~1 minute per 100M positions on one core.
2. Replace the pool build (`shuffle --out poolN.bin ...`) by a manifest
   `poolN.list` that lists the `.rfp` files in the pool, and train with
   `--data poolN.list --shuffle-buffer 67108864`. Delete the shuffled pools
   once a net trained this way matches (section 5).
3. netloop: ingest pipes `gunzip -c | nnue-trainer pack s<seed>.rfp -`
   instead of `convert`; `pool` writes a manifest instead of shuffling.
4. Datagen needs no change (text stays the interchange format; CI
   artifacts stay gzipped text). Writing `.rfp` directly from datagen would
   need the encoder in the engine crate; not worth it while pack takes a
   minute per seed.
5. CUDA trainer: nothing to do; it shares this loader through
   `nnue_trainer::cli`. Use `--loader-threads 0` to stay at 2 CPU threads.
6. Metal trainer (own Swift feed of 32-byte files): either feed it through
   a FIFO now (`nnue-trainer stream --data ... > fifo`, and train with
   `--data fifo --epoch-size <positions>`: its feed only `fopen`/`fread`s and
   the stream never ends), or port `decode_chunk` (~120 lines) and the
   shuffle buffer to Swift.

## 9. Shared record-source interface

`stream::ChunkSource` (name, chunk count, chunk length, `read_chunk(i, out:
&mut Vec<Rec>)`) is the seam: any game-ordered source split into
independently readable blocks can implement it and reuse the chunk
interleave + shuffle buffer: `.rfp` files, raw `.bin` files (65K-record
chunks), and a T80 binpack reader (one binpack chunk = one `ChunkSource`
chunk, decoded to `Rec`s). `train` only sees `Feeder::batch`.
