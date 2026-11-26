# Syzygy tablebase format: clean-room notes

Written for rustfish's clean-room Syzygy prober (`src/syzygy.rs`), November
2026. **Method:** the format was learned only from public prose (forum posts
by the format's author and others, the Chess Programming Wiki) and from
inspecting and experimenting with the downloaded 3-4-5 piece files. No code
that reads, writes or probes these files was read (no probing libraries, no
generator or engine source, no code blocks in forum posts), and nothing was
recalled from such code. Every byte-level fact below was established by
experiment; the "How established" notes say how. The source log is at the
end.

Square numbering throughout: a1 = 0, b1 = 1, ..., h8 = 63 (rank-major).
Piece codes (as stored in the files): 1..6 = P N B R Q K for White, 9..14
for Black (+8).

## 1. Files and naming

- One file per material signature and kind: `<white side>v<black side>`
  with each side written K first, then Q R B N P (e.g. `KRPvKR`). The files
  cover both colour assignments: `KRPvKR` also answers K+R vs K+R+P with the
  colours swapped (swap colours and, if pawns are present, mirror ranks).
  Names list the stronger side first; for a position, look up
  `white-v-black`, otherwise `black-v-white` with colours swapped.
- `.rtbw` = WDL (win/draw/loss under the 50-move rule, both sides to move
  stored), `.rtbz` = DTZ (distance to zeroing, one side to move stored).
- Magic (first 4 bytes): WDL `71 e8 23 5d`, DTZ `d7 66 0c a5`.
- The last 16 bytes are a checksum (prose: a 128-bit cityhash-based key);
  the prober ignores them but the layout check uses "data ends 16 bytes
  before end of file".
- Positions with castling rights are not in the tables. Positions with an en
  passant right are not stored either (prose: "probe as if ep is not
  possible, then correct the result").

## 2. Header

| offset | content |
|---|---|
| 0..3 | magic |
| 4 | high nibble: number of pieces n; low nibble flags: bit 0 = two tables (WDL with non-symmetric material: white-to-move table 0, black-to-move table 1), bit 1 = has pawns |
| pawnless: 5 | order byte: low nibble = order for table 0, high nibble = table 1 |
| pawnless: 6..6+n-1 | piece bytes: low nibble = piece code in table 0's order, high nibble = table 1 |
| pawns: 5.. | for each of 4 files (a, b, c, d): order byte, a second order byte if both colours have pawns, then n piece bytes (nibbles as above) |

Then pad to an even offset. Symmetric material (e.g. `KQvKQ`, `KPvKP`) has a
single WDL table for White to move; Black to move is probed by swapping
colours (mirroring ranks when pawns exist). DTZ files have one table
(pawnless) or one table per file (pawns).

**How established:** byte 4 = 0x31/0x41/0x51 for 3/4/5-piece pawnless, 0x33,
0x43, 0x42 (`KPvKP`) for pawn files; the nibble pairs decode to the piece
codes of the file name in two different orders (one per side to move); a
constant table "always win" sits in the slot expected for White to move in
`KRvK`/`KQvK`.

## 3. Table descriptors

For every table (pawnless: table 0 then 1; pawns: file-major, then side), in
order:

- byte `flags`. If bit 7 is set the table is constant: the next byte is the
  value for every position (2-byte descriptor, no data).
- otherwise: `block_log` (log2 block size in bytes: 5 or 6 in WDL, up to 10
  in DTZ), `idx_bits` (log2 of the sparse-index spacing), `extra` (number
  of additional block-length entries), u32 `nblocks`, `max_len`, `min_len`
  (Huffman code lengths), then `max_len - min_len + 1` u16 values
  `offs[l]`, u16 `nsym`, then `nsym` symbols of 3 bytes each, then pad to
  even.
- DTZ only, after all descriptors: for each non-constant table with flag bit
  1 ("mapped"), four value maps (each: length byte + that many bytes), in
  the order win, loss, cursed win, blessed loss; then pad to even.

All multi-byte integers are little-endian.

## 4. Compression: pairs + canonical Huffman

The author's prose describes a Re-Pair pass (repeatedly replace frequent
adjacent pairs with a new symbol, at most 4095 symbols, each symbol expanding
to at most 256 values, symbol tree stored at 3 bytes per symbol) followed by
canonical Huffman coding where "for each code length the number of symbols
with that length" is stored, and fixed-size blocks of compressed data, each
covering at most 65536 positions. The exact encoding was then established by
experiment:

- **Symbol entry** (3 bytes b0 b1 b2): `left = b0 | (b1 & 0xf) << 8`,
  `right = b1 >> 4 | b2 << 4`. `right == 0xfff` marks a leaf whose value is
  `left`; otherwise the symbol expands to `left` followed by `right`
  (children may have larger indices than the parent). Symbols used only
  inside other symbols take no Huffman code.
- **Code lengths:** `offs[l]` is the index of the first symbol whose code has
  length l; longer codes have smaller symbol indices, so the number of
  symbols of length l is `offs[l-1] - offs[l]` (for the shortest length,
  whatever completes the code space).
- **Canonical codes:** the longest codes are numerically smallest. With
  `base[max_len] = 0` and `base[l] = (base[l+1] + count[l+1]) >> 1`, a code of
  length l is any l-bit value `>= base[l]`, and its symbol is
  `offs[l] + (code - base[l])`. Bits are read most significant first.
- **Decoding a position:** find its block (section 6), decode symbols from
  the block start, adding each symbol's expansion length, until the symbol
  covering the wanted offset; then walk down the pair tree to the leaf.

How established: assuming this layout, `KRvK` black-to-move decodes from one
64-byte block to exactly 31332 values using 209 of 512 bits (the rest zero),
and every block of every 3-5 piece file decodes to exactly its stated
length; the code-length counts from `offs` satisfy Kraft equality (= 1)
exactly when one zero-frequency symbol is excluded.

## 5. Index of a position

### Pawnless tables

Pieces are taken in the table's stored order (like pieces adjacent). Groups:
a leading group, then one group per run of identical piece codes.

- **Leading group "111"** when the first three listed pieces are each the
  only piece of their code (31332 values); otherwise **"K2"**: the first two
  pieces (the two kings) form a 462-value pair index.
- **Symmetry:** mirror files if the first piece is on files e-h, mirror
  ranks if on ranks 5-8 (applied to all pieces), so it lies in a1-d1-d4.
  Then for the pieces of the leading group in order: at the first one not
  on the a1-h8 diagonal, transpose the board (a1-h8 reflection) if it is
  above the diagonal (rank > file); stop there.
- **111 index** (p1, p2, p3 = first three squares after symmetry; "skip"
  means counting only squares not already taken by the earlier pieces):
  - p1 off the diagonal (b1 c1 d1 c2 d2 d3 = t 0..5):
    `t*3906 + skip(p2)*62 + skip(p3)`;
  - p1 on a1 b2 c3 d4 (d = 0..3), p2 strictly below the diagonal (file >
    rank, 28 squares numbered in square order = b): `23436 + (d*28 + b)*62 + skip(p3)`;
  - p1, p2 on the diagonal (j = index of p2 among the 7 other diagonal
    squares), p3 below: `30380 + (d*7 + j)*28 + b(p3)`;
  - all three on the diagonal (k = index of p3 among the other 6):
    `31164 + (d*7 + j)*6 + k`.
- **K2 index:** enumerate, in this order, (1) first king on b1 c1 d1 c2 d2 d3
  with the second king on any square that is not adjacent and not equal, in
  square order; (2) first king on a1 b2 c3 d4 with the second strictly below
  the diagonal; (3) first king on a1 b2 c3 d4 with the second on the
  diagonal. Adjacent placements are omitted: 462 pairs.
- **Other groups** (k identical pieces): squares sorted ascending, each
  reduced by the number of earlier-group pieces on lower squares (s'), digit
  = `sum_j C(s'_j, j+1)`; the group has `C(64 - placed, k)` values.
- **Combining:** the digits form a mixed-radix number. The order nibble gives
  the radix slot of the leading group, counted from the least significant
  end; the other groups fill the remaining slots in piece order. Example:
  `KQvKR` table 0 has order 1, so the fourth piece takes slot 0 (least
  significant) and the lead slot 1: `index = lead * 61 + p4`; with order 0
  it would be `p4 * 31332 + lead`.

### Pawn tables

- **Leading group:** all pawns of the colour of the first listed piece (the
  generator puts a pawn first). Pawn squares are ordered
  a2,h2,a3,h3,...,a7,h7, b2,g2,...,b7,g7, c2,f2,..., d2,e2,...,d7,e7 (48
  squares; "edge distance" classes, ranks upward, the left file first). The
  first leading pawn in that order decides the file table; if it is on
  files e-h, mirror all files. Table for file f (0..3) has
  `sum_{i=0..5} C(47 - 12f - 2i, k-1)` lead values.
- **Lead digit:** for the first pawn at local position q (its rank - 2):
  `sum_{i<q} C(47 - 12f - 2i, k-1)`, plus for the other leading pawns,
  numbered `r = 47 - (position in the order)` and sorted ascending,
  `sum_j C(r_j, j+1)`.
- **Second pawn group** (pawns of the other colour, if any): squares
  numbered `sq - 8` (a2 = 0 .. h7 = 47) minus the number of already placed
  pawns on lower squares; `C(48 - placed pawns, k)` values.
- **Pieces:** as in the pawnless case (`C(64 - placed, k)`).
- **Combining:** order byte 1 nibble = slot of the leading group, order byte
  2 nibble = slot of the second pawn group; the other groups fill the
  remaining slots in piece order (slot 0 least significant). With one pawn
  colour, only the first applies.
- Colour swapping for pawn files also mirrors ranks (sq ^ 56).

How established: index sizes (sum of block lengths per table) for all files
match these formulas exactly (e.g. 31332 = 6*63*62 + 4*28*62 + 4*7*28 +
4*7*6; pawn file sizes 252/180/108/36 * 62 * 61 for `KPPvK` revealed the
interleaved order); the 111 diagonal cases were located with my own
retrograde solver for KRK/KQK DTZ; K2 order, group order and the pawn
numbering were chosen among alternatives by self-consistency (WDL(p) = best
child) and, for the multi-pawn lead digit, by comparison with the lichess
tablebase API.

## 6. Sparse index, block lengths, data

After the descriptors (and DTZ maps):

1. For each non-constant table: `ceil(size / 2^idx_bits)` sparse entries of
   6 bytes (u32 block, u16 offset). Entry k says that position
   `k * 2^idx_bits + 2^(idx_bits-1)` is at that offset in that block.
2. For each table: `nblocks + extra` u16 block lengths (positions - 1).
3. Pad to 64 bytes; each table's blocks (`nblocks * 2^block_log` bytes),
   each table's data starting 64-byte aligned.

Lookup: from the sparse entry, offset += idx - entry position; step blocks
backwards/forwards by the block lengths until the offset is inside a block.

## 7. Values

### WDL

Stored values 0..4 = loss, blessed loss, draw, cursed win, win for the side
to move, assuming the halfmove clock was just reset. Prose (author):
positions where the side to move has a winning capture are **don't-cares**
(any stored value), and a position whose best result is reached by a capture
may store a worse value; positions with illegal piece placement are also
don't-cares. Correct probing therefore:

`WDL = max(stored, max over legal captures of -WDL(after capture))`, with
an early exit on a winning capture. En passant: not stored; treat ep
captures as captures in the max, and if the only legal moves are en passant
captures, the result is the best of them (the stored value would describe a
position without that right). Stalemates and mates are stored normally.

### DTZ

- Flags: bit 0 = side stored (0 White, 1 Black, relative to the file's
  colours); bit 1 = value maps present; bit 2 = wins stored in plies; bit 3 =
  losses stored in plies.
- Stored symbol v (after the map of its WDL class if mapped). In plies:
  wins `v + 1` (bit 2) or `2v + 1`; losses `v + 1` (bit 3) or `2v + 2`;
  cursed wins `101 + 2v`; blessed losses `100 + 2v`. Values stored in full
  moves may be one ply away from the exact distance (prose: "full moves if
  you can normally afford to lose a ply"); the generator uses plies where
  the 100-ply limit needs it.
- Don't-cares: draws; positions whose best move is a winning zeroing move
  (capture or pawn move): DTZ there is 1 (101 for a cursed win). A
  checkmating move also counts as DTZ 1.
- The side not stored: one ply of search (DTZ = 1 + best child distance,
  zeroing moves count 1).
- 50-move rule: a win with clock c is a real win when `c + |DTZ| <= 100`.

## 8. What was established how, and confidence

| Part | Source | Confidence |
|---|---|---|
| file kinds, naming, one/two-sided, WDL5 values, DTZ semantics (cursed, blessed, 50-move), capture don't-cares, ep not stored | prose | high (also confirmed by experiment) |
| Re-Pair + canonical Huffman, block sizes, 4095 symbols, 3-byte symbol entries, 462 king pairs, 111/K2 names, binomial like pieces | prose (concepts only) | high |
| header bytes, piece/order nibbles, descriptor fields, symbol bit packing, canonical code convention, sparse-index semantics, layout and alignment, DTZ flags and maps | reverse-engineered from the files | high: every 3-5 piece file parses with exact layout (end of data = file length - 16) and block lengths summing to the computed index size |
| index scheme (symmetry, triangle/diagonal cases, K2 order, group radix order, pawn order and numbering, second pawn group) | reverse-engineered by experiment | high: 0 mismatches in self-consistency (12495 pawnless + 4865 pawn positions in the prototype; Rust test) and 9793/9794 WDL agreement with the lichess API |
| DTZ ply conversion | experiment vs lichess API | high for the formula; values stored in moves are within 1 ply (as the prose says) |

## 9. Validation (Rust prober vs the lichess tablebase API)

Oracle: 3768 API queries (1 request/s, cached): 18 random legal positions
for each of the 145 3-5 piece material classes (both sides to move, pawns
included), 27 en passant positions, and 900 boundary queries (known
wins/losses re-queried with the halfmove clock at 99, 100 and 101 minus
|DTZ|). Every answer also labels each child position (with the child's
clock), giving 52592 labelled positions (4999 at clock 0, 153 with an en
passant right, 6115 cursed wins / blessed losses).

- **WDL at clock 0: 4999/4999 agree.** With the clock applied (win if
  `clock + |DTZ| <= 100`): 51942/52592; all 650 differences are the same
  case: a table storing full moves gives a DTZ one ply short of the exact
  value and the exact total is 101 (the root filter ranks a total of 100 as
  "uncertain", below a sure win).
- **DTZ: 44764 exact, 7828 one ply off (moves granularity), 0 other.**
- Conventions where the oracle differs: a mated position is DTZ -1 there
  (0 here); in an earlier sample a loss with DTZ exactly -100 at clock 0 was
  called a blessed loss there (the tables and the 50-move rule count it as a
  loss).
- Self-consistency (WDL = best child over all legal moves): prototype 12495
  pawnless + 4865 pawn positions, 0 mismatches; `cargo test` repeats it on
  12 random positions per class, and checks 4986 recorded oracle answers
  (`tests/data/syzygy_oracle.txt`). Tests skip when the files are absent
  (`RUSTFISH_SYZYGY_TEST_PATH`, default `~/rustfish-data/syzygy`).
- Probe cost (M3, tables in memory): raw lookup ~0.55 us, full WDL probe
  with capture resolution ~0.9 us, DTZ probe ~10-14 us.
- Play-outs (engine with tables vs itself, 40-60 ms/move): KBNvK mates in 41
  plies; with clock 41 (41 + DTZ 59 = 100) it mates at ply 59; with clock
  42 (cursed) the game is drawn by the 50-move rule, as the tables say.

## 10. Source log

Prose read (fetched through a summariser instructed to skip all code
blocks):

1. https://www.chessprogramming.org/Syzygy_Bases : overview, sizes, WDL50+/DTZ50+, 5-valued scale, compression and checksum remarks, forum index.
2. http://www.talkchess.com/forum/viewtopic.php?t=47681 (2013, "New 6-piece tablebases"): file kinds, WDL50+/DTZ50+, one-sided DTZ, draws/winning captures as don't-cares, 64-byte blocks, capture round in WDL probes.
3. http://www.talkchess.com/forum/viewtopic.php?t=59947 (2016, "Question to syzygy author"): capture don't-cares, 462 king placements, binomial like pieces, "K2"/"K3"/"111", piece order chosen for compression, value order loss..win, <= 4095 symbols, block sizes (32/64 WDL, up to 1024 DTZ), value mapping step.
4. http://www.talkchess.com/forum3/viewtopic.php?f=7&t=68236 (2018-19, "Re-Pair compression questions"): Re-Pair then Huffman, 3 bytes per symbol, <= 256 values per symbol, counts per code length, 65536 positions per block.
5. http://www.talkchess.com/forum/viewtopic.php?t=60722 (2016, "syzygy questions"): WDL exact after zeroing moves, ep positions not stored (probe then correct), DTZ in moves vs plies.
6. http://www.talkchess.com/forum3/viewtopic.php?f=7&t=71896 (2019, "Syzygy DTZ data explanation?"): DTZ in plies, cursed/blessed, root ranking idea.
7. http://www.talkchess.com/forum/viewtopic.php?t=65390 (2017, "understanding DTZ"): remark that DTZ is stored halved.
8. https://talkchess.com/forum3/viewtopic.php?f=7&t=82613 (2023, "EGTB encoding"): DTZ stores the smaller side.
9. http://www.talkchess.com/forum3/viewtopic.php?f=7&t=68755 (2018): N-1 piece probes come from the compression scheme.
10. Two web searches (result titles/snippets only, no result opened): snippets mentioned that files store piece order as codes with the king = 6 and a "leading pawn group" (from a code-review page and a generated wiki of the generator; neither page was opened).

Data and oracle:

11. https://tablebase.lichess.ovh/tables/standard/ (3-4-5 WDL and DTZ download, md5 list): files verified by md5.
12. https://tablebase.lichess.ovh/standard?fen=... : black-box oracle, at most 1 request/s, answers cached locally.
