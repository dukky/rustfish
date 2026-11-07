# nnue-trainer (v2)

Std-only trainer for the `(768 -> H) x 2 -> B` SCReLU nets in `src/nnue.rs`
(`B` = output buckets by piece count; the engine currently runs H = 256, B = 1).
See the module docs in `src/main.rs` for the maths and quantisation.

```bash
cargo build --release
T=target/release/nnue-trainer
gunzip -c datagen-s9-n8000.gz | $T convert - s9.bin   # text -> 32-byte records
$T shuffle --out pool.bin s4.bin s5.bin s8.bin s9.bin  # optional global shuffle
$T train --data pool.bin --val val.bin --out net.bin --hidden 256 --epochs 20
$T eval --net ../../net/nnue.bin --val val.bin          # same metric for any i16 net
```

Rules that the v1 history taught us (see `docs/nnue-experiments.md`):

- **Always train with `--val`** on positions never trained on, and compare
  candidates by validation loss *before* spending CI games. The quantised
  validation loss is printed too; it should be within ~1% of the float one.
- Change **one variable per experiment** and record it in the log.
- Train on the **whole valid pool**, many epochs, cosine LR decay
  (`--lr 1e-3 --lr-end 1e-5`, batch 16384). Two epochs on one generation's
  data (what v1 did on a 2-core box) leaves most of the net's strength unused.
- Every weight is clipped to +-1.98 while training, so output weights export
  to |w| <= 127 (the engine's fast path) with no post-hoc clipping.

Throughput: ~1.1M positions/s at H = 256 on the 8-core M3 (8 threads).

## King buckets (input buckets)

`--king-buckets <layout>` (train and eval; default `none`) gives each
perspective its own block of 768 feature rows per bucket of its own king
square: `feature = bucket*768 + side*384 + kind*64 + sq`, with squares in
the perspective's orientation. Mirrored layouts also flip every file
(`sq ^ 7`) for a perspective whose king is on files e-h.

| layout | KB | mirror | buckets (perspective-relative king square) |
|---|---|---|---|
| `none` | 1 | no | today's net, bit-for-bit |
| `m1` | 1 | yes | mirroring only |
| `kb4` | 4 | no | rank 1 a-d / rank 1 e-h / rank 2 / ranks 3-8 |
| `kb4m` | 4 | yes | rank 1 a-b / rank 1 c-d / rank 2 / ranks 3-8 |
| `kb8m` | 8 | yes | rank 1: a, b, c, d each; rank 2: a-b, c-d; ranks 3-4; ranks 5-8 |

The net file grows to `(KB*768*H + H + B*(2H+1))` i16; nothing in the file
names the layout, so pass the same `--king-buckets` to `eval` and set the
engine's `KING_LAYOUT` (`src/nnue.rs`) to the matching `LAYOUT_*` constant
when embedding (`scripts/netloop.py embed --king-buckets <layout>` does that). `--init` from a one-bucket net copies its 768 rows into
every bucket (start loss equals the one-bucket net's for unmirrored
layouts). Checkpoints store the layout id in the high half of the buckets
header word, so one-bucket checkpoints are unchanged. The `--mirror` data
augmentation still applies (to the board, before buckets are chosen).

### Factoriser (`--factorise`, KB > 1 only)

Training also keeps a shared 768 x H block `F`; every bucket's effective
row is `clamp(W[kb][f] + F[f], +-1.98)` and `F` gets the sum of all
buckets' gradients for its feature, so rarely hit buckets start from (and
keep learning) what every bucket shares. Every exported net (`<out>.eN.bin`
and `<out>`) holds the merged rows: the file format and the engine are
unchanged, and `eval` needs no flag. Checkpoints store `W` and `F` (header
word 1 bit 40 = factorised; `--resume` needs the same flag).

- From scratch: `W = 0`, `F` = the one-bucket init (the same RNG draws as a
  KB=1 net with the same seed), so step 0 equals that KB=1 init.
- `--init` from a one-bucket net: `F` = its rows, `W = 0`. From a net that
  already has all KB buckets: `W` = the net, `F = 0`.

Warm starts are exact for what the layout can express: an unmirrored
layout (kb4) from a `none` net reproduces it exactly, and a mirrored
layout (kb4m, kb8m) from an `m1` net reproduces it exactly. A mirrored
layout cannot reproduce a `none` net: every mirrored bucket serves both a
king on file x (unflipped) and on file 7-x (flipped), so one row would have
to equal the source rows of square s and s ^ 7 at once. It starts as that
net read with mirroring (`eval --king-buckets m1` gives its loss).

### Active-only weight decay (`--wd-active-only`, default on for KB > 1)

AdamW's decay on the bucketed FT rows `W` is applied only to rows whose
feature occurs in the batch, so buckets that are rarely hit are not shrunk
toward zero between visits. `F`, the FT bias and the output layer decay as
before; `--no-wd-active-only` restores plain decay. It only matters with
`--wd > 0` (the default is 0). KB=1 nets keep plain decay and train
bit-for-bit as before.

## Threat inputs (`--threats <none|ep|ad>`, default `none`)

Extra input blocks of 768 rows after the king buckets' rows (so the FT has
`(KB + TB) * 768` rows and the threat weights sit between the bucketed rows
and the FT bias in the file). A threat row uses the base feature index of
its piece (`side*384 + kind*64 + sq` in the perspective's orientation, with
the perspective's mirror flip) but is not king-bucketed and not factorised.
It is active when the piece has the mode's property (kings never):

- `ep` (TB = 1, "en prise"): attacked by an enemy piece of lower value
  (P < N = B < R < Q), or attacked and not defended. About 3.4 active rows
  per position and 1.1 changed per ply on random game positions.
- `ad` (TB = 2): block 0 = attacked by the enemy, block 1 = defended by its
  own side. About 19 active rows, 3.5 changed per ply.

Attacks are plain attack sets with full occupancy (no x-rays), computed
from the 32-byte records with ray walks (`threat_masks`); the definitions are
the engine's `ThreatMode` (`src/nnue.rs`), and both sides test the same
hand-checked positions and agree with the same naive definition.
`--init` from a net without threat rows (and/or one-bucket) starts the threat
rows at 0, so the warm-started net evaluates exactly like its source.
Checkpoint header word 1 carries the threat id at bit 48 (0 = none, so older
checkpoints are unchanged). `eval` takes `--threats` too. The engine runs a
threat net when `THREATS` in `src/nnue.rs` is set to the same mode.
