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
