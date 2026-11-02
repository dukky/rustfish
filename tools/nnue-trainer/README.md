# nnue-trainer

Trains the `(768 -> 256) x 2 -> 1` SCReLU network that `src/nnue.rs` runs,
from the self-play text format written by the engine's `datagen` command.
Std only, no dependencies, like the engine.

## Build

```bash
cd tools/nnue-trainer
cargo build --release
```

## Data

Training data is the `Datagen` workflow's artifact, gzipped. Rust std cannot
inflate gzip, so decompress first (the workflow's files are plain text inside):

```bash
gh run download <run-id> -n datagen-s1-n5000
gunzip -c datagen-s1-n5000.gz > data.txt
```

One position per line, from White's point of view:

```text
<FEN> | <score_cp> | <result 1.0 | 0.5 | 0.0>
```

## Train

```bash
./target/release/nnue-trainer \
  --data data.txt --out net.bin \
  --epochs 1 --batch 1024 --lr 0.001 --threads 4
```

Then in the engine:

```bash
printf 'setoption name EvalFile value net.bin\nsetoption name UseNNUE value true\n' | ./target/release/rustfish
```

Useful options:

- `--data FILE` (repeatable), `--out FILE`
- `--epochs N`, `--batch N`, `--lr F` (Adam), `--lambda F` (WDL vs score
  blend, default 0.5)
- `--threads N`, `--shuffle-buffer N` (positions held in RAM; default 65536)
- `--max-positions N` per epoch (subsample a large file)
- `--checkpoint FILE --checkpoint-every N` and `--resume FILE`
- `--seed N`, `--log-every N` (optimizer steps)

The shuffle buffer keeps memory flat on 65M+ position files while decorrelating
consecutive samples. `--max-positions` is applied per epoch, so
`--epochs 3 --max-positions 5000000` reads the first 15M positions over three
passes rather than sampling randomly across the file.

## Loss and scale

`loss = lambda * (sigmoid(eval/400) - result)^2 + (1-lambda) * ((eval - score)/400)^2`

The network is trained directly in the runtime's quantized units (accumulator
clamped at `QA = 255`, output scaled by `SCALE / (QA * QB)`), so export is a
plain round to `i16` in the exact `src/nnue.rs` layout: feature weights
`[768][H]` feature-major, feature bias `[H]`, output weights `[2H]` (side to
move first), output bias. There is no separate float-to-quantized rescaling
step, and `H` is fixed at 256 to match the inference code.

Export prints whether the output weights fit the fast `i32` SIMD path
(`|w| <= 127`) or need the wide `i64` scalar path.

## Measured throughput

This 2-core box (3 GB), 3000-node datagen data: **~94k positions/s**
training at `H=256`, batch 512, 2 threads. A 65M-position pass is therefore
~11 minutes of compute here; on a 4-core CI runner expect roughly half that.
Reading/parsing the text file is included in that figure.
