# Desktop (Ryzen 9600X) setup

Goal: make the 9600X the data + generation box (CPU trainer, projected ~4M pos/s at
12 threads, about 2x the laptop's Metal trainer), and optionally a local datagen box.
The laptop then only runs experiments. Written 2026-11-08 for setup before Wed 11 Nov.

Assumes Linux (Ubuntu 24.04 or similar) or WSL2 on Windows. Under WSL2 keep all data
inside the Linux filesystem (`~/...`), not `/mnt/c`, or I/O will be very slow.

## 1. Toolchain and repo

```bash
sudo apt update && sudo apt install -y build-essential git curl python3 pkg-config
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && . ~/.cargo/env
# GitHub CLI (for CI artifacts and the merge queue)
sudo apt install -y gh && gh auth login            # HTTPS; scope: repo, workflow
git clone https://github.com/dukky/rustfish.git ~/dev/rustfish && cd ~/dev/rustfish
git config user.name "Andreas Holley"; git config user.email "<your email>"
```

Pushing changes under `.github/workflows` needs SSH (`git@github.com:dukky/rustfish.git`);
everything else works over HTTPS.

## 2. Build and check (x86 must use target-cpu=native)

```bash
export RUSTFLAGS="-C target-cpu=native"      # put this in ~/.bashrc
cargo build --release
printf 'perfttest\nselftest 100\nquit\n' | ./target/release/rustfish   # 8/8, selftest ok
python3 scripts/bench.py --engine ./target/release/rustfish --depth 5
cargo build --release --manifest-path tools/nnue-trainer/Cargo.toml
cargo test  --release --manifest-path tools/nnue-trainer/Cargo.toml    # 17 tests
```

## 3. Measure the trainer (first thing to report back)

```bash
mkdir -p ~/rustfish-data && cd ~/rustfish-data
T=~/dev/rustfish/tools/nnue-trainer/target/release/nnue-trainer
# small real data: one CI datagen seed (see section 4), or any converted .bin
for th in 6 12; do
  $T train --data s45.bin --val val.bin --out bench$th.bin --hidden 512 --buckets 8 \
     --king-buckets kb8m --factorise --threads $th --max-batches 3000 2>&1 | grep -E "kernels|pos/s"
done
```

The log must print `kernels avx512`. Expect ~3.5-5M pos/s at 12 threads plain, less with
kb8m + factorise. Also note the free disk (`df -h ~`): a full pool is ~32 GB per 1B
positions, and a pool rebuild needs the new pool's size free (`shuffle --consume`).

## 4. Data: fetch CI seeds and validation sets

All training data is on CI as datagen artifacts (90-day retention). Seeds 14-18 were
generated at half score scale and need x2; seed 7 is quarantined; seeds >= 19 are at
net scale. `scripts/train_sweep.py` implements exactly this (it is what the CI sweep
uses), so prefer it over hand scripts:

```bash
cd ~/dev/rustfish
python3 scripts/train_sweep.py --help        # subcommands: plan / pool / ...
```

Validation sets (must match the laptop's byte for byte; train_sweep builds them):
- `val.bin`: last 1M positions of seed 9.
- `fix/val17x2.bin`: last 1M positions of seed 17, scores x2. Never train on these.

Manual alternative for one seed:

```bash
id=$(gh run list -R dukky/rustfish -w datagen.yml -L 100 --json databaseId,displayTitle \
     -q '.[] | select(.displayTitle|test("seed 45 ")) | .databaseId' | head -1)
gh run download $id -R dukky/rustfish -D dl45
gunzip -c dl45/*/*.gz | $T convert - s45.bin
$T scalecheck --net ~/dev/rustfish/net/nnue.bin --buckets 8 --king-buckets kb8m --data s45.bin --n 200000
# slope ~1.13 = net scale (fine); ~0.57 = half scale (needs x2, see scripts/netloop.py x2_copy)
```

## 5. Train a generation (the current recipe)

Pool = every valid seed (or a recent window, see plan.md disk notes), shuffled once:

```bash
$T shuffle --out pool8.bin <seed files...>          # --consume old.bin to free the old pool
$T train --data pool8.bin --val val.bin --out g8.bin   --hidden 512 --buckets 8 --epochs 10 \
   --lr 1e-3 --lr-end 1e-5 --threads 12
$T train --data pool8.bin --val val.bin --out g8m1.bin --hidden 512 --buckets 8 --king-buckets m1 \
   --init g8.bin --epochs 1 --lr 2e-4 --lr-end 1e-5 --keep-best --threads 12
$T train --data pool8.bin --val val.bin --out g8kb8m.bin --hidden 512 --buckets 8 --king-buckets kb8m \
   --factorise --init g8m1.bin --epochs 4 --lr 2e-4 --lr-end 1e-6 --keep-best --threads 12
```

If the laptop's g7e5 test (5 vs 10 epochs) tied, use `--epochs 5` for g8.

## 6. Validate, embed, queue

```bash
for n in main g8kb8m; do   # main = net/nnue.bin from origin/main
  $T eval --net $n.bin --val val.bin        --buckets 8 --king-buckets kb8m --best-scale | tail -1
  $T eval --net $n.bin --val fix/val17x2.bin --buckets 8 --king-buckets kb8m --best-scale | tail -1
done
# better on both -> embed (builds, runs cargo test/perft/bench, pushes nnue-g8kb8m, records the row)
cd ~/dev/rustfish && python3 scripts/netloop.py embed --name g8kb8m --net ~/rustfish-data/g8kb8m.bin \
   --base origin/main --hidden 512 --buckets 8 --king-buckets kb8m --baseline <main net name> --notes "..."
python3 scripts/queue.py add nnue-g8kb8m:net-gate --front --expect 8
```

The queue (CI) then plays the net gate and merges on H1. Nothing else is needed locally.

## 7. Optional: local datagen when not training

```bash
echo "datagen D2000.txt 1000000 5000 11 2000 0" | ./target/release/rustfish > D2000.log 2>&1 &
$T convert D2000.txt D2000.bin       # when done (or in chunks)
```

Use seeds >= 2000 (CI uses < 1000, the laptop 1000s). Rebuild the binary and start a new
seed whenever main gets a new net. Don't run datagen and training together (they compete
for the same cores).

## 8. Notes

- The laptop's chain scripts (`~/rustfish-data/chain*.sh` on the laptop) are the reference
  for automating 4-6 as one unattended script.
- AVX-512 engine kernels exist on branch `nnue-avx512` (parked: no whole-engine gain at
  512 wide); re-check with `nnue-isa-check.yml` if nets get wider.
- Keep `~/rustfish-data` out of any file indexer / backup that rescans large files.
