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

Needs one converted seed, `s45.bin` (fetch it with the manual commands in section 4 first).
The speed test doesn't need the real validation sets: a throwaway slice of the seed works.

```bash
mkdir -p ~/rustfish-data && cd ~/rustfish-data      # wherever s45.bin is
T=~/dev/rustfish/tools/nnue-trainer/target/release/nnue-trainer
tail -c 32000000 s45.bin > valb.bin        # last 1M positions; overlaps training: speed test only
for th in 6 12; do
  echo "== $th threads, plain"
  $T train --data s45.bin --val valb.bin --out b$th.bin --hidden 512 --buckets 8 \
     --threads $th --max-batches 1500 2>&1 | grep -E "kernels|pos/s"
  echo "== $th threads, kb8m + factorise"
  $T train --data s45.bin --val valb.bin --out bk$th.bin --hidden 512 --buckets 8 \
     --king-buckets kb8m --factorise --threads $th --max-batches 1500 2>&1 | grep -E "kernels|pos/s"
done
```

`--max-batches 1500` is ~25M positions (fits in one ~31M seed), a minute or two per run.
Report the four pos/s numbers.

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
gunzip -c dl45/*/*.gz | $T convert - s45.bin   # NOTE: doubles the data (shard + merged artifacts); fine for the speed test only, use dl() in the long test
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
S=2004   # a NEW seed for every run: the file name and the seed both come from $S
ls D$S.* 2>/dev/null && echo "seed $S already used" || \
  { echo "datagen D$S.txt 1000000 5000 11 $S 0" | ./target/release/rustfish > D$S.log 2>&1 & }
$T convert D$S.txt D$S.bin       # when done (or in chunks)
```

Use seeds >= 2000 (CI uses < 1000, the laptop 1000s), and never reuse one: the same
seed replays the same games. (D2002 and D2003 were run with seed 2001 by mistake and
are copies of D2001; 2002 and 2003 count as used.) Rebuild the binary and start a new
seed whenever main gets a new net. Don't run datagen and training together (they compete
for the same cores).

## 8. Notes

- The laptop's chain scripts (`~/rustfish-data/chain*.sh` on the laptop) are the reference
  for automating 4-6 as one unattended script.
- AVX-512 engine kernels exist on branch `nnue-avx512` (parked: no whole-engine gain at
  512 wide); re-check with `nnue-isa-check.yml` if nets get wider.
- Keep `~/rustfish-data` out of any file indexer / backup that rescans large files.

## Measured (2026-11-08, 9600X, s45.bin, 1500 batches)

| config | 6 threads | 12 threads | M3 Metal |
|---|---|---|---|
| plain 512x8 | 3.94M pos/s | **4.08M** | 1.79M |
| kb8m + factorise | 1.83M | **1.98M** | 1.32M |

Identical losses at 6 and 12 threads (deterministic). SMT adds only 4-8%: the trainer is
memory-bound on 6 cores. A generation on ~1.7B positions (g 10 epochs + m1 + kb8m 4 epochs)
takes ~2.2 h here vs ~4.3 h on the laptop, so the desktop should run generations. The CPU
kb8m path is relatively slower (2x below plain vs 1.35x on Metal): a target for optimisation.

## Remote access: SSH over Tailscale

```bash
# desktop (Ubuntu / WSL2)
sudo apt install -y openssh-server && sudo systemctl enable --now ssh
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --ssh            # log in with the same account as the laptop
tailscale ip -4                    # note the 100.x address; MagicDNS name = machine name
# laptop (macOS): install Tailscale from the App Store, log in, then
ssh <user>@<desktop-name>          # or tailscale ssh <user>@<desktop-name>
```

- `--ssh` lets Tailscale handle auth (no keys needed); otherwise add the laptop's
  `~/.ssh/id_ed25519.pub` to the desktop's `~/.ssh/authorized_keys`.
- WSL2: run Tailscale inside WSL (above) or on Windows with SSH into WSL via
  `wsl` port forwarding; inside-WSL is simpler. Keep the PC from sleeping (Windows
  power settings: sleep never on AC).
- Long jobs: run them in `tmux` (`sudo apt install tmux; tmux new -s train`) so they
  survive disconnects; reattach with `tmux attach -t train`.
- Copying nets/logs: `scp <user>@<desktop-name>:~/rustfish-data/g8kb8m.bin .`

## A useful few-hour test: a recent-data net (gen 8 candidate)

Answers an open question for Wednesday (plan.md disk item 2): does a net trained only on
recent data (CI seeds 19+, from the NNUE-era engines) match one trained on everything?
It also exercises the whole pipeline on the desktop. ~1B positions, ~3-4 h, ~60 GB disk.

```bash
cd ~/rustfish-data; T=~/dev/rustfish/tools/nnue-trainer/target/release/nnue-trainer
dl() {  # dl <seed> <out.bin>: the merged artifact only (runs also upload per-shard artifacts)
  id=$(gh run list -R dukky/rustfish -w datagen.yml -L 300 --json databaseId,displayTitle,conclusion \
       -q ".[] | select(.conclusion==\"success\" and (.displayTitle|test(\"seed $1 \"))) | .databaseId" | head -1)
  [ -z "$id" ] && return 1
  name=$(gh api repos/dukky/rustfish/actions/runs/$id/artifacts --jq '.artifacts[].name' | grep -E "^datagen-s$1-n[0-9]+$" | head -1)
  [ -z "$name" ] && return 1
  gh run download $id -R dukky/rustfish -n $name -D dl$1 && gunzip -c dl$1/*.gz | $T convert - $2; rm -rf dl$1
}
# 1. seeds 19-60 (net scale, no x2); ~1.15B positions, ~37 GB. Fewer seeds if disk is short.
for s in $(seq 19 60); do [ -f s$s.bin ] || dl $s s$s.bin; done
# 2. validation sets (must match the laptop): val.bin = last 1M of seed 9; val17x2 = last 1M of seed 17, x2
dl 9 raw9.bin && tail -c 32000000 raw9.bin > val.bin && rm raw9.bin
dl 17 raw17.bin && tail -c 32000000 raw17.bin > val17.bin && rm raw17.bin
python3 - <<'PY'
import array
a = array.array('h'); a.frombytes(open('val17.bin', 'rb').read())
for i in range(12, len(a), 16):            # score = i16 #12 of each 32-byte record
    a[i] = max(-32768, min(32767, 2 * a[i]))
open('val17x2.bin', 'wb').write(a.tobytes())
PY
rm val17.bin
sha256sum val.bin val17x2.bin    # must be 7d2d9157...e3af85dd and abe51b80...8e7cc404b1
df -h ~                          # shuffle needs ~the pool size free again (~37 GB for 19-60)
# 3. pool (exclude nothing else: seeds >= 19 never contain the val tails), then the recipe
$T shuffle --out poolR.bin s{19..60}.bin 2>/dev/null || $T shuffle --out poolR.bin s*.bin
$T train --data poolR.bin --val val.bin --out gR.bin --hidden 512 --buckets 8 --epochs 10 --lr 1e-3 --lr-end 1e-5 --threads 12
$T train --data poolR.bin --val val.bin --out gRm1.bin --hidden 512 --buckets 8 --king-buckets m1 --init gR.bin --epochs 1 --lr 2e-4 --lr-end 1e-5 --keep-best --threads 12
$T train --data poolR.bin --val val.bin --out gRkb8m.bin --hidden 512 --buckets 8 --king-buckets kb8m --factorise --init gRm1.bin --epochs 4 --lr 2e-4 --lr-end 1e-6 --keep-best --threads 12
# 4. compare with main's net (best-scale; lower is better on both = candidate)
git -C ~/dev/rustfish pull -q && cp ~/dev/rustfish/net/nnue.bin main.bin
for n in main gRkb8m; do for v in val.bin val17x2.bin; do
  echo "$n $v $($T eval --net $n.bin --val $v --buckets 8 --king-buckets kb8m --best-scale | tail -1)"; done; done
```

If `shuffle` with the brace list fails because some seeds are missing, list the files
that exist (`ls s*.bin`). If gRkb8m wins on both sets it can go straight to section 6
(embed + net-gate); either way the numbers decide whether gen 8 drops the old data.
