# AGENTS.md

## Project goal
Build a **no-dependency Rust chess engine** that can eventually **beat Stockfish level 17**.

Current engine name: `rustfish`
Language: Rust
Dependencies: none beyond Rust stdlib and project-local scripts using Python stdlib.

---

## High-level working rules
- Prioritize **correctness first**, then **search strength**, then **performance**, then **refactor polish**.
- Every meaningful engine change should be validated with at least some of:
  - compile check
  - perft smoke suite
  - selftest / make-undo invariants
  - quick UCI search smoke test
  - quick gauntlet or benchmark if relevant
- When measuring strength, prefer **frozen/versioned binaries** so results are reproducible.

---

## Repository layout
- `src/main.rs` — remaining core types / board state / entrypoint
- `src/book.rs` — built-in opening book
- `src/eval.rs` — evaluation logic
- `src/movegen.rs` — move generation, attack detection, UCI move parsing/helpers
- `src/search.rs` — search, pruning, move ordering, SEE
- `src/tt.rs` — zobrist hashing + transposition table
- `src/uci.rs` — UCI/perft/selftest handlers
- `scripts/gauntlet.py` — gauntlet runner vs Stockfish or other UCI engines
- `scripts/bench.py` — benchmark runner across fixed positions
- `plan.md` — project roadmap / progress tracking

---

## Core commands

### Compile
Debug:
```bash
cargo check
```

Release:
```bash
cargo build --release
```

---

## UCI smoke test
Quick sanity check that the engine responds correctly:
```bash
printf 'uci\nisready\nposition startpos\ngo depth 4\nquit\n' | cargo run --quiet
```

With release binary:
```bash
printf 'uci\nisready\nposition startpos\ngo depth 4\nquit\n' | ./target/release/rustfish
```

---

## Perft / correctness

### Perft smoke suite
Runs built-in known positions:
```bash
printf 'perfttest\nquit\n' | cargo run --quiet
```

### Single perft
```bash
printf 'position startpos\nperft 4\nquit\n' | cargo run --quiet
```

### Divide
```bash
printf 'position startpos\ndivide 3\nquit\n' | cargo run --quiet
```

### Selftest / make-undo invariant checks
```bash
printf 'selftest 200\nquit\n' | cargo run --quiet
```

---

## Benchmarking
Run the benchmark harness on the release binary:
```bash
python3 scripts/bench.py --engine ./target/release/rustfish --depth 5
```

Notes:
- This reports nodes / time / NPS over a fixed set of positions.
- Use this to catch large performance regressions after search/eval/refactor changes.

---

## Gauntlet / strength testing

### Basic gauntlet vs Stockfish
```bash
python3 scripts/gauntlet.py \
  --stockfish "$(which stockfish)" \
  --rustfish ./target/release/rustfish \
  --games 20 \
  --movetime 50 \
  --max-plies 160 \
  --stockfish-option "UCI_LimitStrength=true" \
  --stockfish-option "UCI_Elo=1800"
```

### Fixed-depth gauntlet
```bash
python3 scripts/gauntlet.py \
  --stockfish "$(which stockfish)" \
  --rustfish ./target/release/rustfish \
  --games 20 \
  --depth 1 \
  --max-plies 160 \
  --stockfish-option "UCI_LimitStrength=true" \
  --stockfish-option "UCI_Elo=1800"
```

### Elo ladder quick sweep
```bash
for E in 1600 1700 1800 1900 2000; do
  echo "=== SF ELO $E ==="
  python3 scripts/gauntlet.py \
    --stockfish "$(which stockfish)" \
    --rustfish ./target/release/rustfish \
    --games 20 \
    --movetime 30 \
    --max-plies 220 \
    --rustfish-option "Hash=256" \
    --stockfish-option "UCI_LimitStrength=true" \
    --stockfish-option "UCI_Elo=$E"
  echo
 done
```

### Better longer gauntlet
```bash
python3 scripts/gauntlet.py \
  --stockfish "$(which stockfish)" \
  --rustfish ./target/release/rustfish \
  --games 200 \
  --movetime 200 \
  --max-plies 220 \
  --rustfish-option "Hash=256" \
  --stockfish-option "UCI_LimitStrength=true" \
  --stockfish-option "UCI_Elo=1800"
```

---

## Frozen binary workflow
Use this when comparing versions so results are not contaminated by rebuilds.

Create a versioned binary:
```bash
VER="rustfish_$(date +%Y%m%d_%H%M%S)" && cargo build --release && mkdir -p bin && cp ./target/release/rustfish "./bin/$VER" && echo "$VER"
```

Run gauntlet against that exact binary:
```bash
python3 scripts/gauntlet.py --stockfish "$(which stockfish)" --rustfish "./bin/$VER" --games 300 --movetime 200 --max-plies 220 --rustfish-option "Hash=256" --stockfish-option "UCI_LimitStrength=true" --stockfish-option "UCI_Elo=1800"
```

---

## Important UCI options supported by rustfish
- `Hash` — TT size in MB
- `Clear Hash`
- `OwnBook` — enable/disable built-in opening book

Example:
```bash
printf 'uci\nsetoption name OwnBook value false\nsetoption name Hash value 256\nisready\nquit\n' | ./target/release/rustfish
```

---

## Validation checklist after engine changes
For search/eval changes:
1. `cargo check`
2. `perfttest`
3. `selftest 100` (or more)
4. UCI smoke (`go depth 4` or `go movetime 100`)
5. quick gauntlet sample

For refactors:
1. `cargo check`
2. `perfttest`
3. `selftest 100+`
4. benchmark smoke
5. quick gauntlet smoke

---

## Measurement guidance
- Small gauntlets are noisy; prefer 100+ games for real comparisons.
- `UCI_LimitStrength` Stockfish Elo is only an approximation.
- Always compare versions under the **same time control**, **same max plies**, and **same Stockfish settings**.
- Prefer versioned binaries for serious comparisons.

---

## Current strategic priorities
1. Strengthen search heuristics without breaking correctness.
2. Improve evaluation stability, especially king safety / pawn structure / endgames.
3. Track practical Elo progress with frozen binaries and gauntlets.
4. Preserve no-dependency constraint.
