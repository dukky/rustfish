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
- `scripts/freeze.py` — build + freeze a versioned binary with metadata and bench output
- `scripts/sweep.py` — run candidate variants against a fixed baseline with gauntlet + SPRT
- `scripts/test_change.py` — one-command validation for correctness, bench, and A/B smoke
- `scripts/rating_pool.py` — run a local round-robin rating pool across fixed engine binaries
- `scripts/openings.txt` — reusable opening suite for diversified gauntlets
- `plan.md` — project roadmap / progress tracking
- `post-2100-revised.md` — current strategic direction (what was tried, what was rejected, what to do next); read this before picking new work
- `docs/archive/` — superseded planning/review notes kept for history; do **not** treat their recommendations as current (several of their priorities were later tested and rejected — see `post-2100-revised.md`)

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

## One-command change validation
Use this as the default “test the current change” workflow:
```bash
python3 scripts/test_change.py \
  --baseline ./bin/rustfish_BASELINE \
  --games 24 \
  --movetime 20 \
  --parallel 4
```

This runs:
- `cargo check`
- `perfttest`
- `selftest 100`
- release build
- release UCI smoke
- fixed benchmark
- optional A/B smoke against `--baseline`

Notes:
- `--baseline` is optional, but should usually point at the current frozen champion.
- Logs from the A/B smoke are written under `tmp/test-change/`.

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
  --openings-file ./scripts/openings.txt \
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
  --openings-file ./scripts/openings.txt \
  --stockfish-option "UCI_LimitStrength=true" \
  --stockfish-option "UCI_Elo=1800"
```

### Generic A/B gauntlet between frozen versions
```bash
python3 scripts/gauntlet.py \
  --engine-a ./target/release/rustfish \
  --engine-b ./bin/rustfish_BASELINE \
  --engine-a-name candidate \
  --engine-b-name baseline \
  --engine-a-option "OwnBook=false" \
  --engine-b-option "OwnBook=false" \
  --engine-a-option "Hash=256" \
  --engine-b-option "Hash=256" \
  --games 48 \
  --movetime 30 \
  --max-plies 220 \
  --parallel 4 \
  --openings-file ./scripts/openings.txt \
  --jsonl ./tmp/ab.jsonl
```

### Local rating pool across frozen binaries
```bash
python3 scripts/rating_pool.py \
  --engine current=./bin/rustfish_9b0b346_20260320_225723 \
  --engine f4143cd=./bin/rustfish_f4143cd_20260320_220450 \
  --engine old=./bin/rustfish_99bdd5d_20260320_205751 \
  --engine sf=/opt/homebrew/bin/stockfish \
  --engine-option current:OwnBook=false \
  --engine-option f4143cd:OwnBook=false \
  --engine-option old:OwnBook=false \
  --common-option Hash=256 \
  --games 48 \
  --movetime 100 \
  --parallel 2 \
  --anchor current \
  --anchor-rating 0
```

Notes:
- This gives a clearer local ladder than `UCI_LimitStrength`, because every opponent is a fixed binary.
- Treat the output as a local pool rating, not a public Elo list.
- Keep the same openings, hash, thread count, and time control across runs.

### CCRL-scale rating in GitHub Actions (preferred absolute strength number)
The `CCRL ladder` workflow (`.github/workflows/ccrl-ladder.yml`) builds rustfish and
the CCRL-rated anchors in `scripts/anchors.json`, plays them with fastchess at 120+1
on GitHub runners, and fits one rating with `scripts/ccrl_ladder.py`. Nothing runs
on this laptop:
```bash
gh workflow run ccrl-ladder.yml -f games_per_anchor=96 -f tc=120+1
gh run watch   # result lands in the run's job summary + ccrl-ladder-results artifact
```
Notes:
- Add anchors (e.g. stronger ones as rustfish improves) by editing `scripts/anchors.json`
  with ratings from the CCRL Blitz complete list.
- Use `-f anchors=a,b,c` to rerun a subset; `-f games_per_anchor=32` for a quick look.
- This is the number to quote; `UCI_Elo` ladders are only a rough cross-check.

### Elo ladder quick sweep vs Stockfish
```bash
for E in 1600 1700 1800 1900 2000; do
  echo "=== SF ELO $E ==="
  python3 scripts/gauntlet.py \
    --stockfish "$(which stockfish)" \
    --rustfish ./target/release/rustfish \
    --games 20 \
    --movetime 30 \
    --max-plies 220 \
    --openings-file ./scripts/openings.txt \
    --rustfish-option "Hash=256" \
    --stockfish-option "UCI_LimitStrength=true" \
    --stockfish-option "UCI_Elo=$E"
  echo
 done
```

### Longer gauntlet with JSONL logging
```bash
python3 scripts/gauntlet.py \
  --stockfish "$(which stockfish)" \
  --rustfish ./target/release/rustfish \
  --games 200 \
  --movetime 200 \
  --max-plies 220 \
  --openings-file ./scripts/openings.txt \
  --rustfish-option "Hash=256" \
  --stockfish-option "UCI_LimitStrength=true" \
  --stockfish-option "UCI_Elo=1800" \
  --jsonl ./tmp/stockfish-1800.jsonl
```

### SPRT early-stop A/B test
```bash
python3 scripts/gauntlet.py \
  --engine-a ./target/release/rustfish \
  --engine-b ./bin/rustfish_BASELINE \
  --engine-a-name candidate \
  --engine-b-name baseline \
  --engine-a-option "OwnBook=false" \
  --engine-b-option "OwnBook=false" \
  --engine-a-option "Hash=256" \
  --engine-b-option "Hash=256" \
  --games 96 \
  --movetime 30 \
  --max-plies 220 \
  --openings-file ./scripts/openings.txt \
  --sprt-h0-elo -5 \
  --sprt-h1-elo 5 \
  --sprt-min-games 24 \
  --jsonl ./tmp/sprt.jsonl
```

Notes:
- `gauntlet.py` supports both `--stockfish`/`--rustfish` mode and generic `--engine-a`/`--engine-b` mode.
- Use `--openings-file ./scripts/openings.txt` for diversified tests instead of repeated `startpos`.
- Use `--jsonl` when you want machine-readable logs.
- When SPRT is enabled, the runner forces sequential execution for correct early stopping.

---

## Variant sweeps
Use `sweep.py` to test multiple candidate option sets against one frozen baseline:
```bash
python3 scripts/sweep.py \
  --baseline ./bin/rustfish_BASELINE \
  --candidate ./target/release/rustfish \
  --baseline-option "OwnBook=false" \
  --candidate-option "OwnBook=false" \
  --baseline-option "Hash=256" \
  --candidate-option "Hash=256" \
  --variant normal:BookStyle=normal \
  --variant grinder:BookStyle=grinder \
  --variant chaos:BookStyle=chaos \
  --games 48 \
  --movetime 30 \
  --openings-file ./scripts/openings.txt \
  --sprt-h0-elo -5 \
  --sprt-h1-elo 5
```

---

## Frozen binary workflow
Use this when comparing versions so results are not contaminated by rebuilds.

Create a versioned binary with metadata and fixed bench output:
```bash
python3 scripts/freeze.py --note "describe the accepted change"
```

This writes:
- a frozen binary under `bin/`
- a sidecar JSON metadata file with git info and bench output

Run gauntlet against that exact binary:
```bash
python3 scripts/gauntlet.py \
  --engine-a ./target/release/rustfish \
  --engine-b ./bin/rustfish_BASELINE \
  --engine-a-name candidate \
  --engine-b-name baseline \
  --engine-a-option "OwnBook=false" \
  --engine-b-option "OwnBook=false" \
  --engine-a-option "Hash=256" \
  --engine-b-option "Hash=256" \
  --games 48 \
  --movetime 30 \
  --max-plies 220 \
  --parallel 4 \
  --openings-file ./scripts/openings.txt
```

---

## Important UCI options supported by rustfish
- `Hash` — TT size in MB
- `Clear Hash`
- `OwnBook` — enable/disable built-in opening book
- `BookStyle` — opening-book personality: `normal`, `grinder`, `chaos`

Example:
```bash
printf 'uci\nsetoption name OwnBook value false\nsetoption name BookStyle value grinder\nsetoption name Hash value 256\nisready\nquit\n' | ./target/release/rustfish
```

---

## Profiling on macOS
The release profile sets `strip = true`, so a plain build cannot be profiled.
Recipe:
```bash
CARGO_PROFILE_RELEASE_STRIP=false CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release
printf 'position fen <busy middlegame FEN>\ngo movetime 10000\nquit\n' | ./target/release/rustfish > /dev/null 2>&1 &
sample "$(pgrep -n rustfish)" 7 -file tmp/profile.txt
```
Notes:
- Attach by PID (`pgrep`), not by name; name lookup often fails.
- `lto = true` + `codegen-units = 1` inline almost everything into
  `negamax`/`quiescence`, so function-level self-time is misleading —
  aggregate by the source `file.rs:line` suffixes in the sample output
  instead, and treat hot *lines* inside big functions as inlined callees.
- Expect underattribution of small, widely-inlined helpers
  (`is_square_attacked` profiled at 4% but was worth +26% NPS when
  rewritten). Confirm any speedup with interleaved candidate/baseline
  bench runs, never single before/after numbers (fanless M3 Air drifts).
- Fixed-depth bench node counts are the acceptance gate for speed work:
  identical nodes = search behavior unchanged, NPS delta is the result.
- On x86-64 (Ryzen desktop) always build with
  `RUSTFLAGS="-C target-cpu=native"`: the default x86-64 baseline lacks
  POPCNT/BMI, so the bitboard hot loops fall back to libcalls without it.
  Measured no effect on Apple Silicon (defaults are already modern), so
  M3-frozen binaries stay flag-free; rebuild per machine for comparisons.

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
- For engine-vs-engine A/B, also keep the **same openings file**, **same hash**, and **same book settings**.
- Prefer frozen binary A/B tests for acceptance decisions, then use Stockfish as a secondary external check.
- Use `scripts/test_change.py` for the default local loop, and `scripts/sweep.py` when screening multiple variants.
- Prefer versioned binaries for serious comparisons.

---

## Current strategic priorities
1. Strengthen search heuristics without breaking correctness.
2. Improve evaluation stability, especially king safety / pawn structure / endgames.
3. Track practical Elo progress with frozen binaries and gauntlets.
4. Preserve no-dependency constraint.
