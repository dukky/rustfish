# rustfish

A chess engine in Rust with no dependencies beyond the standard library, aiming
to beat Stockfish 17 at skill level 17.

**Live progress dashboard: https://rustfish-dash.dev.duk.im** — current strength,
the test in flight, the merge queue, recent results, strength over time and the
roadmap. It reads everything live from this repository (`tests/*.jsonl`, the
`queue-state` branch, `plan.md`, `scripts/queue-settings.json`) and GitHub
Actions, so it never needs updating by hand; redeploy only when
`deploy/dashboard/` itself changes (see its README).

| | |
|---|---|
| Strength | **~3174 CCRL Blitz** (95% CI 3151-3198) on the internal 120+1 anchor ladder, 1440 games, 0 time losses (2026-11-04). Official CCRL rating pending submission. |
| Binaries | [GitHub releases](https://github.com/dukky/rustfish/releases/latest): Windows x86-64 (`v3` AVX2 and `v2` SSE4.2), Linux x86-64 (`v3`/`v2`), macOS arm64. |
| Roadmap | [`plan.md`](plan.md) |
| How changes are tested and merged | [`docs/development.md`](docs/development.md) |
| Working on the code | [`AGENTS.md`](AGENTS.md), and [`docs/handoff.md`](docs/handoff.md) for a new agent |
| NNUE (in progress) | [`docs/nnue.md`](docs/nnue.md) |
| CCRL submission | [`docs/ccrl-submission.md`](docs/ccrl-submission.md) |

## Playing

rustfish is a normal UCI engine. Point a GUI (Cute Chess, Arena, Banksia,
Nalwald, …) at the binary and it will play. No opening book or network file is
needed: the NNUE net is compiled into the executable, and the built-in book is
**off by default** (`OwnBook=false`), so the GUI's own openings are used.

```
uci
isready
ucinewgame
position startpos moves e2e4 e7e5
go wtime 120000 btime 120000 winc 1000 binc 1000
```

The `bench` command searches a fixed set of positions and prints a node count;
that count is the build signature and is recorded in each release's notes.

### UCI options

| Option | Default | Notes |
|---|---|---|
| `Hash` | 64 MB | transposition table size (1-2048 MB) |
| `Threads` | 1 | CCRL lists are single-CPU |
| `Move Overhead` | 10 ms | safety margin subtracted from the clock |
| `OwnBook` | **false** | built-in opening book; leave off under a GUI book |
| `BookStyle`, `BookFile` | `normal`, empty | only used when `OwnBook=true` |
| `UseNNUE`, `EvalFile` | `true`, empty | embedded net is used by default |
| `Clear Hash` | — | button |
| `FutilityMargin*`, `Lmr*`, `Lmp*`, `Null*`, `ReverseFutility*` | tuned | SPSA tuning knobs; leave at defaults |

## Build from source

Rust stable (2021 edition), no external crates:

```bash
cargo build --release
./target/release/rustfish    # UCI engine on stdin/stdout
printf 'bench\nquit\n' | ./target/release/rustfish
```

Run the test suite with `cargo test --release` and the built-in checks with
`printf 'perfttest\nselftest 300\nquit\n' | ./target/release/rustfish`.

## Licence

Apache License 2.0 — see [`LICENSE`](LICENSE).
