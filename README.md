# rustfish

A chess engine in Rust with no dependencies beyond the standard library, aiming
to beat Stockfish 17 at skill level 17.

**Live progress dashboard: https://rustfish-dash.dev.duk.im**: current strength,
the test in flight, the merge queue, recent results, strength over time and the
roadmap. It reads everything live from this repository (`tests/*.jsonl`, the
`queue-state` branch, `plan.md`, `scripts/queue-settings.json`) and GitHub
Actions, so it never needs updating by hand; redeploy only when
`deploy/dashboard/` itself changes (see its README).

| | |
|---|---|
| Strength | see the dashboard; history in `tests/ladder-log.jsonl` (CCRL Blitz scale) |
| Roadmap | [`plan.md`](plan.md) |
| How changes are tested and merged | [`docs/development.md`](docs/development.md) |
| Working on the code | [`AGENTS.md`](AGENTS.md), and [`docs/handoff.md`](docs/handoff.md) for a new agent |
| NNUE (in progress) | [`docs/nnue.md`](docs/nnue.md) |

## Build and run

```bash
cargo build --release
./target/release/rustfish    # UCI engine
printf 'bench\nquit\n' | ./target/release/rustfish
```
