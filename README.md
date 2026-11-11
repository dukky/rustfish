# rustfish

A chess engine in Rust with no dependencies beyond the standard library.
Stockfish 17 at skill level 17 (the original goal) is passed; the targets are
now places on the CCRL Blitz list (single-CPU engines, best versions; list of
2026-10-31, numbers in `scripts/ccrl-targets.json`):

| Target | CCRL Blitz | Rank today |
|---|---|---|
| Short term | 3400 | ~#131 of 865 (reached 2026-11-06) |
| Mid term: top 100 | 3506 | #100 (list of 2026-11-07); **at the line: ladders 3510, 3508, 3508 (2026-11-09..10)** |
| Long term: top 10 | 3751 | #10 |

rustfish measures **~3508** (three CCRL ladders in a row: 3510 on 2026-11-09,
3508.5 and 3507.6 on 2026-11-09/10, net g7kb8m plus the LMR improvements;
each pentanomial CI is about ±18). That is at the top-100 line; each CI still
overlaps it, so top 100 needs a clear margin from the next net. Before that it
was **3492** (2026-11-08) and **3463** (2026-11-07, net g4).

**Live progress dashboard: https://rustfish-dash.dev.duk.im**: current strength,
the test in flight, the merge queue, recent results, strength over time and the
roadmap. It reads everything live from this repository (`tests/*.jsonl`, the
`queue-state` branch, `plan.md`, `scripts/queue-settings.json`) and GitHub
Actions, so it never needs updating by hand; redeploy only when
`deploy/dashboard/` itself changes (see its README).

| | |
|---|---|
| Strength | **~3508** CCRL Blitz (ladders 2026-11-09..10, g7kb8m; latest 3507.6, CI 3490–3525); live on the dashboard, history in `tests/ladder-log.jsonl` |
| Roadmap | [`plan.md`](plan.md) |
| How changes are tested and merged | [`docs/development.md`](docs/development.md) |
| Working on the code | [`AGENTS.md`](AGENTS.md), and [`docs/handoff.md`](docs/handoff.md) for a new agent |
| Engine | NNUE (768 inputs x 8 king buckets -> 512) x 2 -> 8 output buckets, SCReLU; PVS alpha-beta search |
| NNUE | [`docs/nnue.md`](docs/nnue.md), experiments in [`docs/nnue-experiments.md`](docs/nnue-experiments.md) |

## Build and run

```bash
cargo build --release
./target/release/rustfish    # UCI engine
printf 'bench\nquit\n' | ./target/release/rustfish
```
