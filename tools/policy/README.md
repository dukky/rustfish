# Policy order v0 — Phase 0/1 prototype

Status: **Phase 0/1 scaffold**, not an engine candidate and not queued.
Owner: Hermes, 2026-11-24. This is a starting point for Claude when it
picks the search project back up.

## Goal

Add a small **learned move scorer** that guides alpha-beta move ordering
(and later reductions), without replacing NNUE or adding searchless play.
The first milestone is **fewer nodes at equal tactical quality**, not Elo.

## Why this exists

The engine's search is at a local maximum: local pruning/ordering tweaks
exchange nodes for quality at roughly 1:1. The peer evidence says the gap to
Clover is node economy, not eval shape. A learned policy prior is a
different policy class; the first cheap version is a move-feature scorer
used in `MovePicker`, not a transformer and not MCTS.

## Pipeline (Phase 0)

```bash
python3 -m venv .venv-policy
.venv-policy/bin/pip install -r tools/policy/requirements.txt

# 1. Sample positions from PGNs. One JSONL row per sampled position.
.venv-policy/bin/python tools/policy/extract_positions.py \
  --pgn /path/to/games/*.pgn \
  --out /tmp/policy-positions.jsonl \
  --every 20 --max 1000 --min-ply 8 --max-ply 80

# 2. Label each position with a deep fixed-node search.
#    Emits per-move features and the best move from the engine.
.venv-policy/bin/python tools/policy/label_positions.py \
  --engine ./target/release/rustfish \
  --positions /tmp/policy-positions.jsonl \
  --out /tmp/policy-data.jsonl \
  --nodes 50000

# 3. Train a linear pairwise ranker and report top-1 / mean-rank metrics.
.venv-policy/bin/python tools/policy/train_ranker.py \
  --data /tmp/policy-data.jsonl \
  --out /tmp/policy-model.json
```

The scripts are deliberately dependency-light: `python-chess` for PGN/legal
moves, `numpy` for the model. No GPU and no PyTorch are required for v0.

## Data format

Each line of `policy-data.jsonl`:

```json
{
  "game_id": "game-7",
  "ply": 24,
  "fen": "...",
  "best": "e2e4",
  "score_cp": 31,
  "features": ["from_file", "from_rank", "..."],
  "moves": [
    {"uci": "e2e4", "label": 1, "x": [4, 1, 4, 3, ...]},
    {"uci": "g1f3", "label": 0, "x": [6, 0, 5, 2, ...]}
  ]
}
```

The feature vector is intentionally small and explainable:

- from/to file/rank, piece type, captured value, promotion value;
- capture / check / castle / en-passant flags;
- to-center and from-center, relative rank;
- attacked/defended destination flags;
- quiet / pawn-push flags.

Later versions can add SEE, history, killer/counter, TT-move, static-eval
delta and threat flags; those are not in v0 because they depend on live
search state that offline python-chess does not have.

## Phase 1 model

`train_ranker.py` fits a pairwise logistic ranker: for each position the
engine's best move is the positive, up to 12 legal alternatives are sampled
as negatives. Features are standardised from the training split. Evaluation
is grouped by game, so positions from the same game never leak across
splits. Reported metrics:

- top-1 accuracy: best move scores highest;
- mean rank of the engine's best move;
- a simple heuristic baseline (capture value + check + centre).

The model file is JSON (`mean`, `std`, `weights`, `features`, metrics).
It is not loaded by the engine yet.

## Phase 1 integration (next step)

Planned, not implemented here:

1. Add `PolicyOrder` option and a JSON reader under `src/` (or a tiny
   generated Rust table), default off.
2. In `MovePicker`, compute the feature vector for each candidate move and
   add `lambda * policy_score` to the existing move score. Keep captures,
   killers and SEE ordering intact.
3. Start at the root and high-depth nodes only if inference is measurable;
   otherwise all non-qsearch nodes.
4. Gate at fixed nodes vs main and Clover, then LTC. No STC-only merge.

Accept only if nodes-to-depth drops at equal tactical quality and NPS cost
stays under ~2–3%. Kill if the model only learns opening/book patterns or
the inference cost eats the node gain.

## Non-goals

- No searchless play.
- No transformer / LLM policy in v0.
- No policy-driven pruning until ordering is proven.
- No source copying from Clover or Lc0; this is an independent feature
  scorer trained on our own search labels.

## Initial results

Run on 200 positions from the local ladder PGNs, labels from a 50k-node
rustfish search (200 positions, 126 train / 74 test, 1,474 training pairs):

- top-1 accuracy: **13.5%**
- heuristic baseline top-1: **12.2%**
- mean rank of the engine's best move: **14.3**

That is a weak signal: the model is learning something above the heuristic
but nowhere near useful for move ordering. The pipeline is the deliverable;
the model is not. This is expected for a first 18-feature linear scorer on
200 positions.

Do not quote these as engine strength. They only show that the pipeline
runs and that the model is learning above chance.

Next model steps, in order:

1. Add live-search features that the offline extractor lacks: SEE, history,
   killer/counter, TT move, static-eval delta, threat flags.
2. Move from linear to a small MLP (2x64) and train on 10k-100k positions.
3. Replace top-1 with the metric that matters: nodes-to-depth at fixed
   tactical quality when the score is used as an ordering bonus.
4. Only then implement `PolicyOrder` in `MovePicker`.

The current scripts are a clean starting point, not a candidate.

