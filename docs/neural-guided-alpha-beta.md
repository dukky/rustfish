# Neural-guided alpha-beta: a CPU-first alternative to lc0

**Status:** proposal for Opus + M3 validation. Not implemented, not queued.
**Date:** 2026-11-06
**Author:** Hermes, from a discussion with dukky.

## Goal

Get out of the "Stockfish feature-checklist" derivative trap without giving up
blitz strength:

- keep alpha-beta/PVS as the search — it is the right algorithm for CPU blitz;
- add a **learned policy** to the existing value net;
- use the policy for **move ordering and selective search**, not as a replacement
  search;
- train it cheaply from labels we already generate, with no lc0-style RL loop.

The target is a genuinely measurable idea ("policy-guided alpha-beta") that can
be ablated, not a claim that MCTS is better.

## What not to do

- **Do not replace alpha-beta with MCTS + NNUE.** lc0 works because it has a
  large CNN policy/value net and GPU batching. A scalar NNUE value has no move
  prior; CPU MCTS is slow and tactically weaker than alpha-beta.
- **Do not clone lc0's architecture.** A small CPU policy/value head is the
  point: cheap inference, cheap training.
- **Do not use the policy to prune before it has been validated.** Ordering is
  a safe first use; pruning is a later experiment.

The Arimaa evidence points the same way: SHARP, the 2015 Challenge winner, was a
traditional alpha-beta searcher with strong move ordering and evaluation;
earlier MCTS attempts in Arimaa were unsuccessful. The lesson is selectivity +
evaluation, not a search-paradigm swap.
(Reference: D. J. Wu, "Designing a Winning Arimaa Program", ICGA Journal 38(1),
2015.)

## Design

### Network

Start from the current value net (or the next king-bucketed version):

- shared trunk: `768 -> H` (H=256, or 512 if the NPS budget allows);
- **value head:** the existing `2H -> 1` SCReLU output (unchanged);
- **policy head:** a small head from the trunk to move logits.

Move encoding, simplest practical version:

- `from * 64 + to` = 4096 logits for normal moves;
- promotions need a piece dimension, so either `4096 * 4` (16k) or a separate
  promotion head; 16k i16 weights is ~32 KB per H=256 head, acceptable;
- mask illegal moves at inference; never allow an illegal policy target.

Keep the policy head shallow (e.g. `2H -> 256 -> 4096+`) so inference stays
cheap. Evaluate the policy **only where it is used** (root, first few plies, or
nodes where ordering matters), not at every leaf.

### Training

Cheapest useful version, no RL:

1. Generate positions from the existing self-play pool.
2. For each position, record a **search-derived policy label**:
   - best move from a shallow search (e.g. depth 8-10), or
   - a soft distribution from the search's root scores / visit counts.
3. Reuse the existing value labels (search score + game result).
4. Train value + policy jointly:
   - value loss: current value objective;
   - policy loss: cross-entropy against the best move / soft target, legal-move
     masked, with label smoothing.
5. First run: policy-only supervision from existing games; no new datagen
   format required if the best move is recovered from search during labeling.

This is much cheaper than lc0: supervised labels from our own search, one small
head, no self-play RL, no GPU-scale training.

### Search integration, in risk order

**P1 — policy ordering only (low risk).**
Use policy scores in `MovePicker` to order quiets/captures after the TT move and
before history. Keep all existing pruning unchanged. Expected gain: fewer nodes
to the best move; small Elo, but clean.

**P2 — policy in LMR/pruning (medium risk).**
Use policy rank/score to adjust LMR reductions or SEE/futility thresholds.
One variable at a time; validate at LTC.

**P3 — root-only hybrid (high risk, optional).**
At the root, use a small PUCT/MCTS or policy argmax to choose among the top-k
moves, then search each with alpha-beta. This is the "lc0-like" flavour without
replacing the main search. Only attempt if P1/P2 show the policy is informative.

## Validation plan

Do not spend CI games until the policy is demonstrably better than the existing
ordering.

1. **Policy accuracy:** top-1/top-3 agreement with the search best move on a
   fixed held-out set; compare against history/killer ordering.
2. **Ordering efficiency:** nodes to find the best move, and node count at fixed
   depth, policy on/off.
3. **NPS cost:** policy head inference overhead at the intended use sites.
4. **Gate:** same-shape value net; `net-gate`/`stc` only after validation loss
   and ordering metrics improve.
5. **Ablation:** value-only baseline vs value+policy ordering vs value+policy
   pruning. One change per run.
6. **LTC check** for anything that passes stc.

## Risks

- Policy head inference may cost more than the ordering gain, especially at
  H=512.
- Search-best-move labels are noisy and may teach the policy to imitate one
  depth, not general ordering.
- Handcrafted ordering (TT, killers, history) is strong; the policy may add only
  a few Elo.
- A root MCTS hybrid can lose tactics; keep alpha-beta below the root.
- "Novel" is not automatic: publish the ablation, including negative results.

## What would make it interesting

A clean, reproducible result:

> "Adding a supervised policy head to a small NNUE and using it only for move
> ordering improves alpha-beta blitz strength by X Elo at Y% NPS cost."

That is a real contribution — different from another Stockfish feature, cheap
to reproduce, and measurable. If it fails, the negative result is still useful
and honest.

## First experiment (smallest useful)

1. Add a policy head to the existing 256-wide net.
2. Label 100k positions with best move from a depth-8 search.
3. Train policy-only for a few epochs; measure top-1/top-3 accuracy.
4. If accuracy is clearly above history ordering, wire it into `MovePicker`
   behind a UCI option and run a short stc gate.
5. Only then consider LMR/pruning or a root hybrid.
