# Policy-order v0 (Phase 0/1) — status note

This is a starting point for the learned move-ordering experiment, not a
candidate and not queued.

- **Branch:** `feat-policy-order-v0`
- **Code:** `tools/policy/`
- **What exists:** PGN position sampling, deep-search labelling, per-move
  feature extraction, pairwise linear ranker training, and a documented
  Phase-1 integration plan.
- **What does not exist:** engine integration (`PolicyOrder`), a trained
  model in the binary, or any strength claim.

Initial smoke test: 200 positions, 50k-node labels, 74 held-out positions.
Top-1 move accuracy **13.5%** vs **12.2%** for a simple capture/check/centre
heuristic; mean rank **14.3**. Weak but above chance. The next step is live
search features (SEE, history, killers, TT, eval delta) and a small MLP,
then a nodes-to-depth gate — not more linear-model tuning.

See `tools/policy/README.md` for commands, data format, and the integration
plan. Do not copy Clover or Lc0 source into the repo; the prototype is
independent and trained on our own labels.

## NPS budget and training-cost recommendation (2026-11-24)

At the current ~1.5M NPS CI number, **2% of NPS is ~13 ns per node**. With
~30 candidate moves per node that is **~0.4 ns per move**, so a neural policy
cannot run at every node.

Practical envelope:

| Where it runs | Share of nodes | Budget per selected node | Feasible model |
|---|---:|---:|---|
| Every node | 100% | ~13 ns | only a linear/tiny arithmetic scorer; no MLP per move |
| Ambiguous/quiet nodes | ~5% | ~267 ns | tiny MLP, roughly 18 -> 32 -> 1 (~1k params) |
| High-depth/PV nodes | ~1% | ~1.3 us | small MLP, 64-128 hidden (~10k-100k params) |
| Root only | negligible | milliseconds | 1-5M params possible, but only guides a few moves |

**Recommendation:** start with a tiny linear/tiny-MLP scorer on **1-5% of
nodes**, gated by "no TT move / depth >= 4-6 / many quiets." Do not start
with a large policy net; the NPS budget will consume the ordering gain.

Training cost is dominated by **label generation**, not the model:

- NNUE labels come from self-play results; policy labels need deep-search
  best moves.
- At 10M nodes/s: 1M positions at 50k nodes is ~1.4 h of search; 10M
  positions is ~14 h; 1M-node labels are 10x that.
- Add feature extraction, dedup, held-out splits, and iteration. A small MLP
  is easier to train than the 1024-wide NNUE; the deep-search label pipeline
  is the real project.
- A transformer/LLM policy is not justified at this compute scale.

## Follow-up experiment (997 positions)

I expanded to **997 labelled positions** from the ladder PGNs (50k-node
labels, game-level split). Neither the linear ranker nor a 32/64-hidden MLP
improved:

- linear: top-1 **5.5%**, mean rank 16.9
- MLP 64: top-1 **10.9%**, mean rank 14.5 (diverged with lr 0.01; lower-lr
  run was worse)
- 200-position linear smoke test: **13.5%** (small-sample variance; do not
  treat it as the model's real quality)

The conclusion is not “more data will fix it.” The offline feature set is
missing the live signals that make ordering work: history, killers, TT move,
SEE, static-eval delta and threat flags. The next iteration must extract
those from the search, not add parameters to this model.

## Handover

**Status:** Phase 0/1 scaffold complete; no engine integration; no queue
candidate. Branch `feat-policy-order-v0` contains the pipeline, NPS budget,
and this note. It is being queued as a **docs/tools speedup** so it lands on
main and Claude can pick it up.

**Next steps for Claude:**

1. Add live-search features to the labeler: SEE, history, killer/counter,
   TT move, static-eval delta, threat flags.
2. Generate 10k–100k labels at a deeper fixed-node teacher; keep game-level
   splits.
3. Train a small MLP only after the feature set changes; do not tune the
   current 18-feature model.
4. Gate on nodes-to-depth at equal tactical quality, then fixed-node vs
   main/Clover, then LTC. Start at 1–5% of nodes; no every-node inference.
5. If ordering shows a signal, implement `PolicyOrder` in `MovePicker`;
   policy must never prune before ordering is proven.

**Resources:** use the 9600X for deep-search label generation (not datagen);
use the 1080 Ti only for model training. AVX-512 is for local/Lichess; CCRL
needs an AVX2 path.

Do not copy Clover or Lc0 source; this is an independent feature scorer.


