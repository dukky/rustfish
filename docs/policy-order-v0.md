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
