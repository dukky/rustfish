# Peer analysis and AI-engine forum reactions, 2026-11-08

**Author:** Hermes (analysis, not an audited fact base)
**Purpose:** record the peer-engine deep dive and the community reaction to
AI-assisted engines, so it can inform rustfish's feature priorities and any
CCRL submission.
**Status:** worth a look, not taken as fact. Ratings are version- and
anchor-specific, architecture notes are mostly from project READMEs rather
than a code audit, and forum reactions are a sample of a split community.

## Snapshot of the peers

Ratings below are CCRL Blitz single-CPU anchors used by the queue, unless
noted. The anchor versions are not always the projects' current releases.

- **Endspiel 2.0.0 — 3508** (805 games), Rust, repo May 2026. From-scratch
  bitboards, search, NNUE inference and UCI. Own layer-stacked,
  king-bucketed net; v2.0 uses a wider 1536 feature transformer and score
  labels from Leela Chess Zero's public T80 data. Trained with Bullet.
  Syzygy up to 7 pieces, MultiPV, Ponder, SMP, books, Chess960, runtime
  AVX-512/AVX2 dispatch, PGO, Pi 5 and Android builds. AI-assisted
  development (Claude Code, later Codex/Grok/OpenCode); GPL-3.0-or-later.
  Its own ABOUT says the remaining gap is data and compute, not new code.
- **Reckless 0.7 — 3491**, Rust, repo since Dec 2022. Bullet trainer.
  0.7 net is `(768 -> 384) x 2 -> 1 x 4`, four output buckets and material
  scaling. 0.8-dev moved to a threat-based, king-bucketed feature set and
  added NUMA/SMP.
- **Koivisto 6 / 9 — 3487 / 3631**, C++, repo since 2020. NNUE since v5
  (2021); own trainer lineage CudAD -> Grapheus; billions of positions.
  v9 dates from Jan 2023 and development largely stopped in 2024; still a
  strong example of search + data maturity.
- **Stash 37 — 3417**, C, repo since 2020, active. HCE, no NNUE; custom
  hand-crafted evaluation and tuning. Shows the still-real ceiling of a
  well-tuned classical eval.
- **Berserk 8 — 3553**, C, repo since 2021, active. NNUE and Bullet;
  mature search. Release 14 is current in 2026; the anchor is an older v8.
- **Altair 7.0.0 — 3579**, C++20, repo since 2022. NNUE from v5.0.9;
  `Europa` net trained on 3.1B FENs including DFRC. Data was bootstrapped
  from its classical evaluation, then recycled in reinforcement-style
  cycles.
- **Carp 3.0.1 — 3525**, Rust, repo since 2022. A didactic engine with a
  simple NNUE and the marlinflow trainer. Clean proof that a simple Rust
  NNUE engine can reach 3525.
- **Viridithas 20 — current Rust leader**, Rust, repo since 2022. The
  anchor's v4.0 (3024) is ancient; v20 shipped in June 2026. NNUE trained
  on its own self-play and continually retuned. Structurally the closest
  long-running Rust analogue to rustfish.
- **Trebuchet — no CCRL anchor**, Rust, by Jonny Reckless with Claude Code.
  King-bucketed horizontal-mirror `768x8 -> 1024 -> 8` SCReLU; a
  known-engine-author experiment in agentic development.

## Where rustfish sits

- Rustfish: Rust, from-scratch bitboards/search/UCI, NNUE
  `(768 x 8 king buckets -> 512) x 2 -> 8 output buckets`, SCReLU, i16,
  embedded net, own std-only trainer, pure self-play (around 1B positions
  at the time of writing). No Syzygy, no Ponder, root-level SMP.
- Measured 3474.3 (95% CI 3456.1-3492.5) on the new 3318-3631 anchor set;
  the 2026-11-08 19:23 ladder on g7kb8m was 3492 [3475, 3510].
- The gap to this peer group is mostly **data volume, data quality, and
  threat features**, not a missing core algorithm. Search is modern:
  PVS, aspiration, TT, NMP, RFP, futility, LMP, LMR (including cut-node),
  IIR, SEE, killers, counter-moves, history, continuation history,
  ProbCut, singular extensions, check extensions, node-fraction TM.

## Forum threads and reactions

### Endspiel (the normal, accepted AI-assisted path)

- CCRL 2.0 thread:
  https://kirill-kryukov.com/chess/discussion-board/viewtopic.php?t=19333
- CCRL 2.1 thread:
  https://kirill-kryukov.com/chess/discussion-board/viewtopic.php?t=19421
- 2.0 announcement: wider 1536 feature transformer, Lc0 T80 score labels,
  +114 self-play vs 1.7, estimated 3450-3500. Around five replies and under
  a hundred views; mostly "sounds good, looking forward to it" and a
  metadata request. No AI controversy. 2.1 is a normal update thread
  (faster NNUE, retuned search, 50-move handling).
- Why it was accepted: original code, own net, GPL, transparent data
  sources, and a conventional UCI/engine submission. AI assistance is
  disclosed in ABOUT.md but is not the focus of the CCRL thread.

### CODA (the flashpoint)

- https://talkchess.com/viewtopic.php?t=86503
- Author announces Coda 0.9.0, every line written by Claude with human
  direction/testing. Initial reaction is positive ("Looks amazing. Strong
  and fast. I love it."), then the thread turns:
  - Andrew Grant (Ethereal author) calls it "fully-automated theft of all
    open-source IP", says it fails licence obligations, and predicts a
    wave of "slop-engines" that will discourage real developers.
  - Werner says the CCRL team decided not to test it.
  - Gabor Szots questions whether AI engines make sense and calls it a
    show-off.
  - Others defend it as an honest experiment: "the agent produced a
    genuinely strong, original engine"; "what matters here is intent".
  - Rebel reports Coda 0.9 scored 63.4% against a 3578 pool, implying
    around 3700: "Welcome to the new future."
- The dispute is about provenance/licensing and gatekeeping, not merely
  about Claude writing code.

### Fable 5.1 (24-hour benchmark)

- https://talkchess.com/viewtopic.php?t=86707
- Steve Maughan's "24 hours from scratch, scored by Elo" benchmark.
  Reaction: "Impressive, and scary, but not unexpected"; questions about
  prompt, token usage, agent setup, and whether the model consulted
  existing source. More constructive than CODA; provenance is the concern,
  not blanket rejection.

### Leaf (EXchess reborn with Claude)

- https://talkchess.com/viewtopic.php?t=86093
- Dan Homan returns after 8-9 years to add NNUE/TDLeaf to EXchess with
  Claude Code. Reaction: "thrilled to see this new vector of
  experimentation." Mostly technical. A balanced comment notes Claude is a
  real time-saver for refactoring, but new/medium changes can contain bugs,
  performance regressions, or require a lot of prompting.

### FunComp (AI-assisted hobby engine)

- https://talkchess.com/viewtopic.php?t=86481&start=70
- Normal release-thread reception. FunComp 2.0 reaches about 3280,
  +245 over 1.0, supports Lazy SMP, and plans NNUE next. The author says it
  is written with Claude Code. No backlash; discussion is about strength
  and testing.

### Triumviratus (clone accusation)

- https://talkchess.com/viewtopic.php?t=86907
- The author addresses claims that Triumviratus is a Stockfish clone
  because of fast progress and AI use. The recurring pattern: rapid
  AI-assisted progress invites suspicion even when the code is original.

### NARC Genesys and Barbarossa (AI as accelerator)

- https://talkchess.com/viewtopic.php?p=999138
- https://talkchess.com/viewtopic.php?p=998615
- Both are "I used Claude Code/Codex to implement and test ideas faster"
  posts; reception is neutral-to-positive and focused on SPRT results and
  strength.

### Trebuchet (agentic experiment)

- https://github.com/reckless-technology/trebuchet
- By Jonny Reckless and Claude Code; king-bucketed 768x8 -> 1024 -> 8
  SCReLU. No large TalkChess thread, but it is the cleanest example of a
  known engine author trying full agentic development.

## What to learn from the peer group

1. **Threat-based NNUE inputs** are the common next feature. Reckless 0.8,
   Stash-related work and Stockfish all moved toward piece-attack/defence
   features. Rustfish's `feat-threat-hist` was search-side (+4.7); the
   net-side threat inputs are the larger version.
2. **Data scale and labels are the biggest gap.** Endspiel uses billions
   of positions and Lc0 T80 labels; Altair uses 3.1B FENs; Koivisto uses
   billions. Rustfish's pure self-play got to 3474; the next 30-50 likely
   needs more and better labels, not another search trick.
3. **A faster trainer** (Bullet, marlinflow, Grapheus, or a GPU path)
   enables more variants per day. The custom trainer is fine but is now a
   throughput constraint.
4. **Simple nets can reach this level.** Carp 3.0.1 at 3525 and Reckless
   0.7 at 3491 with a 384-wide net show that a 512-wide king-bucket net is
   not inherently behind. Complexity is not the missing Elo.
5. **Engineering polish is secondary.** Syzygy, Ponder, MultiPV, SMP and
   books are excellent product features, but at 1-CPU CCRL Blitz they are
   worth only a few Elo each. The main event remains net/data quality.
6. **AI assistance is not disqualifying; provenance is.** Endspiel is
   tested and treated normally. CODA was refused testing amid licence and
   provenance objections. For any rustfish submission: state the AI
   assistance plainly, keep the code original, document the trainer and
   data sources, and do not ship a borrowed net or data without
   attribution.

## Caveats

- Ratings are from CCRL Blitz anchors and are relative to that anchor set
  and date; a project's current release may be stronger than its anchor
  (e.g. Berserk 14 vs anchor 8, Viridithas 20 vs anchor 4.0).
- Architecture descriptions are from READMEs/ABOUT files and release notes,
  not an independent source audit. Treat dimensions and data claims as
  project-reported.
- Forum reactions are a non-random sample, often strongly held; they are
  useful for understanding community norms and submission risk, not for
  estimating strength.
- This note is Hermes's analysis and should be reviewed, challenged, and
  corrected before being treated as fact.
