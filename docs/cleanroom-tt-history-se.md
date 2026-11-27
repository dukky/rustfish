# Clean-room spec: TT replacement, quiet history, singular extensions

Written 27 Nov 2026 for the owner's decision of the same day: the two
remaining items of the 26 Nov provenance audit (plan.md, "Provenance audit")
are rewritten clean-room before release.

- **(A)** transposition-table store / replacement policy (`src/tt.rs`);
- **(B1)** the quiet-move history table, its bonus and its gravity update
  (`src/search.rs`, `src/main.rs`);
- **(B2)** the singular-extension / multi-cut block inside the move loop of
  `negamax_body` (`src/search.rs`).

## Method

The author of this spec and of the new code did not read the flagged
function bodies (the TT `probe`/`store` bodies and the two tt.rs unit tests
that pin the replacement policy; `threat_slot`, `history_score`,
`history_threats`, `history_bonus`, `apply_gravity`, `apply_history_delta`
and the history constants; the singular-extension block and its two
constants), their git history, any other engine's code, or web
implementations. What was read: type and struct declarations, function
signatures, the call sites that use them, and black-box measurements of the
current engine (counters added at call sites outside the flagged bodies, in
a throw-away build). The concepts are the public ones described on the Chess
Programming Wiki (transposition table, replacement schemes, history
heuristic, singular extensions, multi-cut), restated here in our own words.
Every constant below is our own, chosen by the reasoning or measurement given
next to it. The old code is deleted by function/block boundaries, unread,
after the new code passes the tests.

Two lines of the flagged region were seen while locating boundaries: the
first statement of `store` (it loads the table's current age into a local)
and, in `main.rs`, the doc comment on the history type (layout
`[side][from attacked][to attacked][from][to]`, gravity bound 16384); both
are interface facts the audit already states. Nothing else of the flagged
code was seen.

## Interface (unchanged unless stated)

### Transposition table
- `TransTable::{new, clear, new_search, prefetch}` (kept as they are) and
  `probe(&self, key) -> Option<TTEntry>`,
  `store(&self, key, depth, score, static_eval: Option<i32>, bound, best: Option<Move>)`
  (both rewritten).
- Storage is kept: 64-byte buckets of four 16-byte slots; each slot is two
  `AtomicU64` (`data` = packed entry, `key` = zobrist key XOR data, so a torn
  concurrent write is detected by `key ^ data != zobrist`). Packed data:
  move 17 bits, score i16, static eval i16, depth+1 (u8, 0 = empty), bound
  2 bits, age 5 bits. `new_search` advances the 5-bit age (mod 32).
- Callers: the main search stores after every node (exact / lower / upper,
  no move on fail-low), ProbCut stores lower bounds at depth-3, the Syzygy
  cutoff stores depth+6, qsearch stores depth 0. The search never cuts on an
  entry while a singular verification excludes a move, and never stores the
  verification result.

### History
- Quiet history is read by the move picker for every quiet move (scored as
  history + continuation history + killer/counter bonuses + an escape bonus),
  by `move_order_score` (root ordering), and its sign decides whether a quiet
  is a "priority quiet" (searched first, reduced one ply less).
- At a beta cutoff by a quiet move the search gives that move `+bonus(depth)`
  and every quiet tried before it `-bonus(depth)` (call site unchanged), for
  the main, continuation and capture histories.
- `apply_gravity(h: &mut i16, delta)` is shared with the continuation history
  (`[side][prev to][from][to]`) and capture history
  (`[side][piece][to][victim]`); their score/update functions are not part
  of this rewrite and keep calling it.
- Every 4 searches all history tables are halved (`decay_history`), on a
  new game they are zeroed, and Lazy SMP workers copy them.
- The `ThreatHistory` option (default on) turns threat context off when
  false. `ThreatEscapeBonus` (SPSA-tuned) is a separate ordering term and is
  not changed.

### Singular extension
- Inputs at the point of the block (top of the move loop, after the excluded
  move is skipped): the node's depth, ply, alpha/beta, `pv_node`,
  `cut_node`, `in_check`, the TT entry and TT move, `search.excluded[ply]`
  (the move a verification search skips; the rest of the node already
  handles it: no TT cutoff, no null move, no ProbCut, no RFP, no store, and
  "only the excluded move was legal" returns alpha).
- Output: `extension` (plies added to the first move's depth, may be
  negative), or an early return of a score from the node.
- The `SeNegativeExtension` option (SPSA-tuned, default 1) is "plies the TT
  move loses when it is not singular but its TT score is at least beta"; the
  new design keeps that role and that option.

## (A) TT replacement: design

What a TT must do: keep the entries most likely to save work later, under a
fixed budget of four slots per bucket. Value of an entry grows with the
depth of the search that produced it (a deeper result replaces more work)
and is near zero once it belongs to a finished search whose subtree the game
has left (most of a previous search's positions are on branches that were not
played). Exact bounds are worth a little more than one-sided ones.

**Same position already in the bucket** (key matches):
1. Overwrite it, except keep the old entry when all of these hold: it was
   written in the current search, it is deeper than the new result by more
   than `TT_SAME_KEY_SLACK` = 2 plies, and the new result is not exact.
   Reason: a fresh result for the same position comes from the latest window
   and move ordering and should normally win; a much deeper lower/upper bound
   from the same search is the one thing worth keeping over it. The slack is
   small but non-zero so the re-searches of the same node at depth d-1, d-2
   (IIR, LMR re-search) refresh it.
2. A store with no move keeps the move already in the slot (the old best
   move is still the best ordering guess for this position).

**New position** (no key match): pick the victim in the bucket, in this
order:
1. an empty slot;
2. otherwise the slot with the lowest *keep value*, where entries from an
   older search always rank below entries from the current search (stale
   entries are evicted first regardless of depth), then by depth, then
   non-exact below exact; ties go to the earliest slot.
There is no linear age weight: "written in this search" is a single bit of
the ordering. Reason given above; the measurement below checks it.

**Probe:** scan the four slots, accept the one whose `key ^ data` equals the
probed key and whose depth byte is non-zero.

Measurable goals for (A):
- `cargo test`, perft, selftest unchanged; bench node count may change only
  through TT contents.
- TT stress test (scratch tool: 12 fixed 60-ply game sequences, every
  position of one side searched to a fixed depth with the TT kept between
  moves, Hash 1 MB and a large hash as control): total nodes within noise of
  main or better. Baseline measured on main: depth 13, Hash 1: 46.39M
  nodes; Hash 64: 46.11M (the hash size changes the total by 0.6%, so the
  policy has little leverage at these node counts).
- STC non-regression SPRT, then LTC non-regression (coordinator).

## (B1) Quiet history: design

What it must do: remember, per side, which quiet moves recently caused
beta cutoffs, so later nodes try them early and reduce them less. Two
competing needs: specificity (a move's value depends on the position) and
fast learning (few updates per entry).

Own design: a **factorised** history, the sum of two tables, each updated
with the same bonus:
- `B[side][from][to]`: the plain from-to ("butterfly") table, context free,
  learns fast.
- `C[side][ctx][piece][to]`: a piece-to table split by a 2-bit danger
  context of the move, computed from the opponent's attack maps by attacker
  value (`ThreatMaps`):
  - bit 0, *threatened*: the moving piece stands on a square attacked by a
    cheaper enemy piece (`by_lesser(piece)`); such a move is often an
    escape;
  - bit 1, *contested*: the destination is attacked by any enemy piece;
    such a move puts the piece where it can be challenged.
  With `ThreatHistory` off the context is always 0.
- Score = `B + C`.

Gravity (each table entry, i16): `h += delta - h * |delta| / cap`, the
standard self-limiting update (published as "history gravity"): repeated
equal bonuses converge to the cap, and a cut by a move with a large positive
value moves it less than one by a neutral move. Caps are chosen from our own
interfaces, not from any other engine:
- `GRAVITY_CAP = 15000` for every gravity-updated table that has one entry
  per move (continuation and capture histories). Reasons: the move picker
  adds history to fixed killer/counter bonuses spaced 10000 apart
  (30000/20000/10000), and the SPSA-tuned escape bonus is ~8000; a single
  table saturating at 15000 keeps a strongly proven history move between
  "one killer step" and "two killer steps", and below the i16 limit with
  room for the update's overshoot.
- The factorised quiet history splits that cap: `B` (move-specific) is
  bounded at 12000 and `C` (coarse danger context) at 3000, so `B + C` spans
  the same ±15000 as one table. (v1 split it evenly, 7500 + 7500, and lost
  15.1 +- 6.9 Elo at STC, 2688 games: with half the cap, the specific
  from-to evidence weighed too little against the generalising piece-to
  table, the continuation history and the escape bonus.)
- `bonus(depth) = min(20 * depth * depth + 40 * depth, 1800)`: grows with
  the size of the refuted subtree (quadratic is a mild proxy for an
  exponential tree), saturates by depth ~9 so the deepest nodes do not
  overwrite everything learnt below them in one cut; 1800 is 12% of the
  cap, so about eight consistent cuts bring an entry close to saturation.

Measurable goals for (B1): bench changes (new Bench trailer); STC
non-regression SPRT; LTC non-regression (coordinator).

## (B2) Singular extension: design

What it must do: a TT move that is much better than every alternative is
"singular": the line depends on it, so search it deeper. When the test shows
other moves are also good, the same search gives two useful by-products: if
an alternative already beats beta the node can be cut (multi-cut idea: several
moves fail high), and if the TT move is not alone and already at or above
beta it can be searched a little shallower.

Eligibility (all must hold): not the root (ply > 0); no move already
excluded at this ply; depth >= `SE_DEPTH` = 6 (the verification searches
about half the depth: below 6 that is 2 plies or less, too shallow to say
much and too frequent to pay for); the move is the TT move; the TT entry is
a lower bound or exact, with depth >= depth - 3 (a much shallower entry is
not a reliable reference score); the TT score is not a mate or tablebase
score (|score| < TB_WIN - MAX_PLY).

Verification: `sbeta = tt_score - margin`, with
`margin = SE_MARGIN_BASE + depth * SE_MARGIN_STEP_X2 / 2`
(own constants, `SE_MARGIN_BASE` = 8, `SE_MARGIN_STEP_X2` = 3, i.e. 8 + 1.5
cp per ply: at depth 8, 20 cp). The node is searched with the TT move
excluded at depth `(depth - 1) / 2`, null window `[sbeta - 1, sbeta]`, same
`cut_node` flag. Reason for the margin: the reference score is from a full
search, the alternatives from a half-depth one, so the gap must exceed the
noise between two searches of different depth; deeper reference scores are
more trusted per ply but the verification depth also grows, so a margin
increasing slowly with depth keeps the extension rate roughly level.

Outcome with `v` the verification score:
1. `v < sbeta`: singular, `extension = 1`.
2. else if `sbeta >= beta`: at least one other move beats beta at reduced
   depth while the TT move is expected to as well: return `sbeta`
   (multi-cut; not on mate-range scores, guaranteed by eligibility).
3. else if `tt_score >= beta`: `extension = -SeNegativeExtension`
   (the TT move is not alone and already expected to fail high).
4. else 0.

Measurable goals for (B2): black-box counters on main (bench 16, at the
call site of `extension` and at the loop boundaries around the block, not
inside it): of first-move visits at depth >= 6 (79.4k), 7.3% got +1, 3.0%
got -1, 4.9% returned early, none got +-2. The new design should land in the
same broad band (several percent each), not necessarily the same numbers;
then an STC non-regression SPRT (fixed-node CI screens first if it lags),
then LTC non-regression (coordinator). Expect to iterate on the margin: this
component is worth ~47 Elo.

## Credits
Transposition tables and replacement schemes, the history heuristic
(Schaeffer), singular extensions (Anantharaman, Campbell and Hsu) and
multi-cut (Björnsson and Marsland) are published ideas; see CREDITS.md.
