# Post-2100 Revised

## Why this exists
The original `post-2100-plan.md` was useful as a directional map, but several of its highest-priority ideas did not survive longer A/B testing in this engine.

This note is the corrected version:
- what actually worked
- what was tried and rejected
- what now looks most plausible from the repo's real measurement history

---

## What the engine seems to have learned

### 1. Single-thread structure matters
The strongest accepted post-2100 gains came from structural single-thread search work, not from local parameter tuning.

Accepted examples:
- TT static eval reuse
- staged pick-next move picker

### 2. Micro-tuning mostly does not matter much now
Small pruning-margin and local ordering tweaks are usually neutral or negative.

### 3. Theory is not enough
Many ideas that are strong in principle or in stronger engines did not pay off here in the forms tested.

### 4. Longer A/B matters
Short screens are useful for killing obvious losers, but acceptance decisions need longer confirmation.

---

## Proven winners so far

### TT static eval reuse
Accepted at commit `f4143cd`.

Why it mattered here:
- reduced repeated eval work
- improved pruning/qsearch eval fallback
- fit the current engine cleanly

### Staged pick-next move picker
Accepted at commit `9b0b346`.

Why it mattered here:
- better early cutoffs
- cleaner move ordering structure
- practical Elo gain confirmed on longer A/B

### Scalar hot-path cleanup that helps search behavior
Some speed and structure changes paid off, but only when they improved useful search behavior instead of just raw NPS.

---

## Rejected for now

These were tried seriously enough to matter, and should not be assumed to be near-term winners anymore.

- clustered TT
- pawn hash / pawn-eval cache groundwork
- capture history
- correction history
- ProbCut
- singular extension variants tried so far
- root-split SMP as a practical strength gain
- pin-aware/full-legality SEE attacker selection (2026-07-03: ~14% NPS cost, no node reduction at fixed depth; patch archived in `tmp/patches/`)

This does not mean these ideas are bad forever.
It means they have not paid off yet in this engine, with this surrounding architecture, in the tested forms.

---

## What this probably means

### Engine-specific reality beats general theory
An idea can be strong in top engines and still fail here because of:
- current board representation
- current TT/search coupling
- current eval noise
- current move ordering already absorbing much of the gain
- insufficient surrounding infrastructure
- implementation details not yet mature enough

So the right lesson is not "those ideas are nonsense."
The right lesson is "they are not currently the best next patch here."

---

## Best remaining frontier

### 1. Incremental eval groundwork
This is still the clearest unfinished structural eval item.

Why it remains attractive:
- helps both speed and richer eval
- compounds with future work
- raises the ceiling instead of chasing one heuristic

Good starting areas:
- material / phase bookkeeping
- PST accumulation
- pawn summaries
- king-square dependent cached pieces where practical

### 2. Better endgame knowledge and scaling
This now looks more attractive than many middlegame ornament terms.

Most plausible areas:
- rook endings
- pawn races
- king-and-pawn opposition-ish logic
- cleaner endgame scaling

Why:
- conversion quality matters more now
- these terms are often more stable than speculative search gadgets

### 3. One genuinely different search concept
If search work continues, it should be something structurally distinct.

Probably not:
- another margin sweep
- another tiny move-order score tweak
- another local null/LMR threshold change

Possibly yes:
- a better extension/reduction logic class
- stronger search-state reuse
- a trust/stability concept that fits this engine better than the rejected correction-history attempt

---

## Measurement policy going forward

Use the following hierarchy:

### Primary acceptance
Frozen binary A/B vs current accepted champion.

Recommended:
- quick loser screen: `24g / 20ms / parallel 4`
- real confirmation: `48-96g / 30-100ms / parallel 2`

### External anchor
Use Stockfish only as a ladder anchor, not as the only acceptance metric.

Recommended ladder:
- `2100`
- `2200`
- `2300`

### What not to trust much
- raw bench NPS as Elo
- one short run vs one Stockfish Elo setting
- high-parallel local confirmation on this machine

---

## Revised strategic summary

The original post-2100 note was correct about the big strategic move:
- stop expecting tiny tuning wins
- prioritize structural single-thread work

What changed is the shortlist of likely winners.

The updated practical order is:
1. incremental eval groundwork
2. endgame quality/scaling
3. one truly different search concept
4. only later revisit larger TT/SMP architecture bets if the surrounding engine gets stronger

---

## Bottom line

The engine did improve after 2100, but not through most of the initially favored ideas.

The empirical lesson is:
- structural single-thread work was the right direction
- only some structural ideas actually paid off here
- future progress should be guided by measured repo behavior, not just engine theory
