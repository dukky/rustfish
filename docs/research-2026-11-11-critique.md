# Critique of the 2026-11-11 Coda / Lc0 / compute notes

Review of `docs/research-2026-11-11-coda-lc0-compute.md` (Hermes/DeepSeek,
branch `notes/hermes-coda-lc0-research`, copied to main alongside this) and
`docs/peer-analysis-2026-11-08.md`. Read against our own logs, not taken as
given. Written 2026-11-11.

## Accepted

- **CI policy.** GitHub-hosted runners are for builds and tests; no bulk
  datagen or training. Done: datagen workflow disabled, queue lanes moved to
  18 test shards, bursts off (`0d3af12`).
- **Rescoring is the best-evidenced lever in the notes** (Viridithas: +50,
  +25, +24 on successive rescoring cycles, then +4-8). It is also the
  cheapest for us: rescore our own positions with g7kb8m at higher nodes on
  the desktop, no new format, no licence question.
- **Separate hardware lanes** (desktop trainer, Mac Metal trainer, 1080 Ti if
  the CUDA port works, CI for gates); don't split one net across devices.
- **Step 3 of its plan is the right shape of test:** T80 on the *current* net
  shape, one variable, gated in games.

## Disputed or corrected

1. **"More self-play mostly teaches the net to imitate itself."** Our own
   record says otherwise so far: g4 +26.7, g5kb8m +30.4, g6kb8m +17.2,
   g7kb8m +30.6 in net-gates, with no sign of shrinking returns. Our labels
   come from a 5000-node search, not the raw net, so this is expert
   iteration (search improves the policy), which the note itself names as
   the case where self-play works. Returns will diminish; they have not yet.
   Treat "self-play has hit its ceiling" as a hypothesis to test, not a fact.
2. **The -20 Elo (Coda self-play vs T80) does not transfer directly.** It was
   measured at Coda's strength with a ~1024-wide threat net that has the
   capacity to use better labels; at our 512-wide net the label ceiling may
   bind less. Unknown either way, which is why it needs our own arm.
3. **Peer evidence cuts both ways.** Panda (3523, same 512 shape, self-play)
   and Carp (3525) sit above us without T80; Endspiel uses T80 labels *and*
   a 1536-wide net and sits at 3505, level with us. That doesn't show T80
   is worthless, but it doesn't make T80 "the highest-value next
   experiment" either.
4. **"~37% usable" is Coda's T80 filter rate applied to our data.** Our
   datagen already filters at write time (`src/datagen.rs`: positions in
   check, captures/promotions/en passant and mate scores are never written),
   so CI position counts are already usable positions. The note's
   "~2.7B raw -> ~1.0B usable" since pool7 likely undercounts by ~2.7x.
5. **Cost is understated.** T80 needs a Stockfish binpack reader (a
   non-trivial format) or a converter, TB-scale downloads and disk we don't
   have spare, a calibration from Lc0 Q/WDL to our centipawn scale and
   sigmoid, and ODbL/DbCL attribution and share-alike checks before any CCRL
   release. Days of engineering before the first result.

## What we do instead (in plan.md)

A three-arm test on the desktop, current net shape, same position count,
gated in games, one variable each:

| Arm | Data | Cost |
|---|---|---|
| A | own self-play (baseline, the gen-8 recipe) | none: already the plan |
| B | own self-play rescored by g7kb8m at higher nodes | a `rescore` tool, desktop CPU time |
| C | Lc0 T80 subset (quiet-filtered, calibrated) | binpack reader + calibration + licence check |

B first (cheap, best-evidenced); C if B and A plateau or when the
binpack reader exists. The note's later recommendation (Coda-0.4-class net:
1024 FT, more king buckets, threat inputs) overlaps our own long-term plan
and is being probed now with g7h1024 on fixed pool7.

## Not checked

- Coda timeline and process details (taken as background only).
- The Lichess bot limits section: plausible, not verified here.
