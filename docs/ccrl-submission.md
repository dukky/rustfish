# CCRL submission

Everything CCRL testers need, what is already in place, and the exact steps to
publish the first rated release. Decisions were taken on 2026-11-04; do not tag
or announce anything until the owner says go.

## Decisions

| Item | Decision |
|---|---|
| Licence | **Apache-2.0** (`LICENSE`) |
| Engine identity | `id name Rustfish 1.0.0`, `id author dukky` |
| Version source | `Cargo.toml` (`1.0.0`); the `id name` line uses `env!("CARGO_PKG_VERSION")` |
| Opening book | **off by default** (`OwnBook=false`); all internal test workflows already passed `OwnBook=false` |
| Release timing | prepare now, tag only on explicit owner go-ahead |

## What CCRL needs

CCRL is not a formal submission desk: a new engine is announced on a chess
forum and a tester picks it up. From the CCRL board's own "How to Submit a New
Engine?" thread: announce the release on a chess forum (TalkChess is the usual
one, the CCRL Discussion Board also works) and give a rough idea of its
strength. The engine page then links the release and lists the author.

Testers run Windows GUIs (Cute Chess, Arena, Banksia) and normally download a
64-bit Windows executable. Their conditions, per the CCRL site:

- ponder off, 1 CPU, generic opening book of up to 12 moves;
- engine opening books off — a tester will set `OwnBook=false` if the option
  exists, but the default should already be off;
- 3-4-5 piece tablebases are available; rustfish does not probe them, which is
  allowed;
- Blitz and 40/15 lists are separate; single-CPU lists are the target.

## What CCRL asks you to state

The CCRL "How to get your engine tested" sticky asks for these details with the
download link:

- engine name and version number;
- estimated strength;
- your name and nationality;
- whether it is open source, and a homepage;
- credit where it is due (required for closed source, good practice anyway);
- **whether it was coded on your own or AI assisted**.

Rustfish's answers: **Rustfish 1.0.0**; **~3174 CCRL Blitz** estimate; **dukky,
United Kingdom**; open source under **Apache-2.0** at
https://github.com/dukky/rustfish; **AI assisted** — the code was written with
AI coding agents under the author's direction, and all design, testing, tuning
and release decisions are the author's. Provide the release URL with Windows
AVX2 and SSE4.2 executables (the sticky says testers use AVX2).

## What is ready

- Public repository and public release downloads.
- `.github/workflows/release.yml` builds and smoke-tests five binaries on a
  `v*` tag: Windows x86-64 `v3` (AVX2/BMI2) and `v2` (SSE4.2), Linux x86-64
  `v3`/`v2`, macOS arm64. Windows builds use an 8 MB main-thread stack.
- Each build runs `uci` / `isready` / `go depth 6` and `bench`; the release
  notes list every binary's bench node count.
- NNUE net is embedded; no external file is required.
- `Hash` default 64 MB, `Threads` default 1, `Move Overhead` default 10 ms.
- `stop` and `isready` work during search (`go infinite` + `stop` tested), so
  Cute Chess analysis mode is safe.
- Internal strength evidence: **3174.1 CCRL Blitz** (95% CI 3151.1-3197.6),
  1440 games at 120+1, **0 time losses**, measured 2026-11-04 on `030df68`
  (`tests/ladder-log.jsonl`). This is an anchor-ladder estimate, not an
  official CCRL rating; quote it as "estimated".

## Blocker before the first release

`OwnBook` defaulted to `true` in engine code until branch `ccrl-identity`
(build-verified, bench unchanged). The release must be cut **after** that
branch merges; otherwise a tester who leaves the default in place gets a book
move instantly instead of a search, and CCRL plays its own openings.

## Cutting the release

Only after the owner says go:

1. Confirm `main` has the identity branch and no SPRT is in flight.
2. Run the release workflow without publishing, in a quiet runner window:
   ```bash
   gh workflow run release.yml -f tag=vtest
   gh run watch <id> --interval 30 --exit-status
   ```
   Download one Linux artifact and run `scripts/ccrl_check.py <binary>` plus a
   `go infinite` / `stop` check. The Windows `.exe` is smoke-tested by the
   workflow itself.
3. Tag the release commit (annotated) and push the tag:
   ```bash
   git tag -a v1.0.0 -m "Rustfish 1.0.0"
   git push origin v1.0.0
   ```
   The workflow creates `v1.0.0` with the five binaries attached. If a build
   fails, delete the tag (`git push --delete origin v1.0.0 && git tag -d v1.0.0`)
   and re-tag after the fix.
4. Edit the release notes to add the strength estimate, UCI options and licence.
   The workflow's auto-generated body already has the binary/bench table; keep
   it and prepend the release summary:
   ```bash
   body=$(gh release view v1.0.0 --json body -q .body)
   { cat <<'EOF'
   Rustfish 1.0.0 — a zero-dependency UCI chess engine in Rust (Apache-2.0).

   Estimated ~3174 CCRL Blitz (95% CI 3151-3198) from 1440 games at 120+1
   against rated anchors, 0 time losses, measured on <release-sha>. This is a
   self-measured anchor-ladder estimate, not an official CCRL rating.

   No external files are needed: the NNUE net is embedded and the built-in
   book is off by default (`OwnBook=false`). `v3` binaries need AVX2/BMI2
   (Haswell / Zen 1 or newer); use `v2` on older CPUs.
   EOF
     echo; echo "$body"; } > /tmp/rustfish-release-notes.md
   gh release edit v1.0.0 --notes-file /tmp/rustfish-release-notes.md
   ```
5. Post the announcement (below) on TalkChess and the CCRL Discussion Board.

## Announcement draft

> **Rustfish 1.0.0 — a zero-dependency Rust engine**
>
> I have released Rustfish 1.0.0, a UCI chess engine written in Rust with no
> dependencies outside the standard library. It uses an NNUE evaluation
> (768→256×2→1, net embedded in the binary) and a PVS search with aspiration,
> TT, NMP, RFP, futility/LMP pruning, LMR, SEE, singular extensions, killers,
> counter-moves and continuation history.
>
> - Download (Windows x86-64 `v3`/`v2`, Linux x86-64 `v3`/`v2`, macOS arm64):
>   https://github.com/dukky/rustfish/releases/tag/v1.0.0
> - Author: dukky (United Kingdom). Open source, Apache-2.0; source at
>   https://github.com/dukky/rustfish.
> - Coded on your own or AI assisted: **AI assisted** — the code was written
>   with AI coding agents under my direction; all design, testing, tuning and
>   release decisions are mine.
> - Protocol: UCI. Single-threaded by default (`Threads=1`), `Hash=64`.
> - No external book or network file; the built-in book is off by default
>   (`OwnBook=false`), so a GUI book is used.
> - Estimated strength: **~3170 CCRL Blitz** (95% CI ~3150-3200) from 1440
>   games at 120+1 against rated anchors, with no time losses. This is a
>   self-measured anchor-ladder estimate, not an official CCRL rating.
>
> The `v3` binary needs AVX2/BMI2 (Haswell / Zen 1 or newer); use `v2` on
> older CPUs. `bench` prints a node count that identifies the build; the
> release notes list the expected value for every binary.
>
> Thanks — any feedback or games are welcome.

## Verification checklist

Run on a downloaded release binary before announcing:

- [ ] `uci` prints `id name Rustfish 1.0.0`, `id author dukky`, `uciok`
- [ ] `OwnBook` advertises `default false`
- [ ] `scripts/ccrl_check.py <binary>` passes every check
- [ ] `bench` twice gives the same node count
- [ ] one game in Cute Chess on Windows with `OwnBook=false`, ponder off,
      `Hash=64`, `Threads=1` finishes with no time loss or illegal move
- [ ] release page lists all five binaries and their bench counts

## Open items for the owner

1. **Forum account / announcement** — the announcement must come from the
   owner's TalkChess and CCRL board accounts.
2. **Tag and release** — deliberately not done yet.
3. **Windows GUI smoke test** — CI runs the `.exe`, but one real Cute Chess
   game on Windows is still worth doing before submitting.
4. **Tuning options** — the SPSA spin options remain advertised; harmless, but
   testers should leave them alone. Hiding them in release builds would break
   the SPSA workflow unless it sets them by name without reading `uci` first.
