# CCRL submission

What CCRL testers need from rustfish, how to cut a release that gives it to
them, and the decisions still open before submitting.

## What testers need
CCRL testers mostly run Windows GUIs (Cute Chess, Arena, Banksia) and download
binaries; few build from source. A submission should point them at one page:

- **Release page:** `https://github.com/dukky/rustfish/releases/latest` (each
  tag `v*` produces one; see below).
- **Binaries** attached to every release (built by
  `.github/workflows/release.yml`):

  | File | Platform | CPU requirement |
  |---|---|---|
  | `rustfish-<tag>-windows-x86-64-v3.exe` | Windows | AVX2/BMI2 (Haswell, Zen 1 or newer) |
  | `rustfish-<tag>-windows-x86-64-v2.exe` | Windows | POPCNT/SSE4.2 (most CPUs since 2009) |
  | `rustfish-<tag>-linux-x86-64-v3` | Linux | AVX2/BMI2 |
  | `rustfish-<tag>-linux-x86-64-v2` | Linux | POPCNT/SSE4.2 |
  | `rustfish-<tag>-macos-arm64` | macOS, Apple Silicon | none |

  Testers should prefer `v3`, falling back to `v2` if it will not start.
  The Windows builds are linked with an 8 MB main-thread stack
  (`/STACK:8388608`), because the search runs on the main thread and the MSVC
  default is 1 MB.
- **Bench signature:** the release notes list the `bench` node count of each
  binary. All binaries should report the same count; a tester can type
  `bench` in a console to check that a download is the right build.

### UCI behaviour (as of this branch, read from `src/main.rs`, `src/uci.rs`, `src/search.rs`)
- `id name Rustfish ZeroDep`, `id author pi+chatgpt`.
- Single thread by default (`Threads` default 1). CCRL runs most lists at one
  thread, so nothing needs setting.
- `go` supports `wtime/btime/winc/binc/movestogo`, `movetime`, `depth`,
  `nodes` and `infinite`.
- `stop` works: stdin is read on its own thread (`spawn_input_reader` in
  `src/uci.rs`), which flags the running search to stop. The search checks the
  flag and answers with `bestmove`. `go infinite` searches until `stop`.
  `isready` is answered during a search, too.
- No pondering (no `Ponder` option). CCRL usually tests with ponder off.

### UCI options
| Option | Type | Default | Notes for testers |
|---|---|---|---|
| `Hash` | spin 1-2048 | 64 | TT size in MB; testers set their standard size |
| `Threads` | spin 1-32 | 1 | |
| `Clear Hash` | button | | |
| `Move Overhead` | spin 0-5000 | 10 | ms kept back per move for GUI/network lag |
| `OwnBook` | check | **true** | built-in opening book; see open items |
| `BookStyle` | combo `normal`/`grinder`/`chaos` | normal | only matters with OwnBook on |
| `BookFile` | string | empty | external book path |
| `FutilityMarginD1`, `FutilityMarginD2`, `ReverseFutilityMarginD1..D3`, `NullMarginBase`, `NullMarginDepthScale`, `NullReductionBase`, `NullReductionDepthDiv`, `LmpLimitD1..D3`, `LmrMinDepth`, `LmrMinMoveIndex`, `LmrExtraDepth`, `LmrExtraMoveIndex` | spin | tuned | search tuning; testers should leave these alone |

**OwnBook is on by default.** CCRL plays from its own opening books and
expects engine books to be off. Tell testers to set `OwnBook=false`, or ship
releases with the book off by default (open item 3). This document does not
change the default.

## Cutting a release
1. Make sure `main` is green (CI) and the commit you want is on it.
2. Tag it and push the tag:
   ```bash
   git tag v1.0
   git push origin v1.0
   ```
3. The `Release` workflow builds the five binaries, runs a UCI smoke
   (`uci`, `isready`, `position startpos`, `go depth 6`, `quit`) and `bench`
   on each, then creates the GitHub release `v1.0` with the binaries attached
   and the bench node counts in the notes. If any build or smoke test fails,
   no release is created. Fix it, delete the tag
   (`git push --delete origin v1.0 && git tag -d v1.0`) and tag again.

To test the workflow without publishing anything, run it by hand from the
Actions tab (or `gh workflow run release.yml -f tag=vtest`). In that mode it
uploads the binaries as workflow artifacts only. It creates no release and no
tag.

## Open items for the owner
1. **Licence.** The repo has no `LICENSE` file. CCRL and most testers expect
   an open-source licence before they distribute or list an engine (MIT or
   GPL-3.0 are the usual choices). Add one before the first release.
2. **Engine identity.** `id name Rustfish ZeroDep` / `id author pi+chatgpt`
   (`src/main.rs`, the `uci` handler) is what appears on the rating list. Pick
   the final name, author string and a version number (for example
   `id name Rustfish 1.0`). CCRL lists engines by name plus version, and the
   current string has no version. `Cargo.toml` is still `0.1.0`.
3. **OwnBook default.** Testers usually turn engine books off, but a tester
   who misses it would play rustfish's book over their own openings. Shipping
   releases with `OwnBook` defaulting to `false` removes that risk.
4. **Exposed tuning options.** The 16 search-tuning spin options show up in
   every GUI's engine dialog. Consider hiding them in release builds or
   removing them, so testers don't change them by accident.
5. **stop / go infinite.** Already implemented on `main` (see above); nothing
   to decide. Just check it once in a Windows GUI (Cute Chess "analyze" mode
   uses `go infinite` + `stop`) with the downloaded `.exe`.
