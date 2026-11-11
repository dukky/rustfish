#!/usr/bin/env bash
# Profile-guided (PGO) release build of rustfish.
#
#   scripts/build-pgo.sh [output-path]      (default: target/release-pgo/rustfish)
#
# 1. builds an instrumented binary (-Cprofile-generate) in target/pgo-gen,
# 2. runs a training workload on it: `bench` at the default and a deeper
#    depth, a short single-threaded `datagen` (the CI datagen hot path) and a
#    1 s `go movetime`,
# 3. merges the raw profiles with the toolchain's llvm-profdata,
# 4. rebuilds with -Cprofile-use in target/pgo-use and copies the binary to
#    the output path. target/release is never touched.
#
# Existing RUSTFLAGS (e.g. CI's `-C target-cpu=x86-64-v3`) are passed through
# to both builds. PGO only changes code layout/inlining, never semantics: the
# bench node count and `evalhash` must equal a plain build's. The script
# prints both for the PGO binary; with PGO_REFERENCE=<plain build> it fails on
# any difference (PGO_SKIP_CHECK=1 skips this).
#
# Env knobs: PGO_BENCH_DEPTH (deeper bench in the workload, default 12),
# PGO_DATAGEN_GAMES (default 40), PGO_TARGET (cargo --target triple; the
# binary must run on this machine), CARGO_BUILD_JOBS (cargo -j), PGO_KEEP=1
# keeps target/pgo-gen and the profiles. Also runs in the bash of Windows
# runners (paths for rustc are converted with cygpath).
#
# Needs llvm-profdata matching rustc's LLVM:
#   rustup component add llvm-tools-preview    (done automatically if missing)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# Paths given by the caller are relative to the caller's directory.
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }
OUT=$(abspath "${1:-$ROOT/target/release-pgo/rustfish}")
if [ -n "${PGO_REFERENCE:-}" ]; then PGO_REFERENCE=$(abspath "$PGO_REFERENCE"); fi
cd "$ROOT"
BENCH_DEPTH=${PGO_BENCH_DEPTH:-12}
DG_GAMES=${PGO_DATAGEN_GAMES:-40}
BASE_FLAGS=${RUSTFLAGS:-}
PROF_DIR="$ROOT/target/pgo-profiles"

CARGO_ARGS=(--release --quiet)
if [ -n "${CARGO_BUILD_JOBS:-}" ]; then CARGO_ARGS+=(-j "$CARGO_BUILD_JOBS"); fi
SUBDIR=release
if [ -n "${PGO_TARGET:-}" ]; then CARGO_ARGS+=(--target "$PGO_TARGET"); SUBDIR="$PGO_TARGET/release"; fi
EXE=""
NATIVE_PROF_DIR=$PROF_DIR   # the profile dir as rustc and the engine see it
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) EXE=.exe; NATIVE_PROF_DIR=$(cygpath -m "$PROF_DIR") ;;
esac

find_profdata() {
  local sysroot p
  sysroot=$(rustc --print sysroot)
  for p in "$sysroot"/lib/rustlib/*/bin/llvm-profdata$EXE; do
    if [ -x "$p" ]; then echo "$p"; return 0; fi
  done
  return 1
}

PROFDATA=$(find_profdata || true)
if [ -z "$PROFDATA" ]; then
  echo "build-pgo: installing llvm-tools-preview" >&2
  rustup component add llvm-tools-preview >&2
  PROFDATA=$(find_profdata) || { echo "build-pgo: llvm-profdata not found" >&2; exit 1; }
fi

t0=$(date +%s)
rm -rf "$PROF_DIR"
mkdir -p "$PROF_DIR"

echo "build-pgo: [1/4] instrumented build (RUSTFLAGS='$BASE_FLAGS')" >&2
RUSTFLAGS="$BASE_FLAGS -Cprofile-generate=$NATIVE_PROF_DIR" \
  cargo build "${CARGO_ARGS[@]}" --target-dir target/pgo-gen
GEN=target/pgo-gen/$SUBDIR/rustfish$EXE

echo "build-pgo: [2/4] training workload" >&2
printf 'bench\nbench %s\nquit\n' "$BENCH_DEPTH" | "$GEN" | grep 'Nodes searched' >&2
echo "datagen $NATIVE_PROF_DIR/datagen.txt $DG_GAMES 5000 1 7 0" | "$GEN" 2>&1 | grep 'datagen done' >&2
printf 'uci\nisready\nposition startpos\ngo movetime 1000\nquit\n' | "$GEN" | grep bestmove >&2

echo "build-pgo: [3/4] merging profiles" >&2
"$PROFDATA" merge -o "$PROF_DIR/merged.profdata" "$PROF_DIR"/*.profraw

echo "build-pgo: [4/4] optimized build" >&2
RUSTFLAGS="$BASE_FLAGS -Cprofile-use=$NATIVE_PROF_DIR/merged.profdata" \
  cargo build "${CARGO_ARGS[@]}" --target-dir target/pgo-use
mkdir -p "$(dirname "$OUT")"
cp "target/pgo-use/$SUBDIR/rustfish$EXE" "$OUT"

if [ "${PGO_KEEP:-0}" != 1 ]; then rm -rf target/pgo-gen "$PROF_DIR"/*.profraw "$PROF_DIR/datagen.txt"; fi

if [ "${PGO_SKIP_CHECK:-0}" != 1 ]; then
  signature() {
    printf 'bench\nquit\n' | "$1" | awk '/Nodes searched/ {printf "bench %s", $4}'
    printf 'evalhash\nquit\n' | "$1" | awk -F': *' '/hash/ {printf " evalhash %s", $2}'
  }
  s=$(signature "$OUT")
  echo "build-pgo: PGO $s" >&2
  if [ -n "${PGO_REFERENCE:-}" ]; then
    r=$(signature "$PGO_REFERENCE")
    if [ "$s" != "$r" ]; then
      echo "build-pgo: MISMATCH vs plain build $PGO_REFERENCE: $r" >&2
      exit 1
    fi
    echo "build-pgo: identical to plain build $PGO_REFERENCE" >&2
  fi
fi
echo "build-pgo: done in $(( $(date +%s) - t0 ))s -> $OUT" >&2
