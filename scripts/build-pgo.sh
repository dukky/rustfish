#!/usr/bin/env bash
# Profile-guided (PGO) release build of rustfish.
#
#   scripts/build-pgo.sh [output-path]      (default: target/release-pgo/rustfish)
#
# 1. builds an instrumented binary (-Cprofile-generate) in target/pgo-gen,
# 2. runs a training workload on it: `bench` at the default and a deeper
#    depth, plus a short single-threaded `datagen` (the CI datagen hot path),
# 3. merges the raw profiles with the toolchain's llvm-profdata,
# 4. rebuilds with -Cprofile-use in target/pgo-use and copies the binary to
#    the output path. target/release is never touched.
#
# Existing RUSTFLAGS (e.g. CI's `-C target-cpu=x86-64-v3`) are passed through
# to both builds. PGO only changes code layout/inlining, never semantics: the
# bench node count and `evalhash` must equal a plain build's (checked at the
# end; set PGO_SKIP_CHECK=1 to skip).
#
# Env knobs: PGO_BENCH_DEPTH (deeper bench in the workload, default 12),
# PGO_DATAGEN_GAMES (default 40), CARGO_BUILD_JOBS (cargo -j), PGO_KEEP=1 keeps
# target/pgo-gen and the profiles.
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
MERGED="$PROF_DIR/merged.profdata"
JOBS=()
if [ -n "${CARGO_BUILD_JOBS:-}" ]; then JOBS=(-j "$CARGO_BUILD_JOBS"); fi

find_profdata() {
  local sysroot p
  sysroot=$(rustc --print sysroot)
  for p in "$sysroot"/lib/rustlib/*/bin/llvm-profdata; do
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
RUSTFLAGS="$BASE_FLAGS -Cprofile-generate=$PROF_DIR" \
  cargo build --release --quiet "${JOBS[@]}" --target-dir target/pgo-gen
GEN=target/pgo-gen/release/rustfish

echo "build-pgo: [2/4] training workload" >&2
printf 'bench\nbench %s\nquit\n' "$BENCH_DEPTH" | "$GEN" | grep 'Nodes searched' >&2
echo "datagen $PROF_DIR/datagen.txt $DG_GAMES 5000 1 7 0" | "$GEN" 2>&1 | grep 'datagen done' >&2
printf 'uci\nisready\nposition startpos\ngo movetime 1000\nquit\n' | "$GEN" | grep bestmove >&2

echo "build-pgo: [3/4] merging profiles" >&2
"$PROFDATA" merge -o "$MERGED" "$PROF_DIR"/*.profraw

echo "build-pgo: [4/4] optimized build" >&2
RUSTFLAGS="$BASE_FLAGS -Cprofile-use=$MERGED" \
  cargo build --release --quiet "${JOBS[@]}" --target-dir target/pgo-use
mkdir -p "$(dirname "$OUT")"
cp target/pgo-use/release/rustfish "$OUT"

if [ "${PGO_KEEP:-0}" != 1 ]; then rm -rf target/pgo-gen "$PROF_DIR"/*.profraw "$PROF_DIR/datagen.txt"; fi

if [ "${PGO_SKIP_CHECK:-0}" != 1 ]; then
  # Print the PGO binary's bench and evalhash; with PGO_REFERENCE=<plain
  # build> also require them to match it (a mismatch means PGO changed
  # behaviour, e.g. through UB, and the binary must not be used).
  b=$(printf 'bench\nquit\n' | "$OUT" | awk '/Nodes searched/ {print $4}')
  e=$(printf 'evalhash\nquit\n' | "$OUT" | awk -F': ' '/hash/ {printf "%s ", $2}')
  echo "build-pgo: PGO bench $b evalhash $e" >&2
  if [ -n "${PGO_REFERENCE:-}" ]; then
    rb=$(printf 'bench\nquit\n' | "$PGO_REFERENCE" | awk '/Nodes searched/ {print $4}')
    re=$(printf 'evalhash\nquit\n' | "$PGO_REFERENCE" | awk -F': ' '/hash/ {printf "%s ", $2}')
    if [ "$b" != "$rb" ] || [ "$e" != "$re" ]; then
      echo "build-pgo: MISMATCH vs $PGO_REFERENCE: bench $rb evalhash $re" >&2
      exit 1
    fi
    echo "build-pgo: matches $PGO_REFERENCE" >&2
  fi
fi
echo "build-pgo: done in $(( $(date +%s) - t0 ))s -> $OUT" >&2
