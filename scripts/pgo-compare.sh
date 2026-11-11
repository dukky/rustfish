#!/usr/bin/env bash
# Interleaved speed comparison of two rustfish binaries that must search
# identically (e.g. a plain and a PGO build):
#
#   scripts/pgo-compare.sh <base> <candidate> [rounds] [bench-depth] [datagen-games] [datagen-threads]
#
# Fails unless bench node counts and evalhash match. Then alternates the run
# order every round (the fanless M3 and shared CI runners drift) and prints
# bench NPS and datagen games/s per round plus the mean gain of candidate over
# base. datagen-games=0 skips datagen.
set -euo pipefail
A=$1; B=$2; R=${3:-7}; D=${4:-13}; G=${5:-100}; T=${6:-1}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

bench_nodes() { printf 'bench\nquit\n' | "$1" | awk '/Nodes searched/ {print $4}'; }
evalhash() { printf 'evalhash\nquit\n' | "$1" | awk -F': *' '/hash/ {printf "%s ", $2}'; }
na=$(bench_nodes "$A"); nb=$(bench_nodes "$B")
ea=$(evalhash "$A"); eb=$(evalhash "$B")
echo "bench nodes: base $na candidate $nb"
echo "evalhash:    base $ea"
echo "             cand $eb"
if [ "$na" != "$nb" ] || [ "$ea" != "$eb" ]; then echo "MISMATCH: binaries do not search identically"; exit 1; fi
echo "identical: yes"

nps() { printf 'bench %s\nquit\n' "$D" | "$1" | awk '/Nodes\/second/ {print $3}'; }
gps() { echo "datagen $TMP/dg.txt $G 5000 $T 3 0" | "$1" 2>&1 | sed -n 's/^datagen done.* \([0-9.]*\) games\/s.*/\1/p'; }
summary() { awk -v what="$1" '{print; sa+=$4; sb+=$6; n++} END {printf "%s mean: base %.2f candidate %.2f gain %+.2f%%\n", what, sa/n, sb/n, (sb/sa-1)*100}'; }

for i in $(seq 1 "$R"); do
  if [ $((i % 2)) = 1 ]; then a=$(nps "$A"); b=$(nps "$B"); else b=$(nps "$B"); a=$(nps "$A"); fi
  echo "nps $i base $a candidate $b"
done | summary "bench $D NPS"

if [ "$G" -gt 0 ]; then
  for i in $(seq 1 $(( (R + 1) / 2 ))); do
    if [ $((i % 2)) = 1 ]; then a=$(gps "$A"); b=$(gps "$B"); else b=$(gps "$B"); a=$(gps "$A"); fi
    echo "datagen $i base $a candidate $b"
  done | summary "datagen ${G}g x${T}t games/s"
fi
