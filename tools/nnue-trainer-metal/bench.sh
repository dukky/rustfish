#!/bin/sh
# Per-stage GPU time of one 16384 batch for the standard shapes.
# usage: ./bench.sh <records.bin>
set -e
cd "$(dirname "$0")"
for h in 256 512 1024; do
  for b in 1 8; do
    ./nnue-trainer-metal bench --data "$1" --hidden $h --buckets $b
  done
done
