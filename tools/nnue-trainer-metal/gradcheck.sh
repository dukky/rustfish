#!/bin/sh
# GPU gradient vs CPU reference + finite differences over several shapes.
# usage: ./gradcheck.sh <records.bin>
set -e
cd "$(dirname "$0")"
D="$1"
./nnue-trainer-metal gradcheck --data "$D" --hidden 16 --buckets 1 --batch 64 --mirror
./nnue-trainer-metal gradcheck --data "$D" --hidden 32 --buckets 8 --batch 300
./nnue-trainer-metal gradcheck --data "$D" --hidden 256 --buckets 8 --batch 2048
./nnue-trainer-metal gradcheck --data "$D" --hidden 1024 --buckets 1 --batch 1000 --mirror
