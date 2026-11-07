#!/bin/sh
# GPU gradient vs CPU reference + finite differences over several shapes,
# with and without king buckets.
# usage: ./gradcheck.sh <records.bin>
set -e
cd "$(dirname "$0")"
D="$1"
./nnue-trainer-metal gradcheck --data "$D" --hidden 16 --buckets 1 --batch 64 --mirror
./nnue-trainer-metal gradcheck --data "$D" --hidden 32 --buckets 8 --batch 300
./nnue-trainer-metal gradcheck --data "$D" --hidden 256 --buckets 8 --batch 2048
./nnue-trainer-metal gradcheck --data "$D" --hidden 1024 --buckets 1 --batch 1000 --mirror
./nnue-trainer-metal gradcheck --data "$D" --hidden 32 --buckets 8 --batch 300 --king-buckets kb8m
./nnue-trainer-metal gradcheck --data "$D" --hidden 16 --buckets 1 --batch 64 --mirror --king-buckets kb4m
./nnue-trainer-metal gradcheck --data "$D" --hidden 64 --buckets 1 --batch 1000 --mirror --king-buckets kb4
./nnue-trainer-metal gradcheck --data "$D" --hidden 32 --buckets 1 --batch 500 --king-buckets m1
./nnue-trainer-metal gradcheck --data "$D" --hidden 256 --buckets 8 --batch 2048 --king-buckets kb8m
./nnue-trainer-metal gradcheck --data "$D" --hidden 64 --buckets 8 --batch 777 --mirror --king-buckets kb8m
# Factoriser (W + F merged; F's gradient = sum over buckets) and
# active-only weight decay (default on for KB > 1).
./nnue-trainer-metal gradcheck --data "$D" --hidden 32 --buckets 8 --batch 300 --king-buckets kb4 --factorise
./nnue-trainer-metal gradcheck --data "$D" --hidden 16 --buckets 1 --batch 64 --mirror --king-buckets kb4m --factorise
./nnue-trainer-metal gradcheck --data "$D" --hidden 32 --buckets 8 --batch 300 --king-buckets kb8m --factorise
./nnue-trainer-metal gradcheck --data "$D" --hidden 64 --buckets 8 --batch 1000 --mirror --king-buckets kb8m --factorise
./nnue-trainer-metal gradcheck --data "$D" --hidden 256 --buckets 8 --batch 2048 --mirror --king-buckets kb8m --factorise
./nnue-trainer-metal gradcheck --data "$D" --hidden 32 --buckets 8 --batch 300 --king-buckets kb8m --factorise --no-wd-active-only
