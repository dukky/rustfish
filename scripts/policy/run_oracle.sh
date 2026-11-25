#!/bin/bash
# usage: run_a.sh <tag> <deep> <shallow> <arms...>; 2 shards over pos962
cd ~/agent-runs/policy
tag=$1; deep=$2; sh=$3; shift 3; arms="$*"
mkdir -p out/$tag
for shard in 0 1; do
  ( awk -v s=$shard 'NR%2==s' ~/agent-runs/qs/pos962.txt | while read l; do
      echo "POS $l"
      printf "oracletest $deep $sh $arms | position $l\nquit\n" | nice -n 19 bin/po
    done > out/$tag/shard$shard.txt ) &
  echo $! >> out/$tag/pids
done
wait
echo DONE > out/$tag/done
