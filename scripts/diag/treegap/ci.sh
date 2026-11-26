#!/usr/bin/env bash
# Tree-gap CI plans (run by peer-match.yml's tree job with limit=tree:<plan>).
#   bb:<depth>          black box: rustfish and the peer, go depth <depth> and
#                       go movetime 1000, one search at a time; plus the
#                       treestats build at depth min(<depth>,16), parallel
#   kn:<depth>:<k1>/<k2>/...  treestats build at go depth <depth> for each knob
#                       set (k = name_value:name_value, "base" = defaults), parallel
set -euo pipefail
plan=$1; shard=$2; shards=$3
here=$(dirname "$0")
mkdir -p out
awk -v s="$shard" -v n="$shards" 'NR % n == s' "$here/positions.txt" > out/pos.txt
wc -l out/pos.txt
R="python3 $here/run.py --positions out/pos.txt --option Hash=16 --option Threads=1"
J=$(nproc)
IFS=: read -r kind rest <<<"$plan"
case "$kind" in
  bb)
    d=$rest
    $R --engine engines/rustfish-ts --depth $(( d < 16 ? d : 16 )) --jobs "$J" --out out/rf-ts.jsonl
    for rep in 1 2; do
      $R --engine engines/peer --depth "$d" --timeout 90 --out out/peer-d$d-r$rep.jsonl
      $R --engine engines/rustfish --depth "$d" --timeout 90 --out out/rf-d$d-r$rep.jsonl
    done
    $R --engine engines/rustfish --movetime 1000 --out out/rf-mt1000.jsonl
    $R --engine engines/peer --movetime 1000 --out out/peer-mt1000.jsonl
    ;;
  kn)
    d=${rest%%:*}; sets=${rest#*:}
    IFS=/ read -ra arr <<<"$sets"
    for ks in "${arr[@]}"; do
      k=$ks; [ "$k" = base ] && k=""
      $R --engine engines/rustfish-ts --depth "$d" --jobs "$J" --knobs "${k//_/=}" --out "out/kn-d$d-${ks//:/+}.jsonl"
    done
    ;;
  cnt)
    # node-counting convention, black box: tiny searches, all info lines
    for e in rustfish peer; do
      while read -r pos; do
        for d in 1 2 3; do
          echo "== $e $pos depth $d"
          (printf 'uci\nsetoption name Hash value 16\nisready\nucinewgame\nposition %s\ngo depth %s\n' "${pos/#fen/fen}" "$d"; sleep 1; printf 'quit\n') | engines/$e | grep -E '^info|^bestmove' | tee -a out/cnt.txt
        done
      done < "$here/count.txt"
    done
    ;;
  *) echo "unknown plan $plan"; exit 1 ;;
esac
ls -la out
