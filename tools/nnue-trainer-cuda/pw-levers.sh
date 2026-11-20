#!/bin/bash
# GPU A/B of the pairwise-head trainer levers (desktop, 1080 Ti): the pairwise pilot's shape
# recipe (gpu-jobs/pw-scratch.sh: H1024, plain 4 -> m1 1 -> kb8m factorise 2, --half, the
# 1.27B newest positions, val-g10) with one lever per variant, in every stage:
#   sp1 / sp2  --ft-l1-group W (chunk group-lasso on the pairwise products; SP1 / SP2 below;
#              the CPU H256 sweep had group slightly ahead of plain --ft-l1 per chunk saved)
#   ql2        --qat-l2 (integer L2; the net gets the RFNT trailer)
# The baseline is the pilot itself ($D/pw/s1024kb8m.bin, same recipe/data/val, trainer
# default path byte-identical); BASE=1 re-trains it with this trainer too.
# Each variant ~2.5-3 h on an idle GPU. Start only when the GPU is free (waits for it).
# Results: $L/summary.log (val-g10 / valL1008 best-scale, nonzero L1 chunks before/after
# the trainer's activity-sort permutation, on 20k val-g10 positions).
# usage: pw-levers.sh [variant ...]   (default: sp1 sp2 ql2)
#   TRAINER_DIR = a build of branch trainer-pw-levers (tools/nnue-trainer{,-cuda}/target/release)
set -u
D=~/rustfish-data; B=${TRAINER_DIR:-~/pwlever-build}
G=$B/tools/nnue-trainer-cuda/target/release/nnue-trainer-cuda; T=$B/tools/nnue-trainer/target/release/nnue-trainer
L=$D/gpu-jobs/pw-levers; O=$D/pw-levers; mkdir -p $L $O
SP1=${SP1:-1e-5}; SP2=${SP2:-3e-5}
log() { echo "$(date +%T) $*" >> $L/summary.log; }
q=0; while [ $q -lt 3 ]; do
  if pgrep -f "nnue-trainer-cuda" > /dev/null; then q=0; else q=$((q + 1)); fi; sleep 60
done
cd $D
FILES="$(ls $D/local/D20*.rfp | grep -vE "D200[01]\.rfp" | sort -V | tr "\n" " ")"
N=$(cat $D/shape.count)
DATA="$(for f in $FILES; do echo -n "--data $f "; done)--shuffle-buffer 67108864 --stream-stride 4 --loader-threads 0 --epoch-size $N"
C="--val $D/val-g10.bin --buckets 8 --threads 2 --half --hidden 1024 --head pairwise"
measure() {  # name net
  for v in val-g10.bin valL1008.bin; do
    log "$1 $v: $($T eval --net $2 --val $D/$v --buckets 8 --king-buckets kb8m --best-scale --threads 4 2>/dev/null | grep 'best scale')"
  done
  log "$1 chunks: $($T permute --net $2 --data $D/val-g10.bin --n 20000 --buckets 8 --king-buckets kb8m --out $O/$1.perm.bin 2>/dev/null | grep active)"
}
chain() {  # name "lever flags"
  local n=$1 X=$2
  log "start $n ($X): $(echo $FILES | wc -w) files, $N positions"
  $G train $DATA $C $X --epochs 4 --lr 1e-3 --lr-end 1e-5 --out $O/$n.bin > $L/$n.log 2>&1 &&
  $G train $DATA $C $X --king-buckets m1 --init $O/$n.bin --epochs 1 --lr 2e-4 --lr-end 1e-5 --keep-best --out $O/${n}m1.bin > $L/${n}m1.log 2>&1 &&
  $G train $DATA $C $X --king-buckets kb8m --factorise --init $O/${n}m1.bin --epochs 2 --lr 2e-4 --lr-end 1e-6 --keep-best --out $O/${n}kb8m.bin > $L/${n}kb8m.log 2>&1 || { log "$n chain failed"; return 1; }
  grep '^epoch\|keep-best' $L/${n}kb8m.log | cut -c1-130 | while read l; do log "$n kb8m: $l"; done
  measure $n $O/${n}kb8m.bin
}
measure pilot $D/pw/s1024kb8m.bin
[ "${BASE:-0}" = 1 ] && chain base ""
for v in ${@:-sp1 sp2 ql2}; do
  case $v in
    sp1) chain sp1 "--ft-l1-group $SP1" ;;
    sp2) chain sp2 "--ft-l1-group $SP2" ;;
    ql2) chain ql2 "--qat-l2" ;;
    *) log "unknown variant $v" ;;
  esac
done
log "done"
