#!/bin/bash
cd ~/agent-runs/qs/see
E=~/agent-runs/qs/orc/evalcand.py
until grep -q NODESDONE p2.log; do sleep 10; done
nice -n 19 python3 pscale.py > runs/pscale.txt 2>&1
rg() { local n=$1; shift; env "$@" nice -n 19 python3 $E ./kn2 see-$n --par 1; }
rg noop100 TNOISE=100
rg noop99 TNOISE=99
rg noop101 TNOISE=101
for a in "pc100 PCSH=-100" "pc200 PCSH=-200" "pcev PCEV=1" "qcap400 QPCAP=400" "qcap300 QPCAP=300" "ccap400 CPCAP=400" "cpev CPEV=1" "cpm125 CPM=125" "cpm80 CPM=80"; do
  set -- $a; rg $1 TNOISE=100 $2
done
rg noop98 TNOISE=98
rg noop102 TNOISE=102
echo REGDONE
