#!/bin/bash
cd ~/agent-runs/qs/see
Q=~/agent-runs/qs/qrun.py
until grep -q ST99DONE p1.log; do sleep 10; done
arm() { # name envs...
  local n=$1; shift
  for s in 100 99 101; do env TNOISE=$s "$@" nice -n 19 python3 $Q ./kn2 runs/n-$n-s$s.json --nostats; done
}
arm pc100 PCSH=-100
arm pc200 PCSH=-200
arm pcev PCEV=1
arm qcap400 QPCAP=400
arm qcap300 QPCAP=300
arm ccap400 CPCAP=400
arm cpev CPEV=1
arm cpm125 CPM=125
arm cpm80 CPM=80
echo NODESDONE
