#!/bin/bash
cd ~/agent-runs/qs/see
Q=~/agent-runs/qs/qrun.py
SEELOG=runs/st100.log nice -n 19 python3 $Q ./st runs/st100.json --nostats
echo ST100DONE
for s in 100 99 101; do env TNOISE=$s nice -n 19 python3 $Q ./kn runs/n-noop$s.json --nostats; done
echo NOOPDONE
TNOISE=99 SEELOG=runs/st99.log nice -n 19 python3 $Q ./st runs/st99.json --nostats
echo ST99DONE
