#!/bin/bash
# Sequential 1-thread pipeline: nodes (962 d13, s100) + regret (696 d14) per arm.
Q=/private/tmp/claude-501/-Users-andreasholley-dev-stockfish-challenge/4c85bf7b-729e-4ca9-80d9-7d20684e957e/scratchpad/qs
ORC=/private/tmp/claude-501/-Users-andreasholley-dev-stockfish-challenge/4c85bf7b-729e-4ca9-80d9-7d20684e957e/scratchpad/tree/orc
cd $Q
arm() { # name envs...
  N=$1; shift
  [ -f runs/n-$N.json ] || env "$@" nice -n 20 python3 qrun.py bin/kn2 runs/n-$N.json --nostats
  env "$@" nice -n 20 python3 $ORC/evalcand.py $Q/bin/kn2 qs-$N --par 1 | tail -1
}
arm main TNOISE=100
arm noop99 TNOISE=99
arm A QWP=400 QWN=800 QWB=900 QEP=300 QEN=500 QEB=600 QER=800
arm P QWP=400 QEP=300
arm C QWP=300 QWN=600 QWB=700 QEP=200 QEN=400 QEB=500 QER=600
arm noop101 TNOISE=101
echo PIPE1DONE
