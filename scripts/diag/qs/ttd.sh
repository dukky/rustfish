#!/bin/bash
# Interleaved time-to-depth: pos962 depth 13, 1 thread, plain builds.
cd /private/tmp/claude-501/-Users-andreasholley-dev-stockfish-challenge/4c85bf7b-729e-4ca9-80d9-7d20684e957e/scratchpad/qs
while pgrep -f "pipe2.sh" >/dev/null; do sleep 10; done
for r in 1 2 3; do for b in main-clean P A; do
  nice -n 20 python3 qrun.py bin/$b runs/t-$b-r$r.json --nostats
done; done
echo TTDDONE
