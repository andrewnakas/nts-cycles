#!/bin/bash
# Skin basis for the look fit: the same light-group render with the hero skin's base colour changed one way at a time
# (value, saturation, hue), so nts/lookfit.py models skin albedo from real renders (SSS included) instead of an
# image-space multiply. Waits until the reference job has made both batches of references, so it never takes their gap.
#   lg_skin_basis.sh            -> data/look/lg/<episode>_lg_skin<v|s|h>/
J=~/nts-cycles/datagen/jobs/lg_frame.sh
refs_done() { n=$(grep -c "^REF5 .*DONE" ~/nts-cycles/data/look/refs/REF5_DONE 2>/dev/null); echo "${n:-0}"; }
while [ "$(refs_done)" -lt 2 ]; do sleep 60; done
echo "refs ready: $(date)"
for spec in "zero-day:295,430,605" "agentic-time:170" "last-programmer:450" "ram-shortage:950" "honeypot:190" \
            "please-thank-you:355" "center-a-div:500" "pair-programming:50"; do
  ep=${spec%%:*}; fr=${spec#*:}
  for b in "v value=0.55" "s sat=1.4" "h hue=-0.03"; do
    free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)
    if [ "$free" -lt 6 ]; then echo "STOP: ${free}G free"; exit 1; fi
    NTS_SKIN="${b#* }" $J $ep $fr ${ep}_lg_skin${b%% *}
  done
  echo "$ep skin basis done"
done
echo "SKIN BASIS DONE"
