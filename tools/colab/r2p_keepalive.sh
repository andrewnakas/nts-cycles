#!/bin/bash
# Mac, background: keep a Colab runtime from being reclaimed while r2p_two.sh runs train, and copy checkpoints home.
#   tools/colab/r2p_keepalive.sh <session> [tags=f1]
# Every 4 min: one short kernel exec (counts as activity). Every 30 min: pull last.pt, train.log and the newest sample
# of each run to runs/r2p_<tag>{a,b}. Stops when both runs wrote DONE.json or the runtime is gone (never re-creates it:
# the ssh proxy would silently create a fresh CPU runtime).
R=~/Documents/nts-cycles; S=$1; TAG=${2:-f1}; cd $R
C() { HOME=$HOME/.colab-account-home colab "$@" 2>&1 | grep -v '^\[colab\]'; }
i=0
while true; do
  if ! C sessions | grep "^\[$S\]" | grep -q "Variant: GPU"; then echo "$(date +%H:%M) RUNTIME GONE"; exit 1; fi
  P=/tmp/r2p_ka_$S.py; cat > $P <<'PY'
import os, glob
for t in "ab":
    l = sorted(glob.glob(f"/content/k{t}/working/out/train.log"))
    last = open(l[0]).read().strip().splitlines()[-1][:90] if l else "(no log)"
    print(t, "DONE" if os.path.exists(f"/content/k{t}/working/out/DONE.json") else "RUN", last)
PY
  st=$(C exec -s $S --timeout 60 -f $P)
  echo "$(date +%H:%M) $st" | tr '\n' ' '; echo
  if [ $((i % 8)) = 0 ] || [ $(echo "$st" | grep -c DONE) = 2 ]; then
    for t in a b; do mkdir -p runs/r2p_$TAG$t
      tools/colab/cssh.sh $S "cd /content/k$t/working/out 2>/dev/null && tar cf - \$(ls sample_*.jpg 2>/dev/null | tail -1) train.log last.pt DONE.json 2>/dev/null" < /dev/null | tar xf - -C runs/r2p_$TAG$t 2>/dev/null
    done
  fi
  [ $(echo "$st" | grep -c DONE) = 2 ] && { echo "BOTH DONE"; exit 0; }
  i=$((i + 1)); sleep 240
done
