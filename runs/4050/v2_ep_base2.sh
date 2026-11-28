#!/bin/bash
# Second worker for the per-shot base + tone step of a whole-episode run (runs/v2_ep_render.sh does the shots in
# order; this one takes the shots given, so two shots are in work at a time). A shot that already has frames is
# refused by nr_clip.sh and skipped by whichever worker comes second. CPU only. Nothing is deleted.
#   v2_ep_base2.sh "<tag> <tag> ..." [look=v11e]
cd ~/nts-cycles; LK=${2:-v11e}
for t in $1; do
  until [ -f data/eval/ab/$t/$LK/DONE ]; do sleep 60; done
  NATIVE=data/eval/ab/$t/$LK GEOM=data/raw/${t}_$LK BASE=ns6e TONE=${TONE:-x29} TONE_EXTRA="${TONE_EXTRA:---garment-curve 1.8 --fringe-skin --skin-rim --skin-close --shot-hold --hands-tight}" bash tools/nr_clip.sh base $t
done
echo EP_BASE2_DONE "$1"
