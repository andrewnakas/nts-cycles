#!/bin/bash
# Whole-episode run, part 2: the edited key of each shot (data/v2/<tag>/<keys tag>_keys, pasted by v2_p7keys.py)
# carried over its shot with the e1 recipe -> data/v2/<tag>/<name>. One shot after the other; start several of
# these for shots in parallel. CPU only. Nothing is deleted.
# 2026-11-24: plus the border rules of design log row 122 (CARRY_EXTRA, default "--plain-fill --seam 3 --skin-agree 9";
# CARRY_EXTRA="" gives the plain e1 recipe, builds e1-e5).
#   v2_ep_carry.sh <keys tag> <name> "<tag>:<key frame> ..." [look=v11e]
cd ~/nts-cycles; KT=$1; NAME=$2; LK=${4:-v11e}
for s in $3; do t=${s%%:*}; k=${s##*:}
  GEOM=data/raw/${t}_$LK BASE=ns6e TONE=${TONE:-x29} EXTRA="--hair-rim 3 --plain-local 0.12 --follow 1 --stretch 2 ${CARRY_EXTRA---plain-fill --seam 3 --skin-agree 9}" bash tools/nr_clip.sh carry $t $KT $NAME $k
done
echo EP_CARRY_DONE $NAME "$3"
