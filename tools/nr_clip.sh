#!/bin/bash
# 4050: the NR-1 v2 route (build y2, the design log rows 100-103) for ONE clip, from the native render to the
# frames that go to the episode's finish. CPU only; the keyframe edit itself runs on Kaggle
# (Mac: tools/kaggle/push_pilot.sh p2 "<min>|n|<keys>|sd:t85x3@n+sdonly+nofull|").
#   nr_clip.sh base  <tag>                        steadied base + tone -> data/v2/<tag>/{$BASE, ${TONE}t, ${TONE}w}
#   nr_clip.sh keys  <tag> <frame[,frame...]>     toned keyframes + head boxes for the edit -> data/v2/pilot2/keys/
#                                                 (key_<tag>_<frame>.png, boxes.json; the old boxes.json is kept
#                                                 as boxes.json.before_<tag>)
#   nr_clip.sh carry <tag> <keys tag> <name> <frame[,frame...]>
#                                                 the edited keys (data/v2/<tag>/<keys tag>_keys, pasted there by
#                                                 runs/v2_p7keys.py <kind> <keys tag> ${TONE}t) carried over the shot
#                                                 and composed -> data/v2/<tag>/<name>
# One key per shot: the frame list holds one frame of every shot of the clip (cuts: <native>/cuts.json).
# Env: NATIVE (default data/eval/ab/<tag>/native), MASKS (default data/eval/ab/<tag>/masks),
#      GEOM (slim EXRs with Depth, Vector, Region; default data/raw/<tag>_native, else data/raw/<tag>_v9d: the same
#      geometry in both), SURF (the clip's surf pass, render_lr.py mode "surf"; default data/raw/<tag>_surf),
#      BASE=ns6w, TONE=x19.
#      TONE_EXTRA: more nts.tone options for the base step (e.g. "--garment-curve 1.8").
#      TIN: the tone tag the keys were edited from, when it is not TONE (a re-render of the clip with another cloth
#      look: the keys stay, their detail is taken against the frames they were made from and put on the new ones).
#      NTS_WORKERS: frames processed in parallel by the tone and carry steps (nts/par.py; 0 = one per core, default 1;
#      the output is the same bytes at any count; about 1.4 GB per worker). NTS_THREADS: the steadier's threads (3).
# Nothing is deleted; a step refuses to write into a folder that already has frames.
cd ~/nts-cycles; export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
P="nice -n 10 $HOME/scene-project/py"; CMD=$1; T=$2
[ -z "$T" ] && sed -n '2,18p' $0 && exit 1
A=data/eval/ab/$T; O=data/v2/$T; BASE=${BASE:-ns6w}; TONE=${TONE:-x19}
NATIVE=${NATIVE:-$A/native}; MASKS=${MASKS:-$A/masks}
if [ -z "$GEOM" ]; then GEOM=data/raw/${T}_native; [ -n "$(ls $GEOM/*.exr 2>/dev/null | head -n 1)" ] || GEOM=data/raw/${T}_v9d; fi
count() { ls $1 2>/dev/null | grep -c "^f_"; }
fresh() { [ "$(count $1)" -gt 0 ] && echo "$1 already has frames: pick another name" && exit 1; }
# the y2 recipe; the options given first win over the defaults that follow them in the carry call
Y2="--outline 1.5 --hair-over --own 0 --fine-keep 1 --limit 120 --hair-limit 120 --hp 128 --hp-rel 3.5 --skin-mid 0.3 --skin-mid-plain 1.0 --skin-limit-plain 120 --plain-soft $EXTRA"   # EXTRA: more gen.carry options for a trial build
case "$CMD" in
base)
  fresh $O/$BASE; fresh $O/${TONE}t; mkdir -p $O
  $P -m gen.steady2 $GEOM $NATIVE $NATIVE $O/$BASE --bands 2 --energy-alpha 0.2 --fine-alpha 0.08 --fine-tol 40 --edge-reset 12 2>&1 | tail -n 1
  $P -m nts.tone apply $O/$BASE $MASKS $O/${TONE}t --geom $GEOM --tone 0.7 --share-power 0.5 --rest-tone 1.0 --garment-tone 0.8 \
     $TONE_EXTRA --weights-out $O/${TONE}w --params $O/$TONE.json 2>&1 | tail -n 1
  echo "NR_BASE_DONE $T base $(count $O/$BASE) tone $(count $O/${TONE}t)"
  ;;
keys)
  K=data/v2/pilot2/keys; mkdir -p $K; [ -f $K/boxes.json ] && cp -n $K/boxes.json $K/boxes.json.before_$T
  $P - "$T" "$3" "$O/${TONE}t" "$MASKS" "$K" <<'PY'
import json, os, sys
import numpy as np
from PIL import Image
t, frames, tone, masks, K = sys.argv[1:6]
bp = K + "/boxes.json"
boxes = json.load(open(bp)) if os.path.exists(bp) else {}
for f in (int(x) for x in frames.split(",")):
    im = Image.open("%s/f_%04d.png" % (tone, f)).convert("RGB")
    name = "key_%s_%d.png" % (t, f)
    im.save("%s/%s" % (K, name))
    m = np.asarray(Image.open("%s/m_%04d.png" % (masks, f)).convert("RGB"))
    ys, xs = np.nonzero((m[..., 0] > 127) | (m[..., 2] > 127))  # face or hair
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    s = int(min(1.5 * max(xs.max() - xs.min(), ys.max() - ys.min()), im.width, im.height)) // 2 * 2
    x0 = int(min(max(cx - s / 2, 0), im.width - s))
    y0 = int(min(max(cy - s / 2, 0), im.height - s))
    boxes[name] = [x0, y0, x0 + s, y0 + s]
    print(name, boxes[name])
json.dump(boxes, open(bp, "w"), indent=1)
PY
  ;;
carry)
  KT=$3; NAME=$4; FR=$5; fresh $O/$NAME; S="--surf ${SURF:-data/raw/${T}_surf}"
  $P -m gen.carry $GEOM $O/${KT}_keys $O/${TIN:-$TONE}t $O/$NAME --keys $FR --compose $O/${TONE}t $O/${TONE}w --masks $MASKS $S $Y2 \
     --skin-limit 6 --limit 12 --skin-even 0 --hp 16 --hp-rel 28 --hp-ref 480 --soft 0 --agc 0.8 --split 1 2>&1 | tail -n 2
  echo "NR_CARRY_DONE $T $NAME $(count $O/$NAME)"
  ;;
*) sed -n '2,18p' $0 ;;
esac
