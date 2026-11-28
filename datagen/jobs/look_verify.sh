#!/bin/bash
# Real renders of the look-basis frames with a look fit applied (NTS_LOOK -> render_lr.apply_look), to check the fit's
# prediction against Cycles itself (the look plan "done" criterion 3: within 0.5 dE).
#   look_verify.sh <fit.json> <tag>   -> data/look/verify/<tag>/<episode>/f_####.exr, then tools/look_verify.py
FIT=$(realpath $1) TAG=$2
gap() { while pgrep -x blender >/dev/null; do sleep 10; done; }
for spec in "zero-day:295,430,605" "agentic-time:170" "last-programmer:450" "ram-shortage:950" "honeypot:190" \
            "please-thank-you:355" "center-a-div:500" "pair-programming:50"; do
  ep=${spec%%:*}; fr=${spec#*:}; OUT=~/nts-cycles/data/look/verify/$TAG/$ep
  mkdir -p $OUT; ranges=$(echo $fr | sed -E "s/([0-9]+)/\1-\1/g")
  gap
  (cd ~/scene-project && NTS_LOOK=$FIT ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- \
     ~/scene-project/$ep --pathtrace --frames $ranges --nts native --nts-out $OUT --samples 64) < /dev/null > $OUT.log 2>&1
  echo "VERIFY RENDER $ep $(ls $OUT/f_*.exr 2>/dev/null | wc -l) frames $(grep -o "apply_look.*" $OUT.log | head -c 200)"
done
cd ~/nts-cycles && ~/scene-project/py tools/look_verify.py $FIT data/look/verify/$TAG ${LGROOT:-data/look/lg}
echo "LOOK VERIFY DONE $TAG"
