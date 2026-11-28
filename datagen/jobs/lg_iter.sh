#!/bin/bash
# One look-fit iteration around a previous fit: light-group renders of the 10 look-basis frames with that fit's skin
# and hair as real materials (lights at 1), plus a skin basis around it, then refit and add absolute settings.
#   lg_iter.sh <fit.json> <tag>  -> data/look/lg_<tag>/<ep>_lg{,_skinv,_skins,_skinh}, runs/look/fit_<tag>.json
FIT=$(realpath $1) TAG=$2
R=~/nts-cycles/data/look/lg_$TAG
cd ~/nts-cycles && ~/scene-project/py tools/look_compose.py base $FIT $R
gap() { while pgrep -x blender >/dev/null; do sleep 10; done; }
for spec in "zero-day:295,430,605" "agentic-time:170" "last-programmer:450" "ram-shortage:950" "honeypot:190" \
            "please-thank-you:355" "center-a-div:500" "pair-programming:50"; do
  ep=${spec%%:*}; fr=${spec#*:}; ranges=$(echo $fr | sed -E "s/([0-9]+)/\1-\1/g")
  for v in base v s h; do
    suf=""; [ $v != base ] && suf=_skin$v
    OUT=$R/${ep}_lg$suf
    free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); [ "$free" -lt 6 ] && { echo "STOP: ${free}G free"; exit 1; }
    gap
    (cd ~/scene-project && NTS_LIGHTGROUPS=1 NTS_LOOK=$R/look_$v.json ~/blender-4.5/blender -b --gpu-backend vulkan \
       -P cutscene/scene.py -- ~/scene-project/$ep --pathtrace --frames $ranges --nts native --nts-out $OUT \
       --samples 64) < /dev/null > $OUT.log 2>&1
  done
  echo "ITER $TAG $ep done"
done
cd ~/nts-cycles
dirs=$(ls -d $R/*_lg | tr "\n" ",")
nice ~/scene-project/py nts/lookfit.py fit data/look/brief_v1.json ${dirs%,} runs/look/fit_$TAG.json \
  --holdout zero-day_f0295,please-thank-you_f0355,honeypot_f0190,ram-shortage_f0950,center-a-div_f0500,agentic-time_f0170 \
  --steps 15 > runs/look/fit_$TAG.log 2>&1
~/scene-project/py tools/look_compose.py absolute $R/look_base.json runs/look/fit_$TAG.json
grep -E "^ +[a-z].*->" runs/look/fit_$TAG.log
echo "ITER $TAG FIT DONE"
