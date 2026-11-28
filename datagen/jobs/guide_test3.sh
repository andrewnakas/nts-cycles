#!/bin/bash
# Guide mode with lights and world off (NTS_GUIDE_DARK=1) at 32 and 64 spp: per-frame cost and guided v3 score.
D=~/nts-cycles/data/raw
gap() { while pgrep -x blender >/dev/null; do sleep 10; done; }
for spp in 64 32; do
  name=zd_gdark_s$spp
  gap
  rm -rf $D/$name
  (cd ~/scene-project && NTS_GUIDE_DARK=1 ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- \
     ~/scene-project/zero-day --pathtrace --frames 400-459 --nts guide --nts-out $D/$name --samples $spp) < /dev/null > $D/$name.log 2>&1
  a=$(stat -c %Y $D/$name/f_0400.exr); b=$(stat -c %Y $D/$name/f_0459.exr)
  echo "$name frames=$(ls $D/$name/f_*.exr | wc -l) steady=$(echo "scale=2; ($b-$a)/59" | bc)s/frame overlap=$(pgrep -x blender | wc -l)"
  cd ~/nts-cycles
  nice ~/scene-project/py baselines/guided.py data/raw/zd_perf_s0 $D/$name runs/guided/gd_s$spp --frames 400-459 \
     --reject 0.15 --reject-face 0.02 --sigma 0.7 > /dev/null 2>&1
  nice ~/scene-project/py eval/metrics.py data/eval/zd/gt runs/guided/gd_s$spp --frames 400-459 --cuts data/raw/zd_perf_s0/cuts.json
done
echo "GUIDE TEST3 DONE"
