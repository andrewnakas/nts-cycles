#!/bin/bash
# Guide-mode renders (render_lr --nts guide: no denoise, no diffuse bounces, uniform samples) at 16 and 64 spp for
# zero-day 400-459; wall time per frame, then guided.py v3 settings scored against the native GT.
D=~/nts-cycles/data/raw
gap() { while pgrep -x blender >/dev/null; do sleep 10; done; }
for spp in 16 64; do
  name=zd_gmode_s$spp
  gap
  rm -rf $D/$name; start=$(date +%s)
  (cd ~/scene-project && ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/zero-day \
     --pathtrace --frames 400-459 --nts guide --nts-out $D/$name --samples $spp) < /dev/null > $D/$name.log 2>&1
  echo "$name frames=$(ls $D/$name/f_*.exr 2>/dev/null | wc -l) wall=$(( $(date +%s) - start ))s"
  cd ~/nts-cycles
  nice ~/scene-project/py baselines/guided.py data/raw/zd_perf_s0 $D/$name runs/guided/gm_s$spp --frames 400-459 \
     --reject 0.15 --reject-face 0.02 --sigma 0.7 > /dev/null 2>&1
  nice ~/scene-project/py eval/metrics.py data/eval/zd/gt runs/guided/gm_s$spp --frames 400-459 --cuts data/raw/zd_perf_s0/cuts.json
done
echo "GUIDE TEST2 DONE"
