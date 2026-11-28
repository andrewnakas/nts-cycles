#!/bin/bash
# Rasterised (Eevee) albedo guide for baselines/guided.py, zero-day 400-459: a full-res unjittered guide and a low-res
# one at the Cycles input's exact jitter (perf, seed 0, Halton start 0), so both albedos come from the same renderer.
D=~/nts-cycles/data/raw
gap() { while pgrep -x blender >/dev/null; do sleep 10; done; }
for run in "zd_eguide_native:native" "zd_eguide_perf:perf"; do
  IFS=: read name mode <<< "$run"
  gap
  rm -rf $D/$name
  (cd ~/scene-project && ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/zero-day \
     --frames 400-459 --nts $mode --nts-out $D/$name --nts-seed 0 --nts-halton 0) < /dev/null > $D/$name.log 2>&1
  a=$(stat -c %Y $D/$name/f_0400.exr 2>/dev/null); b=$(stat -c %Y $D/$name/f_0459.exr 2>/dev/null)
  echo "$name frames=$(ls $D/$name/f_*.exr 2>/dev/null | wc -l) steady=$(echo "scale=2; ($b-$a)/59" | bc)s/frame"
done
cd ~/nts-cycles
V3="--reject 0.15 --reject-face 0.02 --sigma 0.7"
nice ~/scene-project/py baselines/guided.py data/raw/zd_perf_s0 $D/zd_eguide_native runs/guided/eg_pair --frames 400-459 $V3 \
   --lr-albedo $D/zd_eguide_perf > runs/guided/eg_pair.log 2>&1
nice ~/scene-project/py eval/metrics.py data/eval/zd/gt runs/guided/eg_pair --frames 400-459 --cuts data/raw/zd_perf_s0/cuts.json
nice ~/scene-project/py baselines/guided.py data/raw/zd_perf_s0 $D/zd_eguide_native runs/guided/eg_mixed --frames 400-459 $V3 \
   > runs/guided/eg_mixed.log 2>&1
nice ~/scene-project/py eval/metrics.py data/eval/zd/gt runs/guided/eg_mixed --frames 400-459 --cuts data/raw/zd_perf_s0/cuts.json
echo "EGUIDE TEST DONE"
