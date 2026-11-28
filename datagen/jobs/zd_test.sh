#!/bin/bash
# zero-day typing shot (frames 400-459): NTS low-res input (perf, seed 0) and native ground truth, both with region masks
cd ~/scene-project
D=~/nts-cycles/data/raw
for run in "zd_perf_s0:perf:0" "zd_native:native:0"; do
  IFS=: read name mode seed <<< "$run"
  rm -rf $D/$name; start=$(date +%s)
  ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/zero-day --pathtrace \
     --frames 400-459 --nts $mode --nts-out $D/$name --nts-seed $seed --samples 48 > $D/$name.log 2>&1
  echo "$name frames=$(ls $D/$name/f_*.exr 2>/dev/null | wc -l) wall=$(( $(date +%s) - start ))s"
done
