#!/bin/bash
# Strand hair test: zero-day f430 (hero) and procgen seed 2 (honeypot f190), ref5p2 look, 256 spp, shell-only vs strands.
# Waits for GPU gaps (the scene-project queue comes first).
O=$HOME/nts-cycles/data/strand_test
mkdir -p $O
for spec in "0:zero-day:430:0" "0:zero-day:430:120000,3" "2:honeypot:190:120000,3"; do
  IFS=: read seed ep f strands <<< "$spec"
  tag=${ep}_${f}_s${seed}_$( [ "$strands" = 0 ] && echo shell || echo strands )
  while pgrep -x blender >/dev/null; do sleep 15; done
  envs=""; [ "$seed" != 0 ] && envs="NTS_PROCGEN=$seed"; [ "$strands" != 0 ] && envs="$envs NTS_STRANDS=$strands"
  (cd ~/scene-project && env $envs ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- \
     ~/scene-project/$ep --pathtrace --frames $f-$f --look ref5p2 --nts native --nts-out $O/$tag --samples 256) \
     > $O/$tag.log 2>&1
  echo "$tag: $(grep -a -o "strand_hair: .*" $O/$tag.log | head -1) frames=$(ls $O/$tag/f_*.exr 2>/dev/null | wc -l) $(grep -a -o "Time:[^|]*" $O/$tag.log | tail -1)" >> $O/done.txt
done
echo STRAND_TEST_DONE >> $O/done.txt
