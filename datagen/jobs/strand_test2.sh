#!/bin/bash
# Strand hair v3 (length capped at the shell edge): zero-day f430 hero, 256 spp. Waits for GPU gaps.
O=$HOME/nts-cycles/data/strand_test
while pgrep -x blender >/dev/null; do sleep 15; done
(cd ~/scene-project && NTS_STRANDS=150000,1.5 ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- \
   ~/scene-project/zero-day --pathtrace --frames 430-430 --look ref5p2 --nts native --nts-out $O/zd430_v3 --samples 256) \
   > $O/zd430_v3.log 2>&1
echo "zd430_v3: $(grep -a -o "strand_hair: .*" $O/zd430_v3.log | head -1) frames=$(ls $O/zd430_v3/f_*.exr 2>/dev/null | wc -l)" >> $O/done.txt
while pgrep -x blender >/dev/null; do sleep 15; done
(cd ~/scene-project && NTS_PROCGEN=2 NTS_STRANDS=150000,1.5 ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- \
   ~/scene-project/honeypot --pathtrace --frames 190-190 --look ref5p2 --nts native --nts-out $O/hp190_v3 --samples 256) \
   > $O/hp190_v3.log 2>&1
echo "hp190_v3: frames=$(ls $O/hp190_v3/f_*.exr 2>/dev/null | wc -l)" >> $O/done.txt
echo STRAND_V3_DONE >> $O/done.txt
