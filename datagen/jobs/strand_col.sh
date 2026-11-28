#!/bin/bash
# Strand colour calibration on zero-day f430 (hero), 256 spp: shell reference vs strands at several NTS_STRAND_COL.
O=$HOME/nts-cycles/data/strand_test
for v in 0.45,0.4 0.35,0.3; do
  while pgrep -x blender >/dev/null; do sleep 15; done
  (cd ~/scene-project && NTS_STRANDS=150000,1.5 NTS_STRAND_MEL=$v ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- \
     ~/scene-project/zero-day --pathtrace --frames 430-430 --look ref5p2 --nts native --nts-out $O/mel_$v --samples 256) > $O/mel_$v.log 2>&1
done
echo STRAND_COL_DONE >> $O/done.txt
