#!/bin/bash
# A/B input clip: zero-day 420-431 (the reference frames, data/eval/zd), production settings + --look ref5p2,
# native 1080x1920. Starts after stage A stops; waits for GPU gaps.
O=$HOME/nts-cycles/data/eval/zd/ntsr_in
until grep -q "STOP\|STAGE_A_DONE" ~/nts-cycles/data/stage_a/log.txt; do sleep 30; done
while pgrep -x blender >/dev/null; do sleep 15; done
(cd ~/scene-project && ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/zero-day \
   --pathtrace --frames 420-431 --look ref5p2 --nts native --nts-out $O --samples 48) > $O.log 2>&1
echo "AB_CLIP_DONE $(ls $O/f_*.exr 2>/dev/null | wc -l) frames" >> ~/nts-cycles/data/stage_a/log.txt
