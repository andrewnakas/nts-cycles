#!/bin/bash
# Clean-room look search round: one Blender session, frame 430 rendered per candidate, then ranked against the
# visual reference. Waits for a gap in the training job (packing, no Blender) so they never share VRAM.
ROUND=$1
OUT=~/nts-cycles/data/look/$ROUND
gap() { while true; do if ! pgrep -x blender >/dev/null && { pgrep -f "nts/datasets/[p]ack.py" >/dev/null || ! pgrep -f "train_v0[.]sh" >/dev/null; }; then return; fi; sleep 3; done; }
gap
cd ~/scene-project && NTS_LOOK_RES=${NTS_LOOK_RES:-1080x1920} NTS_LOOK_SAMPLES=${NTS_LOOK_SAMPLES:-48} NTS_LOOK_SEARCH=~/nts-cycles/datagen/look_$ROUND.json ~/blender-4.5/blender -b --gpu-backend vulkan \
  -P cutscene/scene.py -- ~/scene-project/zero-day --pathtrace --frames 430-430 --nts native --nts-out $OUT > $OUT.log 2>&1
Z=~/nts-cycles/data/eval/zd
~/scene-project/py ~/nts-cycles/eval/look_score.py $OUT $Z/ref5_x5_f0430/chain_2.png $Z/masks/m_0430.png --top 12
echo LOOK DONE
