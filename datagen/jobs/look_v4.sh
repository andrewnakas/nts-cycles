#!/bin/bash
# Asset/look pass 2026-11-12: zero-day f430 at production quality (48 spp, native) with look variants, view +0.3 EV
# (as ref5p3), to data/look/v4/<look>_f0430.png to compare by eye/scoreboard with the reference. Run (4050): look_v4.sh
O=$HOME/nts-cycles/data/look/v4; mkdir -p $O; cd ~/nts-cycles
LOOKS=${*:-ref5p2 v4h v4s v4hs v4hsl}
for lk in $LOOKS; do
  [ -f $O/${lk}_f0430.png ] && continue
  while pgrep -x blender >/dev/null; do sleep 30; done
  (cd ~/scene-project && ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/zero-day \
     --pathtrace --frames 430-430 --look $lk --nts native --samples 48 --nts-out $O/tmp_$lk) > $O/$lk.log 2>&1
  ~/scene-project/py tools/stage_a_pack.py $O/tmp_$lk $O/tmp_$lk.npz >> $O/$lk.log 2>&1
  ~/scene-project/py - $O/tmp_$lk.npz $O/tmp_$lk/meta.json $O/${lk}_f0430.png >> $O/$lk.log 2>&1 <<'PY'
import json, sys, numpy as np
from PIL import Image
sys.path.insert(0, "."); from nts.tonemap import Tonemapper
z = np.load(sys.argv[1]); v = json.load(open(sys.argv[2]))["view"]; v[2] += 0.3
Image.fromarray(Tonemapper(*v)(z["rgb"][0].astype(np.float32).transpose(1, 2, 0))).save(sys.argv[3])
PY
  rm -rf $O/tmp_$lk $O/tmp_$lk.npz; echo "$lk $(date +%H:%M)" >> $O/log.txt
done
echo "LOOK_V4_DONE $LOOKS" >> $O/log.txt
