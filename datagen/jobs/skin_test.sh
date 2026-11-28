#!/bin/bash
# Skin look test (native 1080x1920, zero-day frames 420-431): which material correction reproduces the reference look.
# Starts each render only while the training job is packing (no Blender running), so the two never share VRAM.
D=~/nts-cycles/data/raw
gap() { while true; do if ! pgrep -x blender >/dev/null && pgrep -f "nts/datasets/[p]ack.py" >/dev/null; then return; fi
        if ! pgrep -f "train_v0[.]sh" >/dev/null && ! pgrep -x blender >/dev/null; then return; fi; sleep 3; done; }
for v in "skA:value=0.7,sat=1.1,detail=0.6" "skB:value=0.55,sat=1.15,detail=0.6"; do
  name=${v%%:*}; spec=${v#*:}
  gap
  (cd ~/scene-project && NTS_SKIN=$spec ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- \
     ~/scene-project/zero-day --pathtrace --frames 420-431 --nts native --nts-out $D/zd_$name --samples 48) > $D/zd_$name.log 2>&1
  ~/scene-project/py ~/nts-cycles/tools/exr_to_png.py $D/zd_$name ~/nts-cycles/data/eval/zd/skin_$name
  echo "$name ($spec) frames=$(ls $D/zd_$name/f_*.exr | wc -l)"
done
echo SKIN DONE
