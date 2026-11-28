#!/bin/bash
# Native 1080x1920 stills (with region masks) for the look variety set: these are what the reference job turns into
# visual references (pass 2), and what the look search renders per candidate. Waits for training-job gaps.
R=~/nts-cycles/data/look/refs
mkdir -p $R
gap() { while true; do if ! pgrep -x blender >/dev/null && { pgrep -f "nts/datasets/[p]ack.py" >/dev/null || ! pgrep -f "train_v0[.]sh" >/dev/null; } && ! pgrep -f "look_search[.]sh" >/dev/null; then return; fi; sleep 3; done; }
for spec in "zero-day:60,185,295,430,500,605,720" "agentic-time:170,310" "last-programmer:450" "ram-shortage:790,950"; do
  ep=${spec%%:*}; fr=${spec#*:}
  ranges=$(echo $fr | sed -E "s/([0-9]+)/\1-\1/g")
  gap
  (cd ~/scene-project && ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/$ep --pathtrace --frames $ranges --nts native --nts-out $R/exr_$ep --samples 48) > $R/exr_$ep.log 2>&1
  ~/scene-project/py ~/nts-cycles/tools/exr_to_png.py $R/exr_$ep $R/png_$ep --masks $R/png_$ep
  for f in $(echo $fr | tr , " "); do
    F=$(printf %04d $f)
    cp $R/png_$ep/f_$F.png $R/native_${ep}_f$F.png
    cp $R/png_$ep/m_$F.png $R/m_${ep}_$F.png
  done
  rm -rf $R/exr_$ep
  echo "$ep done: $fr"
done
echo REFS DONE
