#!/bin/bash
# One native render with light-group passes: lg_frame.sh <episode> <frames a,b,c> <out_name>
EP=$1 FR=$2 OUT=~/nts-cycles/data/look/lg/$3
mkdir -p ~/nts-cycles/data/look/lg
gap() { while true; do if ! pgrep -x blender >/dev/null && { pgrep -f "nts/datasets/[p]ack.py" >/dev/null || ! pgrep -f "train_v0[.]sh" >/dev/null; } && ! pgrep -f "look_search[.]sh" >/dev/null; then return; fi; sleep 3; done; }
ranges=$(echo $FR | sed -E "s/([0-9]+)/\1-\1/g")
gap
cd ~/scene-project && NTS_LIGHTGROUPS=1 ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/$EP \
  --pathtrace --frames $ranges --nts native --nts-out $OUT --samples 64 > $OUT.log 2>&1
echo "LG DONE $EP $FR $(ls $OUT/f_*.exr 2>/dev/null | wc -l) frames"
