#!/bin/bash
# Variety A/B (2026-11-12): our candidate look on the 24 look-reference stills that have pass-2 refs
# (data/look/refs/{native,ref,m}_<ep>_f####.png, 9 episodes). 48 spp native + look, view +0.3 EV, film grain 2 ->
# data/look/variety/<look>/<ep>_f####.png. Waits for GPU gaps. Run (4050): [ONLY="<ep>_f#### ..."] [GRAIN=0] variety_v6.sh [look=v6] [ev=0.3]
LK=${1:-v6}; EVX=${2:-0.3}; R=~/nts-cycles; O=$R/data/look/variety/$LK; mkdir -p $O; cd $R
gpu_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 700 ]; }  # 6 GB card shared with another GPU job: wait until nothing else is on it
for f in data/look/refs/ref_*.png; do
  b=$(basename $f .png); b=${b#ref_}; ep=${b%_f*}; fr=$((10#${b##*_f}))
  [ -n "$ONLY" ] && [[ " $ONLY " != *" $b "* ]] && continue  # ONLY="zero-day_f0430 ..." for quick look iterations
  [ -f $O/${ep}_f$(printf %04d $fr).png ] && continue
  while pgrep -x blender >/dev/null || ! gpu_free; do sleep 30; done
  (cd ~/scene-project && ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/$ep \
     --pathtrace --frames $fr-$fr --look $LK --nts native --samples 48 --nts-out /tmp/var_$LK) > /tmp/var_$LK.log 2>&1
  ~/scene-project/py tools/stage_a_pack.py /tmp/var_$LK /tmp/var_$LK.npz --rgb-only >/dev/null 2>&1
  GRAIN=${GRAIN:-2} EVX=$EVX OUT=$O/${ep}_f$(printf %04d $fr).png ~/scene-project/py - /tmp/var_$LK.npz /tmp/var_$LK/meta.json <<'PY'
import json, os, sys, numpy as np
from PIL import Image
sys.path.insert(0, "."); from nts.tonemap import Tonemapper
z = np.load(sys.argv[1]); v = json.load(open(sys.argv[2]))["view"]; v[2] += float(os.environ["EVX"])
a = Tonemapper(*v)(z["rgb"][0].astype(np.float32).transpose(1, 2, 0)).astype(np.float32)
lum = a.mean(-1, keepdims=True) / 255
rng = np.random.default_rng(int(z["frames"][0]))
a = a + rng.normal(0, float(os.environ["GRAIN"]), a.shape[:2])[..., None] * (4 * lum * (1 - lum)).clip(0.25, 1)  # tools/film_grain.py, sigma 2
Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8)).save(os.environ["OUT"])
PY
  [ -f $O/${ep}_f$(printf %04d $fr).png ] && echo "$ep $fr ok $(date +%H:%M)" >> $O/log.txt || { echo "$ep $fr FAIL" >> $O/log.txt; cp /tmp/var_$LK.log $O/fail_${ep}_$fr.log; }
  rm -rf /tmp/var_$LK /tmp/var_$LK.npz
done
echo "VARIETY_DONE $LK" >> $O/log.txt
