#!/bin/bash
# 4050: render_roi.sh <look> "<ep>:<a-b>:<tag> ..." [spp=128]: head-box renders of whole clips for the generative
# model's evaluation -> data/heads/train_eval_<tag>/ (tools/pack_heads.py + heads_to_train.py); EXRs are deleted.
LK=$1; SPP=${3:-128}; R=~/nts-cycles; cd $R
gpu_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 700 ]; }
for spec in $2; do
  IFS=: read ep fr tag <<< "$spec"
  [ -f data/heads/train_eval_$tag/eval_$tag.boxes.json ] && continue
  while pgrep -x blender >/dev/null || ! gpu_free; do sleep 20; done
  T=$R/data/tmp/roi_$tag; rm -rf $T
  (cd ~/scene-project && ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/$ep \
     --pathtrace --frames $fr --look $LK --nts roi --samples $SPP --nts-out $T) > runs/roi_$tag.log 2>&1
  ~/scene-project/py tools/pack_heads.py $T data/heads/eval_$tag.npz | tail -1
  ~/scene-project/py tools/heads_to_train.py data/heads/train_eval_$tag data/heads/eval_$tag.npz | tail -1
  rm -rf $T data/heads/eval_$tag.npz
done
