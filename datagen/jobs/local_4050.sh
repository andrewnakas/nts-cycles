#!/bin/bash
# Training data on the 4050, one shot at a time so the disk never holds more than one shot's raw EXRs:
# render low-res inputs (N seeds, random Halton start) + native GT -> pack to data/packed/<name>_s<seed> ->
# optionally push to the cloud box -> delete the raw EXRs.
#   datagen/jobs/local_4050.sh <episode> <a-b> <name> [seeds=2] [push_host]
# Pick shots from sets the test set doesn't use (zero-day = bullpen/devden/office/startup/studio), e.g.
#   agentic-time (kitchen, ruins, serverroom, street), last-programmer (bedroom), ram-shortage.
# Skips (and waits) while another Blender runs, so it never fights the render queue for VRAM.
set -euo pipefail
EP=$1 RANGE=$2 NAME=$3 SEEDS=${4:-2} PUSH=${5:-}
PY=~/scene-project/py
D=~/nts-cycles/data/raw
P=~/nts-cycles/data/packed
mkdir -p $D $P
wait_gpu() { while pgrep -x blender > /dev/null; do sleep 30; done; }
render() {  # out mode seed halton
  wait_gpu
  (cd ~/scene-project && ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/$EP \
     --pathtrace --frames $RANGE --nts $2 --nts-out $1 --nts-seed $3 --nts-halton $4 --samples 48) > $1.log 2>&1
}
render $D/${NAME}_native native 0 0
for s in $(seq 0 $((SEEDS - 1))); do
  render $D/${NAME}_lr$s perf $((s + 1)) $((RANDOM % 32))
  $PY ~/nts-cycles/nts/datasets/pack.py $D/${NAME}_lr$s $D/${NAME}_native $P/${NAME}_s$s
  rm -rf $D/${NAME}_lr$s
done
rm -rf $D/${NAME}_native
if [ -n "$PUSH" ]; then
  rsync -az $P/${NAME}_s* $PUSH:nts-cycles/data/packed/ && rm -rf $P/${NAME}_s*
fi
echo "$NAME done: $(du -sh $P | cut -f1) packed locally; $(df -h ~ | tail -1 | awk '{print $4}') free"
