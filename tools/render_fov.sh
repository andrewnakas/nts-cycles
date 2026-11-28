#!/bin/bash
# 4050: render_fov.sh <look> <frames a-b> <ep> <tag> [roi spp=128] [lr spp=48]: the two NTS-F inputs for a clip,
# data/raw/<tag>_lr (540x960 jittered, whole frame) and data/raw/<tag>_roi (1080x1920, head box only), with timings
# appended to runs/timing.md. Waits for GPU gaps. Then: ~/scene-project/py -m nts.pipeline data/raw/<tag>_lr data/raw/<tag>_roi <out>
LK=$1; FR=$2; EP=$3; TG=$4; RS=${5:-128}; LS=${6:-48}; R=~/nts-cycles; cd $R; mkdir -p runs
N=$(( ${FR#*-} - ${FR%-*} + 1 ))
for M in perf:$LS:lr roi:$RS:roi; do
  IFS=: read mode spp suf <<< "$M"
  O=$R/data/raw/${TG}_$suf; rm -rf $O
  while pgrep -x blender >/dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 700 ]; do sleep 30; done  # 6 GB card shared with another GPU job
  T0=$(date +%s.%N)
  (cd ~/scene-project && ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/$EP \
     --pathtrace --frames $FR --look $LK --nts $mode --samples $spp --nts-out $O) > runs/fov_${TG}_$suf.log 2>&1
  T1=$(date +%s.%N)
  # per-frame render time from Blender's own log (excludes the scene build), and wall time including the build
  PF=$(grep -a -o "^ *Time: [0-9:.]*" runs/fov_${TG}_$suf.log | awk -F"[: ]+" '{x=$(NF-1)*60+$NF; n++; t+=x} END{if(n) printf "%.2f", t/n}')
  echo "| $TG | $EP $FR | $mode | $spp | $N | $PF | $(echo "$T1 - $T0" | bc | cut -c1-6) |" | tee -a runs/timing.md
  grep -a "^roi:" runs/fov_${TG}_$suf.log
  ls $O/*.exr 2>/dev/null | wc -l
done
