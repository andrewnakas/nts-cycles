#!/bin/bash
# Cheap full-res guide renders for baselines/guided.py: zero-day 400-459 in native mode at 1 and 4 spp.
# Measures wall time per frame, then scores guided.py with each guide against the native GT.
# Waits for GPU gaps like lg_frame.sh, and after the look jobs (they and the reference job go first).
gap() { while true; do if ! pgrep -x blender >/dev/null && { pgrep -f "nts/datasets/[p]ack.py" >/dev/null || ! pgrep -f "train_v0[.]sh" >/dev/null; } && ! pgrep -f "look_search[.]sh" >/dev/null && ! pgrep -f "lg_all[.]sh|lg_batch[.]sh|lg_skin_basis[.]sh" >/dev/null; then return; fi; sleep 10; done; }
D=~/nts-cycles/data/raw
cd ~/scene-project
for spp in 1 4; do
  name=zd_guide_s$spp
  gap
  rm -rf $D/$name; start=$(date +%s)
  ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/zero-day --pathtrace \
     --frames 400-459 --nts native --nts-out $D/$name --samples $spp > $D/$name.log 2>&1
  echo "$name frames=$(ls $D/$name/f_*.exr 2>/dev/null | wc -l) wall=$(( $(date +%s) - start ))s"
  cd ~/nts-cycles
  nice ~/scene-project/py baselines/guided.py data/raw/zd_perf_s0 $D/$name runs/guided/g_s$spp --frames 400-459 > /dev/null 2>&1
  nice ~/scene-project/py eval/metrics.py data/eval/zd/gt runs/guided/g_s$spp --frames 400-459 --cuts data/raw/zd_perf_s0/cuts.json
  cd ~/scene-project
done
echo "GUIDE TEST DONE"
