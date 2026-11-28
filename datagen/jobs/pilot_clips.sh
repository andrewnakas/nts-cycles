#!/bin/bash
# Teacher pilot inputs (plan phase 1): 8 clips x 33 frames of production Cycles (--look ref5p2, 48 spp, native,
# view +0.3 EV = looks/ref5p3), cut to 480x832 around the face by tools/pilot_crop.py. Clip 0 is zero-day 400-432,
# where reference frames exist; 1-4 use the episode's own character, 5-7 unseen procgen people (seeds 301-303).
# EXRs and npz are deleted after cropping. Waits for GPU gaps. Run (4050, in screen): pilot_clips.sh
O=$HOME/nts-cycles/data/pilot; mkdir -p $O; cd ~/nts-cycles
N=33
clips="zd zero-day 400 -
$(for s in 401 402 403 404; do echo "ep$s $(NTS_PICK_LEN=$N ~/scene-project/py tools/stage_a_pick.py $s) -"; done)
$(for s in 301 302 303; do echo "pg$s $(NTS_PICK_LEN=$N ~/scene-project/py tools/stage_a_pick.py $s) $s"; done)"
echo "$clips" > $O/clips.txt
while read name ep f pg; do
  d=$O/$name; [ -f $d/crop.json ] && continue
  while pgrep -x blender >/dev/null; do sleep 30; done
  envs=$([ "$pg" != - ] && echo "NTS_PROCGEN=$pg")
  t0=$(date +%s)
  (cd ~/scene-project && env $envs ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- \
     ~/scene-project/$ep --pathtrace --frames $f-$((f + N - 1)) --look ref5p2 --nts native --samples 48 \
     --nts-out $O/tmp_$name) > $O/$name.log 2>&1
  ~/scene-project/py tools/stage_a_pack.py $O/tmp_$name $O/tmp_$name.npz >> $O/$name.log 2>&1
  ex=$([ $name = zd ] && echo "--extra $HOME/nts-cycles/data/eval/zd/ref5_pass2_clip=ref5")
  ~/scene-project/py tools/pilot_crop.py $O/tmp_$name.npz $O/tmp_$name/meta.json $d 0.3 $ex >> $O/$name.log 2>&1 \
    && echo "$name $ep $f $(( $(date +%s) - t0 ))s ok" >> $O/log.txt || echo "$name $ep $f FAIL" >> $O/log.txt
  rm -rf $O/tmp_$name $O/tmp_$name.npz
done <<< "$clips"
echo "PILOT_CLIPS_DONE $(date +%H:%M)" >> $O/log.txt
