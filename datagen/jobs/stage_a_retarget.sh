#!/bin/bash
# Re-render stage A targets with the fixed strand colour (melanin model, 2026-11-09), at 512 spp (the melanin
# fibre mass doubled the 1024-spp time to ~13 min a pair; OIDN-denoised 512 is plenty for a target). Inputs unchanged.
# Same seed, episode and frame as the original pair (from log.txt), so geometry matches the existing in.npz.
# Replaces tgt.npz only after a successful pack; marks done pairs in data/stage_a/retarget.txt. Waits for GPU gaps.
O=$HOME/nts-cycles/data/stage_a
cd ~/nts-cycles
for d in $O/s*/; do
  s=$(basename $d); seed=${s#s}
  [ -f $d/in.npz ] && [ -f $d/tgt.npz ] || continue
  grep -qx $s $O/retarget.txt 2>/dev/null && continue
  free=$(df --output=avail -BM ~ | tail -1 | tr -dc 0-9)
  [ "$free" -lt 1500 ] && { echo "RETARGET STOP disk ${free}M at $s" >> $O/log.txt; break; }
  read ep f <<< "$(grep -a "^$s in " $O/log.txt | tail -1 | awk '{print $3, $4}')"
  [ -n "$ep" ] || continue
  while pgrep -x blender >/dev/null; do sleep 15; done
  rm -rf $d/tgt
  (cd ~/scene-project && NTS_PROCGEN=$seed NTS_STRANDS=150000,1.5 ~/blender-4.5/blender -b --gpu-backend vulkan \
     -P cutscene/scene.py -- ~/scene-project/$ep --pathtrace --frames $f-$((f + 1)) --look ref5p2 --nts hero \
     --samples 512 --nts-out $d/tgt) > $d/tgt2.log 2>&1
  ~/scene-project/py tools/stage_a_pack.py $d/tgt $d/tgt_new.npz --rgb-only >> $d/tgt2.log 2>&1 \
    && mv $d/tgt_new.npz $d/tgt.npz && echo $s >> $O/retarget.txt || echo "RETARGET FAIL $s" >> $O/log.txt
done
echo "RETARGET_DONE $(wc -l < $O/retarget.txt) $(date +%H:%M)" >> $O/log.txt
