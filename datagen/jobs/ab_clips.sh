#!/bin/bash
# 4050: [SPP=48] [GEOM_ONLY=1] [GBUF_ONLY=1] ab_clips.sh <look|native> "<ep>:<a-b>:<tag> ...": clips for the moving A/B
# vs the reference, native 1080x1920 -> data/eval/ab/<tag>/<look>/f_####.png (AgX, view +0.3 EV, no grain) + masks/m_####.png
# + gbuf_<look>/f_####.{alb.jpg,nrm.png,dep.png} (nts/gbuf.py, for NR-1), and data/raw/<tag>_<look>/f_####.exr slimmed
# to Depth, Vector and Region (motion for the reference pass and for the metrics).
# "native" = the episode's own look (the reference pass's input, as in every earlier request). Waits for GPU gaps.
# GEOM_ONLY=1: 1 spp, no PNGs, only the slim EXRs (when just the motion vectors are missing).
# NOGBUF=1: no G-buffer images (the NR-1 v2 route does not read them; 270 MB per 60 frames).
# MASKS_SUFFIX=_x: the masks go to masks_x, not masks (a second look of a clip whose masks are in use: nothing is overwritten).
# GBUF_ONLY=1: 16 spp into a scratch folder, only the G-buffer images (clips rendered before 2026-11-18 lost theirs to
#              the strip); nothing else of the clip is touched.
# Work files stay on the home disk: /tmp is a 16 GB tmpfs shared with other jobs and filled up (2026-11-12).
LK=$1; R=~/nts-cycles; cd $R; SPP=${SPP:-48}; [ "$GEOM_ONLY" = 1 ] && SPP=1; [ "$GBUF_ONLY" = 1 ] && SPP=16
# The card is shared. The co-tenant's two GPU jobs (otherjob/guard.sh: the detector worker and its llama-server) yield by
# design: that guard stops them within seconds whenever a blender process appears, and restarts them after the card has
# been free for 60 s. So while that guard runs, their memory does not count as taken here; anything else does (a reference
# run, Unreal). Until 2026-11-22 this waited for under 700 MiB in total, which never came: they waited for a blender to
# appear, and this waited for them to leave.
H2=$OTHER_JOB_DIR
gmem() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p="$1" '$1==p{s+=$2} END{print s+0}'; }
gpu_free() {
  local used y=0 p g ok=0
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  for g in $(pgrep -f "^bash guard.sh"); do [ "$(readlink /proc/$g/cwd)" = $H2 ] && ok=1; done
  if [ $ok = 1 ] && [ "$(cat $H2/worker-mode 2>/dev/null)" = on ]; then
    for p in $(pgrep -x llama-server) $(pgrep -f "python -u gpu_dual_worker.py"); do y=$((y + $(gmem $p))); done
  fi
  [ $((used - y)) -lt 700 ]
}
for spec in $2; do
  IFS=: read ep fr tag <<< "$spec"
  O=$R/data/eval/ab/$tag/$LK; G=$R/data/eval/ab/$tag/gbuf_$LK; E=$R/data/raw/${tag}_$LK; mkdir -p $O $R/runs
  [ "$GBUF_ONLY" = 1 ] && E=$R/data/tmp/gb_${tag}_$LK
  if [ "$GEOM_ONLY" = 1 ]; then [ -f $E/GEOM_DONE ] && continue
  elif [ "$GBUF_ONLY" = 1 ]; then [ -f $G/DONE ] && continue
  else [ -f $O/DONE ] && continue; fi
  while pgrep -x blender >/dev/null || ! gpu_free; do sleep 20; done
  while [ "$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)" -lt 4 ]; do sleep 120; done
  rm -rf $E
  LOOKARG=""; [ "$LK" != native ] && LOOKARG="--look $LK"
  (cd ~/scene-project && ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/$ep \
     --pathtrace --frames $fr $LOOKARG --nts native --samples $SPP --nts-out $E) > $R/runs/abclip_${tag}_$LK.log 2>&1
  if [ "$GBUF_ONLY" = 1 ]; then
    ~/scene-project/py tools/exr_tools.py gbuf $E $G | tail -1
    g=$(ls $G/f_*.alb.jpg 2>/dev/null | wc -l)
    echo "$tag $LK $fr gbuf-only spp=$SPP gbuf=$g $(date +%H:%M)" | tee -a $R/data/eval/ab/log.txt
    [ "$g" -gt 0 ] && touch $G/DONE
    rm -rf $E
    continue
  fi
  post() {  # everything after the render is CPU work
  if [ "$GEOM_ONLY" != 1 ]; then
    GB="--gbuf $G"; [ "$NOGBUF" = 1 ] && GB=""
    ~/scene-project/py tools/exr_tools.py png $E $O --masks $R/data/eval/ab/$tag/masks$MASKS_SUFFIX $GB | tail -1
    cp $E/cuts.json $O/cuts.json 2>/dev/null
    [ -n "$(ls $G 2>/dev/null)" ] && touch $G/DONE
  fi
  ~/scene-project/py tools/exr_tools.py strip $E | tail -1
  rm -f $E/lowres_*.png
  n=$(ls $E/f_*.exr 2>/dev/null | wc -l); p=$(ls $O/f_*.png 2>/dev/null | wc -l)
  echo "$tag $LK $fr spp=$SPP exr=$n png=$p $(date +%H:%M)" | tee -a $R/data/eval/ab/log.txt
  [ "$GEOM_ONLY" = 1 ] && [ "$n" -gt 0 ] && touch $E/GEOM_DONE
  [ "$GEOM_ONLY" != 1 ] && [ "$p" -gt 0 ] && touch $O/DONE
  }
  # DEFER=1: convert in the background and start the next render at once, so the card is not left free between
  # clips (the co-tenant jobs would come back after 60 s and be stopped again by the next blender)
  if [ "$DEFER" = 1 ]; then post & else post; fi
done
wait
