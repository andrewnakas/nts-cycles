#!/bin/bash
# Training data v1 on any render box: walks datagen/jobs/train_v1_shots.txt, claims each shot atomically
# (mkdir in ~/nts-data/claims on the 1080 box, which the 4050 can reach), renders native GT + 2 low-res seeds, packs (guide.npy with seed 0), records it.
#   train_v1.sh 4050   packs into ~/nts-cycles/data/packed (offload.sh moves them to the 1080 box)
#   train_v1.sh 1080   packs into ~/nts-data/packed (the storage box); renders ONLY while no other process uses the
#                      GPU, and a watchdog kills our render and releases the shot if one appears
# Log lines: "CLAIM <shot> <box>", "pack: ...", "SHOT DONE <shot> <box> <min>m", "RELEASED <shot> <box> (gpu busy)"
BOX=$1
H1080=$NTS_HOST2
LIST=~/nts-cycles/datagen/jobs/train_v1_shots.txt
PY=~/scene-project/py
D=~/nts-cycles/data/raw
if [ "$BOX" = 1080 ]; then P=~/nts-data/packed; C="bash -c"; else P=~/nts-cycles/data/packed; C="ssh -n $H1080"; fi
mkdir -p $D $P
foreign_gpu() { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v blender | grep -q .; }
gate() {
  if [ "$BOX" = 1080 ]; then
    while foreign_gpu || pgrep -x blender >/dev/null; do sleep 60; done
  else  # the look jobs, the reference job (any non-Blender GPU process) and the guide benchmark go first
    while pgrep -x blender >/dev/null || foreign_gpu || pgrep -f "lg_batch2_queue[.]sh|lg_skin_basis[.]sh|guide_test[.]sh|train_v0[.]sh" >/dev/null; do sleep 60; done
    a=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)
    while [ "$a" -lt 9 ]; do sleep 120; a=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9); done
  fi
}
render() {  # out mode seed halton  -> returns 1 if the 1080's GPU got taken mid-render
  gate
  (cd ~/scene-project && exec ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/$EP \
     --pathtrace --frames $RANGE --nts $2 --nts-out $1 --nts-seed $3 --nts-halton $4 --samples 48) < /dev/null > $1.log 2>&1 &
  bp=$!
  while kill -0 $bp 2>/dev/null; do
    if [ "$BOX" = 1080 ] && foreign_gpu; then pkill -P $bp; kill $bp; pkill -x blender; wait $bp; return 1; fi
    sleep 20
  done
  wait $bp
}
while read EP RANGE NAME; do
  [ -z "$EP" ] && continue
  $C "mkdir ~/nts-data/claims/$NAME" 2>/dev/null || continue   # someone else has it (or it's done)
  echo "CLAIM $NAME $BOX $(date +%H:%M)"; t0=$(date +%s)
  ok=1
  render $D/${NAME}_native native 0 0 || ok=0
  for s in 0 1; do
    [ $ok = 1 ] || break
    render $D/${NAME}_lr$s perf $((s + 1)) $((RANDOM % 32)) || { ok=0; break; }
    g=""; [ $s = 0 ] && g="--guide"
    nice $PY ~/nts-cycles/nts/datasets/pack.py < /dev/null $D/${NAME}_lr$s $D/${NAME}_native $P/${NAME}_s$s $g
    rm -rf $D/${NAME}_lr$s
  done
  rm -rf $D/${NAME}_native $D/${NAME}_lr*
  if [ $ok = 1 ]; then
    echo "SHOT DONE $NAME $BOX $(( ($(date +%s) - t0) / 60 ))m"
  else
    rm -rf $P/${NAME}_s*; $C "rmdir ~/nts-data/claims/$NAME"; echo "RELEASED $NAME $BOX (gpu busy)"
  fi
done < $LIST
echo "TRAIN V1 $BOX LIST DONE"
