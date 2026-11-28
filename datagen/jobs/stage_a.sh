#!/bin/bash
# Stage A paired data (the realism-pass plan): random MPFB person (NTS_PROCGEN=seed) in a random episode frame pair.
#   input : production quality, --look ref5p2, 48 spp, native 1080x1920 + G-buffers
#   target: hero (render_hero.py) 1024 spp + strand-curve hair (strand_hair.py), same look, same geometry and camera
# Each render is packed to fp16 npz and its EXRs deleted. Waits for GPU gaps (the scene-project queue comes first).
# Stops when the 4050 has < 4 GB free. Run: stage_a.sh <first_seed> <count>
O=$HOME/nts-cycles/data/stage_a
mkdir -p $O
S0=${1:-100}; N=${2:-100}
cd ~/nts-cycles
for seed in $(seq $S0 $((S0 + N - 1))); do
  d=$O/s$seed
  [ -f $d/tgt.npz ] && [ -f $d/in.npz ] && continue
  free=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 4 ]; then echo "STOP disk ${free}G free at seed $seed $(date +%H:%M)" >> $O/log.txt; break; fi
  read ep f <<< "$(~/scene-project/py tools/stage_a_pick.py $seed)"
  mkdir -p $d
  for kind in in tgt; do
    [ -f $d/$kind.npz ] && continue
    while pgrep -x blender >/dev/null; do sleep 15; done
    if [ $kind = in ]; then envs="NTS_PROCGEN=$seed"; args="--nts native --samples 48"
    else envs="NTS_PROCGEN=$seed NTS_STRANDS=150000,1.5"; args="--nts hero --samples 1024"; fi
    t0=$(date +%s)
    (cd ~/scene-project && env $envs ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- \
       ~/scene-project/$ep --pathtrace --frames $f-$((f + 1)) --look ref5p2 $args --nts-out $d/$kind) > $d/$kind.log 2>&1
    pk=$([ $kind = tgt ] && echo --rgb-only)
    ~/scene-project/py tools/stage_a_pack.py $d/$kind $d/$kind.npz $pk >> $d/$kind.log 2>&1 \
      || { echo "FAIL s$seed $kind $ep $f" >> $O/log.txt; continue 2; }
    echo "s$seed $kind $ep $f $(( $(date +%s) - t0 ))s $(grep -a -o "procgen {'seed': [0-9]*, 'skin': '[a-z_]*'" $d/$kind.log | head -1 | cut -d"'" -f6)" >> $O/log.txt
  done
done
echo "STAGE_A_DONE $S0..$((S0 + N - 1)) $(date +%H:%M)" >> $O/log.txt
