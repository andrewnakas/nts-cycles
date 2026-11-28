#!/bin/bash
# Mac: one whole episode through the NR-1 v2 route (design log rows 112-115), stage by stage. The heavy work runs on
# the 4050 (ssh), the keyframe edit on Kaggle (one T4, about 6 min for 8 keys). First run by hand on agentic-time
# (2026-11-22): render 2 h, base + tone 22 s a frame per worker, carry 7 s a frame.
# State 2026-11-22: `status` is tested; the other stages are the commands of that hand run put in one place and
# have not been run end to end from here yet.
# 2026-11-23: one key holds over a 158-frame shot (design log row 119).
# 2026-11-24: folders per shot: tone data/v2/<tag>/x29t, build data/v2/<tag>/e8 (env TONE, NAME); the tone options
# are TONE_EXTRA (the standard of design log row 117 plus --hands-tight, row 121). The keys stage waits too long
# for one foreground call from a session: run its lines by hand, one at a time.
# 2026-11-24 (later): build e8 = the e1 recipe plus the border rules of design log row 122; they are the default of
# runs/v2_ep_carry.sh on the 4050 (its CARRY_EXTRA), not passed from here.
# 2026-11-26: gen.carry finds the keyframe positions in each shot's surf pass (render_lr.py mode "surf", at
# data/raw/<tag>_surf: tools/nr_clip.sh SURF), which runs/4050/v2_ep_render.sh does not render yet.
#   nr_episode.sh render <episode> <prefix>   every shot of ~/scene-project/<episode> rendered as its own clip
#                                             <prefix>01.. with the cloth look, then steadied base + tone (two
#                                             workers). The render takes the 4050's GPU: the co-tenant yields to it,
#                                             so start it only when that is fine.
#   nr_episode.sh status <episode> <prefix>   frames per shot: render / base / tone / carried
#   nr_episode.sh keys   <episode> <prefix>   one key per shot (near the middle, eyes open: runs/v2_ep_keys.py), toned
#                                             keyframes to the private Kaggle dataset, klein 4B edit, edits pasted
#                                             back (needs the tone step done for every shot)
#   nr_episode.sh carry  <episode> <prefix> [workers=3]   the keys carried over their shots -> data/v2/<prefix>NN/e8
#   nr_episode.sh finish <episode> <prefix>   the shots as one sequence through the episode's own finish, in a COPY
#                                             of the episode folder -> 4050 data/v2/ep/<episode>_e8/final.mp4
# Env: KIND (keys tag, default k<prefix>), TONE (x29), NAME (e8), TONE_EXTRA (default "--garment-curve 1.8 --fringe-skin --skin-rim --skin-close --shot-hold --hands-tight").
# Only our own renders are uploaded (private dataset); no reference frame is read anywhere. Nothing is deleted; every
# step refuses to write into a folder that already has frames. The user's episode folder is only read.
R=~/Documents/nts-cycles; H=$NTS_HOST; CMD=$1; EP=$2; PRE=$3; KIND=${KIND:-k$PRE}; TN=${TONE:-x29}; NM=${NAME:-e8}; cd $R
[ -z "$PRE" ] && sed -n '2,25p' $0 && exit 1
case "$CMD" in
render)
  ssh $H "cd ~/nts-cycles && [ ! -e runs/v2_ep_$PRE.out ] && (TONE=$TN TONE_EXTRA='${TONE_EXTRA:---garment-curve 1.8 --fringe-skin --skin-rim --skin-close --shot-hold --hands-tight}' nohup setsid bash runs/v2_ep_render.sh $EP $PRE > runs/v2_ep_$PRE.out 2>&1 < /dev/null &) && sleep 5 && head -n 1 runs/v2_ep_$PRE.out | cut -c1-200 || echo 'runs/v2_ep_$PRE.out exists: this prefix was started before'"
  ;;
status)
  ssh $H "cd ~/nts-cycles && for d in data/eval/ab/$PRE[0-9][0-9]; do t=\$(basename \$d); echo \"\$t render \$(ls \$d/v11e 2>/dev/null | grep -c '^f_') base \$(ls data/v2/\$t/ns6e 2>/dev/null | grep -c '^f_') tone \$(ls data/v2/\$t/${TN}t 2>/dev/null | grep -c '^f_') carried \$(ls data/v2/\$t/$NM 2>/dev/null | grep -c '^f_')\"; done; tail -n 1 runs/v2_ep_$PRE.out | cut -c1-80"
  ;;
keys)
  KEYS=$(ssh $H "cd ~/nts-cycles && OMP_NUM_THREADS=2 nice -n 10 ~/scene-project/py runs/v2_ep_keys.py $PRE | tee runs/ep_keys_$PRE.txt | grep '^KEYS ' | cut -c6-")
  [ -z "$KEYS" ] && echo "no keys picked" && exit 1
  touch data/pilot2/.keys_start_$PRE
  echo "keys: $KEYS"
  ssh $H "cd ~/nts-cycles && for s in $KEYS; do TONE=$TN bash tools/nr_clip.sh keys \${s%%:*} \${s##*:} 2>/dev/null | tail -n 1; done" || exit 1
  LIST=$(echo $KEYS | tr ' ' '\n' | sed 's/:/_/' | paste -sd, -)
  for s in $KEYS; do rsync -a $H:nts-cycles/data/v2/pilot2/keys/key_${s%%:*}_${s##*:}.png data/pilot2/d3/keys/ || exit 1; done
  rsync -a $H:nts-cycles/data/v2/pilot2/keys/boxes.json data/pilot2/d3/keys/ && cp data/pilot2/d3/keys/key_$PRE*.png data/pilot2/d3/keys/boxes.json data/pilot2/d/keys/
  kaggle datasets version -p data/pilot2/d3 --dir-mode zip -m "keys: $PRE" -q 2>&1 | grep -v Warning | tail -n 1
  n=0; until [ "$(kaggle datasets status $KAGGLE_USER/nts-pilot2 2>/dev/null | grep -v Warning | tail -n 1)" = ready ] || [ $n -ge 30 ]; do n=$((n+1)); sleep 20; done
  SFX=k bash tools/kaggle/push_pilot.sh p2 "40|n|$LIST|sd:t85x3@n+sdonly+nofull|" 2>&1 | tail -n 1
  sleep 90; n=0; until kaggle kernels status $KAGGLE_USER/nts-pilot-p2k 2>/dev/null | grep -q -i -E "complete|error|cancel" || [ $n -ge 45 ]; do n=$((n+1)); sleep 60; done
  kaggle kernels status $KAGGLE_USER/nts-pilot-p2k 2>&1 | grep -v Warning | tail -n 1
  bash tools/kaggle/push_pilot.sh get p2k | tail -n 1      # the fetch takes the kernel's full suffix (p2k, not p2)
  ssh $H "mkdir -p ~/nts-cycles/data/v2/p7/$KIND"
  for s in $KEYS; do E=data/pilot2/out_p2k/out/head_n_t85x3pn/key_${s%%:*}_${s##*:}.png
    [ $E -nt data/pilot2/.keys_start_$PRE ] && rsync -a $E $H:nts-cycles/data/v2/p7/$KIND/ || { echo "no new edit for $s"; exit 1; }; done
  ssh $H "cd ~/nts-cycles && OMP_NUM_THREADS=2 nice -n 10 ~/scene-project/py runs/v2_p7keys.py $KIND $KIND ${TN}t | tail -n 1"
  echo "look at the edited keys before the carry: data/pilot2/out_p2k/out/head_n_t85x3pn/key_$PRE*.png"
  ;;
carry)
  W=${4:-3}
  ssh $H "cd ~/nts-cycles && K=\$(grep '^KEYS ' runs/ep_keys_$PRE.txt | cut -c6-) && [ -n \"\$K\" ] && for w in \$(seq 1 $W); do L=\$(echo \$K | tr ' ' '\n' | awk -v w=\$w -v n=$W 'NR % n == w % n' | tr '\n' ' '); [ -n \"\$L\" ] && (TONE=$TN nohup setsid bash runs/v2_ep_carry.sh $KIND $NM \"\$L\" > runs/v2_ep_carry_${PRE}_\$w.out 2>&1 < /dev/null &); echo \"worker \$w: \$L\"; done"
  ;;
finish)
  ssh $H "cd ~/nts-cycles && bash runs/v2_ep_finish.sh $EP $PRE $NM 2>&1 | tail -n 3"
  ;;
*) sed -n '2,25p' $0 ;;
esac
