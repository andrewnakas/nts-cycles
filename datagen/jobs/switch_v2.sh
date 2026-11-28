#!/bin/bash
# Switch a box from train_v1 to train_v2 at a shot boundary: wait for the next "SHOT DONE"/"RELEASED" in the v1 log,
# stop v1 (and any Blender it just started), release the claim it made for its next shot, then start v2 in screen.
#   switch_v2.sh 4050|1080
BOX=$1
if [ "$BOX" = 1080 ]; then LOG=~/nts-data/train_v1_1080.txt; C="bash -c"; NEWLOG=~/nts-data/train_v2_1080.txt
else LOG=~/nts-cycles/data/train_v1_4050.txt; C="ssh -n $NTS_HOST2"; NEWLOG=~/nts-cycles/data/train_v2_4050.txt; fi
n0=$(grep -cE "SHOT DONE|RELEASED" $LOG)
while [ "$(grep -cE "SHOT DONE|RELEASED" $LOG)" -le "$n0" ]; do sleep 5; done
for pid in $(ps -eo pid,args | awk '/[t]rain_v1[.]sh '$BOX'/ {print $1}'); do pkill -P $pid; kill $pid; done
sleep 2
pkill -f "nts-out $HOME/nts-cycles/data/raw/" 2>/dev/null   # a Blender v1 had just started for its next shot
last=$(tail -1 $LOG)
if [[ "$last" == CLAIM* ]]; then
  name=$(echo $last | awk '{print $2}')
  $C "rmdir ~/nts-data/claims/$name" && echo "released v1's next claim $name"
  rm -rf ~/nts-cycles/data/raw/${name}_*
fi
echo "SWITCHED $BOX to v2 $(date +%H:%M)" | tee -a $LOG
exec bash ~/nts-cycles/datagen/jobs/train_v2.sh $BOX >> $NEWLOG 2>&1   # run v2 in this screen session (a nested screen dies with it)
