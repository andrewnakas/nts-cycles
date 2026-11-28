#!/bin/bash
# Mac: assemble data/heads/kg/{cg,real,eval} for tools/kaggle/push_r2p.sh from what exists right now:
#   cg    the 4050's head-box renders (data/heads/cg/h*.npz -> tools/heads_to_train.py there, then copied) plus head
#         crops of production-look frames (data/heads/cg_prod, no motion)
#   real  Openverse head crops (data/heads/real)
#   eval  zero-day 400-459 head boxes (the clip of the A/B against the reference)
R=~/Documents/nts-cycles; H=$NTS_HOST; K=$R/data/heads/kg; cd $R
ssh $H 'cd ~/nts-cycles && ~/scene-project/py tools/heads_to_train.py data/heads/train_cg data/heads/cg/h*.npz | tail -1'
rm -rf $K; mkdir -p $K/cg $K/real $K/eval
rsync -a --exclude '*.reg.png' --exclude '*.boxes.json' $H:nts-cycles/data/heads/train_cg/ $K/cg/
cp data/heads/cg_prod/*.jpg $K/cg/ 2>/dev/null
cp data/heads/real/*.jpg $K/real/
rsync -a --include '*.jpg' --exclude '*' $H:nts-cycles/data/heads/train_eval/ $K/eval/
echo "kg: cg=$(ls $K/cg/*.jpg | wc -l) pairs=$(ls $K/cg/*.mv.npy 2>/dev/null | wc -l) real=$(ls $K/real | wc -l) eval=$(ls $K/eval | wc -l) size=$(du -sh $K | cut -f1)"
