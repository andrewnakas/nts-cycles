#!/bin/bash
# Mac: score a teacher run (data/pilot/out_<tag>) per clip: raw teacher and identity-locked (sigma 2, 4) against the
# input and, on zd, the reference. Writes data/pilot/<clip>/scoreboard.json, runs/pilot/<tag>.md and a contact sheet.
R=~/Documents/nts-cycles; TAG=$1; P=$R/data/pilot
UV="uv run --no-project --python 3.11 --with torch==2.2.2 --with torchvision==0.17.2 --with numpy<2 --with facenet-pytorch --with open_clip_torch --with opencv-python-headless --with pillow"
cd $R && mkdir -p runs/pilot && ln -sfn ../eval/cg_faces $P/cg_faces
echo "# Teacher pilot $TAG ($(date +%F))" > runs/pilot/$TAG.md
for c in $P/*/crop.json; do
  d=$(dirname $c); n=$(basename $d); ls -d $P/out_$TAG/out/${n}_s* >/dev/null 2>&1 || continue; span=$(python3 -c "import json;f=json.load(open('$c'))['frames'];print(f'{f[0]}-{f[1]}')")
  meth=("in=in")
  for t in $P/out_$TAG/out/${n}_s*; do s=${t##*_s}
    ln -sfn $t $d/t_${TAG}_s$s; meth+=("raw s$s=t_${TAG}_s$s")
    for sg in 2 4; do $UV python tools/teacher_lock.py $d/in $t $d/masks $d/l_${TAG}_s${s}_g$sg $sg >/dev/null 2>&1
      meth+=("lock s$s g$sg=l_${TAG}_s${s}_g$sg"); done
  done
  [ -d $d/ref5 ] && meth+=("reference=ref5")
  echo -e "\n## $n ($span)" >> runs/pilot/$TAG.md
  $UV python -m eval.scoreboard $d $span runs/ntsr_a/photos_faces.npy "${meth[@]}" 2>&1 | grep -E "^\||Traceback|Error" >> runs/pilot/$TAG.md
done
$UV python tools/pilot_sheet.py $P $TAG runs/pilot/${TAG}_sheet.jpg >/dev/null 2>&1
cat runs/pilot/$TAG.md
