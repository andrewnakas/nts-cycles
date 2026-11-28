#!/bin/bash
# Mac: render_variant.sh <look> <tag> "<env>": look + production strand-curve hair (NTS_STRANDS=150000,1.5; hero renders had it) on zero-day
# 420-431 (4050) -> tonemap +0.3 EV -> film grain 2 -> scoreboard vs B (v4hs+grain) and the reference. Log runs/combo_strands.log
R=~/Documents/nts-cycles; H=$NTS_HOST; LK=$1; TG=$2; EV="$3"
ssh $H "LK=$LK; TG=$TG; EV='$EV'; "'cd ~/nts-cycles && rm -rf /tmp/$TG && while pgrep -x blender >/dev/null; do sleep 30; done
  (cd ~/scene-project && env NTS_STRANDS=150000,1.5 $EV ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- \
     ~/scene-project/zero-day --pathtrace --frames 420-431 --look $LK --nts native --samples 48 --nts-out /tmp/$TG) > /tmp/$TG.log 2>&1
  grep -o "Time:[0-9:.]*" /tmp/$TG.log | tail -1
  ~/scene-project/py tools/stage_a_pack.py /tmp/$TG /tmp/$TG.npz --rgb-only | tail -1
  TG=$TG ~/scene-project/py - <<'PY'
import json, os, numpy as np, sys
from PIL import Image
sys.path.insert(0, "."); from nts.tonemap import Tonemapper
TG = os.environ["TG"]; z = np.load(f"/tmp/{TG}.npz"); v = json.load(open(f"/tmp/{TG}/meta.json"))["view"]; v[2] += 0.3; tm = Tonemapper(*v)
o = f"data/eval/zd/{TG}_in"; os.makedirs(o, exist_ok=True)
for i, f in enumerate(z["frames"]):
    Image.fromarray(tm(z["rgb"][i].astype(np.float32).transpose(1, 2, 0))).save(f"{o}/f_{f:04d}.png")
PY
  ~/scene-project/py tools/film_grain.py data/eval/zd/${TG}_in data/eval/zd/${TG}_in_g2 2 >/dev/null; rm -rf /tmp/$TG /tmp/$TG.npz'
ssh $H "cd ~/nts-cycles/data/eval/zd && tar cf - ${TG}_in ${TG}_in_g2" | tar xf - -C $R/data/eval/zd_sb
cd $R && uv run --no-project --python 3.11 --with torch==2.2.2 --with torchvision==0.17.2 --with 'numpy<2' --with facenet-pytorch --with open_clip_torch --with opencv-python-headless --with pillow python -m eval.scoreboard data/eval/zd_sb 420-431 runs/ntsr_a/photos_faces.npy "p3 input=ntsr5_in" "B v4hs+grain2=v4hs_in_g2" "E v4hs+strands=${TG}_in" "F v4hs+strands+grain2=${TG}_in_g2" "reference=ref5_pass2_clip" 2>&1 | grep -E "^\||Traceback"
python3 -c "
from PIL import Image, ImageDraw
cols=[('B look+grain','v4hs_in_g2'),('E look+strands','${TG}_in'),('F E+grain','${TG}_in_g2'),('reference','ref5_pass2_clip')]
ims=[]
for l,d in cols:
    im=Image.open(f'data/eval/zd_sb/{d}/f_0430.png').convert('RGB').crop((430,620,610,800)).resize((400,400),Image.LANCZOS); ImageDraw.Draw(im).text((5,5),l,fill=(255,255,0)); ims.append(im)
o=Image.new('RGB',(400*len(ims),400)); [o.paste(im,(400*i,0)) for i,im in enumerate(ims)]; o.save('runs/variant_${TG}_f0430.jpg',quality=93)"
echo VARIANT_DONE $TG
