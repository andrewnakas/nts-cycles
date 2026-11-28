#!/bin/bash
# Mac: render_clip.sh <look> [frames=420-431] [ep=zero-day]: render a clip with a look on the 4050 (48 spp native,
# view +0.3 EV, no grain) -> data/eval/zd_sb/<look>_clip/f_####.png on the Mac. Scoring is separate (eval.scoreboard).
R=~/Documents/nts-cycles; H=$NTS_HOST; LK=$1; FR=${2:-420-431}; EP=${3:-zero-day}; SPP=${SPP:-48}; TG=clip_${LK}${TAG}
ssh $H "LK=$LK; TG=$TG; FR=$FR; EP=$EP; SPP=$SPP; "'cd ~/nts-cycles && rm -rf /tmp/$TG && while pgrep -x blender >/dev/null; do sleep 30; done
  (cd ~/scene-project && ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- \
     ~/scene-project/$EP --pathtrace --frames $FR --look $LK --nts native --samples $SPP --nts-out /tmp/$TG) > /tmp/$TG.log 2>&1
  grep -o "Time:[0-9:.]*" /tmp/$TG.log | tail -1
  grep -o "Time: *[0-9:.]*" /tmp/$TG.log | awk -F"[: ]+" -v t="$TG" -v s="$SPP" -v e="$EP" "{x=\$(NF-1)*60+\$NF; n++; tot+=x} END{if(n) printf \"| %s | %s | native | %s | %d | %.2f |\n\", t, e, s, n, tot/n}" >> runs/timing.md
  ~/scene-project/py tools/stage_a_pack.py /tmp/$TG /tmp/$TG.npz --rgb-only | tail -1
  TG=$TG ~/scene-project/py - <<'"'PY'"'
import json, os, numpy as np, sys
from PIL import Image
sys.path.insert(0, "."); from nts.tonemap import Tonemapper
TG = os.environ["TG"]; z = np.load(f"/tmp/{TG}.npz"); v = json.load(open(f"/tmp/{TG}/meta.json"))["view"]; v[2] += 0.3; tm = Tonemapper(*v)
o = f"data/eval/zd/{TG}"; os.makedirs(o, exist_ok=True)
for i, f in enumerate(z["frames"]):
    Image.fromarray(tm(z["rgb"][i].astype(np.float32).transpose(1, 2, 0))).save(f"{o}/f_{f:04d}.png")
PY
  rm -rf /tmp/$TG /tmp/$TG.npz'
ssh $H "cd ~/nts-cycles/data/eval/zd && tar cf - $TG" | tar xf - -C $R/data/eval/zd_sb
echo CLIP_DONE $TG $(ls $R/data/eval/zd_sb/$TG | wc -l)
