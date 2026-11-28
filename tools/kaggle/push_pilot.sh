#!/bin/bash
# Mac: NR-1 v2 pilots on Kaggle (one T4 each). Nothing here trains; only our own Cycles renders are uploaded
# (the reference frames never leave the eval folders).
#   push_pilot.sh data [tone=x5t]  4050 -> data/pilot2/d: evalf/eval_<clip>_<frame>.jpg (native look, q95, all frames
#                                  of the 4 old clips) + keys/key_<clip>_<frame>.png (24 toned keyframes from
#                                  data/v2/<clip>/<tone>) + keys/boxes.json -> private Kaggle dataset nts-pilot2.
#                                  This uploads to the user's Kaggle account: the user runs this step.
#   push_pilot.sh i1 ["extra r2p_train args"]   kernel nts-pilot-i1: e2 (kernel nts-r2p-e2) on evalf, ~1 h
#   push_pilot.sh p2 [minutes=80]               kernel nts-pilot-p2: FLUX.2 klein 4B edits of the keyframes
#   push_pilot.sh p2 [minutes=80] evalf         the same on the v9d-look frames already in dataset r2p-heads
#                                  (data/evalf), so no new upload is needed; head boxes come from the 4050's masks
#   push_pilot.sh get i1|p2        output -> data/pilot2/out_<mode>; i1 also -> 4050 data/eval/ab/<clip>/r2p_e2n
R=~/Documents/nts-cycles; H=$NTS_HOST; U=$KAGGLE_USER; DS=nts-pilot2; CMD=$1; cd $R
case "$CMD" in
data)
  TONE=${2:-x5t}; D=$R/data/pilot2/d; mkdir -p $D
  ssh $H "cd ~/nts-cycles && TONE=$TONE nice -n 15 ~/scene-project/py - && tar cf - -C data/v2/pilot2 evalf keys" <<'PY' | tar xf - -C $D
import glob, json, os, sys
import numpy as np
from PIL import Image
tone, out = os.environ["TONE"], "data/v2/pilot2"
os.makedirs(out + "/evalf", exist_ok=True)
os.makedirs(out + "/keys", exist_ok=True)
KEYS = {"at140": (145, 153, 170, 178, 187, 195), "pty332": (336, 344, 356, 364, 368, 376),
        "rs345": (350, 358, 375, 383, 392, 400), "zd400": (405, 413, 430, 438, 447, 455)}
n, boxes = 0, {}
for t, ks in KEYS.items():
    src = "data/eval/zd/gt" if t == "zd400" else "data/eval/ab/%s/native" % t
    for p in sorted(glob.glob(src + "/f_*.png")):
        f = int(os.path.basename(p)[2:6])
        Image.open(p).convert("RGB").save("%s/evalf/eval_%s_%d.jpg" % (out, t, f), quality=95, subsampling=0)
        n += 1
    for f in ks:
        im = Image.open("data/v2/%s/%s/f_%04d.png" % (t, tone, f)).convert("RGB")
        name = "key_%s_%d.png" % (t, f)
        im.save("%s/keys/%s" % (out, name))
        m = np.asarray(Image.open("data/eval/ab/%s/masks/m_%04d.png" % (t, f)).convert("RGB"))
        ys, xs = np.nonzero((m[..., 0] > 127) | (m[..., 2] > 127))  # face or hair
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        s = int(min(1.5 * max(xs.max() - xs.min(), ys.max() - ys.min()), im.width, im.height)) // 2 * 2
        x0 = int(min(max(cx - s / 2, 0), im.width - s))
        y0 = int(min(max(cy - s / 2, 0), im.height - s))
        boxes[name] = [x0, y0, x0 + s, y0 + s]
json.dump(boxes, open(out + "/keys/boxes.json", "w"), indent=1)
print(n, "frames,", len(boxes), "keyframes", file=sys.stderr)
PY
  echo "$(ls $D/evalf | wc -l | tr -d ' ') frames, $(ls $D/keys/*.png | wc -l | tr -d ' ') keyframes in $D ($(du -sh $D | cut -f1))"
  echo "{\"title\": \"$DS\", \"id\": \"$U/$DS\", \"licenses\": [{\"name\": \"CC-BY-4.0\"}]}" > $D/dataset-metadata.json
  if kaggle datasets status $U/$DS >/dev/null 2>&1; then kaggle datasets version -p $D --dir-mode zip -m "$TONE" -q 2>&1 | tail -1
  else kaggle datasets create -p $D --dir-mode zip -q 2>&1 | tail -1; fi
  ;;
i1|p2)
  TA=$2; K=/tmp/kgk_pilot_$CMD$SFX; mkdir -p $K; BX="{}"   # SFX=<suffix>: a second kernel of the same mode (nts-pilot-p2<suffix>)
  if [ "$3" = evalf ]; then DS=r2p-heads
    BX=$(ssh $H "cd ~/nts-cycles && nice -n 15 ~/scene-project/py -" <<'PY'
import json
import numpy as np
from PIL import Image
KEYS = {"at140": (145, 153, 170, 178, 187, 195), "pty332": (336, 344, 356, 364, 368, 376),
        "rs345": (350, 358, 375, 383, 392, 400), "zd400": (405, 413, 430, 438, 447, 455)}
boxes = {}
for t, ks in KEYS.items():
    for f in ks:
        m = np.asarray(Image.open("data/eval/ab/%s/masks/m_%04d.png" % (t, f)).convert("RGB"))
        h, w = m.shape[:2]
        ys, xs = np.nonzero((m[..., 0] > 127) | (m[..., 2] > 127))  # face or hair
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        s = int(min(1.5 * max(xs.max() - xs.min(), ys.max() - ys.min()), w, h)) // 2 * 2
        x0, y0 = int(min(max(cx - s / 2, 0), w - s)), int(min(max(cy - s / 2, 0), h - s))
        boxes["key_%s_%d.png" % (t, f)] = [x0, y0, x0 + s, y0 + s]
print(json.dumps(boxes))
PY
)
  fi
  python3 - "$CMD" "$TA" $K/run_pilot.py "$DS" "$BX" <<'PY'
import base64, json, sys
s = open("tools/kaggle/run_pilot.py").read()
s = s.replace('MODE = "i1"', f'MODE = "{sys.argv[1]}"', 1)
s = s.replace('ARGS = ""  ', f'ARGS = "{sys.argv[2]}"  ', 1)
s = s.replace('DS = "nts-pilot2"', f'DS = "{sys.argv[4]}"', 1)
s = s.replace('BOXES = {}  ', 'BOXES = ' + repr(json.loads(sys.argv[5])) + '  ', 1)
if sys.argv[1] == "i1":
    s = s.replace('CODE_B64 = ""', 'CODE_B64 = "' + base64.b64encode(open("gen/r2p_train.py", "rb").read()).decode() + '"', 1)
open(sys.argv[3], "w").write(s)
PY
  KS=""; [ $CMD = i1 ] && KS=", \"kernel_sources\": [\"$U/nts-r2p-e2\"]"
  cat > $K/kernel-metadata.json <<J
{"id": "$U/nts-pilot-$CMD$SFX", "title": "nts-pilot-$CMD$SFX", "code_file": "run_pilot.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$U/$DS"]$KS}
J
  kaggle kernels push -p $K --accelerator NvidiaTeslaT4 2>&1 | grep -v Warning | tail -1
  ;;
get)
  O=$R/data/pilot2/out_$2; mkdir -p $O
  kaggle kernels output $U/nts-pilot-$2 -p $O -q >/dev/null 2>&1
  find $O -name "*.jpg" -o -name "*.png" | wc -l | tr -d ' ' | sed 's/$/ images/'
  [ "$2" = i1 ] && bash tools/evalf_to_ab.sh $O/out/evalf e2n
  ;;
*) sed -n '2,11p' $0 ;;
esac
