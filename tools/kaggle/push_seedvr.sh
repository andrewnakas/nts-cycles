#!/bin/bash
# NR-1 pilot P1: SeedVR2-3B on head-crop clips of our toned renders (see run_seedvr.py for provenance).
#   push_seedvr.sh data [tone=x19t]          build the head crops per shot and add them to the private dataset nts-pilot2
#        (private quota is full, 2026-11-21: the new version = keys + crops + 4 evalf frames; the full evalf set
#        stays on the Mac in data/pilot2/d and must be put back before an i1 run that reads it)
#   push_seedvr.sh run "<min>|<shots>|<variants>|<dit>"   push the kernel nts-pilot-v1$SFX (T4 x2)
#        shots: comma list out of at140,pty332,rs345,zd400a,zd400b (empty = all); variants: a,b,c,m,n,l
#   push_seedvr.sh get [name=v1]             fetch the output to data/pilot3/out_<name>
# The account runs at most 2 batch GPU sessions and queued ones count. Never touch the user's own kernels.
R=~/Documents/nts-cycles; H=$NTS_HOST; U=$KAGGLE_USER; DS=nts-pilot2; CMD=$1; cd $R
case "$CMD" in
data)
  TONE=${2:-x19t}; D=$R/data/pilot3/d; mkdir -p $D
  ssh $H "cd ~/nts-cycles && TONE=$TONE nice -n 15 ~/scene-project/py - && tar cf - -C data/v2/pilot3 crops" <<'PY' | tar xf - -C $D
import glob, json, os, sys
import numpy as np
from PIL import Image
tone, out = os.environ["TONE"], "data/v2/pilot3"
CUTS = {"zd400": 439}   # a cut inside the clip: two shots, two crops
meta = {}
for t in ("at140", "pty332", "rs345", "zd400"):
    fs = sorted(int(os.path.basename(p)[2:6]) for p in glob.glob("data/v2/%s/%s/f_*.png" % (t, tone)))
    parts = [(t, fs)] if t not in CUTS else [(t + "a", [f for f in fs if f < CUTS[t]]), (t + "b", [f for f in fs if f >= CUTS[t]])]
    for shot, frames in parts:
        x0 = y0 = 10 ** 6
        x1 = y1 = 0
        for f in frames:
            m = np.asarray(Image.open("data/eval/ab/%s/masks/m_%04d.png" % (t, f)).convert("RGB"))
            ys, xs = np.nonzero((m[..., 0] > 127) | (m[..., 2] > 127))  # face or hair
            x0, x1, y0, y1 = min(x0, xs.min()), max(x1, xs.max()), min(y0, ys.min()), max(y1, ys.max())
        H, W = m.shape[:2]
        s = int(min(1.4 * max(x1 - x0, y1 - y0), W, H)) // 2 * 2
        bx = int(min(max((x0 + x1) / 2 - s / 2, 0), W - s))
        by = int(min(max((y0 + y1) / 2 - s / 2, 0), H - s))
        os.makedirs("%s/crops/%s" % (out, shot), exist_ok=True)
        for f in frames:
            im = Image.open("data/v2/%s/%s/f_%04d.png" % (t, tone, f)).convert("RGB")
            im.crop((bx, by, bx + s, by + s)).save("%s/crops/%s/f_%04d.png" % (out, shot, f))
        meta[shot] = dict(clip=t, box=[bx, by, bx + s, by + s], frames=frames, tone=tone)
        print(shot, len(frames), "frames, box", meta[shot]["box"], file=sys.stderr)
json.dump(meta, open(out + "/crops/shots.json", "w"), indent=1)
PY
  U3=$R/data/pilot2/d3; mkdir -p $U3/evalf
  rsync -a $R/data/pilot2/d/keys $U3/ && rsync -a $D/crops $U3/
  for t in at140 pty332 rs345 zd400; do cp -n $(ls $R/data/pilot2/d/evalf/eval_${t}_*.jpg | head -n 1) $U3/evalf/; done
  echo "$(find $U3/crops -name '*.png' | wc -l | tr -d ' ') crops, $(ls $U3/keys/*.png | wc -l | tr -d ' ') keyframes in $U3 ($(du -sh $U3 | cut -f1))"
  echo "{\"title\": \"$DS\", \"id\": \"$U/$DS\", \"licenses\": [{\"name\": \"CC-BY-4.0\"}]}" > $U3/dataset-metadata.json
  kaggle datasets version -p $U3 --dir-mode zip -m "keys + head crops ($TONE)" -q 2>&1 | tail -1
  ;;
run)
  K=/tmp/kgk_pilot_v1$SFX; mkdir -p $K
  python3 - "$2" $K/run_seedvr.py <<'PY'
import sys
s = open("tools/kaggle/run_seedvr.py").read()
assert s.count('ARGS = ""  ') == 1
open(sys.argv[2], "w").write(s.replace('ARGS = ""  ', f'ARGS = "{sys.argv[1]}"  ', 1))
PY
  cat > $K/kernel-metadata.json <<J
{"id": "$U/nts-pilot-v1$SFX", "title": "nts-pilot-v1$SFX", "code_file": "run_seedvr.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$U/$DS"]}
J
  kaggle kernels push -p $K --accelerator NvidiaTeslaT4 2>&1 | grep -v Warning | tail -1
  ;;
get)
  N=${2:-v1}; O=$R/data/pilot3/out_$N; mkdir -p $O
  kaggle kernels output $U/nts-pilot-$N -p $O -q >/dev/null 2>&1
  find $O -name "*.jpg" | wc -l | tr -d ' ' | sed 's/$/ images/'
  ;;
*) sed -n '2,8p' $0 ;;
esac
