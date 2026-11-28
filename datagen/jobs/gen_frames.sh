#!/bin/bash
# 4050: [CLIP=2] [OUT=data/frames/train_cg] [PFX=w] [GBUF=0] gen_frames.sh <first> <last> [look=v9d] [runs=3] [spp=48]
# Whole-frame CG side of the generative realism model. Launch i renders <runs> random runs of CLIP consecutive frames
# (CLIP=2: the original consecutive pairs) from one episode at native 1080x1920 (production settings) and writes them as
# training images to OUT (tools/frames_to_train.py, with G-buffer images when GBUF=1); EXRs are deleted.
# NR-1 data (2026-11-18): CLIP=12 OUT=data/frames/train_clip PFX=c GBUF=1 ... (runs of 12 for the temporal student).
# Shots listed in data/eval/holdout.json (held-out clips: whole shot; eval clips: their frame window) are never picked.
# Every third launch is the hero, the others a random MPFB person (NTS_PROCGEN=<i>). Episodes rotate.
# Waits for a free GPU (no blender, under 700 MiB used: the card is shared) and 3 GB of disk.
A=$1; B=$2; LK=${3:-v9d}; K=${4:-3}; SPP=${5:-48}; R=~/nts-cycles; cd $R
CLIP=${CLIP:-2}; OUT=${OUT:-data/frames/train_cg}; PFX=${PFX:-w}; GB=""; [ "${GBUF:-0}" = 1 ] && GB=--gbuf
mkdir -p $OUT data/tmp runs
gpu_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 700 ]; }
EPS=($(cd ~/scene-project && for d in */; do [ -f $d/timeline.json ] && [ -f $d/voice.wav ] && echo ${d%/}; done))
for i in $(seq $A $B); do
  N=$PFX$(printf %04d $i); [ -f $OUT/$N.done ] && continue
  ep=${EPS[$(( (i * 7 + 3) % ${#EPS[@]} ))]}
  FR=$(python3 - $HOME/scene-project/$ep/timeline.json $i $K $CLIP $ep $R/data/eval/holdout.json <<'PY'
import json, os, random, sys
t = json.load(open(sys.argv[1])); rng = random.Random(1000 + int(sys.argv[2])); k = int(sys.argv[3]); L = int(sys.argv[4])
ep = sys.argv[5]
bad = []  # frame ranges of this episode that must never be rendered for training
if os.path.exists(sys.argv[6]):
    h = json.load(open(sys.argv[6]))
    bad += [tuple(c["shot"]) for c in h.get("heldout", []) if c["episode"] == ep]
    bad += [tuple(c["frames"]) for c in h.get("eval", []) if c["episode"] == ep]
shots = [(round(s["t0"] * t["fps"]) + 1, round(s["t1"] * t["fps"])) for l in t["lines"] for s in l.get("shots", [])]
shots = [(a, min(b, t["frames"])) for a, b in shots if min(b, t["frames"]) - a >= L + 10]
picks = []
for _ in range(400):
    a, b = rng.choice(shots)
    f = rng.randint(a + 3, b - L - 2)
    if any(f <= hb and f + L - 1 >= ha for ha, hb in bad):
        continue
    if all(abs(f - p) > L + 1 for p in picks):
        picks.append(f)
    if len(picks) >= k:
        break
print(",".join(f"{f}-{f + L - 1}" for f in sorted(picks)))
PY
)
  [ -z "$FR" ] && { echo "$N $ep: no frames"; continue; }
  while pgrep -x blender >/dev/null || ! gpu_free; do sleep 30; done
  while [ "$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)" -lt 3 ]; do sleep 120; done
  T=$R/data/tmp/frames_$N; rm -rf $T; t0=$(date +%s)
  PG=""; [ $(( i % 3 )) -ne 0 ] && PG=$i
  (cd ~/scene-project && NTS_PROCGEN=$PG ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/$ep \
     --pathtrace --frames $FR --look $LK --nts native --samples $SPP --nts-out $T) > runs/frames_$N.log 2>&1
  ~/scene-project/py tools/frames_to_train.py $T $OUT $N $GB 2>&1 | tail -1
  ok=FAIL; [ -f $OUT/$N.done ] && ok=ok
  echo "$N $ep procgen=${PG:-hero} frames=$FR $(( $(date +%s) - t0 ))s $ok" | tee -a $(dirname $OUT)/log.txt
  [ $ok = ok ] && rm -f runs/frames_$N.log
  rm -rf $T
done
echo GEN_FRAMES_DONE
