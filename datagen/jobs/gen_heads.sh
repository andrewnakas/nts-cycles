#!/bin/bash
# 4050: gen_heads.sh <first> <last> [look=v9d] [pairs=8] [spp=96]: CG side of the generative realism model's data.
# Launch i renders <pairs> random consecutive frame pairs from one episode as full-resolution head boxes
# (render_lr mode "roi": about 2.5 s per frame, most of a launch is the scene build) and packs them to
# data/heads/cg/h<i>.npz (tools/pack_heads.py); the EXRs are deleted. Every third launch is the hero himself, the
# others a random MPFB person (NTS_PROCGEN=<i>: skin, age, gender, hair, outfit, face). Episodes rotate.
# Waits for a free GPU (no blender, under 700 MiB used: the card is shared with another GPU job) and 2 GB of disk.
A=$1; B=$2; LK=${3:-v9d}; K=${4:-8}; SPP=${5:-96}; R=~/nts-cycles; cd $R; mkdir -p data/heads/cg data/tmp runs
gpu_free() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 700 ]; }
EPS=($(cd ~/scene-project && for d in */; do [ -f $d/timeline.json ] && [ -f $d/voice.wav ] && echo ${d%/}; done))
for i in $(seq $A $B); do
  N=h$(printf %04d $i); O=data/heads/cg/$N.npz; [ -f data/heads/train_cg/$N.boxes.json ] && continue
  ep=${EPS[$(( i % ${#EPS[@]} ))]}
  FR=$(python3 - $HOME/scene-project/$ep/timeline.json $i $K <<'PY'
import json, random, sys
t = json.load(open(sys.argv[1])); rng = random.Random(int(sys.argv[2])); k = int(sys.argv[3])
shots = [(round(s["t0"] * t["fps"]) + 1, round(s["t1"] * t["fps"])) for l in t["lines"] for s in l.get("shots", [])]
shots = [(a, min(b, t["frames"])) for a, b in shots if min(b, t["frames"]) - a >= 12]
picks = set()
for _ in range(200):
    a, b = rng.choice(shots)
    f = rng.randint(a + 3, b - 4)
    if all(abs(f - p) > 2 for p in picks):
        picks.add(f)
    if len(picks) >= k:
        break
print(",".join(f"{f}-{f + 1}" for f in sorted(picks)))
PY
)
  [ -z "$FR" ] && { echo "h$i $ep: no frames"; continue; }
  while pgrep -x blender >/dev/null || ! gpu_free; do sleep 20; done
  while [ "$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)" -lt 2 ]; do sleep 120; done  # a launch needs ~0.3 GB while it runs
  T=$R/data/tmp/heads_$i; rm -rf $T; t0=$(date +%s)
  PG=""; [ $(( i % 3 )) -ne 0 ] && PG=$i
  (cd ~/scene-project && NTS_PROCGEN=$PG ~/blender-4.5/blender -b --gpu-backend vulkan -P cutscene/scene.py -- ~/scene-project/$ep \
     --pathtrace --frames $FR --look $LK --nts roi --samples $SPP --nts-out $T) > runs/heads_$i.log 2>&1
  ~/scene-project/py tools/pack_heads.py $T $O 2>&1 | tail -1
  # straight to training images (data/heads/train_cg, ~2 MB per launch); the 20-40 MB .npz is not kept: the disk is shared
  [ -f $O ] && ~/scene-project/py tools/heads_to_train.py data/heads/train_cg $O 2>&1 | tail -1
  ok=FAIL; [ -f data/heads/train_cg/$N.boxes.json ] && ok=ok
  echo "h$i $ep procgen=${PG:-hero} frames=$FR $(( $(date +%s) - t0 ))s $ok" | tee -a data/heads/cg/log.txt
  [ $ok = ok ] && rm -f runs/heads_$i.log $O
  rm -rf $T
done
