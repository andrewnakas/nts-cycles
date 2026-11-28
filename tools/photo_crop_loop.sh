#!/bin/bash
# Mac: while the Openverse downloads run, turn photos into head crops (data/heads/real) and delete each processed
# original, so the nearly full disk only ever holds a few minutes of downloads. photos.jsonl (licence, creator,
# source page per photo) is kept. Stops when no fetch_openverse.py is running and nothing is left to crop.
cd ~/Documents/nts-cycles
while true; do
  for d in data/photos_ov data/photos_ov2 data/photos_ov3; do
    [ -d $d/img ] || continue
    ls $d/img/*.jpg >/dev/null 2>&1 || continue
    ls $d/img/*.jpg > /tmp/crop_batch_$$.txt
    .venv-photo/bin/python tools/photo_heads.py $d data/heads/real 2>&1 | tail -1
    python3 - $d /tmp/crop_batch_$$.txt <<'PY'
import json, os, sys
done = {json.loads(l)["photo"] for l in open("data/heads/real/heads.jsonl")}
for p in open(sys.argv[2]).read().split():
    if os.path.basename(p)[:-4] in done and os.path.exists(p):
        os.remove(p)
PY
  done
  echo "$(date +%H:%M) crops=$(ls data/heads/real/*.jpg | wc -l) free=$(df -h ~ | tail -1 | awk '{print $4}')"
  pgrep -f fetch_openverse.py >/dev/null || { ls data/photos_ov*/img/*.jpg >/dev/null 2>&1 || break; }
  sleep 240
done
echo CROP_LOOP_DONE
