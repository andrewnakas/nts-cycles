#!/bin/bash
# Whole-episode run, part 3: the shots of an episode put back into one frame sequence and through the episode finish
# (scene-project cutscene/finish.py: subtitles, voice, music, end card) in a COPY of the episode folder, because
# finish.py clears <episode>/overlays. The user episode folder is only read.
#   v2_ep_finish.sh <episode> <prefix> <name>     name = a build under data/v2/<prefix>NN/<name>, or "render" for
#                                                 the frames the pass started from (data/eval/ab/<prefix>NN/$LK).
#                                                 A list "e3,e2" takes, shot by shot, the first of these builds
#                                                 that is complete (as many frames as the fullest of them); the
#                                                 output folder is named after the first.
# -> data/v2/ep/<episode>_<name>/{frames/f_####.jpg, final.mp4, final_phone.mp4}. Nothing is deleted outside that copy.
cd ~/nts-cycles; EP=$1; PRE=$2; NAME=$3; LK=${LK:-v11e}; SRC=$HOME/scene-project/$EP; E=$PWD/data/v2/ep/${EP}_${NAME%%,*}
[ -n "$(ls $E/frames 2>/dev/null | head -n 1)" ] && echo "$E/frames already has frames" && exit 1
mkdir -p $E/frames
for f in timeline.json episode.json voice.wav music.wav sfx.wav; do [ -f $SRC/$f ] && cp $SRC/$f $E/; done
# the end card of some episodes is the store mockup photo, which finish.py looks for beside the episode folder
# (episode.json "cta_mockup": "merch/mockups/..."): a copy beside the copy (token-bill, 2026-11-24)
M=$(sed -n 's/.*"cta_mockup": *"\([^"]*\)".*/\1/p' $E/episode.json 2>/dev/null | head -n 1)
[ -n "$M" ] && [ -f "$HOME/scene-project/$M" ] && mkdir -p "$(dirname "$(dirname $E)/$M")" && cp -n "$HOME/scene-project/$M" "$(dirname $E)/$M"
OMP_NUM_THREADS=3 nice -n 10 $HOME/scene-project/py - "$E" "$PRE" "$NAME" "$LK" <<"PY"
import glob, json, os, sys
from PIL import Image
E, pre, name, lk = sys.argv[1:5]
n = round(json.load(open(E + "/timeline.json"))["duration"] * 30)
got = {}
for d in sorted(glob.glob("data/v2/%s[0-9][0-9]" % pre)):
    t = os.path.basename(d)
    if name == "render":
        src = "data/eval/ab/%s/%s" % (t, lk)
    else:
        cnt = {k: len(glob.glob("%s/%s/f_*.png" % (d, k))) for k in name.split(",")}
        src = "%s/%s" % (d, next(k for k in name.split(",") if cnt[k] == max(cnt.values())))
        print(t, "<-", src, cnt)
    for p in glob.glob(src + "/f_*.png"):
        got[int(os.path.basename(p)[2:6])] = p
# a showcase shot that was not rendered (runs/v2_ep_render.sh leaves it out when the finish covers it with the product
# photo): its frames hold the last frame before it, so nothing new shows if the photo starts a frame late
tl, ej = json.load(open(E + "/timeline.json")), json.load(open(E + "/episode.json"))
cover = set()
if ej.get("cta_mockup"):
    for l in tl["lines"]:
        if l.get("showcase"):
            cover.update(range(round(l["start"] * 30) + 1, round(l["end"] * 30) + 1))
miss = [f for f in range(1, n + 1) if f not in got and f not in cover]
assert not miss, "frames missing: %s" % miss[:10]
held = 0
for f in range(1, n + 1):
    if f not in got:
        got[f] = got[f - 1]
        held += 1
    Image.open(got[f]).convert("RGB").save("%s/frames/f_%04d.jpg" % (E, f), quality=95, subsampling=0)
print("frames", n, "held (showcase under the product photo)", held)
PY
[ "$(ls $E/frames | wc -l)" -gt 0 ] || exit 1
# an episode that has not been through the queue yet has no music and no sound effects: made here, in the copy,
# by the same two steps the queue runs before the finish (queue.sh)
cd ~/scene-project
DUR=$(./py -c "import json,sys;print(json.load(open(sys.argv[1]))['duration'])" $E/timeline.json)
[ -f $E/music.wav ] || nice -n 10 ./py cutscene/music.py $E/music.wav $DUR 2>&1 | tail -n 1
[ -f $E/sfx.wav ] || nice -n 10 ./py cutscene/sfx.py $E 2>&1 | tail -n 1
nice -n 10 ./py cutscene/finish.py $E --music $E/music.wav 2>&1 | tail -n 2 \
  && nice -n 10 ffmpeg -nostdin -y -v error -i $E/final.mp4 -vf scale=720:-2 -c:v libx264 -crf 25 -preset slow -c:a aac -b:a 128k -movflags +faststart $E/final_phone.mp4
echo "EP_FINISH_DONE $EP ${NAME%%,*} $(ls -la $E/final.mp4 2>/dev/null | cut -d" " -f5)"
