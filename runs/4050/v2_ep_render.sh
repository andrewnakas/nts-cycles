#!/bin/bash
# Whole-episode run of the NR-1 v2 route (plan step D1), part 1: every shot of an episode rendered with the cloth
# look as its own clip <prefix>01.. (shots are independent: hard cuts), then steadied base + tone per shot.
#   v2_ep_render.sh <episode> <prefix> [look=v11e]
# Renders wait for GPU gaps (ab_clips.sh: a running blender, or a GPU held by anything but the yielding co-tenant).
# Env TONE_EXTRA: the tone step's extra options (default "--garment-curve 1.8 --fringe-skin --skin-rim --skin-close --shot-hold --hands-tight"). Outputs: data/eval/ab/<prefix>NN/<look>, masks, data/raw/<prefix>NN_<look>, data/v2/<prefix>NN/{ns6e,x28t,x28w} (env TONE, default x28).
# 2026-11-24: the showcase shot at the end is left out when the episode has a product photo for it (env SHOW=render
# renders it all the same); runs/v2_ep_finish.sh fills its frames.
# Nothing is deleted.
cd ~/nts-cycles; EP=$1; PRE=$2; LK=${3:-v11e}
SPECS=$(python3 - $EP $PRE <<"PY"
import json, os, sys
ep, pre = sys.argv[1:3]
d = os.path.expanduser("~/scene-project/%s/" % ep)
j = json.load(open(d + "timeline.json"))
e = json.load(open(d + "episode.json")) if os.path.exists(d + "episode.json") else {}
# the closing showcase shot is not rendered when the finish covers it with the store's product photo
# (episode.json "cta_mockup"; finish.py product_frame): 150 frames a episode saved. SHOW=render keeps it.
skip = bool(e.get("cta_mockup")) and os.environ.get("SHOW", "skip") == "skip"
sh = sorted((round(s["t0"] * 30) + 1, round(s["t1"] * 30)) for l in j["lines"] if not (skip and l.get("showcase"))
            for s in l.get("shots", []))
print(" ".join("%s:%d-%d:%s%02d" % (ep, a, b, pre, i + 1) for i, (a, b) in enumerate(sh)))
PY
)
echo "specs: $SPECS"
(DEFER=1 NOGBUF=1 bash datagen/jobs/ab_clips.sh $LK "$SPECS" > runs/ep_render_$PRE.out 2>&1; echo EP_RENDER_LOOP_DONE >> runs/ep_render_$PRE.out) &
# base + tone: two workers, odd and even shots (each waits for its shot's render)
TAGS=$(for s in $SPECS; do echo ${s##*:}; done)
bash runs/v2_ep_base2.sh "$(echo "$TAGS" | awk 'NR % 2 == 1' | tr '\n' ' ')" $LK > runs/v2_ep_base_${PRE}_a.out 2>&1 &
bash runs/v2_ep_base2.sh "$(echo "$TAGS" | awk 'NR % 2 == 0' | tr '\n' ' ')" $LK > runs/v2_ep_base_${PRE}_b.out 2>&1 &
wait
echo EP_BASE_DONE $EP $PRE
