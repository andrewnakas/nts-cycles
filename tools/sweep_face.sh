#!/bin/bash
# sweep_face.sh "tag:args" ... : guided.py on FR (default 430-459), scored by eval/run_eval.score_shot (mask face metrics)
cd ~/nts-cycles
FR=${FR:-430-459}
for spec in "$@"; do
  tag=${spec%%:*}; args=${spec#*:}
  nice ~/scene-project/py baselines/guided.py data/raw/zd_perf_s0 data/raw/zd_native runs/guided/sf_$tag --frames $FR $args > /dev/null 2>&1
  nice ~/scene-project/py - "$tag" "$FR" <<'PY'
import json, sys
sys.path.insert(0, ".")
from eval.run_eval import score_shot
tag, fr = sys.argv[1], sys.argv[2]
sh = json.load(open("eval/testset.json"))["shots"][0]
sh["frames"] = [int(x) for x in fr.split("-")]
sh["static"] = sh["frames"]
sh["variants"] = {tag: f"runs/guided/sf_{tag}"}
r = score_shot(sh)[tag]
print(tag, " ".join(f"{m} {r.get(m, 0):.3f}" for m in ("psnr", "terr", "face_psnr", "face_hp_psnr", "face_terr", "flicker")))
PY
done
