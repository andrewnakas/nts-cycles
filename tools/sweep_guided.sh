#!/bin/bash
# sweep_guided.sh <tag> "<guided.py args>" ... : run guided.py on 400-411 per arg set and score whole frame + face box
cd ~/nts-cycles
for spec in "$@"; do
  tag=${spec%%:*}; args=${spec#*:}
  nice ~/scene-project/py baselines/guided.py data/raw/zd_perf_s0 data/raw/zd_native runs/guided/sw_$tag --frames ${FR:-400-411} $args > /dev/null 2>&1
  w=$(nice ~/scene-project/py eval/metrics.py data/eval/zd/gt runs/guided/sw_$tag --frames ${FR:-400-411} --cuts data/raw/zd_perf_s0/cuts.json | sed 's/.*: //')
  fc=$(nice ~/scene-project/py eval/metrics.py data/eval/zd/gt runs/guided/sw_$tag --frames ${FR:-400-411} --cuts data/raw/zd_perf_s0/cuts.json --crop 470,650,580,780 | sed 's/.*: //')
  echo "$tag [$args] whole: $w | face: $fc"
done
