#!/bin/bash
# Mac, unattended: waits for the 4050 pilot clips (datagen/jobs/pilot_clips.sh) -> data/pilot -> private Kaggle dataset
# nts-pilot (JPG q97, ~30 MB) -> tools/kaggle/run_teacher.py on 2x T4 -> data/pilot/out_<tag> -> tools/teacher_lock.py
# and eval/scoreboard.py. Log: runs/teacher_<tag>.log, last line TEACHER_DONE.
# Run: auto_teacher.sh <tag> "<teacher args>" [dataset] [clip glob]   e.g. auto_teacher.sh k13 "--strengths 0.3,0.5"
# A dataset other than nts-pilot skips the 4050 wait/pull (clips already in data/pilot), e.g. nts-pilot-z "*z".
R=~/Documents/nts-cycles; H=$NTS_HOST; U=$KAGGLE_USER; TAG=${1:-k13}; TA=${2:-}; DS=${3:-nts-pilot}; CL=${4:-*}
log() { echo "$(date +%H:%M) $*"; }
[ $DS = nts-pilot ] && until ssh $H "grep -q PILOT_CLIPS_DONE ~/nts-cycles/data/pilot/log.txt 2>/dev/null"; do sleep 900; done
[ $DS = nts-pilot ] && log "clips: $(ssh $H 'grep -c ok ~/nts-cycles/data/pilot/log.txt') ok"
[ $DS = nts-pilot ] && mkdir -p $R/data/pilot && ssh $H 'cd ~/nts-cycles/data/pilot && tar cf - */in */masks */crop.json zd/ref5 clips.txt log.txt' | tar xf - -C $R/data/pilot
if ! kaggle datasets status $U/$DS >/dev/null 2>&1; then
  T=/tmp/kgpilot; rm -rf $T && mkdir -p $T/clips && cp $R/tools/teacher_wan.py $T/
  for d in $R/data/pilot/$CL/in; do n=$(basename $(dirname $d)); mkdir -p $T/clips/$n/in
    for f in $d/f_*.png; do sips -s format jpeg -s formatOptions 97 $f --out $T/clips/$n/in/$(basename ${f%.png}).jpg >/dev/null; done; done
  echo "{\"title\": \"$DS\", \"id\": \"$U/$DS\", \"licenses\": [{\"name\": \"CC-BY-4.0\"}]}" > $T/dataset-metadata.json
  kaggle datasets create -p $T --dir-mode zip -q 2>&1 | grep -v Warning | tail -1 | sed "s/^/$(date +%H:%M) /"
  rm -rf $T; sleep 600
fi
K=/tmp/kgteach_$TAG; rm -rf $K && mkdir $K && sed "s|^ARGS = .*|ARGS = \"$TA\"|" $R/tools/kaggle/run_teacher.py > $K/run_teacher.py
cat > $K/kernel-metadata.json <<J
{"id": "$U/nts-teacher-$TAG", "title": "nts-teacher-$TAG", "code_file": "run_teacher.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$U/$DS"]}
J
kaggle kernels push -p $K --accelerator NvidiaTeslaT4 2>&1 | grep -v Warning | tail -1 | sed "s/^/$(date +%H:%M) /"
while true; do
  st=$(kaggle kernels status $U/nts-teacher-$TAG 2>&1 | grep -v Warning | tail -1); log "$st"
  echo "$st" | grep -qiE "complete|error|cancel" && break; sleep 1200
done
O=$R/data/pilot/out_$TAG; rm -rf $O && mkdir -p $O && kaggle kernels output $U/nts-teacher-$TAG -p $O -q >/dev/null 2>&1
grep -E "frames|Error|error" $O/teacher.log | tail -20 | sed "s/^/$(date +%H:%M) /"
[ -x $R/tools/pilot_eval.sh ] && $R/tools/pilot_eval.sh $TAG 2>&1 | tail -30
log "TEACHER_DONE"
