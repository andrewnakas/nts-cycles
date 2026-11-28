#!/bin/bash
# Mac: push_r2p.sh <tag> "<train args, empty = sanity only>" [data_dir=data/heads/kg] [dataset=nts-r2p]
# Uploads <data_dir> (cg/ real/ eval/) + gen/r2p_train.py as the private Kaggle dataset <dataset> (new version if it
# exists) and starts kernel nts-r2p-<tag> on a T4. NODATA=1 skips the upload (dataset already current).
# MEGA=<n> has the kernel fetch n public-domain photos (megalith-small) as extra real tiles; only crisp ones are kept
# (SHARP=<x> sets the cut-off, default 7.5, 0 = keep all).
# AIX=<n> (needs MEGA) adds n Apache-2.0 generated photoreal images (ZImageTurboGen, max 3000) to that pool.
# KSRC=<kernel slug> attaches that kernel's output; run_r2p.py warm-starts from its last.pt.
# BATCH=<n> (with NOSANITY=1) sets the batch size, default 2.
# GRADE=1 also trains the shot-level grade head (gen/grade.py) first, ~10 min (NR-1, 2026-11-18).
R=~/Documents/nts-cycles; U=$KAGGLE_USER; TAG=$1; TA=$2; DD=${3:-data/heads/kg}; DS=${4:-nts-r2p}; cd $R
# a warm start resumes at the source checkpoint's step (e2: ~16k), so the default --steps 30000 would end a long run early
if [ -n "$KSRC" ] && [ -n "$TA" ] && ! echo "$TA" | grep -q -- "--steps"; then
  echo "push_r2p: KSRC warm start needs an explicit --steps (e.g. --steps 60000); not pushed"; exit 1; fi
if [ "$NODATA" != 1 ]; then
  T=/tmp/kg_r2p_$TAG; rm -rf $T && mkdir -p $T && cp gen/r2p_train.py $T/ && cp -R $DD $T/data
  echo "{\"title\": \"$DS\", \"id\": \"$U/$DS\", \"licenses\": [{\"name\": \"CC-BY-4.0\"}]}" > $T/dataset-metadata.json
  if kaggle datasets status $U/$DS >/dev/null 2>&1; then kaggle datasets version -p $T --dir-mode zip -m "$TAG" -q 2>&1 | tail -1
  else kaggle datasets create -p $T --dir-mode zip -q 2>&1 | tail -1; fi
  rm -rf $T
  for i in $(seq 1 40); do kaggle datasets status $U/$DS 2>&1 | grep -q ready && break; sleep 15; done
fi
K=/tmp/kgk_r2p_$TAG; rm -rf $K && mkdir $K
python3 - "$TA" $K/run_r2p.py <<'PY'
import base64, sys
s = open("tools/kaggle/run_r2p.py").read()
s = s.replace('ARGS = "--hours 11"', f'ARGS = "{sys.argv[1]}"', 1)
import os
s = s.replace("NOSANITY = 0 ", f"NOSANITY = {int(os.environ.get('NOSANITY', 0))} ", 1)
s = s.replace("SHARP = 7.5 ", f"SHARP = {float(os.environ.get('SHARP', 7.5))} ", 1)
s = s.replace("AIX = 0 ", f"AIX = {int(os.environ.get('AIX', 0))} ", 1)
s = s.replace("MEGA = 0 ", f"MEGA = {int(os.environ.get('MEGA', 0))} ", 1)
s = s.replace("GRADE = 0", f"GRADE = {int(os.environ.get('GRADE', 0))}", 1)
s = s.replace("BATCH = 2 ", f"BATCH = {int(os.environ.get('BATCH', 2))} ", 1)
s = s.replace('GRADE_B64 = ""', 'GRADE_B64 = "' + base64.b64encode(open("gen/grade.py", "rb").read()).decode() + '"', 1)
s = s.replace('CODE_B64 = ""', 'CODE_B64 = "' + base64.b64encode(open("gen/r2p_train.py", "rb").read()).decode() + '"', 1)
open(sys.argv[2], "w").write(s)
PY
cat > $K/kernel-metadata.json <<J
{"id": "$U/nts-r2p-$TAG", "title": "nts-r2p-$TAG", "code_file": "run_r2p.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$U/$DS"]$( [ -n "$KSRC" ] && echo ", \"kernel_sources\": [\"$U/$KSRC\"]" )}
J
kaggle kernels push -p $K --accelerator NvidiaTeslaT4 2>&1 | tail -1
