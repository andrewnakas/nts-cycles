#!/bin/bash
# Mac: train two R2P variants side by side on one big Colab GPU (600-unit account, HOME=~/.colab-account-home).
#   tools/colab/r2p_two.sh <session> <gpu> "<train args>" [batch=4]
# Variants (env): A="MEGA:AIX" (default 8000:0), B="MEGA:AIX" (default 6000:2000). Each run has its own work dir
# /content/k{a,b} and its own HF cache (a shared cache let one run delete a file the other was reading).
# Warm start: runs/r2p_e2/out/last.pt. Data: data/heads/pub/data. Then run r2p_keepalive.sh <session> in the background:
# Colab reclaims a runtime whose kernel is idle (~1.5 h) even with background jobs running.
set -e
R=~/Documents/nts-cycles; S=$1; GPU=$2; TA=$3; BS=${4:-4}; cd $R
C() { HOME=$HOME/.colab-account-home colab "$@" 2>&1 | grep -v '^\[colab\]'; }
C sessions | grep -q "^\[$S\]" || C new -s $S --gpu $GPU
C sessions | grep "^\[$S\]" | grep -q "Hardware: $GPU" || { echo "no $GPU runtime"; exit 1; }
T=$(mktemp -d)
for v in a:${A:-8000:0} b:${B:-6000:2000}; do IFS=: read t m x <<<"$v"; mkdir -p $T/k$t
python3 - "$t" "$m" "$x" "$TA" "$BS" $T/k$t/job.py <<'PY'
import sys
t, m, x, ta, bs, out = sys.argv[1:]
s = open("tools/kaggle/run_r2p.py").read().replace("/kaggle/", f"/content/k{t}/")
s = s.replace('ARGS = "--hours 11"', f'ARGS = "{ta}"', 1)
s = s.replace("NOSANITY = 0 ", "NOSANITY = 1 ", 1).replace("ok = 2 if NOSANITY", f"ok = {bs} if NOSANITY", 1)
s = s.replace("MEGA = 0 ", f"MEGA = {m} ", 1).replace("AIX = 0 ", f"AIX = {x} ", 1)
open(out, "w").write(s)
PY
done
cp gen/r2p_train.py runs/r2p_e2/out/last.pt $T/
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -cf - -C $T . -C $R/data/heads pub/data | tools/colab/cssh.sh $S 'set -e
mkdir -p /content/stage /content/r2p && tar xf - -C /content/stage 2>/dev/null; mv /content/stage/pub /content/r2p/; cd /content
for t in a b; do mkdir -p k$t/input/r2p/ck k$t/working k$t/tmp; cp stage/k$t/job.py k$t/; cp stage/r2p_train.py k$t/input/r2p/
  cp stage/last.pt k$t/input/r2p/ck/; ln -sfn /content/r2p/pub/data k$t/input/r2p/data
  (cd k$t && HF_HOME=/content/hf$t setsid nohup python3 job.py > job.log 2>&1 < /dev/null &); done
sleep 3; ps aux | grep -c "[j]ob.py"'
rm -rf $T
