#!/bin/bash
# Mac: train R2P on a Colab GPU with the same job as Kaggle (tools/kaggle/run_r2p.py, paths moved to /content/k).
#   tools/colab/r2p_colab.sh <session> <gpu: A100|L4|...> "<train args>" [batch=4] [resume=runs/r2p_e2/out/last.pt]
# env: MEGA, AIX, SHARP as for push_r2p.sh. CHOME = HOME holding the Colab account's CLI config (default: the account
# with the compute units). Data: data/heads/pub/data (the public r2p-heads set), streamed over the ssh bridge.
# Output on the VM: /content/k/working/out (fetch with r2p_colab_get.sh). Log: /content/k/job.log
set -e
R=~/Documents/nts-cycles; S=$1; GPU=$2; TA=$3; BS=${4:-4}; CK=${5:-runs/r2p_e2/out/last.pt}
CHOME=${CHOME:-$HOME/.colab-account-home}; cd $R
C() { HOME=$CHOME colab "$@" 2>&1 | grep -v '^\[colab\]'; }
B() { ssh -o ProxyCommand="env HOME=$CHOME colab ssh --proxy-mode --identity $HOME/.ssh/id_ed25519 -s $S" -o StrictHostKeyChecking=no -i $HOME/.ssh/id_ed25519 \
      -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 root@colab-$S "$@"; }
if ! C sessions | grep -q "^\[$S\]"; then C new -s $S --gpu $GPU; fi
C sessions | grep "^\[$S\]"
T=$(mktemp -d); mkdir -p $T/k/input/r2p/ck
python3 - "$TA" "$BS" $T/k/job.py <<'PY'
import os, sys
s = open("tools/kaggle/run_r2p.py").read().replace("/kaggle/", "/content/k/")
s = s.replace('ARGS = "--hours 11"', f'ARGS = "{sys.argv[1]}"', 1)
s = s.replace("NOSANITY = 0 ", "NOSANITY = 1 ", 1).replace("ok = 2 if NOSANITY", f"ok = {int(sys.argv[2])} if NOSANITY", 1)
for k, d in (("MEGA", "0"), ("AIX", "0"), ("SHARP", "7.5")):
    v = os.environ.get(k, d)
    s = s.replace(f"{k} = {d} ", f"{k} = {v} ", 1)
open(sys.argv[3], "w").write(s)
PY
cp gen/r2p_train.py $T/k/input/r2p/; [ -f "$CK" ] && cp "$CK" $T/k/input/r2p/ck/last.pt
grep -E "^(ARGS|MEGA|AIX|SHARP|NOSANITY|ok) =" $T/k/job.py
COPYFILE_DISABLE=1 tar czf - -C $T k -C $R/data/heads/pub/data/.. data | B 'mkdir -p /content && cd /content && tar xzf - && mv data k/input/r2p/ && du -sh k/input && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
B 'mkdir -p /content/k/working /content/k/tmp && cd /content/k && nohup python3 job.py > job.log 2>&1 < /dev/null & sleep 5; echo started; head -c 600 /content/k/job.log'
rm -rf $T
