#!/bin/bash
# Mac: tools/colab/r2p_colab_get.sh <session> [dest=runs/r2p_<session>] [log|fetch] : tail the job log, or copy its output.
S=$1; D=${2:-runs/r2p_$1}; CHOME=${CHOME:-$HOME/.colab-account-home}; cd ~/Documents/nts-cycles
B() { ssh -o ProxyCommand="env HOME=$CHOME colab ssh --proxy-mode --identity $HOME/.ssh/id_ed25519 -s $S" -o StrictHostKeyChecking=no -i $HOME/.ssh/id_ed25519 \
      -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@colab-$S "$@"; }
if [ "${3:-log}" = log ]; then B 'grep -v -E "it/s|Fetching|Loading|warn" /content/k/job.log | tail -15; tail -3 /content/k/working/out/train.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
else mkdir -p $D && B 'cd /content/k/working && tar cf - out' | tar xf - -C $D && cp /dev/null $D/.fetched && ls $D/out; fi
