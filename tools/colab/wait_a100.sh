#!/bin/bash
# Mac: retry for a Colab A100 every 10 min (up to <hours>, default 12), then start the R2P run on it.
#   tools/colab/wait_a100.sh <session> "<train args>" [hours]   env as for r2p_colab.sh
S=$1; TA=$2; END=$(( $(date +%s) + ${3:-12} * 3600 )); CHOME=${CHOME:-$HOME/.colab-account-home}
cd ~/Documents/nts-cycles
while [ $(date +%s) -lt $END ]; do
  HOME=$CHOME colab new -s $S --gpu A100 2>&1 | grep -E "Service Unavailable|Error" | tail -1 | cut -c1-120
  if HOME=$CHOME colab sessions 2>&1 | grep -q "^\[$S\].*A100"; then
    echo "$(date '+%F %T') got A100"; sleep 20
    for i in 1 2 3; do tools/colab/r2p_colab.sh $S A100 "$TA" 4 && break; sleep 60; done
    echo "WAIT_A100_DONE"; exit 0
  fi
  echo "$(date '+%F %T') no A100, retry in 10 min"; sleep 600
done
echo "WAIT_A100_GAVE_UP"
