#!/bin/bash
# Mac, unattended: phase 1 pilot with the Wan 2.1 14B teacher on one Colab A100 session (the 1.3B ran on Kaggle).
# Uploads tools/teacher_wan.py + data/pilot/<clip glob>/in, runs it, pulls out/ to data/pilot/out_<tag>/out, stops the
# session, then tools/pilot_eval.sh <tag>. Log: runs/teacher_<tag>.log, last line TEACHER_DONE.
# Run: auto_teacher14.sh <tag> "<teacher args>" "<clip glob>"
S=nts-t14; R=~/Documents/nts-cycles; TAG=${1:-c14}; TA=${2:-}; CL=${3:-*}
C=(ssh -o "ProxyCommand=colab ssh --proxy-mode -s $S" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 root@colab-$S)
log() { echo "$(date +%H:%M) $*"; }
until colab sessions 2>&1 | grep -q "\[$S\]"; do
  colab new -s $S --gpu A100 --high-mem 2>&1 | grep -q READY && { log "session A100"; break; }
  log "no A100, retry in 30 min"; sleep 1800
done
cd $R && COPYFILE_DISABLE=1 tar cf - --no-xattrs tools/teacher_wan.py $(ls -d data/pilot/$CL/in) 2>/dev/null \
  | "${C[@]}" 'mkdir -p /content/t && cd /content/t && tar xf - && mv data/pilot clips && pip install -q -U diffusers transformers accelerate ftfy >/dev/null 2>&1; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /content | tail -1'
"${C[@]}" "cd /content/t && nohup python3 tools/teacher_wan.py --clips clips --out out --model Wan-AI/Wan2.1-T2V-14B-Diffusers $TA > teacher.log 2>&1 & sleep 5; echo started"
log "running"
while sleep 900; do
  "${C[@]}" 'ps aux | grep -q "[t]eacher_wan.py" && echo run' 2>/dev/null | grep -q run || break
  log "$("${C[@]}" 'grep -c frames /content/t/teacher.log' 2>/dev/null) clip-strengths done"
done
O=$R/data/pilot/out_$TAG; rm -rf $O && mkdir -p $O
"${C[@]}" 'cd /content/t && rm -f out/prompt_embeds.pt && tar cf - out teacher.log' | tar xf - -C $O
colab stop -s $S >/dev/null 2>&1; log "session stopped"
grep -E "frames|Error|error" $O/teacher.log | tail -20 | sed "s/^/$(date +%H:%M) /"
$R/tools/pilot_eval.sh $TAG 2>&1 | tail -40
log "TEACHER_DONE"
