#!/bin/bash
# batch 2 starts after batch 1 (lg_all.sh) so the two never share the GPU
while ps -eo args | grep -q "[l]g_all.sh"; do sleep 20; done
~/nts-cycles/datagen/jobs/lg_batch.sh BATCH2 "center-a-div:300,500" "honeypot:60,190" "pair-programming:50" "scam-altman:160" "please-thank-you:70,355,475" "ram-shortage:120,375" "last-programmer:300"
