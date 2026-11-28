#!/bin/bash
# Move finished packed shots off the 4050 so rendering can run all night: rsync to the 1080 box, verify by checksum,
# then delete locally. A shot counts as finished when a job log (data/*.txt) has its "pack: N frames -> <dir>" line.
# test_zd_s0 is the eval shot: copied but kept locally. Log: data/offload.txt ("OFFLOADED <shot> <size>").
DST=$NTS_HOST2:nts-data/packed
KEEP="test_zd_s0"
ssh $NTS_HOST2 "mkdir -p ~/nts-data/packed"
while true; do
  for dir in $(grep -ho "pack: [0-9]* frames -> [^ ]*" ~/nts-cycles/data/*.txt 2>/dev/null | awk '{print $5}' | sort -u); do
    [ -d "$dir" ] || continue
    shot=$(basename $dir)
    grep -q "COPIED $shot\b" ~/nts-cycles/data/offload.txt 2>/dev/null && [[ " $KEEP " == *" $shot "* ]] && continue
    size=$(du -sh $dir | cut -f1)
    free=$(ssh -n $NTS_HOST2 "df --output=avail -BG ~ | tail -1" | tr -dc 0-9)
    if [ "${free:-0}" -lt 25 ]; then echo "PAUSED 1080 disk ${free}G free $(date +%H:%M)"; sleep 600; continue; fi
    rsync -a $dir $DST/ || { echo "FAIL rsync $shot"; continue; }
    if [ -z "$(rsync -a --checksum --dry-run --itemize-changes $dir $DST/ | grep -v '^\.d')" ]; then
      if [[ " $KEEP " == *" $shot "* ]]; then echo "COPIED $shot $size"; else rm -rf $dir && echo "OFFLOADED $shot $size $(date +%H:%M)"; fi
    else
      echo "FAIL verify $shot"
    fi
  done
  sleep 120
done
