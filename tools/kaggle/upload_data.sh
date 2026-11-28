#!/bin/bash
# Mac: upload the 4050's stage A pairs as 8 private Kaggle datasets (nts-stage-a-1..8, ~650 MB each, staged one at a
# time: the Mac has ~1.4 GB free) plus nts-code. Re-run to publish new versions (e.g. after a retarget).
H=$NTS_HOST; U=$KAGGLE_USER; T=/tmp/kgup
pairs=$(ssh $H 'cd ~/nts-cycles/data/stage_a && for s in s*; do [ -f $s/tgt.npz ] && [ -f $s/in.npz ] && echo $s; done')
n=$(echo "$pairs" | wc -l); q=$(( (n + 7) / 8 ))
up() {  # dir slug
  echo "{\"title\": \"$2\", \"id\": \"$U/$2\", \"licenses\": [{\"name\": \"CC-BY-4.0\"}]}" > $1/dataset-metadata.json
  if kaggle datasets status $U/$2 >/dev/null 2>&1; then kaggle datasets version -p $1 --dir-mode tar -m update -q 2>&1 | tail -1
  else kaggle datasets create -p $1 --dir-mode tar -q 2>&1 | tail -1; fi
}
for part in 1 2 3 4 5 6 7 8; do
  sel=$(echo "$pairs" | sed -n "$(( (part - 1) * q + 1 )),$(( part * q ))p"); [ -n "$sel" ] || continue
  rm -rf $T && mkdir -p $T
  ssh $H "cd ~/nts-cycles/data/stage_a && tar cf - $(echo $sel | sed 's/\(s[0-9]*\)/\1\/in.npz \1\/tgt.npz/g')" | tar xf - -C $T
  up $T nts-stage-a-$part; echo "part $part: $(echo $sel | wc -w) pairs"
done
rm -rf $T && mkdir -p $T && (cd ~/Documents/nts-cycles && COPYFILE_DISABLE=1 tar cf $T/nts.tar --no-xattrs nts 2>/dev/null)
up $T nts-code; rm -rf $T; echo UPLOAD_DONE
