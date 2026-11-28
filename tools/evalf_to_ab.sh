#!/bin/bash
# Mac: tools/evalf_to_ab.sh <evalf_dir> <run>   R2P whole-frame eval output (eval_<tag>_<frame>.jpg) -> 4050
# ~/nts-cycles/data/eval/ab/<tag>/r2p_<run>/f_####.jpg, the layout gen.steady, eval.flicker_mv and eval.nr_board read.
SRC=$1; RUN=$2; T=$(mktemp -d)
for p in "$SRC"/eval_*.jpg; do
  b=$(basename "$p" .jpg); f=${b##*_}; t=${b#eval_}; t=${t%_*}
  mkdir -p "$T/$t/r2p_$RUN"; cp "$p" "$T/$t/r2p_$RUN/$(printf 'f_%04d.jpg' $((10#$f)))"
done
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -cf - -C "$T" . | ssh $NTS_HOST "tar xf - -C ~/nts-cycles/data/eval/ab/"
for d in "$T"/*/; do t=$(basename "$d"); echo "$t r2p_$RUN: $(ls "$d/r2p_$RUN" | wc -l | tr -d ' ') frames"; done
rm -rf "$T"
