#!/bin/bash
# Light-group native renders for a batch of look-variety frames, then native PNGs + masks for the references.
#   lg_batch.sh <batch_name> "<episode>:<f,f,...>" ["<episode>:<f,...>" ...]
# Writes data/look/lg/<episode>_lg/ and data/look/refs/native_<episode>_f####.png, then appends
# "<BATCH> REFS DONE" to data/look/refs.txt.
B=$1; shift
J=~/nts-cycles/datagen/jobs/lg_frame.sh
R=~/nts-cycles/data/look/refs
mkdir -p $R
for spec in "$@"; do
  ep=${spec%%:*}; fr=${spec#*:}
  $J $ep $fr ${ep}_lg
  D=~/nts-cycles/data/look/lg/${ep}_lg
  ~/scene-project/py ~/nts-cycles/tools/exr_to_png.py $D $D/png --masks $D/png
  for f in $(echo $fr | tr , " "); do F=$(printf %04d $f)
    cp $D/png/f_$F.png $R/native_${ep}_f$F.png; cp $D/png/m_$F.png $R/m_${ep}_$F.png; done
  echo "$B $ep done"
done
echo "$B REFS DONE" | tee -a ~/nts-cycles/data/look/refs.txt
