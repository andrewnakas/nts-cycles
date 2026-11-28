#!/bin/bash
# Light-group native renders of the 12 look-variety frames (one Blender session per episode), then the native PNGs +
# masks the reference job turns into references (REQUEST_3) and the fitter reads.
J=~/nts-cycles/datagen/jobs/lg_frame.sh
R=~/nts-cycles/data/look/refs
mkdir -p $R
for spec in "zero-day:60,185,295,430,500,605,720" "agentic-time:170,310" "last-programmer:450" "ram-shortage:790,950"; do
  ep=${spec%%:*}; fr=${spec#*:}
  $J $ep $fr ${ep}_lg
  D=~/nts-cycles/data/look/lg/${ep}_lg
  ~/scene-project/py ~/nts-cycles/tools/exr_to_png.py $D $D/png --masks $D/png
  for f in $(echo $fr | tr , " "); do F=$(printf %04d $f)
    cp $D/png/f_$F.png $R/native_${ep}_f$F.png; cp $D/png/m_$F.png $R/m_${ep}_$F.png; done
  echo "$ep done"
done
echo "REFS DONE" | tee -a ~/nts-cycles/data/look/refs.txt
