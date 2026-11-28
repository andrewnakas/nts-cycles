#!/bin/bash
# Mac: pull the nts-facezoom-sdxl kernel output, merge each strength into zero-day 420-431, score vs the reference, face sheet.
# Run: facezoom_eval.sh <tag> "<strengths>"   e.g. facezoom_eval.sh hi "0.7 0.85 1.0"
R=~/Documents/nts-cycles; TAG=$1; ST=$2; O=$R/data/facezoom/zd/out_$TAG; cd $R
until kaggle kernels status $KAGGLE_USER/nts-facezoom-sdxl 2>&1 | grep -qiE "complete|error|cancel"; do sleep 60; done
rm -rf $O && mkdir -p $O && kaggle kernels output $KAGGLE_USER/nts-facezoom-sdxl -p $O -q >/dev/null 2>&1
uvr() { uv run --no-project --python 3.11 --with torch==2.2.2 --with torchvision==0.17.2 --with 'numpy<2' --with opencv-python-headless --with pillow --with facenet-pytorch --with open_clip_torch python "$@"; }
M=(in=ntsr5_in); C=("input=ntsr5_in")
for s in $ST; do uvr tools/facezoom.py merge data/eval/zd_sb/ntsr5_in data/eval/zd_sb/masks data/facezoom/zd $O/out/s$s data/eval/zd_sb/fz${TAG}_s$s >/dev/null 2>&1
  M+=("sdxl s$s=fz${TAG}_s$s"); C+=("sdxl $s=fz${TAG}_s$s"); done
M+=("reference=ref5_pass2_clip"); C+=("reference=ref5_pass2_clip")
uvr -m eval.scoreboard data/eval/zd_sb 420-431 runs/ntsr_a/photos_faces.npy "${M[@]}" 2>&1 | grep -E "^\||Traceback|Error" > runs/scoreboard_facezoom_$TAG.md
python3 - "${C[@]}" <<'PY'
import sys
from PIL import Image, ImageDraw
ims = []
for c in sys.argv[1:]:
    l, d = c.split("=")
    im = Image.open(f"data/eval/zd_sb/{d}/f_0430.png").convert("RGB").crop((420, 600, 640, 820)).resize((360, 360), Image.LANCZOS)
    ImageDraw.Draw(im).text((5, 5), l, fill=(255, 255, 0)); ims.append(im)
o = Image.new("RGB", (360 * len(ims), 360)); [o.paste(im, (360 * i, 0)) for i, im in enumerate(ims)]
o.save(sys.argv[0].replace("-", "") and "runs/facezoom_f0430_latest.jpg", quality=92)
PY
cat runs/scoreboard_facezoom_$TAG.md
