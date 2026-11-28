#!/bin/bash
# Eval only: face match on a CPU Kaggle kernel (run_ident.py), for when the Mac's heavy slots are taken.
#   push_ident.sh <set>=<local dir of *.npz> [...]     adds the crops to nts-pilot2 (faces/<set>) and pushes nts-ident
#   push_ident.sh get                                  prints the tables (data/pilot2/out_ident/identity_<set>.md)
R=~/Documents/nts-cycles; U=$KAGGLE_USER; DS=nts-pilot2; U3=$R/data/pilot2/d3; cd $R
if [ "$1" = get ]; then
  O=$R/data/pilot2/out_ident; mkdir -p $O; kaggle kernels output $U/nts-ident -p $O -q >/dev/null 2>&1; cat $O/identity_*.md; exit 0
fi
SETS=""
for a in "$@"; do s=${a%%=*}; d=${a#*=}; mkdir -p $U3/faces/$s; rsync -a "$d"/ $U3/faces/$s/; SETS="$SETS,$s"; done
# a failed upload (400 = the private storage quota: move old sets from $U3/faces to data/pilot2/d3_faces_old) must show
kaggle datasets version -p $U3 --dir-mode zip -m "faces ${SETS#,}" 2>&1 | grep -E "Error|being created" | tail -2
sleep 90; until kaggle datasets status $U/$DS 2>&1 | grep -q ready; do sleep 20; done   # the status still says "ready" (old version) right after the upload
K=/tmp/kgk_ident; mkdir -p $K
python3 - "${SETS#,}" $K/run_ident.py <<'PY'
import base64, sys
s = open("tools/kaggle/run_ident.py").read()
s = s.replace('CODE_B64 = ""', 'CODE_B64 = "' + base64.b64encode(open("eval/nr_identity.py", "rb").read()).decode() + '"', 1)
s = s.replace('SETS = ""', f'SETS = "{sys.argv[1]}"', 1)
open(sys.argv[2], "w").write(s)
PY
cat > $K/kernel-metadata.json <<J
{"id": "$U/nts-ident", "title": "nts-ident", "code_file": "run_ident.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": false, "enable_internet": true,
 "dataset_sources": ["$U/$DS"]}
J
kaggle kernels push -p $K 2>&1 | grep -v Warning | tail -1
