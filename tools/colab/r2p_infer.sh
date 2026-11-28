#!/bin/bash
# Mac: render the eval crops + whole eval frames of an R2P checkpoint on a cheap Colab GPU (600-unit account).
#   tools/colab/r2p_infer.sh <session> <gpu> <ckpt> <out_dir>
# Pings the kernel while it works (idle runtimes get reclaimed), pulls out/{eval,evalf}, then stops the runtime.
R=~/Documents/nts-cycles; S=$1; GPU=$2; CK=$3; OUT=$4; cd $R
C() { HOME=$HOME/.colab-account-home colab "$@" 2>&1 | grep -v '^\[colab\]'; }
C sessions | grep -q "^\[$S\]" || C new -s $S --gpu $GPU
C sessions | grep "^\[$S\]" | grep -q "Hardware: $GPU" || { echo "no $GPU runtime"; exit 1; }
P=/tmp/r2p_ping_$S.py; echo 'print("ok")' > $P
( while sleep 240; do C exec -s $S --timeout 60 -f $P > /dev/null; done ) & PING=$!
T=$(mktemp -d); mkdir -p $T/data; cp gen/r2p_train.py $T/; cp $CK $T/ck.pt
cp -R data/heads/pub/data/eval data/heads/pub/data/evalf $T/data/
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -cf - -C $T . | tools/colab/cssh.sh $S 'mkdir -p /content/inf && cd /content/inf && tar xf - 2>/dev/null
pip install -q -U diffusers peft lpips > /dev/null 2>&1; pip uninstall -q -y torchao > /dev/null 2>&1
python3 r2p_train.py --data data --out out --infer ck.pt > infer.log 2>&1; tail -3 infer.log >&2' 2>&1 | grep -v "Ignoring unknown"
rm -rf $T; mkdir -p $OUT
tools/colab/cssh.sh $S 'cd /content/inf/out && tar cf - eval evalf' < /dev/null | tar xf - -C $OUT
kill $PING; ls $OUT/evalf | wc -l; C stop -s $S | tail -1
