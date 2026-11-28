"""Kaggle run 7 edits (made from the toned native keyframes) -> key teacher frames.
py runs/v2_p7keys.py <kind> <tag> [tone=x19t] [full_kind]
  kind       data/v2/p7/<kind>/key_<clip>_<frame>.png: edited head crops (at the crop size)
  tag        writes data/v2/<clip>/<tag>_keys/f_####.png = the toned frame (data/v2/<clip>/<tone>) with the edited
             head pasted into its box (feathered inside the box, as runs/v2_p2build.sh did against the native frame)
  full_kind  optional: data/v2/p7/<full_kind>/key_*.png whole-frame edits -> data/v2/<clip>/<tag>_full/f_####.png
The teacher input of these keys is the toned frame itself (data/v2/<clip>/<tone>). Nothing is deleted."""
import json, os, re, sys
import numpy as np
from PIL import Image
kind, tag = sys.argv[1:3]
tone = sys.argv[3] if len(sys.argv) > 3 else "x19t"
fkind = sys.argv[4] if len(sys.argv) > 4 else None
boxes = json.load(open("data/v2/pilot2/keys/boxes.json"))
n = nf = 0
for name, (x0, y0, x1, y1) in sorted(boxes.items()):
    clip, f = re.match(r"key_(\w+)_(\d+)\.png", name).groups()
    f = int(f)
    bp = "data/v2/%s/%s/f_%04d.png" % (clip, tone, f)
    p = "data/v2/p7/%s/%s" % (kind, name)
    if os.path.exists(p) and os.path.exists(bp):
        base = np.asarray(Image.open(bp).convert("RGB")).astype(np.float32)
        ed = np.asarray(Image.open(p).convert("RGB").resize((x1 - x0, y1 - y0), Image.LANCZOS)).astype(np.float32)
        s = x1 - x0
        r = max(8, s // 16)
        H, W = base.shape[:2]
        ys, xs = np.mgrid[0:y1 - y0, 0:s]
        dd = np.full((y1 - y0, s), 1e9, np.float32)   # a box edge on the frame edge needs no feather
        if x0 > 0: dd = np.minimum(dd, xs + 0.5)
        if x1 < W: dd = np.minimum(dd, s - xs - 0.5)
        if y0 > 0: dd = np.minimum(dd, ys + 0.5)
        if y1 < H: dd = np.minimum(dd, (y1 - y0) - ys - 0.5)
        w = np.clip(dd / r, 0, 1)[..., None]
        out = base.copy()
        out[y0:y1, x0:x1] = base[y0:y1, x0:x1] * (1 - w) + ed * w
        d_ = "data/v2/%s/%s_keys" % (clip, tag)
        os.makedirs(d_, exist_ok=True)
        Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save("%s/f_%04d.png" % (d_, f))
        n += 1
    if fkind:
        p = "data/v2/p7/%s/%s" % (fkind, name)
        if os.path.exists(p) and os.path.exists(bp):
            W, H = Image.open(bp).size
            d_ = "data/v2/%s/%s_full" % (clip, tag)
            os.makedirs(d_, exist_ok=True)
            Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS).save("%s/f_%04d.png" % (d_, f))
            nf += 1
print(n, "key teacher frames,", nf, "whole-frame keys for", tag)
