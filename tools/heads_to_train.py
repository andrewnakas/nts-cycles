"""pack_heads .npz files -> training images for the generative realism model (gen/r2p_train.py).

Run: heads_to_train.py <out_dir> <h*.npz ...> [--size 512] [--ev 0.3]
Per frame: <name>_<f>.jpg (AgX display image, resized to --size), <name>_<f>.reg.png (region mask: R face, G hands,
B hair) and, when frame f-1 is in the same file with the same box, <name>_<f>.mv.npy (float16, quarter size, x 2: where
each pixel was in frame f-1, as an offset in resized pixels) so the pair can be used for the temporal loss.
Also writes <name>.boxes.json. Runs in ~/scene-project/py on the 4050 (numpy, Pillow, OCIO).
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts.tonemap import Tonemapper  # noqa: E402

a = sys.argv
opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
size, ev = int(opt("--size", 512)), float(opt("--ev", 0.3))
dst = a[1]
os.makedirs(dst, exist_ok=True)
files = [x for x in a[2:] if x.endswith(".npz")]
n = pairs = 0
for p in files:
    name = os.path.basename(p)[:-4]
    if os.path.exists(os.path.join(dst, name + ".boxes.json")):
        continue
    z = np.load(p, allow_pickle=True)
    v = list(z["view"])
    tm = Tonemapper(v[0], v[1], float(v[2]) + ev)
    frames, boxes = [int(f) for f in z["frames"]], z["boxes"].tolist()
    for f, b in zip(frames, boxes):
        im = Image.fromarray(tm(z[f"{f}_rgb"].astype(np.float32)))
        s = size / im.width
        im.resize((size, size), Image.LANCZOS).save(os.path.join(dst, f"{name}_{f}.jpg"), quality=96)
        Image.fromarray(z[f"{f}_reg"]).resize((size, size), Image.NEAREST).save(os.path.join(dst, f"{name}_{f}.reg.png"))
        if f - 1 in frames and boxes[frames.index(f - 1)] == b:
            mv = z[f"{f}_mv"].astype(np.float32)
            ch = [np.asarray(Image.fromarray(mv[..., i]).resize((size // 4, size // 4), Image.BILINEAR)) for i in (0, 1)]  # motion is smooth: quarter size
            np.save(os.path.join(dst, f"{name}_{f}.mv.npy"), (np.stack(ch, -1) * s).astype(np.float16))
            pairs += 1
        n += 1
    json.dump({str(f): b for f, b in zip(frames, boxes)}, open(os.path.join(dst, name + ".boxes.json"), "w"))
print(f"heads_to_train: {n} images, {pairs} pairs -> {dst}")
