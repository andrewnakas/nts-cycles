"""Linear fp16 frames (<dir>/f_####.npy, H x W x 3) -> PNGs through a render's view transform.
Run (with Blender's OCIO, e.g. ~/scene-project/py on the 4050): tonemap_npy.py <npy_dir> <meta.json> <png_dir> [ev]
ev: extra exposure on top of the render's view (e.g. 0.3 for a look that only changes the view)."""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts.tonemap import Tonemapper  # noqa: E402

src, meta, out = sys.argv[1:4]
view = json.load(open(meta))["view"]
view[2] += float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
tm = Tonemapper(*view)
os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(os.path.join(src, "f_*.npy"))):
    Image.fromarray(tm(np.load(f).astype(np.float32))).save(os.path.join(out, os.path.basename(f)[:-4] + ".png"))
print("tonemap_npy:", len(glob.glob(os.path.join(out, "f_*.png"))), "frames ->", out)
