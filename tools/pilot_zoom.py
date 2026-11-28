"""Teacher pilot, zoomed framing: a 240x416 window around the head of each pilot clip, upscaled 2x to 480x832, so the
teacher sees a ~185 px face instead of ~90 px (small faces are where video diffusion drifts most). Writes
data/pilot/<clip>z/{in,masks,ref5}/ and crop.json. Run (Mac): pilot_zoom.py <pilot_dir>"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

P = sys.argv[1]
for c in sorted(glob.glob(os.path.join(P, "*", "crop.json"))):
    d = os.path.dirname(c)
    if d.endswith("z"):
        continue
    meta = json.load(open(c))
    ms = sorted(glob.glob(os.path.join(d, "masks", "m_*.png")))
    m = np.asarray(Image.open(ms[len(ms) // 2]).convert("RGB")) > 127
    k = m[..., 0] if m[..., 0].sum() >= 100 else m.any(-1)
    ys, xs = np.nonzero(k)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    x0 = int(np.clip(cx - 120, 0, m.shape[1] - 240)); y0 = int(np.clip(cy - 416 // 3, 0, m.shape[0] - 416))
    box = (x0, y0, x0 + 240, y0 + 416)
    out = d + "z"
    for sub, rs in (("in", Image.LANCZOS), ("masks", Image.NEAREST), ("ref5", Image.LANCZOS)):
        if not os.path.isdir(os.path.join(d, sub)):
            continue
        os.makedirs(os.path.join(out, sub), exist_ok=True)
        for f in sorted(os.listdir(os.path.join(d, sub))):
            Image.open(os.path.join(d, sub, f)).convert("RGB").crop(box).resize((480, 832), rs).save(os.path.join(out, sub, f))
    json.dump({**meta, "zoom": {"x0": x0, "y0": y0, "scale": 2}}, open(os.path.join(out, "crop.json"), "w"))
    print(os.path.basename(out), box)
