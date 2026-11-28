"""Identity lock for teacher frames (plan phase 1): keep the input's geometry, take the teacher's realism.
Inside the soft face/hair/hands mask: out = colour_transfer(low(input)) + (teacher - low(teacher)), where
colour_transfer matches Lab mean/std per region to the teacher's and low() is a Gaussian of `sigma` px. Outside the
mask (screens, text, set) the input is kept. Also writes the raw teacher frames' sheet row in pilot_eval.sh.
Run: teacher_lock.py <in_dir> <teacher_dir> <masks_dir> <out_dir> <sigma>"""
import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image

src, tdir, mdir, out, sigma = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
os.makedirs(out, exist_ok=True)
low = lambda a: cv2.GaussianBlur(a, (0, 0), sigma)
for f in sorted(glob.glob(os.path.join(src, "f_*.png"))):
    n = os.path.basename(f)
    i = np.asarray(Image.open(f).convert("RGB"))
    t = np.asarray(Image.open(os.path.join(tdir, n)).convert("RGB"))
    m = np.asarray(Image.open(os.path.join(mdir, "m_" + n[2:])).convert("RGB")) > 127
    il, tl = cv2.cvtColor(i, cv2.COLOR_RGB2LAB).astype(np.float32), cv2.cvtColor(t, cv2.COLOR_RGB2LAB).astype(np.float32)
    lo = low(il)
    for r in range(3):  # face, hands, hair: per-region Lab mean/std transfer of the input's low band
        k = m[..., r]
        if k.sum() < 50:
            continue
        mi, si, mt, st = il[k].mean(0), il[k].std(0) + 1e-3, tl[k].mean(0), tl[k].std(0) + 1e-3
        lo[k] = (lo[k] - mi) * np.clip(st / si, 0.7, 1.4) + mt
    res = np.clip(lo + (tl - low(tl)), 0, 255).astype(np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_LAB2RGB).astype(np.float32)
    w = cv2.GaussianBlur(cv2.dilate(m.any(-1).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 3)[..., None]
    Image.fromarray((w * res + (1 - w) * i).round().astype(np.uint8)).save(os.path.join(out, n))
print(f"teacher_lock: sigma {sigma} -> {out}")
