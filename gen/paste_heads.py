"""Lay translated head crops (gen/r2p_train.py eval output) back over a clip's frames.

Run: paste_heads.py <frames_dir> <crops_dir> <boxes.json> <out_dir> [--feather 24] [--name eval_zd400]
         [--masks <dir of m_####.png> [--grow 14]]
--masks limits the paste to the head itself: the Region mask's face (R) and hair (B), grown by --grow pixels and
softened, so whatever the model did to the background inside the box (it garbles screen text) is not used.
frames_dir holds f_####.png (1080x1920, the same display transform the crops were made with); crops_dir holds
<name>_<frame>.png squares; boxes.json maps frame -> [x0, y0, x1, y1]. Each crop is resized to its box and blended
with a feathered edge (no feather on sides that touch the frame edge). Frames without a crop are copied.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

a = sys.argv
opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
src, crops, boxes, dst = a[1], a[2], json.load(open(a[3])), a[4]
F, name = float(opt("--feather", 24)), opt("--name", "eval_zd400")
masks, grow = opt("--masks", None), int(opt("--grow", 14))
os.makedirs(dst, exist_ok=True)
n = 0
for fn in sorted(os.listdir(src)):
    if not (fn.startswith("f_") and fn.endswith(".png")):
        continue
    f = int(fn[2:6])
    im = np.asarray(Image.open(os.path.join(src, fn)).convert("RGB")).astype(np.float32)
    cp = os.path.join(crops, f"{name}_{f}.png")
    if str(f) in boxes and os.path.exists(cp):
        x0, y0, x1, y1 = boxes[str(f)]
        h, w = y1 - y0, x1 - x0
        c = np.asarray(Image.open(cp).convert("RGB").resize((w, h), Image.LANCZOS)).astype(np.float32)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        big = np.float32(1e6)
        d = np.minimum(np.minimum(xs + 0.5 if x0 > 0 else big, w - xs - 0.5 if x1 < im.shape[1] else big),
                       np.minimum(ys + 0.5 if y0 > 0 else big, h - ys - 0.5 if y1 < im.shape[0] else big))
        t = np.clip(d / F, 0, 1)
        m = (t * t * (3 - 2 * t))[..., None]
        if masks:
            from PIL import ImageFilter
            r = np.asarray(Image.open(os.path.join(masks, f"m_{f:04d}.png")).convert("RGB"))
            head = Image.fromarray(((r[..., 0] > 127) | (r[..., 2] > 127)).astype(np.uint8) * 255).crop((x0, y0, x1, y1))
            head = head.filter(ImageFilter.MaxFilter(2 * grow + 1)).filter(ImageFilter.GaussianBlur(grow / 2))
            m = m * (np.asarray(head).astype(np.float32) / 255)[..., None]
        im[y0:y1, x0:x1] = im[y0:y1, x0:x1] * (1 - m) + c * m
        n += 1
    Image.fromarray(np.clip(im + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(dst, fn))
print(f"paste_heads: {n} heads pasted -> {dst}")
