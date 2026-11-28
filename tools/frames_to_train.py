"""Full-frame native render (render_lr mode "native", multilayer EXR) -> whole-frame training images for the
generative realism model (gen/r2p_train.py, cgf/ folder).

Run: frames_to_train.py <exr_dir> <out_dir> <name> [--ev 0.3] [--gbuf]
Per frame f: <name>_<f>.jpg (AgX display image, full size), <name>_<f>.reg.png (R face, G hands, B hair) and, when
frame f-1 was rendered too, <name>_<f>.mv.npy (float16, 1/8 size: where each pixel was in frame f-1, as an offset in
full-size pixels, current -> previous). With --gbuf also <name>_<f>.alb.jpg / .nrm.png / .dep.png (nts/gbuf.py: albedo,
camera-space normal, log depth) for the NR-1 G-buffer adapter. Runs in ~/scene-project/py on the 4050.
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts import conventions as cv  # noqa: E402
from nts.io_exr import read_layers, stack  # noqa: E402
from nts.tonemap import Tonemapper  # noqa: E402

a = sys.argv
src, dst, name = a[1], a[2], a[3]
ev = float(a[a.index("--ev") + 1]) if "--ev" in a else 0.3
gb = "--gbuf" in a
if gb:
    from nts.gbuf import save_gbuf  # noqa: E402
os.makedirs(dst, exist_ok=True)
v = list(json.load(open(os.path.join(src, "meta.json")))["view"])
tm = Tonemapper(v[0], v[1], float(v[2]) + ev)
files = {int(os.path.basename(p)[2:6]): p for p in glob.glob(os.path.join(src, "f_*.exr"))}
n = pairs = 0
for f, p in sorted(files.items()):
    ch = read_layers(p)
    rgb = np.nan_to_num(stack(ch, "Image.R", "Image.G", "Image.B"), nan=0.0, posinf=0.0, neginf=0.0)
    if rgb.max() <= 0:
        continue
    Image.fromarray(tm(rgb.astype(np.float32))).save(os.path.join(dst, f"{name}_{f}.jpg"), quality=93)
    reg = (np.clip(stack(ch, "Region.R", "Region.G", "Region.B"), 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(reg).save(os.path.join(dst, f"{name}_{f}.reg.png"))
    if gb:
        save_gbuf(ch, os.path.join(dst, f"{name}_{f}"))
    if f - 1 in files:
        mv = cv.mv_image(stack(ch, "Vector.X", "Vector.Y")).astype(np.float32)
        h, w = mv.shape[:2]
        q = [np.asarray(Image.fromarray(mv[..., i]).resize((w // 8, h // 8), Image.BILINEAR)) for i in (0, 1)]
        np.save(os.path.join(dst, f"{name}_{f}.mv.npy"), np.stack(q, -1).astype(np.float16))
        pairs += 1
    n += 1
if n:
    open(os.path.join(dst, name + ".done"), "w").write(f"{n} {pairs}\n")
print(f"frames_to_train: {n} images, {pairs} pairs -> {dst}")
