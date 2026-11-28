"""Teacher pilot: one clip's packed npz -> 480x832 portrait crops around the face, through the render's view (+ev).
Writes <out>/in/f_####.png, <out>/masks/m_####.png (R face, G hands, B hair, as eval/scoreboard.py reads them) and
<out>/crop.json. The window is fixed per clip (locked-off cameras), centred on the face of the middle frame.
--extra <png_dir>=<label> crops other full-frame PNG sequences (e.g. the reference's output) with the same window.
Run (4050, Blender's python for OCIO): pilot_crop.py <clip.npz> <meta.json> <out_dir> <ev> [--extra dir=label ...]"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts.tonemap import Tonemapper  # noqa: E402

W, H = 480, 832
src, meta, out, ev = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
extra = [a.split("=", 1) for a in sys.argv[5:] if a != "--extra"]
z = np.load(src)
reg, frames = z["region"], z["frames"]
mid = reg[len(reg) // 2]
m = (mid & 1) > 0 if ((mid & 1) > 0).sum() >= 400 else mid > 0
if m.sum() < 400:
    sys.exit(f"pilot_crop: no person in {src}")
ys, xs = np.nonzero(m)
fh, fw = mid.shape
cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
x0 = int(np.clip(cx - W // 2, 0, fw - W))
y0 = int(np.clip(cy - H // 3, 0, fh - H))  # face in the upper third, upper body below
view = json.load(open(meta))["view"]
view[2] += ev
tm = Tonemapper(*view)
for d in ("in", "masks"):
    os.makedirs(os.path.join(out, d), exist_ok=True)
for i, f in enumerate(frames):
    rgb = z["rgb"][i, :, y0:y0 + H, x0:x0 + W].astype(np.float32).transpose(1, 2, 0)
    Image.fromarray(tm(rgb)).save(os.path.join(out, "in", f"f_{f:04d}.png"))
    r = reg[i, y0:y0 + H, x0:x0 + W]
    Image.fromarray((np.stack([(r & 1) > 0, (r & 2) > 0, (r & 4) > 0], -1) * 255).astype(np.uint8)).save(
        os.path.join(out, "masks", f"m_{f:04d}.png"))
    for d, lab in extra:
        p = os.path.join(d, f"f_{f:04d}.png")
        if os.path.exists(p):
            os.makedirs(os.path.join(out, lab), exist_ok=True)
            Image.open(p).convert("RGB").crop((x0, y0, x0 + W, y0 + H)).save(os.path.join(out, lab, f"f_{f:04d}.png"))
json.dump({"x0": x0, "y0": y0, "w": W, "h": H, "frames": [int(frames[0]), int(frames[-1])], "ev": ev},
          open(os.path.join(out, "crop.json"), "w"))
print(f"pilot_crop: {len(frames)} frames at ({x0},{y0}) -> {out}")
