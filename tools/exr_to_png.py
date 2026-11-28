"""Tonemap a render_lr folder's Image layer to AgX PNGs (f_####.png), with the folder's own view settings.
Run: exr_to_png.py <exr_dir> <out_dir> [--masks <dir>]   (--masks also writes the Region AOV as m_####.png)"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts.io_exr import Sequence, read_layers, stack  # noqa: E402
from nts.tonemap import Tonemapper  # noqa: E402

seq = Sequence(sys.argv[1])
os.makedirs(sys.argv[2], exist_ok=True)
masks = sys.argv[sys.argv.index("--masks") + 1] if "--masks" in sys.argv else None
if masks:
    os.makedirs(masks, exist_ok=True)
tm = Tonemapper(*seq.meta["view"])
for f in seq.frames:
    Image.fromarray(tm(seq.read(f).rgb)).save(os.path.join(sys.argv[2], f"f_{f:04d}.png"))
    if masks:
        ch = read_layers(seq.files[f])
        if "Region.R" in ch:
            reg = stack(ch, "Region.R", "Region.G", "Region.B") > 0.5
            Image.fromarray((reg * 255).astype(np.uint8)).save(os.path.join(masks, f"m_{f:04d}.png"))
