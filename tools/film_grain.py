"""Film-grain finishing pass: per-frame random luminance grain (a real camera trait; much of the reference's lead on
the realism probe was grain-like texture, the realism-pass plan). Display-space, stronger in mid-tones.
Run: film_grain.py <in_png_dir> <out_png_dir> [sigma=3] [seed=0]"""
import os
import sys

import numpy as np
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
os.makedirs(out, exist_ok=True)
for f in sorted(os.listdir(src)):
    if not f.endswith(".png"):
        continue
    a = np.asarray(Image.open(os.path.join(src, f)).convert("RGB")).astype(np.float32)
    rng = np.random.default_rng(seed * 100003 + int("".join(c for c in f if c.isdigit()) or 0))
    lum = a.mean(-1, keepdims=True) / 255
    n = rng.normal(0, sigma, a.shape[:2])[..., None] * (4 * lum * (1 - lum)).clip(0.25, 1)  # mid-tone weighted
    Image.fromarray(np.clip(a + n, 0, 255).round().astype(np.uint8)).save(os.path.join(out, f))
print("film_grain:", sigma, "->", out)
