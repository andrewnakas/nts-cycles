"""Bidirectional steady: combine gen/steady2.py run forward and with --reverse into non-causal NR-1 targets
(2026-11-19; the design log rows 11 and 18).

Each one-way steady lags: its history comes only from the past (or only from the future). Averaging the two coarse
bands cancels the lag. The fine bands are averaged too, but two partly independent textures lose energy when
averaged, so the mean fine band is rescaled to the two inputs' mean local energy (15 px window, gain within
[1/--gain, --gain]; Heitz & Neyret 2018, variance-preserving blending, paper). --bands 2 does that per sub-band as
gen/steady2.py does.

With gen/steady2.py --key-stride (a keys.json in the forward folder), each direction is weighted by how far its
last keyframe is: at frame f between keyframes p <= f <= q the forward result gets (q - f) / (q - p), so every frame
leans on the nearer keyframe and the result is continuous through keyframes.

Run: python -m gen.steady_bi <forward_dir> <backward_dir> <out_dir> [--gain 1.6] [--bands 1|2]
CPU only (numpy, Pillow). Frames present in both folders are written as f_####.png.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gen.steady2 import lowpass, split, win  # noqa: E402


def frames(d):
    out = {}
    for p in glob.glob(os.path.join(d, "f_*.png")) + glob.glob(os.path.join(d, "f_*.jpg")):
        out[int(os.path.basename(p)[2:6])] = p
    return out


def main():
    a = sys.argv
    if len(a) < 4:
        sys.exit(__doc__)
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    gmax, nb = float(opt("--gain", 1.6)), int(opt("--bands", 1))
    fw, bw = frames(a[1]), frames(a[2])
    os.makedirs(a[3], exist_ok=True)
    kp = os.path.join(a[1], "keys.json")
    keys = sorted(__import__("json").load(open(kp))) if os.path.exists(kp) else None

    def wf_of(f):
        """weight of the forward result at frame f"""
        if not keys:
            return 0.5
        p, q = max([k for k in keys if k <= f], default=None), min([k for k in keys if k >= f], default=None)
        if p is None or q is None or p == q:
            return 0.5
        return (q - f) / (q - p)

    def gain(target_e, h):
        eh = win((h * h).sum(-1))
        return np.clip(np.sqrt((target_e + 1e-3) / (eh + 1e-3)), 1 / gmax, gmax)[..., None]
    n = 0
    for f in sorted(set(fw) & set(bw)):
        F = np.asarray(Image.open(fw[f]).convert("RGB")).astype(np.float32)
        B = np.asarray(Image.open(bw[f]).convert("RGB")).astype(np.float32)
        cf, cb = lowpass(F), lowpass(B)
        ff, fb = F - cf, B - cb
        bands = list(zip(split(ff), split(fb))) if nb == 2 else [(ff, fb)]
        fine, wf = 0, wf_of(f)
        for x, y in bands:
            m = wf * x + (1 - wf) * y
            e = wf * win((x * x).sum(-1)) + (1 - wf) * win((y * y).sum(-1))
            fine = fine + m * gain(e, m)
        out = wf * cf + (1 - wf) * cb + fine
        Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(a[3], f"f_{f:04d}.png"))
        n += 1
    print(f"steady_bi: {n} frames -> {a[3]} (bands {nb})")


if __name__ == "__main__":
    main()
