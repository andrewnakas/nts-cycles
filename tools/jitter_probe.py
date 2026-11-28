"""Measure the real image shift between low-res frames of static content and compare it with jitter.json.

Run: jitter_probe.py <lr_dir> [--frames a-b]   (a static-camera stretch; moving pixels are masked out by their MVs)
Prints, per frame pair, the measured d such that frame b at x matches frame a at x + d (render px, image space),
next to the lens-shift difference from jitter.json. With conventions.SAMPLE_SIGN right, d = SAMPLE_SIGN * (jb - ja).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts.io_exr import Sequence  # noqa: E402
from nts.warp import bilinear  # noqa: E402

seq = Sequence(sys.argv[1])
a0, b0 = map(int, sys.argv[sys.argv.index("--frames") + 1].split("-")) if "--frames" in sys.argv else (2, 8)
fr = {f: seq.read(f) for f in range(a0, b0 + 1)}
still = np.ones(fr[a0].depth.shape, bool)
for f in fr:
    still &= np.hypot(fr[f].mv[..., 0], fr[f].mv[..., 1]) < 0.01
still[:6], still[-6:], still[:, :6], still[:, -6:] = False, False, False, False
ys0, xs0 = np.nonzero(still)
cy, cx = int(np.median(ys0)), int(np.median(xs0))
sl = (slice(max(cy - 150, 6), cy + 150), slice(max(cx - 150, 6), cx + 150))
h, w = still.shape
ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
ys, xs, still = ys[sl], xs[sl], still[sl]
L = {f: np.log1p(np.maximum(fr[f].rgb, 0)).mean(-1) for f in fr}


def search(a, b, cx0, cy0, step, n):
    g = np.arange(-n, n + 1) * step
    return min((float(np.abs(bilinear(L[a], xs + cx0 + dx, ys + cy0 + dy) - L[b][sl])[still].mean()), cx0 + dx, cy0 + dy)
               for dx in g for dy in g)


for a in range(a0, b0):
    b = a + 1
    best = search(a, b, 0.0, 0.0, 0.2, 6)
    best = search(a, b, best[1], best[2], 0.04, 5)
    best = search(a, b, best[1], best[2], 0.01, 4)
    ja, jb = np.array(fr[a].jitter), np.array(fr[b].jitter)
    print(f"{a}->{b}: measured d=({best[1]:+.2f},{best[2]:+.2f}) err {best[0]:.4f}   "
          f"lens-shift jb-ja=({jb[0] - ja[0]:+.3f},{jb[1] - ja[1]:+.3f})", flush=True)
