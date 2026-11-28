"""Whole-episode run: one key frame per shot, picked without a person: near the shot's middle, eyes open.
  py runs/v2_ep_keys.py <prefix> [look=v11e] [window=12] [-v]
Reads the shots' renders and masks (data/eval/ab/<prefix>NN/{<look>,masks}); prints "<tag>:<frame>" per shot and one
line "KEYS <tag>:<frame> ..." (the list v2_ep_carry.sh takes). Nothing is written.
Eyes open = many pixels in the eye band of the face that are not skin-coloured (whites, irises); a blink hides them.
A frame counts as open when it and its two neighbours on each side have at least 85 % of the best count within the
window; the one nearest the shot's middle is taken. Checked on agentic-time against the picks made by eye (design log row 115)."""
import glob
import os
import sys

import numpy as np
from PIL import Image

a = [x for x in sys.argv[1:] if x != "-v"]
pre, lk, win = a[0], (a[1] if len(a) > 1 else "v11e"), int(a[2]) if len(a) > 2 else 12
verbose = "-v" in sys.argv


def score(t, f):
    im = np.asarray(Image.open("data/eval/ab/%s/%s/f_%04d.png" % (t, lk, f)).convert("RGB")).astype(np.float32)
    m = np.asarray(Image.open("data/eval/ab/%s/masks/m_%04d.png" % (t, f)).convert("RGB"))
    face = (m[..., 0] > 127) & (m[..., 2] < 128)
    if face.sum() < 200:
        return 0.0
    ys = np.nonzero(face.any(1))[0]
    y0, h = ys[0], ys[-1] - ys[0] + 1
    q = (im[..., 0] - im[..., 2]) / (im[..., 0] + im[..., 2] + 1)      # skin is red over blue; whites and irises are not
    band = np.zeros(face.shape, bool)
    band[int(y0 + 0.15 * h):int(y0 + 0.6 * h)] = True
    eye = face & band & (q < 0.4 * np.median(q[face])) & (im.mean(-1) > 40)
    return 1000.0 * eye.sum() / face.sum()


keys = []
for d in sorted(glob.glob("data/eval/ab/%s[0-9][0-9]" % pre)):
    t = os.path.basename(d)
    fs = sorted(int(os.path.basename(p)[2:6]) for p in glob.glob("%s/%s/f_*.png" % (d, lk)))
    if not fs:
        continue
    mid = (fs[0] + fs[-1]) // 2
    cand = [f for f in fs if abs(f - mid) <= win]
    sc = {f: score(t, f) for f in cand}
    best = max(sc.values())
    ok = [f for f in cand if sc[f] >= 0.85 * best]
    # ... and two frames clear of a blink on both sides (a lid that is just closing still scores as open)
    calm = [f for f in ok if all(sc.get(g, 0.0) >= 0.85 * best for g in range(f - 2, f + 3))]
    k = min(calm or ok, key=lambda f: (abs(f - mid), f))
    keys.append("%s:%d" % (t, k))
    print("%s:%d  (shot %d-%d, middle %d, eye score %.1f of best %.1f)" % (t, k, fs[0], fs[-1], mid, sc[k], best))
    if verbose:
        print("   " + " ".join("%d=%.0f" % (f, sc[f]) for f in cand))
print("KEYS " + " ".join(keys))
