"""Temporal steadying of a generative pass: the change the model made (output - render) is accumulated along Blender's
exact motion, so invented detail sticks to the surface instead of being re-invented every frame.

  d_f   = generated_f - render_f                    what the model added to this frame
  acc_f = a * d_f + (1 - a) * warp(acc_{f-1})       where the warped render matches the current render
        = d_f                                        at cuts, newly visible pixels, and where the render changed
  out_f = render_f + acc_f
By default the render itself is never blended over time, so nothing lags: only the added detail is averaged.
With --base b (0 < b < 1) the finished frame is accumulated as well (render noise is the larger part of the flicker):
  hist_f = b * out_f + (1 - b) * clamp(warp(hist_{f-1}), 3x3 min/max of out_f +- --slack)
The clamp keeps history from trailing behind anything that really changed; newly visible pixels take out_f.

Run: python -m gen.steady <geom_exr_dir> <render_dir> <gen_dir> <out_dir> [--alpha 0.25] [--tol 6] [--frames a-b]
         [--base 0.2] [--slack 4]
     geom_exr_dir: any render_lr folder of the shot (Vector, Depth); frame dirs hold f_####.png|jpg at output size.
Prints kept = std(acc) / std(d): how much of the added detail survives the averaging.
Needs numpy, Pillow, OpenImageIO (~/scene-project/py on the 4050).
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts import warp as W  # noqa: E402
from nts.io_exr import Sequence  # noqa: E402


def frame(d, f):
    p = glob.glob(os.path.join(d, f"f_{f:04d}.*"))
    return np.asarray(Image.open(p[0]).convert("RGB")).astype(np.float32) if p else None


def minmax3(im):
    p = np.pad(im, ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = im.shape[:2]
    v = [p[i:i + h, j:j + w] for i in range(3) for j in range(3)]
    return np.minimum.reduce(v), np.maximum.reduce(v)


def main():
    a = sys.argv
    if len(a) < 5:
        sys.exit(__doc__)
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    alpha, tol = float(opt("--alpha", 0.25)), float(opt("--tol", 6))
    base, slack = float(opt("--base", 1.0)), float(opt("--slack", 4))
    hist = None
    seq = Sequence(a[1])
    span = tuple(map(int, opt("--frames", "0-999999").split("-")))
    os.makedirs(a[4], exist_ok=True)
    acc = prev_x = prev_depth = None
    n, sa, sd = 0, 0.0, 0.0
    for f in seq.frames:
        if not span[0] <= f <= span[1]:
            continue
        x, y = frame(a[2], f), frame(a[3], f)
        if x is None or y is None:
            acc = None
            continue
        fr = seq.read(f)
        hw = x.shape[:2]
        depth = W.upsample_nearest(fr.depth, seq.scale, hw)
        d = y - x
        if acc is None or fr.reset:
            acc, hist = d, None
        else:
            mv = W.hr_motion(fr.mv, fr.depth, seq.scale, hw)
            same = np.exp(-np.abs(W.warp(prev_x, mv) - x).mean(-1) / tol)  # 1 where the render shows the same surface
            same[W.disocclusion(prev_depth, depth, mv)] = 0
            w = ((1 - alpha) * same)[..., None]
            acc = d * (1 - w) + W.warp(acc, mv) * w
        out = x + acc
        if base < 1:
            if hist is not None:
                lo, hi = minmax3(out)
                h = np.clip(W.warp(hist, mv), lo - slack, hi + slack)
                wb = ((1 - base) * ~W.disocclusion(prev_depth, depth, mv))[..., None]
                out = out * (1 - wb) + h * wb
            hist = out
        Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(a[4], f"f_{f:04d}.png"))
        prev_x, prev_depth = x, depth
        sa, sd, n = sa + float(acc.std()), sd + float(d.std()), n + 1
    print(f"steady: {n} frames -> {a[4]}; alpha {alpha}, kept {sa / max(sd, 1e-9):.2f}")


if __name__ == "__main__":
    main()
