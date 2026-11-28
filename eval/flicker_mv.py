"""True-motion flicker and follow ratio for finished clips, using Blender's own Vector and Depth passes.

eval.flicker_flow warps with Farneback flow, whose error grows with texture, so detailed looks are penalised. Here the
previous frame of every method is warped with the exact motion from a geometry render of the same shot (any render_lr
folder: native 1080x1920, or the low-res perf folder, whose motion is dilated and upsampled), and newly visible
pixels (depth test) are left out.

  flicker  mean |cur - warp(prev)| inside the eroded face mask, 8-bit display units. Lower = steadier.
  follow   how much of the true motion the picture follows, on face pixels that move: the frame-to-frame change is
           regressed on the change the method's own current frame would show if it moved rigidly by the true motion.
           1.0 = moves with the surface; below 1 = lags or sticks to the screen. Frames are blurred (sigma 1.5 px)
           first so grain does not dominate. Compare with the native render's value, not with 1.

Needs numpy, Pillow, OpenImageIO only (runs in ~/scene-project/py on the 4050). Eval only.
Run: python -m eval.flicker_mv <geom_exr_dir> <a-b> <masks_dir|region> label=<frames_dir> ...
     ("region" takes the face mask from the geometry render's Region AOV; frames_dir holds 1080x1920 f_####.png|jpg)
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts import warp as W  # noqa: E402
from nts.io_exr import Sequence, read_layers  # noqa: E402


def erode(m, n):
    for _ in range(n):
        p = np.pad(m, 1, mode="constant")
        m = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m


def blur(a, sigma=1.5):
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r, r)
        p = np.pad(a, pad, mode="edge")
        n = a.shape[ax]
        a = sum(k[i] * np.take(p, np.arange(i, i + n), axis=ax) for i in range(2 * r + 1))
    return a


def frame(d, f):
    p = glob.glob(os.path.join(d, f"f_{f:04d}.*"))
    return np.asarray(Image.open(p[0]).convert("RGB")).astype(np.float32)


def geometry(seq, f0, f1, masks):
    """Per frame f (f0 < f <= f1): motion at output size, valid-history mask, face mask."""
    out, prev_depth = {}, None
    for f in range(f0, f1 + 1):
        fr = seq.read(f)
        h, w = fr.depth.shape
        s = seq.scale
        hw = (round(h * s), round(w * s))
        depth = W.upsample_nearest(fr.depth, s, hw)
        if prev_depth is not None and not fr.reset:
            mv = W.hr_motion(fr.mv, fr.depth, s, hw)
            if masks == "region":
                face = W.upsample_nearest(read_layers(seq.files[f])["Region.R"], s, hw) > 0.5
            else:
                face = np.asarray(Image.open(os.path.join(masks, f"m_{f:04d}.png")).convert("RGB"))[..., 0] > 127
            ok = erode(face, 4) & ~W.disocclusion(prev_depth, depth, mv)
            out[f] = (mv.astype(np.float32), ok)
        prev_depth = depth
    return out


def main():
    a = sys.argv
    if len(a) < 5:
        sys.exit(__doc__)
    seq = Sequence(a[1])
    f0, f1 = map(int, a[2].split("-"))
    geo = geometry(seq, f0, f1, a[3])
    print("| method | flicker | flicker p90 | follow |\n|---|---|---|---|")
    for lab, d in (x.split("=", 1) for x in a[4:]):
        res, num, den = [], 0.0, 0.0
        prev = frame(d, f0)
        for f in range(f0 + 1, f1 + 1):
            cur = frame(d, f)
            if f in geo:
                mv, ok = geo[f]
                res.append(np.abs(cur - W.warp(prev, mv)).mean(-1)[ok])
                moving = ok & (np.hypot(mv[..., 0], mv[..., 1]) > 0.3)
                if moving.sum() > 50:
                    cb, pb = blur(cur.mean(-1)), blur(prev.mean(-1))
                    pred = cb - W.warp(cb, -mv)   # change if the current frame had moved rigidly by the true motion
                    num += float(((cb - pb) * pred)[moving].sum())
                    den += float((pred * pred)[moving].sum())
            prev = cur
        r = np.concatenate(res)
        print(f"| {lab} | {r.mean():.2f} | {np.percentile(r, 90):.2f} | {num / den if den else float('nan'):.2f} |")


if __name__ == "__main__":
    main()
