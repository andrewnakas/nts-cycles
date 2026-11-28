"""This frame's samples for every output pixel of the temporal accumulator (baselines/accumulate.py, nts/rt).

Each output pixel finds its own position in the jittered render and takes the render pixels around it (inverse
mapping, as any resampler does); see PROVENANCE.md.
"""
import numpy as np

from . import conventions as cv


def gather(rgb, jitter, out_hw, foot):
    """This frame's samples for every output pixel, found from the render side (inverse mapping, as any resampler):
    the output pixel's centre in the render's jittered sample grid, u = (X + 0.5) / sx - 0.5 - ox with sx = W / w
    (render pixel i's sample lies at u = i), and the 2 x 2 render pixels around it, weighted by a Gaussian of their
    distance from it in render pixels (sigma foot; separable). Pixels outside the render get no weight.
    -> (weighted sum (H, W, c), weight (H, W)), float32 as the GPU computes it."""
    h, w, c = rgb.shape
    H, W = out_hw
    ox, oy = cv.sample_offset(*jitter)
    k = np.float32(1.0 / (2.0 * foot * foot))

    def axis(n_out, n_in, s, o):
        u = (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) / np.float32(s) - np.float32(0.5) - np.float32(o)
        i0 = np.floor(u).astype(np.int64)
        idx = np.stack([i0, i0 + 1], 1)
        d = idx.astype(np.float32) - u[:, None]
        wt = np.where((idx >= 0) & (idx < n_in), np.exp(-(d * d) * k), np.float32(0.0)).astype(np.float32)
        return np.clip(idx, 0, n_in - 1), wt

    ix, wx = axis(W, w, W / w, ox)
    iy, wy = axis(H, h, H / h, oy)
    acc = np.zeros((H, W, c), np.float32)
    for a in range(2):
        rows = rgb[iy[:, a]].astype(np.float32)
        for b in range(2):
            acc += (wy[:, a, None, None] * wx[None, :, b, None]) * rows[:, ix[:, b]]
    return acc, (wy.sum(1)[:, None] * wx.sum(1)[None, :]).astype(np.float32)

