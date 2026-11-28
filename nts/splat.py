"""Jitter-aware splat: write every low-res sample at its true high-res sub-pixel position with a Gaussian footprint.

Gives a sparse high-res canvas sum(w*c) plus its confidence sum(w). With a static camera and N jitter phases the
canvas converges to a real multi-sample reconstruction instead of an interpolation (Xiao et al. 2020 zero-upsample,
here with a footprint and an explicit confidence channel; see PROVENANCE.md).
"""
import numpy as np

from . import conventions as cv


def splat(rgb, jitter, scale, out_hw, sigma=0.5, radius=1):
    """rgb (h, w, c) low-res -> (sum (H, W, c), weight (H, W)). sigma and radius are in high-res pixels."""
    h, w, c = rgb.shape
    H, W = out_hw
    pos = cv.sample_positions_hr(h, w, *jitter, scale).reshape(-1, 2)
    col = rgb.reshape(-1, c)
    base = np.floor(pos).astype(np.int64)
    acc = np.zeros((H * W, c), np.float64)
    wsum = np.zeros(H * W, np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            tx, ty = base[:, 0] + dx, base[:, 1] + dy
            ok = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
            d2 = (tx + 0.5 - pos[:, 0]) ** 2 + (ty + 0.5 - pos[:, 1]) ** 2
            wt = np.exp(-d2 / (2 * sigma * sigma)) * ok
            idx = np.where(ok, ty * W + tx, 0)
            wsum += np.bincount(idx, wt, H * W)
            for k in range(c):
                acc[:, k] += np.bincount(idx, wt * col[:, k], H * W)
    return acc.reshape(H, W, c).astype(np.float32), wsum.reshape(H, W).astype(np.float32)
