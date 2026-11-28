"""History reprojection: motion-vector upsampling with nearest-depth dilation, bilinear backward warp, disocclusion.

All motion here is image space, current -> previous (conventions.mv_image).

Speed (2026-11-24): the sampling positions, flat gather indices and fractional weights of a motion field are worked
out once (Warper) and reused for every buffer warped along it, the pixel grid is cached per shape, and gathers use
flat np.take instead of 2-D fancy indexing. The arithmetic is the same expressions in the same order as before, so
results are bit-identical (tests/test_speed_equivalence.py).
"""
import functools

import numpy as np


@functools.lru_cache(maxsize=8)
def _grid(h, w):
    """(ys, xs) float32 pixel-index grids, read-only and shared."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ys.flags.writeable = xs.flags.writeable = False
    return ys, xs


def grid(h, w):
    return _grid(int(h), int(w))


class _Bilinear:
    """Gather indices and weights of a bilinear lookup at (x, y), for any number of buffers of one shape."""
    __slots__ = ("h", "w", "i", "fx", "fy", "gx", "gy")

    def __init__(self, h, w, x, y):
        x, y = np.clip(x, 0, w - 1.001), np.clip(y, 0, h - 1.001)
        x0, y0 = x.astype(np.int64), y.astype(np.int64)
        self.h, self.w = h, w
        self.fx, self.fy = x - x0, y - y0
        self.gx, self.gy = 1 - self.fx, 1 - self.fy
        self.i = y0 * w + x0

    def __call__(self, img):
        h, w = self.h, self.w
        assert img.shape[:2] == (h, w), (img.shape, (h, w))
        flat = img.reshape((h * w,) + img.shape[2:])
        i = self.i
        fx, fy, gx, gy = self.fx, self.fy, self.gx, self.gy
        if img.ndim == 3:
            fx, fy, gx, gy = fx[..., None], fy[..., None], gx[..., None], gy[..., None]
        # img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) + img[y0 + 1, x0] * (1 - fx) * fy
        #   + img[y0 + 1, x0 + 1] * fx * fy, evaluated left to right as before, with fewer temporaries
        out = flat.take(i, axis=0) * gx
        out *= gy
        t = flat.take(i + 1, axis=0) * fx
        t *= gy
        out += t
        t = flat.take(i + w, axis=0) * gx
        t *= fy
        out += t
        t = flat.take(i + (w + 1), axis=0) * fx
        t *= fy
        out += t
        return out


def bilinear(img, x, y):
    """Sample img (H, W[, c]) at continuous pixel-index coordinates x, y (pixel i's centre is at i); edges clamped."""
    h, w = img.shape[:2]
    return _Bilinear(h, w, x, y)(img)


def dilate_mv(mv, depth, r=1):
    """Per low-res pixel, take the motion of the nearest surface in a (2r+1)^2 window, so edges of moving foreground
    objects carry the foreground motion (the FSR 2 / Karis 2014 trick)."""
    h, w = depth.shape
    dp = np.pad(depth, r, mode="edge")
    mp = np.pad(mv, ((r, r), (r, r), (0, 0)), mode="edge")
    best, out = depth.copy(), mv.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = dp[r + dy:r + dy + h, r + dx:r + dx + w]
            closer = d < best
            np.copyto(best, d, where=closer)
            np.copyto(out, mp[r + dy:r + dy + h, r + dx:r + dx + w], where=closer[..., None])
    return out


def upsample_nearest(a, scale, out_hw):
    H, W = out_hw
    ys = np.minimum((np.arange(H) / scale).astype(np.int64), a.shape[0] - 1)
    xs = np.minimum((np.arange(W) / scale).astype(np.int64), a.shape[1] - 1)
    if scale == 1 and a.shape[:2] == (H, W):
        return a.copy()
    return a[ys][:, xs]


def upsample_bilinear(a, scale, out_hw):
    H, W = out_hw
    ys, xs = grid(H, W)
    return bilinear(a, (xs + 0.5) / scale - 0.5, (ys + 0.5) / scale - 0.5)


def _dilate_at(mv, depth, ys, xs, r=1):
    """dilate_mv(mv, depth, r)[ys][:, xs] without dilating the pixels that are not kept: each kept pixel looks at its
    own (2r+1)^2 window (edges repeated) in the same order with the same strict test, so the result is identical."""
    h, w = depth.shape
    best = depth[ys][:, xs]
    out = mv[ys][:, xs]
    for dy in range(-r, r + 1):
        yy = np.clip(ys + dy, 0, h - 1)
        for dx in range(-r, r + 1):
            xx = np.clip(xs + dx, 0, w - 1)
            d = depth[yy][:, xx]
            closer = d < best
            np.copyto(best, d, where=closer)
            np.copyto(out, mv[yy][:, xx], where=closer[..., None])
    return out


def hr_motion(mv_lr, depth_lr, scale, out_hw):
    """Low-res render-pixel motion -> high-res pixel motion field (H, W, 2). When the field is smaller than the
    render (scale < 1: nts.tone's quarter-size votes) only the pixels nearest sampling keeps are dilated."""
    if scale < 1:
        H, W = out_hw
        ys = np.minimum((np.arange(H) / scale).astype(np.int64), mv_lr.shape[0] - 1)
        xs = np.minimum((np.arange(W) / scale).astype(np.int64), mv_lr.shape[1] - 1)
        return _dilate_at(mv_lr, depth_lr, ys, xs) * scale
    return upsample_nearest(dilate_mv(mv_lr, depth_lr), scale, out_hw) * scale


class Warper:
    """Backward warps along one high-res motion field: Warper(mv)(buf) == warp(buf, mv), with the lookup worked out
    once for all the buffers that move with the same motion."""

    def __init__(self, mv_hr):
        H, W = mv_hr.shape[:2]
        ys, xs = grid(H, W)
        self.mv = mv_hr
        self.px, self.py = xs + mv_hr[..., 0], ys + mv_hr[..., 1]
        self._b = _Bilinear(H, W, self.px, self.py)

    def __call__(self, prev_hr):
        return self._b(prev_hr)

    def disocclusion(self, prev_depth_hr, cur_depth_hr, rel=0.05, match=False):
        return _disocclusion(prev_depth_hr, cur_depth_hr, self.px, self.py, rel, match)


def warp(prev_hr, mv_hr):
    """Backward-warp a high-res buffer from frame n-1 into frame n."""
    return Warper(mv_hr)(prev_hr)


def _disocclusion(prev_depth_hr, cur_depth_hr, px, py, rel, match=False):
    H, W = cur_depth_hr.shape[:2]
    off = (px < 0) | (py < 0) | (px > W - 1) | (py > H - 1)
    # the previous depths of the 2x2 footprint, taken as they are: bilinear depth would invent surfaces at silhouettes
    x0, y0 = np.clip(px, 0, W - 2).astype(np.int64), np.clip(py, 0, H - 2).astype(np.int64)
    i = y0 * W + x0
    flat = prev_depth_hr.reshape(-1)
    taps = (flat.take(i), flat.take(i + 1), flat.take(i + W), flat.take(i + (W + 1)))
    if match:  # another surface only if no tap is this one: a footprint that reaches over a silhouette still matches
        d = np.minimum(np.minimum(np.abs(taps[0] - cur_depth_hr), np.abs(taps[1] - cur_depth_hr)),
                       np.minimum(np.abs(taps[2] - cur_depth_hr), np.abs(taps[3] - cur_depth_hr)))
    else:  # the nearest tap: a background pixel next to a nearer surface fails
        d = np.abs(np.minimum(np.minimum(taps[0], taps[1]), np.minimum(taps[2], taps[3])) - cur_depth_hr)
    return off | (d > rel * cur_depth_hr)


def disocclusion(prev_depth_hr, cur_depth_hr, mv_hr, rel=0.05, match=False):
    """True where the reprojected previous surface is not the current one (newly visible, or off-screen history).
    match: another surface when no previous depth of the 2x2 footprint is within rel of this one (default: when the
    nearest of them is not)."""
    H, W = mv_hr.shape[:2]
    ys, xs = grid(H, W)
    return _disocclusion(prev_depth_hr, cur_depth_hr, xs + mv_hr[..., 0], ys + mv_hr[..., 1], rel, match)
