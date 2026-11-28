"""Analytical temporal upscaler (no network): this frame's jittered samples + reprojected, clipped, weighted history.

It's the floor NTS-1 must beat and the check that conventions.py is right: with a wrong jitter or MV sign it falls
apart. It is also the reference the GPU port (nts/rt) is tested against. Per output pixel:
  - this frame's samples: the output pixel's position in the jittered render and the 2x2 render pixels around it,
    Gaussian-weighted by their distance in render pixels (nts/gather.py);
  - the previous output and its weight, warped along the motion (Catmull-Rom; the four corner taps left out and the
    rest renormalised, as a GPU reads it in five filtered lookups), dropped where the depth test finds another surface;
  - the history clamped to the per-channel min/max of the render pixel's 3x3 neighbourhood (Karis 2014), widened
    about its middle by expand >= 1, never narrowed; history that had to move far to fit loses weight;
  - one weighted sum of history, samples and the bilinear frame (the last at a small fixed weight, prior).
The constants are conventions.ACCUM (fitted by nts/rt/fit.py). Run with Blender's Python (OpenImageIO + OCIO):
  accumulate.py <lr_dir> <out_dir> [--fmt png] [--linear-exr] [--frames a-b] [--sample-sign sx,sy] [--mv-sign sx,sy]
                [--foot f] [--expand e] [--wmax w] [--shrink k] [--prior w0] [--rel r]
                [--history catmull5|catmull|bilinear] [--space rgb|ycocg] [--clip clamp|centre] [--static-expand e]
                [--depth-test nearest|match]
--history catmull keeps all 16 Catmull-Rom taps; bilinear samples the history bilinearly. --space ycocg takes the box in
luma/chroma (YCoCg) instead of RGB; --clip centre moves history along the line to the box's middle until it is inside,
instead of clamping each channel. --static-expand e widens the box by e >= --expand where the pixel did not move
(motion under 0.01 px).
"""
import os
import sys
import warnings

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts import conventions as cv  # noqa: E402
from nts import warp as wp  # noqa: E402
from nts.io_exr import Sequence  # noqa: E402
from nts.gather import gather  # noqa: E402
from nts.tonemap import Tonemapper  # noqa: E402


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def to_ycocg(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return np.stack([r * 0.25 + g * 0.5 + b * 0.25, r * 0.5 - b * 0.5, g * 0.5 - r * 0.25 - b * 0.25], -1)


def from_ycocg(a):
    y, co, cg = a[..., 0], a[..., 1], a[..., 2]
    return np.stack([y + co - cg, y + cg, y - co - cg], -1)


def minmax_box(rgb_lr, expand, out_hw, space="rgb"):
    """Per-channel min and max of the current frame's 3x3 low-res neighbourhood (Karis 2014), widened about its middle
    by expand >= 1, taken at high res from the render pixel each output pixel falls in (no blend of neighbouring boxes,
    so no sample of the neighbourhood ever lies outside its box). space "ycocg": the box of the samples in YCoCg."""
    h, w, _ = rgb_lr.shape
    p = np.pad(to_ycocg(rgb_lr) if space == "ycocg" else rgb_lr, ((1, 1), (1, 1), (0, 0)), mode="edge")
    win = np.stack([p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    lo, hi = win.min(0), win.max(0)
    if expand != 1.0:
        c, r = (lo + hi) * np.float32(0.5), (hi - lo) * np.float32(0.5 * expand)
        lo0, lo, hi = lo, c - r, c + r
        if space == "rgb":  # never widened below zero unless the samples go there (resolve.wgsl says why)
            lo = np.maximum(lo, np.minimum(lo0, np.float32(0.0)))
    H, W = out_hw
    ys, xs = (np.arange(H) * h) // H, (np.arange(W) * w) // W
    return lo[ys][:, xs], hi[ys][:, xs]


def cr_weights(t):
    """Catmull-Rom weights for taps at -1, 0, 1, 2 given the fractional offset t (as gen/steady2.py)."""
    t2, t3 = t * t, t * t * t
    return (-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1, -1.5 * t3 + 2 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2)


def warp_catmull(img, mv, corners=True):
    """Backward warp of an (H, W, c) buffer with Catmull-Rom sampling, edges clamped, all in float32: rows of four taps
    summed left to right, then the four rows (the order nts/rt/shaders/resolve.wgsl uses). corners=False: the four
    corner taps left out and the other twelve weights renormalised to sum to one."""
    H, W = mv.shape[:2]
    ys, xs = wp.grid(H, W)
    x = np.clip(xs + mv[..., 0], 0, W - 1).astype(np.float32)
    y = np.clip(ys + mv[..., 1], 0, H - 1).astype(np.float32)
    x0, y0 = np.floor(x), np.floor(y)
    wx, wy = cr_weights((x - x0)[..., None]), cr_weights((y - y0)[..., None])
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    flat = img.reshape((H * W,) + img.shape[2:])
    cols = [np.clip(x0 + i - 1, 0, W - 1) for i in range(4)]
    out = np.zeros_like(img)
    for j in range(4):
        yy = np.clip(y0 + j - 1, 0, H - 1) * W
        row = np.zeros_like(img)
        for i in range(4):
            if corners or (i in (1, 2) or j in (1, 2)):
                row += flat.take(yy + cols[i], axis=0) * wx[i]
        out += row * wy[j]
    if not corners:
        out /= 1.0 - (wx[0] + wx[3]) * (wy[0] + wy[3])  # the corners' share: (wx0 + wx3)(wy0 + wy3)
    return out


class Accumulator:
    def __init__(self, scale, out_hw, wmax=cv.ACCUM["wmax"], expand=cv.ACCUM["expand"], mv_sign=(1.0, 1.0),
                 foot=cv.ACCUM["foot"], prior=cv.ACCUM["prior"], history="catmull5", space="rgb", clip="clamp",
                 static_expand=None, shrink=cv.ACCUM["shrink"], rel=cv.ACCUM["rel"], depth_test="nearest",
                 sigma=None):
        """shrink: how fast history weight falls with how far the clip moved it; rel: depth tolerance of the
        disocclusion test (relative); prior: weight of the bilinear frame in the sum (> 0); foot: footprint of this
        frame's samples, a Gaussian sigma in render px (nts/gather.py); depth_test: "nearest" compares the nearest of
        the previous depths under the reprojected footprint, "match" keeps the history if any of them matches.
        sigma: deprecated (0.2.0's footprint in output px); taken as foot = sigma / scale."""
        if sigma is not None:
            warnings.warn("Accumulator(sigma=) is deprecated: use foot= (render px, sigma / scale)", DeprecationWarning,
                          stacklevel=2)
            foot = sigma / scale
        assert expand >= 1.0 and prior > 0.0, (expand, prior)  # the box is never narrowed below the samples' min/max
        assert history in ("bilinear", "catmull", "catmull5") and space in ("rgb", "ycocg") and clip in ("clamp", "centre")
        assert static_expand is None or static_expand >= expand, static_expand
        assert depth_test in ("nearest", "match"), depth_test
        self.depth_test = depth_test
        assert foot > 0, foot
        self.foot = foot
        self.scale, self.hw, self.wmax, self.mv_sign = scale, out_hw, wmax, np.array(mv_sign, np.float32)
        self.expand, self.prior, self.history = float(expand), prior, history
        self.space, self.clip, self.static_expand = space, clip, static_expand
        self.shrink, self.rel = np.float32(shrink), rel
        self.h = self.w = self.depth = None

    def step(self, fr, reject=None):
        """reject: optional (H, W) multiplier in [0, 1] on the history weight (e.g. from a full-res guide's change)."""
        s, ws = gather(fr.rgb, fr.jitter, self.hw, self.foot)
        base = wp.upsample_bilinear(fr.rgb, self.scale, self.hw)
        depth = wp.upsample_nearest(fr.depth, self.scale, self.hw)
        if fr.reset or self.h is None:
            hist, whist = base, np.zeros(self.hw, np.float32)
        else:
            mv = wp.hr_motion(fr.mv * self.mv_sign, fr.depth, self.scale, self.hw)
            if self.history == "bilinear":
                hist = wp.warp(self.h, mv)
            else:
                hist = warp_catmull(self.h, mv, corners=self.history == "catmull")
            whist = wp.warp(self.w, mv)
            whist[wp.disocclusion(self.depth, depth, mv, self.rel, self.depth_test == "match")] = 0.0
            if reject is not None:
                whist = whist * reject
            lo, hi = minmax_box(fr.rgb, self.expand, self.hw, self.space)
            if self.static_expand is not None:  # surfaces that didn't move keep sub-pixel detail a tight box can't hold
                slo, shi = minmax_box(fr.rgb, self.static_expand, self.hw, self.space)
                still = (np.hypot(mv[..., 0], mv[..., 1]) < 0.01)[..., None]
                lo, hi = np.where(still, slo, lo), np.where(still, shi, hi)
            hh = to_ycocg(hist) if self.space == "ycocg" else hist
            if self.clip == "centre":  # along the line to the box's middle, just far enough to be inside
                c, e = (lo + hi) * np.float32(0.5), (hi - lo) * np.float32(0.5) + np.float32(1e-6)
                v = hh - c
                m = (np.abs(v) / e).max(-1, keepdims=True)
                clipped = np.where(m > 1, c + v / m, hh)
            else:
                clipped = np.clip(hh, lo, hi)
            if self.space == "ycocg":
                clipped = from_ycocg(clipped)
            # history that had to move a lot to fit the neighbourhood is less trustworthy: shrink its weight
            moved = np.abs(clipped - hist).sum(-1) / (np.abs(hist).sum(-1) + 1e-3)
            whist *= np.clip(1.0 - self.shrink * moved, 0.0, 1.0)
            hist = clipped
        tot = whist + ws
        w0 = np.float32(self.prior)  # the bilinear frame as one more term of fixed weight in the same sum
        out = (hist * whist[..., None] + s + base * w0) / np.maximum(tot + w0, 1e-6)[..., None]
        self.h, self.w, self.depth = out, np.minimum(tot, self.wmax), depth
        return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    seq = Sequence(sys.argv[1])
    dst = sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    fmt = arg("--fmt", "png")
    if arg("--sample-sign"):
        cv.SAMPLE_SIGN = tuple(map(float, arg("--sample-sign").split(",")))
    mv_sign = tuple(map(float, arg("--mv-sign", "1,1").split(",")))  # extra sign on top of conventions.MV_SIGN
    span = tuple(map(int, arg("--frames").split("-"))) if arg("--frames") else None
    ow, oh = seq.meta["out"]
    num = {k: float(arg("--" + k, v)) for k, v in cv.ACCUM.items()}
    if arg("--sigma"):  # deprecated: 0.2.0's footprint in output px
        num["foot"] = float(arg("--sigma")) / seq.scale
    acc = Accumulator(seq.scale, (oh, ow), mv_sign=mv_sign, history=arg("--history", "catmull5"),
                      space=arg("--space", "rgb"), clip=arg("--clip", "clamp"), depth_test=arg("--depth-test", "nearest"),
                      static_expand=float(arg("--static-expand")) if arg("--static-expand") else None, **num)
    tm = Tonemapper(*seq.meta["view"])
    for f in seq.frames:
        if span and not span[0] <= f <= span[1]:
            continue
        out = acc.step(seq.read(f))
        Image.fromarray(tm(out)).save(os.path.join(dst, f"f_{f:04d}.{fmt}"), quality=92)
        if "--linear-exr" in sys.argv:
            np.save(os.path.join(dst, f"f_{f:04d}.npy"), out.astype(np.float16))
    print(f"accumulate: {len(seq.frames)} frames -> {dst}")


if __name__ == "__main__":
    main()
