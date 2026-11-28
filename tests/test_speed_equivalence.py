"""The 2026-11-24 speed pass must not change a single output byte. Each fast primitive is checked against the version
it replaced, copied here verbatim as the reference, on random data with the awkward cases (NaN, inf, ties in depth,
1-pixel-wide arrays, bool masks, wide windows). Synthetic, no data needed:
  python tests/test_speed_equivalence.py     or     pytest tests/test_speed_equivalence.py
The whole-stage check on real frames (steadied base, tone, carry: byte-identical files at any --workers / --threads)
is in docs/SPEED.md.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gen import steady2 as st  # noqa: E402
from gen import structure as gs  # noqa: E402
from nts import tone as tn  # noqa: E402
from nts import warp as wp  # noqa: E402
from nts.par import pmap  # noqa: E402

RNG = np.random.default_rng(7)


# ---- the replaced versions, verbatim ------------------------------------------------------------------------------
def ref_bilinear(img, x, y):
    h, w = img.shape[:2]
    x, y = np.clip(x, 0, w - 1.001), np.clip(y, 0, h - 1.001)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    fx, fy = x - x0, y - y0
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) +
            img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def ref_dilate_mv(mv, depth, r=1):
    h, w = depth.shape
    dp = np.pad(depth, r, mode="edge")
    mp = np.pad(mv, ((r, r), (r, r), (0, 0)), mode="edge")
    best, out = depth.copy(), mv.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = dp[r + dy:r + dy + h, r + dx:r + dx + w]
            closer = d < best
            best = np.where(closer, d, best)
            out = np.where(closer[..., None], mp[r + dy:r + dy + h, r + dx:r + dx + w], out)
    return out


def ref_upsample_nearest(a, scale, out_hw):
    H, W = out_hw
    ys = np.minimum((np.arange(H) / scale).astype(np.int64), a.shape[0] - 1)
    xs = np.minimum((np.arange(W) / scale).astype(np.int64), a.shape[1] - 1)
    return a[ys][:, xs]


def ref_hr_motion(mv_lr, depth_lr, scale, out_hw):
    return ref_upsample_nearest(ref_dilate_mv(mv_lr, depth_lr), scale, out_hw) * scale


def ref_warp(prev_hr, mv_hr):
    H, W = mv_hr.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return ref_bilinear(prev_hr, xs + mv_hr[..., 0], ys + mv_hr[..., 1])


def ref_disocclusion(prev_depth_hr, cur_depth_hr, mv_hr, rel=0.05):
    H, W = mv_hr.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    px, py = xs + mv_hr[..., 0], ys + mv_hr[..., 1]
    off = (px < 0) | (py < 0) | (px > W - 1) | (py > H - 1)
    x0, y0 = np.clip(px, 0, W - 2).astype(np.int64), np.clip(py, 0, H - 2).astype(np.int64)
    pd = np.minimum(np.minimum(prev_depth_hr[y0, x0], prev_depth_hr[y0, x0 + 1]),
                    np.minimum(prev_depth_hr[y0 + 1, x0], prev_depth_hr[y0 + 1, x0 + 1]))
    return off | (np.abs(pd - cur_depth_hr) > rel * cur_depth_hr)


def ref_st_box(a, r):
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax)
        n = a.shape[ax]
        a = (np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=ax) - np.take(c, np.arange(0, n), axis=ax)) / (2 * r + 1)
    return a


def ref_warp_cr(img, mv):
    H, Wd = mv.shape[:2]
    ys, xs = np.mgrid[0:H, 0:Wd].astype(np.float32)
    x, y = np.clip(xs + mv[..., 0], 0, Wd - 1), np.clip(ys + mv[..., 1], 0, H - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    wx, wy = st.cr_weights((x - x0)[..., None]), st.cr_weights((y - y0)[..., None])
    out = np.zeros_like(img)
    for j in range(4):
        yy = np.clip(y0 + j - 1, 0, H - 1)
        row = np.zeros_like(img)
        for i in range(4):
            row += img[yy, np.clip(x0 + i - 1, 0, Wd - 1)] * wx[i]
        out += row * wy[j]
    return out


def ref_tn_box(a, r):
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax, dtype=np.float64)
        n = a.shape[ax]
        a = ((np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=ax) - np.take(c, np.arange(0, n), axis=ax))
             / (2 * r + 1)).astype(np.float32)
    return a


def ref_max3(m):
    p = np.pad(m, 1, mode="edge")
    h, w = m.shape
    return np.maximum.reduce([p[i:i + h, j:j + w] for i in range(3) for j in range(3)])


def ref_maxr(m, r):
    for ax in (0, 1):
        pad = [(0, 0)] * m.ndim
        pad[ax] = (r, r)
        p = np.pad(m, pad, mode="edge")
        n = m.shape[ax]
        m = np.maximum.reduce([np.take(p, np.arange(i, i + n), axis=ax) for i in range(2 * r + 1)])
    return m


def ref_to_lab(img):
    c = img / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)
    xyz = lin @ tn.M / tn.WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return 116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])


def ref_gauss(a, sigma):
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r, r)
        p = np.pad(a, pad, mode="edge")
        n = a.shape[ax]
        a = sum(k[i] * np.take(p, np.arange(i, i + n), axis=ax) for i in range(2 * r + 1))
    return a


# ---- checks ---------------------------------------------------------------------------------------------------------
def same(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
    assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), what


def depth_field(h, w):
    d = RNG.choice([1.0, 1.5, 3.0], (h, w)).astype(np.float32) + RNG.integers(0, 2, (h, w)).astype(np.float32)
    d[RNG.random((h, w)) < 0.02] = np.inf  # background with no hit
    return d


def test_warp_primitives():
    for h, w in ((64, 48), (37, 5), (2, 3)):
        img3, img1 = RNG.random((h, w, 3)).astype(np.float32), RNG.random((h, w)).astype(np.float32)
        mv = RNG.normal(0, 4, (h, w, 2)).astype(np.float32)
        for img in (img3, img1, img3.astype(np.float64)[..., :1]):
            same(wp.warp(img, mv), ref_warp(img, mv), ("warp", h, w, img.shape))
            x, y = RNG.uniform(-3, w + 3, (h, w)), RNG.uniform(-3, h + 3, (h, w))
            same(wp.bilinear(img, x, y), ref_bilinear(img, x, y), ("bilinear", h, w))
        d = depth_field(h, w)
        same(wp.disocclusion(d, d[::-1].copy(), mv), ref_disocclusion(d, d[::-1].copy(), mv), "disocclusion")
        same(wp.Warper(mv).disocclusion(d, d[::-1].copy()), ref_disocclusion(d, d[::-1].copy(), mv), "Warper.disocclusion")
        same(wp.dilate_mv(mv, d), ref_dilate_mv(mv, d), "dilate_mv")
        for sc in (0.25, 0.5, 1.0, 2.0, 1.5):
            hw = (max(1, int(h * sc)), max(1, int(w * sc)))
            same(wp.hr_motion(mv, d, sc, hw), ref_hr_motion(mv, d, sc, hw), ("hr_motion", sc))
            same(wp.upsample_nearest(d, sc, hw), ref_upsample_nearest(d, sc, hw), ("upsample_nearest", sc))
        same(st.warp_cr(img3, mv), ref_warp_cr(img3, mv), "warp_cr")


def test_filters():
    for shape in ((61, 47), (1, 9), (9, 1), (24, 30, 3)):
        for dt in (np.float32, np.float64):
            a = (RNG.random(shape) * 100).astype(dt)
            if dt == np.float64:
                a.flat[RNG.integers(0, a.size)] = np.nan
            for r in (1, 2, 3, 4, 9, 31):
                same(tn.maxr(a, r), ref_maxr(a, r), ("maxr", shape, dt, r))
                same(tn.maxr(-a, r), ref_maxr(-a, r), ("maxr neg", shape, dt, r))
                same(tn.box(a, r), ref_tn_box(a, r), ("tone.box", shape, dt, r))
                if dt == np.float32:
                    same(st.box(a, r), ref_st_box(a, r), ("steady2.box", shape, r))
            if a.ndim == 2:
                same(tn.max3(a), ref_max3(a), ("max3", shape, dt))
            for sg in (0.7, 1.5, 3.0):
                same(gs.gauss(a, sg), ref_gauss(a, sg), ("gauss", shape, dt, sg))
    b = RNG.integers(0, 2, (40, 33)).astype(bool)
    same(tn.maxr(b, 6), ref_maxr(b, 6), "maxr bool")
    same(tn.max3(b), ref_max3(b), "max3 bool")
    i = RNG.integers(0, 9, (20, 21))
    same(st.box(i, 2), ref_st_box(i, 2), "steady2.box int")


def test_to_lab():
    img = RNG.integers(0, 256, (90, 70, 3)).astype(np.float32)
    for a, b in zip(tn.to_lab(img), ref_to_lab(img)):
        same(a, b, "to_lab levels")
    off = img + RNG.random(img.shape).astype(np.float32)
    for a, b in zip(tn.to_lab(off), ref_to_lab(off)):
        same(a, b, "to_lab non-integer")
    for a, b in zip(tn.to_lab(img.astype(np.float64)), ref_to_lab(img.astype(np.float64))):
        same(a, b, "to_lab float64")
    every = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, -1)  # each level once: uint8 decodes in float64
    for u8 in (img.astype(np.uint8), every):
        for a, b in zip(tn.to_lab(u8), ref_to_lab(u8)):
            same(a, b, "to_lab uint8")
        for a, b in zip(tn.to_lab(u8.astype(np.float32)), ref_to_lab(u8.astype(np.float32))):
            same(a, b, "to_lab float32 levels, all 256")


def _square(x):
    return x * x


def test_pmap_order():
    assert pmap(_square, range(7), 1) == [x * x for x in range(7)]
    assert pmap(_square, range(7), 3) == [x * x for x in range(7)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
