"""Synthetic checks of NTS-G frame generation (nts/framegen.py) and of the NTS-F quality-mode ladder. No data needed:
  python tests/test_framegen_modes.py     or     pytest tests/test_framegen_modes.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from baselines.accumulate import Accumulator  # noqa: E402
from nts import conventions as cv  # noqa: E402
from nts import warp as wp  # noqa: E402
from nts.framegen import between, between_surf  # noqa: E402
from nts.surface import Surf  # noqa: E402
from nts.io_exr import Frame  # noqa: E402


def texture(x, y):
    return np.stack([0.5 + 0.4 * np.sin(x * 0.23) * np.cos(y * 0.17), 0.5 + 0.35 * np.sin(x * 0.11 + y * 0.29),
                     0.5 + 0.3 * np.cos(x * 0.19 - y * 0.07)], -1).astype(np.float32) * 255


def scene(t, h=96, w=128):
    """Frame at time t: a wall sliding right 3 px per frame, and in front of it a square sliding left 5 px per
    frame. Returns (image, depth, motion to t-1, motion to t+1), motion in nts.conventions image space."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    img = texture(xs - 3 * t, ys)
    depth = np.full((h, w), 10.0, np.float32)
    mv_prev = np.zeros((h, w, 2), np.float32)
    mv_prev[..., 0] = -3
    sq = (xs >= 70 - 5 * t) & (xs < 100 - 5 * t) & (ys >= 30) & (ys < 70)
    img[sq] = texture(xs + 5 * t + 400, ys)[sq]
    depth[sq] = 2.0
    mv_prev[sq, 0] = 5
    return img, depth, mv_prev, -mv_prev


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / max(float(np.mean((a - b) ** 2)), 1e-9))


def test_framegen_beats_blend():
    a, da, _, a_next = scene(0)
    b, db, b_prev, _ = scene(2)
    gt = scene(1)[0]
    out, conf = between(a, da, a_next, b, db, b_prev)
    inner = (slice(2, -2), slice(8, -8))  # the frame edge has surface neither neighbour saw
    p, q = psnr(out[inner], gt[inner]), psnr((0.5 * (a + b))[inner], gt[inner])
    assert p > 40 and p > q + 15, (p, q)
    assert conf[inner].mean() > 0.95


def test_framegen_occlusion_order():
    """Where the square covers the wall, the square must win whatever the order of the inputs."""
    a, da, _, a_next = scene(0)
    b, db, b_prev, _ = scene(2)
    gt = scene(1)[0]
    out, _ = between(a, da, a_next, b, db, b_prev)
    sq = (slice(32, 68), slice(67, 93))
    assert np.abs(out[sq] - gt[sq]).mean() < 1.0


def test_quality_modes_converge():
    """NTS-F at every quality mode's scale: a static, jittered scene accumulated over the mode's jitter phases must
    beat bilinear upsampling of a single frame clearly (native AA, x1: beat the single jittered frame)."""
    for name, (w, h) in cv.QUALITY_MODES.items():
        s = cv.OUT[0] / w
        lw, lh = 18, 32  # a small crop with the mode's exact scale
        H, W = int(round(lh * s)), int(round(lw * s))
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        gt = texture(xs * 2.1, ys * 2.1) / 255
        acc = Accumulator(s, (H, W), wmax=32.0, expand=8.0)  # a loose box: the scene is static
        phases = cv.jitter_phases(lh * cv.OUT[1] / H)
        for i, j in enumerate(cv.jitter_seq(phases, phases)):
            ox, oy = cv.sample_offset(*j)  # where each render pixel samples the scene, in output px
            gy, gx = np.mgrid[0:lh, 0:lw].astype(np.float32)
            p = np.stack([(gx + 0.5 + ox) * s, (gy + 0.5 + oy) * s], -1)
            lr = texture(p[..., 0] * 2.1 - 1.05, p[..., 1] * 2.1 - 1.05) / 255  # point samples at the true positions
            fr = Frame(f=i, rgb=lr, depth=np.ones((lh, lw), np.float32), mv=np.zeros((lh, lw, 2), np.float32),
                       jitter=j, reset=i == 0)
            out = acc.step(fr)
            if i == 0:
                first = out
        inner = (slice(3, -3), slice(3, -3))
        e_acc = float(np.abs(out - gt)[inner].mean())
        e_one = float(np.abs(first - gt)[inner].mean())
        assert e_acc < 0.8 * e_one, (name, s, e_acc, e_one)


def test_quality_mode_table():
    assert set(cv.QUALITY_MODES) == {"native_aa", "quality", "balanced", "performance", "ultra_performance"}
    for name, (w, h) in cv.QUALITY_MODES.items():
        cv.scale_of({"out": list(cv.OUT), "render": [w, h]})  # aspect kept exactly
    assert [round(cv.quality_mode(m)[1], 2) for m in ("native_aa", "quality", "balanced", "performance", "ultra_performance")] \
        == [1.0, 1.5, 1.71, 2.0, 3.0]
    _ = wp  # (imported for the side effect of checking nts.warp imports cleanly)


def surf_scene(t, h=96, w=128):
    """The surf pass of scene(t): each pixel's point on the wall's or the square's own surface, and the object."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.stack([(xs - 3 * t) / 300, ys / 300, np.full_like(xs, 0.5)], -1)
    oid = np.full((h, w), 0.3125, np.float32)
    sq = (xs >= 70 - 5 * t) & (xs < 100 - 5 * t) & (ys >= 30) & (ys < 70)
    g[sq] = np.stack([(xs + 5 * t) / 300, ys / 300, np.full_like(xs, 0.9)], -1)[sq]
    oid[sq] = 0.75
    return Surf(g, np.stack([(g * g).sum(-1), oid, oid * oid], -1))


def test_framegen_surface_mode():
    """Surface mode at a fractional in-between time: every pixel of frame n finds its own surface in A and B, so the
    square's outline is frame n's own, and the wall it uncovers comes from B."""
    a, b, gt = scene(0)[0], scene(1)[0], scene(0.5)[0]
    out, conf = between_surf(a, surf_scene(0), b, surf_scene(1), surf_scene(0.5))
    inner = (slice(2, -2), slice(8, -8))
    v, q = psnr(out[inner], gt[inner]), psnr((0.5 * (a + b))[inner], gt[inner])
    (_, da, _, a_next), (_, db, b_prev, _) = scene(0), scene(1)
    vec, _ = between(a, da, a_next * 0.5, b, db, b_prev * 0.5)  # vector mode on the same frames
    assert v > 40 and v > q + 10 and v > psnr(vec[inner], gt[inner]), (v, q, psnr(vec[inner], gt[inner]))
    assert conf[inner].mean() > 0.95


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
