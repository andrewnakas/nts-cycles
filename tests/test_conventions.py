"""Synthetic checks of the splat / warp / jitter plumbing. Plain functions: run with pytest, or `python tests/test_conventions.py`
(the 4050's Blender Python has no pytest).

Blender's own signs (MV_SIGN, SAMPLE_SIGN) can't be proven synthetically; baselines/accumulate.py --sample-sign sweeps
them on a real render. These tests check that the code applies whatever the convention says consistently.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts import conventions as cv  # noqa: E402
from nts import warp as wp  # noqa: E402
from nts.splat import splat  # noqa: E402


def pattern(x, y):
    """A smooth-but-detailed band-limited test image, defined in continuous high-res pixel coordinates."""
    return np.stack([0.5 + 0.4 * np.sin(x * 0.9) * np.cos(y * 0.7), 0.5 + 0.4 * np.sin(x * 0.31 + y * 0.53),
                     0.5 + 0.3 * np.cos(x * 0.47 - y * 0.21)], axis=-1).astype(np.float32)


def render_lr(h, w, scale, jitter, shift=(0.0, 0.0)):
    """Sample the pattern where conventions say a jittered low-res pixel samples (point samples, no filter)."""
    p = cv.sample_positions_hr(h, w, *jitter, scale)
    return pattern(p[..., 0] - shift[0], p[..., 1] - shift[1])


def test_halton_phases():
    assert cv.jitter_phases(960) == 32 and cv.jitter_phases(1920) == 8
    seq = cv.jitter_seq(64, 32)
    assert seq[0] == seq[32] and len(set(seq[:32])) == 32
    assert all(-0.5 <= a < 0.5 and -0.5 <= b < 0.5 for a, b in seq)


def test_resets():
    assert cv.resets([1, 2, 3, 5, 6, 25, 26], [1, 25]) == [True, False, False, True, False, True, False]


def test_sample_positions_follow_sign():
    p = cv.sample_positions_hr(2, 2, 0.25, 0.25, 2.0)
    ox, oy = cv.sample_offset(0.25, 0.25)
    assert np.allclose(p[0, 0], [(0.5 + ox) * 2, (0.5 + oy) * 2])
    assert ox == 0.25 * cv.SAMPLE_SIGN[0] and oy == 0.25 * cv.SAMPLE_SIGN[1]


def test_splat_static_converges():
    """Static scene, 32 jitter phases: the splat canvas must beat bilinear upsampling of one frame by a wide margin."""
    h, w, s = 24, 16, 2.0
    H, W = int(h * s), int(w * s)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    gt = pattern(xs + 0.5, ys + 0.5)
    acc, wsum = np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32)
    for j in cv.jitter_seq(32, 32):
        a, b = splat(render_lr(h, w, s, j), j, s, (H, W))
        acc, wsum = acc + a, wsum + b
    rec = acc / wsum[..., None]
    up = wp.upsample_bilinear(render_lr(h, w, s, (0.0, 0.0)), s, (H, W))
    inner = (slice(2, -2), slice(2, -2))
    e_rec, e_up = np.abs(rec - gt)[inner].mean(), np.abs(up - gt)[inner].mean()
    assert e_rec < 0.5 * e_up, (e_rec, e_up)
    # the wrong sign must be clearly worse: proves the splat really uses the offsets
    wrong = np.zeros_like(acc)
    for j in cv.jitter_seq(32, 32):
        a, _ = splat(render_lr(h, w, s, j), (-j[0], -j[1]), s, (H, W))
        wrong += a
    e_wrong = np.abs(wrong / wsum[..., None] - gt)[inner].mean()
    assert e_wrong > 1.5 * e_rec, (e_wrong, e_rec)


def test_warp_translation():
    """Content moves +3 px right per frame: mv (current -> previous) is (-3, 0); warping frame n-1 predicts frame n."""
    H, W = 32, 48
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    prev, cur = pattern(xs, ys), pattern(xs - 3.0, ys)
    mv = np.zeros((H, W, 2), np.float32)
    mv[..., 0] = -3.0
    inner = (slice(2, -2), slice(6, -6))
    assert np.abs(wp.warp(prev, mv) - cur)[inner].max() < 1e-4
    assert np.abs(wp.warp(prev, -mv) - cur)[inner].mean() > 0.05


def test_blender_vector_to_image():
    """Blender Vector.Y is +up: an object moving down the screen has prev above it, i.e. Vector.Y > 0 -> image mv.y < 0."""
    v = np.array([[[2.0, 1.5]]], np.float32)
    assert np.allclose(cv.mv_image(v), [[[2.0, -1.5]]])


def test_dilate_prefers_foreground():
    depth = np.full((5, 5), 10.0, np.float32)
    depth[2, 2] = 1.0
    mv = np.zeros((5, 5, 2), np.float32)
    mv[2, 2] = (4.0, -1.0)
    d = wp.dilate_mv(mv, depth)
    assert np.allclose(d[1, 1], (4.0, -1.0)) and np.allclose(d[0, 0], 0.0)


def test_disocclusion():
    H, W = 8, 8
    prev = np.full((H, W), 5.0, np.float32)
    cur = prev.copy()
    cur[3:5, 3:5] = 1.0  # a near object appeared
    occ = wp.disocclusion(prev, cur, np.zeros((H, W, 2), np.float32))
    assert occ[3:5, 3:5].all() and not occ[:2].any()


if __name__ == "__main__":
    names = [n for n in dir() if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print(f"{len(names)} passed")
