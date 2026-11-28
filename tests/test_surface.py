"""nts/surface.py on a synthetic scene with known correspondences: a curved sheet seen at the keyframe and, moved,
rotated and scaled, at another frame, with an occluder that hides part of the sheet at the keyframe only."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts import surface as S  # noqa: E402

H, W = 120, 160
SHEET, BLOCK = 0.3125, 0.75  # object random numbers


def sheet_params(A, t):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    Ai = np.linalg.inv(A)
    sx = Ai[0, 0] * (xs - t[0]) + Ai[0, 1] * (ys - t[1])
    sy = Ai[1, 0] * (xs - t[0]) + Ai[1, 1] * (ys - t[1])
    return sx, sy


def render(A, t, block=None, uv=False, drift=0.0):
    """Surf pass of the sheet under p = A s + t (s in [0,100] x [0,80]); block = (x0, y0, x1, y1) pixel rectangle of a
    second object in front. uv: add a UV layer; drift: shift the Generated coordinates by this much (in units of one
    pixel's worth), as a deforming mesh's Generated coordinates can move."""
    sx, sy = sheet_params(A, t)
    g = np.stack([sx / 100 + drift * 0.01, sy / 100, 0.2 + 0.05 * np.sin(sx / 15)], -1)
    u = np.stack([0.1 + sx / 140, 0.2 + sy / 140], -1)
    on = (sx >= 0) & (sx <= 100) & (sy >= 0) & (sy <= 80)
    oid = np.where(on, SHEET, 0.0)
    g = np.where(on[..., None], g, 0.0)
    if block:
        x0, y0, x1, y1 = block
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
        b = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
        g[b] = np.stack([xs / 300, ys / 300, np.full_like(xs, 0.9)], -1)[b]
        oid = np.where(b, BLOCK, oid)
    sid = np.stack([(g * g).sum(-1), oid, oid * oid], -1)
    u = np.where(on[..., None], u, 0.0)
    if block:
        u[b] = np.stack([0.7 + xs / 400, 0.6 + ys / 400], -1)[b]
    uvl = np.concatenate([u, (u * u).sum(-1, keepdims=True)], -1).astype(np.float32) if uv else None
    return S.Surf(g.astype(np.float32), sid.astype(np.float32), uv=uvl), (sx, sy, on)


def rot(deg, k):
    a = np.radians(deg)
    return k * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def test_match_against_truth(uv=False, drift=0.0):
    Ak, tk = rot(0, 1.0), np.array([20.0, 20.0])
    Af, tf = rot(10, 1.2), np.array([30.0, 10.0])
    block = (60, 40, 90, 70)
    key, _ = render(Ak, tk, block, uv)
    cur, (sx, sy, on) = render(Af, tf, None, uv, drift)
    m, v = S.match(S.Key(key), cur)
    u_true = np.stack([Ak[0, 0] * sx + Ak[0, 1] * sy + tk[0], Ak[1, 0] * sx + Ak[1, 1] * sy + tk[1]], -1)
    # what the keyframe shows of the sheet at u_true: inside the frame and not under the block (1 px margins)
    ux, uy = u_true[..., 0], u_true[..., 1]
    seen = on & (ux >= 1) & (ux <= W - 2) & (uy >= 1) & (uy <= H - 2)
    hidden = on & (ux > block[0] - 1.5) & (ux < block[2] + 0.5) & (uy > block[1] - 1.5) & (uy < block[3] + 0.5)
    inner = seen & ~hidden
    # interior of the visible part: well away from the sheet's and the block's outlines
    sxk, syk = ux - tk[0], uy - tk[1]
    deep = inner & (sxk > 2) & (sxk < 98) & (syk > 2) & (syk < 78) & \
        ~((ux > block[0] - 3) & (ux < block[2] + 2) & (uy > block[1] - 3) & (uy < block[3] + 2))
    err = np.hypot(*(m - u_true).transpose(2, 0, 1))
    assert v[deep].mean() > 0.99, v[deep].mean()
    assert np.percentile(err[deep & (v > 0)], 99) < 0.02, np.percentile(err[deep & (v > 0)], 99)
    core = hidden & (ux > block[0] + 1) & (ux < block[2] - 2) & (uy > block[1] + 1) & (uy < block[3] - 2)
    assert core.sum() > 100 and v[core].max() == 0, "a point the keyframe does not show got a match"
    assert v[~on].max() == 0


def test_uv_survives_generated_drift():
    """With UVs, Generated coordinates that move by 3 pixels' worth between the frames (seen on at140's hands) do not
    move the matches."""
    test_match_against_truth(uv=True, drift=3.0)


def test_generated_alone_follows_drift():
    """Without UVs the drifted coordinates are matched where they point, 3 px off: why the surf pass has UVs."""
    Ak, tk = rot(0, 1.0), np.array([20.0, 20.0])
    key, _ = render(Ak, tk)
    cur, (sx, sy, on) = render(rot(10, 1.2), np.array([30.0, 10.0]), drift=3.0)
    m, v = S.match(S.Key(key), cur)
    ux = sx + tk[0]
    err = np.abs(m[..., 0] - ux)
    assert np.percentile(err[(v > 0) & on], 50) > 2  # matched where the drifted coordinates point: 3 px off


def test_mixed_pixels_are_left_out():
    g = np.zeros((4, 4, 3), np.float32) + 0.5
    sid = np.stack([(g * g).sum(-1), np.full((4, 4), SHEET), np.full((4, 4), SHEET ** 2)], -1).astype(np.float32)
    a, b = np.array([0.2, 0.3, 0.4]), np.array([0.6, 0.3, 0.4])
    g[1, 1] = (a + b) / 2  # half the samples on each of two points of the same object
    sid[1, 1, 0] = ((a * a).sum() + (b * b).sum()) / 2
    sid[2, 2, 1:] = (SHEET + BLOCK) / 2, (SHEET ** 2 + BLOCK ** 2) / 2  # two objects
    # a ray through the clear part of a card in front: the card's values plus the sheet's (a negative spread)
    g[3, 0] = a + b
    sid[3, 0] = (a * a).sum() + (b * b).sum(), SHEET + BLOCK, SHEET ** 2 + BLOCK ** 2
    s = S.Surf(g, sid)
    assert s.ids[1, 1] == -1 and s.ids[2, 2] == -1 and s.ids[3, 0] == -1 and s.ids[0, 0] >= 0 and s.ids[3, 3] >= 0


if __name__ == "__main__":
    test_match_against_truth()
    test_uv_survives_generated_drift()
    test_generated_alone_follows_drift()
    test_mixed_pixels_are_left_out()
    print("surface OK")


def test_settle_evens_offsets_at_one_depth():
    """settle(): per-pixel error on a map shrinks, a smooth map stays as it is, a near strip and the far surface
    behind it lend each other nothing, and pixels without a match neither change nor lend."""
    rng = np.random.default_rng(3)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    g = np.stack([xs, ys], -1)
    far = np.stack([3.0 + 0.02 * xs + 0.01 * ys, -2.0 + 0.015 * ys], -1)  # offsets that change slowly
    near = np.stack([np.full_like(xs, 11.0), np.full_like(xs, 6.0)], -1)
    strip = (xs >= 70) & (xs < 90)
    depth = np.where(strip, 2.0, 5.0).astype(np.float32)
    clean = (g + np.where(strip[..., None], near, far)).astype(np.float32)
    valid = np.ones((H, W), np.float32)
    hole = (ys >= 40) & (ys < 50) & (xs >= 20) & (xs < 30)
    valid[hole] = 0.0
    noisy = (clean + rng.normal(0, 0.05, clean.shape)).astype(np.float32)
    noisy[hole] = g[hole] + 40.0  # whatever a pixel without a match holds
    out = S.settle(noisy, valid, depth)
    ok = valid > 0
    err_in, err_out = np.abs(noisy - clean)[ok], np.abs(out - clean)[ok]
    assert err_out.mean() < 0.4 * err_in.mean(), (err_in.mean(), err_out.mean())
    # one neighbour from the other depth would add 8 px / 25 or more, one from the hole 40 px / 25
    assert err_out.max() < 0.15, err_out.max()
    assert np.array_equal(out[hole], noisy[hole])
    still = S.settle(clean, valid, depth)
    plain = (xs >= 100) & (xs < 150) & (ys >= 10) & (ys < 110)  # far surface, away from the strip, the hole and the frame's edge
    assert np.abs(still - clean)[plain | ((xs >= 75) & (xs < 85))].max() < 2e-3
    # on a window of the frame: the same values as on the whole frame, away from the window's own edge
    x0, y0, x1, y1 = 30, 10, 120, 60
    part = S.settle(noisy[y0:y1, x0:x1], valid[y0:y1, x0:x1], depth[y0:y1, x0:x1], (x0, y0, x1, y1))
    r = S.SETTLE
    assert np.array_equal(part[r:-r, r:-r], out[y0 + r:y1 - r, x0 + r:x1 - r])
    assert S.settle(noisy, valid, depth, r=0) is noisy
