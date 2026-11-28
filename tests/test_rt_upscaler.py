"""nts.rt (the GPU upscaler) against baselines/accumulate.py Accumulator, frame by frame, on a synthetic clip with
the awkward parts: a foreground moving over a panning background (dilation, disocclusion, off-screen history), render
noise (the clip at work), HDR colour, a reset in the middle, Halton jitter, all five scale factors.
  compat build: at least 99.9 % of values within 1e-3 (relative above 1) and 60 dB between the two
  fast build (half floats, hardware bilinear): 45 dB
Needs wgpu and a GPU adapter; skipped without them (CI).
  python tests/test_rt_upscaler.py     or     pytest tests/test_rt_upscaler.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from baselines.accumulate import Accumulator  # noqa: E402
from nts import conventions as cv  # noqa: E402
from nts.io_exr import Frame  # noqa: E402

try:
    import wgpu  # noqa: F401
    from nts.rt import gpu
    from nts.rt.upscaler import RTUpscaler
    DEV, NAME = gpu.device()
except Exception as e:  # no wgpu, or no adapter
    DEV, NAME, WHY = None, None, f"{type(e).__name__}: {e}"

# (render w, h, scale): output = render * scale, exactly
CASES = [(48, 40, 1.0), (48, 40, 1.5), (42, 35, 12 / 7), (48, 40, 2.0), (32, 24, 3.0)]
V_FG, V_BG = np.array([1.3, -0.7]), np.array([0.25, 0.1])  # render px per frame


def clip(w, h, n=12, reset_at=7, seed=0):
    """Synthetic jittered low-res frames: (rgb, depth, mv) sampled at each pixel's jittered sample position."""
    rng = np.random.default_rng(seed)
    jit = cv.jitter_seq(n, 32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    out = []
    for t in range(n):
        ox, oy = cv.sample_offset(*jit[t])
        x, y = xs + 0.5 + ox, ys + 0.5 + oy
        bx, by = x - V_BG[0] * t, y - V_BG[1] * t       # background texture coordinates (it pans)
        fx, fy = x - (10 + V_FG[0] * t), y - (12 + V_FG[1] * t)
        fg = (np.abs(fx - 6) < 6) & (np.abs(fy - 5) < 5)
        bg_rgb = np.stack([0.5 + 0.4 * np.sin(0.9 * bx + 0.3 * by), 0.4 + 0.3 * np.cos(0.6 * by),
                           0.3 + 0.25 * np.sin(0.5 * bx) * np.cos(0.8 * by)], -1)
        fg_rgb = np.stack([2.5 + 1.5 * np.sin(1.3 * fx), 1.2 + 0.8 * np.cos(1.1 * fy), 6.0 + 2.0 * np.sin(fx + fy)], -1)
        rgb = np.where(fg[..., None], fg_rgb, bg_rgb) + rng.normal(0, 0.03, (h, w, 3))
        depth = np.where(fg, 2.0 + 0.01 * fx, 5.0 + 0.002 * by)
        mv = np.where(fg[..., None], -V_FG, -V_BG)  # image space, current -> previous
        out.append(Frame(f=t, rgb=rgb.astype(np.float32), depth=depth.astype(np.float32),
                         mv=mv.astype(np.float32), jitter=jit[t], reset=(t == 0 or t == reset_at)))
    return out


def psnr(a, ref):
    mse = float(np.mean((a.astype(np.float64) - ref) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(float(ref.max()) ** 2 / mse)


def run(w, h, scale, build, entry="compute", expand=1.0, prior=0.1, history="bilinear", **kw):
    H, W = round(h * scale), round(w * scale)
    for n, m in ((H, h), (W, w)):  # the GPU maps output -> render pixel as (X * w) // W: numpy's float floor must agree
        assert (np.minimum((np.arange(n) / scale).astype(np.int64), m - 1) == (np.arange(n) * m) // n).all()
    ref = Accumulator(scale, (H, W), **dict(NP, **kw, expand=expand, prior=prior, history=history))
    up = RTUpscaler(scale, (H, W), build=build, dev=DEV, entry=entry,
                    **dict(NP, **kw, expand=expand, prior=prior, history=history))
    worst = []
    for fr in clip(w, h):
        r = ref.step(fr)
        g = up.step(fr)
        assert g.shape == r.shape
        tol = 1e-3 * np.maximum(1.0, np.abs(r))
        worst.append((float(np.mean(np.abs(g - r) <= tol)), psnr(g, r), float(np.abs(g - r).max())))
    return worst


NP = dict(sigma=0.3, wmax=12.0, expand=1.0, shrink=4.0, rel=0.05)  # hand-set constants, given to both sides


def _skip():
    if DEV is None:
        import pytest
        pytest.skip(f"no wgpu adapter ({WHY})")


def test_compat_matches_accumulator():
    _skip()
    for w, h, s in CASES:
        res = run(w, h, s, "compat")
        frac, db = min(r[0] for r in res), min(r[1] for r in res)
        print(f"compat x{s:.3f}: worst frame {100 * frac:.3f} % within tol, {db:.1f} dB, "
              f"max abs {max(r[2] for r in res):.2e}")
        assert frac >= 0.999 and db >= 60, (s, res)


def test_compat_other_constants():
    _skip()  # wider footprint, widened box, longer history, heavier prior
    res = run(48, 40, 2.0, "compat", wmax=32.0, expand=1.5, sigma=0.7, prior=0.3)
    assert min(r[0] for r in res) >= 0.999 and min(r[1] for r in res) >= 60, res


def test_fragment_entry_matches_compute():
    _skip()  # the resolve pass as a full-screen fragment shader: same arithmetic, so the same bar as compute
    for w, h, s in CASES:
        res = run(w, h, s, "compat", entry="fragment")
        assert min(r[0] for r in res) >= 0.999 and min(r[1] for r in res) >= 60, (s, res)
    res = run(48, 40, 2.0, "fast", entry="fragment")
    assert min(r[1] for r in res) >= 45, res


def test_fast_close_to_accumulator():
    _skip()
    for w, h, s in CASES:
        res = run(w, h, s, "fast")
        db = min(r[1] for r in res)
        print(f"fast x{s:.3f}: worst frame {db:.1f} dB")
        assert db >= 45, (s, res)


def test_catmull_history():
    _skip()  # Catmull-Rom colour history: compat exact against numpy, fast (3x3 filtered lookups) close
    for hist in ("catmull", "catmull5"):
        for w, h, s in CASES:
            res = run(w, h, s, "compat", history=hist)
            assert min(r[0] for r in res) >= 0.999 and min(r[1] for r in res) >= 60, (hist, s, res)
            res = run(w, h, s, "fast", history=hist)
            print(f"{hist} fast x{s:.3f}: worst frame {min(r[1] for r in res):.1f} dB")
            assert min(r[1] for r in res) >= 45, (hist, s, res)


def test_ycocg_centre_clip():
    _skip()  # the box in YCoCg, and clipping toward its middle, each alone and together
    for opts in (dict(space="ycocg"), dict(clip="centre"), dict(space="ycocg", clip="centre", expand=1.25)):
        for w, h, s in CASES[1::2]:
            res = run(w, h, s, "compat", **opts)
            assert min(r[0] for r in res) >= 0.999 and min(r[1] for r in res) >= 60, (opts, s, res)
            res = run(w, h, s, "fast", **opts)
            assert min(r[1] for r in res) >= 45, (opts, s, res)


def test_static_expand():
    _skip()  # a wider box where the pixel did not move: with the background held still, most of the frame
    global V_BG
    keep, V_BG = V_BG, np.zeros(2)
    try:
        for w, h, s in CASES[::2]:
            res = run(w, h, s, "compat", expand=1.1, static_expand=1.6)
            assert min(r[0] for r in res) >= 0.999 and min(r[1] for r in res) >= 60, (s, res)
        H, W = round(40 * 2.0), round(48 * 2.0)  # the factor must change the result, or the test checks nothing
        a = Accumulator(2.0, (H, W), **dict(NP, expand=1.1), prior=0.1, static_expand=1.6)
        b = Accumulator(2.0, (H, W), **dict(NP, expand=1.1), prior=0.1)
        diff = max(float(np.abs(a.step(fr) - b.step(fr)).max()) for fr in clip(48, 40))
        assert diff > 1e-3, diff
    finally:
        V_BG = keep


def test_half_builds():
    _skip()  # float32 history with the texture unit's filtering (fast_hist32); half floats with numpy's (half)
    for b in ("fast_hist32", "half"):
        for w, h, s in CASES:
            res = run(w, h, s, b, history="catmull5")
            assert min(r[1] for r in res) >= 45, (b, s, res)


def albedo(depth, scale=1.0):
    """A synthetic albedo: brighter on the near box than on the background, with a fine stripe texture."""
    h, w = depth.shape
    x = (np.arange(w)[None, :] + 0.5) / scale
    base = np.where(depth < 3, 0.8, 0.3)[..., None] * np.array([1.0, 0.8, 0.6], np.float32)
    return (base * (1 + 0.2 * np.sin(x * 2.1))[..., None]).astype(np.float32)


def test_guided_matches_numpy():
    _skip()  # guided mode against baselines/guided.py GuidedAccumulator: shading accumulated, albedo change rejection
    from baselines.guided import GuidedAccumulator
    import nts.warp as wp
    for w, h, s in CASES[1:]:
        H, W = round(h * s), round(w * s)
        for build, bar in (("compat", 60), ("fast", 45)):
            ref = GuidedAccumulator(s, (H, W), eps=0.03, reject=0.15, **NP, prior=0.03, history="catmull5")
            up = RTUpscaler(s, (H, W), build=build, dev=DEV, **NP, prior=0.03, guided=True, eps=0.03, reject=0.15)
            worst = float("inf")
            for fr in clip(w, h):
                a_lr = albedo(fr.depth)
                a_hr = albedo(wp.upsample_nearest(fr.depth, s, (H, W)), s)
                r = ref.step(fr, a_lr, a_hr)
                g = up.step(fr, a_lr, a_hr)
                if build == "compat":
                    tol = 1e-3 * np.maximum(1.0, np.abs(r))
                    assert np.mean(np.abs(g - r) <= tol) >= 0.999, (s, build)
                worst = min(worst, psnr(g, r))
            print(f"guided {build} x{s:.3f}: worst frame {worst:.1f} dB")
            assert worst >= bar, (s, build, worst)


def test_self_guided_matches_numpy():
    _skip()  # self-guided mode: an albedo accumulator feeding the guided one, the albedo history copied on the GPU
    from baselines.guided import GuidedAccumulator
    from nts.rt.upscaler import SelfGuided
    for w, h, s in CASES[1:]:
        H, W = round(h * s), round(w * s)
        for build, bar in (("compat", 60), ("fast", 40)):
            alb = Accumulator(s, (H, W), **dict(NP, wmax=32.0, sigma=0.5, expand=2.0), prior=0.03, history="catmull5")
            ref = GuidedAccumulator(s, (H, W), eps=0.03, reject=None, **NP, prior=0.03, history="catmull5")
            up = SelfGuided(s, (H, W), build=build, dev=DEV, **NP, prior=0.03)
            assert up.on_gpu
            worst = float("inf")
            for fr in clip(w, h):
                # the clip's jitters put some samples exactly on a pixel boundary (at x3, (0 + 0.5 + 1/6) * 3 = 2):
                # numpy (float64) and the GPU (float32) then floor to neighbouring pixels, an equally valid tie whose
                # far tap weighs 4e-6 at sigma 0.3 but 0.011 at the albedo's 0.5. Off the tie, they agree.
                fr.jitter = (fr.jitter[0] + 1e-3, fr.jitter[1] + 1e-3)
                fr.albedo = albedo(fr.depth)
                rgb = fr.rgb
                fr.rgb = fr.albedo
                a_hr = alb.step(fr)
                fr.rgb = rgb
                r = ref.step(fr, fr.albedo, a_hr)
                g = up.step(fr)
                if build == "compat":
                    tol = 1e-3 * np.maximum(1.0, np.abs(r))
                    assert np.mean(np.abs(g - r) <= tol) >= 0.999, (s, build)
                worst = min(worst, psnr(g, r))
            up.release()
            print(f"self-guided {build} x{s:.3f}: worst frame {worst:.1f} dB")
            assert worst >= bar, (s, build, worst)


def test_fitted_constants():
    _skip()  # the constants nts/rt/fit.py chose, weight shrink and depth tolerance included, against numpy
    c = dict(sigma=0.586, expand=1.32, wmax=16.8, shrink=11.0, prior=0.0117, rel=0.169, history="catmull5")
    for w, h, s in CASES[1:]:
        H, W = round(h * s), round(w * s)
        for build, bar in (("compat", 60), ("fast", 45)):
            ref = Accumulator(s, (H, W), **c)
            up = RTUpscaler(s, (H, W), build=build, dev=DEV, **c)
            worst = float("inf")
            for fr in clip(w, h):
                fr.jitter = (fr.jitter[0] + 1e-3, fr.jitter[1] + 1e-3)  # off the exact ties (test_self_guided)
                r, g = ref.step(fr), up.step(fr)
                if build == "compat":
                    tol = 1e-3 * np.maximum(1.0, np.abs(r))
                    assert np.mean(np.abs(g - r) <= tol) >= 0.999, (s, build)
                worst = min(worst, psnr(g, r))
            up.release()
            print(f"fitted {build} x{s:.3f}: worst frame {worst:.1f} dB")
            assert worst >= bar, (s, build, worst)


def test_box_never_shrinks():
    _skip()  # a box narrower than the samples' own min/max is refused
    import pytest
    with pytest.raises(AssertionError):
        RTUpscaler(2.0, (80, 96), dev=DEV, expand=0.9)


if __name__ == "__main__":
    print("adapter:", NAME or WHY)
    test_compat_matches_accumulator()
    test_compat_other_constants()
    test_fragment_entry_matches_compute()
    test_fast_close_to_accumulator()
    test_catmull_history()
    test_ycocg_centre_clip()
    test_guided_matches_numpy()
    test_self_guided_matches_numpy()
    test_fitted_constants()
    print("ok")
