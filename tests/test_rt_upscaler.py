"""nts.rt (the GPU upscaler) against baselines/accumulate.py Accumulator, frame by frame, on a synthetic clip with
the awkward parts: a foreground moving over a panning background (dilation, disocclusion, off-screen history), render
noise (the clip at work), HDR colour, a reset in the middle, Halton jitter, all five scale factors.
  compat build: at least 99.9 % of values within 1e-3 (relative above 1) and 60 dB between the two
  fast build (half floats, hardware bilinear): 45 dB
Needs wgpu and a GPU adapter (a software one will do: NTS_RT_ADAPTER=llvmpipe); skipped without them, or failed
with NTS_RT_REQUIRE=1 (the CI job that runs them on Mesa's lavapipe).
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
    from nts.rt.upscaler import DEPTH_TEST, FITTED, RTUpscaler
    DEV, NAME = gpu.device()
except Exception as e:  # no wgpu, or no adapter
    DEV, NAME, WHY = None, None, f"{type(e).__name__}: {e}"
    DEPTH_TEST, FITTED = "match", {}

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


NP = dict(foot=0.2, wmax=12.0, expand=1.0, shrink=4.0, rel=0.05,  # hand-set constants, given to both sides,
          depth_test=DEPTH_TEST)  # and RTUpscaler's depth test (numpy's default may differ)


def _skip():
    if DEV is None:
        assert not os.environ.get("NTS_RT_REQUIRE"), f"NTS_RT_REQUIRE is set but there is no wgpu adapter ({WHY})"
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
    res = run(48, 40, 2.0, "compat", wmax=32.0, expand=1.5, foot=0.35, prior=0.3)
    assert min(r[0] for r in res) >= 0.999 and min(r[1] for r in res) >= 60, res


def test_depth_tests_match_numpy():
    _skip()  # both depth tests (nearest footprint depth; any footprint depth matching), every scale
    for depth_test in ("nearest", "match"):
        for w, h, s in CASES:
            res = run(w, h, s, "compat", depth_test=depth_test)
            assert min(r[0] for r in res) >= 0.999 and min(r[1] for r in res) >= 60, (depth_test, s, res)


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
            ref = GuidedAccumulator(s, (H, W), eps=0.03, reject=0.15, **NP, prior=0.03,
                                    history="catmull5")
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
            alb = Accumulator(s, (H, W), **dict(NP, wmax=32.0, foot=0.25, expand=2.0), prior=0.03, history="catmull5")
            ref = GuidedAccumulator(s, (H, W), eps=0.03, reject=None, **NP, prior=0.03,
                                    history="catmull5")
            up = SelfGuided(s, (H, W), build=build, dev=DEV, **NP, prior=0.03)
            assert up.on_gpu
            worst = float("inf")
            for fr in clip(w, h):
                # the clip's jitters put some output pixels exactly on a render pixel's sample (at x3, u = (X +
                # 0.5) / 3 - 0.5 - 1/6 is a whole number): numpy (float64) and the GPU (float32) then floor to
                # neighbouring pixels, an equally valid tie whose far tap weighs 4e-6 at foot 0.2 but 3e-4 at the
                # albedo's 0.25. Off the tie, they agree.
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
    c = dict(FITTED, history="catmull5", depth_test=DEPTH_TEST)
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
    try:
        RTUpscaler(2.0, (80, 96), dev=DEV, expand=0.9)
    except AssertionError:
        return
    raise AssertionError("expand < 1 was accepted")


def test_engine_conventions():
    """dispatch() on the engine's own textures, given device depth (reversed infinite, standard finite, reversed
    orthographic), motion in UV / NDC / render px pointing either way, and the jitter as a projection offset: the
    same frames as Accumulator on linear depth and render-px motion toward the previous frame."""
    _skip()
    from nts.rt.upscaler import depth_device, depth_ortho, motion_scale, offset_from_ndc
    near, far = 0.5, 40.0
    conv = [(depth_device(near), lambda z: near / z, "uv", "previous"),
            (depth_device(near, far, reversed=False), lambda z: (1 / near - 1 / z) / (1 / near - 1 / far), "ndc",
             "current"),
            (depth_ortho(near, far, reversed=True), lambda z: (far - z) / (far - near), "px", "current")]
    U = wgpu.TextureUsage
    for (w, h, s), (dmode, to_device, units, toward) in zip([(48, 40, 2.0), (42, 35, 12 / 7), (32, 24, 3.0)], conv):
        H, W = round(h * s), round(w * s)
        ms = motion_scale(units, toward, (h, w))
        ref = Accumulator(s, (H, W), **dict(NP, prior=0.1, history="catmull5"))
        up = RTUpscaler(s, (H, W), build="compat", dev=DEV, depth=dmode, mv_scale=ms,
                        **dict(NP, prior=0.1, history="catmull5"))
        tex = {k: DEV.create_texture(size=(w, h, 1), format=f, usage=U.TEXTURE_BINDING | U.COPY_DST)
               for k, f in (("color", "rgba32float"), ("depth", "r32float"), ("mv", "rg32float"))}
        worst = []
        for fr in clip(w, h):
            c = np.concatenate([fr.rgb, np.ones((h, w, 1), np.float32)], -1)
            d = to_device(fr.depth.astype(np.float64)).astype(np.float32)
            m = (fr.mv / np.array(ms, np.float32)).astype(np.float32)
            for k, a in (("color", c), ("depth", d), ("mv", m)):
                a = np.ascontiguousarray(a)
                DEV.queue.write_texture({"texture": tex[k]}, a, {"bytes_per_row": a.strides[0]}, (w, h, 1))
            ox, oy = cv.sample_offset(*fr.jitter)
            off = offset_from_ndc(-2 * ox / w, 2 * oy / h, (h, w))  # as an engine's projection jitter
            enc = DEV.create_command_encoder()
            up.dispatch(enc, tex["color"], tex["depth"], tex["mv"], off, reset=fr.reset)
            DEV.queue.submit([enc.finish()])
            g, r = up.read(), ref.step(fr)
            tol = 1e-3 * np.maximum(1.0, np.abs(r))
            worst.append((float(np.mean(np.abs(g - r) <= tol)), psnr(g, r)))
        frac, db = min(x[0] for x in worst), min(x[1] for x in worst)
        print(f"engine x{s:.3f} {units}/{toward} depth mode {dmode[0]}: {100 * frac:.3f} % within tol, {db:.1f} dB")
        assert frac >= 0.999 and db >= 60, (s, worst)
        assert up.output is up.hist[(up.t - 1) % 2]


def test_scene_depth_buffer():
    """The raster test scene drawn as a game draws it, then upscaled two ways: dispatch() binding its reversed-Z
    depth32float depth buffer, UV motion and colour target as they are, and step() given the numpy side's linear
    depth and render-px motion read back from the same frame. Also checks the scene's motion points to the previous
    frame: warping the previous frame by it beats no warp and the opposite warp."""
    _skip()
    from nts import warp as wp
    from nts.rt.scene import NEAR, Renderer
    from nts.rt.upscaler import depth_device, motion_scale
    r = Renderer(DEV)
    h, w, s = 45, 80, 2.0
    kw = dict(NP, prior=0.1, history="catmull5")
    a = RTUpscaler(s, (90, 160), build="compat", dev=DEV, **kw)
    b = RTUpscaler(s, (90, 160), build="compat", dev=DEV, depth=depth_device(NEAR),
                   mv_scale=motion_scale("uv", "previous", (h, w)), **kw)
    jit = cv.jitter_seq(10, 32)
    prev, errs = None, []
    for i in range(10):
        off = cv.sample_offset(*jit[i])
        rt, x = r.render("strafe", i, (h, w), off, lod=1 / s)
        ga = a.step(Frame(f=i, rgb=x["rgb"], depth=x["depth"], mv=x["mv"], jitter=jit[i], reset=i == 0))
        enc = DEV.create_command_encoder()
        b.dispatch(enc, rt["color"], rt["depth"], rt["motion"], off, reset=i == 0)
        DEV.queue.submit([enc.finish()])
        gb = b.read()
        tol = 1e-3 * np.maximum(1.0, np.abs(ga))
        assert np.mean(np.abs(gb - ga) <= tol) >= 0.999 and psnr(gb, ga) >= 60, (i, psnr(gb, ga))
        if prev is not None:
            errs.append([float(np.abs(wp.warp(prev, x["mv"] * k) - x["rgb"]).mean()) for k in (1, 0, -1)])
        prev = x["rgb"]
    e = np.mean(errs, 0)
    print(f"scene: depth buffer path matches; warp error along motion {e[0]:.4f}, none {e[1]:.4f}, reversed {e[2]:.4f}")
    assert e[0] < min(e[1], e[2]), e


def test_broken_inputs_recover():
    """NaN, inf and huge values from the engine (colour, depth and motion, frames 3-4) never reach the output, and two
    frames after the inputs are clean again the frames match a clean run (fast and compat builds)."""
    _skip()
    w, h, s = 48, 40, 2.0
    for build in ("compat", "fast"):
        clean = RTUpscaler(s, (80, 96), build=build, dev=DEV)
        hurt = RTUpscaler(s, (80, 96), build=build, dev=DEV)
        rng = np.random.default_rng(3)
        for t, fr in enumerate(clip(w, h, reset_at=99)):
            a = clean.step(fr)
            if t in (3, 4):
                rgb, depth, mv = fr.rgb.copy(), fr.depth.copy(), fr.mv.copy()
                for arr, vals in ((rgb, (np.nan, np.inf, -np.inf, 1e30)), (depth, (np.nan, np.inf, 0.0, -1.0)),
                                  (mv, (np.nan, np.inf, 1e30))):
                    idx = rng.integers(0, [h, w], (40, 2))
                    arr[idx[:, 0], idx[:, 1]] = rng.choice(vals, 40)[:, None] if arr.ndim == 3 else rng.choice(vals, 40)
                fr = Frame(f=fr.f, rgb=rgb, depth=depth, mv=mv, jitter=fr.jitter, reset=fr.reset)
            b = hurt.step(fr)
            assert np.isfinite(b).all(), (build, t)
            if t >= 7:
                assert psnr(b, a) >= 45, (build, t, psnr(b, a))
        print(f"broken inputs, {build}: always finite; frame {t}: {psnr(b, a):.1f} dB from the clean run")


def test_uneven_render_size():
    """A rounded render size (scale 47 / 31 across, 36 / 24 down) on a still, smooth image: the output converges to
    the image at the output pixel centres, as well at the far edge as at the near one (one scale for both axes put
    the far edge's samples up to half a pixel off)."""
    _skip()
    w, h, W, H = 31, 24, 47, 36
    sx, sy = W / w, H / h

    def img(x, y):  # render-px coordinates
        return np.stack([0.5 + 0.4 * np.sin(0.7 * x + 0.2 * y), 0.5 + 0.3 * np.cos(0.5 * y - 0.3 * x),
                         0.4 + 0.3 * np.sin(0.4 * x) * np.cos(0.6 * y)], -1).astype(np.float32)

    up = RTUpscaler(1.5, (H, W), build="compat", dev=DEV, in_hw=(h, w), wmax=64.0)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    jit = cv.jitter_seq(48, 32)
    for t in range(48):
        ox, oy = cv.sample_offset(*jit[t])
        fr = Frame(f=t, rgb=img(xs + 0.5 + ox, ys + 0.5 + oy), depth=np.full((h, w), 5.0, np.float32),
                   mv=np.zeros((h, w, 2), np.float32), jitter=jit[t], reset=t == 0)
        out = up.step(fr)
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    gt = img((X + 0.5) / sx, (Y + 0.5) / sy)
    near, far = psnr(out[:, 2:W // 3], gt[:, 2:W // 3]), psnr(out[:, -W // 3:-2], gt[:, -W // 3:-2])
    print(f"uneven size: near third {near:.1f} dB, far third {far:.1f} dB")
    assert near >= 30 and far >= near - 1.5, (near, far)


def test_export_interface(tmp_path):
    """nts/rt/export.py writes the shaders and interface.json; a host built from that file alone (export.Host) gives
    RTUpscaler's frames, for both builds, with a reversed-Z depth buffer and UV motion."""
    _skip()
    from nts.rt import export
    from nts.rt.upscaler import depth_device, motion_scale
    w, h, s = 48, 40, 2.0
    H, W = round(h * s), round(w * s)
    U = wgpu.TextureUsage
    dmode, ms = depth_device(0.5), motion_scale("uv", "previous", (h, w))
    for build, cfmt, mfmt in (("compat", "rgba32float", "rg32float"), ("fast", "rgba16float", "rg16float")):
        d = tmp_path / build
        export.write(str(d), build)
        host = export.Host(str(d), DEV, (H, W), (h, w), depth=dmode, mv_scale=ms)
        up = RTUpscaler(s, (H, W), build=build, dev=DEV, depth=dmode, mv_scale=ms)
        tex = {k: DEV.create_texture(size=(w, h, 1), format=f, usage=U.TEXTURE_BINDING | U.COPY_DST)
               for k, f in (("color", cfmt), ("depth", "r32float"), ("mv", mfmt))}
        npt = {"rgba32float": np.float32, "rg32float": np.float32, "rgba16float": np.float16, "rg16float": np.float16}
        for fr in clip(w, h):
            c = np.concatenate([fr.rgb, np.ones((h, w, 1), np.float32)], -1).astype(npt[cfmt])
            dd = (0.5 / fr.depth).astype(np.float32)
            m = (fr.mv / np.array(ms, np.float32)).astype(npt[mfmt])
            for k, a in (("color", c), ("depth", dd), ("mv", m)):
                a = np.ascontiguousarray(a)
                DEV.queue.write_texture({"texture": tex[k]}, a, {"bytes_per_row": a.strides[0]}, (w, h, 1))
            off = cv.sample_offset(*fr.jitter)
            for runner in ("host", "up"):
                enc = DEV.create_command_encoder()
                if runner == "host":
                    t_host = host.frame(enc, tex["color"], tex["depth"], tex["mv"], off, reset=fr.reset)
                else:
                    up.dispatch(enc, tex["color"], tex["depth"], tex["mv"], off, reset=fr.reset)
                DEV.queue.submit([enc.finish()])
            ft = np.float32 if "32" in _hist_format(build) else np.float16
            raw = DEV.queue.read_texture({"texture": t_host}, {"bytes_per_row": W * 4 * np.dtype(ft).itemsize},
                                         (W, H, 1))
            g = np.frombuffer(raw, ft).reshape(H, W, 4)[..., :3].astype(np.float32)
            r = up.read()
            assert np.array_equal(g, r), (build, fr.f, float(np.abs(g - r).max()))
        print(f"export {build}: host from interface.json = RTUpscaler, bit for bit")


def _hist_format(build):
    from nts.rt.upscaler import BUILDS
    return BUILDS[build]["hist"]


def _drs_frame(t, hw, out_hw, jit):
    """A frame of a scene defined in display pixels (a panning background, a nearer block moving faster), rendered
    at render size hw with this frame's jitter: rgb, depth, and motion in render px of this size."""
    (h, w), (H, W) = hw, out_hw
    sx, sy = W / w, H / h
    ox, oy = cv.sample_offset(*jit)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = (xs + 0.5 + ox) * sx, (ys + 0.5 + oy) * sy
    return _drs_image(X, Y, t), sx, sy


def _drs_image(X, Y, t):
    bx, by = X - 0.6 * t, Y - 0.2 * t
    fx, fy = X - (18 + 1.5 * t), Y - (22 + 0.5 * t)
    fg = (np.abs(fx) < 9) & (np.abs(fy) < 7)
    bg = np.stack([0.5 + 0.4 * np.sin(0.35 * bx + 0.1 * by), 0.4 + 0.3 * np.cos(0.3 * by),
                   0.3 + 0.25 * np.sin(0.25 * bx) * np.cos(0.4 * by)], -1)
    fgc = np.stack([2.0 + 0.8 * np.sin(0.6 * fx), 1.0 + 0.5 * np.cos(0.5 * fy), 3.0 + np.sin(0.3 * (fx + fy))], -1)
    rgb = np.where(fg[..., None], fgc, bg).astype(np.float32)
    depth = np.where(fg, 2.0, 5.0).astype(np.float32)
    v = np.where(fg[..., None], np.array([1.5, 0.5]), np.array([0.6, 0.2]))  # display px per frame
    return rgb, depth, v


def test_dynamic_resolution():
    """One upscaler, the render size changing every 6 frames (x2, x1.48, x1.67 of a 60 x 80 display) while things
    move: inputs in max-size textures, the frame in their top-left. A constant size gives the fixed-size upscaler's
    frames bit for bit; changing sizes stay finite and within 1 dB of a fixed-size run at the smallest size used."""
    _skip()
    H, W, n = 60, 80, 36
    sizes = [(30, 40), (40, 54), (36, 48)]
    mh, mw = 40, 54
    jit = cv.jitter_seq(n, 32)
    U = wgpu.TextureUsage
    tex = {k: DEV.create_texture(size=(mw, mh, 1), format=f, usage=U.TEXTURE_BINDING | U.COPY_DST)
           for k, f in (("color", "rgba32float"), ("depth", "r32float"), ("mv", "rg32float"))}
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)

    def run(size_of, max_hw):
        up = RTUpscaler(W / size_of(0)[1], (H, W), build="compat", dev=DEV, in_hw=size_of(0), max_in_hw=max_hw)
        outs, scores = [], []
        for t in range(n):
            hw = size_of(t)
            (rgb, depth, v), sx, sy = _drs_frame(t, hw, (H, W), jit[t])
            mv = (-v / np.array([sx, sy])).astype(np.float32)  # render px of this frame, toward the previous one
            for k, a in (("color", np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], -1)),
                         ("depth", depth), ("mv", mv)):
                a = np.ascontiguousarray(a)
                DEV.queue.write_texture({"texture": tex[k]}, a, {"bytes_per_row": a.strides[0]}, (hw[1], hw[0], 1))
            enc = DEV.create_command_encoder()
            up.dispatch(enc, tex["color"], tex["depth"], tex["mv"], cv.sample_offset(*jit[t]), reset=t == 0,
                        render_hw=hw)
            DEV.queue.submit([enc.finish()])
            o = up.read()
            assert np.isfinite(o).all(), t
            outs.append(o)
            if t >= 12:
                scores.append(psnr(o, _drs_image(X + 0.5, Y + 0.5, t)[0]))
        up.release()
        return outs, float(np.mean(scores))

    fixed = {hw: run(lambda t, hw=hw: hw, hw) for hw in sizes}
    same, _ = run(lambda t: sizes[0], (mh, mw))  # a constant size inside larger textures
    for a, b in zip(same, fixed[sizes[0]][0]):
        assert np.array_equal(a, b)
    _, drs = run(lambda t: sizes[(t // 6) % 3], (mh, mw))
    low = min(v[1] for v in fixed.values())
    print("dynamic resolution: fixed " + ", ".join(f"{hw[1]}x{hw[0]} {v[1]:.2f}" for hw, v in fixed.items()) +
          f" dB; changing every 6 frames {drs:.2f} dB")
    assert drs >= low - 1.0, (drs, low)


def test_no_negative_colour():
    """Non-negative input never comes out negative: the widened box stops at zero (Catmull-Rom history overshoots
    at the HDR block's edges; before, -0.05 got through on frame 4 of this sequence)."""
    _skip()
    from nts.rt.webcheck import IN_HW, OUT_HW, scene
    jit = cv.jitter_seq(12, cv.jitter_phases(IN_HW[0]))
    for build in ("compat", "fast"):
        up = RTUpscaler(OUT_HW[1] / IN_HW[1], OUT_HW, build=build, dev=DEV, in_hw=IN_HW)
        low = []
        for t in range(12):
            off = cv.sample_offset(*jit[t])
            rgb, depth, mv = scene(t, IN_HW, OUT_HW, off)
            o = up.step(Frame(f=t, rgb=rgb, depth=depth, mv=mv, jitter=jit[t], reset=t == 0), read=True)
            low.append(float(o.min()))
        up.release()
        assert min(low) >= 0.0, (build, low)


if __name__ == "__main__":  # without pytest (e.g. Blender's Python): every test, in order
    import inspect
    import pathlib
    import tempfile
    print("adapter:", NAME or WHY)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        kw = {"tmp_path": pathlib.Path(tempfile.mkdtemp())} if "tmp_path" in inspect.signature(f).parameters else {}
        f(**kw)
        print("passed", n, flush=True)
    print(f"ok: {len(tests)} tests")
