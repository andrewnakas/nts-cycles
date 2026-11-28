"""RTUpscaler: baselines/accumulate.py Accumulator on the GPU, in two compute passes per frame.

  pass A, render size (shaders/prep.wgsl):     motion of the nearest surface + depth -> aux; 3x3 colour min and max
  pass B, output size (shaders/resolve.wgsl):  this frame's samples gathered from the render (nts/gather.py),
                                               reprojected history, depth test, clip to the
                                               min/max box, one weighted sum with the bilinear frame as a fixed-weight
                                               prior -> history (rgb = the frame, a = weight)
aux and history are ping-ponged: frame t writes slot t % 2 and reads the other one.

Builds:
  compat  float32 everywhere and numpy's own bilinear formula, so the output matches Accumulator
          (tests/test_rt_upscaler.py)
  fast    half-float buffers and the texture unit's bilinear filter (its sub-texel weights are quantised)

Matches baselines/accumulate.py Accumulator: the clip box is the plain min/max of the 3x3 render
neighbourhood (Karis 2014; widened by expand >= 1, never shrunk) and the bilinear frame enters the blend as one more
fixed-weight term rather than through a ramp on the accumulated weight. Same step() as Accumulator: step(fr) takes an nts.io_exr.Frame (rgb, depth, mv, jitter,
reset) and returns the (H, W, 3) float32 frame; step(fr, read=False) leaves it on the GPU. Depth beyond 65000 scene
units is treated as 65000 (background).
"""
import math
import os
import struct
import warnings

import numpy as np
import wgpu

from .. import conventions as cv
from ..io_exr import Frame
from . import gpu

SHADERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
DEPTH_CLAMP = 65000.0
STRIDE = 256  # one uniform slot per frame (minUniformBufferOffsetAlignment)
PARAMS = struct.Struct("<4I4f4f4f4f4f4I4f4I")
EXP_MASK = 0x7f800000  # float exponent bits: the shaders' NaN / inf tests take it from the uniform (finite3)

FETCH_COMPAT = """
fn fetch(t: texture_2d<f32>, x: f32, y: f32, bx: f32, by: f32, size: vec2<f32>) -> vec4<f32> {
  // nts/warp.py _Bilinear: clip to [0, n - 1.001], then t00 gx gy + t01 fx gy + t10 gx fy + t11 fx fy, left to right
  let cx = min(max(x, 0.0), bx);
  let cy = min(max(y, 0.0), by);
  let i = vec2<i32>(i32(cx), i32(cy));
  let fx = cx - f32(i.x);
  let fy = cy - f32(i.y);
  let gx = 1.0 - fx;
  let gy = 1.0 - fy;
  var o = textureLoad(t, i, 0) * gx * gy;
  o += textureLoad(t, i + vec2<i32>(1, 0), 0) * fx * gy;
  o += textureLoad(t, i + vec2<i32>(0, 1), 0) * gx * fy;
  o += textureLoad(t, i + vec2<i32>(1, 1), 0) * fx * fy;
  return o;
}
"""

CR_WEIGHTS = """
fn cr_w(t: f32) -> vec4<f32> {
  // Catmull-Rom weights of the taps at -1, 0, 1, 2 for the fractional offset t (baselines/accumulate.py cr_weights)
  let t2 = t * t;
  let t3 = t * t * t;
  return vec4<f32>(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                   0.5 * t3 - 0.5 * t2);
}
"""

FETCH_COMPAT += CR_WEIGHTS + """
fn hist_cr(px: f32, py: f32, W: i32, H: i32) -> vec3<f32> {
  // baselines/accumulate.py warp_catmull: clamp to [0, n - 1], 4x4 taps with clamped indices, rows then columns;
  // CR5: the corner taps left out and the rest renormalised
  let x = min(max(px, 0.0), f32(W - 1));
  let y = min(max(py, 0.0), f32(H - 1));
  let x0 = floor(x);
  let y0 = floor(y);
  let wx = cr_w(x - x0);
  let wy = cr_w(y - y0);
  var o = vec3<f32>(0.0);
  for (var j = 0; j < 4; j++) {
    let yy = clamp(i32(y0) + j - 1, 0, H - 1);
    var r = vec3<f32>(0.0);
    for (var i = 0; i < 4; i++) {
      if (!CR5 || (i == 1 || i == 2 || j == 1 || j == 2)) {
        let xx = clamp(i32(x0) + i - 1, 0, W - 1);
        r += textureLoad(hist_prev, vec2<i32>(xx, yy), 0).rgb * wx[i];
      }
    }
    o += r * wy[j];
  }
  if (CR5) {
    o /= 1.0 - (wx.x + wx.w) * (wy.x + wy.w);
  }
  return o;
}
"""

FETCH_COMPAT += """
fn gather2(at: vec2<f32>, lsize: vec2<f32>) -> vec3<f32> {
  return vec3<f32>(0.0);  // fast build only
}
"""

FETCH_FAST = """
fn fetch(t: texture_2d<f32>, x: f32, y: f32, bx: f32, by: f32, size: vec2<f32>) -> vec4<f32> {
  let c = vec2<f32>(min(max(x, 0.0), bx), min(max(y, 0.0), by));
  return textureSampleLevel(t, lin, (c + 0.5) / size, 0.0);
}
"""

FETCH_FAST += """
fn gather2(at: vec2<f32>, lsize: vec2<f32>) -> vec3<f32> {
  // the render-side gather's 2 x 2 taps as one filtered lookup at their balance point (render px)
  return min(textureSampleLevel(color_lr, lin, (at + 0.5) / lsize, 0.0).rgb, vec3<f32>(65000.0));
}
"""

FETCH_FAST += CR_WEIGHTS + """
fn hist_cr(px: f32, py: f32, W: i32, H: i32) -> vec3<f32> {
  // the same 4x4 Catmull-Rom with the middle two taps of each axis merged into one bilinear lookup at their balance
  // point (their weights are both positive): 3x3 hardware-filtered lookups, edges clamped by the sampler
  let x = min(max(px, 0.0), f32(W - 1));
  let y = min(max(py, 0.0), f32(H - 1));
  let x0 = floor(x);
  let y0 = floor(y);
  let wx = cr_w(x - x0);
  let wy = cr_w(y - y0);
  let sx = wx.y + wx.z;
  let sy = wy.y + wy.z;
  let u = (vec3<f32>(x0 - 1.0, x0 + wx.z / sx, x0 + 2.0) + 0.5) / f32(W);
  let v = (vec3<f32>(y0 - 1.0, y0 + wy.z / sy, y0 + 2.0) + 0.5) / f32(H);
  let a = vec3<f32>(wx.x, sx, wx.w);
  let b = vec3<f32>(wy.x, sy, wy.w);
  if (CR5) {  // the middle lookup and its four side neighbours; the corners' share renormalised away
    var o = textureSampleLevel(hist_prev, lin, vec2<f32>(u.y, v.y), 0.0).rgb * (sx * sy);
    o += textureSampleLevel(hist_prev, lin, vec2<f32>(u.x, v.y), 0.0).rgb * (wx.x * sy);
    o += textureSampleLevel(hist_prev, lin, vec2<f32>(u.z, v.y), 0.0).rgb * (wx.w * sy);
    o += textureSampleLevel(hist_prev, lin, vec2<f32>(u.y, v.x), 0.0).rgb * (sx * wy.x);
    o += textureSampleLevel(hist_prev, lin, vec2<f32>(u.y, v.z), 0.0).rgb * (sx * wy.w);
    return o / (1.0 - (wx.x + wx.w) * (wy.x + wy.w));
  }
  var o = vec3<f32>(0.0);
  for (var j = 0; j < 3; j++) {
    var r = vec3<f32>(0.0);
    for (var i = 0; i < 3; i++) {
      r += textureSampleLevel(hist_prev, lin, vec2<f32>(u[i], v[j]), 0.0).rgb * a[i];
    }
    o += r * b[j];
  }
  return o;
}
"""

ENTRY_COMPUTE = """
@group(0) @binding(7) var hist_out: texture_storage_2d<FMT, write>;
@group(0) @binding(12) var out_img: texture_storage_2d<FMT, write>;  // guided mode: the remodulated frame

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let X = i32(gid.x);
  let Y = i32(gid.y);
  if (X >= i32(P.sizes.x) || Y >= i32(P.sizes.y)) {
    return;
  }
  let r = resolve_px(X, Y);
  textureStore(hist_out, vec2<i32>(X, Y), r);
  if (GUIDED) {  // the history holds shading; the frame is shading x (full-res albedo + eps)
    var o = max(r.rgb, vec3<f32>(0.0)) * (textureLoad(alb_hr, vec2<i32>(X, Y), 0).rgb + P.flt.y);
    if (FAST) {
      o = min(o, vec3<f32>(65000.0));
    }
    textureStore(out_img, vec2<i32>(X, Y), vec4<f32>(o, 1.0));
  }
}
"""

ENTRY_FRAGMENT = """
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  // one triangle covering the whole target
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  return resolve_px(i32(pos.x), i32(pos.y));
}
"""

# fast: half-float buffers (with the 65000 clamps and finite check they need); hw: the texture unit's filtering.
# fast_hist32 is the fast build with its history in float32: in half floats the history's slow update rounds a little
# differently every frame and the recursion keeps the error, which measured as 5 % more static flicker on our shots
# (motion, depth and the clip box in half floats measured the same as float32). It doubles the history's memory
# traffic: +0.15 to 0.26 ms at 1080p on an RTX 4050 laptop GPU. Filtering a float32 texture needs the device feature
# float32-filterable; without it the history falls back to half floats.
BUILDS = {
    "compat": dict(color="rgba32float", mv="rg32float", aux="rgba32float", stat="rgba32float", hist="rgba32float",
                   fast=False, hw=False),
    "fast": dict(color="rgba16float", mv="rg16float", aux="rgba16float", stat="rgba16float", hist="rgba16float",
                 fast=True, hw=True),
    "half": dict(color="rgba16float", mv="rg16float", aux="rgba16float", stat="rgba16float", hist="rgba16float",
                 fast=True, hw=False),  # half floats with numpy's own filtering: a diagnostic build
}
BUILDS["fast_hist32"] = dict(BUILDS["fast"], hist="rgba32float")


def np_type(fmt):
    return np.float32 if "32" in fmt else np.float16


def shader(name, **fields):
    src = open(os.path.join(SHADERS, name)).read()
    for k, v in fields.items():
        src = src.replace("{{" + k + "}}", v)
    assert "{{" not in src, f"unfilled field in {name}"
    return src


def sources(b, wg=((16, 16), (32, 4)), history="catmull5", space="rgb", clip="clamp", guided=False, entry="compute",
            depth_test="match"):
    """The final WGSL of both passes (prep, resolve) for a build (BUILDS[name]) and the design switches: what
    RTUpscaler compiles, and what nts/rt/export.py writes out for engines."""
    fast, hw, frag = b["fast"], b["hw"], entry == "fragment"
    (gx, gy), (bx, by) = wg
    yc, gd = str(space == "ycocg").lower(), str(guided).lower()
    sa = shader("prep.wgsl", F_AUX=b["aux"], F_STAT=b["stat"], FAST=str(fast).lower(), WG=f"{gx}, {gy}", YCOCG=yc,
                F_COLOR=b["color"], GUIDED=gd, GX=str(gx), GY=str(gy), TN=str((gx + 2) * (gy + 2)))
    sb = shader("resolve.wgsl", FAST=str(fast).lower(), CATMULL=str(history != "bilinear").lower(), YCOCG=yc,
                CR5=str(history == "catmull5").lower(), GUIDED=gd, CENTRE=str(clip == "centre").lower(),
                DMATCH=str(depth_test == "match").lower(),
                SAMPLER="@group(0) @binding(8) var lin: sampler;" if hw else "", HW=str(hw).lower(),
                FETCH=FETCH_FAST if hw else FETCH_COMPAT,
                ENTRY=ENTRY_FRAGMENT if frag else ENTRY_COMPUTE.replace("FMT", b["hist"]).replace("WG", f"{bx}, {by}"))
    return sa, sb


# Constants chosen by nts/rt/fit.py (2026-11-27): six fit shots, J = PSNR - 3 (ln tErr + ln flicker), Catmull-Rom
# (5-lookup) history, fast build. Held-out shots against the hand-set start: -0.07 dB, -9 % tErr, -14 % flicker.
# conventions.ACCUM, shared with the numpy accumulator. A refit with this frame's samples gathered from the render
# side and the matching depth test (2026-11-28, same shots) found no better point.
FITTED = dict(cv.ACCUM)
# A fit on the raster test scene (nts/rt/scene.py, 2026-11-28, Performance: foot 0.1875, expand 2.27, shrink 20) gained
# 0.3 to 0.5 dB at Performance and Ultra Performance there but lost 0.9 dB at Native AA: one set of constants is kept.
DEPTH_TEST = "match"  # RTUpscaler's default for depth_test=


# -- what an engine gives: its conventions, turned into the upscaler's -------------------------------------------------
# The upscaler works in linear view depth (larger = farther), motion in render pixels pointing from this frame to the
# previous one (x right, y down), and the sample offset: render pixel (i, j) samples the image point
# (i + 0.5 + ox, j + 0.5 + oy), in render pixels, x right, y down.

def depth_linear():
    """Depth already linear view depth (scene units: distance along the view axis)."""
    return (0, 0.0, 0.0)


def depth_device(near, far=math.inf, reversed=True):
    """A perspective depth buffer in [0, 1] (D3D, Vulkan, Metal, and OpenGL's window depth): reversed = 1 at the near
    plane, 0 at the far one; far may be math.inf (infinite far plane). Linear depth is 1 / (A d + B)."""
    inv_far = 0.0 if math.isinf(far) else 1.0 / far
    a = 1.0 / near - inv_far
    return (1, a, inv_far) if reversed else (1, -a, 1.0 / near)


def depth_ortho(near, far, reversed=False):
    """An orthographic depth buffer in [0, 1]: linear depth is A d + B."""
    return (2, near - far, far) if reversed else (2, far - near, near)


def motion_scale(units, toward, render_hw):
    """The per-axis factor taking an engine's motion vectors to render pixels from this frame to the previous one.
    units: "px" (render pixels, y down), "uv" (texture coordinates, y down, 1 = the whole image) or "ndc" (y up,
    2 = the whole image); toward: "previous" (the vector points from this frame's position to the previous one)
    or "current" (from the previous position to this one)."""
    h, w = render_hw
    sx, sy = {"px": (1.0, 1.0), "uv": (float(w), float(h)), "ndc": (w / 2.0, -h / 2.0)}[units]
    k = {"previous": 1.0, "current": -1.0}[toward]
    return (sx * k, sy * k)


def offset_from_ndc(jx, jy, render_hw):
    """The sample offset when the engine jitters by adding (jx, jy) to the projected x, y in NDC (clip.xy += (jx, jy)
    * clip.w; y up): the image moves by (jx w / 2, -jy h / 2) render px, so each pixel sees the point that far back."""
    h, w = render_hw
    return (-jx * w / 2.0, jy * h / 2.0)


class EngineInputs:
    """An engine's own textures, bound as they are (no copy): colour (float formats; filterable in the fast build),
    depth (a depth32float depth buffer or any float colour format), motion (rg float formats)."""

    def __init__(self, up, color, depth, mv, alb=None, alb_hr=None):
        assert alb_hr is None or tuple(alb_hr.size[:2]) == up.hw[::-1], (alb_hr.size, up.hw)
        self.color, self.depth, self.mv, self.alb, self.alb_hr = color, depth, mv, alb, alb_hr

    def covers(self, hw):
        """Render-size inputs at least the frame's size (dynamic resolution: the frame fills their top-left)."""
        h, w = hw
        return all(t is None or (t.size[0] >= w and t.size[1] >= h) for t in (self.color, self.depth, self.mv, self.alb))


class Inputs:
    """One frame's inputs at render size: colour (rgb, 1), depth, motion (render px, current -> previous)."""

    def __init__(self, up):
        h, w = up.in_hw
        use = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
        self.color = up.texture((w, h), up.b["color"], use)
        self.depth = up.texture((w, h), "r32float", use)
        self.mv = up.texture((w, h), up.b["mv"], use)
        if up.guided:  # albedo at render size and at display size
            H, W = up.hw
            self.alb = up.texture((w, h), up.b["color"], use)
            self.alb_hr = up.texture((W, H), up.b["color"], use)


class RTUpscaler:
    def __init__(self, scale, out_hw, wmax=FITTED["wmax"], expand=FITTED["expand"], mv_scale=(1.0, 1.0),
                 foot=FITTED["foot"], build="compat", rel=FITTED["rel"], prior=FITTED["prior"],
                 shrink=FITTED["shrink"], dev=None, slots=1, wg=((16, 16), (32, 4)), in_hw=None,
                 entry="compute", history="catmull5", space="rgb", clip="clamp", static_expand=None,
                 guided=False, eps=0.03, reject=0.15, depth=depth_linear(), max_in_hw=None, mv_sign=None,
                 depth_test=DEPTH_TEST):
        """foot: footprint of this frame's samples, a Gaussian's sigma in render px (each output pixel takes the 2 x 2
        render pixels around its own position in the render). depth_test: "match" keeps the history where any of the
        previous frame's depths under it matches (within rel), "nearest" compares the nearest of them only.
        wg: workgroup size of (prep, resolve), or one size for both. in_hw: render size when out / scale is not a
        whole number of pixels (timing runs); otherwise derived. entry: the resolve pass as a compute shader writing
        a storage texture, or as a full-screen triangle whose fragment shader writes a render target. history: the
        colour history read with bilinear or Catmull-Rom sampling (the weight is always bilinear). space: the box in
        RGB or YCoCg. clip: history clamped per channel, or moved along the line to the box's middle ("centre").
        static_expand: box widening where the pixel did not move (>= expand; None = expand). guided: the engine
        also gives the albedo at render and at display size (step(fr, alb_lr, alb_hr)): shading = colour / (albedo +
        eps) is accumulated, history weight is scaled by exp(-change / reject) where the display-size albedo warped
        from the last frame differs from this one (reject 0: off), and the frame is shading x (display albedo + eps),
        as baselines/guided.py GuidedAccumulator (compute entry only). mv_scale: per-axis factor on the motion
        vectors (motion_scale()); depth: depth_linear(), depth_device() or depth_ortho(). max_in_hw: the largest
        render size dispatch() will be given (dynamic resolution; default in_hw)."""
        self.scale, self.hw = float(scale), tuple(out_hw)
        H, W = self.hw
        if in_hw is None:
            in_hw = (round(H / self.scale), round(W / self.scale))
            assert round(in_hw[0] * self.scale) == H and round(in_hw[1] * self.scale) == W, (out_hw, scale)
        h, w = self.in_hw = tuple(in_hw)
        self.max_in_hw = tuple(max_in_hw or in_hw)
        assert self.max_in_hw[0] >= h and self.max_in_hw[1] >= w, (max_in_hw, in_hw)
        self.cur_hw = self.prev_hw = self.in_hw  # this frame's render size and the last one's
        wg = tuple(tuple(x) for x in wg) if isinstance(wg[0], (tuple, list)) else (tuple(wg), tuple(wg))
        assert entry in ("compute", "fragment"), entry
        assert history in ("bilinear", "catmull", "catmull5") and space in ("rgb", "ycocg") and clip in ("clamp", "centre")
        self.history, self.space, self.clip = history, space, clip
        assert depth_test in ("nearest", "match"), depth_test
        self.depth_test = depth_test
        self.foot = float(foot)
        assert self.foot > 0.0, foot
        self.static_expand = float(expand if static_expand is None else static_expand)
        assert not guided or entry == "compute", "guided mode writes two images: compute entry only"
        self.guided, self.eps, self.reject = bool(guided), float(eps), float(reject or 0.0)
        assert self.static_expand >= expand, (static_expand, expand)
        self.build, self.b, self.wg, self.slots, self.entry = build, BUILDS[build], wg, slots, entry
        assert expand >= 1.0 and prior > 0.0, (expand, prior)  # the box is never shrunk below the samples' min/max
        if mv_sign is not None:  # 0.2.0's name for the same per-axis factor
            warnings.warn("RTUpscaler(mv_sign=) is now mv_scale=", DeprecationWarning, stacklevel=2)
            mv_scale = mv_sign
        self.wmax, self.expand, self.mv_scale = wmax, float(expand), tuple(mv_scale)
        assert depth[0] in (0, 1, 2), depth
        self.depth = (int(depth[0]), float(depth[1]), float(depth[2]))
        self.rel, self.prior, self.shrink = rel, prior, shrink
        if dev is None:
            dev, self.adapter = gpu.device()
        else:
            self.adapter = "(given device)"
        self.dev = dev
        if self.b["hw"] and "32" in self.b["hist"] and "float32-filterable" not in dev.features:
            self.b = dict(self.b, hist="rgba16float")
        U = wgpu.TextureUsage
        rw = U.STORAGE_BINDING | U.TEXTURE_BINDING
        mh, mw = self.max_in_hw
        self.aux = [self.texture((mw, mh), self.b["aux"], rw) for _ in range(2)]
        self.lo = self.texture((mw, mh), self.b["stat"], rw)
        self.hi = self.texture((mw, mh), self.b["stat"], rw)
        extra = U.RENDER_ATTACHMENT if entry == "fragment" else 0
        self.hist = [self.texture((W, H), self.b["hist"], rw | U.COPY_SRC | extra) for _ in range(2)]
        self.hist_views = [t.create_view() for t in self.hist]
        self.ubuf = dev.create_buffer(size=STRIDE * slots, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        if self.guided:  # shading at render size (written by pass A), the remodulated frame at display size
            self.shade = self.texture((mw, mh), self.b["color"], rw)
            self.out = self.texture((W, H), self.b["hist"], U.STORAGE_BINDING | U.COPY_SRC | U.TEXTURE_BINDING)
        else:  # the guided bindings get 1x1 stand-ins, never read or written
            self.shade = self.texture((1, 1), self.b["color"], rw)
            self.out = self.texture((1, 1), self.b["hist"], U.STORAGE_BINDING)
        self.dummy = self.texture((1, 1), self.b["color"], U.TEXTURE_BINDING)
        self.inp = Inputs(self)
        self.inps = [self.inp, Inputs(self)] if self.guided else [self.inp]  # step(): this frame's and the last
        self._pipelines()
        self._groups = {}
        self._engine = {}  # dispatch(): EngineInputs by the textures' ids (holding them, so an id is never reused)
        self.alb_prev = None  # dispatch() in guided mode: our copy of the last frame's display albedo
        self.t = 0

    # -- setup ------------------------------------------------------------------------------------------------------
    def texture(self, size, fmt, usage):
        return self.dev.create_texture(size=(size[0], size[1], 1), format=fmt, usage=usage)

    def _pipelines(self):
        hw, frag = self.b["hw"], self.entry == "fragment"
        C = wgpu.ShaderStage.COMPUTE
        VB = (wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.VERTEX) if frag else C  # visibility of pass B bindings
        filt = "float" if hw else "unfilterable-float"

        def ubo(i, vis=C):
            return {"binding": i, "visibility": vis, "buffer": {"type": wgpu.BufferBindingType.uniform,
                                                              "has_dynamic_offset": True,
                                                              "min_binding_size": PARAMS.size}}

        def tex(i, st="unfilterable-float", vis=C):
            return {"binding": i, "visibility": vis, "texture": {"sample_type": st, "view_dimension": "2d"}}

        def out(i, fmt):
            return {"binding": i, "visibility": C,
                    "storage_texture": {"access": "write-only", "format": fmt, "view_dimension": "2d"}}

        la = [ubo(0), tex(1), tex(2), tex(3), out(4, self.b["aux"]), out(5, self.b["stat"]), out(6, self.b["stat"]),
              tex(7), out(8, self.b["color"])]
        lb = [ubo(0, VB), tex(1, filt, VB), tex(2, vis=VB), tex(3, vis=VB), tex(4, vis=VB), tex(5, vis=VB),
              tex(6, filt, VB)]
        lb += [tex(10, vis=VB), tex(11, filt, VB)]
        if not frag:
            lb += [out(7, self.b["hist"]), out(12, self.b["hist"])]
        if hw:
            lb.append({"binding": 8, "visibility": VB, "sampler": {"type": "filtering"}})
            self.sampler = self.dev.create_sampler(mag_filter="linear", min_filter="linear")
        lb.append({"binding": 9, "visibility": VB, "sampler": {"type": "non-filtering"}})  # gathers
        self.near = self.dev.create_sampler()
        self.la = self.dev.create_bind_group_layout(entries=la)
        self.lb = self.dev.create_bind_group_layout(entries=lb)
        sa, sb = sources(self.b, self.wg, self.history, self.space, self.clip, self.guided, self.entry,
                         self.depth_test)
        self.pa = self._pipeline(sa, self.la)
        if frag:
            m = self.dev.create_shader_module(code=sb)
            self.pb = self.dev.create_render_pipeline(
                layout=self.dev.create_pipeline_layout(bind_group_layouts=[self.lb]),
                vertex={"module": m, "entry_point": "vs", "buffers": []},
                primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
                fragment={"module": m, "entry_point": "fs", "targets": [{"format": self.b["hist"]}]})
        else:
            self.pb = self._pipeline(sb, self.lb)

    def _pipeline(self, src, layout):
        return self.dev.create_compute_pipeline(
            layout=self.dev.create_pipeline_layout(bind_group_layouts=[layout]),
            compute={"module": self.dev.create_shader_module(code=src), "entry_point": "main"})

    def groups(self, inp, par, prev=None):
        """Bind groups of both passes for one input set, writing slot `par` and reading slot 1 - par; guided mode
        also reads the display-size albedo of `prev`, the last frame's inputs."""
        prev = prev or inp
        key = (id(inp), par, id(prev))
        if key not in self._groups:
            u = {"buffer": self.ubuf, "offset": 0, "size": PARAMS.size}
            v = lambda t: t.create_view()  # noqa: E731
            ga = self.dev.create_bind_group(layout=self.la, entries=[
                {"binding": 0, "resource": u}, {"binding": 1, "resource": v(inp.color)},
                {"binding": 2, "resource": v(inp.depth)}, {"binding": 3, "resource": v(inp.mv)},
                {"binding": 4, "resource": v(self.aux[par])}, {"binding": 5, "resource": v(self.lo)},
                {"binding": 6, "resource": v(self.hi)},
                {"binding": 7, "resource": v(inp.alb if self.guided else self.dummy)},
                {"binding": 8, "resource": v(self.shade)}])
            eb = [{"binding": 0, "resource": u}, {"binding": 1, "resource": v(self.shade if self.guided else inp.color)},
                  {"binding": 2, "resource": v(self.aux[par])}, {"binding": 3, "resource": v(self.aux[1 - par])},
                  {"binding": 4, "resource": v(self.lo)}, {"binding": 5, "resource": v(self.hi)},
                  {"binding": 6, "resource": v(self.hist[1 - par])}]
            eb += [{"binding": 10, "resource": v(inp.alb_hr if self.guided else self.dummy)},
                   {"binding": 11, "resource": v(prev.alb_hr if self.guided else self.dummy)}]
            if self.entry == "compute":
                eb += [{"binding": 7, "resource": v(self.hist[par])}, {"binding": 12, "resource": v(self.out)}]
            if self.b["hw"]:
                eb.append({"binding": 8, "resource": self.sampler})
            eb.append({"binding": 9, "resource": self.near})
            self._groups[key] = (ga, self.dev.create_bind_group(layout=self.lb, entries=eb))
        return self._groups[key]

    # -- per frame --------------------------------------------------------------------------------------------------
    def params(self, offset, reset):
        ox, oy = offset
        H, W = self.hw
        h, w = self.cur_hw
        hp, wp = self.prev_hw
        sx, sy = W / w, H / h  # exact per axis: a rounded render size (1707 x 960 for 2560 x 1440 at x1.5) differs
        k = 1.0 / (2.0 * self.foot * self.foot)  # render px: the same at every render size
        return PARAMS.pack(W, H, w, h,
                           ox, oy, sx, k,
                           self.expand, self.wmax, self.mv_scale[0], self.mv_scale[1],
                           w - 1.001, h - 1.001, W - 1.001, H - 1.001,
                           self.rel, self.prior, self.shrink, DEPTH_CLAMP,
                           self.static_expand, self.eps, self.reject, 0.0,
                           int(bool(reset)), self.depth[0], wp, hp,
                           self.depth[1], self.depth[2], sy, 0.0,
                           EXP_MASK, 0, 0, 0)

    def upload(self, inp, fr, alb_lr=None, alb_hr=None):
        h, w = self.in_hw
        q = self.dev.queue
        c = np.empty((h, w, 4), np_type(self.b["color"]))
        c[..., :3] = np.minimum(fr.rgb, 65000.0) if self.b["fast"] else fr.rgb
        c[..., 3] = 1.0
        q.write_texture({"texture": inp.color}, c, {"bytes_per_row": c.strides[0]}, (w, h, 1))
        d = np.ascontiguousarray(fr.depth, np.float32)
        q.write_texture({"texture": inp.depth}, d, {"bytes_per_row": d.strides[0]}, (w, h, 1))
        m = np.ascontiguousarray(fr.mv, np_type(self.b["mv"]))
        q.write_texture({"texture": inp.mv}, m, {"bytes_per_row": m.strides[0]}, (w, h, 1))
        if self.guided:
            H, W = self.hw
            for tex, img, (ww, hh) in ((inp.alb, alb_lr, (w, h)), (inp.alb_hr, alb_hr, (W, H))):
                if not isinstance(img, np.ndarray):  # a GPU texture: copied in step()
                    continue
                c = np.empty((hh, ww, 4), np_type(self.b["color"]))
                c[..., :3] = np.minimum(img, 65000.0) if self.b["fast"] else img
                c[..., 3] = 1.0
                q.write_texture({"texture": tex}, c, {"bytes_per_row": c.strides[0]}, (ww, hh, 1))

    def encode(self, enc, inp, slot=0, timer=None, k=0, prev=None):
        """Record one frame (both passes) reading inputs `inp` and uniform slot `slot`; timer passes k, k + 1. prev:
        the last frame's inputs (guided mode's albedo change test)."""
        ga, gb = self.groups(inp, self.t % 2, prev)
        off = [slot * STRIDE]
        (h, w), (H, W) = self.cur_hw, self.hw
        (ax, ay), (bx, by) = self.wg
        p = enc.begin_compute_pass(timestamp_writes=timer.writes(k) if timer else None)
        p.set_pipeline(self.pa)
        p.set_bind_group(0, ga, off)
        p.dispatch_workgroups(math.ceil(w / ax), math.ceil(h / ay))
        p.end()
        if self.entry == "fragment":
            p = enc.begin_render_pass(
                color_attachments=[{"view": self.hist_views[self.t % 2], "load_op": wgpu.LoadOp.clear,
                                    "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 0)}],
                timestamp_writes=timer.writes(k + 1) if timer else None)
            p.set_pipeline(self.pb)
            p.set_bind_group(0, gb, off)
            p.draw(3)
        else:
            p = enc.begin_compute_pass(timestamp_writes=timer.writes(k + 1) if timer else None)
            p.set_pipeline(self.pb)
            p.set_bind_group(0, gb, off)
            p.dispatch_workgroups(math.ceil(W / bx), math.ceil(H / by))
        p.end()
        self.t += 1
        self.prev_hw = self.cur_hw

    def step(self, fr, alb_lr=None, alb_hr=None, read=True, timer=None):
        """One frame: upload, both passes, and (read=True) the frame back. Guided mode takes the albedo at render
        size (h, w, 3) and display size (H, W, 3)."""
        assert self.guided == (alb_hr is not None), "albedo in guided mode, and only there"
        inp, prev = self.inps[self.t % len(self.inps)], self.inps[(self.t - 1) % len(self.inps)]
        self.upload(inp, fr, alb_lr, alb_hr)
        self.dev.queue.write_buffer(self.ubuf, 0, self.params(cv.sample_offset(*fr.jitter), fr.reset or self.t == 0))
        enc = self.dev.create_command_encoder()
        if self.guided and not isinstance(alb_hr, np.ndarray):  # display albedo already on the GPU (SelfGuided)
            enc.copy_texture_to_texture({"texture": alb_hr}, {"texture": inp.alb_hr}, (self.hw[1], self.hw[0], 1))
        self.encode(enc, inp, 0, timer, prev=prev)
        if timer:
            timer.resolve(enc, 2)
        self.dev.queue.submit([enc.finish()])
        return self.read() if read else None

    def dispatch(self, enc, color, depth, mv, offset, reset=False, albedo=None, albedo_hr=None, timer=None,
                 render_hw=None, mv_scale=None):
        """Record one frame into the engine's command encoder, reading its own textures at render size (colour,
        depth, motion; guided mode: albedo at render and display size). offset: the frame's sample offset (render
        px; offset_from_ndc() for a projection jitter). The frame is then in .output (rgb, a = accumulated weight; in
        guided mode rgb only) until the next dispatch. The uniform is written by queue.write_buffer, so submit each
        frame before recording the next. Dynamic resolution: render_hw = this frame's render size (at most
        max_in_hw; the inputs may be larger, the frame in their top-left) and mv_scale = motion_scale() for that size
        when the vectors' units depend on it (UV, NDC)."""
        self.cur_hw = tuple(render_hw or self.in_hw)
        assert self.cur_hw[0] <= self.max_in_hw[0] and self.cur_hw[1] <= self.max_in_hw[1], (render_hw, self.max_in_hw)
        if mv_scale is not None:
            self.mv_scale = tuple(mv_scale)
        assert self.guided == (albedo_hr is not None), "albedo in guided mode, and only there"
        key = (id(color), id(depth), id(mv), id(albedo), id(albedo_hr))
        inp = self._engine.get(key)
        if inp is None:
            if len(self._engine) >= 16:  # an engine that makes new textures every frame: don't hold them all
                self._engine.clear()
                self._groups.clear()
            inp = self._engine[key] = EngineInputs(self, color, depth, mv, albedo, albedo_hr)
        assert inp.covers(self.cur_hw), ("inputs smaller than the render size", self.cur_hw)
        prev = None
        if self.guided:
            if self.alb_prev is None or self.alb_prev.alb_hr.format != albedo_hr.format:
                self.alb_prev = EngineInputs(self, None, None, None, None, self.texture(
                    self.hw[::-1], albedo_hr.format, wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST))
                self._groups.clear()
            prev = self.alb_prev
        self.dev.queue.write_buffer(self.ubuf, 0, self.params(offset, reset or self.t == 0))
        self.encode(enc, inp, 0, timer, prev=prev)
        if self.guided:  # the next frame's albedo-change test reads this frame's display albedo
            enc.copy_texture_to_texture({"texture": albedo_hr}, {"texture": self.alb_prev.alb_hr},
                                        (self.hw[1], self.hw[0], 1))
        if timer:
            timer.resolve(enc, 2)

    @property
    def output(self):
        """The texture holding the last frame: display size, rgb = the frame (a = its accumulated weight, except in
        guided mode)."""
        return self.out if self.guided else self.hist[(self.t - 1) % 2]

    def release(self):
        """Free the GPU textures now (many short-lived upscalers on a shared GPU: fits, benchmarks)."""
        for inp in self.inps:
            for t in [inp.color, inp.depth, inp.mv] + ([inp.alb, inp.alb_hr] if self.guided else []):
                t.destroy()
        for t in [self.lo, self.hi, self.shade, self.out, self.dummy] + self.aux + self.hist:
            t.destroy()
        self._groups.clear()

    def read(self, weight=False):
        """The last frame written, (H, W, 3) float32 (with weight=True: (H, W, 4), a = accumulated weight)."""
        H, W = self.hw
        t = np_type(self.b["hist"])
        src = self.out if self.guided and not weight else self.hist[(self.t - 1) % 2]
        raw = self.dev.queue.read_texture({"texture": src},
                                          {"bytes_per_row": W * 4 * np.dtype(t).itemsize}, (W, H, 1))
        a = np.frombuffer(raw, t).reshape(H, W, 4).astype(np.float32)
        return a if weight else a[..., :3]


class SelfGuided:
    """Self-guided mode: the engine gives only the albedo at render size (a deferred renderer's G-buffer), not at
    display size. One upscaler accumulates the albedo (long history, wider box: albedo carries no lighting noise); a
    second, guided one accumulates shading = colour / (albedo + eps) and multiplies by the first one's output. The
    albedo history is copied on the GPU, never read back. No albedo-change rejection here: the albedo it would test
    is itself a history. As nts/pipeline.py --self-guide, with the min/max box. fr.albedo: (h, w, 3)."""

    SHARED = ("build", "mv_scale", "depth", "rel", "shrink", "prior", "history", "space", "clip", "wg", "in_hw",
              "depth_test")

    def __init__(self, scale, out_hw, alb_wmax=32.0, alb_expand=2.0, alb_foot=0.25, dev=None, **kw):
        self.up = RTUpscaler(scale, out_hw, dev=dev, guided=True, reject=0.0, **kw)
        self.dev = self.up.dev
        self.alb = RTUpscaler(scale, out_hw, wmax=alb_wmax, expand=alb_expand, foot=alb_foot, dev=self.dev,
                              **{k: v for k, v in kw.items() if k in self.SHARED})
        self.on_gpu = self.alb.b["hist"] == self.up.b["color"]  # else (fast_hist32) read back and upload

    def step(self, fr, read=True, timer=None):
        a = Frame(f=fr.f, rgb=fr.albedo, depth=fr.depth, mv=fr.mv, jitter=fr.jitter, reset=fr.reset)
        self.alb.step(a, read=not self.on_gpu)
        hr = self.alb.hist[(self.alb.t - 1) % 2] if self.on_gpu else self.alb.read()
        return self.up.step(fr, fr.albedo, hr, read=read, timer=timer)

    def read(self, weight=False):
        return self.up.read(weight)

    def release(self):
        self.up.release()
        self.alb.release()
