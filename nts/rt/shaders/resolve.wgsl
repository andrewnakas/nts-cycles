// nts.rt pass B, one thread per output pixel: the rest of baselines/accumulate.py Accumulator.step.
//   this frame's samples (nts/gather.py), bilinear fallback, reprojected history and weight, disocclusion
//   (nts/warp.py _disocclusion), clip to the min/max box, weight shrink, then one weighted sum of history, samples and
//   the bilinear frame (a fixed-weight prior). Output rgb = the frame, a = its weight.
// Fields in double braces are filled in by nts/rt/upscaler.py; FETCH is the bilinear lookup of the build: numpy's
// four-tap formula with textureLoad (compat) or the texture unit's own filter (fast).

struct Params {
  sizes: vec4<u32>,  // W, H (output), w, h (render)
  jit: vec4<f32>,    // ox, oy (image-space sample offset, render px), scale x (W / w), k = 1 / (2 foot^2)
  acc: vec4<f32>,    // box expand (>= 1), wmax, motion sign x, motion sign y
  lim: vec4<f32>,    // f32(w - 1.001), f32(h - 1.001), f32(W - 1.001), f32(H - 1.001): the bilinear edge clip
  tun: vec4<f32>,    // depth tolerance, prior weight, weight shrink slope, depth clamp
  flt: vec4<f32>,    // box widening on a pixel that did not move (>= expand), albedo eps, albedo reject tau, 0
  flags: vec4<u32>,  // reset, depth mode (0 linear, 1 z = 1 / (A d + B), 2 z = A d + B), the previous frame's
                     // render w, h (dynamic resolution: the render size may change every frame)
  dep: vec4<f32>,    // A, B of the depth mode (upscaler.py depth_device / depth_ortho), scale y (H / h), 0
  bits: vec4<u32>,   // 0x7f800000 (the float exponent mask), 0, 0, 0: see finite3
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var color_lr: texture_2d<f32>;
@group(0) @binding(2) var aux_cur: texture_2d<f32>;
@group(0) @binding(3) var aux_prev: texture_2d<f32>;
@group(0) @binding(4) var lo_lr: texture_2d<f32>;
@group(0) @binding(5) var hi_lr: texture_2d<f32>;
@group(0) @binding(6) var hist_prev: texture_2d<f32>;
{{SAMPLER}}
@group(0) @binding(9) var near: sampler;
@group(0) @binding(10) var alb_hr: texture_2d<f32>;    // guided mode: this frame's albedo at display size
@group(0) @binding(11) var alb_prev: texture_2d<f32>;  // guided mode: the last frame's

const FAST: bool = {{FAST}};  // half-float buffers: clamp to 65000, fall back on non-finite results
const HW: bool = {{HW}};      // the texture unit's bilinear filter (FETCH, gather2) instead of numpy's formula
const CATMULL: bool = {{CATMULL}};  // colour history by Catmull-Rom (hist_cr in FETCH) instead of bilinear
const GUIDED: bool = {{GUIDED}};    // colour_lr holds shading; the history holds shading
const CR5: bool = {{CR5}};          // ... without its four corner taps (five filtered lookups in the fast build)
const YCOCG: bool = {{YCOCG}};      // the box is in YCoCg (pass A): clip there
const CENTRE: bool = {{CENTRE}};    // clip along the line to the box's middle instead of per channel
const DMATCH: bool = {{DMATCH}};    // depth test: another surface when no footprint depth matches (else: the nearest)

fn to_ycocg(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(c.r * 0.25 + c.g * 0.5 + c.b * 0.25, c.r * 0.5 - c.b * 0.5, c.g * 0.5 - c.r * 0.25 - c.b * 0.25);
}

fn from_ycocg(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

{{FETCH}}

fn finite3(v: vec3<f32>) -> bool {
  // the exponent bits, not v == v, and the mask from the uniform, not a constant: a shader compiler may assume NaN and
  // inf never occur (Vulkan allows it unless the shader declares otherwise) and fold away any test it recognises,
  // which an NVIDIA driver was seen to do with the constant mask (an RTX 4050: NaN went through to the frame)
  let m = vec3<u32>(P.bits.x);
  return all((bitcast<vec3<u32>>(v) & m) != m);
}

fn finite1(v: f32) -> bool {
  return (bitcast<u32>(v) & P.bits.x) != P.bits.x;
}

// one render pixel of this frame's samples, weighted (rgb = weight x colour, a = weight); zero weight: not read
fn tap(i: i32, j: i32, wt: f32) -> vec4<f32> {
  if (wt <= 0.0) {
    return vec4<f32>(0.0);
  }
  var c = textureLoad(color_lr, vec2<i32>(i, j), 0).rgb;
  if (FAST) {
    c = min(c, vec3<f32>(65000.0));
  }
  return vec4<f32>(wt * c, wt);
}

// The whole per-pixel computation; the entry points (ENTRY, below) only decide where its result goes.
fn resolve_px(X: i32, Y: i32) -> vec4<f32> {
  let W = i32(P.sizes.x);
  let H = i32(P.sizes.y);
  let w = i32(P.sizes.z);
  let h = i32(P.sizes.w);
  let ox = P.jit.x;
  let oy = P.jit.y;
  let sx = P.jit.z;  // output / render size, per axis (a rounded render size makes them differ slightly)
  let sy = P.dep.z;
  let k = P.jit.w;
  let fX = f32(X);
  let fY = f32(Y);

  let lsize = vec2<f32>(textureDimensions(color_lr));
  var s = vec3<f32>(0.0);
  var ws = 0.0;
  // 1. this frame's samples, found from the render side: (u, v) is this output pixel's centre in the render's
  //    jittered sample grid (render pixel i's sample lies at u = i), and the 2 x 2 render pixels around it are
  //    weighted by a separable Gaussian of their distance from it in render pixels (k = 1 / (2 foot^2)). Render
  //    pixels outside the render get no weight. As nts/gather.py.
  let u = (fX + 0.5) / sx - 0.5 - ox;
  let v = (fY + 0.5) / sy - 0.5 - oy;
  let i0 = i32(floor(u));
  let j0 = i32(floor(v));
  let du = f32(i0) - u;
  let dv = f32(j0) - v;
  let wx0 = select(0.0, exp(-(du * du) * k), i0 >= 0 && i0 < w);
  let wx1 = select(0.0, exp(-((du + 1.0) * (du + 1.0)) * k), i0 + 1 >= 0 && i0 + 1 < w);
  let wy0 = select(0.0, exp(-(dv * dv) * k), j0 >= 0 && j0 < h);
  let wy1 = select(0.0, exp(-((dv + 1.0) * (dv + 1.0)) * k), j0 + 1 >= 0 && j0 + 1 < h);
  let tx = wx0 + wx1;
  let ty = wy0 + wy1;
  if (HW) {
    // separable weights: the four taps are one bilinear lookup at their balance point, times tx * ty
    if (tx > 0.0 && ty > 0.0) {
      let g = gather2(vec2<f32>(f32(i0) + wx1 / tx, f32(j0) + wy1 / ty), lsize);
      s = g * (tx * ty);
      ws = tx * ty;
    }
  } else {
    let t00 = tap(i0, j0, wy0 * wx0);
    let t01 = tap(i0 + 1, j0, wy0 * wx1);
    let t10 = tap(i0, j0 + 1, wy1 * wx0);
    let t11 = tap(i0 + 1, j0 + 1, wy1 * wx1);
    s = ((t00.rgb + t01.rgb) + t10.rgb) + t11.rgb;
    ws = ((t00.a + t01.a) + t10.a) + t11.a;
  }

  // 2. the bilinear frame at the pixel centre (warp.upsample_bilinear)
  if (!finite3(s) || !finite1(ws)) {  // a broken input pixel: this frame's samples are left out here
    s = vec3<f32>(0.0);
    ws = 0.0;
  }
  let cx = (fX + 0.5) / sx - 0.5;
  let cy = (fY + 0.5) / sy - 0.5;
  var base = fetch(color_lr, cx, cy, P.lim.x, P.lim.y, lsize).rgb;
  if (FAST) {
    base = min(base, vec3<f32>(65000.0));
  }
  base = select(vec3<f32>(0.0), base, finite3(base));

  // 3. history along the nearest surface's motion
  let li = vec2<i32>((X * w) / W, (Y * h) / H);  // the render pixel this output pixel falls in
  let ac = textureLoad(aux_cur, li, 0);
  var hist = base;
  var wh = 0.0;
  if (P.flags.x == 0u) {
    let px = fX + ac.x;
    let py = fY + ac.y;
    let hp = fetch(hist_prev, px, py, P.lim.z, P.lim.w, vec2<f32>(f32(W), f32(H)));
    hist = hp.rgb;
    wh = hp.a;
    if (CATMULL) {
      hist = hist_cr(px, py, W, H);
    }
    if (!finite3(hist) || !finite1(wh)) {  // never let a broken value live on in the history
      hist = base;
      wh = 0.0;
    }
    // another surface: off screen, or the nearest previous depth in the 2x2 output footprint differs
    let off = px < 0.0 || py < 0.0 || px > f32(W - 1) || py > f32(H - 1);
    let qx = i32(min(max(px, 0.0), f32(W - 2)));
    let qy = i32(min(max(py, 0.0), f32(H - 2)));
    let wprev = i32(P.flags.z);  // the previous frame's render size: its depth was stored at that size
    let hprev = i32(P.flags.w);
    let ax0 = (qx * wprev) / W;
    let ay0 = (qy * hprev) / H;
    // the four output pixels of the footprint fall on one render texel or on two neighbours per axis: one gather
    // fetches the 2x2 render texels, and only those numpy's footprint covers enter the minimum (exact)
    let g = textureGather(2, aux_prev, near, (vec2<f32>(f32(ax0), f32(ay0)) + 1.0) /
                          vec2<f32>(textureDimensions(aux_prev)));
    let ex = ((qx + 1) * wprev) / W != ax0;
    let ey = ((qy + 1) * hprev) / H != ay0;
    var pd = g.w;  // gather order: x (i0, j1), y (i1, j1), z (i1, j0), w (i0, j0)
    var dd = abs(g.w - ac.z);
    if (ex) {
      pd = min(pd, g.z);
      dd = min(dd, abs(g.z - ac.z));
    }
    if (ey) {
      pd = min(pd, g.x);
      dd = min(dd, abs(g.x - ac.z));
    }
    if (ex && ey) {
      pd = min(pd, g.y);
      dd = min(dd, abs(g.y - ac.z));
    }
    let miss = select(abs(pd - ac.z), dd, DMATCH);
    if (off || miss > P.tun.x * ac.z) {
      wh = 0.0;
    }
    if (GUIDED && P.flt.z > 0.0) {
      // another surface by its albedo: the last frame's display albedo, warped here, against this frame's
      let pa = fetch(alb_prev, px, py, P.lim.z, P.lim.w, vec2<f32>(f32(W), f32(H))).rgb;
      let ca = textureLoad(alb_hr, vec2<i32>(X, Y), 0).rgb;
      let d = abs(pa - ca);
      let ch = (d.x + d.y + d.z) / ((pa.x + pa.y + pa.z) + (ca.x + ca.y + ca.z) + 3.0 * P.flt.y);
      wh *= exp(-ch / P.flt.z);
    }
    // the min/max box of that render pixel's 3x3 neighbourhood (pass A)
    var lo = textureLoad(lo_lr, li, 0).rgb;
    var hi = textureLoad(hi_lr, li, 0).rgb;
    // widened about its middle, never shrunk (upscaler.py checks both factors >= 1): by expand, or on a pixel that
    // did not move by the still-pixel factor
    let e = select(P.acc.x, P.flt.x, sqrt(ac.x * ac.x + ac.y * ac.y) < 0.01);
    if (e != 1.0) {
      let c = (lo + hi) * 0.5;
      let r = (hi - lo) * (0.5 * e);
      // in RGB, never widened below zero unless the samples themselves go there: otherwise history that overshoots
      // at an edge (Catmull-Rom) is let through as a negative colour no renderer made, which a log tonemapper turns
      // into NaN
      lo = select(c - r, max(c - r, min(lo, vec3<f32>(0.0))), !YCOCG);
      hi = c + r;
    }
    var hh = hist;
    if (YCOCG) {
      hh = to_ycocg(hist);
    }
    var cl = min(max(hh, lo), hi);
    if (CENTRE) {
      let c = (lo + hi) * 0.5;
      let e = (hi - lo) * 0.5 + 1e-6;
      let v = hh - c;
      let a = abs(v) / e;
      let m = max(a.x, max(a.y, a.z));
      cl = select(hh, c + v / m, m > 1.0);
    }
    if (YCOCG) {
      cl = from_ycocg(cl);
    }
    // history that had to move a lot to fit the neighbourhood is less trustworthy
    let d = abs(cl - hist);
    let ah = abs(hist);
    let moved = (d.x + d.y + d.z) / ((ah.x + ah.y + ah.z) + 1e-3);
    wh *= clamp(1.0 - P.tun.z * moved, 0.0, 1.0);
    hist = cl;
  }
  // 4. one normalised sum: history, this frame's samples, and the bilinear frame at a fixed weight (where history
  //    and samples are thin it dominates; the stored weight leaves it out)
  let tot = wh + ws;
  var o = (hist * wh + s + base * P.tun.y) / max(tot + P.tun.y, 1e-6);
  if (!finite3(o) || !finite1(tot)) {  // (overflow in the half-float build)
    return vec4<f32>(base, 0.0);
  }
  return vec4<f32>(o, min(tot, P.acc.y));
}

{{ENTRY}}
