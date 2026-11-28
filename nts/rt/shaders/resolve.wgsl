// nts.rt pass B, one thread per output pixel: the rest of baselines/accumulate.py Accumulator.step.
//   jitter splat (nts/splat.py) as a gather, bilinear fallback, reprojected history and weight, disocclusion
//   (nts/warp.py _disocclusion), clip to the min/max box, weight shrink, then one weighted sum of history, samples and
//   the bilinear frame (a fixed-weight prior). Output rgb = the frame, a = its weight.
// Fields in double braces are filled in by nts/rt/upscaler.py; FETCH is the bilinear lookup of the build: numpy's
// four-tap formula with textureLoad (compat) or the texture unit's own filter (fast).

struct Params {
  sizes: vec4<u32>,  // W, H (output), w, h (render)
  jit: vec4<f32>,    // ox, oy (image-space sample offset, render px), scale, k = 1 / (2 sigma^2)
  acc: vec4<f32>,    // box expand (>= 1), wmax, motion sign x, motion sign y
  lim: vec4<f32>,    // f32(w - 1.001), f32(h - 1.001), f32(W - 1.001), f32(H - 1.001): the bilinear edge clip
  tun: vec4<f32>,    // depth tolerance, prior weight, weight shrink slope, depth clamp
  flt: vec4<f32>,    // box widening on a pixel that did not move (>= expand), albedo eps, albedo reject tau, 0
  flags: vec4<u32>,  // reset, 0, 0, 0
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
const HW: bool = {{HW}};      // the texture unit's bilinear filter (FETCH, splat2) instead of numpy's formula
const CATMULL: bool = {{CATMULL}};  // colour history by Catmull-Rom (hist_cr in FETCH) instead of bilinear
const GUIDED: bool = {{GUIDED}};    // colour_lr holds shading; the history holds shading
const CR5: bool = {{CR5}};          // ... without its four corner taps (five filtered lookups in the fast build)
const YCOCG: bool = {{YCOCG}};      // the box is in YCoCg (pass A): clip there
const CENTRE: bool = {{CENTRE}};    // clip along the line to the box's middle instead of per channel

fn to_ycocg(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(c.r * 0.25 + c.g * 0.5 + c.b * 0.25, c.r * 0.5 - c.b * 0.5, c.g * 0.5 - c.r * 0.25 - c.b * 0.25);
}

fn from_ycocg(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

{{FETCH}}

fn finite3(v: vec3<f32>) -> bool {
  // exponent bits, not v == v: a shader compiler may assume NaN and inf never occur
  let e = bitcast<vec3<u32>>(v) & vec3<u32>(0x7f800000u);
  return all(e != vec3<u32>(0x7f800000u));
}

// The whole per-pixel computation; the entry points (ENTRY, below) only decide where its result goes.
fn resolve_px(X: i32, Y: i32) -> vec4<f32> {
  let W = i32(P.sizes.x);
  let H = i32(P.sizes.y);
  let w = i32(P.sizes.z);
  let h = i32(P.sizes.w);
  let ox = P.jit.x;
  let oy = P.jit.y;
  let scale = P.jit.z;
  let k = P.jit.w;
  let fX = f32(X);
  let fY = f32(Y);

  // 1. splat as a gather: every sample whose floor(position) lies within one pixel of (X, Y), per axis, with the
  //    sample position worked out by the same expression as conventions.sample_positions_hr. Positions grow with the
  //    sample index, so the samples taken on each axis are consecutive: the first one and a count are enough (no
  //    arrays, which a GPU compiler may move out of registers). Samples outside the render are skipped (a clamped
  //    index would count an edge sample twice).
  let x0 = i32(floor((fX - 1.0) / scale - 0.5 - ox)) - 1;
  let y0 = i32(floor((fY - 1.0) / scale - 0.5 - oy)) - 1;
  var ix = 0;
  var nx = 0;
  var iy = 0;
  var ny = 0;
  for (var t = 0; t < 5; t++) {
    let i = x0 + t;
    let bx = i32(floor((f32(i) + 0.5 + ox) * scale));
    if (i >= 0 && i < w && bx >= X - 1 && bx <= X + 1) {
      if (nx == 0) {
        ix = i;
      }
      nx++;
    }
    let j = y0 + t;
    let by = i32(floor((f32(j) + 0.5 + oy) * scale));
    if (j >= 0 && j < h && by >= Y - 1 && by <= Y + 1) {
      if (ny == 0) {
        iy = j;
      }
      ny++;
    }
  }
  let lsize = vec2<f32>(f32(w), f32(h));
  var s = vec3<f32>(0.0);
  var ws = 0.0;
  if (HW && nx > 0 && ny > 0 && nx <= 2 && ny <= 2) {
    let sp = splat2(ix, iy, nx, ny, fX, fY, lsize);
    s = sp.rgb;
    ws = sp.a;
  } else {
    for (var a = 0; a < ny; a++) {
      let j = iy + a;
      let dy = fY + 0.5 - (f32(j) + 0.5 + oy) * scale;
      for (var b = 0; b < nx; b++) {
        let i = ix + b;
        let dx = fX + 0.5 - (f32(i) + 0.5 + ox) * scale;
        let wt = exp(-(dx * dx + dy * dy) * k);
        var c = textureLoad(color_lr, vec2<i32>(i, j), 0).rgb;
        if (FAST) {
          c = min(c, vec3<f32>(65000.0));
        }
        s += wt * c;
        ws += wt;
      }
    }
  }

  // 2. the bilinear frame at the pixel centre (warp.upsample_bilinear)
  let cx = (fX + 0.5) / scale - 0.5;
  let cy = (fY + 0.5) / scale - 0.5;
  var base = fetch(color_lr, cx, cy, P.lim.x, P.lim.y, lsize).rgb;
  if (FAST) {
    base = min(base, vec3<f32>(65000.0));
  }

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
    // another surface: off screen, or the nearest previous depth in the 2x2 output footprint differs
    let off = px < 0.0 || py < 0.0 || px > f32(W - 1) || py > f32(H - 1);
    let qx = i32(min(max(px, 0.0), f32(W - 2)));
    let qy = i32(min(max(py, 0.0), f32(H - 2)));
    let ax0 = (qx * w) / W;
    let ay0 = (qy * h) / H;
    // the four output pixels of the footprint fall on one render texel or on two neighbours per axis: one gather
    // fetches the 2x2 render texels, and only those numpy's footprint covers enter the minimum (exact)
    let g = textureGather(2, aux_prev, near, (vec2<f32>(f32(ax0), f32(ay0)) + 1.0) / lsize);
    let ex = ((qx + 1) * w) / W != ax0;
    let ey = ((qy + 1) * h) / H != ay0;
    var pd = g.w;  // gather order: x (i0, j1), y (i1, j1), z (i1, j0), w (i0, j0)
    if (ex) {
      pd = min(pd, g.z);
    }
    if (ey) {
      pd = min(pd, g.x);
    }
    if (ex && ey) {
      pd = min(pd, g.y);
    }
    if (off || abs(pd - ac.z) > P.tun.x * ac.z) {
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
      lo = c - r;
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
  if (FAST && !finite3(o)) {
    o = base;
  }
  return vec4<f32>(o, min(tot, P.acc.y));
}

{{ENTRY}}
