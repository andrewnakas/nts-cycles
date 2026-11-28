// nts.rt pass A, one thread per render pixel: the motion of the nearest surface and the colour neighbourhood.
// Mirrors nts/warp.py dilate_mv (3x3 window, rows then columns, strict <, edges repeated) and the 3x3 per-channel
// min and max of baselines/accumulate.py minmax_box (Karis 2014); pass B widens it about its middle (expand >= 1).
// Fields in double braces are filled in by nts/rt/upscaler.py.

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
@group(0) @binding(2) var depth_lr: texture_2d<f32>;
@group(0) @binding(3) var mv_lr: texture_2d<f32>;
@group(0) @binding(4) var aux_out: texture_storage_2d<{{F_AUX}}, write>;
@group(0) @binding(5) var lo_out: texture_storage_2d<{{F_STAT}}, write>;
@group(0) @binding(6) var hi_out: texture_storage_2d<{{F_STAT}}, write>;
@group(0) @binding(7) var albedo_lr: texture_2d<f32>;                       // guided mode
@group(0) @binding(8) var shade_out: texture_storage_2d<{{F_COLOR}}, write>;  // guided mode: colour / (albedo + eps)

const FAST: bool = {{FAST}};
const GUIDED: bool = {{GUIDED}};
const YCOCG: bool = {{YCOCG}};  // the box in luma/chroma (YCoCg) instead of RGB

fn col(q: vec2<i32>) -> vec3<f32> {
  var v = textureLoad(color_lr, q, 0).rgb;
  if (GUIDED) {  // shading: what the accumulator carries in guided mode (baselines/guided.py)
    v = v / (textureLoad(albedo_lr, q, 0).rgb + P.flt.y);
  }
  if (FAST) {
    return min(v, vec3<f32>(65000.0));
  }
  return v;
}

// The workgroup's tile plus a one-pixel border (edges repeated, as the clamped window) is read once into workgroup
// memory; every pixel then takes its 3x3 window from there.
const GX: i32 = {{GX}};
const GY: i32 = {{GY}};
const TW: i32 = GX + 2;
const TH: i32 = GY + 2;
var<workgroup> tc: array<vec3<f32>, {{TN}}>;
var<workgroup> td: array<f32, {{TN}}>;

@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let w = i32(P.sizes.z);
  let h = i32(P.sizes.w);
  let edge = vec2<i32>(w - 1, h - 1);
  let origin = vec2<i32>(wid.xy) * vec2<i32>(GX, GY) - 1;
  for (var t = i32(li); t < TW * TH; t += GX * GY) {
    let q = clamp(origin + vec2<i32>(t % TW, t / TW), vec2<i32>(0), edge);
    let c = col(q);
    if (YCOCG) {
      tc[t] = vec3<f32>(c.r * 0.25 + c.g * 0.5 + c.b * 0.25, c.r * 0.5 - c.b * 0.5, c.g * 0.5 - c.r * 0.25 - c.b * 0.25);
    } else {
      tc[t] = c;
    }
    td[t] = textureLoad(depth_lr, q, 0).x;
  }
  workgroupBarrier();
  let p = vec2<i32>(gid.xy);
  if (p.x >= w || p.y >= h) {
    return;
  }
  let c0 = (p.y - origin.y) * TW + (p.x - origin.x);  // this pixel's tile index
  let dc = td[c0];
  var best = dc;
  var bq = p;
  var lo = tc[c0];
  var hi = lo;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let t = c0 + dy * TW + dx;
      let d = td[t];
      if (d < best) {
        best = d;
        bq = clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), edge);
      }
      lo = min(lo, tc[t]);
      hi = max(hi, tc[t]);
    }
  }
  // (motion * sign) * scale: render px -> output px, as nts/warp.py hr_motion
  let mv = textureLoad(mv_lr, bq, 0).xy * P.acc.zw * P.jit.z;
  textureStore(aux_out, p, vec4<f32>(mv, min(dc, P.tun.w), 0.0));
  if (GUIDED) {
    textureStore(shade_out, p, vec4<f32>(col(p), 1.0));
  }
  textureStore(lo_out, p, vec4<f32>(lo, 0.0));
  textureStore(hi_out, p, vec4<f32>(hi, 0.0));
}
