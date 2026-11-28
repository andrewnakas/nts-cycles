"""A small raster renderer that feeds nts/rt the way a game does: game-like test shots with a supersampled native
reference, and a working example of the engine contract (upscaler.py: dispatch, depth_device, motion_scale,
offset_from_ndc).

Everything is one unit cube, instanced: a ground with a fine checker and grid lines, thin poles and wires (sub-pixel at
render size), an alpha-tested chain-link fence, a zone plate, a barcode sign, neon bars (HDR), office blocks with lit
windows, a bumpy specular block, a fan turning fast, and blocks crossing the view (disocclusion). The shader writes
what a game's main pass gives an upscaler: HDR colour, motion in UV pointing to the previous frame, albedo, and a
reversed-Z depth buffer with an infinite far plane; the jitter is an offset added to the projection. Procedural
textures are box-filtered over their screen footprint times `lod` (1 / scale at render size, as a game biases its
texture mips for temporal upscaling); the zone plate, the barcode and the specular are left unfiltered (shading
aliasing, as in games).

Shots (60 fps): pan, dolly, strafe, orbit, still (only the fan and the blocks move), fast (a fast turn), cut (two
views, a reset at frame 24). The native reference is the same frame at display size, K jittered samples per pixel
averaged (box filter).

Run: python -m nts.rt.scene score [--shots pan,dolly,...] [--modes <conventions.QUALITY_MODES keys>] [--build fast]
                                 [--frames 40] [--warm 8] [--k 32] [--out 1080x1920] [--adapter NAME] [--json f]
                                 [--opts depth_test=nearest] [--start foot=0.3,expand=1.5,...]
     (--modes: render sizes of the 1080x1920 ladder, scaled to --out; default all. Constants: RTUpscaler's defaults,
     or those given by --start)
     python -m nts.rt.scene still <shot> <frame> <out.png> [--scale 2]  (render size, native and nts/rt side by side)
     python -m nts.rt.scene fit [--fit pan,dolly,still,orbit] [--hold strafe,fast,cut] [--modes performance] and
                                nts/rt/fit.py's [--params] [--start] [--rounds] [--min-gain] [--stab] [--log] [--out f]
                                [--opts depth_test=nearest]
"""
import math
import os
import struct
import sys

import numpy as np
import wgpu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from nts import conventions as cv  # noqa: E402
from nts.io_exr import Frame  # noqa: E402
from nts.rt import gpu  # noqa: E402

FPS = 60.0
NEAR = 0.1
FOV_Y = math.radians(60.0)
SHOTS = ("pan", "dolly", "strafe", "orbit", "still", "fast", "cut")
CUT = 24  # the cut shot's second view starts here (a reset)

SHADER = """
struct Frame {
  vp_jit: mat4x4<f32>,   // view-projection with this frame's jitter: where it is drawn
  vp_cur: mat4x4<f32>,   // without the jitter: motion is between unjittered positions
  vp_prev: mat4x4<f32>,
  cam: vec4<f32>,        // camera position, texture filter scale (lod)
  sun: vec4<f32>,        // direction toward the sun, time
}
struct Inst {
  m_cur: mat4x4<f32>,
  m_prev: mat4x4<f32>,
  n_cur: mat4x4<f32>,    // normal matrix (inverse transpose of m_cur)
  mat: vec4<f32>,        // material, three parameters
  col: vec4<f32>,
}
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> I: array<Inst>;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) nrm: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) cur: vec4<f32>,
  @location(4) prev: vec4<f32>,
  @location(5) @interpolate(flat) id: u32,
}

@vertex
fn vs(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>, @location(2) uv: vec2<f32>,
      @builtin(instance_index) ii: u32) -> VOut {
  let inst = I[ii];
  let w = inst.m_cur * vec4<f32>(p, 1.0);
  var o: VOut;
  o.pos = F.vp_jit * w;
  o.world = w.xyz;
  o.nrm = (inst.n_cur * vec4<f32>(n, 0.0)).xyz;
  o.uv = uv;
  o.cur = F.vp_cur * w;
  o.prev = F.vp_prev * (inst.m_prev * vec4<f32>(p, 1.0));
  o.id = ii;
  return o;
}

struct FOut {
  @location(0) color: vec4<f32>,
  @location(1) motion: vec2<f32>,  // UV, toward the previous frame
  @location(2) albedo: vec4<f32>,
  @location(3) lin: f32,           // linear view depth (for the numpy side only; an engine has the depth buffer)
}

fn hash(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn pint(x: f32, lw: f32) -> f32 {
  // the integral from 0 to x of a pulse train: 1 on [k, k + lw), 0 elsewhere
  return floor(x) * lw + min(fract(x), lw);
}

fn pulse(x: f32, w: f32, start: f32, lw: f32) -> f32 {
  // that pulse train shifted by start, box-filtered over a footprint w
  let ww = max(w, 1e-5);
  return (pint(x - start + 0.5 * ww, lw) - pint(x - start - 0.5 * ww, lw)) / ww;
}

fn checker(p: vec2<f32>, w: vec2<f32>) -> f32 {
  // a box-filtered checker of unit squares
  let ww = max(w, vec2<f32>(1e-5));
  let i = 2.0 * (abs(fract((p - 0.5 * ww) * 0.5) - 0.5) - abs(fract((p + 0.5 * ww) * 0.5) - 0.5)) / ww;
  return 0.5 - 0.5 * i.x * i.y;
}

@fragment
fn fs(in: VOut) -> FOut {
  let inst = I[in.id];
  let m = u32(inst.mat.x + 0.5);
  var N = normalize(in.nrm);
  let an = abs(N);
  // planar texture coordinates from the dominant axis (derivatives taken here, in uniform control flow)
  let q = select(select(in.world.xz, in.world.xy, an.z > an.y), in.world.zy, an.x > an.y && an.x > an.z);
  let fq = fwidth(q) * F.cam.w;
  let V = normalize(F.cam.xyz - in.world);
  if (dot(N, V) < 0.0) {
    N = -N;
  }
  var alb = inst.col.rgb;
  var emit = vec3<f32>(0.0);
  var spec = 0.04;
  var shin = 32.0;
  var lit = true;
  if (m == 0u) {  // ground: 1 m checker, grid lines every 0.5 m (3 cm)
    alb = mix(vec3<f32>(0.40, 0.38, 0.35), vec3<f32>(0.62, 0.60, 0.55), checker(q, fq));
    let l = max(pulse(q.x * 2.0, fq.x * 2.0, -0.03, 0.06), pulse(q.y * 2.0, fq.y * 2.0, -0.03, 0.06));
    alb = mix(alb, vec3<f32>(0.07), l);
  } else if (m == 2u) {  // zone plate, unfiltered
    let r = in.uv - 0.5;
    alb = vec3<f32>(0.5 + 0.45 * cos(inst.mat.y * dot(r, r)));
  } else if (m == 3u) {  // barcode, unfiltered
    let b = floor(in.uv.x * inst.mat.y);
    let on = hash(vec2<f32>(b, 7.0)) > 0.5 && in.uv.y > 0.12 && in.uv.y < 0.88;
    alb = select(vec3<f32>(0.85), vec3<f32>(0.03), on);
  } else if (m == 4u) {  // chain-link fence: alpha test on a diagonal wire lattice
    let p = in.world.xy / inst.mat.y;
    let d = vec2<f32>(p.x + p.y, p.x - p.y);
    let dn = abs(d - round(d));
    if (min(dn.x, dn.y) > inst.mat.z) {
      discard;
    }
    spec = 0.6;
    shin = 60.0;
  } else if (m == 5u) {  // bumpy metal: specular sparkle
    let b = in.world * inst.mat.y;
    N = normalize(N + 0.35 * vec3<f32>(sin(b.x) * cos(b.z), sin(b.y + b.x), cos(b.z) * sin(b.y)));
    spec = 1.6;
    shin = 240.0;
  } else if (m == 6u) {  // neon: HDR emission
    emit = inst.col.rgb * inst.mat.y;
    alb = vec3<f32>(0.05);
  } else if (m == 7u) {  // facade: filtered window grid, some windows lit
    let p = q / vec2<f32>(1.2, 1.5);
    let fp = fq / vec2<f32>(1.2, 1.5);
    let inside = pulse(p.x, fp.x, 0.15, 0.7) * pulse(p.y, fp.y, 0.2, 0.6);
    let on = hash(floor(p) + inst.mat.y) > 0.55;
    alb = mix(vec3<f32>(0.36, 0.34, 0.33), vec3<f32>(0.05, 0.07, 0.10), inside);
    emit = select(vec3<f32>(0.0), vec3<f32>(2.6, 2.0, 1.2), on) * inside;
    spec = 0.04 + 0.4 * inside;
    shin = 90.0;
  } else if (m == 8u) {  // sky
    let d = normalize(in.world - F.cam.xyz);
    emit = mix(vec3<f32>(1.05, 1.1, 1.2), vec3<f32>(0.25, 0.45, 0.95), clamp(d.y * 2.5, 0.0, 1.0));
    alb = vec3<f32>(0.0);
    lit = false;
  }
  var c = emit;
  if (lit) {
    let L = normalize(F.sun.xyz);
    let ndl = max(dot(N, L), 0.0);
    let Hv = normalize(L + V);
    let sp = spec * pow(max(dot(N, Hv), 0.0), shin) * ndl;
    c += alb * (0.35 * (0.6 + 0.4 * N.y) + 2.2 * ndl) + vec3<f32>(2.2 * sp);
  }
  var o: FOut;
  o.color = vec4<f32>(c, 1.0);
  o.motion = (in.prev.xy / in.prev.w - in.cur.xy / in.cur.w) * vec2<f32>(0.5, -0.5);
  o.albedo = vec4<f32>(alb, 1.0);
  o.lin = in.cur.w;
  return o;
}
"""

ACCUM = """
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var srca: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> acc: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> acca: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = textureDimensions(src);
  if (gid.x >= s.x || gid.y >= s.y) {
    return;
  }
  let i = gid.y * s.x + gid.x;
  let p = vec2<i32>(gid.xy);
  acc[i] += textureLoad(src, p, 0);
  acca[i] += textureLoad(srca, p, 0);
}
"""

FRAME_U = struct.Struct("<48f4f4f")  # three mat4, cam, sun
INST = 3 * 16 + 4 + 4                 # floats per instance
STRIDE = 256


# -- geometry and scene ----------------------------------------------------------------------------------------------
def cube():
    """The unit cube [-0.5, 0.5]^3 as 36 vertices: position, normal, face uv."""
    v = []
    for ax in range(3):
        for sgn in (-1.0, 1.0):
            n = np.zeros(3)
            n[ax] = sgn
            u, w = [a for a in range(3) if a != ax]
            corners = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
            for a, b in corners:
                p = np.zeros(3)
                p[ax], p[u], p[w] = 0.5 * sgn, a, b
                v.append(np.concatenate([p, n, [a + 0.5, 0.5 - b]]))
    return np.array(v, np.float32)


def trs(t=(0, 0, 0), s=(1, 1, 1), rz=0.0, ry=0.0):
    """Translate x rotate (about y, then z) x scale, as a 4x4 (column vectors)."""
    cz, sz, cy, sy = math.cos(rz), math.sin(rz), math.cos(ry), math.sin(ry)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    m = np.eye(4)
    m[:3, :3] = Ry @ Rz @ np.diag(s)
    m[:3, 3] = t
    return m


def objects(time, cam):
    """Every instance at a time: (model matrix, material (id, p1, p2, p3), colour). The order never changes, so the
    same index is the same object in the previous frame."""
    o = [(trs(cam, (4000, 4000, 4000)), (8, 0, 0, 0), (0, 0, 0))]  # sky, centred on the camera
    o.append((trs((0, -0.05, -20), (220, 0.1, 220)), (0, 0, 0, 0), (0.5, 0.5, 0.5)))
    for k in range(41):  # fence poles every 0.5 m and three wires
        o.append((trs((-10 + 0.5 * k, 0.6, -6), (0.04, 1.2, 0.04)), (1, 0, 0, 0), (0.30, 0.28, 0.26)))
    for y in (0.35, 0.75, 1.15):
        o.append((trs((0, y, -6), (20, 0.012, 0.012)), (1, 0, 0, 0), (0.25, 0.25, 0.27)))
    o.append((trs((6.5, 1.2, -4.2), (6, 2.4, 0.02)), (4, 0.07, 0.1, 0), (0.55, 0.57, 0.60)))  # chain link
    o.append((trs((-6.5, 1.9, -10), (3, 3, 0.05)), (2, 900.0, 0, 0), (1, 1, 1)))  # zone plate
    o.append((trs((2.5, 2.2, -8.6), (4, 1, 0.05)), (3, 90.0, 0, 0), (1, 1, 1)))  # barcode
    for k, c in enumerate(((1.0, 0.25, 0.6), (0.2, 0.9, 1.0), (1.0, 0.85, 0.2))):  # neon bars
        o.append((trs((2.5, 2.95 + 0.16 * k, -8.55), (2.6, 0.05, 0.05)), (6, 8.0, 0, 0), c))
    for k, (x, z, wd, ht) in enumerate(((-16, -30, 9, 18), (-6, -34, 7, 24), (3, -28, 8, 14), (12, -32, 10, 20),
                                        (22, -26, 6, 11), (-26, -24, 8, 9))):  # office blocks
        o.append((trs((x, ht / 2, z), (wd, ht, 6)), (7, float(k * 17), 0, 0), (0.4, 0.4, 0.4)))
    o.append((trs((1.3, 0.6, -2.8), (1.2, 1.2, 1.2), ry=0.5), (5, 9.0, 0, 0), (0.75, 0.6, 0.35)))  # bumpy block
    o.append((trs((-2.2, 1.5, -3.6), (0.4, 3, 0.4)), (1, 0, 0, 0), (0.55, 0.25, 0.2)))  # pillars
    o.append((trs((3.8, 1.5, -5.2), (0.4, 3, 0.4)), (1, 0, 0, 0), (0.2, 0.35, 0.55)))
    hub = np.array((-3.0, 2.6, -7.5))
    a0 = 2 * math.pi * 0.6 * time  # the fan: 0.6 turns per second
    o.append((trs(hub, (0.25, 0.25, 0.3)), (1, 0, 0, 0), (0.2, 0.2, 0.2)))
    for k in range(4):
        a = a0 + k * math.pi / 2
        c = (0.85, 0.85, 0.8) if k % 2 else (0.8, 0.3, 0.15)
        o.append((trs(hub + 0.8 * np.array((math.cos(a), math.sin(a), 0)), (1.6, 0.22, 0.04), rz=a), (1, 0, 0, 0),
                  c))
    for k, (v, z, c) in enumerate(((2.5, -3.0, (0.2, 0.6, 0.25)), (-1.8, -4.6, (0.7, 0.7, 0.75)),
                                   (3.2, -5.4, (0.9, 0.5, 0.1)))):  # blocks crossing the view, wrapping round
        x = (v * time + 5.0 * k + 8.0) % 16.0 - 8.0
        o.append((trs((x, 0.45, z), (0.9, 0.9, 0.9)), (1, 0, 0, 0), c))
    return o


def camera(shot, i):
    """(position, yaw, pitch) of a shot at frame i; yaw 0 looks down -z."""
    d = math.radians
    if shot == "pan":
        return (0.0, 1.7, 3.0), d(-6 + 0.25 * i), d(-4)
    if shot == "dolly":
        return (0.3, 1.7, 3.0 - 2.5 * i / FPS), d(3), d(-4)
    if shot == "strafe":
        return (-1.2 + 2.5 * i / FPS, 1.6, 2.5), d(2), d(-3)
    if shot == "orbit":
        a = d(-12 + 0.5 * i)
        c = np.array((0.0, 1.2, -5.0))
        p = c + 8.0 * np.array((math.sin(a), 0.07, math.cos(a)))
        return tuple(p), a, d(-4)
    if shot == "still":
        return (0.5, 1.6, 2.5), d(5), d(-4)
    if shot == "fast":
        return (0.0, 1.7, 3.0), d(-30 + 1.5 * i), d(-4)
    if shot == "cut":
        return camera("pan", i) if i < CUT else ((2.0, 1.4, -0.5), d(-25 + 0.25 * (i - CUT)), d(-2))
    raise ValueError(shot)


def view_matrix(pos, yaw, pitch):
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    fwd = np.array((-sy * cp, sp, -cy * cp))
    right = np.array((cy, 0.0, -sy))
    up = np.cross(right, fwd)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = right, up, -fwd
    m[:3, 3] = -m[:3, :3] @ np.array(pos)
    return m


def projection(aspect, jx=0.0, jy=0.0):
    """Reversed-Z, infinite far plane: depth = NEAR / view depth. (jx, jy) is added to x, y in NDC."""
    f = 1.0 / math.tan(FOV_Y / 2)
    p = np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0], [0, 0, 0, NEAR], [0, 0, -1, 0]], np.float64)
    t = np.eye(4)
    t[0, 3], t[1, 3] = jx, jy  # clip.xy += (jx, jy) * clip.w
    return t @ p


# -- renderer --------------------------------------------------------------------------------------------------------
class Renderer:
    """Renders a shot's frames at any size with any sample offset; keeps the last frame's textures on the GPU."""

    def __init__(self, dev=None, max_slots=64):
        self.dev = dev or gpu.device()[0]
        d = self.dev
        self.vb = d.create_buffer_with_data(data=cube().tobytes(), usage=wgpu.BufferUsage.VERTEX)
        n = len(objects(0.0, (0, 0, 0)))
        self.n = n
        self.ib = d.create_buffer(size=n * INST * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self.ub = d.create_buffer(size=STRIDE * max_slots, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.max_slots = max_slots
        V, Fr = wgpu.ShaderStage.VERTEX, wgpu.ShaderStage.FRAGMENT
        self.layout = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": V | Fr, "buffer": {"type": "uniform", "has_dynamic_offset": True,
                                                             "min_binding_size": FRAME_U.size}},
            {"binding": 1, "visibility": V | Fr, "buffer": {"type": "read-only-storage"}}])
        self.group = d.create_bind_group(layout=self.layout, entries=[
            {"binding": 0, "resource": {"buffer": self.ub, "offset": 0, "size": FRAME_U.size}},
            {"binding": 1, "resource": {"buffer": self.ib, "offset": 0, "size": n * INST * 4}}])
        m = d.create_shader_module(code=SHADER)
        self.targets = ("rgba16float", "rg16float", "rgba8unorm", "r32float")
        self.pipe = d.create_render_pipeline(
            layout=d.create_pipeline_layout(bind_group_layouts=[self.layout]),
            vertex={"module": m, "entry_point": "vs", "buffers": [{
                "array_stride": 32, "step_mode": "vertex", "attributes": [
                    {"format": "float32x3", "offset": 0, "shader_location": 0},
                    {"format": "float32x3", "offset": 12, "shader_location": 1},
                    {"format": "float32x2", "offset": 24, "shader_location": 2}]}]},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={"format": "depth32float", "depth_write_enabled": True, "depth_compare": "greater"},
            fragment={"module": m, "entry_point": "fs", "targets": [{"format": f} for f in self.targets]})
        self.acc_pipe = d.create_compute_pipeline(layout="auto",
                                                  compute={"module": d.create_shader_module(code=ACCUM),
                                                           "entry_point": "main"})
        self._rt = {}

    def _targets(self, hw):
        if hw not in self._rt:
            h, w = hw
            U = wgpu.TextureUsage
            use = U.RENDER_ATTACHMENT | U.TEXTURE_BINDING | U.COPY_SRC
            tex = [self.dev.create_texture(size=(w, h, 1), format=f, usage=use) for f in self.targets]
            depth = self.dev.create_texture(size=(w, h, 1), format="depth32float",
                                            usage=U.RENDER_ATTACHMENT | U.TEXTURE_BINDING)
            self._rt[hw] = dict(color=tex[0], motion=tex[1], albedo=tex[2], lin=tex[3], depth=depth)
        return self._rt[hw]

    def _uniform(self, shot, i, hw, offset, lod):
        h, w = hw
        pos, yaw, pitch = camera(shot, i)
        ppos, pyaw, ppitch = camera(shot, max(i - 1, 0)) if not (shot == "cut" and i == CUT) else (pos, yaw, pitch)
        aspect = w / h
        jx, jy = -2.0 * offset[0] / w, 2.0 * offset[1] / h  # upscaler.offset_from_ndc inverted
        vp_jit = projection(aspect, jx, jy) @ view_matrix(pos, yaw, pitch)
        vp_cur = projection(aspect) @ view_matrix(pos, yaw, pitch)
        vp_prev = projection(aspect) @ view_matrix(ppos, pyaw, ppitch)
        sun = np.array((0.45, 0.85, 0.3))
        sun /= np.linalg.norm(sun)
        mats = [np.asarray(m, np.float32).T.ravel() for m in (vp_jit, vp_cur, vp_prev)]  # column-major for WGSL
        return FRAME_U.pack(*np.concatenate(mats), *pos, lod, *sun, i / FPS), pos, ppos

    def _instances(self, shot, i, pos, ppos):
        t = i / FPS
        prev_t = t if (shot == "cut" and i == CUT) else (i - 1) / FPS if i > 0 else t
        cur, prev = objects(t, pos), objects(prev_t, ppos)
        a = np.zeros((self.n, INST), np.float32)
        for k, ((m, mat, col), (mp, _, _)) in enumerate(zip(cur, prev)):
            a[k, 0:16] = m.T.ravel()
            a[k, 16:32] = mp.T.ravel()
            a[k, 32:48] = np.linalg.inv(m).ravel()  # (M^-1)^T, column-major = M^-1 row-major
            a[k, 48:52] = mat
            a[k, 52:55] = col
        return a

    def _pass(self, enc, rt, slot):
        p = enc.begin_render_pass(
            color_attachments=[{"view": rt[k].create_view(), "load_op": "clear", "store_op": "store",
                                "clear_value": (0, 0, 0, 0)} for k in ("color", "motion", "albedo", "lin")],
            depth_stencil_attachment={"view": rt["depth"].create_view(), "depth_load_op": "clear",
                                      "depth_store_op": "store", "depth_clear_value": 0.0})
        p.set_pipeline(self.pipe)
        p.set_vertex_buffer(0, self.vb)
        p.set_bind_group(0, self.group, [slot * STRIDE])
        p.draw(36, self.n)
        p.end()

    def render(self, shot, i, hw, offset=(0.0, 0.0), lod=1.0, read=True):
        """Frame i of a shot at size hw = (h, w), each pixel sampling at (x + 0.5 + ox, y + 0.5 + oy). Returns the
        GPU textures (color, motion, albedo, lin, depth) and, with read=True, numpy arrays: rgb, albedo, linear depth
        and motion in render px toward the previous frame (the numpy side's conventions)."""
        rt = self._targets(tuple(hw))
        u, pos, ppos = self._uniform(shot, i, hw, offset, lod)
        q = self.dev.queue
        q.write_buffer(self.ib, 0, self._instances(shot, i, pos, ppos).tobytes())
        q.write_buffer(self.ub, 0, u)
        enc = self.dev.create_command_encoder()
        self._pass(enc, rt, 0)
        q.submit([enc.finish()])
        if not read:
            return rt, None
        h, w = hw
        rgb = self._read(rt["color"], np.float16, 4)[..., :3].astype(np.float32)
        mv = self._read(rt["motion"], np.float16, 2).astype(np.float32) * np.array((w, h), np.float32)
        alb = self._read(rt["albedo"], np.uint8, 4)[..., :3].astype(np.float32) / 255.0
        lin = self._read(rt["lin"], np.float32, 1)[..., 0]
        return rt, dict(rgb=rgb, mv=mv, albedo=alb, depth=lin)

    def native(self, shot, i, hw, k=32):
        """The reference: frame i at display size, k jittered samples per pixel averaged; (rgb, albedo) float32."""
        h, w = hw
        rt = self._targets(tuple(hw))
        u0, pos, ppos = self._uniform(shot, i, hw, (0.0, 0.0), 1.0)
        q = self.dev.queue
        q.write_buffer(self.ib, 0, self._instances(shot, i, pos, ppos).tobytes())
        size = w * h * 16
        B = wgpu.BufferUsage
        acc = [self.dev.create_buffer(size=size, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST) for _ in range(2)]
        group = self.dev.create_bind_group(layout=self.acc_pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": rt["color"].create_view()}, {"binding": 1, "resource": rt["albedo"].create_view()},
            {"binding": 2, "resource": {"buffer": acc[0], "offset": 0, "size": size}},
            {"binding": 3, "resource": {"buffer": acc[1], "offset": 0, "size": size}}])
        offs = [(cv.halton(j + 1, 2) - 0.5, cv.halton(j + 1, 3) - 0.5) for j in range(k)]
        for j0 in range(0, k, self.max_slots):
            enc = self.dev.create_command_encoder()
            for s, off in enumerate(offs[j0:j0 + self.max_slots]):
                q.write_buffer(self.ub, s * STRIDE, self._uniform(shot, i, hw, off, 1.0)[0])
                self._pass(enc, rt, s)
                p = enc.begin_compute_pass()
                p.set_pipeline(self.acc_pipe)
                p.set_bind_group(0, group)
                p.dispatch_workgroups(math.ceil(w / 8), math.ceil(h / 8))
                p.end()
            q.submit([enc.finish()])
        rgb = np.frombuffer(q.read_buffer(acc[0]), np.float32).reshape(h, w, 4)[..., :3] / k
        alb = np.frombuffer(q.read_buffer(acc[1]), np.float32).reshape(h, w, 4)[..., :3] / k
        for b in acc:
            b.destroy()
        return rgb, alb

    def _read(self, tex, dt, c):
        w, h = tex.size[:2]
        raw = self.dev.queue.read_texture({"texture": tex}, {"bytes_per_row": w * c * np.dtype(dt).itemsize},
                                          (w, h, 1))
        return np.frombuffer(raw, dt).reshape(h, w, c)


def render_hw(out_hw, scale):
    H, W = out_hw
    h, w = round(H / scale), round(W / scale)
    assert round(h * scale) == H and round(w * scale) == W, (out_hw, scale)
    return h, w


_NATIVE = {}


class SceneShot:
    """One shot at one scale in nts/rt/fit.py's Shot interface (name, n, warm, scale, hw, gt, frame(i)): render
    frames are drawn when asked for, the tonemapped native frames once per shot and display size (shared across
    scales). albedo=True: frame(i).albedo is the render-size albedo and .alb_hr[i] the display-size one."""

    def __init__(self, shot, scale, n, warm, tm, out_hw=(1080, 1920), k=32, renderer=None, albedo=False):
        self.r = renderer or Renderer()
        self.name, self.shot, self.scale, self.n, self.warm, self.hw = f"{shot}_x{scale:g}", shot, float(scale), n, \
            warm, tuple(out_hw)
        self.in_hw = render_hw(self.hw, self.scale)
        self.scale = self.hw[1] / self.in_hw[1]  # exact (Balanced is 12/7, not 1.714)
        self.jit = cv.jitter_seq(n, cv.jitter_phases(self.in_hw[0]))
        self.reset = [i == 0 or (shot == "cut" and i == CUT) for i in range(n)]
        self.albedo = albedo
        key = (shot, n, warm, self.hw, k, albedo, id(tm))
        if key not in _NATIVE:
            _NATIVE.clear()  # one shot's reference at a time, kept tonemapped (8 bit): the float frames are 1.6 GB
            gt, alb = [], []
            for i in range(0 if albedo else warm, n):
                rgb, a = self.r.native(shot, i, self.hw, k)
                if i >= warm:
                    gt.append(tm(rgb))
                if albedo:
                    alb.append(a)
            _NATIVE[key] = gt, (alb if albedo else None)
        self.gt, self.alb_hr = _NATIVE[key]  # shared between the scales of a shot, never written

    def frame(self, i):
        ox, oy = cv.sample_offset(*self.jit[i])
        _, a = self.r.render(self.shot, i, self.in_hw, (ox, oy), lod=1.0 / self.scale)
        return Frame(f=i, rgb=a["rgb"], depth=a["depth"], mv=a["mv"], albedo=a["albedo"] if self.albedo else None,
                     jitter=self.jit[i], reset=self.reset[i])


def main():
    from baselines.accumulate import arg
    from nts.rt.fit import VIEW, score
    from nts.tonemap import Tonemapper
    a = sys.argv
    if len(a) < 2 or a[1] not in ("score", "still", "fit"):
        sys.exit(__doc__)
    dev, name = gpu.device(arg("--adapter"))
    tm = Tonemapper(*VIEW)
    r = Renderer(dev)
    H, W = map(int, arg("--out", "1080x1920").split("x"))
    k = int(arg("--k", 32))
    if a[1] == "still":
        from PIL import Image
        from nts.rt.upscaler import RTUpscaler
        shot, f, path, s = a[2], int(a[3]), a[4], float(arg("--scale", 2))
        sh = SceneShot(shot, s, f + 1, f, tm, (H, W), k, r)
        up = RTUpscaler(s, (H, W), build=arg("--build", "fast"), dev=dev)
        for i in range(f + 1):
            fr = sh.frame(i)
            out = up.step(fr, read=i == f)
        lr = tm(np.repeat(np.repeat(fr.rgb, int(math.ceil(s)), 0), int(math.ceil(s)), 1)[:H, :W])
        Image.fromarray(np.concatenate([lr, tm(out), sh.gt[0]], 1)).save(path)
        print(f"{path}: render (nearest), nts/rt, native; {shot} frame {f} x{s:g}")
        return
    if a[1] == "fit":
        import json
        import time
        from nts.rt.fit import DEFAULTS, search
        modes = arg("--modes", "performance").split(",")
        n, warm, build = int(arg("--frames", 32)), int(arg("--warm", 8)), arg("--build", "fast")
        t0 = time.time()

        def shots(names):
            return [SceneShot(sh, cv.OUT[0] / cv.QUALITY_MODES[m][0], n, warm, tm, (H, W), k, r)
                    for sh in names.split(",") if sh for m in modes]
        fit, hold = shots(arg("--fit", "pan,dolly,still,orbit")), shots(arg("--hold", "strafe,fast,cut"))
        best = dict(DEFAULTS)
        if arg("--start"):
            best.update({x.split("=")[0]: float(x.split("=")[1]) for x in arg("--start").split(",")})
        names = arg("--params", ",".join(DEFAULTS)).split(",")
        print(f"scene fit: {len(fit)} shots, {len(hold)} held out, modes {modes}, frames {warm}-{n - 1}, {build} on "
              f"{name}, {W}x{H} (references in {time.time() - t0:.0f} s)", flush=True)
        opts = dict(x.split("=") for x in arg("--opts").split(",")) if arg("--opts") else {}
        res = search(fit, hold, best, names, dev, build, tm, opts, int(arg("--rounds", 3)),
                     float(arg("--min-gain", 0.005)), float(arg("--stab", 3)), arg("--log"))
        res.update(adapter=name, frames=[warm, n], modes=modes, out=[H, W])
        print("best:", " ".join(f"{k}={v:.4g}" for k, v in res["best"].items()), f"({(time.time() - t0) / 60:.0f} min)")
        if arg("--out-json"):
            json.dump(res, open(arg("--out-json"), "w"), indent=1)
        return
    shots = arg("--shots", ",".join(SHOTS)).split(",")
    modes = arg("--modes", ",".join(cv.QUALITY_MODES)).split(",")
    scales = [cv.OUT[0] / cv.QUALITY_MODES[m][0] for m in modes]  # both [0] are heights (1080 / render h)
    n, warm, build = int(arg("--frames", 40)), int(arg("--warm", 8)), arg("--build", "fast")
    opts = dict(x.split("=") for x in arg("--opts").split(",")) if arg("--opts") else {}
    params = {x.split("=")[0]: float(x.split("=")[1]) for x in arg("--start").split(",")} if arg("--start") else {}
    print(f"scene: {build} build on {name}, {W}x{H} output, frames {warm}-{n - 1} scored, native k={k}, "
          f"constants {params or 'default'}, {opts or 'default switches'}")
    print(f"{'shot':<10} {'mode':<18} | {'PSNR':>6} {'tErr':>6} {'flicker':>7} | {'bilinear PSNR':>13} {'tErr':>6} "
          f"{'flicker':>7}")
    for shot in shots:
        for md, s in zip(modes, scales):
            sh = SceneShot(shot, s, n, warm, tm, (H, W), k, r)
            rt = score([sh], params, dev, build, tm, opts)["rows"][0]
            bl = score([sh], dict(params, wmax=1e-6, prior=1e3), dev, build, tm, opts)["rows"][0]  # ~ bilinear
            print(f"{shot:<10} {md:<18} | {rt['psnr']:6.2f} {rt['terr']:6.3f} {rt['flicker']:7.4f} | "
                  f"{bl['psnr']:13.2f} {bl['terr']:6.3f} {bl['flicker']:7.4f}", flush=True)
            if arg("--json"):
                import json
                with open(arg("--json"), "a") as fh:
                    fh.write(json.dumps(dict(shot=shot, mode=md, scale=sh.scale, build=build, adapter=name,
                                             out=[H, W], params=params, opts=opts, rt=rt, bilinear=bl)) + "\n")


if __name__ == "__main__":
    main()
