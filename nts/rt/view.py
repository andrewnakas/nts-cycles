"""Watch nts.rt live: a packed clip upscaled on the GPU every frame, split screen against plain bilinear upscaling.

Left of the split: the render (low res) stretched to output size with bilinear filtering. Right: nts.rt's output.
The GPU time of nts.rt's two passes and the display rate are in the window title. Display tone is a plain exposure +
Reinhard + sRGB curve for viewing, not Blender's AgX (the eval's frames are tonemapped properly).

Controls: move the mouse to move the split; space pauses; right arrow steps one frame while paused; wheel zooms about
the pointer; 0 resets the zoom; r restarts the clip (the history starts over, as on a cut).

Run: python -m nts.rt.view <pack_dir> [--frames N] [--build fast|compat] [--zoom 1] [--fps 30] [--exposure 0.35]
         [--start sigma=0.47,prior=0.03] [--opts history=catmull5]
     python -m nts.rt.view <pack_dir> --record out.mp4 [--size 1080x1080] [--zoom 2] [--center x,y] [--loops 2]
--record renders off screen (no window) and pipes the frames to ffmpeg, with the split sweeping across once per loop.
Data: an nts/datasets/pack.py folder (lr.npy with 12 channels, or only rgb, depth, mv x, mv y; meta.json).
"""
import json
import math
import os
import struct
import subprocess
import sys
import time

import numpy as np
import wgpu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from baselines.accumulate import arg  # noqa: E402
from nts.io_exr import Frame  # noqa: E402
from nts.rt import gpu  # noqa: E402
from nts.rt.upscaler import STRIDE, Inputs, RTUpscaler  # noqa: E402

DISPLAY = """
struct V {
  view: vec4<f32>,  // split x (window px), exposure (EV), zoom, 0
  geo: vec4<f32>,   // centre x, y (output px), window w, h
  out: vec4<f32>,   // output w, h, 0, 0
}
@group(0) @binding(0) var<uniform> U: V;
@group(0) @binding(1) var rt: texture_2d<f32>;
@group(0) @binding(2) var lr: texture_2d<f32>;
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var nearest: sampler;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn tone(c: vec3<f32>) -> vec3<f32> {
  let e = max(c, vec3<f32>(0.0)) * exp2(U.view.y);
  return pow(e / (1.0 + e), vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let q = U.geo.xy + (pos.xy - 0.5 * U.geo.zw) / U.view.z;  // output pixel under this window pixel
  let uv = q / U.out.xy;
  if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
    return vec4<f32>(0.08, 0.08, 0.08, 1.0);
  }
  if (abs(pos.x - U.view.x) < 1.0) {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
  }
  var c: vec3<f32>;
  if (pos.x < U.view.x) {
    c = textureSampleLevel(lr, lin, uv, 0.0).rgb;
  } else {
    c = textureSampleLevel(rt, nearest, uv, 0.0).rgb;
  }
  return vec4<f32>(tone(c), 1.0);
}
"""
VSTRUCT = struct.Struct("<12f")


def load(d, n=None):
    meta = json.load(open(os.path.join(d, "meta.json")))
    L = np.load(os.path.join(d, "lr.npy"), mmap_mode="r")
    ch = [0, 1, 2, 9, 10, 11] if L.shape[1] == 12 else slice(None)
    n = min(n or len(L), len(L))
    frames = []
    for i in range(n):
        a = np.asarray(L[i, ch], np.float32)
        frames.append(Frame(f=i, rgb=np.ascontiguousarray(a[:3].transpose(1, 2, 0)), depth=np.ascontiguousarray(a[3]),
                            mv=np.ascontiguousarray(a[4:6].transpose(1, 2, 0)), jitter=tuple(meta["jitter"][i]),
                            reset=bool(meta["reset"][i]) or i == 0))
    return frames, float(meta["scale"])


class Player:
    """The clip's inputs resident on the GPU, nts.rt run once per shown frame, and the split-screen display pass."""

    def __init__(self, dev, frames, scale, fmt, build="fast", **kw):
        self.dev, self.frames = dev, frames
        h, w = frames[0].depth.shape
        self.out = (round(h * scale), round(w * scale))
        self.up = RTUpscaler(scale, self.out, build=build, dev=dev, slots=len(frames), **kw)
        self.inputs = []
        for i, fr in enumerate(frames):
            inp = Inputs(self.up)
            self.up.upload(inp, fr)
            self.inputs.append(inp)
            dev.queue.write_buffer(self.up.ubuf, i * STRIDE, self.up.params(fr.jitter, fr.reset))
        self.timer = gpu.Timer(dev, 2)
        self.ubuf = dev.create_buffer(size=VSTRUCT.size, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.lin = dev.create_sampler(mag_filter="linear", min_filter="linear")
        self.nearest = dev.create_sampler()
        m = dev.create_shader_module(code=DISPLAY)
        self.pipe = dev.create_render_pipeline(
            layout="auto", vertex={"module": m, "entry_point": "vs", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            fragment={"module": m, "entry_point": "fs", "targets": [{"format": fmt}]})
        self.groups, self.f, self.ms = {}, -1, None

    def group(self, f):
        key = (f, (self.up.t - 1) % 2)
        if key not in self.groups:
            v = lambda t: t.create_view()  # noqa: E731
            self.groups[key] = self.dev.create_bind_group(layout=self.pipe.get_bind_group_layout(0), entries=[
                {"binding": 0, "resource": {"buffer": self.ubuf, "offset": 0, "size": VSTRUCT.size}},
                {"binding": 1, "resource": v(self.up.hist[key[1]])},
                {"binding": 2, "resource": v(self.inputs[f].color)},
                {"binding": 3, "resource": self.lin}, {"binding": 4, "resource": self.nearest}])
        return self.groups[key]

    def frame(self, view, target, advance=1, restart=False):
        """Advance the clip (0: show the same frame again), run nts.rt on it, draw into the texture view `target`."""
        enc = self.dev.create_command_encoder()
        if restart:
            self.f = -1
        if advance or self.f < 0:
            self.f = (self.f + advance) % len(self.frames) if self.f >= 0 else 0
            self.up.encode(enc, self.inputs[self.f], self.f, self.timer, 0)
            self.timer.resolve(enc, 2)
        self.dev.queue.write_buffer(self.ubuf, 0, VSTRUCT.pack(*view))
        p = enc.begin_render_pass(color_attachments=[{"view": target, "load_op": wgpu.LoadOp.clear,
                                                      "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 1)}])
        p.set_pipeline(self.pipe)
        p.set_bind_group(0, self.group(self.f))
        p.draw(3)
        p.end()
        self.dev.queue.submit([enc.finish()])
        if advance:
            t = self.timer.read(2)
            self.ms = None if t is None else float(t.sum())


def consts():
    kw = {}
    if arg("--start"):
        kw.update({k: float(v) for k, v in (x.split("=") for x in arg("--start").split(","))})
    if arg("--opts"):
        kw.update(dict(x.split("=") for x in arg("--opts").split(",")))
    return kw


def record(dev, frames, scale, path):
    W, H = map(int, arg("--size", "1080x1080").split("x"))
    fmt = "rgba8unorm"
    pl = Player(dev, frames, scale, fmt, build=arg("--build", "fast"), **consts())
    oh, ow = pl.out
    zoom = float(arg("--zoom", 1))
    cx, cy = map(float, arg("--center", f"{ow / 2},{oh / 2}").split(","))
    tex = dev.create_texture(size=(W, H, 1), format=fmt, usage=wgpu.TextureUsage.RENDER_ATTACHMENT |
                             wgpu.TextureUsage.COPY_SRC)
    fps = int(arg("--fps", 30))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s",
                           f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf",
                           "16", path], stdin=subprocess.PIPE)
    loops, n, ms = int(arg("--loops", 2)), len(frames), []
    for k in range(loops * n):
        split = W * (0.15 + 0.7 * (0.5 - 0.5 * math.cos(2 * math.pi * (k % n) / n)))  # sweeps across and back
        pl.frame((split, float(arg("--exposure", 0.35)), zoom, 0, cx, cy, W, H, ow, oh, 0, 0), tex.create_view())
        if pl.ms is not None:
            ms.append(pl.ms)
        raw = dev.queue.read_texture({"texture": tex}, {"bytes_per_row": W * 4}, (W, H, 1))
        ff.stdin.write(bytes(raw))
    ff.stdin.close()
    ff.wait()
    print(f"view: {loops * n} frames -> {path} ({W}x{H}, zoom {zoom:g}); nts.rt GPU time per frame, median "
          f"{np.median(ms):.3f} ms (one frame per submit, with a readback each frame)" if ms else f"-> {path}")


def window(dev, adapter, frames, scale):
    from rendercanvas.glfw import RenderCanvas, loop

    oh, ow = (round(frames[0].depth.shape[0] * scale), round(frames[0].depth.shape[1] * scale))
    fit = min(1400 / ow, 900 / oh, 1.0)
    canvas = RenderCanvas(size=(int(ow * fit), int(oh * fit)), title="nts.rt", update_mode="continuous",
                          max_fps=float(arg("--fps", 30)), vsync=False)
    ctx = canvas.get_wgpu_context()
    fmt = ctx.get_preferred_format(adapter)
    ctx.configure(device=dev, format=fmt)
    pl = Player(dev, frames, scale, fmt, build=arg("--build", "fast"), **consts())
    st = dict(split=None, zoom=float(arg("--zoom", 1)), cx=ow / 2, cy=oh / 2, pause=False, step=0, restart=False,
              t=time.perf_counter(), shown=0, ms=[])

    def on(ev):
        t = ev["event_type"]
        r = canvas.get_pixel_ratio()
        if t == "pointer_move":
            st["split"] = ev["x"] * r
        elif t == "key_down":
            k = ev["key"]
            if k == " ":
                st["pause"] = not st["pause"]
            elif k == "ArrowRight":
                st["step"] = 1
            elif k == "0":
                st.update(zoom=1.0, cx=ow / 2, cy=oh / 2)
            elif k == "r":
                st["restart"] = True
        elif t == "wheel":
            W, H = canvas.get_physical_size()
            base = min(W / ow, H / oh)  # window px per output px with the whole frame in view
            z0 = st["zoom"] * base
            z1 = min(max(z0 * math.exp(-ev["dy"] / 300), base), 16.0 * base)
            px, py = ev["x"] * r, ev["y"] * r
            qx, qy = st["cx"] + (px - W / 2) / z0, st["cy"] + (py - H / 2) / z0  # keep the pixel under the pointer
            st.update(zoom=z1 / base, cx=qx - (px - W / 2) / z1, cy=qy - (py - H / 2) / z1)

    canvas.add_event_handler(on, "pointer_move", "key_down", "wheel")

    @canvas.request_draw
    def draw():
        W, H = canvas.get_physical_size()
        split = W / 2 if st["split"] is None else st["split"]
        adv = 0 if st["pause"] else 1
        if st["step"]:
            adv, st["step"] = st["step"], 0
        z = st["zoom"] * min(W / ow, H / oh)
        pl.frame((split, float(arg("--exposure", 0.35)), z, 0, st["cx"], st["cy"], W, H, ow, oh, 0, 0),
                 ctx.get_current_texture().create_view(), advance=adv, restart=st["restart"])
        st["restart"] = False
        if adv and pl.ms is not None:
            st["ms"].append(pl.ms)
        st["shown"] += 1
        now = time.perf_counter()
        if now - st["t"] > 0.5:
            ms = f"{np.median(st['ms']):.3f} ms" if st["ms"] else "n/a"
            canvas.set_title(f"nts.rt {pl.up.build}  frame {pl.f}/{len(frames)}  GPU {ms} (prep + resolve)  "
                             f"{st['shown'] / (now - st['t']):.0f} fps shown  |  left: bilinear  right: nts.rt  "
                             f"{'(paused)' if st['pause'] else ''}")
            st.update(t=now, shown=0, ms=[])

    loop.run()


def main():
    a = sys.argv
    if len(a) < 2 or a[1].startswith("-"):
        sys.exit(__doc__)
    frames, scale = load(a[1], int(arg("--frames")) if arg("--frames") else None)
    dev, adapter_name = gpu.device(arg("--adapter"))
    if arg("--record"):
        record(dev, frames, scale, arg("--record"))
    else:
        window(dev, dev.adapter, frames, scale)


if __name__ == "__main__":
    main()
