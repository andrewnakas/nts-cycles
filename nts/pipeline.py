"""NTS-F: foveated temporal upscaling of a Cycles shot, one entry point from render folders to finished frames.

Two renders of the same frames (scene-project cutscene/scene.py --nts perf and --nts roi):
  lr_dir   540x960 jittered, the whole frame, cheap.
  roi_dir  1080x1920 with only a box around the hero's head path-traced (render_lr.setup_roi), at a high sample count.
Per frame:
  1. the low-res frame is upscaled with jittered temporal accumulation (baselines/accumulate.py). With --self-guide
     its lighting is accumulated separately from its albedo, which gets a long history (albedo has no lighting noise).
  2. the head box is steadied, not upscaled: lighting (rgb / albedo) is accumulated over a few frames along the exact
     motion and multiplied back by the current frame's albedo, so texture stays exact and only render noise is averaged.
  3. the box is laid over the upscaled frame with a feathered edge, in scene-linear, then tonemapped once (AgX) and
     finished (nts/finish.py) if asked.
The face is never upscaled, so it cannot swim or change identity.

Run: python -m nts.pipeline <lr_dir> <roi_dir|none> <out_dir> [--frames a-b] [--self-guide] [--roi-wmax 3]
         [--feather 24] [--ev 0.3] [--fmt png|jpg] [--finish [key=value,...]] [--save-parts]
Needs numpy, Pillow, OpenImageIO, OCIO (Blender's Python: ~/scene-project/py on the 4050).
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from baselines.accumulate import Accumulator, arg  # noqa: E402
from nts import conventions as cv  # noqa: E402
from nts import warp as wp  # noqa: E402
from nts.io_exr import Frame, Sequence, read_layers, stack  # noqa: E402
from nts.tonemap import Tonemapper  # noqa: E402

EPS = 0.03


class Upscaler:
    """Whole-frame jittered accumulation; optionally demodulated by a self-accumulated albedo."""

    def __init__(self, seq, self_guide):
        ow, oh = seq.meta["out"]
        self.hw, self.self_guide = (oh, ow), self_guide
        self.acc = Accumulator(seq.scale, self.hw)
        # albedo: no lighting noise, so a long history and a clamp loose enough to keep sub-pixel texture
        self.alb = Accumulator(seq.scale, self.hw, wmax=32.0, expand=2.0, foot=0.5 / seq.scale) if self_guide else None

    def step(self, fr):
        if not self.self_guide or fr.albedo is None:
            return self.acc.step(fr)
        rgb, albedo = fr.rgb, fr.albedo
        fr.rgb = albedo
        alb_hr = self.alb.step(fr)
        fr.rgb = rgb / (albedo + EPS)
        shade = self.acc.step(fr)
        fr.rgb = rgb
        return np.maximum(shade, 0) * (alb_hr + EPS)


class Roi:
    """The full-resolution head box: read, steady over time, and hand back with its feather mask."""

    def __init__(self, src, wmax, feather, out_hw):
        self.src, self.wmax, self.feather, self.hw = src, wmax, feather, out_hw
        self.boxes = json.load(open(os.path.join(src, "roi.json")))["frames"]
        self.acc = self.box = self.prev_alb = None

    def mask(self, box):
        x0, y0, x1, y1 = box
        H, W = self.hw
        h, w = y1 - y0, x1 - x0
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        big = np.float32(1e6)  # sides that sit on the frame edge need no feather
        d = np.minimum(np.minimum(xs + 0.5 if x0 > 0 else big, w - xs - 0.5 if x1 < W else big),
                       np.minimum(ys + 0.5 if y0 > 0 else big, h - ys - 0.5 if y1 < H else big))
        t = np.clip(d / self.feather, 0, 1)
        return (t * t * (3 - 2 * t))[..., None]

    def step(self, f, reset):
        box = self.boxes.get(str(f))
        if box is None:
            return None
        x0, y0, x1, y1 = box
        ch = read_layers(os.path.join(self.src, f"f_{f:04d}.exr"))
        crop = lambda a: np.ascontiguousarray(a[y0:y1, x0:x1])  # noqa: E731
        rgb = np.nan_to_num(crop(stack(ch, "Image.R", "Image.G", "Image.B")), nan=0.0, posinf=0.0, neginf=0.0)
        if self.wmax <= 0 or "Albedo.R" not in ch:
            return box, rgb, self.mask(box)
        alb = crop(stack(ch, "Albedo.R", "Albedo.G", "Albedo.B"))
        fr = Frame(f=f, rgb=rgb / (alb + EPS), depth=crop(ch["Depth.V"]),
                   mv=cv.mv_image(crop(stack(ch, "Vector.X", "Vector.Y"))), jitter=(0.0, 0.0),
                   reset=reset or box != self.box)
        if fr.reset or self.acc is None:
            self.acc = Accumulator(1.0, rgb.shape[:2], wmax=self.wmax, expand=1.0, foot=0.3)
            self.prev_alb = None
        rej = None
        if self.prev_alb is not None:  # a surface whose albedo changed (lips, lids, new hair) keeps no history
            prev = wp.warp(self.prev_alb, wp.hr_motion(fr.mv, fr.depth, 1.0, rgb.shape[:2]))
            change = np.abs(prev - alb).sum(-1) / (prev.sum(-1) + alb.sum(-1) + 3 * EPS)
            tau = 0.15
            if "Region.R" in ch:
                tau = np.where(crop(ch["Region.R"]) > 0.5, 0.02, 0.15)
            rej = np.exp(-change / tau).astype(np.float32)
        shade = self.acc.step(fr, rej)
        self.box, self.prev_alb = box, alb
        return box, np.maximum(shade, 0) * (alb + EPS), self.mask(box)


def main():
    a = sys.argv
    if len(a) < 4:
        sys.exit(__doc__)
    seq = Sequence(a[1])
    ow, oh = seq.meta["out"]
    dst = a[3]
    os.makedirs(dst, exist_ok=True)
    span = tuple(map(int, arg("--frames").split("-"))) if arg("--frames") else None
    fmt = arg("--fmt", "png")
    up = Upscaler(seq, "--self-guide" in a)
    roi = None if a[2] == "none" else Roi(a[2], float(arg("--roi-wmax", 3)), float(arg("--feather", 24)), (oh, ow))
    view = list(seq.meta["view"])
    view[2] += float(arg("--ev", 0.3))
    tm = Tonemapper(*view)
    fin = None
    if "--finish" in a:
        from nts.finish import finish
        nxt = arg("--finish", "")
        kw = {k: float(v) for k, v in (x.split("=") for x in nxt.split(",") if "=" in x)} if nxt and "=" in nxt else {}
        fin = lambda im, f: finish(im, frame=f, **kw)  # noqa: E731
    n, t_up, t_roi = 0, 0.0, 0.0
    for f in seq.frames:
        if span and not span[0] <= f <= span[1]:
            continue
        t0 = time.time()
        fr = seq.read(f)
        out = up.step(fr)
        t1 = time.time()
        if roi is not None:
            r = roi.step(f, fr.reset)
            if r is not None:
                (x0, y0, x1, y1), img, m = r
                if "--save-parts" in a:
                    Image.fromarray(tm(out)).save(os.path.join(dst, f"up_{f:04d}.png"))
                out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - m) + img * m
        t_up, t_roi = t_up + t1 - t0, t_roi + time.time() - t1
        im = tm(out)
        if fin:
            im = fin(im, f)
        Image.fromarray(im).save(os.path.join(dst, f"f_{f:04d}.{fmt}"), quality=95)
        n += 1
    print(f"pipeline: {n} frames -> {dst}; upscale {t_up / max(n, 1):.2f} s/frame, roi {t_roi / max(n, 1):.2f} s/frame")


if __name__ == "__main__":
    main()
