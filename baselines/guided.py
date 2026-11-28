"""Albedo-guided temporal upscaler (no network). Texture comes from a full-res guide render, never from a guess.

Offline we can render what a game can't afford: a full-res albedo/normal/depth pass. Shading (radiance / albedo) is
smooth, so upscale only that with the temporal accumulator, then multiply by the full-res albedo:
    shade_lr = rgb_lr / (albedo_lr + eps)
    shade_hr = Accumulator(shade_lr)               jittered samples + reprojected, clipped history (baselines/accumulate.py)
    out      = shade_hr * (albedo_hr + eps)
Pores, fabric weave and screen text then come from the albedo render (exact), and the network-free accumulator only
has to carry lighting. Emission and glossy reflections aren't albedo-modulated, so eps keeps dark-albedo pixels sane.

The guide here is a folder of full-res EXRs with Albedo (and Depth/Normal) layers: the native render's own passes for
the prototype (an upper bound on guide quality), later a cheap few-sample guide render.

Edge fallback (--fallback tau): at hair, lashes and silhouettes a low-res pixel's albedo mixes several surfaces, so
dividing by it is wrong and the result grids. There, compare the low-res albedo with the full-res albedo box-filtered
to the low-res grid; where they disagree, blend toward a plain accumulator run on the colour itself:
    w = clip(1 - mismatch / tau, 0, 1),   out = w * guided + (1 - w) * plain

Skin blur (--skin-blur r): subsurface scattering blurs the skin's albedo texture in the real render, so on skin
(Region face + hands) multiply by the full-res albedo box-blurred over r px instead of the sharp one.

Guide TAA (--guide-taa a): a few-sample guide render is noisy at texture and edge detail, but albedo is fixed to the
surfaces, so accumulate the guide itself over time: warp the previous accumulated guide with the motion vectors, clamp
it to the current guide's 3x3 neighbourhood range (so moving edges can't ghost), and blend a of the new frame in.
Disocclusions (depth test) restart from the current frame.

History rejection (--reject tau): offline we know the full-res albedo of every frame, so warp the previous frame's
albedo with the motion vectors and compare: where they disagree (an eyelid closed, teeth appeared, a hand passed),
the shading history belongs to a different surface. Its weight is scaled by exp(-change / tau).

Measured (2026-11-27, zero-day 400-459, guide = native passes): --reject 0.15 with the accumulator's defaults
(conventions.ACCUM), run as nts.rt.run --guide (the GPU port of GuidedAccumulator, tested equal to it):
  PSNR 40.50 / tErr 0.393 / static flicker 0.206 / face PSNR 33.21.
  --reject-face has not been measured with the current accumulator.
  Tried and dropped: normal-map detail regression, albedo-mismatch fallback, skin-albedo blur.

Run: guided.py <lr_dir> <guide_dir> <out_dir> [--eps 0.03] [--frames a-b] [--foot f] [--expand e] [--wmax w]
               [--fallback 0.15] [--skin-blur 2] [--reject 0.1] [--reject-face 0.03] [--guide-taa 0.2]
               [--lr-albedo <dir>]   low-res albedo from another render (e.g. an Eevee guide at the input's jitter),
                                     so demodulation and remodulation use the same renderer's albedo
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from baselines.accumulate import Accumulator, arg  # noqa: E402
from eval.metrics import box  # noqa: E402
from nts import conventions as cv  # noqa: E402
from nts import warp as wp  # noqa: E402
from nts.io_exr import Frame, Sequence, read_layers, stack  # noqa: E402
from nts.tonemap import Tonemapper  # noqa: E402


class GuidedAccumulator:
    """The guided upscaler's core as one class (what nts/rt runs in guided mode and its test checks against): shading
    = colour / (low-res albedo + eps) accumulated by Accumulator, history weight scaled by exp(-change / reject) where
    the full-res albedo warped from the previous frame disagrees with this frame's, output = max(shading, 0) x
    (full-res albedo + eps). No fallback, skin blur or guide TAA."""

    def __init__(self, scale, out_hw, eps=0.03, reject=None, **acc_kw):
        self.acc, self.scale, self.hw = Accumulator(scale, out_hw, **acc_kw), scale, out_hw
        self.eps, self.reject, self.prev = np.float32(eps), reject, None

    def step(self, fr, alb_lr, alb_hr):
        eps = self.eps
        rej = None
        if self.reject and self.prev is not None and not fr.reset:
            mv = wp.hr_motion(fr.mv * self.acc.mv_sign, fr.depth, self.scale, self.hw)
            prev = wp.warp(self.prev, mv)
            change = np.abs(prev - alb_hr).sum(-1) / (prev.sum(-1) + alb_hr.sum(-1) + 3 * eps)
            rej = np.exp(-change / np.float32(self.reject)).astype(np.float32)
        self.prev = alb_hr
        shaded = Frame(f=fr.f, rgb=(fr.rgb / (alb_lr + eps)).astype(np.float32), depth=fr.depth, mv=fr.mv,
                       jitter=fr.jitter, reset=fr.reset)
        return np.maximum(self.acc.step(shaded, rej), 0) * (alb_hr + eps)


def read_guide(guide_dir, f):
    ch = read_layers(os.path.join(guide_dir, f"f_{f:04d}.exr"))
    reg = stack(ch, "Region.R", "Region.G", "Region.B") if "Region.R" in ch else None
    return np.nan_to_num(stack(ch, "Albedo.R", "Albedo.G", "Albedo.B")), reg


def boxc(a, r):
    return np.stack([box(a[..., k], r) for k in range(a.shape[-1])], -1).astype(np.float32)


def nbr_range(a):
    """Per-pixel min and max over the 3x3 neighbourhood (edges clamped)."""
    h, w = a.shape[:2]
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    win = np.stack([p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    return win.min(0), win.max(0)


def mismatch(alb_lr, alb_hr, scale, eps):
    """Relative disagreement between the low-res albedo and the full-res albedo averaged over each low-res footprint."""
    k = int(round(scale))
    h, w = alb_lr.shape[:2]
    down = alb_hr[:h * k, :w * k].reshape(h, k, w, k, 3).mean((1, 3))
    return (np.abs(alb_lr - down).sum(-1) / (alb_lr.sum(-1) + down.sum(-1) + 3 * eps)).astype(np.float32)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    seq = Sequence(sys.argv[1])
    guide, dst = sys.argv[2], sys.argv[3]
    os.makedirs(dst, exist_ok=True)
    eps = float(arg("--eps", 0.03))
    span = tuple(map(int, arg("--frames").split("-"))) if arg("--frames") else None
    ow, oh = seq.meta["out"]
    num = {k: float(arg("--" + k, cv.ACCUM[k])) for k in ("expand", "wmax")}
    num["foot"] = float(arg("--foot", cv.ACCUM["foot"]))  # render px
    if arg("--sigma"):  # deprecated: 0.2.0's footprint in output px
        num["foot"] = float(arg("--sigma")) / seq.scale
    acc = Accumulator(seq.scale, (oh, ow), **num)
    tau = float(arg("--fallback")) if arg("--fallback") else None
    skin_blur = int(arg("--skin-blur", 0))
    rtau = float(arg("--reject")) if arg("--reject") else None
    rtau_face = float(arg("--reject-face")) if arg("--reject-face") else None
    prev_alb = None
    gtaa = float(arg("--guide-taa")) if arg("--guide-taa") else None
    lr_alb = arg("--lr-albedo")
    g_acc = g_depth = None
    plain = Accumulator(seq.scale, (oh, ow), **num) if tau else None
    tm = Tonemapper(*seq.meta["view"])
    n = 0
    for f in seq.frames:
        if span and not span[0] <= f <= span[1]:
            continue
        fr = seq.read(f)
        if lr_alb:
            fr.albedo = read_guide(lr_alb, f)[0]
        rgb = fr.rgb
        fr.rgb = rgb / (fr.albedo + eps)
        alb_hr, reg = read_guide(guide, f)
        if gtaa:
            depth_hr = wp.upsample_nearest(fr.depth, seq.scale, (oh, ow))
            if g_acc is not None and not fr.reset:
                mvh = wp.hr_motion(fr.mv, fr.depth, seq.scale, (oh, ow))
                hist = wp.warp(g_acc, mvh)
                lo, hi = nbr_range(alb_hr)
                hist = np.clip(hist, lo, hi)
                fresh = wp.disocclusion(g_depth, depth_hr, mvh)[..., None]
                alb_hr = np.where(fresh, alb_hr, (1 - gtaa) * hist + gtaa * alb_hr)
            g_acc, g_depth = alb_hr, depth_hr
        rej = None
        if rtau and prev_alb is not None and not fr.reset:
            mv = wp.hr_motion(fr.mv, fr.depth, seq.scale, (oh, ow))
            prev = wp.warp(prev_alb, mv)
            change = np.abs(prev - alb_hr).sum(-1) / (prev.sum(-1) + alb_hr.sum(-1) + 3 * eps)
            rt = rtau
            if rtau_face and reg is not None:  # faces talk and blink: be stricter there
                rt = np.where(reg[..., 0] > 0.5, rtau_face, rtau)
            rej = np.exp(-change / rt).astype(np.float32)
        prev_alb = alb_hr
        shade = acc.step(fr, rej)
        if skin_blur and reg is not None:
            skin = np.clip(reg[..., 0] + reg[..., 1], 0, 1)[..., None]
            alb_hr = alb_hr * (1 - skin) + boxc(alb_hr, skin_blur) * skin
        out = np.maximum(shade, 0) * (alb_hr + eps)
        if tau:
            fr.rgb = rgb
            w = np.clip(1 - wp.upsample_bilinear(mismatch(fr.albedo, alb_hr, seq.scale, eps)[..., None], seq.scale,
                                                 (oh, ow)) / tau, 0, 1)
            out = w * out + (1 - w) * plain.step(fr)
        Image.fromarray(tm(out)).save(os.path.join(dst, f"f_{f:04d}.png"))
        n += 1
    print(f"guided: {n} frames -> {dst}")


if __name__ == "__main__":
    main()
