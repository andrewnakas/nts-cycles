"""Upscale a render folder on the GPU (nts.rt) into the frames the eval harness reads, as baselines/accumulate.py does.

Run: python -m nts.rt.run <lr_dir> <out_dir> [--build compat|fast] [--frames a-b] [--fmt png] [--wmax w]
         [--expand e] [--prior w0] [--sigma s] [--shrink k] [--rel r] [--history catmull5|catmull|bilinear]
         (constants default to conventions.ACCUM; sigma to conventions.accum_sigma(scale))
         [--adapter NAME] [--compare] [--linear-npy] [--guide <native_dir> [--eps 0.03] [--reject 0.15]]
--self-guide [--alb-wmax 32 --alb-expand 2 --alb-sigma 0.5]: the render's own albedo accumulated as a guide.
--guide: guided mode, with the display-size albedo from that folder's EXRs (Albedo layer) and the render's own albedo.
--compare also runs baselines/accumulate.py Accumulator on the same frames and prints the PSNR of the GPU frame
against it (linear, peak = the frame's maximum). Needs OpenImageIO and OCIO (Blender's Python) plus wgpu.
"""
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from baselines.accumulate import Accumulator, arg  # noqa: E402
from nts.io_exr import Sequence  # noqa: E402
from nts.rt import gpu  # noqa: E402
from nts.rt.upscaler import FITTED, RTUpscaler, SelfGuided  # noqa: E402
from nts.tonemap import Tonemapper  # noqa: E402


def psnr(a, ref):
    mse = float(np.mean((a.astype(np.float64) - ref) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(float(ref.max()) ** 2 / mse)


def main():
    a = sys.argv
    if len(a) < 3:
        sys.exit(__doc__)
    seq, dst = Sequence(a[1]), a[2]
    os.makedirs(dst, exist_ok=True)
    span = tuple(map(int, arg("--frames").split("-"))) if arg("--frames") else None
    fmt = arg("--fmt", "png")
    ow, oh = seq.meta["out"]
    kw = dict(wmax=float(arg("--wmax", FITTED["wmax"])), expand=float(arg("--expand", FITTED["expand"])),
              sigma=float(arg("--sigma")) if arg("--sigma") else None, prior=float(arg("--prior", FITTED["prior"])),
              history=arg("--history", "catmull5"))
    tune = dict(shrink=float(arg("--shrink", FITTED["shrink"])), rel=float(arg("--rel", FITTED["rel"])))
    guide = arg("--guide")
    if guide:
        from baselines.guided import read_guide
        kw.update(guided=True, eps=float(arg("--eps", 0.03)), reject=float(arg("--reject", 0.15)))
    dev, name = gpu.device(arg("--adapter"))
    build = arg("--build", "compat")
    selfg = "--self-guide" in a
    if selfg:  # the render's own albedo only: an albedo history feeds the guided upscaler
        kw.update(eps=float(arg("--eps", 0.03)))
        up = SelfGuided(seq.scale, (oh, ow), alb_wmax=float(arg("--alb-wmax", 32)),
                        alb_expand=float(arg("--alb-expand", 2)), alb_sigma=float(arg("--alb-sigma", 0.5)),
                        build=build, dev=dev, **kw, **tune)
    else:
        up = RTUpscaler(seq.scale, (oh, ow), build=build, dev=dev, **kw, **tune)
    if "--compare" in a and not (guide or selfg):
        ref = Accumulator(seq.scale, (oh, ow), **kw, **tune)
    else:
        ref = None
    timer = gpu.Timer(dev, 2)
    tm = Tonemapper(*seq.meta["view"])
    ms, dbs, t_all = [], [], time.time()
    for f in seq.frames:
        if span and not span[0] <= f <= span[1]:
            continue
        fr = seq.read(f)
        if guide:
            out = up.step(fr, fr.albedo, read_guide(guide, f)[0], timer=timer)
        elif selfg:
            out = up.step(fr, timer=timer)  # the timer covers the shading upscaler only
        else:
            out = up.step(fr, timer=timer)
        t = timer.read()
        if t is not None:
            ms.append(t)
        if ref is not None:
            dbs.append(psnr(out, ref.step(fr)))
        Image.fromarray(tm(out)).save(os.path.join(dst, f"f_{f:04d}.{fmt}"), quality=92)
        if "--linear-npy" in a:
            np.save(os.path.join(dst, f"f_{f:04d}.npy"), out.astype(np.float16))
    n = len(ms) if ms else len(dbs)
    u = up.up if selfg else up
    print(f"rt.run: {u.build} build on {name}, {seq.scale:g}x {u.in_hw[1]}x{u.in_hw[0]} -> {ow}x{oh}, "
          f"frames -> {dst} ({time.time() - t_all:.1f} s wall with file I/O)")
    if ms:
        m = np.array(ms)
        print(f"  GPU per frame (one frame per submit, median): prep {np.median(m[:, 0]):.3f} ms, "
              f"resolve {np.median(m[:, 1]):.3f} ms, total {np.median(m.sum(1)):.3f} ms (use nts.rt.bench for timing)")
    if dbs:
        print(f"  vs Accumulator: min {min(dbs):.1f} dB, median {np.median(dbs):.1f} dB over {n} frames")


if __name__ == "__main__":
    main()
