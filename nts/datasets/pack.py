"""Pack a low-res render_lr folder + its native GT folder into fp16 .npy arrays for training (runs on the 4050, numpy only).

Run: pack.py <lr_dir> <gt_dir> <out_dir> [--frames a-b] [--png-gt <png_dir>] [--masks <png_dir>] [--guide]
out_dir/
  lr.npy     (T, 12, h, w) f16: rgb linear, albedo, normal, depth distance, mv x, mv y (image space, render px)
  gt.npy     (T, 3, H, W) f16: linear ground truth
  region.npy (T, H, W) u8:  bit 1 face, 2 hands, 4 hair (from the GT's Region AOV) -- if present
  guide.npy  (T, 6, H, W) f16: full-res albedo + normal from the native render (--guide): the texture guide of
             baselines/guided.py. Both seeds of a shot share one native render, so pack it with one seed only.
  meta.json  frames, jitter (lens-shift px), reset flags, scale, source dirs
--png-gt / --masks also write the AgX GT frames and m_####.png masks the eval harness reads.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from nts.io_exr import Sequence, read_layers, stack  # noqa: E402
from nts.tonemap import Tonemapper  # noqa: E402


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    lr, gt, out = Sequence(sys.argv[1]), Sequence(sys.argv[2]), sys.argv[3]
    a = sys.argv

    def opt(n):
        return a[a.index(n) + 1] if n in a else None
    span = tuple(map(int, opt("--frames").split("-"))) if opt("--frames") else None
    frames = [f for f in lr.frames if f in gt.files and (not span or span[0] <= f <= span[1])]
    os.makedirs(out, exist_ok=True)
    ow, oh = lr.meta["out"]
    rw, rh = lr.meta["render"]
    L = np.lib.format.open_memmap(os.path.join(out, "lr.npy"), "w+", np.float16, (len(frames), 12, rh, rw))
    G = np.lib.format.open_memmap(os.path.join(out, "gt.npy"), "w+", np.float16, (len(frames), 3, oh, ow))
    R = None
    U = np.lib.format.open_memmap(os.path.join(out, "guide.npy"), "w+", np.float16,
                                  (len(frames), 6, oh, ow)) if "--guide" in a else None
    tm = Tonemapper(*gt.meta["view"]) if opt("--png-gt") else None
    for d in (opt("--png-gt"), opt("--masks")):
        if d:
            os.makedirs(d, exist_ok=True)
    for i, f in enumerate(frames):
        fr = lr.read(f)
        alb = fr.albedo if fr.albedo is not None else np.zeros_like(fr.rgb)
        nor = fr.normal if fr.normal is not None else np.zeros_like(fr.rgb)
        L[i] = np.concatenate([fr.rgb, alb, nor, np.minimum(fr.depth, 1e4)[..., None], fr.mv], -1).transpose(2, 0, 1)
        ch = read_layers(gt.files[f])
        g = np.nan_to_num(stack(ch, "Image.R", "Image.G", "Image.B"), nan=0.0, posinf=0.0, neginf=0.0)
        G[i] = np.minimum(g, 6e4).transpose(2, 0, 1)
        if U is not None:
            U[i] = np.nan_to_num(stack(ch, "Albedo.R", "Albedo.G", "Albedo.B", "Normal.X", "Normal.Y", "Normal.Z")
                                 ).transpose(2, 0, 1)
        if "Region.R" in ch:
            if R is None:
                R = np.lib.format.open_memmap(os.path.join(out, "region.npy"), "w+", np.uint8, (len(frames), oh, ow))
            reg = stack(ch, "Region.R", "Region.G", "Region.B") > 0.5
            R[i] = reg[..., 0] * 1 + reg[..., 1] * 2 + reg[..., 2] * 4
            if opt("--masks"):
                Image.fromarray((reg * 255).astype(np.uint8)).save(os.path.join(opt("--masks"), f"m_{f:04d}.png"))
        if tm:
            Image.fromarray(tm(g)).save(os.path.join(opt("--png-gt"), f"f_{f:04d}.png"))
    json.dump({"frames": frames, "jitter": [list(lr.jit[str(f)]) for f in frames],
               "reset": [bool(lr.reset[f]) for f in frames], "scale": lr.scale, "lr": lr.src, "gt": gt.src,
               "region": R is not None, "guide": U is not None}, open(os.path.join(out, "meta.json"), "w"))
    L.flush(), G.flush()
    print(f"pack: {len(frames)} frames -> {out} (lr {L.nbytes / 1e6:.0f} MB, gt {G.nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
