"""Reading harness_setup / render_lr folders: f_####.exr multilayer + jitter.json + cuts.json + meta.json.

Layers: Image (RGBA scene-linear), Depth.V, Vector.XYZW, and from render_lr.py optionally Albedo.RGB, Normal.XYZ and
CryptoObject00..02 (Cryptomatte). Needs OpenImageIO (Blender's Python has it).
"""
import glob
import json
import os
import re

import numpy as np

from . import conventions as cv


def read_layers(path):
    import OpenImageIO as oiio
    inp = oiio.ImageInput.open(path)
    if not inp:
        raise RuntimeError(f"can't open {path}: {oiio.geterror()}")
    spec = inp.spec()
    px = inp.read_image(0, 0, 0, spec.nchannels, "float")
    inp.close()
    return {n: px[..., i] for i, n in enumerate(spec.channelnames)}


def stack(ch, *names):
    return np.stack([ch[n] for n in names], axis=-1)


class Frame:
    __slots__ = ("f", "rgb", "depth", "mv", "mv_next", "albedo", "normal", "jitter", "reset", "crypto")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class Sequence:
    """A rendered shot folder. Frames come back in order with image-space MV and reset flags already resolved."""

    def __init__(self, src):
        self.src = os.path.abspath(src)
        self.meta = json.load(open(os.path.join(src, "meta.json")))
        self.jit = json.load(open(os.path.join(src, "jitter.json")))["frames"]
        cp = os.path.join(src, "cuts.json")
        self.cuts = json.load(open(cp)) if os.path.exists(cp) else []
        self.files = {int(re.search(r"f_(\d+)\.exr$", p).group(1)): p for p in glob.glob(os.path.join(src, "f_*.exr"))}
        self.frames = sorted(self.files)
        self.scale = cv.scale_of(self.meta)
        self.reset = dict(zip(self.frames, cv.resets(self.frames, self.cuts)))

    def read(self, f):
        ch = read_layers(self.files[f])
        rgb = None  # slim geometry files (Vector, Depth, Region only) have no colour
        if "Image.R" in ch:
            rgb = np.nan_to_num(stack(ch, "Image.R", "Image.G", "Image.B"), nan=0.0, posinf=0.0, neginf=0.0)
        fr = Frame(f=f, rgb=rgb, depth=ch["Depth.V"], mv=cv.mv_image(stack(ch, "Vector.X", "Vector.Y")),
                   jitter=tuple(self.jit[str(f)]), reset=self.reset[f])
        if "Vector.Z" in ch:
            fr.mv_next = cv.mv_next_image(stack(ch, "Vector.Z", "Vector.W"))
        if "Albedo.R" in ch:
            fr.albedo = stack(ch, "Albedo.R", "Albedo.G", "Albedo.B")
        if "Normal.X" in ch:
            fr.normal = stack(ch, "Normal.X", "Normal.Y", "Normal.Z")
        crypto = sorted(n for n in ch if n.startswith("Crypto"))
        if crypto:
            fr.crypto = {n: ch[n] for n in crypto}
        return fr

    def __iter__(self):
        for f in self.frames:
            yield self.read(f)
