"""Compact G-buffer images for the NR-1 teacher's G-buffer adapter and the NR-1 student (2026-11-18).

From one render_lr EXR (layers Albedo.RGB = Cycles Denoising Albedo, Normal.XYZ = Denoising Normal, Depth.V):
  <prefix>.alb.jpg   albedo, linear -> sRGB, 8 bit (q 93)
  <prefix>.nrm.png   normal * 0.5 + 0.5, 8 bit. Cycles writes the denoising normal in camera space (to be confirmed on the
                     first render: surfaces facing the camera should read blue, z near +1)
  <prefix>.dep.png   16 bit, log depth: v = log2(z / Z0) / log2(Z1 / Z0), Z0 = 0.1 m, Z1 = 100 m (0 = background/none)
About 1.5 MB per 1080x1920 frame instead of ~10 MB of extra EXR layers. load_gbuf() inverts all three.
"""
import numpy as np
from PIL import Image

Z0, Z1 = 0.1, 100.0


def srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def save_gbuf(ch, prefix):
    """ch: dict of EXR channels (nts.io_exr.read_layers). Returns the list of files written (empty if no G-buffer)."""
    out = []
    if "Albedo.R" in ch:
        a = np.stack([ch["Albedo.R"], ch["Albedo.G"], ch["Albedo.B"]], -1).astype(np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        Image.fromarray((srgb(a) * 255 + 0.5).astype(np.uint8)).save(prefix + ".alb.jpg", quality=93)
        out.append(prefix + ".alb.jpg")
    if "Normal.X" in ch:
        n = np.stack([ch["Normal.X"], ch["Normal.Y"], ch["Normal.Z"]], -1).astype(np.float32)
        n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
        Image.fromarray((np.clip(n * 0.5 + 0.5, 0, 1) * 255 + 0.5).astype(np.uint8)).save(prefix + ".nrm.png")
        out.append(prefix + ".nrm.png")
    dk = "Depth.V" if "Depth.V" in ch else ("Depth.Z" if "Depth.Z" in ch else None)
    if dk:
        z = np.nan_to_num(ch[dk].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        ok = (z > 0) & (z < 1e9)
        v = np.zeros(z.shape, np.float64)
        v[ok] = np.clip(np.log2(np.maximum(z[ok], Z0) / Z0) / np.log2(Z1 / Z0), 1 / 65535, 1.0)
        Image.fromarray((v * 65535 + 0.5).astype(np.uint16)).save(prefix + ".dep.png")
        out.append(prefix + ".dep.png")
    return out


def load_gbuf(prefix):
    """-> albedo (H,W,3) sRGB 0..1, normal (H,W,3) -1..1, depth (H,W) metres (0 = none); None for a missing file."""
    import os
    alb = nrm = dep = None
    if os.path.exists(prefix + ".alb.jpg"):
        alb = np.asarray(Image.open(prefix + ".alb.jpg").convert("RGB"), np.float32) / 255
    if os.path.exists(prefix + ".nrm.png"):
        nrm = np.asarray(Image.open(prefix + ".nrm.png").convert("RGB"), np.float32) / 255 * 2 - 1
    if os.path.exists(prefix + ".dep.png"):
        v = np.asarray(Image.open(prefix + ".dep.png"), np.float64) / 65535
        dep = np.where(v > 0, Z0 * np.power(Z1 / Z0, v), 0.0).astype(np.float32)
    return alb, nrm, dep
