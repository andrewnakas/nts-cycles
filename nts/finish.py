"""Camera finish: what a real camera does to a frame that a path tracer doesn't. Display-space (AgX output, sRGB
8-bit in and out), deterministic per frame, no learned parts, so it can't change who the person is.

  grade   Oklab chroma roll-off (very saturated colours compress, as on film and phone pipelines) and a global
          saturation scale; a soft toe (blacks never reach 0) and shoulder.
  lens    slight MTF softening plus halation: a wide, warm glow around bright emitters (Blender's glare node is off
          because of the 4.5 bug).
  grain   sensor noise: spatially correlated luminance grain (blob size `grain_px`), stronger in mid-tones, plus
          weaker, coarser chroma noise. A new pattern every frame (seed*100003 + frame), like a real sensor.

Grade parameters are fitted to real photos (`fit`), never to the reference frames.

Run:
  python -m nts.finish apply <in_png_dir> <out_png_dir> [key=value ...]     e.g. grain=2.5 chroma_max=0.16
  python -m nts.finish stats <png_dir | frames.npy> [...]                    chroma / tone stats of each set
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # Blender's Python on the 4050 has no scipy
    def gaussian_filter(a, sigma):
        sig = sigma if isinstance(sigma, (tuple, list)) else (sigma, sigma)
        for ax in (0, 1):
            if sig[ax] <= 0:
                continue
            r = max(1, int(4 * sig[ax] + 0.5))
            k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig[ax]) ** 2).astype(np.float32)
            k /= k.sum()
            pad = [(0, 0)] * a.ndim
            pad[ax] = (r, r)
            q = np.pad(a, pad, mode="symmetric")
            n = a.shape[ax]
            a = sum(k[i] * np.take(q, np.arange(i, i + n), axis=ax) for i in range(2 * r + 1))
        return a

DEFAULT = dict(
    sat=0.92,          # global chroma scale
    chroma_max=0.17,   # Oklab chroma soft limit (tanh)
    toe=0.005,         # black lift (display 0..1); 0.012 read milky
    shoulder=0.9,      # highlights above this roll off
    soften=0.45,       # lens blur sigma, px at 1080x1920
    halation=0.04,     # glow strength
    hal_thresh=0.8,    # luma above which emitters glow
    hal_px=18.0,       # glow radius
    grain=1.6,         # luminance grain sigma, 8-bit units at mid-grey (2.4 per frame raised face flicker 3.4 -> 4.3)
    grain_px=0.55,     # grain blob size (gaussian sigma, px)
    chroma_grain=0.0,  # chroma grain relative to luminance grain (0.35 read as colour speckle next to the reference)
    grain_hold=0.5,    # share of the previous frame's grain field carried over (0 = new grain every frame)
    seed=0,
)


# ---- colour ------------------------------------------------------------------------------------------------------
def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)


def to_oklab(rgb):  # display sRGB 0..1
    return np.cbrt(srgb_to_lin(rgb) @ M1.T) @ M2.T


def from_oklab(lab):
    lms = (lab @ np.linalg.inv(M2).T) ** 3
    return lin_to_srgb(lms @ np.linalg.inv(M1).T)


# ---- stages ------------------------------------------------------------------------------------------------------
def grade(rgb, p):
    lab = to_oklab(rgb)
    c = np.hypot(lab[..., 1], lab[..., 2]) + 1e-6
    cm = p["chroma_max"]
    scale = p["sat"] * cm * np.tanh(c / cm) / c
    lab[..., 1:] *= scale[..., None]
    out = from_oklab(lab)
    t, s = p["toe"], p["shoulder"]
    out = t + (1 - t) * out
    hi = out > s
    out[hi] = s + (1 - s) * np.tanh((out[hi] - s) / (1 - s))
    return np.clip(out, 0, 1)


def lens(rgb, p):
    out = gaussian_filter(rgb, (p["soften"], p["soften"], 0)) if p["soften"] > 0 else rgb
    if p["halation"] > 0:
        luma = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        bright = np.clip((luma - p["hal_thresh"]) / (1 - p["hal_thresh"]), 0, 1)[..., None] * rgb
        glow = gaussian_filter(bright, (p["hal_px"], p["hal_px"], 0)) * np.array([1.0, 0.75, 0.6], np.float32)
        out = 1 - (1 - out) * (1 - p["halation"] * 4 * glow)  # screen blend
    return np.clip(out, 0, 1)


def grain(rgb, p, frame):
    if p["grain"] <= 0:
        return rgb
    h, w = rgb.shape[:2]
    rngs = [np.random.default_rng(int(p["seed"]) * 100003 + f) for f in (frame, frame - 1)]
    hold = float(np.clip(p.get("grain_hold", 0.0), 0.0, 0.95))

    def field(sig):
        """Unit-variance grain; with grain_hold, a mix of this frame's and the previous frame's fields (each frame's
        own field depends only on seed and frame number, so any frame can be finished on its own)."""
        fs = []
        for rng in (rngs if hold > 0 else rngs[:1]):
            n = rng.standard_normal((h, w)).astype(np.float32)
            if sig > 0:
                n = gaussian_filter(n, sig)
                n /= n.std() + 1e-8
            fs.append(n)
        if hold > 0:
            return (fs[0] * (1 - hold) + fs[1] * hold) / float(np.hypot(1 - hold, hold))
        return fs[0]
    luma = rgb.mean(-1, keepdims=True)
    amp = (p["grain"] / 255) * np.clip(4 * luma * (1 - luma), 0.3, 1)
    out = rgb + field(p["grain_px"])[..., None] * amp
    if p["chroma_grain"] > 0:
        cg = np.stack([field(p["grain_px"] * 2.2) for _ in range(2)], -1) * p["chroma_grain"]
        out = out + np.stack([cg[..., 0], -0.5 * (cg[..., 0] + cg[..., 1]), cg[..., 1]], -1) * amp
    return np.clip(out, 0, 1)


def finish(img_u8, frame=0, **kw):
    p = {**DEFAULT, **kw}
    x = img_u8.astype(np.float32) / 255
    x = grain(lens(grade(x, p), p), p, frame)
    return (x * 255 + 0.5).astype(np.uint8)


# ---- stats -------------------------------------------------------------------------------------------------------
def stats(ims):
    """Resolution-independent look statistics: Oklab chroma percentiles, luma percentiles."""
    ch, lu = [], []
    for a in ims:
        a = np.asarray(Image.fromarray(a).resize((270, 480))) if a.shape[0] > 480 else a
        lab = to_oklab(a.astype(np.float32) / 255)
        ch.append(np.hypot(lab[..., 1], lab[..., 2]).ravel())
        lu.append(lab[..., 0].ravel())
    ch, lu = np.concatenate(ch), np.concatenate(lu)
    return {"chroma_p50": float(np.percentile(ch, 50)), "chroma_p90": float(np.percentile(ch, 90)),
            "chroma_p99": float(np.percentile(ch, 99)), "L_p1": float(np.percentile(lu, 1)),
            "L_p50": float(np.percentile(lu, 50)), "L_p99": float(np.percentile(lu, 99))}


def load_set(path):
    if path.endswith(".npy"):
        return list(np.load(path))
    return [np.asarray(Image.open(f).convert("RGB")) for f in sorted(glob.glob(os.path.join(path, "*.png")))]


def main():
    a = sys.argv
    if len(a) < 3:
        sys.exit(__doc__)
    if a[1] == "stats":
        for p in a[2:]:
            print(p, {k: round(v, 4) for k, v in stats(load_set(p)).items()})
        return
    src, out = a[2], a[3]
    kw = {k: float(v) for k, v in (x.split("=") for x in a[4:])}
    os.makedirs(out, exist_ok=True)
    for i, f in enumerate(sorted(glob.glob(os.path.join(src, "*.png")))):
        digits = "".join(c for c in os.path.basename(f) if c.isdigit())
        im = np.asarray(Image.open(f).convert("RGB"))
        Image.fromarray(finish(im, int(digits[-4:]) if digits else i, **kw)).save(os.path.join(out, os.path.basename(f)))
    print("finish ->", out, kw)


if __name__ == "__main__":
    main()
