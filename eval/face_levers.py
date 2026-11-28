"""Which face changes move the realism probe? Image-space ablations on the native face crops of the 24 variety stills
(eval only; used to choose render levers, never as a finishing step or training target). Same face probe as
eval/variety.py (DINOv2 + CLIP, photos vs stage-A CG faces, grain-invariant).
Levers, each applied inside the face mask only:
  dark      skin luma x0.75
  contrast  local contrast (unsharp, sigma 6 px) x1.6: deeper sockets / cheek modelling
  mottle    1-3 px multiplicative colour noise, 6 %
  stubble   lower-face (below the nose line) darkening 12 % with 1 px speckle: beard shadow
  eyes      eye-region saturation x0.6 and brightness x0.85
  all       everything above
Run: python -m eval.face_levers data/look/refs runs/face_levers
"""
import glob
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from eval import scoreboard as S
from eval.face import load_mask
from eval.variety import Clip

refs, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
dino, clip = S.Dino(), Clip()
photos = np.load("runs/ntsr_a/photos_faces.npy", mmap_mode="r")
keep = np.load("runs/ntsr_a/photos_faces_keep.npy")
z = np.load("runs/variety2_feats.npz")
probes = {bb: S.probe(z[f"{bb}_face_r"], np.concatenate([z[f"{bb}_face_c"], z[f"{bb}_face_g"]]))[0] for bb in ("dino", "clip")}
feat = {"dino": lambda c: dino(c)[0], "clip": clip}


def levers(im, face, hy):
    f = face[..., None].astype(np.float32)
    f = gaussian_filter(f, (2, 2, 0))
    x = im.astype(np.float32) / 255
    h, w = face.shape
    yy = np.arange(h)[:, None, None]
    out = {}
    out["dark"] = x * (1 - 0.25 * f)
    blur = gaussian_filter(x, (6, 6, 0))
    out["contrast"] = x + 0.6 * f * (x - blur)
    n = gaussian_filter(rng.standard_normal((h, w, 3)).astype(np.float32), (1, 1, 0))
    n /= n.std()
    out["mottle"] = x * (1 + 0.06 * f * n)
    low = f * (yy > hy)
    sp = gaussian_filter(rng.standard_normal((h, w, 1)).astype(np.float32), 0.6)
    sp /= sp.std()
    out["stubble"] = x * (1 - low * (0.12 + 0.05 * sp))
    eye = f * ((yy > hy - 0.45 * (hy - np.nonzero(face)[0].min())) & (yy < hy - 0.15 * (hy - np.nonzero(face)[0].min())))
    lum = x.mean(-1, keepdims=True)
    out["eyes"] = x * (1 - eye) + eye * 0.85 * (lum + 0.6 * (x - lum))
    a = out["dark"]
    a = a + 0.6 * f * (a - gaussian_filter(a, (6, 6, 0)))
    a = a * (1 + 0.06 * f * n) * (1 - low * (0.12 + 0.05 * sp))
    a = a * (1 - eye) + eye * 0.85 * (a.mean(-1, keepdims=True) + 0.6 * (a - a.mean(-1, keepdims=True)))
    out["all"] = a
    return {k: (np.clip(v, 0, 1) * 255).astype(np.uint8) for k, v in out.items()}


res = {}
for nat in sorted(glob.glob(os.path.join(refs, "native_*.png"))):
    key = os.path.basename(nat)[7:-4]
    ep, fr = key.rsplit("_f", 1)
    face = load_mask(os.path.join(refs, f"m_{ep}_{fr}.png"))[0]
    if face.sum() < 50:
        continue
    im = np.asarray(Image.open(nat).convert("RGB"))
    ys, xs = np.nonzero(face)
    c = ((xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2)
    b = S.haar_box(im, c)
    if b is None:
        s = int(0.75 * max(np.ptp(xs), np.ptp(ys)))
        b = (c[0] - s // 2, c[1] - s // 2, s, s)
    hy = ys.min() + 0.62 * np.ptp(ys)  # about the nose base
    vars_ = {"native": im, **levers(im, face, hy), "reference": np.asarray(Image.open(os.path.join(refs, f"ref_{key}.png")).convert("RGB"))}
    for k, v in vars_.items():
        cr = S.crop(v, b)[None]
        res.setdefault(k, []).append([float(probes[bb](feat[bb](cr))[0]) for bb in ("dino", "clip")])
    print(key, flush=True)
lines = ["| variant | face P(real) DINOv2 / CLIP |", "|---|---|"]
for k, v in res.items():
    m = np.mean(v, 0)
    lines.append(f"| {k} | {m[0]:.3f} / {m[1]:.3f} |")
open(out + ".md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
