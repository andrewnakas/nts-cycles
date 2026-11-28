"""Zero-training baseline for the generative realism pass: GFPGAN (Apache-2.0 code and weights; trained on FFHQ, whose
dataset licence is non-commercial: provenance flagged, see memory genmodel-research) on every frame of a clip.

Run: gfpgan_clip.py <frames_dir> <out_dir> [--weight 0.5] [--frames a-b] [--version 1.4]
Each face is detected, aligned to 512, restored with the StyleGAN2 face prior, and pasted back at the frame's own
resolution. --weight blends restored and input features (lower = closer to the input face).
"""
import glob
import os
import sys

import cv2
from gfpgan import GFPGANer

URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
a = sys.argv
opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
src, dst, w = a[1], a[2], float(opt("--weight", 0.5))
span = tuple(map(int, opt("--frames", "0-99999").split("-")))
os.makedirs(dst, exist_ok=True)
r = GFPGANer(model_path=URL, upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None)
for p in sorted(glob.glob(os.path.join(src, "f_*.png"))):
    f = int(os.path.basename(p)[2:6])
    if not span[0] <= f <= span[1]:
        continue
    img = cv2.imread(p)
    _, faces, out = r.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, weight=w)
    cv2.imwrite(os.path.join(dst, os.path.basename(p)), out if out is not None else img)
    print(f, len(faces), flush=True)
