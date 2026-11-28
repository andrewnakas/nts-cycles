#!/usr/bin/env python3
"""Side-by-side comparison panels for the README: whole frame (scaled) above, the head at 1:1 below.

  python tools/gallery.py <out.jpg> <mask.png|-> <label>=<frame.png> [<label>=<frame.png> ...] [--width 360]
         [--crop 360] [--title "..."] [--quality 88]

Each frame is a 1080x1920 PNG of the same moment. The head crop is centred on the face and hair of the renderer's
Region mask (R face, B hair); with "-" it is the upper middle of the frame. Labels are drawn under each column.
Only reads the frames given; writes one JPEG.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf")


def font(size):
    for f in FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def head_box(mask_path, hw, side):
    h, w = hw
    if mask_path != "-" and os.path.exists(mask_path):
        m = np.asarray(Image.open(mask_path).convert("RGB"))
        ys, xs = np.nonzero((m[..., 0] > 127) | (m[..., 2] > 127))
        if len(ys) > 50:
            cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
            s = int(min(max(side, 1.35 * max(ys.max() - ys.min(), xs.max() - xs.min())), w, h))
            x0, y0 = int(np.clip(cx - s // 2, 0, w - s)), int(np.clip(cy - s // 2, 0, h - s))
            return x0, y0, x0 + s, y0 + s
    s = min(side, w)
    return (w - s) // 2, h // 6, (w - s) // 2 + s, h // 6 + s


def main():
    a = sys.argv[1:]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or not a[i - 1].startswith("--"))]
    if len(pos) < 3:
        sys.exit(__doc__)
    out, mask, cols = pos[0], pos[1], [c.split("=", 1) for c in pos[2:]]
    W, C, q = int(opt("--width", 360)), int(opt("--crop", 360)), int(opt("--quality", 88))
    title = opt("--title", "")
    ims = [Image.open(p).convert("RGB") for _, p in cols]
    fw, fh = ims[0].size
    box = head_box(mask, (fh, fw), C)
    H = int(round(W * fh / fw))
    pad, lab = 6, 34
    top = 40 if title else 0
    sheet = Image.new("RGB", (len(ims) * W + (len(ims) - 1) * pad, top + H + pad + W + lab), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    f1, f2 = font(20), font(22)
    if title:
        d.text((10, 9), title, fill=(235, 235, 235), font=f2)
    for i, ((name, _), im) in enumerate(zip(cols, ims)):
        x = i * (W + pad)
        sheet.paste(im.resize((W, H), Image.LANCZOS), (x, top))
        sheet.paste(im.crop(box).resize((W, W), Image.LANCZOS), (x, top + H + pad))
        tw = d.textlength(name, font=f1)
        d.text((x + (W - tw) / 2, top + H + pad + W + 6), name, fill=(235, 235, 235), font=f1)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sheet.save(out, quality=q, optimize=True, progressive=True)
    print(f"{out}: {sheet.size[0]}x{sheet.size[1]}, head box {box}, {os.path.getsize(out) // 1024} KB")


if __name__ == "__main__":
    main()
