"""Side-by-side comparison of upscaler variants on one shot: labelled full-frame and face-zoom videos (ffmpeg) plus
labelled still crops for an HTML page. Every variant must have f_####.png for the same frames.

Run: side_by_side.py <out_dir> <a-b> <face x0,y0,x1,y1> "Label=dir" ["Label=dir" ...]
"""
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def label(img, text, size):
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, size)
    d.rectangle([0, 0, img.width, size + 12], fill=(0, 0, 0))
    d.text((8, 5), text, fill=(255, 255, 255), font=f)
    return img


def main():
    out, span, box = sys.argv[1], sys.argv[2], tuple(map(int, sys.argv[3].split(",")))
    vs = [a.split("=", 1) for a in sys.argv[4:]]
    a, b = map(int, span.split("-"))
    os.makedirs(os.path.join(out, "full"), exist_ok=True)
    os.makedirs(os.path.join(out, "face"), exist_ok=True)
    n = len(vs)
    cols = 3 if n > 3 else n
    rows = (n + cols - 1) // cols
    for f in range(a, b + 1):
        ims = [Image.open(os.path.join(d, f"f_{f:04d}.png")).convert("RGB") for _, d in vs]
        full = Image.new("RGB", (360 * n, 640))
        face = Image.new("RGB", (600 * cols, 600 * rows))
        for i, ((name, _), im) in enumerate(zip(vs, ims)):
            full.paste(label(im.resize((360, 640), Image.LANCZOS), name, 15), (360 * i, 0))
            fz = im.crop(box).resize((600, 600), Image.NEAREST)  # nearest: show real pixels, no smoothing
            face.paste(label(fz, name, 22), (600 * (i % cols), 600 * (i // cols)))
        full.save(os.path.join(out, "full", f"{f:04d}.png"))
        face.save(os.path.join(out, "face", f"{f:04d}.png"))
    for kind in ("full", "face"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-start_number", str(a), "-i",
                        os.path.join(out, kind, "%04d.png"), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                        os.path.join(out, f"{kind}.mp4")], check=True)
    print(f"side_by_side: {b - a + 1} frames, {n} variants -> {out}/full.mp4, face.mp4")


if __name__ == "__main__":
    main()
