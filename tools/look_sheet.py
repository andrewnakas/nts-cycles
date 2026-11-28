"""Contact sheet for a look-search round: native Cycles, the visual reference, and the top-N candidates (face crop).

Run: look_sheet.py <round_dir> <native.png> <reference.png> <out.jpg> [--top 4] [--frame f####] [--crop x0,y0,x1,y1]
Uses <round_dir>/ranked.json (single frame) or ranked_multi.json (with --frame picking which frame to show).
"""
import json
import os
import sys

from PIL import Image, ImageDraw


def main():
    d, native, ref, out_path = sys.argv[1:5]
    a = sys.argv
    top = int(a[a.index("--top") + 1]) if "--top" in a else 4
    frame = a[a.index("--frame") + 1] if "--frame" in a else None
    box = tuple(map(int, a[a.index("--crop") + 1].split(","))) if "--crop" in a else (220, 440, 860, 1500)
    ranked = os.path.join(d, "ranked_multi.json" if frame else "ranked.json")
    r = json.load(open(ranked))
    ims = [("native Cycles", native), ("reference (reference pass 2)", ref)]
    for i, x in enumerate(r[:top]):
        n, s = x["cand"]["name"], x["score"]
        p = os.path.join(d, f"{n}_{frame}.png" if frame else f"{n}.png")
        ims.append((f"#{i + 1} {n}  dE {s['total']:.2f}  head {s['head']:.1f}  hands {s['hands']:.1f}", p))
    w, h = box[2] - box[0], box[3] - box[1]
    cols = 3
    rows = (len(ims) + cols - 1) // cols
    out = Image.new("RGB", (w * cols, (h + 28) * rows), (16, 16, 16))
    dr = ImageDraw.Draw(out)
    for i, (lab, p) in enumerate(ims):
        im = Image.open(p).convert("RGB").resize((1080, 1920), Image.BICUBIC)
        x, y = (i % cols) * w, (i // cols) * (h + 28)
        out.paste(im.crop(box), (x, y + 28))
        dr.text((x + 8, y + 8), lab, fill=(240, 240, 240))
    out.save(out_path, quality=92)


if __name__ == "__main__":
    main()
