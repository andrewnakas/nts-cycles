"""Contact sheet for a teacher run: one row per clip (middle frame): input | raw/locked teacher variants | reference.
Run: pilot_sheet.py <pilot_dir> <tag> <out.jpg>"""
import glob
import os
import sys

from PIL import Image, ImageDraw

P, tag, out = sys.argv[1:4]
rows = []
for d in sorted(glob.glob(os.path.join(P, "*", "crop.json"))):
    d = os.path.dirname(d)
    fr = sorted(os.listdir(os.path.join(d, "in")))
    mid = fr[len(fr) // 2]
    cols = [("input", "in")] + [(os.path.basename(v).split(f"{tag}_")[1], os.path.basename(v)) for v in
            sorted(glob.glob(os.path.join(d, f"t_{tag}_s*")) + glob.glob(os.path.join(d, f"l_{tag}_s*")))]
    cols += [("reference", "ref5")] if os.path.isdir(os.path.join(d, "ref5")) else []
    ims = []
    for lab, sub in cols:
        im = Image.open(os.path.join(d, sub, mid)).convert("RGB").resize((240, 416))
        ImageDraw.Draw(im).text((4, 4), f"{os.path.basename(d)} {lab}", fill=(255, 255, 0))
        ims.append(im)
    row = Image.new("RGB", (240 * len(ims), 416))
    for i, im in enumerate(ims):
        row.paste(im, (240 * i, 0))
    rows.append(row)
sheet = Image.new("RGB", (max(r.width for r in rows), 416 * len(rows)))
for i, r in enumerate(rows):
    sheet.paste(r, (0, 416 * i))
sheet.save(out, quality=90)
