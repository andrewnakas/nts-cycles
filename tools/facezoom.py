"""Option 1 (2026-11-12): zoomed-face detail teacher. SDXL can paint skin mottling, pores and hair strands only when
the face is large; ours is ~90 px. prep: per frame, the face window (2.1x the Haar box, as the scoreboard cuts it)
upscaled to 1024 px -> <out>/zoom/f_####.jpg + boxes.json. merge: the teacher's 1024 px faces downscaled back into
the frame, blended through a soft face+hair mask (screens, text and background stay the input).
Run (Mac):  facezoom.py prep  <frames_dir> <masks_dir> <out_dir>
            facezoom.py merge <frames_dir> <masks_dir> <out_dir> <teacher_dir> <merged_dir>"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval.face import load_mask  # noqa: E402
from eval.scoreboard import haar_box  # noqa: E402

mode, fdir, mdir, out = sys.argv[1:5]
frames = sorted(f for f in os.listdir(fdir) if f.startswith("f_") and f.endswith(".png"))
if mode == "prep":
    os.makedirs(os.path.join(out, "zoom"), exist_ok=True)
    boxes = {}
    for f in frames:
        im = np.asarray(Image.open(os.path.join(fdir, f)).convert("RGB"))
        ys, xs = np.nonzero(load_mask(os.path.join(mdir, "m_" + f[2:]))[0])
        x, y, w, h = haar_box(im, ((xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2))
        s = int(max(w, h) * 2.1 / 2)
        cx, cy = x + w // 2, y + h // 2
        boxes[f] = [int(cx - s), int(cy - s), int(2 * s)]
        Image.fromarray(im[cy - s:cy + s, cx - s:cx + s]).resize((1024, 1024), Image.LANCZOS).save(
            os.path.join(out, "zoom", f[:-4] + ".jpg"), quality=97)
    json.dump(boxes, open(os.path.join(out, "boxes.json"), "w"))
    print("facezoom prep:", len(boxes), "faces, window", next(iter(boxes.values()))[2], "px -> 1024")
else:
    tdir, mdir_out = sys.argv[5], sys.argv[6]
    os.makedirs(mdir_out, exist_ok=True)
    boxes = json.load(open(os.path.join(out, "boxes.json")))
    for f in frames:
        im = np.asarray(Image.open(os.path.join(fdir, f)).convert("RGB")).astype(np.float32)
        x0, y0, s = boxes[f]
        t = np.asarray(Image.open(os.path.join(tdir, f)).convert("RGB").resize((s, s), Image.LANCZOS)).astype(np.float32)
        reg = np.asarray(Image.open(os.path.join(mdir, "m_" + f[2:])).convert("RGB"))[y0:y0 + s, x0:x0 + s] > 127
        m = (reg[..., 0] | reg[..., 2]).astype(np.float32)  # face + hair
        m = cv2.GaussianBlur(cv2.erode(m, np.ones((3, 3), np.uint8)), (0, 0), 2)[..., None]
        im[y0:y0 + s, x0:x0 + s] = m * t + (1 - m) * im[y0:y0 + s, x0:x0 + s]
        Image.fromarray(im.round().clip(0, 255).astype(np.uint8)).save(os.path.join(mdir_out, f))
    print("facezoom merge ->", mdir_out)
