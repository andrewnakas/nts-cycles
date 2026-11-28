"""Real-photo head crops framed like our CG head boxes (render_lr.setup_roi), for the generative realism model.

Run: photo_heads.py <photos_dir with img/*.jpg> <out_dir> [--size 512]
For each confident, roughly frontal face (MTCNN): a square box 2.9x the face width, centred a little above the face
centre (the CG box is the head bone padded 1.45x, which frames a head the same way). The crop is first resized to a
random native size between 230 and 530 px, the range of our CG boxes at 1080x1920, and then to --size, so the real
photos carry the same amount of true detail as the renders the model will see. Resumable. Writes <id>_<k>.jpg and
heads.jsonl (photo id, box, face probability). Needs torch, facenet-pytorch, Pillow.
"""
import glob
import json
import os
import random
import sys

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
size = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 512
os.makedirs(dst, exist_ok=True)
log = os.path.join(dst, "heads.jsonl")
seen = {json.loads(l)["photo"] for l in open(log)} if os.path.exists(log) else set()
mt = MTCNN(keep_all=True, min_face_size=50, thresholds=[0.7, 0.8, 0.9], device="cpu")
rng = random.Random(0)
n = 0
with open(log, "a") as fp:
    for p in sorted(glob.glob(os.path.join(src, "img", "*.jpg"))):
        pid = os.path.basename(p)[:-4]
        if pid in seen:
            continue
        im = Image.open(p).convert("RGB")
        with torch.no_grad():
            boxes, probs, lm = mt.detect(im, landmarks=True)
        k = 0
        for b, pr, l in zip(boxes if boxes is not None else [], probs if probs is not None else [], lm if lm is not None else []):
            fw, fh = b[2] - b[0], b[3] - b[1]
            eye = np.linalg.norm(l[0] - l[1])
            # confident, big enough, eyes level and well apart (frontal to three-quarter), not a sliver at the edge
            if pr < 0.985 or fw < 70 or eye / fw < 0.33 or abs(l[0][1] - l[1][1]) > 0.25 * eye:
                continue
            cx, cy, half = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2 - 0.12 * fh, 1.45 * fw
            box = [cx - half, cy - half, cx + half, cy + half]
            if box[0] < -0.15 * half or box[1] < -0.15 * half or box[2] > im.width + 0.15 * half or box[3] > im.height + 0.15 * half:
                continue
            crop = im.crop(tuple(int(round(v)) for v in box))  # PIL pads the small overhang with black
            native = min(int(2 * half), rng.randint(230, 530))
            crop = crop.resize((native, native), Image.LANCZOS).resize((size, size), Image.BICUBIC)
            crop.save(os.path.join(dst, f"{pid}_{k}.jpg"), quality=95)
            fp.write(json.dumps({"photo": pid, "k": k, "box": [float(v) for v in box], "prob": float(pr), "native": native}) + "\n")
            k += 1
            n += 1
            if k >= 2:
                break
        if k == 0:
            fp.write(json.dumps({"photo": pid, "k": -1}) + "\n")
        fp.flush()
print(f"photo_heads: +{n} crops -> {dst}")
