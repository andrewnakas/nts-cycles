"""Whole-scene real photos sized like our frames, for the whole-frame generative realism model (realf/ folder).

Run: photo_scenes.py <photos_dir with img/*.jpg> <out_dir>
For each photo with a confident face (MTCNN): the photo is scaled down so the largest face is a random width between
55 and 210 px (the range of faces in our 1080x1920 frames, wide shot to close shot), never scaled up, and kept if at
least 512 px remain on both sides. The trainer takes random 512 px tiles from it, so skin, hair, clothes, hands and
room surfaces all appear at the size the renders show them. Resumable. Writes <id>.jpg and scenes.jsonl (photo id,
scale, face boxes in the saved image). Needs torch, facenet-pytorch, Pillow.
"""
import glob
import json
import os
import random
import sys

import torch
from facenet_pytorch import MTCNN
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
log = os.path.join(dst, "scenes.jsonl")
seen = {json.loads(l)["photo"] for l in open(log)} if os.path.exists(log) else set()
mt = MTCNN(keep_all=True, min_face_size=40, thresholds=[0.7, 0.8, 0.9], device="cpu")
rng = random.Random(0)
n = 0
with open(log, "a") as fp:
    for p in sorted(glob.glob(os.path.join(src, "img", "*.jpg"))):
        pid = os.path.basename(p)[:-4]
        if pid in seen:
            continue
        rec = {"photo": pid, "kept": False}
        try:
            im = Image.open(p).convert("RGB")
            with torch.no_grad():
                boxes, probs = mt.detect(im)
        except Exception:
            boxes = None
        good = [b for b, pr in zip(boxes, probs) if pr > 0.97] if boxes is not None else []
        if good:
            fw = max(b[2] - b[0] for b in good)
            s = min(1.0, rng.uniform(55, 210) / fw)
            w, h = int(im.width * s), int(im.height * s)
            if fw * s >= 45 and min(w, h) >= 512:
                im.resize((w, h), Image.LANCZOS).save(os.path.join(dst, pid + ".jpg"), quality=93)
                rec = {"photo": pid, "kept": True, "scale": s, "faces": [[float(v * s) for v in b] for b in good]}
                n += 1
        fp.write(json.dumps(rec) + "\n")
        fp.flush()
print(f"photo_scenes: +{n} scenes -> {dst}")
