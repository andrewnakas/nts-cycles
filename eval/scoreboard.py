"""Realism / identity / flicker scoreboard: NTS-R variants against the reference on the same clip (eval only).

  realism   P(real) from a grain-invariant logistic probe on frozen DINOv2 ViT-B/14 features of face crops, trained on the filtered
            CC0/CC-BY photos (real) against stage-A Cycles faces of 70 other people (data/eval/cg_faces, input and hero).
            Nothing from the scored clip is used to train it. (KID can't work here: one person's 12 frames are a
            point, not a distribution.)
  identity  facenet (VGGFace2, MIT; eval only) cosine between each method's face and the input render's face.
  flicker   eval.face.face_temporal against the input (frame-to-frame face change the real motion doesn't explain).
  tells     eval.look.ratios vs the input on face (R) and hair (B) masks: bright / detail / sat.

Crops follow tools/colab/fetch_photos.py exactly (OpenCV Haar box x2.1, 256 px), with the box found once per frame on
the input so every method is cut identically. Photos are filtered with CLIP zero-shot (colour photo of a real face, not
B&W / painting / statue / object), since about a third of the raw set is junk.

Run (Mac CPU):
  uv run --python 3.11 --with 'torch==2.2.2' --with 'torchvision==0.17.2' --with 'numpy<2' --with facenet-pytorch \
    --with open_clip_torch --with opencv-python-headless --with pillow python -m eval.scoreboard \
    data/eval/zd_sb 420-431 runs/ntsr_a/photos_faces.npy in=ntsr5_in "run 4=ntsr4_out" "run 5=ntsr5_out" \
    B2=ntsrb2_out "reference=ref5_pass2_clip"
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from eval.face import face_temporal, load_mask
from eval.look import ratios

torch.set_grad_enabled(False)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def haar_box(rgb, near):
    """The Haar detection whose centre is closest to `near` (x, y) and within one box size of it (false hits on
    screens and posters are common), and whose 2.1x crop stays inside the frame."""
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    best = None
    for x, y, w, h in det.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1.1, 6, minSize=(40, 40)):  # pilot crops: wide shots have small faces
        cx, cy, s = x + w // 2, y + h // 2, int(max(w, h) * 1.05)
        d = np.hypot(cx - near[0], cy - near[1])
        if d < max(w, h) and cx - s >= 0 and cy - s >= 0 and cx + s <= rgb.shape[1] and cy + s <= rgb.shape[0]:
            if best is None or d < best[0]:
                best = (d, (x, y, w, h))
    return best and best[1]


def crop(rgb, b, scale=2.1, size=256):
    x, y, w, h = b
    cx, cy, s = x + w // 2, y + h // 2, int(max(w, h) * scale / 2)
    # Close-ups (face fills the frame): a 2.1x window runs off the edge and numpy slicing silently returned a sliver,
    # which scored 0.811 for every method. Shrink the window to fit and move it inside the frame instead.
    s = max(8, min(s, rgb.shape[1] // 2, rgb.shape[0] // 2))
    cx, cy = min(max(cx, s), rgb.shape[1] - s), min(max(cy, s), rgb.shape[0] - s)
    return np.asarray(Image.fromarray(rgb[cy - s:cy + s, cx - s:cx + s]).resize((size, size), Image.LANCZOS))


def clip_filter(photos, cache):
    if os.path.exists(cache):
        return np.load(cache)
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32")
    prompts = ["a colour photograph of a person's face", "a black and white photograph", "a sepia vintage photograph",
               "a painting of a person", "a statue or sculpture", "a close-up photo of an object or a hand",
               "a blurry photo with no face"]
    t = model.encode_text(tok(prompts)); t /= t.norm(dim=-1, keepdim=True)
    keep = []
    for i in range(0, len(photos), 64):
        x = torch.stack([pre(Image.fromarray(p.transpose(1, 2, 0))) for p in photos[i:i + 64]])
        f = model.encode_image(x); f /= f.norm(dim=-1, keepdim=True)
        p = (100 * f @ t.T).softmax(-1).numpy()
        sat = np.array([np.asarray(Image.fromarray(q.transpose(1, 2, 0)).convert("HSV"))[..., 1].mean() for q in photos[i:i + 64]])
        keep += list((p[:, 0] > 0.5) & (sat > 40))
    keep = np.array(keep)
    np.save(cache, keep)
    return keep


class Dino:
    def __init__(self):
        self.m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()

    def __call__(self, crops):  # (N,256,256,3) uint8 -> cls (N,768), centre patch tokens (N*36,768)
        x = np.stack([np.asarray(Image.fromarray(c).resize((224, 224), Image.BICUBIC)) for c in crops])
        x = torch.from_numpy(((x / 255.0 - MEAN) / STD).astype(np.float32)).permute(0, 3, 1, 2)
        cls, pat = [], []
        for i in range(0, len(x), 32):
            o = self.m.forward_features(x[i:i + 32])
            cls.append(o["x_norm_clstoken"].numpy())
            p = o["x_norm_patchtokens"].reshape(-1, 16, 16, 768)[:, 5:11, 5:11]  # face centre (crop is 2.1x the box)
            pat.append(p.reshape(-1, 768).numpy())
        return np.concatenate(cls), np.concatenate(pat)


def probe(real, cg, steps=3000, wd=1e-3):
    """Balanced logistic regression on unit features; returns a function giving P(real), and its leave-out accuracy."""
    x = np.concatenate([real, cg]); x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.concatenate([np.ones(len(real)), np.zeros(len(cg))]).astype(np.float32)
    w8 = np.where(y == 1, 0.5 / len(real), 0.5 / len(cg)).astype(np.float32)
    rng = np.random.default_rng(1); hold = rng.random(len(x)) < 0.2

    def fit(m):
        X, Y, W = (torch.from_numpy(v[m]) for v in (x.astype(np.float32), y, w8))
        lin = torch.nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(lin.parameters(), 1e-2, weight_decay=wd)
        with torch.enable_grad():
            for _ in range(steps):
                opt.zero_grad()
                l = (torch.nn.functional.binary_cross_entropy_with_logits(lin(X)[:, 0], Y, reduction="none") * W).sum() / W.sum()
                l.backward(); opt.step()
        return lin
    lin = fit(~hold)
    p = torch.sigmoid(lin(torch.from_numpy(x[hold].astype(np.float32)))[:, 0]).numpy()
    acc = float(((p > 0.5) == (y[hold] == 1)).mean())
    lin = fit(np.ones(len(x), bool))
    return (lambda f: torch.sigmoid(lin(torch.from_numpy((f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float32)))[:, 0]).numpy()), acc


def grained(crops, rng, sigmas=(2, 4, 8)):
    """Copies of uint8 crops with per-pixel luminance grain. The probe trains on these for both classes: plain grain
    alone took a clean render from P(real) 0.05 to 0.36, so without this the probe is a noise meter."""
    out = []
    for s in sigmas:
        n = rng.normal(0, s, crops.shape[:3])[..., None]
        out.append(np.clip(crops.astype(np.float32) + n, 0, 255).astype(np.uint8))
    return np.concatenate(out)


def cg_crops(d):
    out = []
    for f in sorted(os.listdir(d)):
        im = np.asarray(Image.open(os.path.join(d, f)).convert("RGB"))
        b = haar_box(im, (im.shape[1] // 2, im.shape[0] // 2))  # cgcrops cut a face-mask-centred square
        if b:
            out.append(crop(im, b))
    return np.stack(out)


class FaceID:
    def __init__(self):
        from facenet_pytorch import InceptionResnetV1
        self.m = InceptionResnetV1(pretrained="vggface2").eval()

    def __call__(self, rgb, b):
        x, y, w, h = b
        m = int(0.1 * max(w, h))
        c = np.asarray(Image.fromarray(rgb[y - m:y + h + m, x - m:x + w + m]).resize((160, 160), Image.BILINEAR))
        t = torch.from_numpy((c.astype(np.float32) - 127.5) / 128.0).permute(2, 0, 1)[None]
        e = self.m(t)[0].numpy()
        return e / np.linalg.norm(e)


def main():
    root, span, photos_path = sys.argv[1], sys.argv[2], sys.argv[3]
    methods = [a.split("=", 1) for a in sys.argv[4:]]
    a, b = map(int, span.split("-"))
    frames = range(a, b + 1)
    rng = np.random.default_rng(0)
    load = lambda d, f: np.asarray(Image.open(f"{root}/{d}/f_{f:04d}.png").convert("RGB"))

    photos = np.load(photos_path, mmap_mode="r")
    keep = clip_filter(photos, photos_path.replace(".npy", "_keep.npy"))
    print(f"photos: {keep.sum()}/{len(keep)} kept by the CLIP + colour filter")
    dino, fid = Dino(), FaceID()
    ph_cls, _ = dino(np.asarray(photos[keep]).transpose(0, 2, 3, 1))
    cg = cg_crops(os.path.join(root, "..", "cg_faces"))
    ph_np = np.asarray(photos[keep]).transpose(0, 2, 3, 1)
    sub = ph_np[rng.choice(len(ph_np), min(len(ph_np), 200), replace=False)]
    real_p, acc = probe(np.concatenate([ph_cls, dino(grained(sub, rng))[0]]),
                        np.concatenate([dino(cg)[0], dino(grained(cg, rng))[0]]))
    print(f"realism probe: {keep.sum()} photos vs {len(cg)} CG faces, held-out accuracy {acc:.3f}")

    ref = methods[0][1]
    def face_centre(f):
        ys, xs = np.nonzero(load_mask(f"{root}/masks/m_{f:04d}.png")[0])
        return (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    def mask_box(f):  # Haar misses small or turned faces: a Haar-sized square inside the face mask instead
        ys, xs = np.nonzero(load_mask(f"{root}/masks/m_{f:04d}.png")[0])
        s = int(0.75 * max(xs.ptp(), ys.ptp()))
        return ((xs.min() + xs.max()) // 2 - s // 2, (ys.min() + ys.max()) // 2 - s // 2, s, s) if s >= 20 else None
    boxes = {f: haar_box(load(ref, f), face_centre(f)) for f in frames}
    boxes = {f: b if b is not None else mask_box(f) for f, b in boxes.items()}
    print("face boxes:", {f: tuple(int(v) for v in b) for f, b in boxes.items() if b is not None})
    frames = [f for f in frames if boxes[f] is not None]
    ref_id = {f: fid(load(ref, f), boxes[f]) for f in frames}
    ph_id = [fid(np.asarray(Image.fromarray(p.transpose(1, 2, 0)).resize((256, 256))), (67, 67, 122, 122))
             for p in np.asarray(photos[keep][:200])]
    floor = float(np.mean([ref_id[frames[0]] @ e for e in ph_id]))

    rows = []
    for lab, d in methods:
        ims = {f: load(d, f) for f in frames}
        cls, _ = dino(np.stack([crop(ims[f], boxes[f]) for f in frames]))
        pr = real_p(cls)
        ident = float(np.mean([fid(ims[f], boxes[f]) @ ref_id[f] for f in frames]))
        flick, tf, th = [], [], []
        for i, f in enumerate(frames):
            face, _ = load_mask(f"{root}/masks/m_{f:04d}.png")
            hair = np.asarray(Image.open(f"{root}/masks/m_{f:04d}.png").convert("RGB"))[..., 2] > 127
            g, t = load(ref, f).astype(np.float32), ims[f].astype(np.float32)
            tf.append(ratios(g, t, face)); th.append(ratios(g, t, hair))
            if i:
                flick.append(face_temporal(load(ref, frames[i - 1]).astype(np.float32), g,
                                           ims[frames[i - 1]].astype(np.float32), t, face))
        avg = lambda rs, k: float(np.mean([r[k] for r in rs]))
        rows.append({"method": lab, "p_real": float(pr.mean()), "p_real_min": float(pr.min()),
                     "identity": ident, "flicker": float(np.mean(flick)),
                     **{f"face_{k}": avg(tf, k) for k in ("bright", "detail", "sat")},
                     **{f"hair_{k}": avg(th, k) for k in ("bright", "detail", "sat")}})
        print(json.dumps(rows[-1]))

    hdr = ("method", "P(real)", "identity", "flicker", "face br", "face det", "face sat", "hair br",
           "hair det", "hair sat")
    print(f"\nidentity reference: same render across frames ~1.0; random real photo vs this face = {floor:.3f}")
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for r in rows:
        print(f"| {r['method']} | {r['p_real']:.3f} (min {r['p_real_min']:.3f})"
              f" | {r['identity']:.3f} | {r['flicker']:.2f} | {r['face_bright']:.2f} | {r['face_detail']:.2f}"
              f" | {r['face_sat']:.2f} | {r['hair_bright']:.2f} | {r['hair_detail']:.2f} | {r['hair_sat']:.2f} |")
    with open(os.path.join(root, "scoreboard.json"), "w") as fh:
        json.dump({"frames": [frames[0], frames[-1]], "photos_kept": int(keep.sum()), "id_floor": floor, "rows": rows},
                  fh, indent=1)


if __name__ == "__main__":
    main()
