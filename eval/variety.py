"""Variety A/B on stills: native Cycles vs our candidate(s) vs the reference pass 2 on the look-reference frames
(data/look/refs: native_/ref_/m_<ep>_f####.png, 9 episodes). Eval only.

v2 (2026-11-12). Three crops, each scored by two frozen backbones (DINOv2 ViT-B/14 and OpenCLIP ViT-B/16) with a
grain-invariant photo-vs-CG logistic probe. A method only beats another where both backbones agree.
  face   2.1x Haar box (mask fallback), real = CC0/CC-BY portrait crops, CG = stage-A faces of 70 other people
  cloth  square torso box under the face, real = data/eval/real_scenes_clean/cloth.npy, CG = torso crops of rendered episodes
  frame  the whole 9:16 frame, real = data/eval/real_scenes_clean/frames.npy, CG = rendered episode frames
         (data/eval/cg_scenes/<ep>_f_####.jpg). Cloth and frame probes leave out the scored still's own episode.
Identity is facenet cosine to native on the face box.
Run (Mac): python -m eval.variety <refs_dir> <out_prefix> label=<dir of <ep>_f####.png> [...]
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from eval import scoreboard as S
from eval.face import load_mask

torch.set_grad_enabled(False)
FRAME_HW = (392, 224)  # 9:16-ish, multiple of the 14 px DINOv2 patch


class Clip:
    def __init__(self):
        import open_clip
        self.m, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        self.m.eval()
        self.mean = np.array([0.4815, 0.4578, 0.4082], np.float32)
        self.std = np.array([0.2686, 0.2613, 0.2758], np.float32)

    def __call__(self, ims):
        out = []
        for i in range(0, len(ims), 32):
            x = np.stack([np.asarray(Image.fromarray(c).resize((224, 224), Image.BICUBIC)) for c in ims[i:i + 32]])
            x = torch.from_numpy(((x / 255.0 - self.mean) / self.std).astype(np.float32)).permute(0, 3, 1, 2)
            out.append(self.m.encode_image(x).numpy())
        return np.concatenate(out)


class DinoAny(S.Dino):
    """DINOv2 CLS at a given input size (frames keep their aspect ratio)."""
    def at(self, ims, hw=(224, 224)):
        out = []
        for i in range(0, len(ims), 16):
            x = np.stack([np.asarray(Image.fromarray(c).resize(hw[::-1], Image.BICUBIC)) for c in ims[i:i + 16]])
            x = torch.from_numpy(((x / 255.0 - S.MEAN) / S.STD).astype(np.float32)).permute(0, 3, 1, 2)
            out.append(self.m.forward_features(x)["x_norm_clstoken"].numpy())
        return np.concatenate(out)


def cloth_box(b, shape):
    """Square torso window under a face box (x, y, w, h), same rule as tools/fetch_scenes.py, clamped into the frame."""
    x, y, w, h = b
    s = min(int(2.4 * w), shape[0], shape[1])
    tx = min(max(x + w // 2 - s // 2, 0), shape[1] - s)
    ty = min(max(y + int(1.4 * h), 0), shape[0] - s)
    return tx, ty, s


def cut_cloth(rgb, b):
    tx, ty, s = cloth_box(b, rgb.shape)
    return np.asarray(Image.fromarray(rgb[ty:ty + s, tx:tx + s]).resize((224, 224), Image.LANCZOS))


def small(rgb):
    return np.asarray(Image.fromarray(rgb).resize((270, 480), Image.LANCZOS))


def main():
    refs, out = sys.argv[1], sys.argv[2]
    cands = [a.split("=", 1) for a in sys.argv[3:]]
    rng = np.random.default_rng(0)
    dino, clip, fid = DinoAny(), Clip(), S.FaceID()
    feats = {
        "dino": {"face": lambda c: dino(c)[0], "cloth": lambda c: dino.at(c), "frame": lambda c: dino.at(c, FRAME_HW)},
        "clip": {"face": clip, "cloth": clip, "frame": clip},
    }

    # --- training sets -------------------------------------------------------------------------------------------
    photos = np.load("runs/ntsr_a/photos_faces.npy", mmap_mode="r")
    keep = np.load("runs/ntsr_a/photos_faces_keep.npy")
    real = {"face": np.asarray(photos[keep]).transpose(0, 2, 3, 1)}
    cg = {"face": S.cg_crops("data/eval/cg_faces")}
    cg_ep = {"face": np.array([""] * len(cg["face"]))}
    rs = "data/eval/real_scenes_clean"  # tools/filter_scenes.py output
    real["frame"], real["cloth"] = np.load(f"{rs}/frames.npy"), np.load(f"{rs}/cloth.npy")
    fr, cl, fe, ce = [], [], [], []
    for p in sorted(glob.glob("data/eval/cg_scenes/*.jpg")):
        ep = os.path.basename(p).rsplit("_f_", 1)[0]
        a = np.asarray(Image.open(p).convert("RGB").resize((1080, 1920), Image.BICUBIC))
        fr.append(small(a)); fe.append(ep)
        ys = S.haar_box(a, (540, 700))  # nearest face to the usual hero position
        if ys is not None:
            tx, ty, s = cloth_box(ys, a.shape)
            if s >= 2.4 * ys[2] - 1:  # only torsos that fit, like the photos
                cl.append(cut_cloth(a, ys)); ce.append(ep)
    cg["frame"], cg_ep["frame"] = np.stack(fr), np.array(fe)
    cg["cloth"], cg_ep["cloth"] = np.stack(cl), np.array(ce)
    print({k: (len(real[k]), len(cg[k])) for k in real}, flush=True)

    F = {}
    cache = "runs/variety2_feats.npz"  # training features only depend on the fixed photo / CG sets
    z = dict(np.load(cache)) if os.path.exists(cache) else {}
    for bb in feats:
        for crop in real:
            key = f"{bb}_{crop}"
            if key + "_r" not in z:
                f = feats[bb][crop]
                sub = real[crop][rng.choice(len(real[crop]), min(200, len(real[crop])), replace=False)]
                z[key + "_r"] = np.concatenate([f(real[crop]), f(S.grained(sub, rng))])
                z[key + "_c"], z[key + "_g"] = f(cg[crop]), f(S.grained(cg[crop], rng))
                np.savez(cache, **z)
            F[bb, crop] = (z[key + "_r"], z[key + "_c"], z[key + "_g"])
            print("features", bb, crop, flush=True)
    probes = {}

    def probe(bb, crop, ep):
        k = (bb, crop, ep if crop != "face" else "")
        if k not in probes:
            r, c, cgr = F[bb, crop]
            m = cg_ep[crop] != k[2] if k[2] else np.ones(len(c), bool)
            if m.sum() < 20:  # too few CG left after leaving the episode out: use all
                m = np.ones(len(c), bool)
            probes[k] = S.probe(r, np.concatenate([c[m], cgr[np.tile(m, 3)]]))
            print("probe", k, f"acc {probes[k][1]:.2f}", flush=True)
        return probes[k][0]

    # --- score -----------------------------------------------------------------------------------------------------
    cols = [("native", None)] + cands + [("reference", "ref")]
    labs = [lab for lab, _ in cols]
    rows, sheet = [], []
    for nat in sorted(glob.glob(os.path.join(refs, "native_*.png"))):
        key = os.path.basename(nat)[7:-4]  # <ep>_f####
        ep, frn = key.rsplit("_f", 1)
        paths = {"native": nat, "reference": os.path.join(refs, f"ref_{key}.png")}
        paths.update({lab: os.path.join(d, f"{key}.png") for lab, d in cands})
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        ims = {k: np.asarray(Image.open(p).convert("RGB")) for k, p in paths.items()}
        mk = os.path.join(refs, f"m_{ep}_{frn}.png")
        face = load_mask(mk)[0] if os.path.exists(mk) else None
        if face is None or face.sum() < 50:
            continue
        ys, xs = np.nonzero(face)
        c = ((xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2)
        b = S.haar_box(ims["native"], c)
        if b is None:
            s = int(0.75 * max(np.ptp(xs), np.ptp(ys)))
            b = (c[0] - s // 2, c[1] - s // 2, s, s)
        e0 = fid(ims["native"], b)
        r = {"frame": key}
        for lab in labs:
            im = ims[lab]
            crops = {"face": S.crop(im, b)[None], "cloth": cut_cloth(im, b)[None], "frame": small(im)[None]}
            r[lab] = {(bb, cr): float(probe(bb, cr, ep)(feats[bb][cr](crops[cr]))[0]) for bb in feats for cr in crops}
            r[lab]["id"] = float(fid(im, b) @ e0)
        rows.append(r)
        x, y, w, h = b
        s = int(max(w, h) * 1.3)
        row = []
        for lab in labs:
            im = Image.fromarray(ims[lab]).crop((x + w // 2 - s, y + h // 2 - s, x + w // 2 + s, y + h // 2 + s)).resize((300, 300))
            ImageDraw.Draw(im).text((4, 4), f"{key} {lab}", fill=(255, 255, 0))
            row.append(im)
        sheet.append(row)
        print(key, flush=True)

    crops = ("face", "cloth", "frame")
    avg = lambda r, cr: (r[("dino", cr)] + r[("clip", cr)]) / 2
    beats = lambda a, d, cr: a[("dino", cr)] > d[("dino", cr)] and a[("clip", cr)] > d[("clip", cr)]
    L = [f"# Variety A/B v2 ({len(rows)} stills)", "",
         "Probe held-out accuracy: " + ", ".join(f"{k[0]}/{k[1]}{'/-' + k[2] if k[2] else ''} {v[1]:.2f}"
                                                 for k, v in sorted(probes.items())), "",
         "## Summary (mean P(real), DINOv2 / CLIP; wins = both backbones above reference on that still)", "",
         "| method | face | cloth | frame | identity | wins face | wins cloth | wins frame |", "|---|---|---|---|---|---|---|---|"]
    for lab in labs:
        m = lambda bb, cr: np.mean([r[lab][(bb, cr)] for r in rows])
        L.append(f"| {lab} | " + " | ".join(f"{m('dino', cr):.3f} / {m('clip', cr):.3f}" for cr in crops)
                 + f" | {np.mean([r[lab]['id'] for r in rows]):.2f} | "
                 + " | ".join(f"{sum(beats(r[lab], r['reference'], cr) for r in rows)}/{len(rows)}" for cr in crops) + " |")
    L += ["", "## Per still (backbone-mean P(real) face / cloth / frame, identity)", "",
          "| still | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
    for r in rows:
        L.append(f"| {r['frame']} | " + " | ".join(
            " / ".join(f"{avg(r[l], cr):.2f}" for cr in crops) + f", {r[l]['id']:.2f}" for l in labs) + " |")
    open(out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    W = Image.new("RGB", (300 * len(labs), 300 * len(sheet)))
    for i, row in enumerate(sheet):
        for j, im in enumerate(row):
            W.paste(im, (300 * j, 300 * i))
    W.save(out + "_sheet.jpg", quality=88)


if __name__ == "__main__":
    main()
