"""Merge fetched scene sets (tools/fetch_scenes.py outputs) and keep only modern colour photographs, using OpenCLIP
zero-shot (same idea as eval.scoreboard.clip_filter) plus an HSV saturation floor. Drops sepia/B&W prints,
illustrations, paintings, renders and AI images, which would otherwise teach the probe "old photo = real".
Run: python tools/filter_scenes.py <out_dir> <in_dir> [<in_dir> ...]   -> out_dir/frames.npy, cloth.npy, kept counts
"""
import os
import sys

import numpy as np
import open_clip
import torch
from PIL import Image

torch.set_grad_enabled(False)
out, ins = sys.argv[1], sys.argv[2:]
model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tok = open_clip.get_tokenizer("ViT-B-32")
P = ["a modern colour photograph of a person", "a black and white photograph", "a sepia vintage photograph",
     "an old printed photograph card", "a painting or illustration", "a 3D render or video game screenshot",
     "an AI generated image", "a photo of text or a document"]
t = model.encode_text(tok(P))
t /= t.norm(dim=-1, keepdim=True)


def keep(ims):
    k = []
    for i in range(0, len(ims), 64):
        x = torch.stack([pre(Image.fromarray(a)) for a in ims[i:i + 64]])
        f = model.encode_image(x)
        f /= f.norm(dim=-1, keepdim=True)
        p = (100 * f @ t.T).softmax(-1).numpy()
        sat = np.array([np.asarray(Image.fromarray(a).convert("HSV"))[..., 1].mean() for a in ims[i:i + 64]])
        k += list((p[:, 0] > 0.6) & (sat > 35))
    return np.array(k)


os.makedirs(out, exist_ok=True)
for name in ("frames", "cloth"):
    sets = [np.load(os.path.join(d, name + ".npy")) for d in ins if os.path.exists(os.path.join(d, name + ".npy"))]
    a = np.concatenate(sets)
    k = keep(a)
    np.save(os.path.join(out, name + ".npy"), a[k])
    print(name, f"{k.sum()}/{len(a)} kept")
