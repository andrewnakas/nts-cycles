"""Shot-level photo grade: NR-1 teacher upgrade 4 (2026-11-18, the design log row 6).

The user's notes on the reference ("looked more real and cooler"; ours gave itself away by "skin tone") point at the global
colour and tone of the frame. This is a white-box grade in the spirit of Hu et al. 2018, "Exposure: A White-Box Photo
Post-Processing Framework" (paper): a few global, bounded, interpretable operations predicted from a thumbnail of the
frame and learned adversarially against whole real photos.

  operations   3x3 colour matrix (I + 0.08 tanh), one monotonic 8-segment tone curve on all channels (slopes
               exp(0.35 tanh)), saturation (1 + 0.25 tanh). The predictor's last layer starts at zero: identity grade.
  predictor    frozen DINOv2-S class token of the 224x392 thumbnail -> MLP -> 18 numbers
  realism      hinge GAN, projected discriminator on frozen DINOv2-S patch tokens (2 layers) of the graded thumbnail
               against real scene photos (realf/: indoor scenes with people, CC0/CC-BY), centre-cropped to 9:16
  restraint    L2 on the 18 raw numbers (--w-reg), so the grade only moves as far as realism pulls it
  no flicker   at inference the raw numbers are averaged over a shot (between cuts) and ONE grade is applied to every
               frame of it: the grade cannot flicker and never touches structure, so identity is kept.

Run: grade.py train --data <dir with cgf/ realf/> --out <dir> [--steps 3000] [--batch 16]      (GPU, ~10 min on a T4)
     grade.py apply <grade.pt> <frames_dir> <out_dir> [--cuts a,b,...]  (CPU is fine: thumbnails only)
     grade.py train ... --smoke    tiny random features, no downloads, 3 steps on CPU
Open weights only: DINOv2 (Apache-2.0).
"""
import argparse
import glob
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

TH, TW = 392, 224  # thumbnail (9:16 portrait), multiples of 14 for DINOv2
NP = 9 + 8 + 1


def thumb(im):
    """PIL image -> centre crop to 9:16 -> (3, TH, TW) in [0, 1]."""
    w, h = im.size
    if w / h > 9 / 16:
        nw = int(h * 9 / 16)
        im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w * 16 / 9)
        im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    a = np.asarray(im.convert("RGB").resize((TW, TH), Image.BILINEAR), np.float32) / 255
    return torch.from_numpy(a.copy()).permute(2, 0, 1)


def grade(x, raw):
    """x (b,3,H,W) in [0,1] display space, raw (b,18) -> graded x."""
    b = x.shape[0]
    M = torch.eye(3, device=x.device)[None] + 0.08 * torch.tanh(raw[:, :9]).view(b, 3, 3)
    y = torch.einsum("bij,bjhw->bihw", M, x).clamp(0, 1)
    s = torch.exp(0.35 * torch.tanh(raw[:, 9:17]))  # 8 positive slopes, normalised so 0 -> 0 and 1 -> 1
    knots = torch.cat([torch.zeros(b, 1, device=x.device), torch.cumsum(s, 1)], 1)
    knots = knots / knots[:, -1:]
    t = y * 8
    i = t.floor().clamp(0, 7).long()
    f = t - i
    lo = torch.gather(knots[:, None, None, None, :].expand(*y.shape, 9), -1, i[..., None])[..., 0]
    hi = torch.gather(knots[:, None, None, None, :].expand(*y.shape, 9), -1, (i + 1)[..., None])[..., 0]
    y = lo + (hi - lo) * f
    sat = 1 + 0.25 * torch.tanh(raw[:, 17]).view(b, 1, 1, 1)
    lum = y.mean(1, keepdim=True)
    return (lum + sat * (y - lum)).clamp(0, 1)


class Feats(nn.Module):
    def __init__(self, smoke):
        super().__init__()
        self.net = None
        if not smoke:
            from transformers import AutoModel
            self.net = AutoModel.from_pretrained("facebook/dinov2-small").eval().requires_grad_(False)
        self.dim = 384 if self.net is not None else 48
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, layers=(6, 12)):
        """x in [0,1] -> (class token, [patch token maps of the given layers])."""
        if self.net is None:
            p = F.adaptive_avg_pool2d(x, (4, 4))
            return p.flatten(1), [F.adaptive_avg_pool2d(x, (28, 16)).repeat(1, 16, 1, 1)[:, :48]] * 2
        hs = self.net(pixel_values=(x - self.mean) / self.std, output_hidden_states=True).hidden_states
        h, w = x.shape[-2] // 14, x.shape[-1] // 14
        return hs[-1][:, 0], [hs[i][:, 1:].transpose(1, 2).reshape(x.shape[0], -1, h, w) for i in layers]


class Predictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, NP))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cls):
        return self.mlp(cls.float())


class Heads(nn.Module):
    def __init__(self, dim):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.h = nn.ModuleList(nn.Sequential(sn(nn.Conv2d(dim, 128, 1)), nn.LeakyReLU(0.2), sn(nn.Conv2d(128, 1, 3, padding=1)))
                               for _ in range(2))

    def forward(self, maps):
        return torch.cat([h(m.float()).flatten(1) for h, m in zip(self.h, maps)], 1)


def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    cg = [p for p in sorted(glob.glob(os.path.join(a.data, "cgf", "*.jpg"))) if not p.endswith(".alb.jpg")]
    real = sorted(glob.glob(os.path.join(a.data, "realf", "*.jpg")))
    print(f"grade: {len(cg)} CG frames, {len(real)} real photos", flush=True)
    if a.smoke:
        a.steps, a.batch = 3, 2
    Fe = Feats(a.smoke).to(dev)
    P, D = Predictor(Fe.dim).to(dev), Heads(Fe.dim).to(dev)
    og = torch.optim.Adam(P.parameters(), lr=a.lr, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr=a.lr * 2, betas=(0.5, 0.999))
    cache = {}

    def load(p):
        if p not in cache:
            cache[p] = thumb(Image.open(p))
        t = cache[p]
        return t.flip(-1) if random.random() < 0.5 else t
    for step in range(1, a.steps + 1):
        x = torch.stack([load(random.choice(cg)) for _ in range(a.batch)]).to(dev)
        r = torch.stack([load(random.choice(real)) for _ in range(a.batch)]).to(dev)
        with torch.no_grad():
            cls, _ = Fe(x)
        raw = P(cls)
        y = grade(x, raw)
        _, my = Fe(y)
        lg = -D(my).mean() + a.w_reg * raw.pow(2).mean()
        og.zero_grad()
        lg.backward()
        og.step()
        with torch.no_grad():
            _, mr = Fe(r)
            _, mf = Fe(y.detach())
        ld = F.relu(1 - D(mr)).mean() + F.relu(1 + D(mf)).mean()
        od.zero_grad()
        ld.backward()
        od.step()
        if step % 100 == 0 or a.smoke:
            m = raw.detach()
            print(f"grade step {step} g {lg.item():.4f} d {ld.item():.4f} |raw| {m.abs().mean():.3f} "
                  f"sat {1 + 0.25 * torch.tanh(m[:, 17]).mean():.3f}", flush=True)
    torch.save({"p": P.state_dict(), "dim": Fe.dim}, os.path.join(a.out, "grade.pt"))
    with torch.no_grad():  # sample sheet: 4 CG frames before and after
        x = torch.stack([thumb(Image.open(p)) for p in random.Random(0).sample(cg, min(4, len(cg)))]).to(dev)
        y = grade(x, P(Fe(x)[0]))
        g = torch.cat([torch.cat(list(x), -1), torch.cat(list(y), -1)], -2)
        Image.fromarray((g.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(a.out, "grade_sheet.jpg"))
    print("GRADE_DONE", flush=True)


@torch.no_grad()
def apply(a):
    ck = torch.load(a.ckpt, map_location="cpu")
    Fe = Feats(ck["dim"] != 384)
    P = Predictor(ck["dim"])
    P.load_state_dict(ck["p"])
    files = sorted(glob.glob(os.path.join(a.frames, "f_*.jpg")) + glob.glob(os.path.join(a.frames, "f_*.png")))
    nums = [int(os.path.basename(p)[2:6]) for p in files]
    cuts = sorted(int(c) for c in a.cuts.split(",")) if a.cuts else []
    shot = [sum(n >= c for c in cuts) for n in nums]
    raws = [P(Fe(thumb(Image.open(p))[None])[0])[0] for p in files]
    os.makedirs(a.out, exist_ok=True)
    for s in sorted(set(shot)):
        idx = [i for i, v in enumerate(shot) if v == s]
        r = torch.stack([raws[i] for i in idx]).mean(0, keepdim=True)  # one grade for the whole shot
        print(f"shot {s}: {len(idx)} frames, sat {1 + 0.25 * torch.tanh(r[0, 17]):.3f}, |raw| {r.abs().mean():.3f}")
        for i in idx:
            im = Image.open(files[i]).convert("RGB")
            x = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None]
            y = grade(x, r)[0].permute(1, 2, 0).numpy()
            out = os.path.join(a.out, os.path.basename(files[i])[:-4] + ".png")
            Image.fromarray((y * 255 + 0.5).astype(np.uint8)).save(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "apply"])
    p.add_argument("ckpt", nargs="?")
    p.add_argument("frames", nargs="?")
    p.add_argument("out_dir", nargs="?")
    p.add_argument("--data")
    p.add_argument("--out")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--w-reg", type=float, default=0.05)
    p.add_argument("--cuts", default="")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.mode == "train":
        train(a)
    else:
        a.out = a.out_dir
        apply(a)


if __name__ == "__main__":
    sys.exit(main())
