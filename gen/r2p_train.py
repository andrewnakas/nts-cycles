"""R2P: render-to-photo head translation, the generative realism pass (2026-11-13).

One step of a pretrained latent diffusion UNet turns a CG head crop into a photoreal one. Unpaired: CG crops on one
side, real photo head crops on the other, never the reference's output.

  generator    z = VAE.encode(x);  z' = z + UNet(z, t, "photo")  (the UNet's last conv starts at zero, so training
               starts from "output = input" and only moves as far as the losses pull it);  y = VAE.decode(z').
               Trainable: LoRA on the UNet's attention and conv layers, and its last conv. VAE frozen.
  realism      hinge GAN with a projected discriminator: frozen DINOv2-S features of random crops at several scales,
               small trainable heads on four layers. Real side = photo head crops.
  same face    LPIPS(y, x) keeps structure; a DINOv2 embedding of the face region keeps identity;
               a weak low-frequency term keeps overall brightness and colour near the render.
  no flicker   on consecutive-frame pairs, the previous frame's output is warped by Blender's exact motion and
               the current output is pulled to it where the two input frames agree.

Open weights only: Stable Diffusion 1.5 (CreativeML OpenRAIL-M) with LCM-LoRA (OpenRAIL++) fused in, DINOv2
(Apache-2.0, also used for the identity term), LPIPS (BSD).

NR-1 teacher upgrades (2026-11-18, the design log rows 2-5; every one is off by default, so old runs replay):
  --match 1     content-matched real crops (Richter et al. 2021 paper): frames and photos are cut into fixed windows,
                embedded with the discriminator's DINOv2 (class token), and each CG window keeps its real windows with
                cosine > --match-thr. Frame-tile steps then train on a CG window and one of its matches only.
  --w-r1 x      R1 penalty (Mescheder et al. 2018) on the real side, on the frozen features the heads see, every 4th step
  --w-band x    the 8-32 px band of the change is held to the render on skin and smooth areas; finer detail stays free
  --gbuf 1      G-buffer adapter (ControlNet/T2I-Adapter papers): albedo, normal, log depth and region masks
                (<frame>.alb.jpg/.nrm.png/.dep.png, nts/gbuf.py) through a zero-initialised conv branch added to the
                UNet's conv_in output, so training starts exactly at the warm-start weights

Run: r2p_train.py --data <dir with cg/ real/ eval/> --out <dir> [--hours 11] [--steps 30000] [--batch 2]
       [--size 512] [--rank 32] [--lr 2e-5] [--w-lpips 1.0] [--w-id 1.0] [--w-low 0.3] [--w-temp 2.0] [--w-gan 0.5]
       [--resume ckpt] [--smoke]
     r2p_train.py --infer <ckpt> --data <dir> --out <dir>      translate eval/*.jpg -> out/eval/*.png
--smoke builds tiny random networks (no downloads) and runs 3 steps on CPU: a code-path check only.
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

P = argparse.ArgumentParser()
P.add_argument("--data", required=True)
P.add_argument("--out", required=True)
P.add_argument("--hours", type=float, default=11.0)
P.add_argument("--steps", type=int, default=30000)
P.add_argument("--batch", type=int, default=2)
P.add_argument("--size", type=int, default=512)
P.add_argument("--rank", type=int, default=32)
P.add_argument("--lr", type=float, default=2e-5)
P.add_argument("--t", type=int, default=199)
P.add_argument("--w-lpips", type=float, default=1.0)
P.add_argument("--w-id", type=float, default=4.0)    # run a5 (1.0 / 0.3 / 2.0 / 0.5) aged the face, shifted hair
P.add_argument("--w-low", type=float, default=0.8)   # colour and flickered more than the plain render (3.24 vs 2.23)
P.add_argument("--w-temp", type=float, default=5.0)
P.add_argument("--w-gan", type=float, default=0.6)
# run b2 (id 6.0, gan 0.4, keep-losses on the whole head) kept the face but also left the hair a rendered shell.
# The keep-the-render losses (lpips, identity) now count only this fraction of the change made inside the hair mask.
P.add_argument("--w-hair", type=float, default=0.15)
# run c1 (w-hair 0.15, id 3.0) grew real strands but turned the hair near-black and let the hairline melt into the
# forehead: the hair's own low-frequency colour is now held to the render, and the freedom covers the hair only.
P.add_argument("--w-haircol", type=float, default=4.0)
# whole-frame mode (data has cgf/ and realf/): this share of the steps trains on 512 px tiles of full frames
P.add_argument("--p-frame", type=float, default=0.6)
# run d1 changed hair and little else: cloth, walls and props were held to the render at full weight. On frame tiles
# the keep-losses now see only this share of the change outside faces, hands and printed/on-screen graphics.
P.add_argument("--w-free", type=float, default=0.35)
# extra real photos for frame tiles (any subject; public-domain sets fetched by the runner): folder of jpgs, and the
# share of real tiles drawn from it instead of realf/
P.add_argument("--realx", default=None)
P.add_argument("--p-realx", type=float, default=0.5)
P.add_argument("--match", type=int, default=0)
P.add_argument("--match-thr", type=float, default=0.5)
P.add_argument("--match-k", type=int, default=16)
P.add_argument("--w-r1", type=float, default=0.0)
P.add_argument("--w-band", type=float, default=0.0)
P.add_argument("--band-lo", type=int, default=9)    # box sizes in tile pixels: the band is blur(lo) - blur(hi)
P.add_argument("--band-hi", type=int, default=33)
P.add_argument("--gbuf", type=int, default=0)
P.add_argument("--vae-res", type=int, default=1)  # output = input + (decode(z + r) - decode(z)): untouched areas stay exact
P.add_argument("--gan-start", type=int, default=300)
P.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
P.add_argument("--lcm", default="latent-consistency/lcm-lora-sdv1-5")
P.add_argument("--resume", default=None)
P.add_argument("--infer", default=None)
P.add_argument("--infer-only", default=None)
P.add_argument("--smoke", action="store_true")
A = P.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEV2 = "cuda:1" if torch.cuda.device_count() > 1 else DEV  # discriminator and loss networks (Kaggle gives two T4s)
AMP = DEV == "cuda"
os.makedirs(A.out, exist_ok=True)
LOG = open(os.path.join(A.out, "train.log"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


# ---------------------------------------------------------------------------------------------------------- data
def load_img(p, size):
    im = Image.open(p).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 127.5 - 1


def load_mask(p, size):
    """Two channels from a Region image (R face, B hair): 0 = head mask, grown ~20 px and softened (all ones when
    there is no file); 1 = hair only, lightly softened (zeros when there is no file)."""
    if not os.path.exists(p):
        return torch.cat([torch.ones(1, size, size), torch.zeros(1, size, size)])
    a = np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.NEAREST))
    m = torch.from_numpy(((a[..., 0] > 127) | (a[..., 2] > 127)).astype(np.float32))[None, None]
    soft = lambda t, k: F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="replicate"), k, stride=1)[0]  # noqa: E731
    h = torch.from_numpy((a[..., 2] > 127).astype(np.float32))[None, None]
    return torch.cat([soft(F.max_pool2d(m, 41, stride=1, padding=20), 21), soft(h, 7)])


def to_t(im):
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 127.5 - 1


def pick_window(w, h, size, focus):
    """A square window of a full frame: native scale most of the time, otherwise zoomed in up to 2.5x (close looks at
    skin, cloth, hair). focus() gives a point to stay near (a person), used 70% of the time."""
    s = 1.0 if random.random() < 0.6 else random.uniform(1.0, 2.5)
    k = min(int(round(size / s)), w, h)
    cx, cy = focus() if focus is not None and random.random() < 0.7 else (random.uniform(0, w), random.uniform(0, h))
    x0 = int(min(max(cx - k / 2 + random.uniform(-0.25, 0.25) * k, 0), w - k))
    y0 = int(min(max(cy - k / 2 + random.uniform(-0.25, 0.25) * k, 0), h - k))
    return x0, y0, k


def reg_of(p):
    return p[:-4] + ".reg.png"


GB_SUFFIX = (".alb.jpg", ".nrm.png", ".dep.png", ".reg.png")
GB_CH = 11  # albedo 3, normal 3, log depth 1, region 3 (face, hands, hair), present-flag 1


def images(d):
    """Frame images of a folder: *.jpg / *.png without the G-buffer and mask companions (nts/gbuf.py)."""
    ps = glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png"))
    return sorted(p for p in ps if not p.endswith(GB_SUFFIX))


def load_gb(p, box, S):
    """G-buffer of frame image p cropped to box (x0, y0, x1, y1) and resized to S: (GB_CH, S, S), all zeros (flag 0)
    when the frame has no G-buffer files."""
    stem = p[:-4]
    sz = (S, S) if isinstance(S, int) else tuple(S)  # (w, h) for a whole frame
    if not os.path.exists(stem + ".alb.jpg"):
        return torch.zeros(GB_CH, sz[1], sz[0])

    def part(path, mode, scale, shift, nearest=False):
        im = Image.open(path)
        if mode:
            im = im.convert(mode)
        im = im.crop(box)
        if im.size != sz:
            im = im.resize(sz, Image.NEAREST if nearest else Image.BILINEAR)
        a = np.asarray(im, np.float32)
        a = a[..., None] if a.ndim == 2 else a
        return torch.from_numpy(a.copy()).permute(2, 0, 1) * scale + shift
    alb = part(stem + ".alb.jpg", "RGB", 2 / 255, -1)
    nrm = part(stem + ".nrm.png", "RGB", 2 / 255, -1)
    z = lambda c: torch.zeros(c, sz[1], sz[0])  # noqa: E731
    dep = part(stem + ".dep.png", None, 1 / 65535, 0) if os.path.exists(stem + ".dep.png") else z(1)
    reg = part(stem + ".reg.png", "RGB", 1 / 255, 0, True) if os.path.exists(stem + ".reg.png") else z(3)
    return torch.cat([alb, nrm, dep, (reg > 0.5).float(), torch.ones(1, sz[1], sz[0])])


class Data:
    def __init__(self, root, size):
        self.size = size
        self.cg = sorted(glob.glob(os.path.join(root, "cg", "*.jpg")))
        masked = [p for p in self.cg if os.path.exists(reg_of(p))]
        if len(masked) > 200:  # run 2: only renders that come with a head mask, so the background can be left alone
            self.cg = masked
        self.real = sorted(glob.glob(os.path.join(root, "real", "*.jpg")))
        self.pairs = []  # (previous image, current image, motion file)
        for mv in sorted(glob.glob(os.path.join(root, "cg", "*.mv.npy"))):
            cur = mv[:-7] + ".jpg"
            stem, f = cur[:-4].rsplit("_", 1)
            prev = f"{stem}_{int(f) - 1}.jpg"
            if os.path.exists(cur) and os.path.exists(prev):
                self.pairs.append((prev, cur, mv))
        log(f"data: {len(self.cg)} cg images, {len(self.pairs)} pairs, {len(self.real)} real")
        # whole frames: cgf/ full-size renders (+ region masks, motion at 1/8 size), realf/ photos sized like them
        self.cgf = images(os.path.join(root, "cgf"))
        self.realf = sorted(glob.glob(os.path.join(root, "realf", "*.jpg")))
        self.match = None  # set by build_match(): matched (CG window, real windows) for frame tiles
        self.fpairs = []
        for mv in sorted(glob.glob(os.path.join(root, "cgf", "*.mv.npy"))):
            cur = mv[:-7] + ".jpg"
            stem, f = cur[:-4].rsplit("_", 1)
            prev = f"{stem}_{int(f) - 1}.jpg"
            if os.path.exists(cur) and os.path.exists(prev):
                self.fpairs.append((prev, cur, mv))
        self.faces = {}
        sj = os.path.join(root, "realf", "scenes.jsonl")
        if os.path.exists(sj):
            for l in open(sj):
                r = json.loads(l)
                if r.get("kept"):
                    self.faces[r["photo"]] = r["faces"]
        self.realx = sorted(glob.glob(os.path.join(A.realx, "*.jpg"))) if A.realx else []
        if self.realx:
            log(f"extra real photos: {len(self.realx)}")
        self.frames = bool(self.cgf and self.realf)
        if self.frames:
            log(f"frames: {len(self.cgf)} cg frames, {len(self.fpairs)} pairs, {len(self.realf)} real scenes")

    def realf_batch(self, b):
        out = []
        for _ in range(b):
            extra = self.realx and random.random() < A.p_realx
            p = random.choice(self.realx if extra else self.realf)
            im = Image.open(p).convert("RGB")
            faces = None if extra else self.faces.get(os.path.basename(p)[:-4])

            def focus():
                x0, y0, x1, y1 = random.choice(faces)
                fw, fh = x1 - x0, y1 - y0
                return random.uniform(x0 - 1.5 * fw, x1 + 1.5 * fw), random.uniform(y0 - 0.5 * fh, y1 + 4 * fh)
            x0, y0, k = pick_window(im.width, im.height, self.size, focus if faces else None)
            t = to_t(im.crop((x0, y0, x0 + k, y0 + k)).resize((self.size, self.size), Image.BICUBIC))
            out.append(t.flip(-1) if random.random() < 0.5 else t)
        return torch.stack(out)

    def matched_real(self, picks, b):
        """Real tiles for the CG windows just drawn: each one a random match of its CG window (build_match)."""
        M, out = self.match, []
        for i in picks[:b]:
            nb = M["nbr"][i]
            j = int(random.choice(nb[nb >= 0]))
            f, x0, y0, k = M["real"][j]
            im = Image.open(M["real_files"][f]).convert("RGB")
            j0 = int(k * 0.08)  # small jitter so the same window is never seen pixel-identical twice
            x0 = int(min(max(x0 + random.randint(-j0, j0), 0), im.width - k))
            y0 = int(min(max(y0 + random.randint(-j0, j0), 0), im.height - k))
            t = to_t(im.crop((x0, y0, x0 + k, y0 + k)).resize((self.size, self.size), Image.BICUBIC))
            out.append(t.flip(-1) if random.random() < 0.5 else t)
        return torch.stack(out)

    def cgf_batch(self, b, paired):
        """Tiles of full frames -> (prev, cur, motion, masks cur, masks prev, gbuf cur, gbuf prev, picks). The whole
        tile may change (mask channel 0 is all ones). With matching on, windows come from the matched CG list and
        picks names them, so matched_real() can draw their real partners."""
        S, prevs, curs, mvs, ms, gcs, gps, picks = self.size, [], [], [], [], [], [], []
        paired = paired and bool(self.fpairs)
        M = self.match
        for _ in range(b):
            if M is not None:
                pool = M["cg_paired"] if paired and len(M["cg_paired"]) else M["cg_ok"]
                i = int(pool[random.randrange(len(pool))]) if random.random() > 0.3 or not len(M["cg_person"]) else \
                    int(M["cg_person"][random.randrange(len(M["cg_person"]))])  # 30% from windows with a person in them
                if paired and i not in M["cg_paired_set"]:
                    i = int(M["cg_paired"][random.randrange(len(M["cg_paired"]))]) if len(M["cg_paired"]) else i
                picks.append(i)
                f, x0, y0, k = M["cg"][i]
                cp = self.cgf[f]
                pp, mp_ = M["prev_of"].get(cp, (None, None))
                if not paired:
                    pp = mp_ = None
            else:
                pp, cp, mp_ = random.choice(self.fpairs) if paired else (None, random.choice(self.cgf), None)
            im = Image.open(cp).convert("RGB")
            w, h = im.size
            reg = np.asarray(Image.open(reg_of(cp)).convert("RGB")) if os.path.exists(reg_of(cp)) else None

            def focus():
                ys, xs = np.nonzero(reg[::8, ::8].max(-1) > 127)
                if len(ys) == 0:
                    return random.uniform(0, w), random.uniform(0, h)
                i = random.randrange(len(ys))
                return xs[i] * 8, ys[i] * 8
            if M is not None:
                j0 = int(k * 0.08)
                x0 = int(min(max(x0 + random.randint(-j0, j0), 0), w - k))
                y0 = int(min(max(y0 + random.randint(-j0, j0), 0), h - k))
            else:
                x0, y0, k = pick_window(w, h, S, focus if reg is not None else None)
            box = (x0, y0, x0 + k, y0 + k)
            if A.gbuf:
                gcs.append(load_gb(cp, box, S))
                if pp is not None and paired:
                    gps.append(load_gb(pp, box, S))
            curs.append(to_t(im.crop(box).resize((S, S), Image.BICUBIC)))
            hair = torch.zeros(1, 1, S, S)
            if reg is not None:
                hb = torch.from_numpy((reg[y0:y0 + k, x0:x0 + k, 2] > 127).astype(np.float32))[None, None]
                hair = F.interpolate(hb, size=(S, S), mode="bilinear", align_corners=False)
            keep = torch.zeros(1, 1, S, S)  # faces and hands: held to the render like before
            if reg is not None:
                kb = torch.from_numpy((reg[y0:y0 + k, x0:x0 + k, :2].max(-1) > 127).astype(np.float32))[None, None]
                keep = F.max_pool2d(F.interpolate(kb, size=(S, S), mode="bilinear", align_corners=False), 9, 1, 4)
            ms.append(torch.cat([torch.ones(1, S, S), hair[0], keep[0]]))
            if paired and pp is not None:
                prevs.append(to_t(Image.open(pp).convert("RGB").crop(box).resize((S, S), Image.BICUBIC)))
                m = torch.from_numpy(np.load(mp_).astype(np.float32)).permute(2, 0, 1)[None]
                m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[..., y0:y0 + k, x0:x0 + k]
                mvs.append(F.interpolate(m, size=(S, S), mode="bilinear", align_corners=False)[0] * (S / k))
        m = torch.stack(ms)
        gc = torch.stack(gcs) if A.gbuf else None
        if paired and len(prevs) == b:
            gp = torch.stack(gps) if A.gbuf and len(gps) == b else (torch.zeros_like(gc) if A.gbuf else None)
            return torch.stack(prevs), torch.stack(curs), torch.stack(mvs), m, m, gc, gp, picks
        return None, torch.stack(curs), None, m, None, gc, None, picks

    def real_batch(self, b):
        x = torch.stack([load_img(random.choice(self.real), self.size) for _ in range(b)])
        flip = torch.rand(b) < 0.5
        x[flip] = x[flip].flip(-1)
        return x

    def cg_batch(self, b, paired):
        """-> prev (or None), cur, motion (b,2,H,W in pixels of this size, or None), head masks (cur, prev).
        No flips on pairs: motion has a sign."""
        if paired and self.pairs:
            ps = [random.choice(self.pairs) for _ in range(b)]
            prev = torch.stack([load_img(p[0], self.size) for p in ps])
            cur = torch.stack([load_img(p[1], self.size) for p in ps])
            mv = []
            for p in ps:
                m = torch.from_numpy(np.load(p[2]).astype(np.float32)).permute(2, 0, 1)[None]
                mv.append(F.interpolate(m, size=(self.size, self.size), mode="bilinear", align_corners=False)[0])
            mc = torch.stack([load_mask(reg_of(p[1]), self.size) for p in ps])
            mp = torch.stack([load_mask(reg_of(p[0]), self.size) for p in ps])
            g = torch.zeros(b, GB_CH, self.size, self.size) if A.gbuf else None  # head crops have no G-buffer
            return prev, cur, torch.stack(mv), mc, mp, g, g, []
        ps = [random.choice(self.cg) for _ in range(b)]
        cur = torch.stack([load_img(p, self.size) for p in ps])
        mc = torch.stack([load_mask(reg_of(p), self.size) for p in ps])
        flip = torch.rand(b) < 0.5
        cur[flip], mc[flip] = cur[flip].flip(-1), mc[flip].flip(-1)
        return None, cur, None, mc, None, (torch.zeros(b, GB_CH, self.size, self.size) if A.gbuf else None), None, []


def warp(prev, mv):
    """prev (b,c,H,W) sampled where each current pixel was in the previous frame; mv in pixels (x, y), y down."""
    b, _, h, w = prev.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=prev.device), torch.arange(w, device=prev.device), indexing="ij")
    gx = (xs[None] + mv[:, 0] + 0.5) / w * 2 - 1
    gy = (ys[None] + mv[:, 1] + 0.5) / h * 2 - 1
    inside = (gx.abs() < 1) & (gy.abs() < 1)
    return F.grid_sample(prev, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=False), inside[:, None]


def blur(x, k=33):
    return F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, stride=1)


# -------------------------------------------------------------------------------------------------------- models
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        if A.smoke:
            from diffusers import AutoencoderKL, UNet2DConditionModel
            self.vae = AutoencoderKL(block_out_channels=(16, 32), down_block_types=("DownEncoderBlock2D",) * 2,
                                     up_block_types=("UpDecoderBlock2D",) * 2, latent_channels=4, norm_num_groups=8)
            self.unet = UNet2DConditionModel(block_out_channels=(16, 32), layers_per_block=1, cross_attention_dim=16,
                                             down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                             up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), norm_num_groups=8,
                                             attention_head_dim=4)
            self.register_buffer("ctx", torch.randn(1, 4, 16))
        else:
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(A.base, torch_dtype=torch.float32, safety_checker=None)
            try:  # few-step distilled weights: a much better one-step starting point than the base model
                pipe.load_lora_weights(A.lcm)
                pipe.fuse_lora()
                pipe.unload_lora_weights()
                log("fused", A.lcm)
            except Exception as e:
                log("LCM-LoRA not fused:", repr(e)[:200])
            self.vae, self.unet = pipe.vae, pipe.unet
            with torch.no_grad():
                ids = pipe.tokenizer(["a candid photograph, natural skin, real hair, real fabric, real room"],
                                     padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                     truncation=True, return_tensors="pt").input_ids
                self.register_buffer("ctx", pipe.text_encoder(ids)[0])
            del pipe
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        from peft import LoraConfig
        targets = ["to_k", "to_q", "to_v", "to_out.0", "conv1", "conv2", "conv_shortcut", "proj_in", "proj_out",
                   "ff.net.0.proj", "ff.net.2"]
        self.unet.add_adapter(LoraConfig(r=A.rank, lora_alpha=A.rank, init_lora_weights="gaussian", target_modules=targets))
        nn.init.zeros_(self.unet.conv_out.weight)  # the output is a residual on the input latent: start at zero
        nn.init.zeros_(self.unet.conv_out.bias)
        self.unet.conv_out.requires_grad_(True)
        if not A.smoke:  # 512 px on a 14.5 GB card: recompute activations in the backward pass instead of storing them
            self.unet.enable_gradient_checkpointing()
            self.vae.enable_gradient_checkpointing()
        self.sf = getattr(self.vae.config, "scaling_factor", 0.18215)
        self.gb, self._gfeat = None, None
        if A.gbuf:  # G-buffer adapter: pixels -> latent grid (/8), zero last conv, added to conv_in's output
            c = self.unet.conv_in.out_channels
            n_down = len(self.vae.config.block_out_channels) - 1  # pixels -> latent: /8 for SD 1.5
            chs = [32, 64, 128, 256][:n_down + 1]
            layers = [nn.Conv2d(GB_CH, chs[0], 3, padding=1), nn.SiLU()]
            for i, o in zip(chs[:-1], chs[1:]):
                layers += [nn.Conv2d(i, o, 3, 2, 1), nn.SiLU()]
            self.gb = nn.Sequential(*layers, nn.Conv2d(chs[-1], c, 1))
            nn.init.zeros_(self.gb[-1].weight)
            nn.init.zeros_(self.gb[-1].bias)
            self.unet.conv_in.register_forward_hook(lambda m, i, o: o if self._gfeat is None else o + self._gfeat.to(o.dtype))

    def params(self):
        return [p for p in self.unet.parameters() if p.requires_grad]

    def gb_params(self):
        return list(self.gb.parameters()) if self.gb is not None else []

    def state(self):
        return {k: v.detach().cpu() for k, v in self.unet.state_dict().items() if "lora" in k or k.startswith("conv_out")}

    def forward(self, x, g=None):
        with torch.no_grad():
            z = self.vae.encode(x).latent_dist.mode() * self.sf
        t = torch.full((x.shape[0],), A.t, device=x.device, dtype=torch.long)
        if self.gb is not None:
            g = torch.zeros(x.shape[0], GB_CH, *x.shape[-2:], device=x.device) if g is None else g.to(x.device)
            self._gfeat = self.gb(g)
        r = self.unet(z, t, encoder_hidden_states=self.ctx.expand(x.shape[0], -1, -1)).sample
        self._gfeat = None
        y = self.vae.decode((z + r) / self.sf).sample
        if A.vae_res:  # cancel the VAE's own reconstruction error (fine text, edges): only the learned change remains
            with torch.no_grad():
                y0 = self.vae.decode(z / self.sf).sample
            y = x + (y - y0)
        return y.clamp(-1, 1)


class Disc(nn.Module):
    """Projected discriminator on frozen DINOv2-S patch tokens from four layers; sees random crops at several scales."""

    def __init__(self):
        super().__init__()
        if A.smoke:
            self.net, dim, self.layers = None, 3, [0]
        else:
            from transformers import AutoModel
            self.net = AutoModel.from_pretrained("facebook/dinov2-small").eval().requires_grad_(False)
            dim, self.layers = self.net.config.hidden_size, [3, 6, 9, 12]
        sn = nn.utils.spectral_norm
        self.heads = nn.ModuleList(nn.Sequential(sn(nn.Conv2d(dim, 256, 1)), nn.LeakyReLU(0.2), sn(nn.Conv2d(256, 256, 3, padding=1)),
                                                 nn.LeakyReLU(0.2), sn(nn.Conv2d(256, 1, 1))) for _ in self.layers)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def crops(self, x, n=3):
        """Same random boxes for every image in the batch: the whole head, and two closer looks (skin, hair)."""
        out, s = [], x.shape[-1]
        for i in range(n):
            k = s if i == 0 else int(s * random.uniform(0.3, 0.6))
            x0, y0 = random.randint(0, s - k), random.randint(0, s - k)
            c = x[..., y0:y0 + k, x0:x0 + k]
            out.append(F.interpolate(c, size=(224, 224), mode="bilinear", align_corners=False, antialias=k > 224))
        return torch.cat(out)

    def feats(self, x):
        x = self.crops(x)
        if self.net is None:
            return [F.avg_pool2d(x, 14)]
        hs = self.net(pixel_values=((x + 1) / 2 - self.mean) / self.std, output_hidden_states=True).hidden_states
        return [hs[i][:, 1:].transpose(1, 2).reshape(x.shape[0], -1, 16, 16) for i in self.layers]

    def heads_out(self, feats):
        return torch.cat([h(f.float()).flatten(1) for h, f in zip(self.heads, feats)], 1)

    def forward(self, x):
        return self.heads_out(self.feats(x))

    @torch.no_grad()
    def embed(self, x):
        """(n,3,224,224) in [-1,1] -> unit-length content embedding (DINOv2 class token) for crop matching."""
        if self.net is None:
            return F.normalize(F.adaptive_avg_pool2d(x, 4).flatten(1).float(), dim=1)
        h = self.net(pixel_values=((x + 1) / 2 - self.mean) / self.std).last_hidden_state
        return F.normalize(h[:, 0].float(), dim=1)


class Identity(nn.Module):
    """Embedding of the centre of the head box (where the face sits) from frozen DINOv2-S: class token plus mean
    patch token. Run a5 used FaceNet here; its weights come from VGGFace2, a research-only dataset, so it was
    replaced to keep every network in the training chain under a licence that allows commercial use."""

    def __init__(self):
        super().__init__()
        self.net = None
        if not A.smoke:
            try:
                from transformers import AutoModel
                self.net = AutoModel.from_pretrained("facebook/dinov2-small").eval().requires_grad_(False)
            except Exception as e:
                log("identity loss off:", repr(e)[:200])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, full=False):
        s = x.shape[-1]
        c = x if full else x[..., int(0.22 * s):int(0.80 * s), int(0.21 * s):int(0.79 * s)]  # full: a frame tile
        c = F.interpolate(c, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        h = self.net(pixel_values=((c + 1) / 2 - self.mean) / self.std).last_hidden_state
        return F.normalize(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float(), dim=1)


MATCH_SIZES = (288, 416, 576)  # window sides in source pixels: ~4-16% of a 1080x1920 frame (EPE uses ~7% crops)


def windows(w, h):
    out = []
    for k in MATCH_SIZES:
        if k > min(w, h):
            continue
        st = int(k * 0.75)
        xs = sorted(set(list(range(0, w - k + 1, st)) + [w - k]))
        ys = sorted(set(list(range(0, h - k + 1, st)) + [h - k]))
        out += [(x, y, k) for y in ys for x in xs]
    return out


MATCH_BS = 256  # windows per embedding batch (the CPU match test uses fewer to keep memory low)


@torch.no_grad()
def build_match(data, D):
    """Content-matched crops (Richter et al. 2021, sec. 3.3, with DINOv2 instead of VGG): every CG frame and real photo
    is cut into fixed windows, each window embedded once, and each CG window keeps its --match-k most similar real
    windows above --match-thr. Writes out/match.npz and out/match_sheet.jpg (random CG windows next to their best
    match) so the pairing can be checked by eye."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    real_files = data.realf + data.realx
    cg, person, real = [], [], []
    for fi, p in enumerate(data.cgf):
        w, h = Image.open(p).size
        reg = None
        if os.path.exists(reg_of(p)):
            reg = np.asarray(Image.open(reg_of(p)).convert("RGB"))[::8, ::8].max(-1) > 127
        for x, y, k in windows(w, h):
            cg.append((fi, x, y, k))
            person.append(bool(reg is not None and reg[y // 8:(y + k) // 8, x // 8:(x + k) // 8].any()))
    for fi, p in enumerate(real_files):
        w, h = Image.open(p).size
        real += [(fi, x, y, k) for x, y, k in windows(w, h)]
    cg, real, person = np.array(cg, np.int32), np.array(real, np.int32), np.array(person, bool)
    log(f"match: {len(cg)} CG windows from {len(data.cgf)} frames, {len(real)} real windows from {len(real_files)} photos")

    def crops_of(files, wins):
        by = {}
        for i, (f, x, y, k) in enumerate(wins):
            by.setdefault(int(f), []).append((i, x, y, k))

        def one(f):
            im = Image.open(files[f]).convert("RGB")
            return [(i, np.asarray(im.crop((x, y, x + k, y + k)).resize((224, 224), Image.BILINEAR))) for i, x, y, k in by[f]]
        with ThreadPoolExecutor(4) as ex:
            for lst in ex.map(one, sorted(by)):
                yield from lst

    def embed(files, wins):
        dev = D.mean.device
        E = torch.zeros(len(wins), 48 if D.net is None else D.net.config.hidden_size, dtype=torch.float16)
        buf, idx = [], []

        def flush():
            x = torch.from_numpy(np.stack(buf)).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                E[idx] = D.embed(x).half().cpu()
            buf.clear()
            idx.clear()
        for i, a in crops_of(files, wins):
            buf.append(a)
            idx.append(i)
            if len(buf) == MATCH_BS:
                flush()
        if buf:
            flush()
        return E
    ecg, ere = embed(data.cgf, cg), embed(real_files, real)
    log(f"match: embedded in {(time.time() - t0) / 60:.1f} min")
    dev = D.mean.device
    er = ere.to(dev)
    K = min(A.match_k, len(real))
    nbr = np.full((len(cg), K), -1, np.int32)
    sim = np.zeros((len(cg), K), np.float16)
    for s in range(0, len(cg), 512):
        sc = ecg[s:s + 512].to(dev).float() @ er.float().T
        v, j = sc.topk(K, dim=1)
        v, j = v.cpu().numpy(), j.cpu().numpy()
        thr = -2.0 if A.smoke else A.match_thr
        nbr[s:s + 512] = np.where(v > thr, j, -1)
        sim[s:s + 512] = v
    ok = (nbr >= 0).any(1)
    log(f"match: {ok.mean() * 100:.0f}% of CG windows have a match > {A.match_thr}; top-1 similarity median "
        f"{np.median(sim[:, 0]):.2f}, p10 {np.percentile(sim[:, 0], 10):.2f}; real windows used "
        f"{len(np.unique(nbr[nbr >= 0]))} of {len(real)}")
    np.savez_compressed(os.path.join(A.out, "match.npz"), cg=cg, real=real, nbr=nbr, sim=sim, person=person)
    json.dump({"cg_files": data.cgf, "real_files": real_files}, open(os.path.join(A.out, "match_files.json"), "w"))
    prev_of = {cur: (prev, mv) for prev, cur, mv in data.fpairs}
    has_prev = np.array([data.cgf[f] in prev_of for f in cg[:, 0]], bool)
    data.match = dict(cg=cg, real=real, nbr=nbr, real_files=real_files, prev_of=prev_of,
                      cg_ok=np.nonzero(ok)[0], cg_person=np.nonzero(ok & person)[0], cg_paired=np.nonzero(ok & has_prev)[0])
    data.match["cg_paired_set"] = set(data.match["cg_paired"].tolist())
    rows = []
    for i in np.random.default_rng(0).choice(data.match["cg_ok"], size=min(6, ok.sum()), replace=False):
        f, x, y, k = cg[i]
        a = Image.open(data.cgf[f]).convert("RGB").crop((x, y, x + k, y + k)).resize((192, 192))
        f2, x2, y2, k2 = real[nbr[i][nbr[i] >= 0][0]]
        b = Image.open(real_files[f2]).convert("RGB").crop((x2, y2, x2 + k2, y2 + k2)).resize((192, 192))
        rows.append([to_t(a), to_t(b)])
    if rows:
        grid([[r[0] for r in rows], [r[1] for r in rows]], os.path.join(A.out, "match_sheet.jpg"))


def grid(rows, path):
    ims = [torch.cat([r.cpu() for r in row], -1) for row in rows]
    a = ((torch.cat(ims, -2).clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(a).save(path, quality=90)


@torch.no_grad()
def run_eval(G, tag):
    src = sorted(glob.glob(os.path.join(A.data, "eval", "*.jpg")))
    if not src:
        return
    d = os.path.join(A.out, "eval" if tag is None else f"eval_{tag}")
    os.makedirs(d, exist_ok=True)
    G.eval()
    for i in range(0, len(src), 4):
        x = torch.stack([load_img(p, A.size) for p in src[i:i + 4]]).to(DEV)
        m = torch.stack([load_mask(reg_of(p), A.size) for p in src[i:i + 4]])[:, :1].to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
            y = x * (1 - m) + G(x).float() * m
        for p, o in zip(src[i:i + 4], y):
            a = ((o.float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            Image.fromarray(a).save(os.path.join(d, os.path.basename(p)[:-4] + ".png"))
    G.train()
    log(f"eval: {len(src)} crops -> {d}")


@torch.no_grad()
def run_evalf(G):
    """Whole frames (evalf/*.jpg|png): overlapping tiles blended with a ramp -> out/evalf/<name>.jpg."""
    src = images(os.path.join(A.data, "evalf"))
    if A.infer_only:  # e.g. "at140,zd400": only these clips (inference jobs split by clip)
        src = [p for p in src if any(f"_{t}_" in os.path.basename(p) for t in A.infer_only.split(","))]
    if not src:
        return
    d = os.path.join(A.out, "evalf")
    os.makedirs(d, exist_ok=True)
    S = A.size
    st = S * 3 // 4
    r = (torch.minimum(torch.arange(S) + 1, S - torch.arange(S)).float() / (S - st)).clamp(max=1)
    win = (r[:, None] * r[None, :])[None].to(DEV)
    G.eval()
    for p in src:
        x = to_t(Image.open(p).convert("RGB")).to(DEV)
        H, W = x.shape[1:]
        g = load_gb(p, (0, 0, W, H), (W, H)).to(DEV) if A.gbuf else None
        pos = [(yy, xx) for yy in sorted(set(list(range(0, H - S, st)) + [H - S])) for xx in sorted(set(list(range(0, W - S, st)) + [W - S]))]
        acc, ws = torch.zeros_like(x), torch.zeros(1, H, W, device=DEV)
        for i in range(0, len(pos), 4):
            t = torch.stack([x[:, yy:yy + S, xx:xx + S] for yy, xx in pos[i:i + 4]])
            gt = torch.stack([g[:, yy:yy + S, xx:xx + S] for yy, xx in pos[i:i + 4]]) if g is not None else None
            with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
                y = G(t, gt).float()
            for (yy, xx), o in zip(pos[i:i + 4], y):
                acc[:, yy:yy + S, xx:xx + S] += o * win
                ws[:, yy:yy + S, xx:xx + S] += win
        a = (((acc / ws).clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        Image.fromarray(a).save(os.path.join(d, os.path.basename(p)[:-4] + ".jpg"), quality=95)
    G.train()
    log(f"evalf: {len(src)} frames -> {d}")


def main():
    t_start = time.time()
    random.seed(0)
    torch.manual_seed(0)
    if A.smoke:
        A.size, A.steps, A.gan_start = 64, 3, 1
    G = Generator().to(DEV)
    if A.resume or A.infer:
        sd = torch.load(A.resume or A.infer, map_location="cpu")
        missing = G.unet.load_state_dict(sd["g"], strict=False)
        if G.gb is not None and "gb" in sd:
            G.gb.load_state_dict(sd["gb"])
        log("loaded", A.resume or A.infer, "step", sd.get("step"), "unexpected", len(missing.unexpected_keys),
            "gbuf adapter", "gb" in sd)
    if A.infer:
        run_evalf(G)
        return run_eval(G, None)
    data = Data(A.data, A.size)
    G.train()  # no dropout or batch norm in these networks; train mode is what switches gradient checkpointing on
    D, ID = Disc().to(DEV2), Identity().to(DEV2)
    if A.smoke:
        lp = lambda a, b: (a - b).abs().mean()  # noqa: E731
    else:
        import lpips
        lpn = lpips.LPIPS(net="vgg").to(DEV2).eval().requires_grad_(False)
        lp = lambda a, b: lpn(a.float().to(DEV2), b.float().to(DEV2)).mean().to(DEV)  # noqa: E731
    if A.match:
        build_match(data, D)
    groups = [{"params": G.params()}]
    if G.gb is not None:  # the adapter starts from scratch (zero output), so it gets a larger step than the LoRA
        G.gb.to(DEV)
        groups.append({"params": G.gb_params(), "lr": A.lr * 10})
    og = torch.optim.AdamW(groups, lr=A.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    od = torch.optim.AdamW(D.heads.parameters(), lr=A.lr * 5, betas=(0.5, 0.999))
    sg, sd_ = torch.cuda.amp.GradScaler(enabled=AMP), torch.cuda.amp.GradScaler(enabled=AMP)
    step0 = 0
    if A.resume:
        ck = torch.load(A.resume, map_location="cpu")
        step0 = ck.get("step", 0)
        if A.steps - step0 < 2000:
            log(f"WARNING: resuming at step {step0} with --steps {A.steps}: only {max(A.steps - step0, 0)} steps left")
        if "d" in ck:
            D.heads.load_state_dict(ck["d"])
    log(f"trainable G params {sum(p.numel() for p in G.params()) / 1e6:.1f}M, device {DEV}, args {vars(A)}")
    fb = data.cg_batch(4, False)
    if data.frames:  # two head crops and two frame tiles in the sample sheet
        ft = data.cgf_batch(2, False)
        fg = torch.cat([fb[5][:2], ft[5]]) if A.gbuf else None
        fb = (None, torch.cat([fb[1][:2], ft[1]]), None, torch.cat([fb[3][:2], ft[3][:, :2]]), None, fg)
    fixed, fixed_m = fb[1].to(DEV), fb[3][:, :1].to(DEV)
    fixed_g = fb[5].to(DEV) if A.gbuf else None
    acc = {}
    for step in range(step0 + 1, A.steps + 1):
        paired = step % 2 == 0
        tile = data.frames and random.random() < A.p_frame  # this step: whole-frame tiles, or head crops
        prev, x, mv, mc, mp, gc, gp, picks = (data.cgf_batch if tile else data.cg_batch)(A.batch, paired)
        x, mc, hair, keep = x.to(DEV), mc[:, :1].to(DEV), mc[:, 1:2].to(DEV), mc[:, 2:3].to(DEV)
        if tile and picks:  # matched: each CG window is judged against real windows of similar content
            real = data.matched_real(picks, A.batch).to(DEV)
        else:
            real = (data.realf_batch if tile else data.real_batch)(A.batch).to(DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
            y = x * (1 - mc) + G(x, gc).float() * mc  # only the head changes; the rest of the box is the render itself
            hide = (1 - A.w_hair) * hair  # what the keep-losses do not see: most of the change in hair
            if tile and A.w_free < 1:
                lum = x.float().mean(1, keepdim=True)  # print and screen text: sharp, high local contrast -> kept
                sharp = (F.max_pool2d(lum, 5, 1, 2) + F.max_pool2d(-lum, 5, 1, 2) > 0.6).float()
                free = (1 - keep) * (1 - F.max_pool2d(sharp, 9, 1, 4)) * (1 - hair)
                hide = hide + (1 - A.w_free) * free
            yk = x + (y - x) * (1 - hide)
            loss = {"lpips": A.w_lpips * lp(yk, x), "low": A.w_low * (blur(y) - blur(x)).abs().mean()}
            if A.w_haircol > 0:  # hair keeps the render's colour and brightness; only its fine structure is free
                k = max(3, (A.size // 24) | 1)
                loss["hcol"] = A.w_haircol * ((blur(y, k) - blur(x, k)).abs() * hair).sum() / (hair.sum() * 3 + 1)
            if tile and A.w_band > 0:  # mid band (blobs, smears, blotches) held on skin and smooth surfaces
                xf, yf = x.float(), y.float()
                bnd = lambda z: blur(z, A.band_lo | 1) - blur(z, A.band_hi | 1)  # noqa: E731
                lum = xf.mean(1, keepdim=True)
                sd = (blur(lum * lum, 9) - blur(lum, 9) ** 2).clamp(min=0).sqrt()
                reg = torch.maximum((sd < 0.024).float(), keep) * (1 - hair)  # 0.024 = 3/255 std in [-1, 1] units
                loss["band"] = A.w_band * ((bnd(yf) - bnd(xf)).abs() * reg).sum() / (reg.sum() * 3 + 1)
            if ID.net is not None:
                with torch.no_grad():
                    ex = ID(x.float().to(DEV2), tile)
                loss["id"] = A.w_id * (1 - (ID(yk.float().to(DEV2), tile) * ex).sum(1)).mean().to(DEV)
            if prev is not None:
                prev, mv = prev.to(DEV), mv.to(DEV)
                with torch.no_grad():
                    mp = mp[:, :1].to(DEV)
                    yp = prev * (1 - mp) + G(prev, gp).float() * mp
                    wy, inside = warp(yp.float(), mv)
                    wx, _ = warp(prev, mv)
                    ok = (inside & ((wx - x).abs().mean(1, keepdim=True) < 0.08)).float()  # same surface in both inputs
                loss["temp"] = A.w_temp * ((y.float() - wy).abs() * ok).sum() / (ok.sum() * 3 + 1)
            if step >= A.gan_start:
                loss["gan"] = -A.w_gan * D(y.to(DEV2)).mean().to(DEV)
            tot = sum(loss.values())
        og.zero_grad(set_to_none=True)
        sg.scale(tot).backward()
        sg.unscale_(og)
        torch.nn.utils.clip_grad_norm_(G.params() + G.gb_params(), 1.0)
        sg.step(og)
        sg.update()
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=AMP):  # frozen DINOv2 features
            fr, ff = D.feats(real.to(DEV2)), D.feats(y.detach().to(DEV2))
        r1 = A.w_r1 > 0 and (step % 4 == 0 or A.smoke)  # lazy R1 (every 4th step, weight x4) on the features the heads see
        fr = [f.float().requires_grad_(r1) for f in fr]
        dr = D.heads_out(fr)
        ld = F.relu(1 - dr).mean() + F.relu(1 + D.heads_out([f.float() for f in ff])).mean()
        if r1:
            gr = torch.autograd.grad(dr.sum(), fr, create_graph=True)
            pen = sum(g.pow(2).flatten(1).sum(1).mean() for g in gr)
            ld = ld + A.w_r1 * 4 * 0.5 * pen
            acc["r1"] = acc.get("r1", 0.0) + float(pen) * 4
        od.zero_grad(set_to_none=True)
        sd_.scale(ld).backward()
        sd_.step(od)
        sd_.update()
        for k, v in {**loss, "d": ld}.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        if step % 50 == 0 or A.smoke:
            el = (time.time() - t_start) / 3600
            log(f"step {step} {el:.2f}h " + " ".join(f"{k} {v / (1 if A.smoke else 50):.4f}" for k, v in acc.items()))
            acc = {}
        last = step == A.steps or (time.time() - t_start) / 3600 > A.hours
        if step % 500 == 0 or last or A.smoke:
            with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
                G.eval()
                fy = fixed * (1 - fixed_m) + G(fixed, fixed_g).float() * fixed_m
                G.train()
            grid([list(fixed), list(fy)], os.path.join(A.out, f"sample_{step:06d}.jpg"))
        if step % 2000 == 0 or last or A.smoke:
            ck = {"g": G.state(), "d": D.heads.state_dict(), "step": step, "args": vars(A)}
            if G.gb is not None:
                ck["gb"] = {k: v.detach().cpu() for k, v in G.gb.state_dict().items()}
            torch.save(ck, os.path.join(A.out, "last.pt"))
            if step % 4000 == 0 or last:
                run_eval(G, None if last else f"{step:06d}")
            if last:
                run_evalf(G)
        if last:
            break
    json.dump({"steps": step, "hours": (time.time() - t_start) / 3600}, open(os.path.join(A.out, "DONE.json"), "w"))
    log("R2P_DONE", step)


if __name__ == "__main__":
    main()
