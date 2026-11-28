# Colab VM: generated realism targets at full 1080x1920, by detail transfer from SDXL + depth ControlNet
# (the realism-pass plan, "generated targets"). Our render keeps everything except fine texture:
#   target = render + gate * highpass(SDXL)
#   - SDXL img2img runs on overlapping 1024x1024 tiles at native resolution, so the added texture is pixel-scale,
#     and the tiles are blended with feathered weights.
#   - body: high-pass sigma 2.5 of a strength-0.45 pass; face: only the finest band (sigma 1) of a gentle 0.25 pass,
#     so identity and expression can't move.
#   - gate: 0 where the render already has strong edges (text, screen glyphs, logos, silhouettes), so those stay
#     exactly as rendered; 1 on smooth areas (skin, walls, cloth) that need texture.
# Inputs in /content/gen_in: <tag>_rgb.png, <tag>_depth.png (inverse depth), <tag>_face.png (face mask), 1080x1920.
# Writes /content/gen_out/<tag>_target.png, <tag>_gate.png and <tag>_check.jpg (render | target, + face crop).
# Models: stabilityai/stable-diffusion-xl-base-1.0 (CreativeML Open RAIL++-M), diffusers/controlnet-depth-sdxl-1.0.
import glob
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers>=0.30", "transformers", "accelerate"], check=True)
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
from PIL import Image

TILE, RAMP = 1024, 128
UP = int(os.environ.get("NTS_UP", 1))  # 2: generate on a 2x Lanczos upscale, then box-downsample (SDXL draws texture at
                                       # its own scale; at 1x its output has less fine detail than our renders)
BODY = dict(strength=float(os.environ.get("NTS_BODY", 0.45)), sigma=2.5, gain=1.0)
FACE = dict(strength=float(os.environ.get("NTS_FACE", 0.25)), sigma=1.0, gain=0.8)
EDGE_T = 0.05  # render high-pass (sigma 1.5, luminance) above this keeps the render untouched
PROMPT = ("raw photograph, photorealistic, natural skin texture with pores, detailed individual hair strands, "
          "realistic fabric weave, soft realistic lighting, subtle film grain, 35mm photo")
NEG = "cartoon, cgi, 3d render, plastic skin, smooth skin, video game, illustration, painting, blurry, deformed"
cn = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)
os.makedirs("/content/gen_out", exist_ok=True)


def t(img):
    return torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)[None].cuda()


def blur(x, sigma):
    r = int(3 * sigma + 0.5)
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma * sigma))
    k = (k / k.sum()).view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k, groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.transpose(2, 3), groups=x.shape[1])


def starts(n):
    k = max(1, -(-(n - RAMP) // (TILE - RAMP)))
    return [0] if n <= TILE else [round(i * (n - TILE) / (k - 1)) for i in range(k)]


def window():
    r = torch.arange(TILE, dtype=torch.float32)
    w = torch.minimum(torch.minimum((r + 1) / RAMP, (TILE - r) / RAMP), torch.ones(()))
    return (w[:, None] * w[None, :]).cuda()


def sdxl_tiled(img, depth, strength, seed):
    if UP > 1:
        W0, H0 = img.size
        img, depth = img.resize((W0 * UP, H0 * UP), Image.LANCZOS), depth.resize((W0 * UP, H0 * UP), Image.BILINEAR)
    W, H = img.size
    acc, wsum, win = torch.zeros(1, 3, H, W, device="cuda"), torch.zeros(1, 1, H, W, device="cuda"), window()
    for y in starts(H):
        for x in starts(W):
            box = (x, y, x + TILE, y + TILE)
            g = torch.Generator("cuda").manual_seed(seed + 7919 * y + x)
            out = pipe(prompt=PROMPT, negative_prompt=NEG, image=img.crop(box), control_image=depth.crop(box),
                       strength=strength, controlnet_conditioning_scale=0.8, num_inference_steps=30,
                       guidance_scale=5.0, generator=g).images[0]
            acc[..., y:y + TILE, x:x + TILE] += t(out) * win
            wsum[..., y:y + TILE, x:x + TILE] += win
    return F.avg_pool2d(acc / wsum, UP) if UP > 1 else acc / wsum


def lum(x):
    return (0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3])


def face_crop(face):
    ys, xs = np.nonzero(np.asarray(face) > 127)
    if len(ys) < 50:
        return None
    cy, cx, s = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2, int(1.6 * max(np.ptp(ys), np.ptp(xs))) // 2 + 8
    return (max(cx - s, 0), max(cy - s, 0), cx + s, cy + s)


for rgb_path in sorted(glob.glob("/content/gen_in/*_rgb.png")):
    tag, t0 = os.path.basename(rgb_path)[:-8], time.time()
    img = Image.open(rgb_path).convert("RGB")
    depth = Image.open(f"/content/gen_in/{tag}_depth.png").convert("RGB")
    face = Image.open(f"/content/gen_in/{tag}_face.png").convert("L")
    R = t(img)
    fm = (torch.from_numpy(np.asarray(face, np.float32) / 255)[None, None].cuda() > 0.5).float()
    fm = blur(F.max_pool2d(fm, 31, 1, 15), 8).clamp(0, 1)
    hp_r = (lum(R) - lum(blur(R, 1.5))).abs()
    gate = blur(1 - F.max_pool2d(hp_r / EDGE_T, 5, 1, 2).clamp(0, 1), 2)
    body = sdxl_tiled(img, depth, BODY["strength"], 1)
    to8 = lambda x: Image.fromarray((x[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).clip(0, 255).astype(np.uint8))  # noqa: E731
    to8(body).save(f"/content/gen_out/{tag}_sdxl_body.png")
    detail = (body - blur(body, BODY["sigma"])) * BODY["gain"]
    if fm.max() > 0:
        fs = sdxl_tiled(img, depth, FACE["strength"], 2)
        to8(fs).save(f"/content/gen_out/{tag}_sdxl_face.png")
        detail = detail * (1 - fm) + (fs - blur(fs, FACE["sigma"])) * FACE["gain"] * fm
    T = (R + gate * detail).clamp(0, 1)
    to8(gate.repeat(1, 3, 1, 1)).save(f"/content/gen_out/{tag}_gate.png")
    ti = to8(T)
    ti.save(f"/content/gen_out/{tag}_target.png")
    chk = Image.new("RGB", (2160, 1920 + 1080))
    chk.paste(img, (0, 0))
    chk.paste(ti, (1080, 0))
    box = face_crop(face) or (270, 480, 810, 1020)
    chk.paste(img.crop(box).resize((1080, 1080), Image.LANCZOS), (0, 1920))
    chk.paste(ti.crop(box).resize((1080, 1080), Image.LANCZOS), (1080, 1920))
    chk.save(f"/content/gen_out/{tag}_check.jpg", quality=90)
    print(f"done {tag} {time.time() - t0:.0f}s gate_mean {gate.mean().item():.2f}", flush=True)
print("GEN_TARGETS_DONE")
