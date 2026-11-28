# Colab VM: SDXL + depth ControlNet img2img feasibility test for generated realism targets (the realism-pass plan).
# Inputs in /content/gen_in: <tag>_rgb.png (our render), <tag>_depth.png (inverse depth), <tag>_face.png (face mask).
# Writes /content/gen_out/<tag>_strip.jpg: render | strength 0.3 | strength 0.5 | 0.5 with the face from a 0.2 pass.
# Models: stabilityai/stable-diffusion-xl-base-1.0 (CreativeML Open RAIL++-M), diffusers/controlnet-depth-sdxl-1.0.
import glob
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers>=0.30", "transformers", "accelerate"], check=True)
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 768, 1344
cn = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)
PROMPT = ("raw photograph, photorealistic, natural skin texture with pores, detailed individual hair strands, "
          "soft realistic lighting, subtle film grain, 35mm photo")
NEG = "cartoon, cgi, 3d render, plastic skin, smooth skin, video game, illustration, painting, blurry, deformed"
os.makedirs("/content/gen_out", exist_ok=True)


def gen(img, depth, strength, seed=7):
    g = torch.Generator("cuda").manual_seed(seed)
    return pipe(prompt=PROMPT, negative_prompt=NEG, image=img, control_image=depth, strength=strength,
                controlnet_conditioning_scale=0.8, num_inference_steps=30, guidance_scale=5.0, generator=g).images[0]


try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 28)
except OSError:
    font = ImageFont.load_default()
for rgb_path in sorted(glob.glob("/content/gen_in/*_rgb.png")):
    tag = os.path.basename(rgb_path)[:-8]
    img = Image.open(rgb_path).convert("RGB").resize((W, H), Image.LANCZOS)
    depth = Image.open(f"/content/gen_in/{tag}_depth.png").convert("RGB").resize((W, H), Image.BILINEAR)
    face = Image.open(f"/content/gen_in/{tag}_face.png").convert("L").resize((W, H), Image.NEAREST)
    face = face.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(10))
    a, b, c = gen(img, depth, 0.3), gen(img, depth, 0.5), gen(img, depth, 0.2)
    protected = Image.composite(c, b, face)
    tiles = [("our render", img), ("SDXL 0.3", a), ("SDXL 0.5", b), ("0.5, face from 0.2", protected)]
    strip = Image.new("RGB", (W * 4, H))
    d = ImageDraw.Draw(strip)
    for i, (n, t) in enumerate(tiles):
        strip.paste(t, (W * i, 0))
        d.rectangle([W * i, 0, W * i + W, 40], fill=(0, 0, 0))
        d.text((W * i + 8, 6), n, fill=(255, 255, 255), font=font)
    strip.save(f"/content/gen_out/{tag}_strip.jpg", quality=90)
    print("done", tag, flush=True)
print("GEN_TEST_DONE")
