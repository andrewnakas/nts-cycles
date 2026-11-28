"""Kaggle/Colab: SDXL img2img + tile ControlNet on 1024 px zoomed faces (tools/facezoom.py prep), several strengths.
Models: stabilityai/stable-diffusion-xl-base-1.0 (CreativeML Open RAIL++-M), xinsir/controlnet-tile-sdxl-1.0
(Apache-2.0), madebyollin/sdxl-vae-fp16-fix (MIT; plain SDXL VAE NaNs in fp16 on T4).
Run: python teacher_sdxl_face.py <zoom_dir> <out_dir> [strengths] ; writes <out>/s<strength>/f_####.png"""
import glob, os, sys, time
import torch
from PIL import Image
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

src, out = sys.argv[1], sys.argv[2]
strengths = [float(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "0.25,0.4").split(",")]
dt = torch.float16
cn = ControlNetModel.from_pretrained("xinsir/controlnet-tile-sdxl-1.0", torch_dtype=dt)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dt)
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn,
                                                                   vae=vae, torch_dtype=dt, variant="fp16").to("cuda")
prompt = ("raw close-up photograph of a man's face, natural skin texture with pores and fine blemishes, individual hair "
          "strands, soft natural light, 85mm lens, high detail")
neg = "cgi, 3d render, video game, plastic skin, airbrushed, smooth skin, makeup, painting, cartoon, blurry, deformed"
for f in sorted(glob.glob(os.path.join(src, "f_*.jpg"))):
    im = Image.open(f).convert("RGB")
    for s in strengths:
        od = os.path.join(out, f"s{s}"); os.makedirs(od, exist_ok=True)
        t0 = time.time()
        r = pipe(prompt, negative_prompt=neg, image=im, control_image=im, strength=s, num_inference_steps=30,
                 guidance_scale=5.0, controlnet_conditioning_scale=float(os.environ.get("CN_SCALE", 0.8)), generator=torch.Generator("cuda").manual_seed(0)).images[0]
        r.save(os.path.join(od, os.path.basename(f)[:-4] + ".png"))
        print(os.path.basename(f), s, f"{time.time() - t0:.0f}s", flush=True)
