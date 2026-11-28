"""Phase 1 teacher pilot: Wan 2.1 (Apache-2.0) video-to-video at low strength on 480x832 production clips, to see whether
an open video-diffusion prior makes our renders read as real (judged by eval/scoreboard.py and the user, vs the reference).

Raw output only: the identity lock (keep input low frequencies / geometry) is applied afterwards on the Mac by
tools/teacher_lock.py, which has the region masks.

  clips/<name>/in/f_####.png  (33 frames, 4k+1 as Wan needs)  ->  out/<name>_s<strength>/f_####.png

Runs anywhere diffusers runs. On Kaggle 2x T4 (no bf16): fp16 transformer, fp32 VAE, the UMT5 prompt encoder runs once
on CPU in bf16 (fp16 UMT5 overflows), and clips are split across the GPUs, one worker process each.
  python teacher_wan.py --clips <dir> --out <dir> [--model Wan-AI/Wan2.1-T2V-1.3B-Diffusers] [--strengths 0.3,0.5]
"""
import argparse
import glob
import os
import subprocess
import sys
import time

PROMPT = ("a candid photograph of a real person, natural skin with pores and fine texture, realistic hair with "
          "individual strands, soft natural light, shot on a 35mm camera, sharp focus")
NEG = ("cgi, 3d render, video game, plastic skin, waxy, smooth skin, cartoon, illustration, painting, blurry, "
       "deformed face, distorted eyes, extra fingers, text artifacts, oversaturated")


def encode(model, path):
    """Same as diffusers' Wan _get_t5_prompt_embeds (512 tokens, trimmed to length, zero-padded), on CPU in bf16."""
    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel
    te = UMT5EncoderModel.from_pretrained(model, subfolder="text_encoder", torch_dtype=torch.bfloat16).eval()
    tok = AutoTokenizer.from_pretrained(model, subfolder="tokenizer")
    out = {}
    for k, text in (("pe", PROMPT), ("ne", NEG)):
        t = tok([text], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
                return_attention_mask=True, return_tensors="pt")
        n = int(t.attention_mask.sum())
        with torch.no_grad():
            e = te(t.input_ids, t.attention_mask).last_hidden_state[:, :n].float()
        out[k] = torch.cat([e, e.new_zeros(1, 512 - n, e.shape[2])], 1)
    torch.save(out, path)


def work(a, names):
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVideoToVideoPipeline
    bf16 = torch.cuda.get_device_capability()[0] >= 8  # T4 (7.5) only emulates bf16
    dt = torch.bfloat16 if bf16 else torch.float16
    emb = torch.load(a.embeds)
    vae = AutoencoderKLWan.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVideoToVideoPipeline.from_pretrained(a.model, vae=vae, text_encoder=None, tokenizer=None,
                                                   torch_dtype=dt)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)  # 480p setting
    if a.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pe, ne = emb["pe"].to("cuda", dt), emb["ne"].to("cuda", dt)
    for name in names:
        frames = sorted(glob.glob(os.path.join(a.clips, name, "in", "f_*.png")) +
                        glob.glob(os.path.join(a.clips, name, "in", "f_*.jpg")))
        video = [Image.open(f).convert("RGB") for f in frames]
        w, h = video[0].size
        for s in a.strengths.split(","):
            od = os.path.join(a.out, f"{name}_s{s}")
            if len(glob.glob(os.path.join(od, "f_*.png"))) == len(frames):
                continue
            os.makedirs(od, exist_ok=True)
            t0 = time.time()
            g = torch.Generator("cuda").manual_seed(0)
            out = pipe(video=video, prompt_embeds=pe, negative_prompt_embeds=ne, height=h, width=w,
                       strength=float(s), guidance_scale=a.cfg, num_inference_steps=a.steps, generator=g,
                       output_type="np").frames[0]
            for f, im in zip(frames, out):
                Image.fromarray((np.clip(im, 0, 1) * 255).round().astype(np.uint8)).save(
                    os.path.join(od, os.path.basename(f)[:-4] + ".png"))
            print(f"{name} s{s}: {len(out)} frames {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    ap.add_argument("--strengths", default="0.3,0.5"); ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=5.0); ap.add_argument("--offload", action="store_true")
    ap.add_argument("--embeds", default=None); ap.add_argument("--worker", default=None)
    a = ap.parse_args()
    if a.worker is not None:
        return work(a, a.worker.split(","))
    os.makedirs(a.out, exist_ok=True)
    a.embeds = os.path.join(a.out, "prompt_embeds.pt")
    if not os.path.exists(a.embeds):
        encode(a.model, a.embeds)
    import torch
    names = sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                   for p in glob.glob(os.path.join(a.clips, "*", "in", "f_*.*")))
    names = sorted(set(names))
    n = max(torch.cuda.device_count(), 1)
    procs = []
    for i in range(n):
        part = names[i::n]
        if not part:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        cmd = [sys.executable, __file__, "--clips", a.clips, "--out", a.out, "--model", a.model, "--strengths",
               a.strengths, "--steps", str(a.steps), "--cfg", str(a.cfg), "--embeds", a.embeds,
               "--worker", ",".join(part)] + (["--offload"] if a.offload else [])
        procs.append(subprocess.Popen(cmd, env=env))
    sys.exit(max(p.wait() for p in procs))


if __name__ == "__main__":
    main()
