# Kaggle kernel: NR-1 v2 pilots (the design log row 20 onward). Patched by tools/kaggle/push_pilot.sh.
# Dataset nts-pilot2 = our own renders only: evalf/eval_<clip>_<frame>.jpg (native look, all frames of the 4 old clips)
# and keys/key_<clip>_<frame>.png (toned keyframes, nts/tone.py output) + keys/boxes.json (head boxes).
# MODE "i1": the e2 teacher (kernel source nts-r2p-e2, its last.pt) on evalf -> out/evalf/*.jpg. ~1 h on a T4.
# MODE "p2": FLUX.2 [klein] 4B (Apache-2.0, ungated; 4 steps, guidance 1) edits the toned keyframes with a plain
#   "photograph of this scene" prompt, head crop at HEAD px and the whole frame at about half size ->
#   out/head_<p>/*.png, out/full_<p>/*.png (both written back at the keyframe's own size). No LoRA.
#   A T4 (14.6 GB) cannot run 1024 px edits: the model takes 8.2 GB and attention over image + reference tokens
#   (8,700 at 1024 px) asked for another 6.8 GB (run 1, 2026-11-20). 768 px is 5,100 tokens, about 2.5 GB.
# Nothing is trained in either mode.
import base64, glob, os, subprocess, sys, time
MODE = "i1"
ARGS = ""                # i1: extra gen/r2p_train.py arguments; p2: "minutes|prompts|only|mode|median" (default
                         # "80|ab|||"): prompts = letters of PROMPTS in run order, only = comma list of
                         # <clip>_<frame> keyframes, mode = flags joined with "+": "offload" for the slow, safe
                         # loading (see load() below), "nofull" for heads only, "nohead" for whole frames only,
                         # "sd:0.6,0.45" for extra head edits that START FROM THE INPUT (see edit(), run 7),
                         # "sdonly" for those alone;
                         # median = odd size of a median filter on the input at frame scale, or "r" for a size
                         # that follows the head box (0 or empty = none; output folders end in m<size> or mr)
CODE_B64 = ""            # gen/r2p_train.py (i1)
DS = "nts-pilot2"
BOXES = {}               # p2 on a dataset without keys/ (r2p-heads): {"key_<clip>_<frame>.png": [x0, y0, x1, y1]} head
                         # boxes; the keyframes are then that dataset's evalf/eval_<clip>_<frame>.jpg (v9d look)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True)
os.chdir("/kaggle/working")
root = [d for d in glob.glob("/kaggle/input/**/evalf", recursive=True) if f"/{DS}/" in d and "/out/" not in d]
assert root, f"dataset {DS} not attached"
data = os.path.dirname(root[0])
if MODE == "i1":
    subprocess.run(f"{sys.executable} -m pip install -q -U diffusers peft lpips", shell=True, check=True)
    subprocess.run(f"{sys.executable} -m pip uninstall -q -y torchao", shell=True)  # Kaggle ships 0.10; peft wants 0.16+
    src = "/kaggle/working/r2p_train.py"
    open(src, "wb").write(base64.b64decode(CODE_B64))
    ck = glob.glob("/kaggle/input/**/last.pt", recursive=True)
    assert ck, "teacher checkpoint (kernel source nts-r2p-e2) not attached"
    clips = sorted({os.path.basename(p).split("_")[1] for p in glob.glob(data + "/evalf/eval_*.jpg")})
    print("clips:", clips, "frames:", len(glob.glob(data + "/evalf/eval_*.jpg")), "checkpoint:", ck[0], flush=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run(f"{sys.executable} {src} --data {data} --out out --infer {ck[0]} --infer-only {','.join(clips)} {ARGS}",
                   shell=True, env=env)
    print("written:", len(glob.glob("out/evalf/*.jpg")), flush=True)
if MODE == "p2":
    subprocess.run(f"{sys.executable} -m pip install -q -U git+https://github.com/huggingface/diffusers.git transformers accelerate",
                   shell=True, check=True)
    import gc, inspect, json
    import torch
    HEAD, FULL_W = 768, 576   # px: head crop side; whole-frame width (height follows the frame's shape). Both are
                              # ~590k px: 608 x 1088 ran out of memory on the T4 (run 3)
    from PIL import Image
    from diffusers import Flux2KleinPipeline
    arg = (ARGS.split("|") + ["", "", "", "", ""])[:5]
    MEDR = arg[4] == "r"     # "r": median size follows the head box (5 at a ~450 px head, 11 on a 1080 px close-up, 3 at
                             # ~200 px). Run 4, fixed 5: dots stayed on the close-up, and the small head lost its face
    MED = 0 if MEDR else int(arg[4] or 0)   # the v9d look paints stubble as a 1-2 px speckle; the model turns it into real stubble
    t0, budget = time.time(), 60 * float(arg[0] or 80)
    order, only = arg[1] or "ab", {x for x in arg[2].split(",") if x}
    if os.path.isdir(data + "/keys"):
        boxes = json.load(open(data + "/keys/boxes.json"))
        src = {os.path.basename(p): p for p in sorted(glob.glob(data + "/keys/key_*.png"))}
    else:
        boxes = BOXES
        src = {n: f"{data}/evalf/eval_{n[4:-4]}.jpg" for n in sorted(BOXES)}
        src = {n: p for n, p in src.items() if os.path.exists(p)}
    if only:
        src = {n: p for n, p in src.items() if n[4:-4] in only}
    print("keyframes:", len(src), "from", data, "prompts:", order, flush=True)
    assert src, "no keyframes found"
    PROMPTS = {
        "a": "A real photograph of this exact scene, shot on a full-frame camera. Keep the same person, the same face, "
             "pose, framing, clothes, colours and lighting. Natural skin with fine pores, real hair strands, real fabric.",
        "b": "Turn this 3D render into a photograph. Do not change the person, the composition or the colours.",
        # run 2 (v9d-look input, which has painted stubble): a and b keep the stubble and add eye wrinkles.
        # run 3: c and d remove it but REPLACE THE FACE on every clip (closed mouth, younger, narrower jaw): do not use
        "c": "A real photograph of this exact scene, shot on a full-frame camera. Keep the same person, the same face, "
             "pose, framing, clothes, colours and lighting. He is clean-shaven: no beard, no stubble, no moustache. "
             "Natural skin with fine pores, real hair strands, real fabric.",
        "d": "A real photograph of this exact scene, shot on a full-frame camera. Keep the same person, the same face, "
             "pose, framing, clothes, colours and lighting. He is a clean-shaven young man in his twenties: no beard, "
             "no stubble, no moustache, smooth skin without wrinkles. Natural skin with fine pores, real hair strands.",
        # run 7 (2026-11-21, first run on the toned NATIVE keyframes, which have no painted stubble): b as the baseline,
        # a for skin pores, h for the hair. With b the model redraws the fringe as wavy locks whose outlines read as a
        # "net" over the render's hair (three judges, stills H1/H2); no filter after the fact beat plain hair.
        "h": "Turn this 3D render into a photograph. Do not change the person, the composition or the colours. "
             "Keep the hair exactly as it is: the same locks in the same places, made of fine individual strands.",
        # run 8: a description, for edits WITHOUT a reference image (the "t" variants of edit()): the model's own
        # idea of a photograph fills in the surface, the noised keyframe fixes where everything is
        "p": "A candid photograph of a clean-shaven young man with light brown hair, natural skin with visible "
             "pores and faint freckles, individual hair strands, soft natural light. Shot on a full-frame camera "
             "with an 85mm lens.",
        # run 12: as p without the freckles and the "soft natural light" (the t85 edits lighten the face by 3-8 L
        # and the freckles are a pattern the character does not have: the face-match score falls with the skin's
        # mid band, design log row 97), with the eye colour named (pty332 356 came back with green-brown eyes)
        "n": "A candid photograph of a clean-shaven young man with light brown hair and blue eyes, clear youthful "
             "skin with fine natural pores, individual hair strands. Shot on a full-frame camera with an 85mm lens.",
    }

    def load(fast):
        """The pipeline and, in fast mode, the prompts' embeddings.
        offload: enable_model_cpu_offload moves the text encoder and the transformer (about 8 GB each; the T4 holds
        one at a time) on and off the GPU for every edit: ~105 s per 768 px edit (runs 2 and 3).
        fast: the prompts are fixed, so each is encoded once, the text encoder is dropped, and the transformer and
        the VAE stay on the GPU."""
        pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)
        emb = {}
        if fast:
            t1 = time.time()
            pipe.text_encoder.to("cuda")
            with torch.no_grad():
                for k in order:
                    emb[k] = pipe.encode_prompt(PROMPTS[k], device=torch.device("cuda"))[0].to("cpu")
            pipe.text_encoder.to("cpu")
            pipe.text_encoder = None
            gc.collect()
            torch.cuda.empty_cache()
            pipe.transformer.to("cuda")
            pipe.vae.to("cuda")
            assert pipe._execution_device.type == "cuda", pipe._execution_device
            print(f"fast mode: {len(emb)} prompts encoded, transformer on the GPU, {time.time() - t1:.0f}s, "
                  f"mem {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)
        else:
            pipe.enable_model_cpu_offload()
        try:
            pipe.vae.enable_tiling()
        except Exception as e:
            print("no vae tiling:", repr(e))
        return pipe, emb

    fast = "offload" not in arg[3]   # mode: flags joined with "+": offload, nofull (heads only), sd:<levels>
    try:
        pipe, emb = load(fast)
        msg = None
    except Exception as e:
        msg = repr(e)[:300]
    if msg:
        print("fast mode failed, loading with offload:", msg, flush=True)
        pipe = emb = None
        gc.collect()
        torch.cuda.empty_cache()
        fast = False
        pipe, emb = load(False)
    print("call:", inspect.signature(pipe.__call__), flush=True)

    def start_sigmas(tokens, targets):
        """The `sigmas` to hand the pipeline so that the noise levels it really uses are `targets` (it shifts what
        it is given by a resolution-dependent mu), found by bisection on the scheduler itself -> (sigmas, real)."""
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
        mu = compute_empirical_mu(image_seq_len=tokens, num_steps=len(targets))
        sch = pipe.scheduler

        def real(u):
            sch.set_timesteps(sigmas=u, mu=mu, device="cpu")
            return [float(v) for v in sch.sigmas]
        out = []
        for tgt in targets:
            lo, hi = 1e-5, 1.0
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if real([mid])[0] < tgt else (lo, mid)
            out.append((lo + hi) / 2)
        return out, real(out)

    def edit(im, pk, seed=0, start=None, steps=2, ref=True, init=None, ref_blur=0):
        """One edit at the image's own size, padded on the right/bottom to a multiple of 16 by edge repetition.
        start (0..1, fast mode only): the edit does not begin from pure noise but from the input image itself with
        that share of noise mixed in (Meng et al. 2022, "SDEdit: guided image synthesis and editing with stochastic
        differential equations"; flow-matching form x = (1 - s) x0 + s noise), two steps from there. The input is
        still given as the reference image as well. Why: a plain edit is photographic but redraws the person (darker
        hair in new locks, another jaw, a shaved shadow: sheet look/p3/ceil.jpg), so only a thin band of its detail
        could be used and the result kept the render's smooth skin. Whatever survives the noise (face shape, hair
        mass, colours) cannot be redrawn.
        Run 7 (start 0.3-0.75, two steps, with the reference): the model returns its input, smooth skin and all
        (mean change on the face 2-6 levels): starting point and reference agree, so nothing asks for a change.
        Run 8 therefore varies what the two are. ref=False: no reference image, text only (prompt p), the classic
        SDEdit: the surface comes from the model's prior. init=<image>: the starting point is that image instead
        of the input (the plain, photographic edit), while the reference stays the input: does the reference pull
        the face back? steps: levels fall linearly from start to 0 in that many steps."""
        import numpy as np
        w, h = (im.width + 15) // 16 * 16, (im.height + 15) // 16 * 16
        a = np.asarray(im)
        x = Image.fromarray(np.pad(a, ((0, h - im.height), (0, w - im.width), (0, 0)), mode="edge"))
        # ref_blur (px): the REFERENCE is a low-passed copy of the input. Run 8b: with the sharp input as reference
        # the model copies it, smooth skin included (r85, r92); without any reference it draws real skin and hair
        # but drifts from the face and the mouth (t75, t85). A blurred reference
        # can only pin what survives the blur: shape, expression, colour. The surface is left to the model.
        xr = x
        if ref_blur:
            from PIL import ImageFilter
            xr = x.filter(ImageFilter.GaussianBlur(ref_blur))
        text = {"prompt_embeds": emb[pk].to("cuda")} if emb else {"prompt": PROMPTS[pk]}
        gen = torch.Generator(device="cuda").manual_seed(seed)
        if start:
            assert emb, "start-from-input edits need fast mode (the VAE on the GPU)"
            assert not getattr(pipe.scheduler.config, "use_flow_sigmas", False), "scheduler ignores custom sigmas"
            src0 = x
            if init is not None:
                ai = np.asarray(init.convert("RGB").resize(im.size, Image.LANCZOS))
                src0 = Image.fromarray(np.pad(ai, ((0, h - im.height), (0, w - im.width), (0, 0)), mode="edge"))
            xt = pipe.image_processor.preprocess(src0, height=h, width=w, resize_mode="crop").to("cuda", pipe.vae.dtype)
            with torch.no_grad():
                x0 = pipe._encode_vae_image(image=xt, generator=None).float()
            us, real = start_sigmas(x0.shape[2] * x0.shape[3], [start * (steps - i) / steps for i in range(steps)])
            if name_log.setdefault(("sig", start, steps, x0.shape[2:]), True):
                name_log[("sig", start, steps, x0.shape[2:])] = False
                print(f"start {start}: sigmas given {[round(u, 4) for u in us]} -> used {[round(v, 3) for v in real]}", flush=True)
            noise = torch.randn(x0.shape, generator=gen, device="cuda", dtype=torch.float32)
            lat = (1 - real[0]) * x0 + real[0] * noise
            y = pipe(height=h, width=w, guidance_scale=1.0, num_inference_steps=len(us), sigmas=us,
                     latents=lat, generator=gen, **({"image": xr} if ref else {}), **text).images[0]
        else:
            y = pipe(image=x, height=h, width=w, guidance_scale=1.0, num_inference_steps=4, generator=gen, **text).images[0]
        if y.size != (w, h):
            y = y.resize((w, h), Image.LANCZOS)
        return y.crop((0, 0, im.width, im.height))

    done = failed = 0
    name_log = {}
    flags = arg[3].split("+")
    # sd:<variant>,<variant>...  variant = [r|t|e]<start level>[x<steps>]: r (default) = from the input, with the
    # reference (run 7); t = from the input, no reference, prompt p; e = from the plain edit, with the reference
    import re
    starts = []
    for v in (v for fl in flags if fl.startswith("sd:") for v in fl[3:].split(",") if v):
        # u / q / w: like r with the reference blurred 2 / 4 / 8 px. c = chain (run 12): the start is the result of
        # a first pass (blurred reference 4 px, start 0.92, 4 steps: keeps the face, face match 0.87-0.94, but soft
        # skin and clay hair), then no reference, text only, like t. @<letter> = prompt for this variant, s<n> = seed
        m_ = re.match(r"^([rteuqwc]?)([0-9.]+)(?:x(\d+))?(?:@([a-z]))?(?:s(\d+))?$", v)
        lv = float(m_.group(2))
        lv = lv / 100 if lv > 1 else lv   # "t60x3" means 0.60: run 8 (11-21) took 60 as the level, every variant
        assert 0 < lv <= 1, v             # then started from pure noise and the keyframe was never used
        starts.append((m_.group(1) or "r", lv, int(m_.group(3) or 2), m_.group(4), int(m_.group(5) or 0)))
    plain_out, chain_out = {}, {}
    # heads with the first prompt, the same from the input at each start level, whole frames, then the other
    # prompts' heads: a cut-off run still covers what the next decision needs
    jobs = [] if "nohead" in flags else [("head", order[0], None)] + [("head", order[0], s_) for s_ in starts]
    jobs += [] if "nofull" in flags else [("full", order[0], None)]
    jobs += [] if "nohead" in flags else [("head", k, None) for k in order[1:]]
    if "sdonly" in flags:  # only the start-from-input heads (a second kernel beside a plain run)
        jobs = ([("head", order[0], None)] if any(s_[0] == "e" for s_ in starts) else []) + \
               [("head", order[0], s_) for s_ in starts]
    for kind, pk, st in jobs:
        d = f"out/{kind}_{pk}" + (f"_{st[0]}{round(100 * st[1])}x{st[2]}" + (f"p{st[3]}" if st[3] else "") + (f"s{st[4]}" if st[4] else "") if st else "") + (f"m{arg[4]}" if arg[4] else "")
        os.makedirs(d, exist_ok=True)
        for name, p in src.items():
            if time.time() - t0 > budget or (failed >= 3 and done == 0):
                break
            im = Image.open(p).convert("RGB")
            med = 2 * int((boxes[name][2] - boxes[name][0]) / 200 + 0.5) + 1 if MEDR and name in boxes else MED
            if med >= 3:
                from PIL import ImageFilter
                im = im.filter(ImageFilter.MedianFilter(med))
            msg = None
            try:
                t1 = time.time()
                if kind == "full":
                    small = im.resize((FULL_W, round(im.height * FULL_W / im.width)), Image.LANCZOS)
                    y = edit(small, pk).resize(im.size, Image.LANCZOS)
                else:  # generative detail needs the face large
                    x0, y0, x1, y1 = boxes[name]
                    c = im.crop((x0, y0, x1, y1))
                    cin = c.resize((HEAD, HEAD), Image.LANCZOS)
                    if st is None:
                        ym = plain_out[name] = edit(cin, pk)
                    else:
                        ini = plain_out[name] if st[0] == "e" else None
                        if st[0] == "c":
                            if name not in chain_out:
                                chain_out[name] = edit(cin, pk, start=0.92, steps=4, ref=True, ref_blur=4)
                            ini = chain_out[name]
                        ym = edit(cin, st[3] or ("p" if st[0] in "tc" else pk), seed=st[4], start=st[1], steps=st[2],
                                  ref=st[0] not in "tc", init=ini,
                                  ref_blur={"u": 2, "q": 4, "w": 8}.get(st[0], 0))
                    y = ym.resize(c.size, Image.LANCZOS)
                y.save(f"{d}/{name}")
                done += 1
                print(f"{d[4:]} {name} {time.time() - t1:.0f}s  mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            except Exception as e:
                msg, failed = repr(e)[:300], failed + 1
            if msg:  # outside the except block: the traceback no longer holds the failed call's tensors
                print(f"{d[4:]} {name} FAILED {msg}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                if fast and done == 0 and failed >= 2:  # the transformer + its activations do not fit: go back to offload
                    print("fast mode cannot run an edit, reloading with offload", flush=True)
                    pipe = emb = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    fast, failed = False, 0
                    pipe, emb = load(False)
    print("written:", done, "failed:", failed, f"in {(time.time() - t0) / 60:.0f} min", flush=True)
