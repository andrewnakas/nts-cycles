"""NR-1 pilot P1 (plan v2, step 3): SeedVR2-3B, a one-step video restoration model, on head-crop clips of our own
toned renders. Question: does a restoration prior add real skin and hair (pores, strands) on the render's own shapes,
steady over time, with the face kept? The keyframe edits (P2) are photographic but draw another man.

Provenance (the design log row 84; PROVENANCE.md before anything from here ships):
  - code: github.com/numz/ComfyUI-SeedVR2_VideoUpscaler (Apache-2.0), its standalone inference_cli.py; the commit is
    printed below. The official ByteDance-Seed/SeedVR script (Apache-2.0) needs an 80 GB GPU.
  - weights: SeedVR2-3B (ByteDance-Seed/SeedVR2-3B, tagged Apache-2.0), the fp8 file of numz/SeedVR2_comfyUI
    (tagged Apache-2.0), downloaded by the CLI.
  - colour correction is OFF: the wrapper's colour-fix module may derive from StableSR's (S-Lab licence,
    non-commercial). Colour comes from our own frames (the composite takes detail bands only).
  - open point before shipping: SeedVR v1 was initialised from SD3-Medium; the SeedVR2 paper says the 7B was trained
    from scratch and the 3B distilled from it. Not verified beyond the papers and model cards.
Inputs are our renders only. No reference frame is read here."""
import glob, json, os, re, shutil, subprocess, sys, threading, time

MODE = "v1"
ARGS = ""  # "minutes|shots|variants|dit"   (filled in by push_seedvr.sh)
DS = "nts-pilot2"   # the head crops live in the same private dataset as the keyframes (private quota is full)

T0 = time.time()
_a = (ARGS.split("|") + ["", "", "", ""])[:4]
BUDGET = float(_a[0] or 100) * 60
VARS = [v for v in _a[2].split(",") if v] or ["a", "b"]
DIT = _a[3] or "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
# ds: the crop is scaled by this before the model sees it (a restoration model needs something to restore: on a
# sharp input it does little or over-sharpens); the output side is the same for every variant of a shot.
VAR = {"a": dict(ds=1.0), "b": dict(ds=0.5), "c": dict(ds=0.35),
       "m": dict(ds=1.0, input_noise_scale=0.5), "n": dict(ds=0.5, input_noise_scale=0.5),
       "l": dict(ds=1.0, latent_noise_scale=0.1)}
W = "/kaggle/working"
PY = sys.executable


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, **kw)


def left():
    return BUDGET - (time.time() - T0)


src = glob.glob("/kaggle/input/**/crops/shots.json", recursive=True)
assert src, "dataset %s not mounted" % DS
ROOT = os.path.dirname(os.path.dirname(src[0]))
META = json.load(open(src[0]))
SHOTS = [s for s in _a[1].split(",") if s] or sorted(META)
print("shots", {s: (META[s]["box"], len(META[s]["frames"])) for s in SHOTS}, "variants", VARS, flush=True)

sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader")
sh("git clone -q https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git %s/seedvr2" % W)
sh("cd %s/seedvr2 && git log -1 --format='wrapper commit %%H %%cd' && cat requirements.txt | tr '\\n' ' '" % W)
sh(PY + " -m pip install -q 'omegaconf>=2.3.0' 'diffusers>=0.33.1' 'peft>=0.17.0' 'rotary_embedding_torch>=0.5.3' "
   "gguf einops safetensors psutil tqdm imageio-ffmpeg 2>&1 | tail -n 3")
FF = shutil.which("ffmpeg")
if not FF:
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
HELP = subprocess.run([PY, "inference_cli.py", "--help"], cwd=W + "/seedvr2", capture_output=True, text=True)
HELP = HELP.stdout + HELP.stderr
KNOWN = set(re.findall(r"--[a-z_0-9]+", HELP))
print("python", sys.version.split()[0], "| ffmpeg", FF, "| cli flags:", " ".join(sorted(KNOWN)), flush=True)
ANY = not KNOWN   # --help failed: show why, then pass every flag anyway
if ANY:
    print(HELP[-3000:], flush=True)

# (frames per batch, DiT blocks swapped to the CPU, VAE tile): tried in order, the first that runs is kept
LADDER = [(33, 0, 512), (17, 0, 512), (9, 16, 512), (5, 32, 256)]
state = {"rung": 0}
lock = threading.Lock()
LOG = []


def run_job(shot, var, gpu, cap=None):
    v, info = VAR[var], META[shot]
    frames = info["frames"] if cap is None else info["frames"][:cap]
    side = info["box"][2] - info["box"][0]
    out_side = 1024 if side > 900 else 768
    in_side = max(64, int(round(side * v["ds"] / 2)) * 2)
    name = "%s_%s" % (shot, var) + ("" if cap is None else "_smoke")
    os.makedirs(W + "/in", exist_ok=True)
    vid = "%s/in/%s.mp4" % (W, name)
    r = sh("%s -y -loglevel error -framerate 30 -start_number %d -i %s/crops/%s/f_%%04d.png -frames:v %d "
           "-vf scale=%d:%d:flags=area -c:v libx264 -qp 0 -pix_fmt yuv444p %s" % (
               FF, frames[0], ROOT, shot, len(frames), in_side, in_side, vid))
    if r.returncode != 0:
        return False
    while True:
        with lock:
            rung = state["rung"]
        if rung >= len(LADDER):
            return False
        bs, swap, tile = LADDER[rung]
        bs = min(bs, (len(frames) - 1) // 4 * 4 + 1)
        raw = "%s/raw/%s" % (W, name)
        shutil.rmtree(raw, ignore_errors=True)
        opts = {"--output": raw, "--output_format": "png", "--model_dir": W + "/models/SEEDVR2", "--dit_model": DIT,
                "--resolution": str(out_side), "--max_resolution": str(out_side), "--batch_size": str(bs),
                "--temporal_overlap": "3" if bs < len(frames) else "0", "--seed": "42", "--color_correction": "none",
                "--cuda_device": str(gpu), "--dit_offload_device": "cpu", "--vae_offload_device": "cpu",
                "--vae_encode_tile_size": str(tile), "--vae_decode_tile_size": str(tile)}
        flags = ["--vae_encode_tiled", "--vae_decode_tiled"]
        if swap:
            opts["--blocks_to_swap"] = str(swap)
            flags.append("--swap_io_components")
        for k in ("input_noise_scale", "latent_noise_scale"):
            if k in v:
                opts["--" + k] = str(v[k])
        cmd = [PY, "inference_cli.py", vid]
        for k, x in opts.items():
            if ANY or k in KNOWN:
                cmd += [k, x]
            else:
                print("  flag not in this CLI, dropped:", k, flush=True)
        cmd += [f for f in flags if ANY or f in KNOWN]
        t = time.time()
        lf = "%s/raw_%s.log" % (W, name)
        with open(lf, "w") as fh:
            rc = subprocess.run(cmd, cwd=W + "/seedvr2", stdout=fh, stderr=subprocess.STDOUT,
                                timeout=max(600, left() + 600)).returncode
        dt = time.time() - t
        pngs = sorted(glob.glob(raw + "/**/*.png", recursive=True))
        txt = open(lf, errors="replace").read()
        ok = rc == 0 and len(pngs) >= max(1, len(frames) - 4)   # a count a few short of the input is kept and logged
        print("job %s gpu %d: batch %d swap %d tile %d in %d px -> out %d px: rc %d, %d/%d frames, %.0f s (%.1f s/frame)" % (
            name, gpu, bs, swap, tile, in_side, out_side, rc, len(pngs), len(frames), dt, dt / max(len(frames), 1)), flush=True)
        if not ok:
            keep = [l for l in txt.replace("\r", "\n").split("\n") if re.search(r"(?i)error|memory|traceback|unrecognized|killed", l)]
            print("  log: " + " | ".join(keep[-8:])[:1500] + "\n  tail: " + txt[-700:].replace("\n", " / "), flush=True)
            with lock:
                if state["rung"] == rung:
                    state["rung"] += 1
            if left() < 300:
                return False
            continue
        from PIL import Image
        out = "%s/out/%s" % (W, name)
        os.makedirs(out, exist_ok=True)
        for p, f in zip(pngs, frames):
            Image.open(p).convert("RGB").save("%s/f_%04d.jpg" % (out, f), quality=95, subsampling=0)
        shutil.rmtree(raw, ignore_errors=True)
        LOG.append(dict(name=name, shot=shot, var=var, frames=len(frames), seconds=round(dt, 1), batch=bs, swap=swap,
                        tile=tile, in_side=in_side, out_side=out_side, box=info["box"], **{k: x for k, x in v.items()}))
        return True


# 1. smoke on one GPU (this also downloads the weights once), 2. everything else on both GPUs
ok = run_job(SHOTS[0], VARS[0], 0, cap=9)
print("smoke", "OK" if ok else "FAILED", "after %.0f s; ladder rung %d" % (time.time() - T0, state["rung"]), flush=True)
if ok:
    jobs = [(s, v) for v in VARS for s in SHOTS]
    done = []

    def worker(g):
        for s, v in jobs[g::2]:
            if left() < 420:
                print("out of time before", s, v, flush=True)
                continue
            try:
                if run_job(s, v, g):
                    done.append((s, v))
            except Exception as e:  # a timeout or a crash of one job must not lose the others
                print("job %s_%s failed: %r" % (s, v, e), flush=True)

    th = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
    [t.start() for t in th]
    [t.join() for t in th]
    print("done %d of %d jobs" % (len(done), len(jobs)), flush=True)
os.makedirs(W + "/out", exist_ok=True)
json.dump(dict(log=LOG, meta={s: META[s] for s in SHOTS}, dit=DIT, args=ARGS), open(W + "/out/seedvr_log.json", "w"), indent=1)
for d in ("seedvr2", "models", "in", "raw"):
    shutil.rmtree("%s/%s" % (W, d), ignore_errors=True)
print("P1 finished in %.0f min" % ((time.time() - T0) / 60), flush=True)
