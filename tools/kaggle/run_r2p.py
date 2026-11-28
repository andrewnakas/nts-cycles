# Kaggle kernel: train the render-to-photo head model (gen/r2p_train.py) on the nts-r2p dataset
# (r2p_train.py + data/{cg,real,eval}). ARGS is patched by tools/kaggle/push_r2p.sh. Output: /kaggle/working/out.
import glob, os, shutil, subprocess, sys
ARGS = "--hours 11"
SANITY = "--steps 12 --hours 0.2"
subprocess.run(f"{sys.executable} -m pip install -q -U diffusers peft lpips", shell=True, check=True)
subprocess.run(f"{sys.executable} -m pip uninstall -q -y torchao", shell=True)  # Kaggle ships 0.10; peft refuses anything below 0.16
CODE_B64 = ""  # gen/r2p_train.py, injected by push_r2p.sh so a code change needs no dataset upload
data = os.path.dirname(glob.glob("/kaggle/input/**/data/cg", recursive=True)[0])
src = "/kaggle/working/r2p_train.py"
if CODE_B64:
    import base64
    open(src, "wb").write(base64.b64decode(CODE_B64))
else:
    src = glob.glob("/kaggle/input/**/r2p_train.py", recursive=True)[0]
print("data:", {d: len(os.listdir(os.path.join(data, d))) for d in os.listdir(data)}, flush=True)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True)
MEGA = 0  # push_r2p.sh MEGA=<n>: add n public-domain photos (bitmind/megalith-small, Flickr PD/CC0 marks) as extra real tiles
AIX = 0  # see below
SHARP = 7.5  # push_r2p.sh SHARP=<x>: lowest detail score a megalith photo may have (0 = no sharpness filter)
if MEGA and ARGS:
    subprocess.run(f"{sys.executable} -m pip install -q huggingface_hub pyarrow", shell=True)
    import io
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image, ImageFilter
    Image.MAX_IMAGE_PIXELS = None
    rx, n, seen, soft = "/kaggle/tmp/realx", 0, 0, 0
    os.makedirs(rx, exist_ok=True)
    try:
        for c in range(12):
            for part in range(4):
                if n >= MEGA:
                    break
                f = hf_hub_download("bitmind/megalith-small", f"chunk_{c:04d}/train-{part:05d}-of-00004.parquet", repo_type="dataset")
                pf = pq.ParquetFile(f)
                for b in pf.iter_batches(batch_size=64, columns=["image", "filename"]):
                    for im_, fn in zip(b.column("image").to_pylist(), b.column("filename").to_pylist()):
                        seen += 1
                        try:
                            im = Image.open(io.BytesIO(im_["bytes"])).convert("RGB")
                        except Exception:
                            continue
                        if min(im.size) < 640:
                            continue
                        k = 768 / min(im.size)
                        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
                        if np.asarray(im.convert("HSV"))[::8, ::8, 1].mean() < 40:  # black-and-white and sepia scans
                            continue
                        # crisp photos only (thresholds set on 500 photos, 2026-11-16: keeps ~44%). Detail = mean
                        # |image - blurred image| per 64-px block; the sharpest fifth of the blocks must be crisp
                        # (a soft background is fine), and the flattest tenth must be clean (drops grain, paintings
                        # and wall-to-wall foliage).
                        g = im.convert("L")
                        hp = np.abs(np.asarray(g, np.float32) - np.asarray(g.filter(ImageFilter.GaussianBlur(1.5)), np.float32))
                        hh, ww = hp.shape[0] // 64 * 64, hp.shape[1] // 64 * 64
                        e = hp[:hh, :ww].reshape(hh // 64, 64, ww // 64, 64).mean((1, 3))
                        if np.percentile(e, 80) < SHARP or np.percentile(e, 10) > 5:
                            soft += 1
                            continue
                        im.save(f"{rx}/{os.path.splitext(os.path.basename(fn))[0]}.jpg", quality=92)
                        n += 1
                        if n >= MEGA:
                            break
                    if n >= MEGA:
                        break
                os.remove(os.path.realpath(f))
    except Exception as e:
        print("megalith fetch stopped:", repr(e), flush=True)
    print(f"megalith: kept {n} of {seen} photos ({soft} dropped as soft or grainy)", flush=True)
    # push_r2p.sh AIX=<n>: add n generated photoreal images (lrzjason/ZImageTurboGen: 3,000 images, Apache 2.0, made
    # with Z-Image-Turbo, itself Apache 2.0) to the same pool, as an experiment. Share of the pool = AIX / (MEGA + AIX).
    if AIX:
        from concurrent.futures import ThreadPoolExecutor

        def one(i):
            try:
                im = Image.open(hf_hub_download("lrzjason/ZImageTurboGen", f"zimage_{i}.webp", repo_type="dataset")).convert("RGB")
                im.resize((768, 768), Image.LANCZOS).save(f"{rx}/ai_{i:04d}.jpg", quality=92)
                return 1
            except Exception:
                return 0
        with ThreadPoolExecutor(8) as ex:
            na = sum(ex.map(one, range(min(AIX, 3000))))
        print(f"generated images added: {na}", flush=True)
        n += na
    if n >= 500:
        ARGS += f" --realx {rx}"
os.chdir("/kaggle/working")
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
# push_r2p.sh GRADE=1: train the shot-level grade head (gen/grade.py, ~10 min on a T4) before the main run, so its
# output (out/grade/grade.pt + grade_sheet.jpg) exists even if the long run is cut off.
GRADE = 0
GRADE_B64 = ""
if GRADE and GRADE_B64 and ARGS:
    import base64
    open("/kaggle/working/grade.py", "wb").write(base64.b64decode(GRADE_B64))
    subprocess.run(f"{sys.executable} /kaggle/working/grade.py train --data {data} --out out/grade --steps 3000 2>&1 | tail -40",
                   shell=True, env=env)
# a dozen steps first: any environment problem shows up in the first minutes instead of hours into a long run.
# Batch 2 is tried first; if it does not fit the card, batch 1.
NOSANITY = 0  # push_r2p.sh NOSANITY=1: skip the check (it also writes every eval frame, ~1 h on a T4) and use batch 2
BATCH = 2  # push_r2p.sh BATCH=<n>: the batch used when the sanity check is skipped (1 if batch 2 ran out of memory)
ok = BATCH if NOSANITY else None
for bs in (() if NOSANITY else (2, 1)):
    shutil.rmtree("sanity", ignore_errors=True)
    subprocess.run(f"{sys.executable} {src} --data {data} --out sanity {SANITY} --batch {bs} 2>&1 | tail -25", shell=True, env=env)
    subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader", shell=True)
    if os.path.exists("sanity/DONE.json"):
        ok = bs
        break
print("SANITY", "FAILED" if ok is None else f"OK batch {ok}", flush=True)
if ok and ARGS:
    shutil.rmtree("sanity", ignore_errors=True)
    ck = glob.glob("/kaggle/input/**/last.pt", recursive=True)  # an earlier run's output attached as a kernel source
    if ck and "--resume" not in ARGS:
        ARGS += f" --resume {ck[0]}"
        print("warm start from", ck[0], flush=True)
    subprocess.run(f"{sys.executable} {src} --data {data} --out out {ARGS} --batch {ok}", shell=True, env=env)
