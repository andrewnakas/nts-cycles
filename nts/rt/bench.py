"""GPU cost of nts.rt per frame, per pass, for each output size and quality mode, on a replayed real clip.

The clip's first K frames are put on the GPU once (inputs a game would already have there: colour, depth, motion),
then played in a loop. Each case reports:
  ts   GPU timestamps around each pass, N frames recorded into one command buffer: min, p10, median of prep + resolve
  wall the same N frames by wall clock (submit to done) / N, with timestamps off: the check on the timestamp unit
  paced one frame per submit at 60 fps (a laptop GPU may clock down between short bursts), timestamps, median
Output sizes are W x H. A portrait source is turned to landscape (axes and motion swapped) and resampled with nearest
neighbour to the render size of each case, motion scaled along: timing needs realistic motion, not exact content.

Run: python -m nts.rt.bench [--src <exr_dir | pack.py dir | replay.npz | synthetic>] [--out 1920x1080,2560x1440,3840x2160]
         [--modes all | performance,quality,...] [--build fast | compat | fast,compat] [--entry compute,fragment] [--frames 300] [--k 16]
         [--adapter NAME] [--wg 16x16,32x4] [--mem 256] [--json results.jsonl] [--save-replay replay.npz] [--no-paced]
         [--warm-ms 2000]
         [--variants "base;history=catmull;space=ycocg,clip=centre"]
--variants: RTUpscaler design switches per variant, separated by ";" ("base" = none).
--mem caps the replayed input frames at that many MB of GPU memory (fewer than --k frames at large sizes).
--warm-ms: keep the GPU busy that long before each case is timed (an idle laptop GPU runs at a low clock until
then; 0 = one 30-frame warm-up batch).
"""
import gc
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import wgpu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from baselines.accumulate import arg  # noqa: E402
from nts import conventions as cv  # noqa: E402
from nts.io_exr import Frame  # noqa: E402
from nts.rt import gpu  # noqa: E402
from nts.rt.upscaler import STRIDE, Inputs, RTUpscaler, SelfGuided  # noqa: E402

NATIVE = next(k for k, v in cv.QUALITY_MODES.items() if v == cv.OUT)
SCALES = {k: cv.OUT[0] / v[0] for k, v in cv.QUALITY_MODES.items()}  # mode -> per-axis scale


def load(src, k):
    """Up to k frames as Frames (rgb, depth, mv in render px, jitter)."""
    if src == "synthetic":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
        from test_rt_upscaler import clip
        return clip(540, 960, n=k)
    if src.endswith(".npz"):
        z = np.load(src)
        return [Frame(f=i, rgb=z["rgb"][i].astype(np.float32), depth=z["depth"][i].astype(np.float32),
                      mv=z["mv"][i].astype(np.float32), jitter=tuple(z["jitter"][i]), reset=False)
                for i in range(min(k, len(z["rgb"])))]
    if os.path.exists(os.path.join(src, "lr.npy")):  # nts/datasets/pack.py: rgb, albedo, normal, depth, mv x, mv y
        L = np.load(os.path.join(src, "lr.npy"), mmap_mode="r")
        meta = json.load(open(os.path.join(src, "meta.json")))
        out = []
        for i in range(min(k, len(L))):
            a = np.asarray(L[i], np.float32).transpose(1, 2, 0)
            out.append(Frame(f=i, rgb=np.ascontiguousarray(a[..., :3]), depth=np.ascontiguousarray(a[..., 9]),
                             mv=np.ascontiguousarray(a[..., 10:12]), jitter=tuple(meta["jitter"][i]), reset=False))
        return out
    from nts.io_exr import Sequence
    seq = Sequence(src)
    return [seq.read(f) for f in seq.frames[:k]]


def fit(fr, w, h):
    """The frame turned to the orientation of (w, h) and resampled to it (nearest), motion scaled along."""
    rgb, depth, mv, jit = fr.rgb, fr.depth, fr.mv, fr.jitter
    if (rgb.shape[0] > rgb.shape[1]) != (h > w):
        rgb, depth = rgb.transpose(1, 0, 2), depth.T
        mv, jit = mv.transpose(1, 0, 2)[..., ::-1], jit[::-1]
    sh, sw = depth.shape
    ys, xs = (np.arange(h) * sh) // h, (np.arange(w) * sw) // w
    g = lambda a: np.ascontiguousarray(a[ys][:, xs])  # noqa: E731
    return Frame(f=fr.f, rgb=g(rgb), depth=g(depth), mv=g(mv) * np.array([w / sw, h / sh], np.float32),
                 jitter=tuple(jit), reset=False)


def gpu_load():
    try:
        q = "--query-gpu=utilization.gpu,memory.used,memory.total,clocks.sm,temperature.gpu"
        return subprocess.run(["nvidia-smi", q, "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def case(dev, name, frames, W, H, mode, build, n, k, wg, paced, mem_mb, entry="compute", opts=None, warm_ms=0.0):
    s = SCALES[mode]
    rw, rh = max(1, round(W / s)), max(1, round(H / s))
    per = rw * rh * (16 + 4 + 8 if build == "compat" else 8 + 4 + 4)  # input bytes per replayed frame
    k = max(2, min(k, int(mem_mb * 2 ** 20 // per)))  # stay inside the VRAM budget (shared GPUs)
    opts = dict(opts or {})
    selfg = opts.pop("selfguide", False)
    if selfg:
        opts.update(guided=True, reject=0.0)
    up = RTUpscaler(W / rw, (H, W), build=build, dev=dev, slots=k + 1, wg=wg, in_hw=(rh, rw), entry=entry, **opts)
    alb = None
    if selfg:  # SelfGuided: the albedo history's own two passes, then a GPU copy into the guided upscaler's input
        alb = RTUpscaler(W / rw, (H, W), build=build, dev=dev, slots=k + 1, wg=wg, in_hw=(rh, rw), entry=entry,
                         wmax=32.0, expand=2.0, sigma=0.5, **{q: v for q, v in opts.items() if q in SelfGuided.SHARED})
        assert alb.b["hist"] == up.b["color"], "self-guided timing needs the albedo history in the input's format"
    per_f = 4 if alb else 2  # timed passes per frame
    inputs, ainputs = [], []
    albs = (np.full((rh, rw, 3), 0.5, np.float32), np.full((H, W, 3), 0.5, np.float32)) if up.guided else (None, None)
    for i in range(k):
        inp = Inputs(up)
        fr = fit(frames[i % len(frames)], rw, rh)
        up.upload(inp, fr, *albs)
        inputs.append(inp)
        dev.queue.write_buffer(up.ubuf, i * STRIDE, up.params(fr.jitter, False))
        if alb:
            ai = Inputs(alb)
            alb.upload(ai, Frame(f=fr.f, rgb=albs[0], depth=fr.depth, mv=fr.mv, jitter=fr.jitter, reset=False))
            ainputs.append(ai)
            dev.queue.write_buffer(alb.ubuf, i * STRIDE, alb.params(fr.jitter, False))
    for u in [up] + ([alb] if alb else []):
        u.dev.queue.write_buffer(u.ubuf, k * STRIDE, u.params(fit(frames[0], rw, rh).jitter, True))
    timer = gpu.Timer(dev, per_f * n)

    def frame(enc, f, slot, tm, k0):
        if alb:
            alb.encode(enc, ainputs[f % k], slot, tm, k0)
            enc.copy_texture_to_texture({"texture": alb.hist[(alb.t - 1) % 2]}, {"texture": inputs[f % k].alb_hr},
                                        (W, H, 1))
            k0 += 2
        up.encode(enc, inputs[f % k], slot, tm, k0, prev=inputs[(f - 1) % k])

    def record(m, t0, timed):
        enc = dev.create_command_encoder()
        for j in range(m):
            f = t0 + j
            frame(enc, f, k if f == 0 else f % k, timer if timed else None, per_f * j)
        if timed:
            timer.resolve(enc, per_f * m)
        return enc.finish()

    t_end = time.perf_counter() + warm_ms / 1e3  # warm-up: long enough for a laptop GPU to raise its clock
    while True:
        dev.queue.submit([record(30, 0, False)])
        dev.queue.on_submitted_work_done_sync()
        if time.perf_counter() >= t_end:
            break
    cb = record(n, 30, True)
    dev.queue.submit([cb])
    dev.queue.on_submitted_work_done_sync()
    ts = timer.read(per_f * n)
    cb = record(n, 30 + n, False)
    t0 = time.perf_counter()
    dev.queue.submit([cb])
    dev.queue.on_submitted_work_done_sync()
    wall = (time.perf_counter() - t0) / n * 1e3
    r = dict(adapter=name, build=build, entry=entry, opts=opts or {}, out=f"{W}x{H}", mode=mode, render=f"{rw}x{rh}", frames=n, k=k, wg=[list(g) for g in up.wg],
             wall_ms=round(wall, 4), load_before=gpu_load())
    if ts is not None:
        q = ts.reshape(n, per_f)
        a, b = q[:, 0::2].sum(1), q[:, 1::2].sum(1)  # prep, resolve (both upscalers' in self-guided timing)
        tot = a + b
        r.update(ts_min=round(float(tot.min()), 4), ts_p10=round(float(np.percentile(tot, 10)), 4),
                 ts_med=round(float(np.median(tot)), 4), prep_med=round(float(np.median(a)), 4),
                 resolve_med=round(float(np.median(b)), 4))
    if paced and ts is not None:
        pt = gpu.Timer(dev, per_f)
        got, nxt = [], time.perf_counter()
        for j in range(120):
            enc = dev.create_command_encoder()
            frame(enc, j, j % k, pt, 0)
            pt.resolve(enc, per_f)
            dev.queue.submit([enc.finish()])
            got.append(float(pt.read(per_f).sum()))
            nxt += 1 / 60
            time.sleep(max(0.0, nxt - time.perf_counter()))
        r["paced_med"] = round(float(np.median(got)), 4)
    r["load_after"] = gpu_load()
    for inp in inputs + ainputs:
        for t in (inp.color, inp.depth, inp.mv) + ((inp.alb, inp.alb_hr) if inp in inputs and up.guided else ()):
            t.destroy()
    up.release()
    if alb:
        alb.release()
    return r


def main():
    a = sys.argv
    if "-h" in a or "--help" in a:
        sys.exit(__doc__)
    outs = [tuple(map(int, o.split("x"))) for o in arg("--out", "1920x1080,2560x1440,3840x2160").split(",")]
    modes = [m for m in SCALES if m != NATIVE] if arg("--modes", "all") == "all" else arg("--modes").split(",")
    builds = arg("--build", "fast").split(",")
    entries = arg("--entry", "compute").split(",")  # resolve as compute and/or fragment shader
    flag = {"1": True, "true": True, "0": False, "false": False}
    variants = [{} if v in ("", "base") else {k: flag.get(x, x) for k, x in (p.split("=") for p in v.split(","))}
                for v in arg("--variants", "base").split(";")]
    n, k = int(arg("--frames", 300)), int(arg("--k", 16))
    wg = [tuple(map(int, g.split("x"))) for g in arg("--wg", "16x16,32x4").split(",")]
    wg = (wg[0], wg[-1])  # one size for both passes, or prep,resolve
    dev, name = gpu.device(arg("--adapter"))
    frames = load(arg("--src", "synthetic"), k)
    if arg("--save-replay"):
        np.savez(arg("--save-replay"), rgb=np.stack([f.rgb for f in frames]).astype(np.float16),
                 depth=np.stack([f.depth for f in frames]).astype(np.float32),
                 mv=np.stack([f.mv for f in frames]).astype(np.float16), jitter=np.array([f.jitter for f in frames]))
    print(f"bench: {name} on {platform.node()}, {len(frames)} source frames {frames[0].rgb.shape[1]}x"
          f"{frames[0].rgb.shape[0]}, {n} frames per case, timestamps {'on' if 'timestamp-query' in dev.features else 'off'}")
    print(f"{'out':>10} {'mode':>18} {'render':>10} {'build':>6} | {'min':>6} {'p10':>6} {'median':>6} | "
          f"{'prep':>6} {'resolve':>7} | {'wall':>6} {'paced':>6}  ms")
    for W, H in outs:
        for mode in modes:
            for build, entry, opts in [(b, e, o) for b in builds for e in entries for o in variants]:
                try:
                    r = case(dev, name, frames, W, H, mode, build, n, min(k, len(frames)), wg, "--no-paced" not in a,
                             float(arg("--mem", 256)), entry, opts, float(arg("--warm-ms", 0)))
                except wgpu.GPUError as e:  # e.g. not enough GPU memory left next to other jobs
                    gc.collect()
                    print(f"{W}x{H}".rjust(10), f"{mode:>18}", f"{'':>10} {build:>6} | skipped: {str(e).splitlines()[-1].strip()}")
                    continue
                g = lambda key: f"{r[key]:6.3f}" if key in r else "     -"  # noqa: E731
                short = {"history": "", "space": "", "clip": "", "guided": "guided", "selfguide": "selfguide"}
                tag = (build if entry == "compute" else build + "/fs") + "".join(
                    "/" + (k if v is True else short.get(k, k + "=") + str(v)) for k, v in opts.items())
                print(f"{r['out']:>10} {mode:>18} {r['render']:>10} {tag:>6} | {g('ts_min')} {g('ts_p10')} "
                      f"{g('ts_med')} | {g('prep_med')} {g('resolve_med'):>7} | {g('wall_ms')} {g('paced_med')}",
                      flush=True)
                if arg("--json"):
                    with open(arg("--json"), "a") as fh:
                        fh.write(json.dumps(r) + "\n")
    print("load:", gpu_load() or "n/a (no nvidia-smi)")


if __name__ == "__main__":
    main()
