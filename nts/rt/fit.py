"""Fit nts.rt's constants to native renders by plain coordinate search (no network, no gradients).

Each candidate upscales every fit shot from its first frame on the GPU and scores frames [--warm, --frames) against
the native render, both through Blender's view transform (AgX, AgX - Punchy, +0.35 EV):
  PSNR     8-bit, as eval/metrics.py
  tErr     mean |(out_t - out_t-1) - (gt_t - gt_t-1)|, as eval/run_eval.py
  flicker  temporal std of output luma where the native frames are steadiest (60 % of pixels), as eval/run_eval.py
One number decides: J = PSNR - s * (ln tErr + ln flicker), s = --stab (default 3: 1 % more tErr or flicker costs
as much as 0.03 dB). A step is kept when it raises J by at least --min-gain. Held-out shots (--hold) are scored,
never searched on.

Data: nts/datasets/pack.py folders (lr.npy, gt.npy, meta.json); lr.npy may hold only rgb, depth, mv x, mv y.
Run: python -m nts.rt.fit <pack_dir> ... [--hold <pack_dir>,<pack_dir>] [--frames 40] [--warm 8] [--build fast]
         [--params foot,expand,wmax,shrink,prior,rel] [--start foot=0.25,...] [--rounds 3] [--min-gain 0.005]
         [--stab 3]
         [--adapter NAME] [--log fit.jsonl] [--out fit.json] [--opts history=catmull,space=ycocg,clip=centre]
--opts are RTUpscaler's design switches, held fixed (not searched); --rounds 0 just scores --start on the fit and
held-out shots (an A/B row).
NTS_OCIO_CONFIG points the tonemapper at Blender's config.ocio on a machine without Blender.
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from baselines.accumulate import arg  # noqa: E402
from eval.metrics import luma, psnr  # noqa: E402
from nts.io_exr import Frame  # noqa: E402
from nts.rt import gpu  # noqa: E402
from nts.rt.upscaler import FITTED, RTUpscaler, SelfGuided  # noqa: E402
from nts.tonemap import Tonemapper  # noqa: E402

DEFAULTS = dict(FITTED)  # the search starts from the last fit (the first one started from sigma 0.47, expand 1,
#                          wmax 12, shrink 4, prior 0.03, rel 0.05)
STEPS = dict(foot=1.25, expand=1.15, wmax=1.4, shrink=1.4, prior=1.6, rel=1.5)  # multiplicative steps
LIMITS = dict(foot=(0.05, 0.8), expand=(1.0, 3.0), wmax=(2.0, 64.0), shrink=(0.05, 20.0),
              prior=(0.002, 2.0), rel=(0.005, 0.5))  # expand never below 1: the box is never shrunk below the samples'
#                                                     min/max; rel is capped: a looser depth test lets ghosts through

VIEW = ("AgX", "AgX - Punchy", 0.3499999940395355)


class Shot:
    """The first n frames of a packed shot: low-res rgb, depth and motion in memory (f16), native frames tonemapped.
    albedo=True also keeps the render's albedo (from a full pack, or alb.npy beside a trimmed one) for self-guided
    scoring."""

    def __init__(self, d, n, warm, tm, albedo=False):
        meta = json.load(open(os.path.join(d, "meta.json")))
        L = np.load(os.path.join(d, "lr.npy"), mmap_mode="r")
        G = np.load(os.path.join(d, "gt.npy"), mmap_mode="r")
        self.n = n = min(n, len(L))
        self.name, self.warm, self.scale = os.path.basename(d.rstrip("/")), warm, float(meta["scale"])
        ch = [0, 1, 2, 9, 10, 11] if L.shape[1] == 12 else slice(None)  # a full pack, or one trimmed to these six
        self.albedo = albedo
        alb = None
        if albedo and L.shape[1] == 12:
            ch = ch + [3, 4, 5]
        elif albedo:  # a trimmed pack: the render's albedo beside it as alb.npy (T, 3, h, w)
            alb = np.load(os.path.join(d, "alb.npy"), mmap_mode="r")
        self.lr = np.ascontiguousarray(L[:n, ch])  # rgb, depth, mv x, mv y (+ albedo)
        if alb is not None:
            assert len(alb) >= n, (d, len(alb), n)
            self.lr = np.concatenate([self.lr, alb[:n]], 1)
        self.jit = [tuple(j) for j in meta["jitter"][:n]]
        self.reset = [bool(r) for r in meta["reset"][:n]]
        self.hw = tuple(G.shape[2:])
        self.gt = [tm(np.asarray(G[i], np.float32).transpose(1, 2, 0)) for i in range(warm, n)]  # uint8

    def frame(self, i):
        a = self.lr[i].astype(np.float32)
        return Frame(f=i, rgb=a[:3].transpose(1, 2, 0), depth=a[3], mv=a[4:6].transpose(1, 2, 0),
                     albedo=a[6:9].transpose(1, 2, 0) if self.albedo else None, jitter=self.jit[i], reset=self.reset[i])


def score(shots, params, dev, build, tm, opts=None):
    rows = []
    opts = dict(opts or {})
    selfg = opts.pop("selfguide", False)
    for sh in shots:
        if selfg:  # params may hold the albedo upscaler's alb_wmax, alb_expand, alb_foot, and eps
            up = SelfGuided(sh.scale, sh.hw, build=build, dev=dev, **params, **opts)
        else:
            up = RTUpscaler(sh.scale, sh.hw, build=build, dev=dev, **params, **opts)
        ps, te, gl, tl, prev = [], [], [], [], None
        for i in range(sh.n):
            fr = sh.frame(i)
            out = up.step(fr, read=i >= sh.warm)
            if out is None:
                continue
            t, g = tm(out).astype(np.float32), sh.gt[i - sh.warm].astype(np.float32)
            ps.append(psnr(g, t))
            if prev is not None and not fr.reset:
                te.append(float(np.mean(np.abs((t - prev[1]) - (g - prev[0])))))
            gl.append(luma(g))
            tl.append(luma(t))
            prev = (g, t)
        up.release()
        gs, ts = np.stack(gl), np.stack(tl)
        gstd = gs.std(0)
        steady = gstd < np.percentile(gstd, 60)
        rows.append(dict(shot=sh.name, psnr=float(np.mean(ps)), terr=float(np.mean(te)),
                         flicker=float(ts.std(0)[steady].mean())))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "terr", "flicker")}
    return dict(agg, rows=rows)


def objective(r, stab):
    return r["psnr"] - stab * (math.log(r["terr"]) + math.log(r["flicker"]))


def better(c, b, min_gain, stab=3.0):
    return objective(c, stab) >= objective(b, stab) + min_gain


def main():
    a = sys.argv
    dirs = [x for x in a[1:] if not x.startswith("--") and a[a.index(x) - 1] not in OPTS]
    if not dirs:
        sys.exit(__doc__)
    n, warm, build = int(arg("--frames", 40)), int(arg("--warm", 8)), arg("--build", "fast")
    names = arg("--params", ",".join(DEFAULTS)).split(",")
    best = dict(DEFAULTS)
    if arg("--start"):
        best.update({k: float(v) for k, v in (x.split("=") for x in arg("--start").split(","))})
    min_gain, log, stab = float(arg("--min-gain", 0.005)), arg("--log"), float(arg("--stab", 3))
    dev, name = gpu.device(arg("--adapter"))
    opts = dict(x.split("=") for x in arg("--opts").split(",")) if arg("--opts") else {}
    tm = Tonemapper(*VIEW)
    t0 = time.time()
    fit = [Shot(d, n, warm, tm) for d in dirs]
    hold = [Shot(d, n, warm, tm) for d in arg("--hold").split(",")] if arg("--hold") else []
    print(f"fit: {len(fit)} shots, {len(hold)} held out, frames {warm}-{n - 1} scored, {build} build {opts} on {name} "
          f"(loaded in {time.time() - t0:.0f} s)", flush=True)

    res = search(fit, hold, best, names, dev, build, tm, opts, int(arg("--rounds", 3)), min_gain, stab, log)
    res.update(adapter=name, frames=[warm, n])
    print("best:", " ".join(f"{k}={v:.4g}" for k, v in res["best"].items()), f"({(time.time() - t0) / 60:.0f} min)")
    if arg("--out"):
        json.dump(res, open(arg("--out"), "w"), indent=1)


def search(fit, hold, best, names, dev, build, tm, opts, rounds=3, min_gain=0.005, stab=3.0, log=None):
    """Coordinate search from `best` over the constants in `names`, scored on the fit shots (any objects with the
    Shot interface); the held-out shots are scored at the start and the end only. Returns the result dict."""
    best = dict(best)
    seen = {}

    def run(p, tag):
        key = tuple(sorted((k, round(v, 9)) for k, v in p.items()))
        if key in seen:  # a point already scored (a step back to where the search came from)
            return seen[key]
        t = time.time()
        r = seen[key] = score(fit, p, dev, build, tm, opts)
        print(f"  {tag:<28} J {objective(r, stab):.3f}  psnr {r['psnr']:.3f}  tErr {r['terr']:.4f}  "
              f"flicker {r['flicker']:.4f}  "
              f"({time.time() - t:.0f} s)  " + " ".join(f"{k}={v:.4g}" for k, v in p.items()), flush=True)
        if log:
            with open(log, "a") as fh:
                fh.write(json.dumps(dict(tag=tag, params=p, opts=opts, stab=stab, j=objective(r, stab),
                                         **{k: r[k] for k in ("psnr", "terr", "flicker")})) + "\n")
        return r

    start = dict(best)
    b = run(best, "start")
    first = b
    steps = dict(STEPS)
    for rnd in range(rounds):
        moved = False
        for k in names:
            for d in (steps[k], 1 / steps[k]):
                while True:
                    lo, hi = LIMITS[k]
                    v = min(max(best[k] * d, lo), hi)
                    if v == best[k]:
                        break
                    cand = dict(best, **{k: v})
                    c = run(cand, f"round {rnd + 1} {k} x{d:.3g}")
                    if not better(c, b, min_gain, stab):
                        break
                    best, b, moved = cand, c, True
        if not moved:
            steps = {k: v ** 0.5 for k, v in steps.items()}  # nothing helped: finer steps
    res = dict(start=start, best=best, fit_start={k: first[k] for k in ("psnr", "terr", "flicker")},
               fit_best={k: b[k] for k in ("psnr", "terr", "flicker")}, fit_rows=b["rows"],
               shots=[s.name for s in fit], build=build, opts=opts)
    if hold:
        hs = score(hold, start, dev, build, tm, opts)
        hb = hs if best == start else score(hold, best, dev, build, tm, opts)
        res.update(hold_start=hs, hold_best=hb)
        for tag, r in (("held out, start", hs), ("held out, best", hb)):
            print(f"  {tag:<28} psnr {r['psnr']:.3f}  tErr {r['terr']:.4f}  flicker {r['flicker']:.4f}")
    return res


OPTS = {"--hold", "--frames", "--warm", "--build", "--params", "--start", "--rounds", "--min-gain", "--adapter",
        "--log", "--out", "--opts", "--stab"}

if __name__ == "__main__":
    main()
