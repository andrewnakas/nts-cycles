"""A/B rows for nts.rt design switches and constants on packed shots, one shot in memory at a time.

Each variant upscales every shot from its first frame and is scored as nts/rt/fit.py scores (8-bit PSNR, tErr,
static flicker through Blender's view transform), per shot and as the mean over the fit shots and the held-out ones.
A variant is "base" or comma-separated key=value pairs: RTUpscaler switches (history, space, clip), the
build (build=compat), selfguide=1 (SelfGuided, full packs only, with alb_wmax, alb_expand, alb_foot, eps) and
constants (foot, expand, static_expand, wmax, shrink, prior, rel); --start sets constants shared by every variant.

Run: python -m nts.rt.ab <pack_dir> ... [--hold <pack_dir>,...] --variants "base;history=catmull;space=ycocg,clip=centre"
         [--start foot=0.25,...] [--frames 32] [--warm 8] [--build fast] [--adapter NAME] [--json ab.jsonl]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from baselines.accumulate import arg  # noqa: E402
from nts.rt import gpu  # noqa: E402
from nts.rt.fit import DEFAULTS, VIEW, Shot, score  # noqa: E402
from nts.tonemap import Tonemapper  # noqa: E402

OPTS = {"--hold", "--variants", "--start", "--frames", "--warm", "--build", "--adapter", "--json"}
SWITCHES = {"history", "space", "clip", "depth_test"}
FLAGS = {"selfguide"}  # selfguide=1: SelfGuided (needs full packs); its constants alb_wmax, alb_expand, alb_foot, eps


def parse(v, start, build):
    params, opts = dict(start), {}
    if v not in ("", "base"):
        for k, x in (p.split("=") for p in v.split(",")):
            if k in SWITCHES:
                opts[k] = x
            elif k in FLAGS:
                opts[k] = x in ("1", "true")
            elif k == "build":
                build = x
            else:
                params[k] = float(x)
    return params, opts, build


def main():
    a = sys.argv
    dirs = [x for x in a[1:] if not x.startswith("--") and a[a.index(x) - 1] not in OPTS]
    hold = arg("--hold").split(",") if arg("--hold") else []
    if not dirs or not arg("--variants"):
        sys.exit(__doc__)
    start = dict(DEFAULTS)
    if arg("--start"):
        start.update({k: float(v) for k, v in (x.split("=") for x in arg("--start").split(","))})
    names = arg("--variants").split(";")
    n, warm, build = int(arg("--frames", 32)), int(arg("--warm", 8)), arg("--build", "fast")
    dev, adapter = gpu.device(arg("--adapter"))
    tm = Tonemapper(*VIEW)
    t0 = time.time()
    rows = {v: [] for v in names}
    alb = any("selfguide=1" in v for v in names)
    for d in dirs + hold:
        sh = Shot(d, n, warm, tm, albedo=alb)
        for v in names:
            params, opts, b = parse(v, start, build)
            r = score([sh], params, dev, b, tm, opts)["rows"][0]
            r["held_out"] = d in hold
            rows[v].append(r)
            if arg("--json"):
                with open(arg("--json"), "a") as fh:
                    fh.write(json.dumps(dict(variant=v, start=start, build=build, adapter=adapter, **r)) + "\n")
        print(f"  {sh.name} done ({time.time() - t0:.0f} s)", flush=True)
        del sh
    print(f"\nab: {len(dirs)} fit shots, {len(hold)} held out, frames {warm}-{n - 1}, {build} build on {adapter}, "
          f"start " + " ".join(f"{k}={v:g}" for k, v in start.items()))
    print(f"{'variant':<44} | {'fit PSNR':>8} {'tErr':>7} {'flicker':>7} | {'held PSNR':>9} {'tErr':>7} {'flicker':>7}")
    for v in names:
        f = [r for r in rows[v] if not r["held_out"]]
        h = [r for r in rows[v] if r["held_out"]]
        m = lambda rs, k: f"{np.mean([r[k] for r in rs]):.4f}" if rs else "-"  # noqa: E731
        print(f"{v:<44} | {m(f, 'psnr'):>8} {m(f, 'terr'):>7} {m(f, 'flicker'):>7} | "
              f"{m(h, 'psnr'):>9} {m(h, 'terr'):>7} {m(h, 'flicker'):>7}")


if __name__ == "__main__":
    main()
