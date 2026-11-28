"""Region "look" stage: per-region brightness, contrast, detail (mid/high band) and saturation, driven by the Region
masks Cycles renders (face / hands / rest). Deterministic and geometry-preserving: every operation scales an existing
band of the image, so nothing is invented and nothing can flicker beyond the input.

The parameters are a handful of numbers per region. fit() solves for them so the result's region statistics
(eval/look.py ratios) match a target, e.g. the measured reference pass-2 look. Only these summary numbers are taken from
the target, never its pixels, so the model stays clean-room. The trained enhancer later
replaces the hand-built bands, with these as its Structure/Tone gains.

Run: python nts/look.py fit   <gt.png> <mask.png> <target.png> <params.json>
     python nts/look.py apply <params.json> <in_dir> <masks_dir> <out_dir>
"""
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval.look import KEYS, ratios  # noqa: E402
from eval.metrics import box, load, luma  # noqa: E402

REGIONS = ("face", "hands", "rest")
DEFAULT = {r: {"bright": 1.0, "contrast": 1.0, "detail": 1.0, "shading": 1.0, "sat": 1.0} for r in REGIONS}


def soft_masks(mask_png, shape):
    im = Image.open(mask_png).convert("RGB").resize((shape[1], shape[0]), Image.NEAREST)
    a = np.asarray(im, np.float32) / 255.0
    face, hands = box(a[..., 0], 3), box(a[..., 1], 3)          # feathered edges: no hard seams
    rest = np.clip(1 - face - hands, 0, 1)
    return {"face": face, "hands": hands, "rest": rest}


def apply(img, masks, params):
    """img (H, W, 3) 0..255 display space -> processed, same space."""
    y = luma(img)
    hi = y - box(y, 6)                     # fine detail (pores, weave)
    lo = box(y, 24)
    mid = box(y, 6) - lo                   # broad shading
    chroma = img - y[..., None]

    def field(k):
        return sum(masks[r] * params[r][k] for r in REGIONS)
    region_mean = sum(masks[r] * float((y * masks[r]).sum() / max(masks[r].sum(), 1)) for r in REGIONS)
    b, c, d, s, sat = (field(k) for k in ("bright", "contrast", "detail", "shading", "sat"))
    base = region_mean + c * (lo - region_mean)                  # contrast about each region's own mean
    ynew = b * (base + s * mid + d * hi)
    out = ynew[..., None] + sat[..., None] * b[..., None] * chroma
    return np.clip(out, 0, 255)


def fit(gt, masks_png, target, iters=12):
    """Fixed-point fit: each region's knobs move until its ratios to native match the target's."""
    m = soft_masks(masks_png, gt.shape)
    hard = {r: m[r] > 0.5 for r in REGIONS}
    want = {r: ratios(gt, target, hard[r]) for r in REGIONS}
    p = json.loads(json.dumps(DEFAULT))
    for _ in range(iters):
        got = apply(gt, m, p)
        for r in REGIONS:
            have = ratios(gt, got, hard[r])
            for k in KEYS:
                p[r][k] *= float(np.clip(want[r][k] / max(have[k], 1e-3), 0.8, 1.25))
    err = {r: {k: round(ratios(gt, apply(gt, m, p), hard[r])[k] - want[r][k], 3) for k in KEYS} for r in REGIONS}
    return p, want, err


def main():
    cmd = sys.argv[1]
    if cmd == "fit":
        gt, mask, target, out = sys.argv[2:6]
        p, want, err = fit(load(gt), mask, load(target))
        json.dump({"params": p, "target_ratios": want, "fit_error": err, "target": os.path.abspath(target)},
                  open(out, "w"), indent=1)
        print(json.dumps(err))
    elif cmd == "apply":
        params, src, masks, dst = sys.argv[2:6]
        p = json.load(open(params))["params"]
        os.makedirs(dst, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(src, "f_*.png"))):
            n = int(re.search(r"f_(\d+)", f).group(1))
            img = load(f)
            out = apply(img, soft_masks(os.path.join(masks, f"m_{n:04d}.png"), img.shape), p)
            Image.fromarray((out + 0.5).astype(np.uint8)).save(os.path.join(dst, os.path.basename(f)))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
