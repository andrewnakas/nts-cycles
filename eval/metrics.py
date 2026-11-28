"""Frame-sequence metrics against a reference, in 8-bit display space (AgX-tonemapped PNG/JPG), so any upscaler's output
(ours, another upscaler's frames from the Windows laptop, bicubic) is scored the same way.

Ported from cycles-harness tools/metrics.py (PSNR, SSIM, tErr with cut-skip); tests/test_metrics.py pins it to the
numbers measured there. Added: masked variants, high-pass luminance PSNR (structure change, ignoring tone), sharpness.
Run: metrics.py <gt_dir> <test_dir> [--frames a-b] [--skip N] [--cuts cuts.json] [--crop x0,y0,x1,y1] [--json out.json]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image

FRAME_RE = re.compile(r"f_(\d+)\.(png|jpg|jpeg)$", re.I)


def load(path, size=None):
    im = Image.open(path).convert("RGB")
    if size and im.size != size:
        im = im.resize(size, Image.BICUBIC)
    return np.asarray(im, np.float32)


def frames_in(d):
    out = {}
    for p in glob.glob(os.path.join(d, "f_*.*")):
        m = FRAME_RE.search(p)
        if m:
            out[int(m.group(1))] = p
    return out


def box(a, r):
    """Mean over a (2r+1)^2 window, edges clamped, via summed-area tables (float64: float32 sums lose the variance)."""
    p = np.pad(a.astype(np.float64), ((r + 1, r), (r + 1, r)), mode="edge")
    s = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return (s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]) / (k * k)


def luma(a):
    return a @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def ssim_map(a, b, r=4):
    x, y = luma(a), luma(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = box(x, r), box(y, r)
    vx, vy, cxy = box(x * x, r) - mx * mx, box(y * y, r) - my * my, box(x * y, r) - mx * my
    return ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def ssim(a, b, r=4, mask=None):
    m = ssim_map(a, b, r)
    return float(np.mean(m if mask is None else m[mask]))


def psnr(a, b, mask=None):
    d = (a - b) ** 2
    mse = float(np.mean(d if mask is None else d[mask]))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def highpass(a, r=3):
    """Luminance minus its local mean: the structure (edges, pores, lashes) without the tone."""
    y = luma(a)
    return y - box(y, r)


def hp_psnr(a, b, mask=None):
    """PSNR of high-pass luminance. A tone/lighting change scores high; moved or invented detail scores low."""
    return psnr(highpass(a) + 128, highpass(b) + 128, mask)


def sharpness(a, mask=None):
    """Mean gradient magnitude of luminance: report next to temporal metrics, which reward blur."""
    y = luma(a)
    g = np.hypot(np.diff(y, axis=1)[:-1], np.diff(y, axis=0)[:, :-1])
    return float(np.mean(g if mask is None else g[mask[:-1, :-1]]))


def compare(gt_dir, test_dir, frames=None, skip=0, cuts=(), crop=None):
    gt, te = frames_in(gt_dir), frames_in(test_dir)
    common = sorted(set(gt) & set(te))
    if frames:
        common = [f for f in common if frames[0] <= f <= frames[1]]
    cuts = sorted(cuts)
    rows, prev = [], None
    for f in common:
        g = load(gt[f])
        t = load(te[f], (g.shape[1], g.shape[0]))
        if crop:
            x0, y0, x1, y1 = crop
            g, t = g[y0:y1, x0:x1], t[y0:y1, x0:x1]
        since_cut = f - max([c for c in cuts if c <= f], default=common[0])
        row = {"frame": f, "since_cut": since_cut, "psnr": psnr(g, t), "ssim": ssim(g, t)}
        if prev and prev[0] == f - 1 and f not in cuts:
            row["terr"] = float(np.mean(np.abs((t - prev[2]) - (g - prev[1]))))
        prev = (f, g, t)
        if since_cut >= skip:
            rows.append(row)
    agg = {k: float(np.mean([r[k] for r in rows if k in r])) for k in ("psnr", "ssim", "terr") if any(k in r for r in rows)}
    return agg, rows


def main():
    a = sys.argv

    def opt(n, d=None):
        return a[a.index(n) + 1] if n in a else d
    gt_dir, test_dir = a[1], a[2]
    frames = tuple(map(int, opt("--frames").split("-"))) if opt("--frames") else None
    cuts_path = opt("--cuts", os.path.join(os.path.dirname(os.path.abspath(test_dir)), "cuts.json"))
    cuts = json.load(open(cuts_path)) if os.path.exists(cuts_path) else []
    crop = tuple(map(int, opt("--crop").split(","))) if opt("--crop") else None
    agg, rows = compare(gt_dir, test_dir, frames, int(opt("--skip", 0)), cuts, crop)
    print(f"{test_dir}: PSNR {agg.get('psnr', 0):.2f} dB  SSIM {agg.get('ssim', 0):.4f}  tErr {agg.get('terr', 0):.3f}  ({len(rows)} frames)")
    if opt("--json"):
        json.dump({"agg": agg, "rows": rows}, open(opt("--json"), "w"), indent=1)


if __name__ == "__main__":
    main()
