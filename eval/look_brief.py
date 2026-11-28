"""The look brief: how a visual reference look differs from native Cycles, written down as a few hundred relative
region statistics (a colourist's notes, in numbers). Measured ONCE from reference/native pairs, aggregated over
frames by median, then frozen. The look fit (nts/lookfit.py) targets "the brief applied to our own native render" and
never opens a reference image (PROVENANCE.md, "The look").

Descriptors of one display image (8-bit sRGB floats) and its Region mask (R face, G hands, B hair):
  face grid   6x8 cells in face-box coordinates, full res: mean L (ratio), a, b (difference)
  regions     face / hands / hair at full res on crops, rest at 1/4 res: eval/look.py's measures (bright,
              contrast, detail, shading, sat), L percentiles 5..95 (ratio), hue angle (difference), band energy at 3 scales (ratio)
  bg grid     4x6 cells over the non-character pixels at 1/4 res: mean L (ratio), a, b (difference)
The same layout (crop boxes, face box) is used for native and candidate, so their descriptors line up cell by cell.

Run: look_brief.py measure <refs_dir> <out.json> [--exclude ep_f####,...]
       refs_dir holds native_<ep>_f####.png, ref_<ep>_f####.png and m_<ep>_####.png
"""
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval.look_score import lab  # noqa: E402
from eval.metrics import box, load  # noqa: E402

FACE_GRID = (8, 6)   # rows, cols
BG_GRID = (6, 4)
PCT = (5, 25, 50, 75, 95)
BANDS = (1, 2, 4)
DS = 4
REGIONS = ("face", "hands", "hair")
# how each descriptor compares: ratio (multiplicative) or diff (additive)
KIND = {"L": "ratio", "a": "diff", "b": "diff", "bright": "ratio", "contrast": "ratio", "detail": "ratio",
        "shading": "ratio", "sat": "ratio", "hue": "diff", **{f"p{p}": "ratio" for p in PCT},
        **{f"band{r}": "ratio" for r in BANDS}}


def _bbox(m, margin, shape):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return (max(ys.min() - margin, 0), min(ys.max() + 1 + margin, shape[0]),
            max(xs.min() - margin, 0), min(xs.max() + 1 + margin, shape[1]))


def layout(region):
    """Crop boxes for one frame, from its Region mask (H, W, 3 in 0..1). The face is the hero's: the face pixels
    within 160 px of their median, so a second character elsewhere in frame doesn't stretch the face box."""
    face, hands, hair = (region[..., k] > 0.5 for k in range(3))
    out = {"shape": region.shape[:2]}
    if face.sum() >= 200:
        ys, xs = np.nonzero(face)
        cy, cx = np.median(ys), np.median(xs)
        near = np.zeros_like(face)
        keep = (np.abs(ys - cy) < 160) & (np.abs(xs - cx) < 160)
        near[ys[keep], xs[keep]] = True
        out["face_box"] = _bbox(near, 0, face.shape)
        y0, y1, x0, x1 = out["face_box"]
        head = near | (hair & (np.abs(np.arange(face.shape[0])[:, None] - (y0 + y1) / 2) < 1.5 * (y1 - y0))
                       & (np.abs(np.arange(face.shape[1])[None, :] - (x0 + x1) / 2) < 1.5 * (x1 - x0)))
        out["head"] = _bbox(head, 16, face.shape)
    if hands.sum() >= 200:
        out["hands"] = _bbox(hands, 16, face.shape)
    return out


def crops(lay):
    """Name -> (y0, y1, x0, x1) of the full-res crops the descriptors read."""
    return {k: lay[k] for k in ("head", "hands") if k in lay}


def _region_stats(img, L, ab, m, prefix, out):
    if m.sum() < 100:
        return
    y = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    hy, my = y - box(y, 6), box(y, 24)
    out[f"{prefix}.bright"] = float(y[m].mean())
    out[f"{prefix}.contrast"] = float(y[m].std())
    out[f"{prefix}.detail"] = float(hy[m].std())
    out[f"{prefix}.shading"] = float(my[m].std())
    out[f"{prefix}.sat"] = float(np.hypot(ab[..., 0], ab[..., 1])[m].mean())
    for p, v in zip(PCT, np.percentile(L[m], PCT)):
        out[f"{prefix}.p{p}"] = float(max(v, 3.0))
    out[f"{prefix}.hue"] = float(np.degrees(np.arctan2(ab[..., 1][m].mean(), ab[..., 0][m].mean())))
    for r in BANDS:
        band = box(L, r) - box(L, 2 * r)
        out[f"{prefix}.band{r}"] = float((band[m] ** 2).mean() + 1e-4)


def _grid(Lab, m, box_, grid, prefix, out, min_px):
    y0, y1, x0, x1 = box_
    rows, cols = grid
    ye = np.linspace(y0, y1, rows + 1).round().astype(int)
    xe = np.linspace(x0, x1, cols + 1).round().astype(int)
    for i in range(rows):
        for j in range(cols):
            mm = m[ye[i]:ye[i + 1], xe[j]:xe[j + 1]]
            if mm.sum() < min_px:
                continue
            c = Lab[ye[i]:ye[i + 1], xe[j]:xe[j + 1]][mm]
            out[f"{prefix}.{i}.{j}.L"] = float(max(c[:, 0].mean(), 3.0))
            out[f"{prefix}.{i}.{j}.a"] = float(c[:, 1].mean())
            out[f"{prefix}.{i}.{j}.b"] = float(c[:, 2].mean())


def describe(parts, regions, lay):
    """parts: name -> display image (float 0..255) for each crop in crops(lay), plus "small" = the whole frame at 1/DS
    res. regions: the same keys, with the Region mask cut/downsampled to match. Returns {descriptor: value}."""
    out = {}
    if "head" in parts:
        img, reg = parts["head"], regions["head"]
        Lab = lab(img)
        face, hair = reg[..., 0] > 0.5, reg[..., 2] > 0.5
        hy0, _, hx0, _ = lay["head"]
        fy0, fy1, fx0, fx1 = lay["face_box"]
        fb = (fy0 - hy0, fy1 - hy0, fx0 - hx0, fx1 - hx0)
        fmask = np.zeros_like(face)
        fmask[fb[0]:fb[1], fb[2]:fb[3]] = face[fb[0]:fb[1], fb[2]:fb[3]]
        _grid(Lab, fmask, fb, FACE_GRID, "fgrid", out, 8)
        _region_stats(img, Lab[..., 0], Lab[..., 1:], fmask, "face", out)
        _region_stats(img, Lab[..., 0], Lab[..., 1:], hair, "hair", out)
    if "hands" in parts:
        img, reg = parts["hands"], regions["hands"]
        Lab = lab(img)
        _region_stats(img, Lab[..., 0], Lab[..., 1:], reg[..., 1] > 0.5, "hands", out)
    img, reg = parts["small"], regions["small"]
    Lab = lab(img)
    rest = reg.max(-1) < 0.01
    _region_stats(img, Lab[..., 0], Lab[..., 1:], rest, "rest", out)
    _grid(Lab, rest, (0, img.shape[0], 0, img.shape[1]), BG_GRID, "bg", out, 50)
    return out


def downsample(a, k=DS):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    a = a[:h, :w]
    return a.reshape(h // k, k, w // k, k, *a.shape[2:]).mean((1, 3))


def split(full, lay):
    """Cut a full-res array (image or mask) into the parts describe() reads. Copies, so the full array can be freed."""
    out = {k: full[y0:y1, x0:x1].copy() for k, (y0, y1, x0, x1) in crops(lay).items()}
    out["small"] = downsample(full)
    return out


def describe_full(img, region, lay=None):
    lay = lay or layout(region)
    return describe(split(img, lay), split(region, lay), lay), lay


def kind(name):
    return KIND[name.rsplit(".", 1)[1]]


def compare(native, ref):
    """Per-descriptor change from native to ref: log ratio, or difference (hue wrapped to +-180)."""
    out = {}
    for k in native.keys() & ref.keys():
        if kind(k) == "ratio":
            out[k] = float(np.log(ref[k] / native[k]))
        else:
            d = ref[k] - native[k]
            out[k] = float((d + 180) % 360 - 180) if k.endswith(".hue") else float(d)
    return out


def apply(brief, native):
    """Target descriptors: the brief applied to a native render's own descriptors."""
    out = {}
    for k, v in native.items():
        if k in brief["median"]:
            out[k] = v * float(np.exp(brief["median"][k])) if kind(k) == "ratio" else v + brief["median"][k]
    return out


def measure(refs, out_path, exclude=()):
    per = {}
    for np_path in sorted(glob.glob(os.path.join(refs, "native_*_f*.png"))):
        ep, f = re.search(r"native_(.+)_f(\d+)\.png$", np_path).groups()
        key = f"{ep}_f{f}"
        rp, mp = os.path.join(refs, f"ref_{key}.png"), os.path.join(refs, f"m_{ep}_{f}.png")
        if not (os.path.exists(rp) and os.path.exists(mp)):
            print(f"skip {key}: no ref or mask")
            continue
        nat = load(np_path)
        ref = load(rp, (nat.shape[1], nat.shape[0]))
        reg = np.asarray(Image.open(mp).convert("RGB").resize((nat.shape[1], nat.shape[0]), Image.NEAREST),
                         np.float32) / 255
        dn, lay = describe_full(nat, reg)
        dr, _ = describe_full(ref, reg, lay)
        per[key] = compare(dn, dr)
        print(f"{key}: {len(per[key])} descriptors; face bright x{np.exp(per[key].get('face.bright', np.nan)):.2f} "
              f"detail x{np.exp(per[key].get('face.detail', np.nan)):.2f}  hands sat "
              f"x{np.exp(per[key].get('hands.sat', np.nan)):.2f}", flush=True)
    fit_keys = [k for k in per if k not in exclude]
    names = sorted(set().union(*[per[k].keys() for k in fit_keys]))
    med, mad, n = {}, {}, {}
    for d in names:
        v = np.array([per[k][d] for k in fit_keys if d in per[k]])
        if len(v) < max(1, (len(fit_keys) + 2) // 3):  # a cell seen in under a third of the frames isn't global
            continue
        med[d] = float(np.median(v))
        mad[d] = float(np.median(np.abs(v - med[d])) * 1.4826)
        n[d] = int(len(v))
    brief = {"frames": fit_keys, "excluded": [k for k in per if k in exclude], "median": med, "spread": mad, "n": n,
             "per_frame": per, "note": "relative statistics only; see PROVENANCE.md 'The look'"}
    json.dump(brief, open(out_path, "w"), indent=1)
    write_md(brief, os.path.splitext(out_path)[0] + ".md")
    print(f"brief: {len(med)} descriptors from {len(fit_keys)} frames -> {out_path}")


def write_md(brief, path):
    med, sp = brief["median"], brief["spread"]
    lines = [f"# Look brief ({len(brief['frames'])} frames)", "",
             "Ratios are shown as ×, differences as ±. Spread is the robust std across frames, in the same units.", "",
             "## Regions", "", "| descriptor | " + " | ".join(("face", "hands", "hair", "rest")) + " |",
             "|---|" + "---|" * 4]
    for d in ("bright", "contrast", "detail", "shading", "sat", *[f"p{p}" for p in PCT], "hue",
              *[f"band{r}" for r in BANDS]):
        cells = []
        for r in ("face", "hands", "hair", "rest"):
            k = f"{r}.{d}"
            if k not in med:
                cells.append("–")
            elif kind(k) == "ratio":
                cells.append(f"×{np.exp(med[k]):.2f} (±{sp[k]:.2f})")
            else:
                cells.append(f"{med[k]:+.1f} (±{sp[k]:.1f})")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    for name, (rows, cols) in (("fgrid", FACE_GRID), ("bg", BG_GRID)):
        lines += ["", f"## {'Face grid (face-box coordinates, top row = brow)' if name == 'fgrid' else 'Background grid'}"
                  ": L ×ratio / a ± / b ±", "", "| row | " + " | ".join(str(j) for j in range(cols)) + " |",
                  "|---|" + "---|" * cols]
        for i in range(rows):
            cells = []
            for j in range(cols):
                k = f"{name}.{i}.{j}"
                cells.append(f"×{np.exp(med[k + '.L']):.2f} {med[k + '.a']:+.1f} {med[k + '.b']:+.1f}"
                             if k + ".L" in med else "–")
            lines.append(f"| {i} | " + " | ".join(cells) + " |")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    a = sys.argv
    if len(a) < 4 or a[1] != "measure":
        sys.exit(__doc__)
    ex = tuple(a[a.index("--exclude") + 1].split(",")) if "--exclude" in a else ()
    measure(a[2], a[3], ex)


if __name__ == "__main__":
    main()
