"""Look fit: solve ONE global set of Cycles settings so our renders match the look brief across many frames.

Inputs per frame: a light-group render (datagen/blender/lightgroups.py), with layers LG_<group> (denoised linear
radiance per emitter group), Mist, Region (face R / hands G / hair B) and Image (the production result). The target is
the frozen look brief (eval/look_brief.py) applied to that frame's own native render. No reference image is ever
opened here (PROVENANCE.md, "The look").

Model (exact for lights, first-order for materials):
  L(p)     = sum_g w_g * LG_g                            path tracing is linear in emitter strength
  L       *= 1 + (s - 1) * skin_mask + (h - 1) * hair    RGB albedo gain of skin (s) and hair (h)
  L        = mix(L, haze, 0.55 * mist)                   the production compositor's depth haze (affine)
  display  = AgX(L * 2^exposure)                         Blender's exact view transform, through a 3D LUT
Objective: distance between the candidate's descriptors and the target descriptors. Face-grid and background-grid
CIELAB ΔE, plus region tone, saturation and hue. With --texture, region detail and band energy are added. Held-out
frames report generalisation: the settings must apply to everything, not only to the frames they were fit on.

Run: lookfit.py check <lg_dir>                                   linearity: sum of groups vs the rendered Image
     lookfit.py fit <brief.json> <lg_dir>[,<lg_dir>...] <out.json> [--holdout ep_f####,...] [--steps 25] [--texture] [--skin-range lo,hi]
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
from PIL import Image as PImage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval import look_brief as lb  # noqa: E402
from nts.io_exr import read_layers, stack  # noqa: E402
from nts.tonemap import Tonemapper  # noqa: E402

GROUPS = ("HeroKey", "HeroFill", "HeroRim", "HeroKicker", "HeroEyes", "set", "emit", "world")
HAZE = np.array([0.42, 0.46, 0.55], np.float32)


class AgxLUT:
    """Blender's view transform as a 3D LUT over log2-encoded scene-linear input: fast and exact to < 1/255."""

    def __init__(self, view="AgX", look="AgX - Punchy", exposure=0.0, n=49, lo=-12.0, hi=8.0):
        self.lo, self.hi, self.n = lo, hi, n
        g = 2.0 ** np.linspace(lo, hi, n, dtype=np.float32)
        r, gg, b = np.meshgrid(g, g, g, indexing="ij")
        grid = np.stack([r, gg, b], -1).reshape(-1, 3)
        grid[np.all(grid <= 2.0 ** lo * 1.0001, axis=1)] = 0.0  # the darkest corner means black
        tm = Tonemapper(view, look, exposure)
        img = np.ascontiguousarray(grid.reshape(n * n, n, 3), dtype=np.float32)
        tm.cpu.applyRGB(img)
        self.table = (np.clip(img.reshape(n, n, n, 3), 0, 1) * 255.0).astype(np.float32)

    def __call__(self, lin):
        n = self.n
        x = (np.log2(np.maximum(lin, 2.0 ** self.lo)) - self.lo) * np.float32((n - 1) / (self.hi - self.lo))
        x = np.clip(x, 0, n - 1.0001).astype(np.float32)
        i = x.astype(np.int32)
        f = x - i
        flat = self.table.reshape(-1, 3)
        base = (i[..., 0] * n + i[..., 1]) * n + i[..., 2]
        out = 0.0
        for dr in (0, 1):
            wr = f[..., 0] if dr else 1 - f[..., 0]
            for dg in (0, 1):
                wrg = wr * (f[..., 1] if dg else 1 - f[..., 1])
                for db in (0, 1):
                    w = wrg * (f[..., 2] if db else 1 - f[..., 2])
                    out = out + w[..., None] * flat[base + (dr * n * n + dg * n + db)]
        return out


def load_lg(path):
    ch = read_layers(path)
    groups = {g: stack(ch, f"LG_{g}.R", f"LG_{g}.G", f"LG_{g}.B") for g in GROUPS if f"LG_{g}.R" in ch}
    mist = ch.get("Mist.Z", ch.get("Mist.V"))
    region = stack(ch, "Region.R", "Region.G", "Region.B")
    image = stack(ch, "Image.R", "Image.G", "Image.B")
    return groups, (mist if mist is not None else np.zeros(image.shape[:2], np.float32)), region, image


BASIS = {"v": ("value", 0.55), "s": ("sat", 1.4), "h": ("hue", -0.03)}  # datagen/jobs/lg_skin_basis.sh defaults
# a basis rendered around another point (datagen/jobs/lg_iter.sh) writes its steps to <lg root>/basis.json


def basis_steps(lg_path):
    bj = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(lg_path))), "basis.json")
    return {b: tuple(v) for b, v in json.load(open(bj)).items()} if os.path.exists(bj) else BASIS


def basis_coef(p, steps=BASIS):
    """Exponents of each skin-basis render for skin settings p["skin"] = {value, sat, hue} (HSV node units),
    relative to the base render."""
    sk = p["skin"]
    return {b: (sk[k] / step if k == "hue" else np.log(sk[k]) / np.log(step)) for b, (k, step) in steps.items()}


def compose(groups, mist, region, p, basis=None, steps=BASIS):
    """basis: {b: {group: log(G_b / G)}} from real renders with the skin base colour changed one way at a time; with
    it, skin is exact at the rendered points and log-linear between. Without it, an image-space RGB gain on the skin
    mask stands in (first order only: no SSS, desaturates in AgX)."""
    hair = np.clip(region[..., 2], 0, 1)[..., None]
    if basis:
        c = basis_coef(p, steps)
        lin = sum(p["w"][g] * groups[g] * np.exp(sum(c[b] * basis[b][g] for b in basis)) for g in groups)
        lin = lin * (1 + (np.asarray(p["hair"], np.float32) - 1) * hair)
    else:
        lin = sum(p["w"][g] * groups[g] for g in groups)
        skin = np.clip(region[..., 0] + region[..., 1], 0, 1)[..., None]
        lin = lin * (1 + (np.asarray(p["skin"], np.float32) - 1) * skin + (np.asarray(p["hair"], np.float32) - 1) * hair)
    a = (0.55 * mist)[..., None]
    lin = lin * (1 - a) + HAZE * a
    return lin * 2.0 ** p["exposure"]


def check(lg_dir):
    for path in sorted(glob.glob(os.path.join(lg_dir, "f_*.exr"))):
        groups, mist, region, image = load_lg(path)
        p = p0(groups)
        rec = compose(groups, mist, region, p)
        rel = np.abs(rec - image).mean() / max(image.mean(), 1e-6)
        face = region[..., 0] > 0.5
        tot, ftot = sum(v.mean() for v in groups.values()), sum(v[face].mean() for v in groups.values()) if face.any() else 1
        print(f"{os.path.basename(path)}: sum-vs-Image mean rel err {rel:.4f}\n  frame share " +
              " ".join(f"{g} {groups[g].mean() / max(tot, 1e-9):.2f}" for g in sorted(groups)) + "\n  face share  " +
              (" ".join(f"{g} {groups[g][face].mean() / max(ftot, 1e-9):.2f}" for g in sorted(groups)) if face.any() else "-"))


TONE = ("bright", "contrast", "shading", "sat", *[f"p{q}" for q in lb.PCT])
TEXTURE = ("detail", *[f"band{r}" for r in lb.BANDS])
REGION_W = {"face": 2.0, "hands": 1.5, "hair": 1.0, "rest": 1.0}


def p0(groups, basis=False):
    return {"w": dict.fromkeys(groups, 1.0), "skin": {"value": 1.0, "sat": 1.0, "hue": 0.0} if basis else [1.0] * 3,
            "hair": [1.0] * 3, "exposure": 0.0}


def load_basis(path, region):
    """Log ratios of each skin-basis render to the base render, per group, limited to the (grown) skin mask so the
    separate denoising of each render doesn't add noise elsewhere. None if any basis render is missing."""
    d, f = os.path.dirname(path), os.path.basename(path)
    paths = {b: os.path.join(d + f"_skin{b}", f) for b in BASIS}
    if not all(os.path.exists(q) for q in paths.values()):
        return None
    base, _, _, _ = load_lg(path)
    near = (lb.box(np.clip(region[..., 0] + region[..., 1], 0, 1), 6) > 0.01)[..., None]
    out = {}
    for b, q in paths.items():
        gb, _, _, _ = load_lg(q)
        out[b] = {g: np.where(near & (base[g] > 1e-5) & (gb[g] > 1e-5),
                              np.clip(np.log(np.maximum(gb[g], 1e-5) / np.maximum(base[g], 1e-5)), -3, 3),
                              0).astype(np.float32) for g in base}
    return out


class Frame:
    """One frame's light groups, cut into the parts the descriptors read (full-res head and hands crops, plus the
    whole frame at 1/4 res), and its target descriptors."""

    def __init__(self, key, path, brief, lut):
        groups, mist, region, _ = load_lg(path)
        self.key, self.path, self.lay = key, path, lb.layout(region)
        gp = {g: lb.split(v, self.lay) for g, v in groups.items()}
        mp, rp = lb.split(mist, self.lay), lb.split(region, self.lay)
        self.parts = {k: ({g: gp[g][k] for g in groups}, mp[k], rp[k]) for k in rp}
        bas = load_basis(path, region)
        self.steps = basis_steps(path)
        bas = {b: {g: lb.split(v, self.lay) for g, v in bas[b].items()} for b in bas} if bas else None
        self.basis = {k: {b: {g: bas[b][g][k] for g in groups} for b in bas} for k in rp} if bas else None
        self.native = self.describe(p0(sorted(groups), bool(bas)), lut)
        self.target = lb.apply(brief, self.native)

    def describe(self, p, lut):
        imgs = {k: lut(compose(g, m, r, p, self.basis[k] if self.basis else None, self.steps))
                for k, (g, m, r) in self.parts.items()}
        return lb.describe(imgs, {k: r for k, (_, _, r) in self.parts.items()}, self.lay)


def load_frames(brief, lg_dirs, lut):
    frames = []
    for d in lg_dirs:
        ep = re.sub(r"_lg$", "", os.path.basename(os.path.normpath(d)))  # light-group dirs are named <episode>_lg
        for path in sorted(glob.glob(os.path.join(d, "f_*.exr"))):
            f = int(re.search(r"f_(\d+)", path).group(1))
            frames.append(Frame(f"{ep}_f{f:04d}", path, brief, lut))
            print(f"loaded {frames[-1].key}: {len(frames[-1].target)} target descriptors", flush=True)
    return frames


def grid_de(c, t, prefix):
    cells = {k.rsplit(".", 1)[0] for k in t if k.startswith(prefix + ".")}
    d = [np.sqrt(sum((c[f"{k}.{ch}"] - t[f"{k}.{ch}"]) ** 2 for ch in "Lab")) for k in cells if f"{k}.L" in c]
    return float(np.mean(d)) if d else 0.0


def frame_error(fr, p, lut, texture=False):
    """Descriptor distance in roughly ΔE units: grids in CIELAB ΔE, a 5 % ratio error counts 1, 3° of hue counts 1."""
    c, t = fr.describe(p, lut), fr.target
    parts = {"face_grid": (3.0, grid_de(c, t, "fgrid")), "bg_grid": (1.0, grid_de(c, t, "bg"))}
    names = TONE + (TEXTURE if texture else ())
    for r, w in REGION_W.items():
        errs = [20 * abs(np.log(c[f"{r}.{d}"] / t[f"{r}.{d}"])) for d in names if f"{r}.{d}" in t and f"{r}.{d}" in c]
        if f"{r}.hue" in t and f"{r}.hue" in c:
            errs.append(abs((c[f"{r}.hue"] - t[f"{r}.hue"] + 180) % 360 - 180) / 3)
        if errs:
            parts[r] = (w, float(np.mean(errs)))
    total = sum(w * e for w, e in parts.values()) / sum(w for w, _ in parts.values())
    return total, {k: e for k, (_, e) in parts.items()}


PRIOR = 0.5  # residual per log-unit of light weight away from 1: keep lights near production unless the brief needs it


def frame_residuals(fr, p, lut, texture=False):
    """frame_error as a fixed-length residual vector (its keys come from the masks, not the image), for least squares."""
    c, t = fr.describe(p, lut), fr.target
    keys = sorted(t)
    nf = max(sum(k.startswith("fgrid.") for k in keys), 1)
    nb = max(sum(k.startswith("bg.") for k in keys), 1)
    names = TONE + (TEXTURE if texture else ())
    r = []
    for k in keys:
        pre, d = k.split(".", 1)[0], k.rsplit(".", 1)[1]
        if k not in c:
            r.append(0.0)
        elif pre == "fgrid":
            r.append((c[k] - t[k]) * np.sqrt(3.0 / nf))
        elif pre == "bg":
            r.append((c[k] - t[k]) * np.sqrt(1.0 / nb))
        elif pre in REGION_W and d in names:
            r.append(20 * np.log(c[k] / t[k]) * np.sqrt(REGION_W[pre] / len(names)))
        elif pre in REGION_W and d == "hue":
            r.append(((c[k] - t[k] + 180) % 360 - 180) / 3 * np.sqrt(REGION_W[pre] / len(names)))
    return np.array(r, np.float64)


def pack(p, groups):
    sk = p["skin"]
    skin = [np.log(sk["value"]), np.log(sk["sat"]), sk["hue"]] if isinstance(sk, dict) else list(np.log(sk))
    return np.array([np.log(p["w"][g]) for g in groups] + skin + list(np.log(p["hair"])) + [p["exposure"]])


def unpack(v, groups, basis=False):
    n = len(groups)
    sk = v[n:n + 3]
    skin = ({"value": float(np.exp(sk[0])), "sat": float(np.exp(sk[1])), "hue": float(sk[2])} if basis
            else [float(x) for x in np.exp(sk)])
    return {"w": {g: float(np.exp(v[i])) for i, g in enumerate(groups)}, "skin": skin,
            "hair": [float(x) for x in np.exp(v[n + 3:n + 6])], "exposure": float(v[n + 6])}


def save_preview(fr, p, lut, path):
    groups, mist, region, _ = load_lg(fr.path)
    bas = load_basis(fr.path, region) if fr.basis else None
    PImage.fromarray(lut(compose(groups, mist, region, p, bas, fr.steps)).round().clip(0, 255).astype(np.uint8)).save(path)


def fit(brief_path, lg_dirs, out, holdout=(), steps=25, texture=False, skin_range=None):
    brief = json.load(open(brief_path))
    lut = AgxLUT(*json.load(open(os.path.join(lg_dirs[0], "meta.json")))["view"])  # production view; exposure fit is a delta
    frames = load_frames(brief, lg_dirs, lut)
    train = [f for f in frames if f.key not in holdout]
    test = [f for f in frames if f.key in holdout]
    groups = sorted(set.intersection(*[set(f.parts["small"][0]) for f in frames]))
    use_basis = any(f.basis for f in frames)
    if use_basis:  # the skin basis is real renders on a subset: fit on that subset only, never mix the two skin models
        dropped = [f.key for f in frames if not f.basis]
        if dropped:
            print(f"skin basis present: dropping {len(dropped)} frames without it: {dropped}", flush=True)
        frames = [f for f in frames if f.basis]
        train = [f for f in frames if f.key not in holdout]
        test = [f for f in frames if f.key in holdout]
    print(f"fit on {len(train)} frames, hold out {len(test)}; groups {groups}; skin basis {use_basis}; "
          f"texture term {texture}", flush=True)
    P0 = p0(groups, use_basis)

    def unp(v):
        return unpack(v, groups, use_basis)

    def loss(v, fs):
        return float(np.mean([frame_error(fr, unp(v), lut, texture)[0] for fr in fs]))

    def resid(v):
        pp = unp(v)
        r = [frame_residuals(fr, pp, lut, texture) / np.sqrt(len(train)) for fr in train]
        return np.concatenate(r + [PRIOR * v[:len(groups)]])

    lo, hi = np.full(len(groups) + 7, np.log(0.02)), np.full(len(groups) + 7, np.log(8.0))
    lo[-1], hi[-1] = -3, 3
    if use_basis:  # skin value/sat within 2x of the basis renders, hue within +-0.1 turn
        n = len(groups)
        lo[n:n + 2], hi[n:n + 2] = np.log(0.25), np.log(2.5)
        lo[n + 2], hi[n + 2] = -0.1, 0.1
        if skin_range:  # keep the skin's base colour near the original (identity): darken with light instead
            lo[n], hi[n] = np.log(skin_range[0]), np.log(skin_range[1])
    v = pack(P0, groups)
    start = {"train": loss(v, train), "test": loss(v, test) if test else None}
    # Levenberg-Marquardt on log-parameters, forward-difference Jacobian
    r = resid(v)
    cost, lam, eps = float(r @ r), 1e-2, 0.01
    t0 = time.time()
    for it in range(1, steps + 1):
        J = np.empty((len(r), len(v)))
        for i in range(len(v)):
            e = np.zeros_like(v)
            e[i] = eps
            J[:, i] = (resid(v + e) - r) / eps
        A, g = J.T @ J, J.T @ r
        while True:
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            nv = np.clip(v + step, lo, hi)
            nr = resid(nv)
            if float(nr @ nr) < cost:
                v, r, cost, lam = nv, nr, float(nr @ nr), max(lam / 3, 1e-6)
                break
            lam *= 4
            if lam > 1e6:
                break
        print(f"iter {it} cost {cost:.4f} train {loss(v, train):.3f} lambda {lam:.1e} |step| {np.abs(step).max():.3f} "
              f"{time.time() - t0:.0f}s", flush=True)
        if lam > 1e6 or np.abs(step).max() < 1e-3:
            break
    best = (loss(v, train), v)
    p = unp(best[1])
    per = {}
    for f in frames:
        b, bp = frame_error(f, P0, lut, texture)
        a, ap = frame_error(f, p, lut, texture)
        per[f.key] = {"before": b, "after": a, "before_parts": bp, "after_parts": ap, "held_out": f.key in holdout}
    res = {"brief": brief_path, "params": p, "start": start, "train": best[0],
           "test": loss(best[1], test) if test else None, "per_frame": per}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("params", "start", "train", "test")}, indent=1))
    for k, r in per.items():
        print(f"  {k:28s} {r['before']:.2f} -> {r['after']:.2f}  face grid ΔE {r['before_parts']['face_grid']:.2f} -> "
              f"{r['after_parts']['face_grid']:.2f}{'  (held out)' if r['held_out'] else ''}")
    prev = os.path.splitext(out)[0] + "_preview"
    os.makedirs(prev, exist_ok=True)
    for f in (test or frames)[:6]:
        save_preview(f, P0, lut, os.path.join(prev, f"native_{f.key}.png"))
        save_preview(f, p, lut, os.path.join(prev, f"pred_{f.key}.png"))
    print(f"previews -> {prev}")


def main():
    a = sys.argv
    if len(a) < 3:
        sys.exit(__doc__)
    if a[1] == "check":
        return check(a[2])
    hold = tuple(a[a.index("--holdout") + 1].split(",")) if "--holdout" in a else ()
    steps = int(a[a.index("--steps") + 1]) if "--steps" in a else 25
    sr = tuple(map(float, a[a.index("--skin-range") + 1].split(","))) if "--skin-range" in a else None
    fit(a[2], a[3].split(","), a[4], hold, steps, "--texture" in a, sr)


if __name__ == "__main__":
    main()
