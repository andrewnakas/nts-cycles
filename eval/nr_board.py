"""NR-1 board: one command scores every method on every eval clip and appends a markdown table (2026-11-18).

Per clip and method (8-bit display units; each method is compared with ITS OWN input render: ours with the v9d render,
the reference with the episode's native-look render it was fed):
  flk      true-motion flicker inside the eroded face mask (as eval/flicker_mv.py); p90 alongside
  follow   share of the true motion the face follows (eval/flicker_mv.py); compare with the render's own value
  flkF     true-motion flicker over the whole frame (newly visible pixels left out)
  textdB   PSNR to the input render on print/screen text: 5x5 luma range > 76, grown 9 px, people excluded. Gate >= 38
  lump     mean |8-32 px band change| (difference of box-Gaussians, sigma ~2.5 and ~10.5 px) on smooth, non-person
           areas (9x9 std of the render < 3). Blobs and smears on walls and floors. Lower = cleaner
  blotch   the same band change (luma + chroma) inside the eroded face mask. Lower = cleaner skin
  fineBG   fine-detail energy (|x - box r1|) of the output over the input's, outside face/hands/hair (cloth, walls,
           props): above 1 = texture was added there
  fineF    the same inside the face
  textAf   text PSNR after a per-frame, per-channel affine colour fit to the input render, so a grade is not scored
           as damage (gate v2: >= 30 dB and above the reference's)
  faceL hairL handL   median CIELAB lightness of the output over the input's, per region (tone branch target: the
           look brief's ratios at --tone 1)
  faceC hairC restC   mean chroma over the input's (rest = outside the person)
  hue      face hue shift in degrees (CIELAB, from the mean a, b)
  blk wht  1st and 99th percentile of the output's luma (0..1): black lift and highlight roll-off
  beardL beardF   no-stubble check. Chin + neck zone against the cheeks: lightness ratio and fine-energy ratio of the
           output, each over the same ratio in the input render. 1.00 = the jaw is no darker / no more speckled,
           relative to the cheeks, than in the input. Gate v2 (against the native render): within 5% (beardL >= 0.95,
           beardF <= 1.05)
With --faces <dir>: face crops (every 10th frame, 160 px, output and its input render) go to <dir>/<method>_<clip>.npz
for eval/nr_identity.py (facenet cosine, run on the Mac).

Clips and frame ranges come from data/eval/holdout.json (eval + heldout). Methods:
  label=<subdir>[:<refsubdir>]  folders under data/eval/ab/<clip>/ (ref defaults to v9d). The names "ref5" and "native"
  resolve per clip (zero-day keeps them under data/eval/zd); "@name" is data/v2/<clip>/name. Missing folders are
  skipped.
Run (4050, ~/scene-project/py): python -m eval.nr_board runs/board.md [--clips at140,rs345] [--step 3] [--faces dir]
       render=v9d e2=r2p_e2_b0.3 ref5=ref5:native
Needs numpy, Pillow, OpenImageIO. Eval only: nothing here feeds training.
"""
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts import warp as W  # noqa: E402
from nts.io_exr import Sequence  # noqa: E402
from nts.tone import to_lab  # noqa: E402
from eval.flicker_mv import blur, erode  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def clips():
    h = json.load(open(os.path.join(R, "data/eval/holdout.json")))
    out = {}
    for c in h["eval"] + h["heldout"]:
        t = c["tag"]
        ab = os.path.join(R, "data/eval/ab", t)
        geom = os.path.join(R, "data/raw", f"{t}_v9d")  # the same geometry in both looks; the held-out clips
        if not os.path.isdir(geom):                      # (2026-11-22) were rendered in the native look only
            geom = os.path.join(R, "data/raw", f"{t}_native")
        out[t] = dict(tag=t, frames=tuple(c["frames"]), ab=ab, geom=geom,
                      masks=os.path.join(ab, "masks"), native=os.path.join(ab, "native"), ref5=os.path.join(ab, "ref5"))
    # NR_BOARD_EXTRA="<tag>:<first>-<last>:<geom look> ...": clips outside holdout.json (episode shots, a long-shot
    # test); they have no reference output, so only our own methods can be scored on them.
    for x in os.environ.get("NR_BOARD_EXTRA", "").split():
        t, fr, g = x.split(":")
        ab = os.path.join(R, "data/eval/ab", t)
        out[t] = dict(tag=t, frames=tuple(int(v) for v in fr.split("-")), ab=ab, geom=os.path.join(R, "data/raw", f"{t}_{g}"),
                      masks=os.path.join(ab, "masks"), native=os.path.join(ab, g), ref5=os.path.join(ab, "ref5"))
    if "zd400" in out:  # zero-day's reference input and output predate the ab/ layout
        out["zd400"]["native"] = os.path.join(R, "data/eval/zd/gt")
        out["zd400"]["ref5"] = os.path.join(R, "data/eval/zd/ref5_pass2_clip_mv")
    return out


def resolve(c, name):
    if name.startswith("@"):
        return os.path.join(R, "data/v2", c["tag"], name[1:])
    return c[name] if name in ("ref5", "native") else os.path.join(c["ab"], name)


def frame(d, f):
    p = glob.glob(os.path.join(d, f"f_{f:04d}.*"))
    p = [q for q in p if q[-4:] in (".png", ".jpg")]
    return np.asarray(Image.open(p[0]).convert("RGB")).astype(np.float32) if p else None


# ------------------------------------------------------------------------------------------------- image helpers
def box(a, r):
    """Box mean of radius r along both axes (edge padded), via cumulative sums: O(1) per pixel."""
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax)
        n = a.shape[ax]
        a = (np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=ax) - np.take(c, np.arange(0, n), axis=ax)) / (2 * r + 1)
    return a


def gauss3(a, r):
    return box(box(box(a, r), r), r)  # three box passes ~ Gaussian, sigma = sqrt(r(r+1))


def maxf(m, r):
    """Binary dilation / grey max filter with a (2r+1) square, separable."""
    for ax in (0, 1):
        pad = [(0, 0)] * m.ndim
        pad[ax] = (r, r)
        p = np.pad(m, pad, mode="edge")
        n = m.shape[ax]
        m = np.maximum.reduce([np.take(p, np.arange(i, i + n), axis=ax) for i in range(2 * r + 1)])
    return m


def luma(x):
    return x @ np.array([0.299, 0.587, 0.114], np.float32)


def chroma(x):
    return np.stack([x[..., 0] - x[..., 1], 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]], -1)


def band(y):
    return gauss3(y, 2) - gauss3(y, 10)


def masks_of(c, f, shape):
    p = os.path.join(c["masks"], f"m_{f:04d}.png")
    if not os.path.exists(p):
        z = np.zeros(shape, bool)
        return z, z, z
    a = np.asarray(Image.open(p).convert("RGB"))
    return a[..., 0] > 127, a[..., 1] > 127, a[..., 2] > 127


def geometry(c, f0, f1):
    """f -> (motion at output size, valid history), as eval/flicker_mv.geometry but without the face restriction."""
    seq = Sequence(c["geom"])
    out, prev_depth = {}, None
    for f in range(f0, f1 + 1):
        fr = seq.read(f)
        h, w = fr.depth.shape
        s = seq.scale
        hw = (round(h * s), round(w * s))
        depth = W.upsample_nearest(fr.depth, s, hw)
        if prev_depth is not None and not fr.reset:
            mv = W.hr_motion(fr.mv, fr.depth, s, hw)
            out[f] = (mv.astype(np.float32), ~W.disocclusion(prev_depth, depth, mv))
        prev_depth = depth
    return out


# ------------------------------------------------------------------------------------------------------- scoring
def score(c, d, ref, geo, step, faces_out):
    f0, f1 = c["frames"]
    flk, flkF, num, den = [], [], 0.0, 0.0
    text_se, text_n, lump, blotch = 0.0, 0, [], []
    fe = {"bg": [0.0, 0.0], "face": [0.0, 0.0]}
    crops_o, crops_r = [], []
    aff_se, aff_n, tone = 0.0, 0, {}
    prev = frame(d, f0)
    if prev is None:
        return None
    for f in range(f0 + 1, f1 + 1):
        cur = frame(d, f)
        if cur is None:
            return None
        face, hands, hair = masks_of(c, f, cur.shape[:2])
        if f in geo:
            mv, ok = geo[f]
            e = np.abs(cur - W.warp(prev, mv)).mean(-1)
            okf = ok & erode(face, 4)
            if okf.sum():
                flk.append(e[okf])
            flkF.append(float(e[ok].mean()))
            moving = okf & (np.hypot(mv[..., 0], mv[..., 1]) > 0.3)
            if moving.sum() > 50:
                cb, pb = blur(cur.mean(-1)), blur(prev.mean(-1))
                pred = cb - W.warp(cb, -mv)
                num += float(((cb - pb) * pred)[moving].sum())
                den += float((pred * pred)[moving].sum())
        if (f - f0) % step == 0:
            r = frame(ref, f)
            if r is not None:
                person = maxf(face | hands | hair, 6)
                yr, yo = luma(r), luma(cur)
                rng = maxf(yr, 2) - (-maxf(-yr, 2))
                text = maxf(rng > 76, 4) & ~person
                if text.sum():
                    text_se += float(((cur - r) ** 2)[text].sum())
                    text_n += int(text.sum()) * 3
                    for ch in range(3):  # the same after the best affine map of the render's channel
                        x, y = r[..., ch][text].astype(np.float64), cur[..., ch][text].astype(np.float64)
                        vx = x.var()
                        al = ((x - x.mean()) * (y - y.mean())).mean() / vx if vx > 1e-6 else 0.0
                        aff_se += float(((y - y.mean() - al * (x - x.mean())) ** 2).sum())
                        aff_n += x.size
                smooth = (box(yr * yr, 4) - box(yr, 4) ** 2 < 9) & ~person
                db = np.abs(band(yo) - band(yr))
                if smooth.sum():
                    lump.append(float(db[smooth].mean()))
                fi = erode(face, 4)
                if fi.sum() > 100:
                    co, cr = chroma(cur), chroma(r)
                    dc = sum(np.abs(band(co[..., i]) - band(cr[..., i])) for i in (0, 1))
                    blotch.append(float((db + 0.5 * dc)[fi].mean()))
                fo, fr_ = np.abs(yo - box(yo, 1)), np.abs(yr - box(yr, 1))
                for k, m in (("bg", ~(face | hands | hair)), ("face", fi)):
                    if m.sum():
                        fe[k][0] += float(fo[m].sum())
                        fe[k][1] += float(fr_[m].sum())
                # tone columns and the no-stubble check
                (Lo, ao, bo), (Lr, ar, br) = to_lab(cur), to_lab(r)
                co_, cr_ = np.hypot(ao, bo), np.hypot(ar, br)
                fs = erode(face & ~hair, 2)
                for k, m in (("face", fs), ("hair", erode(hair, 2)), ("hand", erode(hands, 2)), ("rest", ~person)):
                    if m.sum() > 100:
                        if k != "rest":
                            tone.setdefault(k + "L", []).append(float(np.median(Lo[m]) / max(np.median(Lr[m]), 1.0)))
                        if k != "hand":
                            tone.setdefault(k + "C", []).append(float(co_[m].mean() / max(cr_[m].mean(), 1e-3)))
                yy = yo / 255.0
                tone.setdefault("blk", []).append(float(np.percentile(yy[::4, ::4], 1)))
                tone.setdefault("wht", []).append(float(np.percentile(yy[::4, ::4], 99)))
                if fs.sum() > 400:
                    tone.setdefault("hue", []).append(float(np.degrees(
                        np.arctan2(bo[fs].mean(), ao[fs].mean()) - np.arctan2(br[fs].mean(), ar[fs].mean()))))
                    ys, xs = np.nonzero(fs)
                    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                    fh, fw = y1 - y0, x1 - x0
                    Y, X = np.mgrid[0:fs.shape[0], 0:fs.shape[1]]
                    u, v = (X - x0) / max(fw, 1), (Y - y0) / max(fh, 1)
                    chin = fs & (v > 0.80) & (v < 0.97) & (u > 0.15) & (u < 0.85)
                    skin_r = (ar > 0.5 * np.median(ar[fs])) & (Lr > 0.4 * np.median(Lr[fs]))
                    neck = (v > 1.02) & (v < 1.22) & (u > 0.3) & (u < 0.7) & ~(hands | hair) & skin_r
                    beard = chin | neck
                    cheek = fs & (v > 0.42) & (v < 0.62) & (((u > 0.10) & (u < 0.35)) | ((u > 0.65) & (u < 0.90)))
                    if beard.sum() > 200 and cheek.sum() > 200:
                        tone.setdefault("beardL", []).append(float(
                            (np.median(Lo[beard]) / np.median(Lo[cheek])) / (np.median(Lr[beard]) / np.median(Lr[cheek]))))
                        tone.setdefault("beardF", []).append(float(
                            (fo[beard].mean() / fo[cheek].mean()) / (fr_[beard].mean() / fr_[cheek].mean())))
                if faces_out is not None and (f - f0) % 10 == 0 and face.sum() > 400:
                    ys, xs = np.nonzero(face)
                    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
                    s = max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.4 / 2
                    bb = (int(cx - s), int(cy - s), int(cx + s), int(cy + s))
                    crops_o.append(np.asarray(Image.fromarray(cur.astype(np.uint8)).crop(bb).resize((160, 160), Image.BICUBIC)))
                    crops_r.append(np.asarray(Image.fromarray(r.astype(np.uint8)).crop(bb).resize((160, 160), Image.BICUBIC)))
        prev = cur
    if faces_out is not None and crops_o:
        np.savez_compressed(faces_out, out=np.stack(crops_o), ref=np.stack(crops_r))
    fl = np.concatenate(flk) if flk else np.array([np.nan])
    nan = float("nan")
    extra = {k: float(np.mean(tone[k])) if tone.get(k) else nan for k in TONE_COLS}
    extra["textAf"] = 10 * np.log10(255 ** 2 / (aff_se / aff_n)) if aff_n and aff_se > 0 else (99.0 if aff_n else nan)
    return dict(**extra, flk=float(fl.mean()), p90=float(np.percentile(fl, 90)), follow=num / den if den else nan,
                flkF=float(np.mean(flkF)) if flkF else nan,
                textdB=10 * np.log10(255 ** 2 / (text_se / text_n)) if text_n and text_se > 0 else (99.0 if text_n else nan),
                lump=float(np.mean(lump)) if lump else nan, blotch=float(np.mean(blotch)) if blotch else nan,
                fineBG=fe["bg"][0] / fe["bg"][1] if fe["bg"][1] else nan,
                fineF=fe["face"][0] / fe["face"][1] if fe["face"][1] else nan)


TONE_COLS = ("faceL", "hairL", "handL", "faceC", "hairC", "restC", "hue", "blk", "wht", "beardL", "beardF")
COLS = ("flk", "p90", "follow", "flkF", "textdB", "textAf", "lump", "blotch", "fineBG", "fineF") + TONE_COLS


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    out_md = a.pop(0)
    opt = {}
    for k in ("--clips", "--step", "--faces"):
        if k in a:
            i = a.index(k)
            opt[k] = a[i + 1]
            del a[i:i + 2]
    step = int(opt.get("--step", 3))
    C = clips()
    tags = opt["--clips"].split(",") if "--clips" in opt else list(C)
    methods = []
    for m in a:
        lab, spec = m.split("=", 1)
        d, ref = (spec.split(":", 1) + ["v9d"])[:2]
        methods.append((lab, d, ref))
    if "--faces" in opt:
        os.makedirs(opt["--faces"], exist_ok=True)
    rows, per = [], {}
    for t in tags:
        c = C[t]
        if not os.path.isdir(c["geom"]) or not os.path.isdir(c["masks"]):
            print(f"{t}: no geometry or masks yet, skipped", flush=True)
            continue
        geo = geometry(c, *c["frames"])
        for lab, d, ref in methods:
            dd, rr = resolve(c, d), resolve(c, ref)
            if not os.path.isdir(dd) or not os.path.isdir(rr):
                continue
            t0 = time.time()
            fo = os.path.join(opt["--faces"], f"{lab}_{t}.npz") if "--faces" in opt else None
            s = score(c, dd, rr, geo, step, fo)
            if s is None:
                print(f"{t} {lab}: frames missing, skipped", flush=True)
                continue
            per.setdefault(lab, []).append(s)
            rows.append(f"| {t} | {lab} | " + " | ".join(f"{s[k]:.2f}" for k in COLS) + " |")
            print(rows[-1], f"({time.time() - t0:.0f}s)", flush=True)
    head = "| clip | method | " + " | ".join(COLS) + " |\n|" + "---|" * (len(COLS) + 2)
    mean = [f"| mean ({len(v)}) | {lab} | " + " | ".join(f"{np.nanmean([s[k] for s in v]):.2f}" for k in COLS) + " |"
            for lab, v in per.items()]
    txt = (f"\n## nr_board {time.strftime('%Y-%m-%d %H:%M')} (step {step})\n\n{head}\n" + "\n".join(rows) +
           f"\n\n{head}\n" + "\n".join(mean) + "\n")
    with open(out_md, "a") as fp:
        fp.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
