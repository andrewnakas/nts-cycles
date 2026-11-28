"""Two-band temporal steadying of a generative pass (NR-1 phase 2, 2026-11-18; the design log row 12).

gen/steady.py (--alpha 0.2 --base 0.3) lowered flicker on every eval clip but the board showed why it lost
on look: it averages the whole finished frame over time with bilinear warps, and the per-frame invented texture is
uncorrelated between frames, so averaging cancels it. e2 at140 fine-detail ratio (output vs input render): raw 1.21
background / 1.09 face, steadied 0.96 / 0.67. Here the frame is split in two bands:

  coarse   C = blur(.) (3 box passes, sigma ~1.4 px): shading, colour, render noise. Steadied exactly as steady.py:
           the model's coarse change accumulated along Blender's motion where the render shows the same surface
           (--alpha), then the finished coarse frame accumulated TAA-style with a 3x3 min/max clamp (--base, --slack).
  fine     F = y - blur(y): texture. HELD, not averaged: the history's fine band is carried along the motion with a
           Catmull-Rom (bicubic) warp, which keeps 1-2 px detail where bilinear resampling would soften it, and
           refreshed slowly from the current frame (--fine-alpha). Uncorrelated blending loses energy, so the held band
           is rescaled to the current frame's local fine energy (smooth ~15 px window, gain in [1/--gain, --gain]):
           the pattern stays on the surface, the amount of detail stays that of the current frame (histogram/variance
           preserving blending: Heitz & Neyret 2018, paper). Cuts and newly visible pixels take the current frame.
  out = steadied coarse + held fine.

Run: python -m gen.steady2 <geom_exr_dir> <render_dir> <gen_dir> <out_dir> [--alpha 0.2] [--tol 6] [--base 0.3]
         [--slack 4] [--fine-alpha 0.15] [--gain 1.6] [--bands 1|2] [--energy-alpha 1] [--fine-tol 6] [--frames a-b] [--reverse]
         [--key-stride 1] [--edge-reset 0]
--edge-reset r: within r px of a moving silhouette (a depth step whose sides move differently) nothing is held:
coarse history and fine band both come from the current frame there (see moving_edges() and the note in main()).
--energy-alpha a < 1 accumulates the target texture energy along the motion as well (a = share of the current frame),
so the held texture's strength does not pulse with the teacher's per-frame changes (default 1: current frame only).
--fine-tol t (default: --tol) is the render-colour tolerance for holding the fine band. The render's own noise fails
the coarse tolerance on faces (render face flicker ~2.2), which refreshes the fine band from the uncorrelated per-frame
texture far more often than --fine-alpha says; texture belongs to the surface, so a looser tolerance holds it.
--reverse runs backwards in time along Blender's next-frame motion (Vector.ZW, nts/conventions.mv_next_image); with
gen/steady_bi.py the forward and backward results make the non-causal NR-1 targets (the design log row 11).
--key-stride k (NR-1 v2, the design log row 24): the teacher is only needed on keyframes (every k-th frame
number, plus the first and last frame of each shot; `keyframes()` below, written to <out_dir>/keys.json). Between
keyframes nothing new is observed: the coarse change and the held texture are carried along the motion at full weight
where the render shows the same surface, and fall back to the render itself where it does not (newly visible
pixels); the texture's target energy is carried the same way. Run forward and --reverse and blend with
gen/steady_bi.py, which weights each direction by its distance to the keyframe it came from.
--bands 2 matches the energy of the top sub-band (y - box r1, what eval/nr_board.py's fine ratios measure) and the
rest of the fine band separately, so the bicubic warp's loss of the finest detail is restored too.
--threads n (default 3, env NTS_THREADS): the next frame is read and filtered, and last frame's state warped, in n
threads while this frame is worked on (numpy releases the GIL in its loops); the output is byte-identical to --threads 1.
Same inputs and outputs as gen/steady.py. Needs numpy, Pillow, OpenImageIO (~/scene-project/py on the 4050).
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts import warp as W  # noqa: E402
from nts.io_exr import Sequence  # noqa: E402


def frame(d, f):
    p = glob.glob(os.path.join(d, f"f_{f:04d}.*"))
    p = [q for q in p if q[-4:] in (".png", ".jpg")]
    return np.asarray(Image.open(p[0]).convert("RGB")).astype(np.float32) if p else None


def box(a, r):
    """Box mean over a (2r+1) square, edges repeated: running sums along each axis (slices of one cumsum, which is
    bit-identical to the np.take form this replaced and avoids two copies per axis)."""
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax)
        n = a.shape[ax]
        hi = [slice(None)] * a.ndim
        lo = [slice(None)] * a.ndim
        hi[ax], lo[ax] = slice(2 * r + 1, 2 * r + 1 + n), slice(0, n)
        a = c[tuple(hi)] - c[tuple(lo)]
        if a.dtype.kind == "f":
            a /= (2 * r + 1)
        else:
            a = a / (2 * r + 1)
    return a


def moving_edges(depth, mv, r, rel=0.05, slip=(1.0, 3.0)):
    """Weight (0..1) that is 1 within r px of a depth step whose two sides move differently by slip[1] px a frame or
    more, 0 where they differ by under slip[0] px, feathered over r/2 px: a silhouette in fast relative motion.
    There the motion and depth passes are exact but the picture is not: defocus and anti-aliasing spread the moving
    object's colour a few pixels past its outline, so pixels that are 'the same wall' by depth and motion change
    with it. A slowly moving outline (a talking head) is left to the history clamp, which handles a sub-pixel shift."""
    e = np.zeros(depth.shape, np.float32)
    for ax in (0, 1):
        n = depth.shape[ax]
        a0, a1 = np.take(depth, np.arange(0, n - 1), axis=ax), np.take(depth, np.arange(1, n), axis=ax)
        m0, m1 = np.take(mv, np.arange(0, n - 1), axis=ax), np.take(mv, np.arange(1, n), axis=ax)
        s = np.clip((np.abs(m1 - m0).max(-1) - slip[0]) / (slip[1] - slip[0]), 0, 1)
        s = s * (np.abs(a1 - a0) > rel * np.minimum(a0, a1))
        pad = [(0, 0), (0, 0)]
        pad[ax] = (0, 1)
        s = np.pad(s, pad)
        e = np.maximum(e, np.maximum(s, np.roll(s, 1, axis=ax)))
    w = np.clip(box(e, r) * (2 * r + 1), 0, 1)  # a line of weight 1 through the window -> 1 anywhere within r px of it
    return box(w, max(1, r // 2))


def lowpass(a):
    return box(box(box(a, 1), 1), 1)


def win(a):
    """local mean over a smooth ~15 px window (3 box passes, sigma ~4.5 px) for the texture-energy match. A single
    hard-edged 15 px box was used until 2026-11-20: every strong pixel then cast a square plateau in the energy map,
    the gain stepped at its edges, and over a clip that feedback grew a regular lattice of embossed dots on hair and
    skin (the design log row 37)."""
    return box(box(box(a, 4), 4), 4)


def minmax3(im):
    p = np.pad(im, ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = im.shape[:2]
    v = [p[i:i + h, j:j + w] for i in range(3) for j in range(3)]
    return np.minimum.reduce(v), np.maximum.reduce(v)


def cr_weights(t):
    """Catmull-Rom weights for taps at -1, 0, 1, 2 given the fractional offset t."""
    t2, t3 = t * t, t * t * t
    return (-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1, -1.5 * t3 + 2 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2)


def warp_cr(img, mv):
    """Backward warp with Catmull-Rom (bicubic) sampling, edges clamped. img (H, W, c), mv (H, W, 2) current->previous.
    Gathers are flat np.take on precomputed row and column indices; the sums are the same as before, in the same
    order and dtypes, so the result is bit-identical."""
    H, Wd = mv.shape[:2]
    ys, xs = W.grid(H, Wd)
    x, y = np.clip(xs + mv[..., 0], 0, Wd - 1), np.clip(ys + mv[..., 1], 0, H - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    wx, wy = cr_weights((x - x0)[..., None]), cr_weights((y - y0)[..., None])
    flat = img.reshape((H * Wd,) + img.shape[2:])
    cols = [np.clip(x0 + i - 1, 0, Wd - 1) for i in range(4)]
    out = np.zeros_like(img)
    for j in range(4):
        yy = np.clip(y0 + j - 1, 0, H - 1) * Wd
        row = np.zeros_like(img)
        for i in range(4):
            row += flat.take(yy + cols[i], axis=0) * wx[i]
        out += row * wy[j]
    return out


def split(f):
    """fine band -> (top: f - box r1, rest)."""
    top = f - box(f, 1)
    return top, f - top


def keyframes(frames, reset, k):
    """Frames the teacher must run on for --key-stride k: every k-th frame number, plus both ends of every shot."""
    frames = sorted(frames)
    if k <= 1:
        return set(frames)
    keys = {f for f in frames if f % k == 0}
    for i, f in enumerate(frames):
        first = i == 0 or reset.get(f) or frames[i - 1] != f - 1
        last = i == len(frames) - 1 or reset.get(frames[i + 1]) or frames[i + 1] != f + 1
        if first or last:
            keys.add(f)
    return keys


def main():
    a = sys.argv
    if len(a) < 5:
        sys.exit(__doc__)
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    alpha, tol = float(opt("--alpha", 0.2)), float(opt("--tol", 6))
    base, slack = float(opt("--base", 0.3)), float(opt("--slack", 4))
    fa, gmax = float(opt("--fine-alpha", 0.15)), float(opt("--gain", 1.6))
    nb = int(opt("--bands", 1))
    ftol = float(opt("--fine-tol", tol))  # colour tolerance for holding the fine band (default: same as the coarse)

    ea = float(opt("--energy-alpha", 1.0))
    er = int(opt("--edge-reset", 0))  # px round moving silhouettes where the fine band is taken from the current frame
    E = {}  # per sub-band target energy, accumulated along the motion when --energy-alpha < 1
    threads = int(opt("--threads", os.environ.get("NTS_THREADS", 3)))  # 1 = no overlap; the output is the same

    def gain(k, ec, eh, Ew=None, same=None, carry=False):
        """per-pixel gain that gives the held band h the local (~15 px, smooth window) energy of the current band cur,
        within [1/--gain, --gain]; ec = win(|cur|^2), eh = win(|h|^2), Ew = last frame's E[k] warped along the motion.
        With --energy-alpha < 1 that target energy is itself accumulated along the motion, so the amount of texture
        does not pulse with the teacher's frame-to-frame changes. carry (between keyframes): the target energy is the
        carried one at full weight where the surface is the same, and cur's (the render's own fine band) elsewhere."""
        if (ea < 1 or carry) and k in E and Ew is not None:
            we = same if carry else (1 - ea) * same
            ec = ec * (1 - we) + Ew[..., 0] * we
        E[k] = ec
        return np.clip(np.sqrt((ec + 1e-3) / (eh + 1e-3)), 1 / gmax, gmax)[..., None]
    seq = Sequence(a[1])
    span = tuple(map(int, opt("--frames", "0-999999").split("-")))
    rev = "--reverse" in a  # run backwards in time: history comes from frame f+1 along Blender's next-frame motion
    order = seq.frames[::-1] if rev else seq.frames
    if rev:  # going f+1 -> f crosses a cut when f+1 starts a shot (or f+1 is missing)
        fs = set(seq.frames)
        rreset = {f: (f + 1 not in fs) or seq.reset[f + 1] for f in seq.frames}
    os.makedirs(a[4], exist_ok=True)
    ks = int(opt("--key-stride", 1))
    todo = [f for f in seq.frames if span[0] <= f <= span[1]]
    keys = keyframes(todo, seq.reset, ks)
    if ks > 1:
        import json
        json.dump(sorted(keys), open(os.path.join(a[4], "keys.json"), "w"))
    # the base step passes the render as its own teacher: read and filter it once
    same_src = os.path.abspath(a[2]) == os.path.abspath(a[3])

    def prep(f):
        """Everything of frame f that does not depend on the history: read, split into bands, the motion's lookup,
        the moving silhouettes and the current bands' energy. Runs a frame ahead in a thread (--threads)."""
        x = frame(a[2], f)
        y = (x if same_src else frame(a[3], f)) if f in keys else None
        if x is None:
            return dict(x=None, y=y)
        fr = seq.read(f)
        if rev:
            fr.mv, fr.reset = fr.mv_next, rreset[f]
        hw = x.shape[:2]
        depth = W.upsample_nearest(fr.depth, seq.scale, hw)
        xc = lowpass(x)
        if y is None:  # between keyframes: nothing new from the teacher, the render's own fine band is the fallback
            fine, d = x - xc, None
        else:
            yc = xc if y is x else lowpass(y)
            fine, d = y - yc, yc - xc
        bands = split(fine) if nb == 2 else [fine]
        mv = W.hr_motion(fr.mv, fr.depth, seq.scale, hw)
        return dict(x=x, y=y, fr=fr, depth=depth, xc=xc, fine=fine, d=d, bands=bands, mv=mv, wp=W.Warper(mv),
                    ec=[win((fc * fc).sum(-1)) for fc in bands], edge=moving_edges(depth, mv, er) if er > 0 else None)

    import concurrent.futures as cf
    pool = cf.ThreadPoolExecutor(max(1, threads)) if threads > 1 else None
    run = (lambda fn, *q: pool.submit(fn, *q)) if pool else (lambda fn, *q: _Done(fn(*q)))
    live = [f for f in order if span[0] <= f <= span[1]]
    ahead = {}
    saving = []

    acc = hist = held = prev_xc = prev_depth = prev_edge = None
    n, e_in, e_out = 0, 0.0, 0.0
    for i, f in enumerate(live):
        if pool and i + 1 < len(live):
            ahead[i + 1] = run(prep, live[i + 1])  # the next frame's reads and filters, while this one is worked on
        P = (ahead.pop(i) if i in ahead else run(prep, f)).result()
        x, y = P["x"], P["y"]
        cut = rreset[f] if rev else seq.reset[f]
        if x is None or (y is None and (f in keys or acc is None or cut)):  # a keyframe's teacher output is missing
            acc = None
            continue
        fr, depth, xc, fine, d = P["fr"], P["depth"], P["xc"], P["fine"], P["d"]
        if acc is None or fr.reset:
            acc, hist, held, mv, occ, prev_edge = d, None, fine, None, None, None
            E.clear()
            for k, ec in enumerate(P["ec"]):
                E[k] = ec
        else:
            wp = P["wp"]  # every buffer below moves with the same motion: one lookup for all of them
            mv = P["mv"]
            # last frame's state, all warped along this frame's motion at once
            jobs = {"xc": run(wp, prev_xc), "acc": run(wp, acc), "held": run(warp_cr, held, mv)}
            if er > 0 and prev_edge is not None:
                jobs["edge"] = run(wp, prev_edge[..., None])
            if base < 1 and hist is not None:
                jobs["hist"] = run(wp, hist)
            if ea < 1 or d is None:  # gain() blends in the warped target energy only then
                for k in list(E):
                    jobs["E%d" % k] = run(wp, E[k][..., None])
            occ = wp.disocclusion(prev_depth, depth)
            dc = np.abs(jobs["xc"].result() - xc).mean(-1)
            same, same_f = np.exp(-dc / tol), np.exp(-dc / ftol)
            if d is None:  # hold at full weight while the render agrees (no observation to refresh from)
                same, same_f = np.clip(2 - dc / tol, 0, 1), np.clip(2 - dc / ftol, 0, 1)
            same[occ] = same_f[occ] = 0
            if er > 0:
                # nothing is held next to a moving silhouette (at140 166-174, found 2026-11-21): the motion field
                # is exact but steps at the geometric outline, while defocus and anti-aliasing spread the object's
                # colour a few px past it. History warped across that step, then clamped only to the local 3x3
                # range, left a jagged bright fringe round a defocused moving hand, and the seam it cut was carried
                # on along the wall and the hand as trails of thin contour lines.
                # The frame after, the wall the outline has left is 'the same wall' again and far from any outline,
                # but its history is the frame that still had the object's blurred edge on it: the held fine band
                # kept that edge as a thin line, one per frame along the object's path (at140 185-195, a net of
                # contour lines above the hand; found 2026-11-21 on the first build with whole-frame detail). So
                # history taken from a place that was next to a moving outline is not held either.
                edge = P["edge"]
                keep = 1 - edge
                if prev_edge is not None:
                    keep = keep * (1 - np.clip(jobs["edge"].result()[..., 0], 0, 1))
                prev_edge = edge
                same, same_f = same * keep, same_f * keep
            if d is None:
                acc = jobs["acc"].result() * same[..., None]
                wf = same_f[..., None]
            else:
                w = ((1 - alpha) * same)[..., None]
                acc = d * (1 - w) + jobs["acc"].result() * w
                # fine band: carry the held texture along the motion, refresh it slowly, keep the current detail
                wf = ((1 - fa) * same_f)[..., None]
            h = jobs["held"].result() * wf + fine * (1 - wf)
            cy = d is None
            hb = split(h) if nb == 2 else [h]  # match the top (1 px) sub-band separately: warping and blending lose it fastest
            eh = [run(lambda q: win((q * q).sum(-1)), hc) for hc in hb]
            g = [gain(k, P["ec"][k], eh[k].result(), jobs["E%d" % k].result() if "E%d" % k in jobs else None, same_f, cy)
                 for k in range(len(hb))]
            if nb == 2:
                held = sum(gk * hc for gk, hc in zip(g, hb))
            else:
                held = g[0] * h
        out_c = xc + acc
        if base < 1:
            if hist is not None:
                lo, hi = minmax3(out_c)
                hc = np.clip(jobs["hist"].result(), lo - slack, hi + slack)
                wb = ((1 - base) * ~occ * (keep if er > 0 else 1.0))[..., None]
                out_c = out_c * (1 - wb) + hc * wb
            hist = out_c
        out = out_c + held
        im = Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
        path = os.path.join(a[4], f"f_{f:04d}.png")
        saving.append(run(im.save, path))
        prev_xc, prev_depth = xc, depth
        e_in += float(np.abs(fine).mean())
        e_out += float(np.abs(held).mean())
        n += 1
        P = jobs = None  # let this frame's buffers go before the next one is read
    for q in saving:
        q.result()
    if pool:
        pool.shutdown()
    print(f"steady2: {n} frames -> {a[4]}; alpha {alpha} base {base} fine-alpha {fa} bands {nb} energy-alpha {ea} fine-tol {ftol} "
          f"key-stride {ks} ({len(keys)} teacher frames); fine kept {e_out / max(e_in, 1e-9):.2f}")


class _Done:
    """A finished result with the Future interface (sequential runs)."""
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def result(self):
        return self.v


if __name__ == "__main__":
    main()
