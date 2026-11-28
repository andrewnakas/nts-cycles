"""NTS-G: 2x frame generation for Cycles sequences (2026-11-26).

An offline renderer knows, for every pixel, which bit of which surface it shows. The frame n between two finished
frames A (n-1) and B (n+1) is made from that knowledge in one of two ways.

Surface mode (--surf; preferred). The in-between frame's own surf pass is rendered (datagen/blender/render_lr.py mode
"surf": each pixel's surface coordinates and object number, 8 samples, no lights or bounces, a small share of a full
frame's cost). Each of its pixels looks its own surface point up in A's and in B's surf pass (nts/surface.py) and
reads the colour there:
  - both show it: the two colours are mixed by time;
  - one shows it: that one; with --geom, the mean of that colour and the vector-mode frame's (one witness, often
    lit differently next to a moving outline);
  - neither shows it, or the pixel saw more than one surface (an outline, a strand): it is taken from the vector-mode
    frame (below) when the finished frames' geometry is given (--geom), else filled in from around it (push-pull,
    which blurs strands and outlines);
  - no surface at all (the world): the two frames' colours at the same place, mixed by time.
The outlines are the in-between frame's own, so a thin or fast object is where it really is at frame n; the motion
vectors only fill the pixels the surface lookup leaves.

Vector mode (no surf pass for the in-between frame). Cycles' Vector pass gives each pixel of A its motion to frame n
(A's Vector.ZW) and each pixel of B its motion from frame n (B's Vector.XY). Each pixel's COLOUR is carried along its
motion and shared among the four pixels around where it lands, by bilinear weight: forward image warping with a
depth order, as in Mark, McMillan and Bishop,
"Post-rendering 3D warping" (1997), and Gortler et al., "The Lumigraph" (1996) for the fill. At each pixel the nearest
surface comes first and anything behind it fills only what it leaves uncovered, so outlines stay anti-aliased. A and B
are then combined the same way (one surface: mixed by time; different surfaces: nearer first), and what neither covers
is filled by push-pull. The results are never motion vectors for frame n, only colour and coverage.

What neither mode can know: lighting that changes between the frames (it is averaged) and surface that neither
neighbour shows (it is filled).

Use, either way:
  - every frame rendered: --check generates each frame f from f-1 and f+1 and compares it with the real f (the
    synthetic tests are in tests/test_framegen_modes.py).
  - half the frames rendered: render the full frames with Blender's frame step 2, and the surf pass (or, for vector
    mode, nothing more: the Vector pass of n-1 and n+1 already points at the SCENE frames n-1 +- 1) for every frame.

Run: python -m nts.framegen <exr_dir> <frames_dir> [out_dir] [--surf [--geom <dir>]] [--frames a-b] [--check]
                           [--workers n]
  exr_dir        f_####.exr: surf passes of every frame (--surf), or Depth.V and Vector.XYZW (render_lr.py, or slim
                 geometry files) of the finished frames
  frames_dir     finished frames f_####.png (any stage: native render, toned, NR-1 output)
  --geom         with --surf: Depth.V and Vector.XYZW of the finished frames, for the vector-mode frame under
                 the pixels the surface lookup leaves
  --check        every frame f with both neighbours finished is generated from f-1 and f+1 and compared with the real
                 f (PSNR over the frame, and against the plain average of the neighbours); nothing is written unless
                 out_dir is given
  otherwise      every frame number missing from frames_dir whose two neighbours are there is generated into out_dir
Needs numpy, Pillow, OpenImageIO (nts.io_exr).
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts import warp as W  # noqa: E402

DEPTH_REL = 0.05   # two surfaces are the same within this share of their depth (as nts.warp.disocclusion)
SMOOTH = 2         # px: window (2r+1) over which the share of each frame is smoothed where both are used


def splat(img, depth, mv):
    """Carry img's colours to the in-between frame: each pixel moves along its own motion mv (H, W, 2), lands
    between four pixels and gives each its bilinear share. A pixel's coverage by a surface is the sum of the shares it
    got from it (capped at 1); per pixel the nearest surface that landed comes first, and surfaces behind it fill only
    the part it leaves uncovered. (Weighting the shares by each pixel's stretch, from the slopes of its motion, was
    tried and lost 2.4 dB on at140: the slopes of real motion fields are too noisy.)
    Returns (colour (H, W, c) premultiplied by coverage, coverage (H, W) 0..1, depth (H, W), inf where nothing landed)."""
    H, Wd = depth.shape
    ys, xs = W.grid(H, Wd)
    px, py = xs + mv[..., 0], ys + mv[..., 1]
    ok = np.isfinite(px) & np.isfinite(py) & np.isfinite(depth) & (px > -1) & (px < Wd) & (py > -1) & (py < H)
    px, py, d = px[ok].astype(np.float64), py[ok].astype(np.float64), depth[ok].astype(np.float64)
    col = img[ok].astype(np.float64)
    x0, y0 = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
    fx, fy = px - x0, py - y0
    idx, wt, src = [], [], []
    for oy, wy in ((0, 1 - fy), (1, fy)):
        for ox, wx in ((0, 1 - fx), (1, fx)):
            tx, ty, w = x0 + ox, y0 + oy, wx * wy
            k = (tx >= 0) & (tx < Wd) & (ty >= 0) & (ty < H) & (w > 1e-6)
            idx.append(ty[k] * Wd + tx[k])
            wt.append(w[k])
            src.append(np.flatnonzero(k))
    idx, wt, src = np.concatenate(idx), np.concatenate(wt), np.concatenate(src)
    dd = d[src]
    N = H * Wd
    z = np.full(N, np.inf)
    np.minimum.at(z, idx, dd)
    front = dd <= z[idx] * (1 + DEPTH_REL)
    out = np.zeros((N, col.shape[1]))
    alpha = np.zeros(N)
    room = np.ones(N)
    for part in (front, ~front):
        i, w = idx[part], wt[part]
        sw = np.bincount(i, w, N)
        a = np.minimum(sw, room)  # the part of each pixel this layer covers
        has = sw > 0
        for ch in range(col.shape[1]):
            m = np.bincount(i, w * col[src[part], ch], N)
            out[has, ch] += a[has] * m[has] / sw[has]
        alpha += a
        room -= a
    return (out.reshape(H, Wd, -1).astype(np.float32), alpha.reshape(H, Wd).astype(np.float32),
            z.reshape(H, Wd).astype(np.float32))


def _down(x):
    h, w = x.shape[:2]
    if h % 2:
        x = np.concatenate([x, x[-1:]], 0)
    if w % 2:
        x = np.concatenate([x, x[:, -1:]], 1)
    return x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2]


def fill(p, a):
    """Push-pull (Gortler et al. 1996): colour for every pixel from premultiplied colour p (H, W, c) and coverage a
    (H, W) 0..1. Coverage is summed into coarser levels (capped at 1) and each level fills what its finer one leaves
    uncovered. Returns the filled colour (not premultiplied)."""
    pyr = [(p.astype(np.float64), a.astype(np.float64))]
    while min(pyr[-1][1].shape) > 1 and (pyr[-1][1] < 1).any():
        q, b = _down(pyr[-1][0]), _down(pyr[-1][1])
        k = 1 / np.maximum(b, 1)
        pyr.append((q * k[..., None], b * k))
    q, b = pyr[-1]
    for fp, fa in pyr[-2::-1]:
        h, w = fa.shape
        uq = np.repeat(np.repeat(q, 2, 0), 2, 1)[:h, :w]
        ub = np.repeat(np.repeat(b, 2, 0), 2, 1)[:h, :w]
        rest = 1 - fa
        q, b = fp + rest[..., None] * uq, fa + rest * ub
    return (q / np.maximum(b, 1e-9)[..., None]).astype(np.float32)


def _finish(ia, ib, wa, wb, under=None):
    """The two frames' colours ia, ib with per-pixel weights wa, wb (0 where a frame has nothing): the share of A is
    smoothed where both are used (no pixel-by-pixel flips between them at soft outlines), and coverage below 1 is
    filled from `under` (an image of the same frame made another way) or, without one, by push-pull.
    Returns (frame, coverage)."""
    from nts.tone import box
    tot = wa + wb
    share = np.where(tot > 0, wa / np.maximum(tot, 1e-9), 0.5).astype(np.float32)
    both = ((wa > 0) & (wb > 0)).astype(np.float32)
    sm = box(share * both, SMOOTH) / np.maximum(box(both, SMOOTH), 1e-6)
    share = np.where(both > 0, sm, share)[..., None]
    alpha = np.clip(tot, 0, 1)
    col = ia * share + ib * (1 - share)
    out = col * alpha[..., None]
    if (alpha < 1).any():
        out = out + (1 - alpha)[..., None] * (fill(out, alpha) if under is None else under)
    return out.astype(np.float32), alpha.astype(np.float32)


def between(a_img, a_depth, a_mv_next, b_img, b_depth, b_mv_prev, t=0.5):
    """Vector mode: the frame between A and B. a_img / b_img (H, W, c) float; *_depth (H, W) at their own frames;
    a_mv_next: A's image-space motion to the in-between frame, b_mv_prev: B's (both full-resolution,
    nts.conventions signs). Returns (frame, coverage 0..1)."""
    pa, ca, za = splat(a_img, a_depth, a_mv_next)
    pb, cb, zb = splat(b_img, b_depth, b_mv_prev)
    ia, ib = pa / np.maximum(ca, 1e-9)[..., None], pb / np.maximum(cb, 1e-9)[..., None]
    with np.errstate(invalid="ignore"):
        same = np.abs(za - zb) <= DEPTH_REL * np.minimum(za, zb)
    a_first, b_first = ~same & (za < zb), ~same & (zb < za)
    # one surface: mixed by time where both landed, to the union of their coverage; different surfaces: the nearer
    # first, the other fills what it leaves
    tw = (1 - t) * ca + t * cb
    u = np.maximum(ca, cb)
    wa = np.where(same, u * (1 - t) * ca / np.maximum(tw, 1e-9), np.where(a_first, ca, np.minimum(ca, 1 - cb)))
    wb = np.where(same, u * t * cb / np.maximum(tw, 1e-9), np.where(b_first, cb, np.minimum(cb, 1 - ca)))
    return _finish(ia, ib, wa.astype(np.float32), wb.astype(np.float32))


def _taps_are(s, m, ids):
    """1 where the four pixels of s around m (h, w, 2) all show object ids (h, w), else 0."""
    h, w = s.shape
    x0 = np.clip(np.floor(m[..., 0]).astype(np.int64), 0, w - 2)
    y0 = np.clip(np.floor(m[..., 1]).astype(np.int64), 0, h - 2)
    ok = np.ones(ids.shape, bool)
    for dy in (0, 1):
        for dx in (0, 1):
            ok &= s.ids[y0 + dy, x0 + dx] == ids
    return ok.astype(np.float32)


def between_surf(a_img, sa, b_img, sb, sn, t=0.5, under=None):
    """Surface mode: the frame between A and B. a_img / b_img (H, W, c) float; sa, sb, sn: nts.surface.Surf of A, B
    and the in-between frame. Pixels whose surface point neither A nor B shows, or that see more than one surface (an
    outline, a hair strand: the surf pass marks them mixed), are taken from `under` when given: the vector-mode frame
    (between()), which carries such pixels whole; without it they are filled in from around them, which blurs strands
    and outlines (on at140 these 3.5 % of the pixels held 87 % of the error). A pixel whose point only ONE neighbour
    shows (about 1 % of a frame, next to a moving outline) has a single witness, often lit differently there (the
    occluder's contact shadow has just moved off it): with `under` it takes the mean of that neighbour's colour and
    the vector-mode colour (at140n, 58 frames: 44.19 dB; 43.55 dB taking the one colour, 43.61 dB vector mode).
    Returns (frame, coverage 0..1)."""
    from nts import surface as S
    ma, va = S.match(S.Key(sa), sn)
    mb, vb = S.match(S.Key(sb), sn)
    # the colour is read bilinearly: next to an outline its four pixels must all show the object, or the other
    # side's anti-aliased colour is read in (nts.surface also matches points there, for the carry's coordinates)
    va, vb = va * _taps_are(sa, ma, sn.ids), vb * _taps_are(sb, mb, sn.ids)
    ia = W.bilinear(a_img, ma[..., 0], ma[..., 1])
    ib = W.bilinear(b_img, mb[..., 0], mb[..., 1])
    wa, wb = (1 - t) * va, t * vb
    tot = np.maximum(wa + wb, 1e-9)
    wa, wb = wa / tot * (va + vb > 0), wb / tot * (va + vb > 0)  # by time where both show it, else the one that does
    # the world (no surface in any of the three frames): the same place in both frames
    world = sn.world & sa.world & sb.world & (va + vb <= 0)
    wa, wb = np.where(world, 1 - t, wa), np.where(world, t, wb)
    ia = np.where(world[..., None], a_img, ia)
    ib = np.where(world[..., None], b_img, ib)
    out, alpha = _finish(ia, ib, wa.astype(np.float32), wb.astype(np.float32), under)
    if under is not None:
        one = ((va > 0) != (vb > 0))[..., None]
        out = np.where(one, 0.5 * (out + under), out).astype(np.float32)
    return out, alpha


def _frames(d):
    out = {}
    for p in glob.glob(os.path.join(d, "f_*.png")) + glob.glob(os.path.join(d, "f_*.jpg")):
        b = os.path.basename(p)
        if b[2:6].isdigit():
            out[int(b[2:6])] = p
    return out


def _load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32)


_SEQ = {}


def _score(img, frames, f, a, b):
    ref = _load(frames[f])
    mse = lambda x: float(np.mean((x - ref) ** 2))  # noqa: E731
    psnr = lambda e: 10 * np.log10(255.0 ** 2 / max(e, 1e-9))  # noqa: E731
    return psnr(mse(img.astype(np.float32))), psnr(mse(np.clip(0.5 * (a + b) + 0.5, 0, 255).astype(np.uint8).astype(np.float32)))


def _vector(geom, f, a, b):
    """Vector mode for frame f from the full renders' geometry (Depth, Vector) of f-1 and f+1."""
    from nts.io_exr import Sequence
    if geom not in _SEQ:
        _SEQ[geom] = Sequence(geom)
    seq = _SEQ[geom]
    fa, fb = seq.read(f - 1), seq.read(f + 1)
    if fa.mv_next is None:
        raise SystemExit(f"framegen: {seq.files[f - 1]} has no Vector.ZW (next-frame motion)")
    hw = a.shape[:2]
    # each pixel moves with its own motion (no nearest-depth dilation: that is for backward lookups)
    up = lambda m: W.upsample_nearest(m, seq.scale, hw) * seq.scale  # noqa: E731
    da, db = W.upsample_nearest(fa.depth, seq.scale, hw), W.upsample_nearest(fb.depth, seq.scale, hw)
    return between(a, da, up(fa.mv_next), b, db, up(fb.mv))[0]


def _gen(task):
    """Generate frame f from f-1 and f+1: (f, (psnr, psnr of the plain average) or None)."""
    src, frames, f, check, out_dir, surf, geom = task
    a, b = _load(frames[f - 1]), _load(frames[f + 1])
    if surf:
        from nts import surface as S
        rd = lambda q: S.read(os.path.join(src, f"f_{q:04d}.exr"))  # noqa: E731
        out, _ = between_surf(a, rd(f - 1), b, rd(f + 1), rd(f), under=_vector(geom, f, a, b) if geom else None)
    else:
        out = _vector(src, f, a, b)
    img = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    if out_dir:
        Image.fromarray(img).save(os.path.join(out_dir, f"f_{f:04d}.png"))
    return f, (_score(img, frames, f, a, b) if check else None)


def main():
    a = sys.argv[1:]
    if len(a) < 2 or a[0] in ("-h", "--help"):
        sys.exit(__doc__)
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--frames", "--workers", "--geom"))]
    src, fdir = os.path.abspath(pos[0]), pos[1]
    out_dir = pos[2] if len(pos) > 2 else None
    check, surf = "--check" in a, "--surf" in a
    geom = os.path.abspath(opt("--geom")) if surf and opt("--geom") else None
    span = tuple(map(int, opt("--frames", "0-999999").split("-")))
    from nts.par import pmap
    have = {int(os.path.basename(p)[2:6]) for p in glob.glob(os.path.join(src, "f_*.exr"))}
    cp = os.path.join(src, "cuts.json")
    cuts = set(json.load(open(cp))) if os.path.exists(cp) else set()
    frames = _frames(fdir)
    # a cut at f or at f+1 puts f-1 and f+1 in different shots; the surf mode needs frame f's own pass as well
    need = (lambda f: {f - 1, f, f + 1}) if surf else (lambda f: {f - 1, f + 1})
    if geom:  # the full renders' geometry of f-1 and f+1, for the vector-mode frame under the surface mode
        hg = {int(os.path.basename(p)[2:6]) for p in glob.glob(os.path.join(geom, "f_*.exr"))}
        need0 = need
        need = lambda f: need0(f) if {f - 1, f + 1} <= hg else {-1}  # noqa: E731
    ok = lambda f: f - 1 in frames and f + 1 in frames and need(f) <= have and f not in cuts and f + 1 not in cuts  # noqa: E731
    if check:
        todo = [f for f in sorted(frames) if ok(f)]
    else:
        todo = [f for f in range(min(frames, default=0), max(frames, default=-1)) if f not in frames and ok(f)]
    todo = [f for f in todo if span[0] <= f <= span[1]]
    if not todo:
        sys.exit("framegen: nothing to generate (need frames f-1 and f+1, their geometry (and f's surf pass with --surf), "
                 "no cut between them)")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    res = pmap(_gen, [(src, frames, f, check, out_dir, surf, geom) for f in todo], opt("--workers"))
    if check:
        g = [s[0] for _, s in res]
        b = [s[1] for _, s in res]
        for f, (p, q) in res:
            print(f"frame {f}: generated {p:.2f} dB, plain average of the neighbours {q:.2f} dB")
        print(f"framegen check: {len(res)} frames, mean PSNR against the real frame {np.mean(g):.2f} dB "
              f"(median {np.median(g):.2f}, worst {np.min(g):.2f}; plain average {np.mean(b):.2f} dB)")
    else:
        print(f"framegen: {len(res)} frames -> {out_dir}")


if __name__ == "__main__":
    main()
