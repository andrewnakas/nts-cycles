"""Carry a keyframe's added detail through a shot by surface identity (NR-1 v2, 2026-11-20; surface identity
2026-11-26; the design log rows 40 and 126).

gen/steady2.py holds texture by warping last frame's held band again every frame. Each warp resamples the texture, so
over a shot it softens and has to be re-energised, and a teacher is still needed often. Here nothing is warped from
frame to frame. The surf pass (datagen/blender/render_lr.py mode "surf") records for every pixel which object it shows
and which point of that object (its UV and Generated coordinates), and nts/surface.py finds for every pixel of a frame
the keyframe position that shows the same point:

  map_f(x)   = the keyframe position showing the surface point that frame f shows at x     (nts.surface.match)
  valid_f(x) = 1 where the keyframe shows that point; 0 where it was hidden, out of frame or not found
  R_f(x)     = R_k(map_f(x))                       one Catmull-Rom lookup into the keyframe's residual

The texture is resampled once however far the frame is from its keyframe, and every frame is matched on its own, so
nothing builds up along a shot and frames are matched in parallel (--workers). The keyframe's residual
R_k = teacher_k - teacher_input_k then sits on the surface for the whole shot: no flicker from re-generated texture,
and the teacher runs on one frame per shot.

Output: pseudo-teacher frames  teacher_input_f + valid_f * R_f  for every frame, to hand to gen.structure exactly like
a per-frame teacher's output (gen.structure removes the residual's low band and applies the limits).

Seen-again test and re-seating (--photo, --fill; 2026-11-21, design log rows 42 and 60). A matched point can still
look different at another frame (a shadow passes over it, a highlight moves), and the match leaves out every pixel
that shows more than one surface point (a hair strand over skin, an outline: nts.surface marks them mixed). Every
frame has its own render, so the map can be checked against it:
  agree_f(x) = | blur(render_f)(x) - blur(render_k)(map_f(x)) | <= tol        (RGB distance, levels)
Where a pixel fails, or has no match, its coordinates are re-seated from its neighbours: the map's offset (map - x)
is nearly constant over a small rigid patch, so the normalised average of the agreeing offsets around a hole is the
key position of whatever is in the hole, if it moved with its surroundings. The re-seated coordinates are kept only
where they pass the same test; elsewhere the pixel gets no detail on this frame, so a shadow passing over a surface
costs detail only while it is there. Offsets are averaged over neighbours at the pixel's own depth only, and the
re-seated place must have shown the same surface (depth) at the keyframe: a wall uncovered by a moving hand continues
the wall's detail where the keyframe saw that wall, and stays plain where the hand hid it at the keyframe. On at140n
(key 170) the re-seating gives detail to 12% of the frame's wanted part that the match alone leaves without (share
without carried detail 0.34; 0.45 with --fill 0) and lowers flicker (flk 0.80; 0.84 with --fill 0).
(History rejection by colour, as in temporal anti-aliasing: Yang, Liu & Salvi 2020, survey; the re-seating is ours.)

Evened offsets and a short reach (--settle, --fill; 2026-11-27, design log row 126). Every pixel is matched on its
own, so neighbouring pixels' keyframe positions carry small independent errors, and texture read at them shimmers
from frame to frame. Before the test, nts.surface.settle replaces a matched pixel's offset by the mean offset of the
matched pixels around it at its own depth (--settle 2: 5 x 5 px). The pixels without a match are thin (lash lines,
the lip line, outlines, a band of skin coming into view); re-seated from 36 px around (--fill 12, the earlier
default) whose make-up changed with every frame, their positions wobbled and stepped against their matched
neighbours. The surface carry re-seats from about 6 px (--fill 2). Face flicker, 8 eval clips: 0.72 before, 0.70
after (the motion-vector carry: 0.70); hp622n / pty332n 0.82 / 0.52 before, 0.79 / 0.50 after (eval/nr_board.py), with
slightly more of the face carrying detail.

Run: python -m gen.carry <geom_exr_dir> <key_teacher_dir> <teacher_input_dir> <out_dir> --surf <surf_exr_dir>
         [--keys f1,f2,...] [--frames a-b] [--feather 2] [--photo 20] [--fill 2] [--settle 2] [--roi 1]
         [--compose <toned_dir> <weights_dir> [--structure 1] [--hp 8] [--skin-limit 2.5] [--skin-mask 0.5]
          [--limit 6] [--mask 0.5] [--skin-even 1] [--hair-gain 1] [--edge 0] [--soft 0] [--hp-rel 0] [--agc 0]
          [--hair-hp 0] [--hair-limit 0] [--split 0] [--hp-ref 0] [--full <whole_frame_key_dir> [--agree 0] [--full-hp 0] [--full-gain 1]]
          [--masks <region_mask_dir>] [--own 0] [--fine-keep 0] [--skin-mid 1] [--fringe-skin] [--hair-guard] [--skin-ridge 0] [--skin-agree 0] [--plain-fill] [--seam 0] [--hair-flow 0] [--flow-keep 0] [--flow-mix 1] [--hair-mid 1] [--skin-grain 0] [--grain-band 2] [--hair-strands 0] [--outline 1] [--skin-mid-plain g] [--skin-limit-plain 0]]
--surf: the surf pass of the same frames (f_####.exr, render_lr.py mode "surf"); <geom_exr_dir>: the render's own
EXRs, for the shot cuts and the frame list.
--workers n: frames matched and composed in parallel by n processes (0 = one per CPU core; default env NTS_WORKERS,
else 1). The files written are byte-identical whatever n is.
--compose writes the finished frames instead: the keyframe's residual is band-limited once at the keyframe
(gen.structure.residual: --hp, --edge, --soft), carried, and composed in float with gen.structure.compose's limits, so it is
not rounded to 8 bits on the way (weights: the w_####.png files nts.tone writes).
--photo t: tolerance of the seen-again test in levels (full detail up to t, none from 1.5 t; 0 = off, as before
11-21; 20 and 30 flag nearly the same pixels on at140, 20 leaves fewer specks on closed lids); --fill r: radius of the
window the re-seated coordinates are averaged over (3 box passes); --settle r: half-width of the window the matched
offsets are evened out over before the test (0 = off). Both: see "Evened offsets and a short reach" above.
--roi 1 (default): if the keyframe's residual is non-zero only in a part of the frame (a head-crop teacher), each
frame is worked on only around where that part's surface is (see carried_surf()); --roi 0 always works on whole
frames.
<key_teacher_dir> needs the teacher's output only for the keyframes. Default keys: the middle frame of every shot.
Every frame takes the nearest keyframe of its own shot. Where nothing valid reaches a pixel (newly visible surface)
the output is the teacher input itself (no added detail).
Needs numpy, Pillow, OpenImageIO (~/scene-project/py on the 4050).
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gen.steady2 import box, cr_weights, frame, lowpass  # noqa: E402
from gen.structure import band_for, compose, key_residual, outline_fade, residual  # noqa: E402
from nts import warp as W  # noqa: E402
from nts.io_exr import Sequence  # noqa: E402
from nts.tone import load as tone_load  # noqa: E402
from nts.tone import box as tbox, maxr  # noqa: E402
from gen.structure import LUMA, low  # noqa: E402


def shots_of(seq, frames):
    """Runs of consecutive frames with no cut inside."""
    out = []
    for f in frames:
        if out and f == out[-1][-1] + 1 and not seq.reset[f]:
            out[-1].append(f)
        else:
            out.append([f])
    return out


def grid(hw):
    ys, xs = np.mgrid[0:hw[0], 0:hw[1]].astype(np.float32)
    return np.stack([xs, ys], -1)


def grow(b, pad, hw):
    """Rectangle (x0, y0, x1, y1) grown by pad and clipped to the frame."""
    return max(b[0] - pad, 0), max(b[1] - pad, 0), min(b[2] + pad, hw[1]), min(b[3] + pad, hw[0])


def bbox(mask):
    """Bounding rectangle (x0, y0, x1, y1) of a boolean mask, or None if it is empty."""
    ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1) if len(ys) else None


def grid_of(b):
    ys, xs = np.mgrid[b[1]:b[3], b[0]:b[2]].astype(np.float32)
    return np.stack([xs, ys], -1)




def agree(ref, cur, m, tol):
    """0..1: 1 where the frame's render (cur, on the map's rectangle) is within tol of the keyframe's render (ref,
    whole frame) at the carried coordinates, 0 from 1.5 tol on. Both are lightly blurred, so texture and noise do
    not count."""
    d = cur - W.bilinear(ref, m[..., 0], m[..., 1])
    return np.clip(3.0 - 2.0 * np.sqrt((d * d).sum(-1)) / tol, 0.0, 1.0)


def reseat(m, g, ok, r, depth=None):
    """Coordinates from the neighbours' offsets: (map, weight). weight ~ the share of agreeing pixels around.
    depth (this frame's): a pixel takes offsets only from neighbours at about its own depth (layers 12% apart, own
    layer +- half a layer). Without it the wall a moving hand uncovers took a mix of the hand's offsets (tens of px)
    and the wall's (zero): detail sampled along a sheared map, a swirl on the wall that changed with every frame
    (at140 185-195, x13)."""
    if depth is None:
        w = ok.astype(np.float32)
        den = box(box(box(w, r), r), r)
        num = box(box(box((m - g) * w[..., None], r), r), r)
        return g + num / np.maximum(den, 1e-4)[..., None], den
    need = ~ok
    lay = np.log(np.maximum(depth, 1e-6)) / np.log(1.12)
    li = np.floor(lay).astype(np.int32)
    out, den_out = m.copy(), np.zeros(m.shape[:2], np.float32)
    ids, cnt = np.unique(li[need], return_counts=True)
    for b in ids[np.argsort(-cnt)][:6]:
        sel = need & (li == b)
        if int(sel.sum()) < 30:
            continue
        w = (ok & (np.abs(lay - (b + 0.5)) < 1.0)).astype(np.float32)
        den = box(box(box(w, r), r), r)
        num = box(box(box((m - g) * w[..., None], r), r), r)
        out[sel] = (g + num / np.maximum(den, 1e-4)[..., None])[sel]
        den_out[sel] = den[sel]
    return out, den_out


MOTION = False  # the keyframe positions come from the surf pass (--surf) only
PAD = 64      # px of margin around the wanted region at the keyframe: the re-seating (36) and the lowpass, with room to spare


def seen_again(ref, base, f, m, valid, w1, hw, tol, fill, dep, d0):
    """The seen-again test and re-seating (module docstring) for one frame's map m and valid on w1. ref: the
    keyframe's lowpassed render or None (test off). dep: this frame's depth on w1; d0: the keyframe's, whole frame.
    Returns (map, valid, show)."""
    show = valid
    cur = base(f) if ref is not None else None
    if cur is not None:
        a, b, c, d = grow(w1, 3, hw)  # lowpass reaches 3 px
        cur = lowpass(cur[b:d, a:c])[w1[1] - b:w1[3] - b, w1[0] - a:w1[2] - a]
        g = grid_of(w1)
        ag = agree(ref, cur, m, tol)
        # valid < 1 next to a place that had no coordinates: the bilinear step mixed its neighbour's
        # leftover coordinates in, so the map is off there by a share of an unrelated offset. Such pixels
        # used to count as fine from 0.5 up and kept that state for good on a static surface: one thin seam
        # of weaker, misplaced detail per frame along a moving outline (at140 185-195, lines on the wall
        # above the hand, visible once the wall had whole-frame detail). They are re-seated instead.
        ok = (ag > 0.5) & (valid > 0.999)
        if not ok.all():
            m2, den = reseat(m, g, ok, fill, dep)
            ag2 = agree(ref, cur, m2, tol)
            # the place the new coordinates point at must have been this surface at the keyframe too: the
            # wall behind where the hand was at the keyframe has no wall detail there to take
            kx = np.clip(np.rint(m2[..., 0]), 0, hw[1] - 1).astype(np.int64)
            ky = np.clip(np.rint(m2[..., 1]), 0, hw[0] - 1).astype(np.int64)
            same = np.abs(d0[ky, kx] - dep) < 0.15 * dep  # loose: a head may lean in during a shot
            take = ~ok & (den > 0.05) & (ag2 > 0.5) & same
            m = np.where(take[..., None], m2, m).astype(np.float32)
            valid = np.where(take, 1.0, valid).astype(np.float32)
            ag = np.where(take, ag2, ag)
        show = (valid * ag).astype(np.float32)
    return m, valid, show




PAD_SURF = 40  # px around where the wanted surface is: the re-seating's reach (3 box passes, up to --fill 12) + lowpass


_SK = {}  # the keyframe's surface lookup in this process (carried_surf's workers)


def _surf_init(surf_dir, key, roi, boxes, wk):
    if _SK.get("id") == (surf_dir, key):  # forked from the process that made it
        return
    from nts import surface as SF
    ks = SF.read(os.path.join(surf_dir, f"f_{key:04d}.exr"))
    _SK.update(id=(surf_dir, key), K=SF.Key(ks), roi=roi, boxes=boxes, wk=wk, hw=ks.shape)


def _surf_match(f):
    """Frame f's keyframe positions by surface identity: (f, map, valid, window, depth in the window or None)."""
    from nts import surface as SF
    c = _SK
    hw, roi = c["hw"], c["roi"]
    cs = SF.read(os.path.join(c["id"][0], f"f_{f:04d}.exr"))
    w1 = (0, 0, hw[1], hw[0])
    if roi:  # where the wanted surface is in this frame; the keyframe's window when none of it is in view
        want = np.zeros(hw, bool)
        for o, (lo, hi) in c["boxes"].items():
            want |= (cs.ids == o) & (cs.f >= lo).all(-1) & (cs.f <= hi).all(-1)
        b = bbox(want)
        w1 = grow(b, PAD_SURF, hw) if b else c["wk"]
    m, valid = SF.match(c["K"], cs, w1, objects=list(c["boxes"]))
    dep = cs.depth[w1[1]:w1[3], w1[0]:w1[2]] if cs.depth is not None else None
    return f, m, valid, w1, dep


def carried_surf(seq, surf_dir, key, frames, base=None, tol=0.0, fill=2, roi=None, workers=None, settle=2):
    """Yield (f, map, valid, win) for the frames of key's shot (frames: that shot's frame list): the keyframe first,
    then the later frames, then the earlier ones, nearest first. map (h, w, 2: keyframe pixel coordinates) and valid
    (h, w: 0..1) cover the rectangle win = (x0, y0, x1, y1) of frame f. Every frame looks its keyframe positions up
    directly in the keyframe's surf pass (<surf_dir>/f_####.exr, render_lr.py mode "surf"; nts/surface.py), so no
    frame depends on another and nothing accumulates along the shot; workers (nts.par.workers) match that many
    frames at a time in worker processes.
    roi (x0, y0, x1, y1): the part of the keyframe whose detail is wanted; None = all of it, and win is then the
    whole frame. With a roi each frame is worked on only around where that part's surface is in it (PAD_SURF).
    base(f) -> the render of frame f (float RGB, the look the keyframe's teacher was given) or None; with tol > 0 it
    drives the seen-again test and the re-seating (module docstring; fill: the re-seating's radius).
    settle: half-width of the window the matched offsets are evened out over first (nts.surface.settle; 0 = off)."""
    from nts import surface as SF
    ks = SF.read(os.path.join(surf_dir, f"f_{key:04d}.exr"))
    K = SF.Key(ks)
    hw = ks.shape
    full = (0, 0, hw[1], hw[0])
    wk = grow(roi, PAD, hw) if roi else full
    gk = grid_of(wk)
    yield key, gk, np.ones(gk.shape[:2], np.float32), wk
    # the wanted part of the keyframe, as surface: its objects inside roi and the box of their points on each
    r = roi or full
    kid, kg = ks.ids[r[1]:r[3], r[0]:r[2]], ks.f[r[1]:r[3], r[0]:r[2]]
    boxes = {}
    for o in np.unique(kid[kid >= 0]):
        if int(o) in K.foot:
            pts, mg = kg[kid == o], 4 * K.foot[int(o)]
            boxes[int(o)] = (pts.min(0) - mg, pts.max(0) + mg)
    d0 = ks.depth if ks.depth is not None else seq.read(key).depth
    ref = base(key) if base is not None and tol > 0 else None
    if ref is not None:
        ref = lowpass(ref)
    _SK.clear()
    _SK.update(id=(surf_dir, key), K=K, roi=roi, boxes=boxes, wk=wk, hw=hw)
    order = [f for f in frames if f > key] + [f for f in frames if f < key][::-1]
    for f, m, valid, w1, dep in _stream(_surf_match, iter(order), workers, _surf_init, (surf_dir, key, roi, boxes, wk)):
        if dep is None:
            dep = seq.read(f).depth[w1[1]:w1[3], w1[0]:w1[2]]
        m = SF.settle(m, valid, dep, w1, settle)
        m, valid, show = seen_again(ref, base, f, m, valid, w1, hw, tol, fill, dep, d0)
        yield f, m, show, w1


def follow_head(m, v, win, wf, mk, reach, stretch, steps=False):
    """Two rules for a head-crop key that the coordinate map alone does not give (held-out clips, 2026-11-22).
    m, v: this frame's key coordinates and validity in the window win; wf: this frame's region weights (nts.tone);
    mk: this frame's renderer mask or None. Returns (m, v, (weight for light detail, weight for dark detail) or None).

    reach > 0 (--follow): the band just outside the head follows the HEAD. The key's detail reaches a few px past the
    head's outline (dark detail only: strands over the wall, gen.structure.key_residual). Those are wall pixels, so
    the map carried them with the wall: when the head moved, its strands stayed behind as a dotted outline on the
    wall where the head's edge had been at the key (hp622 676: a dotted line beside the ear; ti620, orbit camera: a
    dotted contour 15 px outside the head: "dotted halo by the ear", three judges, batch AJ), and the head's new
    edge was the renderer's bare cut ("helmet hair", "cutout edge"). Now a pixel outside the head but within reach
    of it takes the mean offset (map - x) of the head pixels around it: the head is rigid over a few px, so that is
    where the same strand was at the key. Everything further from the head gets no head detail on this frame, and
    the key's rule for the outline is applied again on this frame's own outline: light detail ends at the
    renderer's outline, dark detail fades out over `reach` px.

    stretch > 1 (--stretch): no detail where the map changes area by more than about that factor. A closing eyelid
    unrolls from the fold it was at the key: its pixels all point into that thin strip, and the lash line and
    crease drawn there were smeared over the whole lid ("closed eyelids smeared", "blurred eyelids", batch AJ). The
    render's own lid shows there instead. Detail is kept up to 0.8 x stretch and gone from 1.2 x stretch.
    steps (the surface carry, --surf): the map's slopes are taken from the smooth side of a step (slopes()). Its map
    steps where two surfaces that moved apart since the key meet (lid and eye, hair and forehead, the lips); central
    differences read that step as a stretch of hundreds, and every such outline lost its detail in a 2-3 px band
    whose strength changed with every frame (11-26, pty332n and hp622n: the whole face-flicker gap to the motion
    carry, whose map is smooth across outlines)."""
    x0, y0, x1, y1 = win
    hero = (np.clip((wf[..., 0].astype(np.float32) + wf[..., 1] + wf[..., 2]) / 255.0, 0, 1) > 0.3)[y0:y1, x0:x1]
    m, v, wgt = np.nan_to_num(m.astype(np.float32)), np.nan_to_num(v.astype(np.float32)), None
    if reach > 0:
        gx, gy = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
        hv = (hero & (v > 0.5)).astype(np.float32)
        n = reach + 2
        den = tbox(hv, n)
        ok = (~hero) & (den * (2 * n + 1) ** 2 >= 3)
        ax = tbox((m[..., 0] - gx) * hv, n) / np.maximum(den, 1e-6)
        ay = tbox((m[..., 1] - gy) * hv, n) / np.maximum(den, 1e-6)
        m[..., 0], m[..., 1] = np.where(ok, gx + ax, m[..., 0]), np.where(ok, gy + ay, m[..., 1])
        v = np.where(ok, 1.0, np.where(hero, v, 0.0)).astype(np.float32)
        true = hero if mk is None else ((mk[..., 0] > 127) | (mk[..., 2] > 127))[y0:y1, x0:x1]
        wgt = (np.clip(low(true.astype(np.float32), 1) * 2, 0, 1), np.clip(low(hero.astype(np.float32), reach) * 2.5, 0, 1))
    if stretch > 1 and steps:
        sx, sy = slopes(m, 1), slopes(m, 0)
        det = sx[..., 0] * sy[..., 1] - sy[..., 0] * sx[..., 1]
    elif stretch > 1:
        det = np.gradient(m[..., 0], axis=1) * np.gradient(m[..., 1], axis=0) - np.gradient(m[..., 0], axis=0) * np.gradient(m[..., 1], axis=1)
    if stretch > 1:
        q = np.where(det > 1e-3, np.maximum(det, 1.0 / np.maximum(det, 1e-3)), 1e3)
        keep = np.clip((1.2 * stretch - q) / (0.4 * stretch), 0, 1).astype(np.float32)
        v = v * tbox(1 - maxr(1 - keep, 1), 1)
    return m, v, wgt



STEP_PX, STEP_RATIO = 1.5, 4.0  # a step in the map: one neighbour's offset over 1.5 px AND over 4 x the other side's


def slopes(m, axis):
    """d(map)/d(axis) (axis 1 = x, 0 = y), (h, w, 2): central differences, but one-sided away from a step (the
    difference to one neighbour over STEP_PX and over STEP_RATIO times the other side's); a pixel with a step on both
    sides keeps the central difference."""
    c = np.gradient(m, axis=axis)
    d = np.diff(m, axis=axis)
    pad = [(0, 0)] * m.ndim
    pad[axis] = (0, 1)
    fwd = np.pad(d, pad, mode="edge")
    pad[axis] = (1, 0)
    bwd = np.pad(d, pad, mode="edge")
    nf, nb = np.linalg.norm(fwd, axis=-1), np.linalg.norm(bwd, axis=-1)
    jf = (nf > STEP_PX) & (nf > STEP_RATIO * nb)
    jb = (nb > STEP_PX) & (nb > STEP_RATIO * nf)
    return np.where(jf[..., None], bwd, np.where(jb[..., None], fwd, c)).astype(np.float32)


def inside(m, roi):
    return (m[..., 0] >= roi[0] - 1) & (m[..., 0] <= roi[2]) & (m[..., 1] >= roi[1] - 1) & (m[..., 1] <= roi[3])


def sample_cr(img, x, y):
    """img (H, W, c) at the points (x, y) (1-D arrays, pixel coordinates), Catmull-Rom, edges clamped: the same
    arithmetic as gen.steady2.warp_cr, for a list of points."""
    H, Wd = img.shape[:2]
    x, y = np.clip(x, 0, Wd - 1), np.clip(y, 0, H - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    wx, wy = cr_weights((x - x0)[..., None]), cr_weights((y - y0)[..., None])
    out = np.zeros((len(x), img.shape[2]), np.float32)
    for j in range(4):
        yy = np.clip(y0 + j - 1, 0, H - 1)
        row = np.zeros_like(out)
        for i in range(4):
            row += img[yy, np.clip(x0 + i - 1, 0, Wd - 1)] * wx[i]
        out += row * wy[j]
    return out


_CTX = None


def _carry_init(ctx):
    global _CTX
    _CTX = ctx


def _stream(fn, it, n, init, initargs):
    """fn over a generator's items, n at a time in worker processes (sequential for n <= 1), results in order. At
    most 2n items are in flight, so carried maps do not pile up in memory."""
    from nts.par import _ctx, workers
    n = workers(n)
    if n <= 1:
        for x in it:
            yield fn(x)
        return
    import collections
    import concurrent.futures as cf
    with cf.ProcessPoolExecutor(n, mp_context=_ctx(), initializer=init, initargs=initargs) as ex:
        q = collections.deque()
        for x in it:
            q.append(ex.submit(fn, x))
            while len(q) > 2 * n:
                yield q.popleft().result()
        while q:
            yield q.popleft().result()


def _carry_frame(item):
    """One carried frame (f, map, valid, window): sample the keyframe's residual along the map and compose it.
    Returns (written, share without carried detail, the plain-skin map compose made or None)."""
    f, m, valid, win = item
    c = _CTX
    a, comp, R, hp, roi, area, k, feather = c["a"], c["comp"], c["R"], c["hp"], c["roi"], c["area"], c["k"], c["feather"]
    kskin_lim, khair_lim = c["kskin_lim"], c["khair_lim"]
    opt = lambda key, d=None: a[a.index(key) + 1] if key in a else d  # noqa: E731
    fo = lambda key, d: float(opt(key, d))  # noqa: E731
    pkey = {"map": c["pkey"]}
    x = frame(a[2], f)
    if x is None:
        return False, 0.0, None
    v = np.clip(box(valid, feather) * 2 - 1, 0, 1) if feather else valid
    wgt = None
    if roi and comp and ("--follow" in a or "--stretch" in a):
        wf = tone_load(comp[1], f, "w")
        if wf is not None:
            reach = max(2, int(round(float(np.sqrt(((np.abs(R).max(-1) > 0.25)).sum())) / 28))) if fo("--follow", 0.0) > 0 else 0
            m, v, wgt = follow_head(m, v, win, wf, tone_load(opt("--masks"), f, "m") if "--masks" in a else None,
                                    reach, fo("--stretch", 0.0), steps="--surf" in a)
    sel = v > 0
    if roi:
        sel &= inside(m, roi)
    Rw = np.zeros(m.shape[:2] + (3,), np.float32)
    Rw[sel] = v[sel][:, None] * sample_cr(R, m[sel][:, 0], m[sel][:, 1])
    if wgt is not None:
        Rw *= np.where(Rw @ LUMA > 0, wgt[0], wgt[1])[..., None]
    Rf = np.zeros_like(x)
    Rf[win[1]:win[3], win[0]:win[2]] = Rw
    out = x + Rf
    made = None
    if comp:
        toned, w = frame(comp[0], f), tone_load(comp[1], f, "w")
        if toned is None:
            return False, 0.0, None
        skin = hair = None
        if w is not None:
            skin, hair = np.clip((w[..., 0] + w[..., 1]) / 255.0, 0, 1), w[..., 2] / 255.0
        fr = mkf = None
        if ("--fringe-skin" in a or "--fringe-zone" in a or "--outline" in a or "--keep-fringe" in a) and "--masks" in a:
            mkf = tone_load(opt("--masks"), f, "m")
        if ("--fringe-skin" in a or "--fringe-zone" in a) and mkf is not None and hair is not None:
            fr = hair * (mkf[..., 0] > 127)
        pio = {}
        if f != k and pkey.get("map") is not None:
            pw = np.zeros(m.shape[:2], np.float32)
            pw[sel] = sample_cr(pkey["map"][..., None], m[sel][:, 0], m[sel][:, 1])[:, 0]
            pio["given"] = np.zeros(x.shape[:2], np.float32)
            pio["given"][win[1]:win[3], win[0]:win[2]] = np.clip(pw, 0, 1)
        out = compose(toned, None, x, fo("--structure", 1.0), hp, skin, fo("--skin-limit", 2.5) * kskin_lim,
                      fo("--skin-mask", 0.5) * kskin_lim, hair, fo("--hair-gain", 1.0), fo("--limit", 6.0), fo("--mask", 0.5),
                      fo("--skin-even", 1.0), pre=Rf, agc=fo("--agc", 0.0),
                      hair_limit=((fo("--hair-limit", 0.0) or fo("--limit", 6.0)) * khair_lim
                                  if khair_lim < 1 else fo("--hair-limit", 0.0)), own=fo("--own", 0.0) * khair_lim,
                      fine_keep=fo("--fine-keep", 0.0), fine_share=kskin_lim, skin_mid=fo("--skin-mid", 1.0), fringe=fr,
                      hair_flow=fo("--hair-flow", 0.0), flow_keep=fo("--flow-keep", 0.0),
                      flow_mix=fo("--flow-mix", 1.0), hair_mid=fo("--hair-mid", 1.0),
                      sil=outline_fade(mkf, fo("--outline", 1.0)) if "--outline" in a else None,
                      mid_plain=fo("--skin-mid-plain", 1.0) if "--skin-mid-plain" in a else None, hair_over="--hair-over" in a, skin_low=fo("--skin-low", 0.0), skin_low_gain=fo("--skin-low-gain", 1.0), fringe_zone=int(fo("--fringe-zone", 0)), plain_zone=fo("--plain-zone", 0.0), hair_keep=fo("--hair-keep", 0.0), hair_rim=fo("--hair-rim", 0.0), plain_local=fo("--plain-local", 0.0), plain_face=None if mkf is None else mkf[..., 0].astype(np.float32) / 255.0, plain_cmin=fo("--plain-cmin", 0.3), plain_soft="--plain-soft" in a, keep_zone=(hair * (mkf[..., 0] > 127) if "--keep-fringe" in a and mkf is not None and hair is not None else None),
                      limit_plain=fo("--skin-limit-plain", 0.0) * kskin_lim, plain_io=pio, plain_fill="--plain-fill" in a, seam=int(fo("--seam", 0)))
        if f == k and pio.get("made") is not None:
            made = pio["made"].astype(np.float32)
    Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(a[3], f"f_{f:04d}.png"))
    return True, 1 - min(float(v[sel].sum()) / area, 1.0), made


def main():
    a = sys.argv[1:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    surf = opt("--surf")
    if len(a) < 4 or not (surf or MOTION):
        sys.exit(__doc__)
    seq = Sequence(a[0])
    span = tuple(map(int, opt("--frames", "0-999999").split("-")))
    feather = int(opt("--feather", 2))
    tol, fill = float(opt("--photo", 20.0)), int(opt("--fill", 2))
    todo = [f for f in seq.frames if span[0] <= f <= span[1]]
    shots = shots_of(seq, todo)
    keys = sorted(map(int, opt("--keys").split(","))) if "--keys" in a else [s[len(s) // 2] for s in shots]
    os.makedirs(a[3], exist_ok=True)
    comp = (a[a.index("--compose") + 1], a[a.index("--compose") + 2]) if "--compose" in a else None
    fo = lambda k, d: float(opt(k, d))  # noqa: E731
    nw = opt("--workers")  # frames composed in parallel (nts.par.workers: 0 = one per core; default env NTS_WORKERS or 1)
    n = lost = 0
    for shot in shots:
        ks = [k for k in keys if k in shot]
        if not ks:
            print(f"carry: shot {shot[0]}-{shot[-1]} has no keyframe, skipped")
            continue
        for k in ks:
            mine = [f for f in shot if min(ks, key=lambda q: (abs(q - f), q)) == k]
            teacher, tin = frame(a[1], k), frame(a[2], k)
            if teacher is None or tin is None:
                print(f"carry: keyframe {k} is missing in {a[1]} or {a[2]}")
                continue
            R = teacher - tin
            hp = band_for(teacher, tin, int(opt("--hp", 8)), fo("--hp-rel", 0.0), fo("--hp-ref", 0.0))
            if comp:
                wk = tone_load(comp[1], k, "w")  # the keyframe's own region weights: hair band, per-region low band
                kskin, khair = (None, None) if wk is None else (np.clip((wk[..., 0] + wk[..., 1]) / 255.0, 0, 1),
                                                                wk[..., 2] / 255.0)
                full = frame(opt("--full"), k) if "--full" in a else None  # whole-frame edit of the keyframe (optional)
                hg = {}
                R, hp, kskin_lim, khair_lim = key_residual(teacher, tin, full, int(opt("--hp", 8)), fo("--hp-rel", 0.0), fo("--edge", 0.0),
                                     fo("--soft", 0.0), khair, int(opt("--hair-hp", 0)), kskin,
                                     bool(int(opt("--split", 0))), fo("--agree", 0.0), fo("--hp-ref", 0.0),
                                     int(opt("--full-hp", 0)), fo("--full-gain", 1.0),
                                     tone_load(opt("--masks"), k, "m") if "--masks" in a else None,
                                     fo("--skin-grain", 0.0), fo("--hair-strands", 0.0), fo("--grain-band", 2.0),
                                     "--feat-guard" in a, "--hair-guard" in a, hg, fo("--skin-ridge", 0.0), fo("--skin-agree", 0.0))
                if "skin_agree" in hg:
                    sk_ = kskin > 0.5
                    print("skin agree, key %d: detail kept on %.2f of the skin (under half on %.3f)" % (k, float(hg["skin_agree"][sk_].mean()), float((hg["skin_agree"][sk_] < 0.5).mean())))
                if "hair_guard" in hg:
                    print("hair guard, key %d: %d px where the edit shows skin on the render's hair" % (k, int((hg["hair_guard"] > 0.5).sum())))
            H, Wd = R.shape[:2]
            roi = bbox(np.abs(R).max(-1) > 0.25) if int(opt("--roi", 1)) else None
            roi = grow(roi, 4, (H, Wd)) if roi else None
            if roi and (roi[2] - roi[0]) * (roi[3] - roi[1]) > 0.6 * H * Wd:
                roi = None
            area = (roi[2] - roi[0]) * (roi[3] - roi[1]) if roi else H * Wd
            ctx = dict(a=a, comp=comp, R=R, hp=hp, kskin_lim=kskin_lim if comp else 1.0,
                       khair_lim=khair_lim if comp else 1.0, roi=roi, area=area, k=k, feather=feather, pkey=None)
            todo_k = [f for f in shot if min(mine) <= f <= max(mine)]
            if surf:  # keyframe positions by surface identity (nts/surface.py)
                frames_k = carried_surf(seq, surf, k, todo_k, lambda q: frame(a[2], q), tol, fill, roi, nw,
                                        int(opt("--settle", 2)))
            # the keyframe comes first and makes the plain-skin map the other frames read (gen.structure.compose,
            # plain_io); every other frame is independent once its coordinates are carried: --workers composes them
            # in parallel while this process carries the map on
            _carry_init(ctx)
            done = [_carry_frame(next(frames_k))]
            ctx["pkey"] = done[0][2]
            for r in _stream(_carry_frame, frames_k, nw, _carry_init, (ctx,)):
                done.append(r)
            for wrote, lo, _ in done:  # in the order of the sequential version, so the printed share is the same
                if wrote:
                    n += 1
                    lost += lo
    print(f"carry: {n} frames -> {a[3]}; keys {keys}; mean share without carried detail {lost / max(n, 1):.3f}")


if __name__ == "__main__":
    main()
