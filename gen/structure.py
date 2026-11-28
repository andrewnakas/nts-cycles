"""Structure branch, composed onto a toned frame (NR-1 v2, 2026-11-19; the design log row 23).

The teacher's change (steadied teacher output minus the render it was given) is added to the tone branch's frame:
  R    = teacher - teacher_input, with its own low band removed (three box passes, radius --hp), so only texture
         and strand-scale structure crosses over, never the teacher's or its input look's tone
  gain = local luminance of the toned frame over that of the teacher's input (clamped 0.4..1.6): texture rides on
         the new tone instead of keeping the contrast it had on a brighter surface
  out  = toned + --structure * gain * R
teacher and teacher_input must show the same geometry as the toned frame (same shot and frames; a different look of
the same render is fine: the residual is pixel-aligned).

The residual is soft-limited (tanh) to --limit levels plus --mask times the toned frame's own local contrast, and on
skin (--weights, the w_####.png files nts.tone writes) to --skin-limit + --skin-mask times it: the teacher may
sharpen what the render shows (lashes, lips, brows, cloth weave) and add pore-scale texture, but it cannot draw
high-contrast marks on smooth surfaces (no stubble on skin, no dot patterns in hair).

Run: python -m gen.structure <toned_dir> <teacher_dir> <teacher_input_dir> <out_dir> [--structure 1.0] [--hp 8]
         [--weights <dir>] [--skin-limit 2.5] [--skin-mask 0.5] [--skin-even 1] [--limit 6] [--mask 0.5] [--hair-gain 1] [--frames a-b]
         [--render <detailed_look_dir> <base_look_dir> [--render-gain 1] [--render-hair 0] [--render-hp <--hp>]]
         [--edge 0] [--soft 0] [--hp-rel 0] [--agc 0] [--hair-hp 0] [--hair-limit 0] [--split 0] [--hp-ref 0]
         [--full <dir> [--agree 0] [--full-hp 0] [--full-gain 1]] [--masks <region_mask_dir>]
         [--own 0] [--fine-keep 0] [--skin-mid 1] [--fringe-skin (needs --masks)] [--hair-over] [--fringe-zone 0 (px, needs --masks)] [--plain-zone 0] [--feat-guard] [--hair-keep 0 (levels) [--keep-fringe]] [--plain-cmin 0.3] [--plain-soft] [--skin-low 0] [--skin-low-gain 1] [--hair-flow 0] [--flow-keep 0] [--flow-mix 1] [--hair-mid 1] [--skin-grain 0] [--grain-band 2] [--hair-strands 0]
--hp-ref s: with --hp-rel; above a head box of s px the band narrows in proportion (see band_for()) and so do the
skin limit and skin masking; under s px the head's skin and hair limits are scaled by side / s (see key_residual()).
--full <dir>: whole-frame edits of the same frames (same names); their detail is added outside the head crop's box.
--agree t: whole-frame detail is taken only where that edit kept the input's shapes within t levels (0 = everywhere).
--masks d: the renderer's region masks (m_####.png). With them the added detail draws no small dark specks on the
            lower face (no stubble: beard_zone(), speck_free()).
--full-hp r, --full-gain g: the whole-frame detail's own band radius (0 = --hp) and gain. It is lightness only and
            is left off print and screen text (text_zone()).
--split 1: remove each region's own low band (skin, hair, rest) instead of one blur across them (see residual()).
--hair-hp h, --hair-limit l: band radius and limit inside the hair mask (0 = as everywhere else).
--agc a: limit by a smooth gain instead of clipping: local RMS of the added detail is held to a x the limit (see
compose()); for teachers whose detail is far stronger than the limits (generative keyframe edits).
--hp-rel r: the band radius becomes (larger side of the part the teacher edited) / r, between 4 and --hp (band_for()).
--soft s (px): Gaussian blur of the teacher's residual (see residual()); for generative keyframe edits.
--edge e (levels, e.g. 25) takes the teacher's detail over an edge-preserving base instead of a plain blur (see
residual()); use it when the teacher's tone differs from its input's (generative keyframe edits).
--render adds a second texture source (needs --weights): the fine band of a more detailed look of the same render
over the look the toned frame was made from (both steadied the same way), brightness only, on plain skin (and on
hair at weight --render-hair). It
goes through the same limits and the same skin cap as the teacher's texture, so stubble painted in that look is held
to the level of the cheeks.
--hair-gain multiplies the residual inside the hair weight (strands are where the teacher's change is most wanted).
Needs numpy and Pillow.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts.tone import box, frames_in, load, maxr, to_lab  # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def low(a, r):
    return box(box(box(a, r), r), r)


def plain(toned, skin, hair, hair_w=0.0, shrink=5, cmin=0.3, feather=2, local=0.0):
    """Where the render-texture source (--render) applies: skin pixels that have the skin's own colour (hue within
    ~25 degrees of the skin's median, chroma 0.45-2x), shrunk by `shrink` px, and hair at weight hair_w. Eyes, teeth,
    lips and a band around them are left out: there a more detailed look of the same render differs in material, not
    in texture (iris colour, the edge of a darkened moustache zone at the lip)."""
    L, a, b = to_lab(toned)
    sel = skin > 0.6
    if sel.sum() < 500:
        return np.zeros_like(skin) if hair is None else hair_w * hair
    am, bm = float(np.median(a[sel])), float(np.median(b[sel]))
    c, cm = np.hypot(a, b), float(np.hypot(am, bm))
    if local > 0:
        # local (fraction of sqrt(skin area)): the skin's own colour is taken from the skin AROUND the pixel, not
        # from one median over all skin in the frame. Held-out clip ti620 (2026-11-22: dark studio, a teal screen
        # lighting one cheek, other skin of the hero large in frame and nearly grey): the median chroma of all skin
        # was 2.7 against 13.7 on the face, the whole face failed the test (plain 0.00 of the face against
        # 0.54-0.75 on the other clips), so every skin pixel was held to the feature limit and the mid band cut:
        # fine detail 2.1 levels RMS of the edit's 8.1 ("orange waxy plastic skin", three judges, batch AJ).
        #  - The reference is the mean skin colour in a window of +-r px, and a pixel passes when it fits the window
        #    round itself OR one of the four windows shifted by r: at the edge between two lights the centred window
        #    mixes both and neither side fits it (before the shrink: 0.81 of the face with one window, 0.90 with
        #    five). Eyes, teeth and lips are smaller than the windows and still stand out from all five.
        #  - Where the surrounding skin is nearly grey its hue means little: the hue test measures the distance
        #    from the local hue line in units of the local chroma (the same as 12.5-25 degrees at that chroma), and
        #    the chroma is not taken under half the frame's median.
        #  - The colour is smoothed over 2 px (x the face scale) first: in a dark frame single noisy pixels fail and
        #    the shrink turns each into a hole of its own size (plain 0.38 -> 0.49 on ti620 with one window).
        r = max(4, int(round(local * float(np.sqrt(sel.sum())))))
        w = sel.astype(np.float32)
        den = box(w, r)
        al = np.where(den > 0.02, box(a * w, r) / np.maximum(den, 1e-3), am)
        bl = np.where(den > 0.02, box(b * w, r) / np.maximum(den, 1e-3), bm)
        q = int(round(2 * max(1.0, float(np.sqrt(sel.sum())) / 300.0)))
        a, b = box(a, q), box(b, q)
        c = np.hypot(a, b)

        def fits(al, bl):
            cl = np.maximum(np.hypot(al, bl), 0.5 * cm)
            dh = np.abs(np.angle(np.exp(1j * (np.arctan2(b, a) - np.arctan2(bl, al)))))
            dt = c * np.sin(np.minimum(dh, np.pi / 2)) / cl
            return np.clip(2 - dt / 0.216, 0, 1) * np.clip((c / cl - cmin) / 0.15, 0, 1) * np.clip((2.3 - c / cl) / 0.3, 0, 1)

        p = fits(al, bl)
        for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r)):
            p = np.maximum(p, fits(np.roll(al, (dy, dx), (0, 1)), np.roll(bl, (dy, dx), (0, 1))))
    else:
        dh = np.abs(np.angle(np.exp(1j * (np.arctan2(b, a) - np.arctan2(bm, am)))))
        p = np.clip(2 - dh / np.deg2rad(12.5), 0, 1) * np.clip((c / cm - cmin) / 0.15, 0, 1) * np.clip((2.3 - c / cm) / 0.3, 0, 1)
    p = box(1 - maxr((1 - p * skin).astype(np.float32), shrink), feather)  # shrink, then feather
    return p if hair is None else np.maximum(p, hair_w * hair)


def _bbox(m, pad):
    """(y0, y1, x0, x1) of the True pixels of m, grown by pad and cut to the frame; None when there are none."""
    ys, xs = np.nonzero(m)
    if not len(ys):
        return None
    return (max(0, int(ys.min()) - pad), min(m.shape[0], int(ys.max()) + pad + 1),
            max(0, int(xs.min()) - pad), min(m.shape[1], int(xs.max()) + pad + 1))


def scalp(hair, skin, r):
    """The hair of the head without brows and lashes (0 / 1 map). The renderer's hair mask, and with it the tone
    step's hair weight, also holds the eyebrows. Scalp hair has pixels more than r px from any skin, brows have none:
    those pixels are the seed, and the seed is grown inside the hair weight (over 0.3) to the hairline, the fringe
    and the sideburns. Why it matters: the border rules of compose (plain_fill, seam) are for the hairline; round
    the brows they smoothed the skin (tbs04 364, build f1: 1,250 px changed in the eye and brow box)."""
    out = np.zeros(hair.shape, np.float32)
    bb = _bbox(hair > 0.3, r + 2)
    if bb is None:
        return out
    q = (slice(bb[0], bb[1]), slice(bb[2], bb[3]))
    m = (hair[q] > 0.3).astype(np.float32)
    g = m * (hair[q] > 0.5) * (maxr((skin[q] > 0.3).astype(np.float32), r) < 0.5)
    for _ in range(600):
        g2 = np.minimum(maxr(g, 2), m)
        if float((g2 - g).sum()) < 0.5:
            break
        g = g2
    out[q] = g
    return out


def not_skin(img, skin):
    """1 where a pixel is clearly not skin: nearly grey (chroma under 0.3 of the skin's median: eye white, grey
    iris, teeth) or a hue more than 60 degrees off the skin's (blue iris). Shading, freckles and lips are skin."""
    L, a, b = to_lab(img)
    sel = skin > 0.6
    if sel.sum() < 500:
        return np.zeros_like(skin)
    am, bm = float(np.median(a[sel])), float(np.median(b[sel]))
    c, cm = np.hypot(a, b), float(np.hypot(am, bm))
    dh = np.abs(np.angle(np.exp(1j * (np.arctan2(b, a) - np.arctan2(bm, am)))))
    far = np.clip((dh - np.deg2rad(60)) / np.deg2rad(15), 0, 1) * (c / cm > 0.3)
    return np.maximum(np.clip((0.3 - c / cm) / 0.1, 0, 1), far).astype(np.float32)


def guided(img, guide, r, eps):
    """Guided filter (He, Sun & Tang 2010, paper): an edge-preserving low band of img (H, W, 3), steered by the
    single-channel guide. Steps much larger than sqrt(eps) stay in the output, texture below it is smoothed away."""
    m = box(guide, r)
    a_den = box(guide * guide, r) - m * m + eps
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        p = img[..., c]
        mp = box(p, r)
        a = (box(guide * p, r) - m * mp) / a_den
        out[..., c] = box(a, r) * guide + box(mp - a * m, r)
    return out


def gauss(a, sigma):
    """Small separable Gaussian blur (edges repeated)."""
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r, r)
        p = np.pad(a, pad, mode="edge")
        n = a.shape[ax]
        sl = [slice(None)] * a.ndim

        def tap(i):  # p shifted by i along ax: a view (the first version took a copy with np.take; same values)
            sl[ax] = slice(i, i + n)
            return p[tuple(sl)]
        a = sum(k[i] * tap(i) for i in range(2 * r + 1))
    return a


def band_for(teacher, tin, hp, rel, ref=0.0):
    """Band radius for a teacher that edited only a part of the frame (a head crop): the edited part's larger side
    / rel, between 4 and hp. A fixed 16 px band is 3.5% of a 450 px head box but 8% of a 206 px one (zd400's far
    shot), where it reaches the size of eyes and mouth and moves the face-match score (x8: 0.849).
    ref > 0: above a head box of ref px the band narrows in proportion (hp * ref / side). On a close-up the render
    already defines the skin's mid-scale structure; the edit's version of it (an 8-16 px cell pattern on pty332's
    1080 px head, judged "scaly") must not replace it, only the finest band is missing."""
    if rel <= 0:
        return hp
    d = np.abs(teacher - tin).max(-1) > 0.5
    ys, xs = np.flatnonzero(d.any(1)), np.flatnonzero(d.any(0))
    if not len(ys):
        return hp
    side = max(int(xs[-1] - xs[0]), int(ys[-1] - ys[0])) + 1
    b = min(hp, max(4, round(side / rel)))
    if ref > 0 and side > ref:
        b = max(4, round(b * ref / side))
    return int(b)


def agreement(teacher, tin, r=24, tol=8.0):
    """1 where the teacher's picture is its input's shapes under a local change of tone, 0 where it shows other
    shapes (a whole-frame generative edit keeps the layout but redraws small faces, hands and the folds and outline
    of clothes). The input's luma is fitted to the teacher's by the best local affine map (the guided filter's
    model, window r); what that map cannot explain at ~3 px and coarser is a shape that moved or was invented. Detail
    is taken from the teacher only where that leftover is under `tol` levels (soft, to 0 at 2 x tol)."""
    a, b = low(teacher @ LUMA, 1), low(tin @ LUMA, 1)
    q = guided(a[..., None], b, r, 4.0)[..., 0]
    e = np.sqrt(low((a - q) ** 2, r // 4))
    return np.clip(2.0 - e / tol, 0.0, 1.0)


def head_box(teacher, tin):
    """(x0, y0, x1, y1) of the part of the frame a head-crop teacher edited, or None."""
    d = np.abs(teacher - tin).max(-1) > 0.5
    ys, xs = np.flatnonzero(d.any(1)), np.flatnonzero(d.any(0))
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1) if len(ys) else None


def text_zone(tin, rng=64.0, grow=9):
    """0..1: 1 on and round print and screen text (and any other hard, fine contrast): the input's 5x5 lightness
    range above rng levels, grown by grow px, with a soft rim. A whole-frame edit redraws letters; its detail there
    cost 5 dB of text PSNR (x11: 21.9-23.5 dB after the colour fit on three clips, tone alone 26.7-29.0)."""
    y = tin @ LUMA
    hi, lo = y, y
    for ax in (0, 1):
        pad = [(2, 2) if a == ax else (0, 0) for a in (0, 1)]
        n = y.shape[ax]
        ph, pl = np.pad(hi, pad, mode="edge"), np.pad(lo, pad, mode="edge")
        hi = np.maximum.reduce([np.take(ph, np.arange(i, i + n), axis=ax) for i in range(5)])
        lo = np.minimum.reduce([np.take(pl, np.arange(i, i + n), axis=ax) for i in range(5)])
    z = ((hi - lo) > rng).astype(np.float32)
    return np.clip(low(np.clip(low(z, grow) * (2 * grow + 1), 0, 1), 3) * 2 - 0.5, 0, 1)


def beard_zone(mask):
    """(zone, face height): zone 0..1 covers the lower face and the top of the neck (upper lip, chin, jaw), in the
    coordinates of the renderer's face mask (mask: m_####.png as float RGB; face = R and not hair): from 0.5 of the
    face's height (the nostrils; 0.58 left dots beside them) down to 0.3 of it below the chin. None without a face."""
    face = (mask[..., 0] > 127) & (mask[..., 2] <= 127)
    ys, xs = np.nonzero(face)
    if ys.size < 400:
        return None, 0
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    Y, X = np.mgrid[0:face.shape[0], 0:face.shape[1]]
    u, v = (X - x0) / max(x1 - x0, 1), (Y - y0) / max(y1 - y0, 1)
    z = np.clip((v - 0.50) / 0.08, 0, 1) * (v < 1.3) * (u > -0.15) * (u < 1.15)
    return z.astype(np.float32), int(y1 - y0)


def speck_free(R, zone, r):
    """R without the small dark specks it draws inside zone: a grey closing of its lightness with a (2r+1) px square
    fills dark dots and hairs narrower than that and leaves light specks, creases wider than it and all shading.
    The user wants no stubble (2026-11-19), and a head edit made from a frame with painted stubble draws real
    stubble even after the median filter: dark dots on the upper lip and chin at 3x zoom (rs345 375, at140 170;
    the board's beard checks did not move, cheeks and chin gain texture alike)."""
    y = R @ LUMA
    c = -maxr(-maxr(y, r), r)
    # the closing's flat tops are squares: on a chin full of specks nearly every pixel was lifted to one, a mosaic of
    # 5 px blocks over the lower face and neck (map of x13 - tone, 2026-11-21; judges: "blotchy smooth skin"). The
    # envelope is smoothed and only what lies under it is lifted.
    c = gauss(c, max(0.8, 0.6 * r))
    return R + (zone * np.maximum(c - y, 0))[..., None]


def skin_grain(tin, R, skin, target, scale, seed=7, hi=2.0):
    """Lightness grain that tops plain skin up to an even fine texture -> (H, W) levels to add to R.
    target: RMS of the 1-3 px band (at a 480 px head box; the band scales with `scale`) in % of the local mean; a
    zone that already has that much (input plus R) gets nothing, a smooth one gets the difference (in power).
    Why: the head edit draws real stubble on the lower face, which is taken out again (speck_free, no stubble), so
    cheeks below the nostrils, chin and jaw end up SMOOTHER relative to the upper face than in the render itself
    (board x16, beard-zone / cheek fine energy against the render's: 0.62-0.70; plain-skin 1-2 px
    band on the lower face 0.6-0.7 % against 1.6 % on the upper face): the "waxy skin" of
    judge batch Q. The grain is band-limited Gaussian noise: as many light grains as dark, so it has pores' texture
    and none of stubble's dark dots; it is made once per keyframe and carried with R, so it sits on the skin.
    Texture synthesis by filtered noise: Perlin 1985, "An image synthesizer"."""
    y = tin @ LUMA
    p = plain(tin, skin, None)
    if float(p.max()) <= 0:
        return np.zeros_like(y)
    s = max(0.5, scale)
    band = lambda v: gauss(v, 0.6 * s) - gauss(v, hi * s)  # noqa: E731
    m = gauss(y, 6.0 * s) + 1.0
    e = 100.0 * np.sqrt(box(band(y + R @ LUMA) ** 2, max(2, int(round(5 * s))))) / m
    n = band(np.random.default_rng(seed).standard_normal(y.shape).astype(np.float32))
    n = n / max(float(n.std()), 1e-6)
    return (p * np.sqrt(np.maximum(target * target - e * e, 0.0)) / 100.0 * m * n).astype(np.float32)


def key_residual(teacher, tin, full, hp, rel, edge=0.0, soft=0.0, hair=None, hair_hp=0, skin=None, split=False,
                 agree=0.0, ref=0.0, full_hp=0, full_gain=1.0, mask=None, grain=0.0, strands=0.0, grain_hi=2.0,
                 feat_guard=False, hair_guard=False, guard_out=None, skin_ridge=0.0, skin_agree=0.0):
    """The detail one keyframe adds -> (R, band radius used for the head, skin budget scale, hair budget scale).
    teacher: the frame with the edited head crop pasted in (it equals tin elsewhere).
    full (optional): a whole-frame edit of the same frame. A whole-frame edit gives clothes, walls and props real
    texture but redraws a small face, so the head (skin and hair inside the head crop's box, grown a little) takes
    the head crop's detail and everything else takes the whole-frame edit's, where agreement() says it kept the
    input's shapes. The wall behind the head therefore has ONE texture: with the head crop's detail on the whole
    box the box showed as a smoother rectangle on a plastered wall (judge stills O, at140).
    Skin budget scale (with ref > 0): ref / head side above a head box of ref px, else 1. The caller multiplies its
    skin limit and skin masking by it: on a close-up an edit made from the v9d look redraws that look's skin cells
    as a net of lines ("scaly", "reptile"); at 6 levels it shows, at 2.7 (1080 px head) it reads as skin.
    Under ref px both scales are side / ref: on a small head the same levels of added detail are a larger change of
    the face (206 px head, zd400 430, face match to the render: 0.87 at full budget, 0.97 at 0.4). The
    scales act on the limits, after which the detail is measured, not on R: a gain before the limiter is undone
    by it (tried: 0.94)."""
    hb = band_for(teacher, tin, hp, rel, ref)
    b = head_box(teacher, tin)
    side = max(b[2] - b[0], b[3] - b[1]) if b else 0
    # under ref px the budget follows the head's AREA ((side / ref)^2; it was side / ref until 2026-11-21): on a
    # 208 px head box (zd400's far shot, a 90 px face) the linear rule still cost 0.03-0.04 of face match against the
    # toned frame (key still 405: tone 0.973-0.983, with detail 0.930-0.953), and the
    # judges of batch Q named tone there, not detail.
    small = min(1.0, side / ref) ** 2 if ref > 0 and side > 0 else 1.0
    k = min(1.0, ref / side, small) if ref > 0 and side > 0 else 1.0
    kh = small
    R = residual(teacher, tin, hb, edge, soft, hair, hair_hp, skin, split)
    if feat_guard and skin is not None:
        # Features are never ADDED. Where the render has plain skin and the edit shows something that is not skin
        # (eye white, iris, lips, teeth: plain() of the edit, grown with the head), the edit drew a feature that the
        # render does not have there, and nothing is taken from it. pty332 356 (y1, batch AF, three judges:
        # "doubled / two-tone irises"): the no-reference edit opened the half-closed eyes, and its iris sat on the
        # render's upper lid. A wider protected strip round the render's own features (plain_zone) removes it too
        # but leaves a smooth patch with a visible edge on the close-up (stills V1, V2).
        # What it removes in practice (stills G1, G2, heads of 300-480 px): the edit's own shading round nose, eye
        # sockets and mouth, whose colour is off the skin's; pores and freckles on plain skin stay. Face match to the
        # render on the key stills: 0.74-0.79 without it, 0.78-0.93 with it.
        # The colour test is made at the scale of a 480 px head: on the close-up (1080 px) every pore and freckle
        # is some pixels wide and counted as "not skin", and the guard emptied the whole face (still G1, pty332).
        te, ti = (teacher, tin) if side <= 480 else (gauss(teacher, side / 400.0), gauss(tin, side / 400.0))
        # ... and there only blobs of side / 100 px and more count (an opening): the blur alone left the cheeks
        # of the close-up nearly empty (still H3), its freckles are 6-10 px wide.
        # 2026-11-21, batch AG: with the test "not plain skin" (plain()) the guard also removed the edit's shading
        # round nose, eye sockets and mouth. Face match rose to 0.78-0.93, and three blind judges turned the result
        # down ("smooth plastic skin"): that shading is what reads as a photograph. The test is now
        # not_skin(): eye white, iris, teeth; blobs of side / 200 px and more (2 px at least).
        ro = max(1, side // 200)
        ne = not_skin(te, skin) * (skin > 0.3)
        ne = maxr(1 - maxr((1 - ne).astype(np.float32), ro), ro)
        pe = box(1 - maxr(ne.astype(np.float32), max(5, side // 64)), 2)
        R = R * (1 - np.clip(plain(ti, skin, None) - pe, 0, 1))[..., None]
    if hair_guard and skin is not None and hair is not None and b:
        # Hair is never REMOVED. The no-reference edit draws its own hairline; seen from above with the head bent
        # (tbs05, a desk shot) it drew the fringe some 20 px shorter, and with the hair open to the edit
        # (compose: hair_over, hair limit 120) the strip between the two hairlines took the edit's forehead: +30 to
        # +56 levels on hair, a pale strip whose lower edge is the hair mask's, a line across the forehead and down
        # the temple in every frame of the shot (458 col 640: render 76-113, edit 118-172, built 106-158).
        # A cap on ALL hair's lightening (compose: hair_keep) was tried on 11-21 and gave back the render's moulded
        # hair cap. Here only the zone counts where the render has hair and the edit shows SKIN at the same place:
        # skin's colour (not_skin), about the face's lightness in the edit, clearly lighter than the render there,
        # and joined to the face's own skin (a highlight on top of the hair is none of these). In that zone the
        # detail may not lighten at all (first try: only its low band at band / 12, as hair_keep does; the strip is
        # 20 px wide between dark neighbours, so its low band is +14 of the +56 and the strip stayed). Darker
        # detail, strands, passes.
        x0, y0, x1, y1 = b
        Lt, Li = to_lab(teacher)[0], to_lab(tin)[0]
        sel = skin > 0.6
        zone = np.zeros(tin.shape[:2], np.float32)
        if sel.sum() >= 500:
            Ls = float(np.median(Lt[sel]))
            cand = ((hair > 0.5) * (1 - not_skin(teacher, skin)) * np.clip((Lt - (Ls - 22.0)) / 6.0, 0, 1)
                    * np.clip((Lt - Li - 6.0) / 6.0, 0, 1)).astype(np.float32)[y0:y1, x0:x1]
            seed = maxr((skin * (1 - hair) > 0.5).astype(np.float32), 3)[y0:y1, x0:x1]
            g = np.minimum(seed, cand)
            for _ in range(max(8, side // 3)):
                g2 = np.maximum(g, np.minimum(maxr(g, 1), cand))
                if float((g2 - g).sum()) < 0.5:
                    break
                g = g2
            ro = max(1, side // 200)
            g = maxr(1 - maxr(1 - g, ro), ro)                     # blobs of side / 200 px and more
            zone[y0:y1, x0:x1] = np.clip(box(maxr(g, 2), 2), 0, 1)
            R = R - (zone * np.maximum(R @ LUMA, 0))[..., None]
        if guard_out is not None:
            guard_out["hair_guard"] = zone
    if skin_ridge > 0 and skin is not None and b:
        # Outlines are never ADDED. The edit draws its own ear, jaw and neck outline, some px beside the render's
        # where the head is turned (tbs05, seen from above: the edit's ear sits about 10 px nearer the face). Its
        # lit rim then lies on the render's plain temple skin, where the detail is open to the edit (limit 120): a
        # pale curve 2-3 px wide, +21 to +32 levels, in every frame of the shot (458 row 612: render 144, edit 200,
        # built 165). The same class as the lines noted on ats06, ti620 and rs345 (design log row 120). Skin has
        # no light detail that fine and that strong of its own (pores are +-5): on plain skin the light part of
        # the detail finer than side / 40 px is capped at skin_ridge levels. Eyes, lips and teeth are not plain
        # skin and keep their glints; broader highlights (nose, cheekbone) are below that band.
        Rl = (R @ LUMA).astype(np.float32)
        fine = Rl - low(Rl, max(2, side // 40))
        R = R - (plain(tin, skin, None) * np.maximum(fine - skin_ridge, 0))[..., None]
    if skin_agree > 0 and skin is not None and b:
        # Shapes are never ADDED on skin. skin_ridge (above) halves the line on tbs05's temple and no more: it is
        # not a thin ridge but the edit's lit skin beside its ear, 9 px wide, on a place where the render has the
        # shadow beside ITS ear (458 row 596: render 109-117, edit 160-177, built 135-146 between flanks of 113 and
        # 119). What tells it from the edit's pores, freckles and shading is that no local change of tone maps the
        # render onto it: agreement() (built for whole-frame edits) fits the render's luma to the edit's in a
        # window and measures what is left. On skin the detail is taken in full where that leftover is under
        # skin_agree levels and not at all from twice that; the window follows the head (24 px at 480).
        # On PLAIN skin only (plain() of the render). On all skin (first version, build e6t) the rule also took the
        # edit's eyes, brows, lids and mouth, whose shapes never match the render's to 9 levels and which carry the
        # look: tbs01 50 and tbs04 364 came out with flat, pale eye sockets and faint brows (look/p6/e6_tbs0104.jpg).
        ag = agreement(teacher, tin, max(8, side // 20), skin_agree)
        ag = 1 - plain(tin, skin, None) * (1 - ag)
        # ... and in its place comes the smooth part of what the agreeing skin around it received (see plain_fill in
        # compose): a patch left at the render's level inside skin the edit made 25 levels darker is a pale patch.
        sk = np.clip(skin, 0, 1)
        keep = 1 - sk * (1 - ag)
        rf = max(3, hb // 12)
        wgt = (sk * ag).astype(np.float32)
        den = low(wgt, 2 * rf)
        Re = low(R.astype(np.float32) * wgt[..., None], 2 * rf) / (den + 1e-3)[..., None]  # per channel: the edit
        # changes the skin's colour too.
        # Only LIGHT shapes are taken out: where the detail is lighter than that smooth part (from 2 levels, in
        # full from 6). With both signs (second version, build f2) the rule removed the edit's brows where the
        # render's are thin and pale, plain skin by colour (tbs04 364: thin render brows in place of the edit's,
        # look/p6/f2_tbs04_zoom.png). Darker shapes on plain skin are brows, creases and shading, which carry the
        # look; the ear band is light.
        g = (1 - keep) * np.clip(4 * den, 0, 1) * np.clip(((R - Re) @ LUMA - 2.0) / 4.0, 0, 1)
        R = R + g[..., None] * (Re - R)
        if guard_out is not None:
            guard_out["skin_agree"] = 1 - g
    if mask is not None and skin is not None:
        zone, fh = beard_zone(mask)
        if zone is not None:
            R = speck_free(R, zone * skin, max(1, int(round(fh / 130.0))))
    if grain > 0 and skin is not None and side > 0:
        R = R + skin_grain(tin, R, skin, grain, side / 480.0, hi=grain_hi)[..., None]
    if strands > 0 and hair is not None and side > 0:
        # not on the fringe over the forehead (hair weight on the renderer's face mask): skin shows through there,
        # and strands drew wood-grain lines on it (key-still scans U-X)
        hs = hair if mask is None else hair * (mask[..., 0] <= 127)
        R = R + hair_strands(tin, hs, strands, side / 480.0)[..., None]
    # w: the head (skin and hair inside the head crop's box, grown a little). The head crop's detail is used there
    # only, with or without a whole-frame edit: on the rest of its box it showed as a rectangle on the wall, and the
    # feathered paste of the crop left a faint frame along the box's border (lump map, at140 170, head detail alone).
    w = wh = None
    if b and skin is not None and hair is not None:
        x0, y0, x1, y1 = b
        hard = np.zeros(tin.shape[:2], np.float32)
        hard[y0:y1, x0:x1] = (skin + hair)[y0:y1, x0:x1] > 0.3
        w = np.clip(low(hard, max(2, side // 32)) * 2.5, 0, 1)
        # Outside the head itself the head edit's detail may only darken, and only close by. Its outline is crisper
        # than the render's, so the difference drew a light line 1-3 px outside jaw, ear and hair and lifted the
        # wall for 8 px (+2 to +4 levels in x13): the "light
        # halo round head, neck and shoulders" all three judges of batch Q named. Light detail ends at the
        # renderer's own outline; dark detail (strands over the wall) reaches side / 64 px; the whole-frame detail
        # still takes over at side / 32.
        true = hard
        if mask is not None:
            true = np.zeros(tin.shape[:2], np.float32)
            true[y0:y1, x0:x1] = ((mask[..., 0] > 127) | (mask[..., 2] > 127))[y0:y1, x0:x1]
        lightw = np.minimum(np.clip(low(true, 1) * 2, 0, 1), w)
        darkw = np.clip(low(hard, max(2, side // 64)) * 2.5, 0, 1)
        wh = np.where(R @ LUMA > 0, lightw, darkw)
    if full is not None:
        # full_hp: the whole-frame detail's own band radius (0 = hp). With the head's 16 px band the edit's
        # 8-32 px mottling came along and read as lumps on smooth walls, floors and cloth (x11, lump 1.1-2.3;
        # head detail alone 0.4-0.7): a smooth surface gets grain and weave, not new blobs. Key
        # stills, lump at 16 / 8 / 4 px and at 4 px with gain 0.6: at140 2.24 / 2.03 / 1.51 / 1.02,
        # rs345 1.59 / 1.34 / 0.94 / 0.67, zd400 1.30 / 1.12 / 0.80 / 0.58.
        Rf = residual(full, tin, full_hp or hp, edge, soft, hair, hair_hp, skin, split, agree)
        # on a close-up (head box above ref px) the whole-frame detail shrinks like the skin budget does: the same
        # weave is drawn larger there and counted as lumps on the sweater (pty332: 0.63 at gain 0.6)
        kf = min(1.0, ref / side) if ref > 0 and side > 0 else 1.0
        Rf = Rf * (full_gain * kf * (1 - text_zone(tin)))[..., None]
        # the whole-frame edit gives texture, not colour: only the lightness of its detail is used. The edit pushes
        # saturated things further (shirt print, Oklab chroma at the letters: render 31.8, edit 39.4-44), and its
        # detail put back what the tone step had taken off (tone 23.1 -> with the detail 29.4-33.9; at140 170,
        # rs345 375 the same), while a camera look goes the other way.
        Rf = np.repeat((Rf @ LUMA)[..., None], 3, -1).astype(np.float32)
        if w is None:
            w = np.zeros(tin.shape[:2], np.float32)
            if b:
                x0, y0, x1, y1 = b
                w[y0:y1, x0:x1] = 1
                w = low(w, max(2, side // 48))
            wh = w
        R = wh[..., None] * R + (1 - w)[..., None] * Rf
    elif wh is not None:
        R = wh[..., None] * R
    return R, hb, k, kh


def along_flow(R, y, w, sigma):
    """R averaged along the strand direction of the picture y, inside region w (0..1), over w's pixels only.
    The direction is the structure tensor's axis of least change (Bigun & Granlund 1987; flow-guided smoothing as in
    Kang, Lee & Chui 2009, "Flow-based image abstraction"): detail that runs along the render's strands passes,
    detail that crosses them averages out. Used on hair: the head edit redraws a combed fringe as wavy locks, and
    the outlines of those locks, laid over the render's straight hair, read as a net of dark curved lines (zoomed
    stills zd400 455, rs345 350, x16). The hair's shape is the render's; the edit supplies strand contrast.
    sigma (px) is for a head whose hair covers 200 x 200 px and scales with the square root of the hair's area."""
    ys, xs = np.nonzero(w > 0.02)
    if len(ys) < 50:
        return R
    sigma = float(np.clip(sigma * np.sqrt(float((w > 0.5).sum())) / 200.0, 1.5, 12.0))
    K = int(np.ceil(2 * sigma))
    y0, y1 = max(int(ys.min()) - K - 2, 0), min(int(ys.max()) + K + 3, R.shape[0])
    x0, x1 = max(int(xs.min()) - K - 2, 0), min(int(xs.max()) + K + 3, R.shape[1])
    gy, gx = np.gradient(gauss(y[y0:y1, x0:x1], 1.0))
    ts = 1.5 * sigma
    jxx, jxy, jyy = gauss(gx * gx, ts), gauss(gx * gy, ts), gauss(gy * gy, ts)
    phi = 0.5 * np.arctan2(2 * jxy, jxx - jyy)  # direction of strongest change; the strand runs across it
    tx, ty = -np.sin(phi), np.cos(phi)
    m = w[y0:y1, x0:x1]
    Rc = R[y0:y1, x0:x1] * m[..., None]
    H, W = m.shape
    gyi, gxi = np.mgrid[0:H, 0:W].astype(np.float32)
    acc, den = np.zeros_like(Rc), np.zeros_like(m)
    for k in range(-K, K + 1):
        wk = float(np.exp(-0.5 * (k / sigma) ** 2))
        fx, fy = np.clip(gxi + k * tx, 0, W - 1.001), np.clip(gyi + k * ty, 0, H - 1.001)
        ix, iy = fx.astype(np.int32), fy.astype(np.int32)
        ax, ay = fx - ix, fy - iy
        for dy, dx, wt in ((0, 0, (1 - ax) * (1 - ay)), (0, 1, ax * (1 - ay)), (1, 0, (1 - ax) * ay), (1, 1, ax * ay)):
            acc += (wk * wt)[..., None] * Rc[iy + dy, ix + dx]
            den += wk * wt * m[iy + dy, ix + dx]
    out = R.copy()
    out[y0:y1, x0:x1] = acc / np.maximum(den, 1e-3)[..., None]
    return out


def hair_strands(tin, hair, amp, scale, seed=11):
    """Fine strands that follow the render's own hair direction -> (H, W) levels to add to R (once per keyframe;
    carried with R, so they sit on the hair). White noise is averaged along the streamlines of the render's strand
    direction (line integral convolution: Cabral & Leedom 1993, "Imaging vector fields using line integral
    convolution"), which gives streaks one pixel wide and some 10-20 px long, as thin lit and shadowed strands lie
    on a head of hair. amp: their RMS in % of the local mean lightness. Why: the head edit's own strands belong to
    its wavy locks and cross the render's combed hair; filtered to the render's direction (along_flow) the hair is
    calm but has lost its strand band (key stills, band under 1 px: 5.9 -> 3.2 levels, the render's own 3.0)."""
    ys, xs = np.nonzero(hair > 0.02)
    out = np.zeros(hair.shape, np.float32)
    if len(ys) < 50:
        return out
    s = max(0.5, scale)
    K = int(round(12 * s))
    y0, y1 = max(int(ys.min()) - K - 2, 0), min(int(ys.max()) + K + 3, hair.shape[0])
    x0, x1 = max(int(xs.min()) - K - 2, 0), min(int(xs.max()) + K + 3, hair.shape[1])
    y = (tin @ LUMA)[y0:y1, x0:x1]
    gy, gx = np.gradient(gauss(y, 1.0))
    ts = 5.0 * s
    jxx, jxy, jyy = gauss(gx * gx, ts), gauss(gx * gy, ts), gauss(gy * gy, ts)
    phi = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    tx, ty = -np.sin(phi), np.cos(phi)
    H, W = y.shape
    n = np.random.default_rng(seed).standard_normal((H, W)).astype(np.float32)
    gyi, gxi = np.mgrid[0:H, 0:W].astype(np.float32)
    acc = n.copy()
    wsum = 1.0
    for sign in (1.0, -1.0):
        px, py = gxi.copy(), gyi.copy()
        dx, dy = sign * tx, sign * ty
        for k in range(1, K + 1):
            px, py = np.clip(px + dx, 0, W - 1), np.clip(py + dy, 0, H - 1)
            ix, iy = np.rint(px).astype(np.int32), np.rint(py).astype(np.int32)
            # the direction field has no sign: keep walking the way we came
            ndx, ndy = tx[iy, ix], ty[iy, ix]
            flip = np.where(ndx * dx + ndy * dy < 0, -1.0, 1.0)
            dx, dy = flip * ndx, flip * ndy
            wk = float(np.exp(-0.5 * (k / (0.5 * K)) ** 2))
            acc += wk * n[iy, ix]
            wsum += wk
    acc = gauss(acc / wsum, 0.5)
    acc = acc - gauss(acc, 2.0 * s)
    acc = acc / max(float(acc[hair[y0:y1, x0:x1] > 0.5].std()) if (hair[y0:y1, x0:x1] > 0.5).any() else 1.0, 1e-6)
    out[y0:y1, x0:x1] = hair[y0:y1, x0:x1] * amp / 100.0 * (gauss(y, 4.0 * s) + 1.0) * acc
    return out


def low_in(D, m, r):
    """Local mean of D taken only over the part of the window that lies in region m (0..1): normalised convolution."""
    return low(D * m[..., None], r) / np.maximum(low(m, r), 1e-3)[..., None]


def outline_fade(mask, v=1.0):
    """0..1 map: the share of the added detail a pixel keeps; 0 on the hero's SKIN outline (jaw, ear, neck, hands
    against the room or the collar), back to 1 about r px on either side (r = v x face width / 150, at least 1).
    mask: the renderer's Region mask (R face, G hands, B hair), whose values are fractional on the outline.
    Why: the outline's pixels are mixtures the renderer has already anti-aliased. Every detail layer treated them
    by a hard rule (light detail ended at the mask's 50 % contour, the regions' weights stop there, the frame's own
    raised detail lifted the rim light up to that contour), and the sum was a stair-stepped jaw with a dark 1 px
    line outside it (exact zooms look/p3/jaw_s1.png, 2026-11-21: the tone-only frame has a smooth ramp, x16 and the
    stills after it do not). Nothing is added on the outline now.
    The hair's outline is left out: strands that cross it are wanted, and a gap in them would cut them off."""
    if mask is None:
        return None
    m = mask[..., :3].astype(np.float32) / 255.0
    ys, xs = np.nonzero(m[..., 0] > 0.5)
    if len(xs) < 50:
        return None
    r = max(1, int(round(v * (xs.max() - xs.min()) / 150.0)))
    cb = box(m.max(-1), r)
    # zero while the blurred coverage is 0.25-0.75 (the mixed pixels and one more on either side), full from 0.05 /
    # 0.95: with the zero zone at 0.4-0.6 the edit's crisper outline still left dark dots one pixel outside the jaw
    q = np.clip((np.abs(2 * cb - 1) - 0.5) / 0.4, 0, 1)
    sb, hb = box(np.maximum(m[..., 0], m[..., 1]), 2 * r + 1), box(m[..., 2], 2 * r + 1)
    share = sb / (sb + hb + 1e-3)   # 1 where the hero's edge is skin, 0 where it is hair
    return (1 - share * (1 - q)).astype(np.float32)


def residual(teacher, tin, hp=8, edge=0.0, soft=0.0, hair=None, hair_hp=0, skin=None, split=False, agree=0.0):
    """What the teacher added below ~hp px. edge = 0: the plain high band of (teacher - tin). edge > 0 (levels): each
    image's detail is taken over its own edge-preserving base (guided filter, radius hp, eps = edge^2), so a
    teacher whose tone differs from its input's (darker hair, tanned skin) leaves no light or dark fringe along the
    hairline, jaw and collar: a step much larger than `edge` stays in the base and never reaches the residual.
    soft > 0 (px): the result is blurred with a Gaussian of that sigma, which takes out the 1-2 px band. A
    generative teacher fills that band to the pixel limit (board 11-21: 2-7x the render's fine energy on faces);
    on a moving surface it is resampled at a new sub-pixel phase every frame and shimmers.
    hair, hair_hp: inside the hair mask the band is hair_hp wide: hair's realism is clumps and locks (10-40 px), not
    single strands, and a render's hair is a smooth cap at that scale (judge batches I-N: "hair a smooth cap").
    split (needs skin and hair): the low band that is removed is each region's OWN local mean (skin, hair, the rest),
    so a tone difference between teacher and input in one region (much darker hair) does not ring into its
    neighbours as a light halo round the head or a pale band on the forehead under the fringe (judge batch N: "halo
    round head", "black cutout line at hairline")."""
    D = teacher - tin
    hh = max(hair_hp, hp)
    if split and skin is not None and hair is not None:
        rest = np.clip(1 - skin - hair, 0, 1)
        R = (skin[..., None] * (D - low_in(D, skin, hp)) + hair[..., None] * (D - low_in(D, hair, hh)) +
             rest[..., None] * (D - low_in(D, rest, hp)))
    else:
        if edge > 0:
            R = (teacher - guided(teacher, teacher @ LUMA, hp, edge * edge)) - (tin - guided(tin, tin @ LUMA, hp, edge * edge))
        else:
            R = D - low(D, hp)
        if hair is not None and hh > hp:
            R = R + hair[..., None] * ((D - low(D, hh)) - R)
    if agree > 0:  # whole-frame teachers: detail only where the teacher kept the input's shapes (see agreement())
        R = R * agreement(teacher, tin, tol=agree)[..., None]
    return gauss(R, soft) if soft > 0 else R


def compose(toned, teacher, tin, s=1.0, hp=8, skin=None, skin_limit=2.5, skin_mask=0.5, hair=None, hair_gain=1.0,
            limit=6.0, mask=0.5, skin_even=1.0, render=None, base=None, render_gain=1.0, render_hair=0.0, render_hp=None,
            edge=0.0, pre=None, soft=0.0, agc=0.0, hair_hp=0, hair_limit=0.0, split=False, agree=0.0, own=0.0,
            own_limit=10.0, fine_keep=0.0, fine_share=1.0, skin_mid=1.0, fringe=None, hair_flow=0.0, flow_keep=0.0, flow_mix=1.0, hair_mid=1.0,
            sil=None, mid_plain=None, limit_plain=0.0, hair_over=False, skin_low=0.0, skin_low_gain=1.0, fringe_zone=0, plain_zone=0.0,
            hair_keep=0.0, plain_cmin=0.3, keep_zone=None, plain_soft=False, hair_rim=0.0, plain_local=0.0, plain_face=None, plain_io=None,
            plain_fill=False, seam=0):
    """mid_plain (gain) / limit_plain (levels): on PLAIN skin (plain(): skin-coloured pixels away from eyes, lips and
    teeth) the added detail's mid band gets this gain and this limit instead of skin_mid / skin_limit. Judge batch U
    (x19): "waxy, poreless, uniform orange skin" in 25 of 36 answers, with face match at
    0.94-0.99. The band above ~1 px (pores in clusters, freckles, mottling) was passed
    at 0.3 on all skin because at full gain the edit's eyes, brows and creases come along and the face-match score
    falls (key stills, mid 1.0 / limit 12: 0.86-0.99). Cheeks, forehead, nose and neck carry the
    surface, the features carry who it is: the first get the edit's band, the second keep the render's.
    pre: a residual that is already band-limited (gen.carry carries residual() of the keyframe); teacher is then
    unused and may be None.
    own > 0: the tone step darkens skin and hair by scaling, so the render's own texture shrinks with them (x0.72 on
    skin, x0.78 on hair on the medium shots, in every band up to 8 px), and the keyframe detail mostly swaps that
    texture for the edit's inside a fixed limit instead of adding to it: the skin ended at 0.75 of the render's fine
    energy and read as "waxy, pore-less" to all three judges of batch Q. With own = 1 skin and hair keep the detail
    layer the frame had before the tone step: the toned frame's own detail (over each region's own local mean, so the
    outline does not ring) is raised by 1 / (the tone's local darkening). Strong steps (eye line, nostrils, the
    hairline) are not raised beyond own_limit levels. The base layer carries the tone, the detail layer is left
    alone: Durand & Dorsey 2002, "Fast bilateral filtering for the display of high-dynamic-range images". Values
    above 1 raise the layer beyond that (the callers scale own by the small-head factor: on a 90 px face the
    render's own 1-4 px detail is features and aliasing, and raising it cost face match, key still zd400 405:
    0.947 at 2, 0.942 at 2.5).
    fine_keep > 0 (px, Gaussian sigma): the same for the added detail's finest band. The detail is scaled with the
    tone (x0.76 on a face; "texture is never amplified"), which is right for its mid band (shapes; the face-match
    score lives there) and wrong for pores and strands: under sigma the scaling is undone, by fine_share of the way
    (the callers pass the head-size scale, so a close-up, which has the band already, and a far head, where 1-2 px
    are features, get less). skin_mid: gain on the skin's part above sigma (the edit's blotches and freckles), which
    is what moves the face-match score.
    fringe (0..1 map, hair weight x the renderer's face mask): the hair mask also covers the forehead under a thin
    fringe, where skin shows through. There the edit's forehead highlight came through at the hair limit: a pale,
    reddish patch of +21 levels over 13 px (rs345 350, zd400 455, at140 145, x16; +12 on x13). On the fringe the
    added detail obeys the skin limit and the skin's mid-band gain; a cap on the hair's broad band alone (sigma 4,
    8 or 5 levels) left +14 to +17 of it (key-still scan J/K, 2026-11-21)."""
    R = residual(teacher, tin, hp, edge, soft, hair, hair_hp, skin, split, agree) if pre is None else pre
    y = toned @ LUMA
    skin0, hair0 = skin, hair
    if fringe_zone > 0 and fringe is not None and skin is not None and hair is not None:
        # fringe_zone (px): hair OPEN to the edit next to skin that is NOT (hair limit 120, skin at the thin band)
        # needs one rule for the whole strip under the fringe. With `fringe` the strip's own pixels are skin and
        # their hair-only neighbours are not (dots, see hair_over); with hair_over the strip follows the hair and
        # the edit's lighter forehead stands as a pale wedge against the skin below it (at140 170, zd400 447,
        # stills X2 / X4 / X5). Here the fringe map is grown by fringe_zone px and feathered; inside it everything
        # (hair-only pixels too) is skin for limits and gains, and the hair's rules fade in over the feather.
        fz = np.clip(box(maxr(fringe.astype(np.float32), fringe_zone), fringe_zone), 0, 1) * np.clip(hair + skin, 0, 1)
        hair, skin, fringe = hair * (1 - fz), np.clip(np.maximum(skin, fz), 0, 1), None
    if hair_over and skin is not None and hair is not None:
        # hair wins where the masks overlap (the fringe over the forehead; use instead of `fringe`). With the hair
        # opened to the edit (hair limit 60) and the fringe held to the skin's limit and mid-band gain, single
        # fringe pixels (face mask 143 of 255, hair 255) kept the render's dark strand (levels 62 43 33) next to
        # hair-only pixels that took the edit's forehead (101 67 53): a dotted line along the hairline (at140 170,
        # stills U1-U4, measured 2026-11-21). The no-reference keys keep the render's hairline closely enough that
        # the fringe can follow the edit as a whole.
        skin = np.clip(skin - hair, 0, 1)
    # plain_zone (fraction of the band hp): width of the strip round eyes, lips and teeth that stays the render's
    # (plain()'s shrink; 5 px when 0). With a wide band the edit's own features come along where they do not sit on
    # the render's: on the close-up (pty332 356, head 1080 px, y1) the edit's open eye, about 30 px taller than the
    # render's half-closed one, was drawn over the upper lid ("doubled / two-tone irises", three judges, batch AF).
    # The strip has to grow with the head: 0.1 of a band of head / 3.5 is 3% of the head.
    # plain_cmin: chroma (as a fraction of the skin's median) under which skin is not "plain". At 0.3 the rim-lit
    # strip along the temple of the close-up (pty332: 146 139 142 against 139 118 115 beside it) counted as a
    # feature and kept the render's surface while the skin beside it took the edit's 57 px band: a pale tab with
    # the edge of plain() ("temple seam", "forehead patch", batch AF).
    pz = max(5, int(round(plain_zone * hp)))
    # plain_soft: the border of plain skin keeps its width RELATIVE TO THE FACE. 5 px of shrink and 2 px of feather
    # were set on faces of ~200 px. On the close-up (pty332, face 857 px) the border of the rim-lit strip along the
    # temple (not plain: hue 35-65 degrees off, chroma 0.24-0.45 of the median) was measured: render kept on one
    # side, +10 to +15 levels from the edit 12 px further on: the "temple seam" and "forehead patch" of batch AF.
    # ps = sqrt(area of the skin) / 300, at least 1 (1 on the other three clips: nothing changes there).
    pf = 2
    if plain_soft and skin is not None:
        ps = max(1.0, float(np.sqrt((skin > 0.6).sum())) / 300.0)
        pz, pf = int(round(pz * ps)), int(round(2 * ps * ps))
        if hair is not None and ps > 1.2:
            # ... and so does the border between the hair's rules and the skin's: the renderer's hair mask is cut
            # to the pixel, the edit's strands were drawn up to that cut and the render's pale skin began behind
            # it (a hook-shaped tab at the temple of the close-up, still N5)
            rs = int(round(1.5 * ps * ps))
            hair, skin = box(hair.astype(np.float32), rs), box(skin.astype(np.float32), rs)
    low_add = None
    if skin_low > 0 and skin is not None:
        # skin_low (fraction of the band hp): on skin the added detail is split at sigma = skin_low * hp. The part
        # above it (the edit's shading of cheeks, forehead and neck) is added on plain skin at skin_low_gain with
        # no limit; the part below it goes through the limits and gains as before. Key stills 2026-11-21, keys
        # t85: the composite with a 128 px band wins with three judges (batch AB) and the one
        # with a 32 px band loses (batch AA), so the judges read the 32-128 px band; the face-match score
        # is already down at the 32 px band (0.74 on at140 with either) and stays at 0.93-0.98 when only the hair
        # is opened (X2), so the score reads the skin's texture below it.
        Rl = low(R, max(1, int(round(skin_low * hp))))  # three box passes, close to a Gaussian of that sigma
        low_add = (skin_low_gain * plain(toned, skin, None, shrink=pz, cmin=plain_cmin, feather=pf))[..., None] * Rl
        R = R - skin[..., None] * Rl
    g = (low(y, hp) + 4.0) / (low(tin @ LUMA, hp) + 4.0)
    R = np.clip(g, 0.4, 1.0)[..., None] * R  # texture is never amplified, only reduced on darker surfaces
    if hair is not None and hair_flow > 0:
        # hair: only what runs along the render's own strands (see along_flow); the fringe over the forehead is
        # left out, skin shows through there
        hw = hair if fringe is None else np.clip(hair - fringe, 0, 1)
        # flow_keep (sigma, px): the band finer than that passes unfiltered (single strands and frizz may cross)
        Rl = gauss(R, flow_keep) if flow_keep > 0 else R
        R = R + (flow_mix * hw)[..., None] * (along_flow(Rl, y, hw, hair_flow) - Rl)
    if render is not None and base is not None and skin is not None and render_gain > 0:
        # second texture source: what a more detailed look of the same render (pores, strands) shows over the look
        # the toned frame was made from. Brightness only, and only on plain skin and hair.
        R2 = (render - base) @ LUMA
        R2 = R2 - low(R2, render_hp or hp)  # a smaller radius keeps pores and strands and drops look edges
        g2 = np.clip((low(y, hp) + 4.0) / (low(base @ LUMA, hp) + 4.0), 0.4, 1.0)
        R = R + (render_gain * plain(toned, skin, hair, render_hair) * g2 * R2)[..., None]
    # soft limit: base levels + a multiple of the toned frame's own local contrast (contrast masking). Isolated
    # high-contrast marks on smooth surfaces (stubble specks on skin, dot patterns in hair and cloth) are
    # compressed; detail where the render already has detail passes.
    local = box(np.sqrt(np.maximum(box(y * y, 3) - box(y, 3) ** 2, 0)), 2)
    lim = limit + mask * local
    if skin is not None and skin_limit > 0:
        lim = lim + skin * ((skin_limit + skin_mask * local) - lim)
    if hair is not None and hair_limit > 0:
        lim = lim + hair * ((hair_limit + mask * local) - lim)
    if fringe is not None and skin_limit > 0:
        lim = lim + fringe * ((skin_limit + skin_mask * local) - lim)
    pl = plain(toned, skin, None, shrink=pz, cmin=plain_cmin, feather=pf) if skin is not None and (mid_plain is not None or limit_plain > 0) else None
    if pl is not None and plain_local > 0:
        # plain_local: the local-colour rule of plain() takes over only where the frame-wide rule collapses: under
        # 0.30 of the face plain (0.15: fully local; plain_face = the renderer's face mask, else all skin). The
        # frame-wide rule finds 0.54-0.75 on seven of the eight gate clips, which therefore stay exactly as built.
        fz = (skin > 0.6) if plain_face is None else (skin > 0.6) & (plain_face > 0.6)
        u = min(1.0, max(0.0, (0.30 - float(pl[fz].mean())) / 0.15)) if fz.sum() > 500 else 0.0
        if u > 0:
            # plain_io (dict, from gen.carry): the local rule's map is made ONCE, on the keyframe, and carried to
            # the other frames with the detail itself ("given"). Made afresh on every frame it changed from frame
            # to frame in a dark, noisy picture (the shrink turns each failing pixel into a hole), and the limits
            # it sets switched with it: ti620 flicker 1.01 (y3) -> 1.29 (y4, the rule per frame).
            if plain_io is not None and plain_io.get("given") is not None:
                pln = plain_io["given"]
            else:
                pln = plain(toned, skin, None, shrink=pz, cmin=plain_cmin, feather=pf, local=plain_local)
                if plain_io is not None:
                    plain_io["made"] = pln
            pl = pl + u * (pln - pl)
    if pl is not None and limit_plain > 0:
        lim = lim + pl * ((limit_plain + skin_mask * local) - lim)
    lim = lim[..., None]
    if agc > 0:
        # Strong detail (a generative teacher's hair strands, +-40 levels) is turned DOWN to the allowed strength,
        # not clipped at it: tanh at the limit squares such detail off into two flat levels, which reads as
        # blocky scribbles (judge batch N: "scribbled white hair strands", "chunky hair edge"). The zone's local
        # RMS is brought to agc x the limit by a smooth gain; only peaks beyond 3x that are rounded off.
        rms = np.sqrt(box(box((R @ LUMA) ** 2, 4), 4))[..., None]
        R = R * (agc * lim / np.maximum(agc * lim, rms))
        R = 3 * agc * lim * np.tanh(R / (3 * agc * lim))
    else:
        R = lim * np.tanh(R / lim)
    if skin is not None and skin_even > 0:
        # No stubble: the teacher works on the v9d look, which painted stubble, and its texture comes out stronger
        # on chin and jaw than on the cheeks (board 11-19: chin/cheek fine energy 1.2-1.4x the native render's).
        # On skin the added texture's local strength (e) is held to a fixed multiple of the fine texture the toned
        # frame already has there (F, zone level): the frame's own median e / F. No skin zone gets more texture,
        # relative to what the render shows, than the typical one, so the jaw keeps the render's relation to the
        # cheeks; eyes, lips and edges have a high F and pass.
        rl = R @ LUMA
        rl = rl - box(rl, 1)  # the finest band (1-2 px), where speckle lives and what the board's fine ratios measure
        e = np.sqrt(box(rl * rl, 4))
        fy = y - box(y, 1)
        F = np.sqrt(box(fy * fy, 8)) + 0.3
        # only skin that received detail sets the typical level: a head-crop teacher adds nothing to arms and hands,
        # and their zeros pulled the median to ~0, which then removed the head's detail too (found 2026-11-20)
        sel = (skin > 0.5) & (e > 0.05)
        if sel.sum() > 500:
            cap = skin_even * float(np.median((e / F)[sel])) * F
            g2 = box(np.minimum(1.0, cap / np.maximum(e, 1e-3)), 2)
            R = R * (1 + skin[..., None] * (g2[..., None] - 1))
    if hair is not None and hair_gain != 1.0:
        R = R * (1 + (hair_gain - 1) * hair[..., None])
    if fine_keep > 0:
        Rm = gauss(R, fine_keep)
        # skin only: hair reaches the wanted strand band through its limit alone (key stills, <1 px band: 6.2-6.3;
        # with this boost 7.5)
        up_ = fine_share * (1.0 / np.clip(g, 0.5, 1.0) - 1.0) * (skin if skin is not None else 1.0)
        Rf = (R - Rm) * (1 + up_)[..., None]
        if skin is not None and (skin_mid != 1.0 or mid_plain is not None):
            sk = skin if fringe is None else np.maximum(skin, fringe)
            gm = skin_mid if pl is None or mid_plain is None else skin_mid + (mid_plain - skin_mid) * pl
            Rm = Rm * (1 + (gm - 1) * sk)[..., None]
        if hair is not None and hair_mid != 1.0:
            # hair: the band coarser than single strands is where the edit's own locks show as dark gaps between
            # clumps ("leopard pattern", "woven net": three judges, stills H1, zd400 447 / 455); strands pass
            hw = hair if fringe is None else np.clip(hair - fringe, 0, 1)
            Rm = Rm * (1 + (hair_mid - 1) * hw[..., None])
        R = Rm + Rf
    if hair_keep > 0 and hair0 is not None and skin0 is not None:
        # Hair is never removed: the hair's low band may lighten by hair_keep levels at most. The edit draws its own hairline; where it has forehead or temple and the render has hair,
        # open hair took the edit's skin: +28 levels on the fringe of pty332 356 (y1), a pale patch with the mask's
        # edge ("temple seam", "forehead patch", three judges, batch AF). Strands (finer than hp / 12) are not capped.
        # On ALL hair, not only next to the skin (tried, still H1: no change): the renderer's fringe over the
        # forehead is hair for 100 px and more, and the patch is as large.
        Rl = low((R @ LUMA).astype(np.float32), max(3, hp // 12))
        # keep_zone (--keep-fringe): only the hair that lies over the face in the renderer's mask. On all hair the
        # cap gave back the render's hair cap ("moulded hair, no loose strands", batch AG).
        R = R - ((hair0 if keep_zone is None else keep_zone) * np.maximum(Rl - hair_keep, 0))[..., None]
    if hair_rim > 0 and hair0 is not None:
        # The head's outer outline where it is hair: the edit may not brighten it. Against a plain wall a generative
        # edit lights the head's outline like a cut-out, and the border pixels took that in full (held-out clips,
        # 2026-11-22: tb600 630 +45 levels on the hair's border pixels, a white line round the head; hp622 652 +27).
        # The render's own rim light is in the base and stays; darker detail (strands across the outline) is not
        # touched. hair_rim = half width of the band in px for a head up to ~300 px, wider on close-ups.
        sk = np.zeros_like(hair0) if skin0 is None else skin0
        cov = np.maximum(hair0, sk).astype(np.float32)
        rr = max(2, int(round(hair_rim * max(1.0, float(np.sqrt((sk > 0.6).sum())) / 300.0))))
        band = np.clip(maxr(cov, rr) - (1 - maxr(1 - cov, rr)), 0, 1)       # rr px on either side of the outline
        hb, sb = box(hair0.astype(np.float32), 2 * rr + 1), box(sk.astype(np.float32), 2 * rr + 1)
        band = band * hb / (hb + sb + 1e-3)                                  # only where that outline is hair
        R = R - (band * np.maximum(R @ LUMA, 0))[..., None]
    out = toned + s * (R if low_add is None else R + low_add)
    if own > 0 and skin is not None and hair is not None:
        y3 = y[..., None]
        d = skin * (y3 - low_in(y3, skin, hp))[..., 0] + hair * (y3 - low_in(y3, hair, hp))[..., 0]
        # not the finest band: that is where what is left of the render's noise lives, and raising it raises
        # flicker (the tone-only frames carry 0.52-0.67 of face flicker); pores come from the
        # keyframe detail's fine band, which is carried and does not flicker
        d = gauss(own_limit * np.tanh(d / own_limit), 1.0)
        out = out + (own * (1.0 / np.clip(g, 0.5, 1.0) - 1.0) * d)[..., None]
    if sil is not None:  # nothing is added on the hero's skin outline (see outline_fade)
        out = toned + sil[..., None] * (out - toned)
    if plain_fill and pl is not None and skin is not None:
        # The border of plain skin follows the plain skin beside it. Plain skin takes the edit's surface whole
        # (limit_plain, mid_plain), its shading included; plain() is shrunk by 5 px from everything that is not
        # skin-coloured skin, and that border (along the hair, round eyes and lips, inside the outline) is held to
        # the feature limit and keeps the render's level. Where the edit shades the skin by more than a few levels
        # the border shows: tbs05 458, temple 147 -> 125, the 2-3 px beside the sideburn stay at 148-155 between
        # that and hair that follows the edit too: a pale line that traces the hair mask, in every frame (exact
        # zooms look/p6/tbs05_temple_px.png, maps tbs05_temple_maps.png). The same reading fits the line just
        # inside the neck's outline on rs345 and those on ats06 and ti620 (design log row 120).
        # On border pixels that are skin-coloured themselves (plain() with next to no shrink) the change in
        # brightness becomes the mean change of the plain skin within ~3 px (a normalised blur over plain pixels
        # only: smooth shading, none of the edit's features), by the share the pixel is not plain already and only
        # within reach of plain skin. Eyes, lips and teeth are not skin-coloured and stay as they were.
        # (First version, dropped: it replaced the border's own low band, sigma 7, by the plain skin's, sigma 14;
        # on a border 2-3 px wide both averages are mostly the neighbours, and the line moved 5 of 18 levels.)
        # Along the HAIR only (within the shrink + 3 px of the hair mask). On every border of plain skin (first
        # version, build e6t) the ring round eyes and lips took the cheeks' mean change in place of its own, and
        # the edit's shading of lids and eye sockets went with it (tbs01 50, tbs04 364: flat, pale eyes).
        rf = max(2, hp // 30)
        sc = scalp(hair0, np.clip(skin0, 0, 1), max(6, hp // 8)) if hair0 is not None else None
        bb = _bbox(sc > 0.5, pz + 6 + 3 * rf) if sc is not None else None
        if bb is not None:
            q = (slice(bb[0], bb[1]), slice(bb[2], bb[3]))
            near = np.clip(box(maxr(sc[q], pz + 3), 2), 0, 1)
            p0 = plain(toned, skin, None, shrink=1, cmin=plain_cmin, feather=1)[q]
            plq = pl[q].astype(np.float32)
            W = np.clip(p0 - plq, 0, 1).astype(np.float32) * near
            # Per channel: the edit's skin is also more saturated, and a border level in brightness only stays a
            # greyish line (tried on brightness first, tbs05_e5n.png).
            A = (out[q] - toned[q]).astype(np.float32)
            den = low(plq, rf)
            Re = low(A * plq[..., None], rf) / (den + 1e-3)[..., None]
            out[q] = out[q] + (W * np.clip(6 * den, 0, 1))[..., None] * (Re - A)
    if seam > 0 and skin0 is not None and hair0 is not None:
        # No pale seam between hair and skin. On the hair's side of the same border the hair's edge pixels (hair
        # weight 0.5-0.9) end up with NO change where the edit shows skin on them: its lighter value is refused
        # (hair is never removed: the limits and guards cap what brightens hair) and nothing darker is on offer,
        # while the hair behind them and the skin in front of them both follow the edit: tbs05 458 row 604, change
        # -14 -13 | 0 0 | -22 -22 -30 levels, a pale line 2 px wide round the sideburn and under the fringe that
        # plain_fill does not reach (not skin-coloured).
        # Within seam + 1 px of the hair / skin border the change in brightness loses its narrow light ridges: its
        # white top-hat (the change minus its morphological opening of radius `seam`: everything light and
        # narrower than 2 x seam + 1 px). Steps, dark strands and anything wider stay as they are. A ridge 2 px
        # wide or more goes whole, a single light strand keeps 0.4 of itself: the amount is capped at 1.5 x the
        # top-hat of the change smoothed by 1 px (a 1 px line keeps 0.40 of its height under that blur, a 2 px
        # line 0.68). (First version, dropped: the top-hat of the smoothed change alone, in a zone made of the
        # weights' product: it took 9 and 4 of the 18 levels.)
        r = int(seam)
        sc = scalp(hair0, np.clip(skin0, 0, 1), max(6, hp // 8))  # not round the brows
        Z = np.clip(box(maxr(sc, r + 1) * maxr((skin0 > 0.3).astype(np.float32), r + 1), 1), 0, 1)
        bb = _bbox(Z > 0, 3 * r + 3)
        if bb is not None:
            q = (slice(bb[0], bb[1]), slice(bb[2], bb[3]))
            A = (out[q] - toned[q]).astype(np.float32)
            Ab = gauss(A, 1.0)
            for c in range(3):  # per channel, as in plain_fill
                a, ab = np.ascontiguousarray(A[..., c]), np.ascontiguousarray(Ab[..., c])
                th = np.minimum(a - maxr(-maxr(-a, r), r), 1.5 * (ab - maxr(-maxr(-ab, r), r)))
                out[q + (c,)] = out[q + (c,)] - Z[q] * np.maximum(th, 0)
    return out


def main():
    a = sys.argv[1:]
    if len(a) < 4:
        sys.exit(__doc__)
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    s, hp = float(opt("--structure", 1.0)), int(opt("--hp", 8))
    span = tuple(map(int, opt("--frames").split("-"))) if "--frames" in a else None
    wdir, lim, msk = opt("--weights"), float(opt("--skin-limit", 2.5)), float(opt("--skin-mask", 0.5))
    hg = float(opt("--hair-gain", 1.0))
    glim, gmsk = float(opt("--limit", 6.0)), float(opt("--mask", 0.5))
    even = float(opt("--skin-even", 1.0))
    rdirs = (a[a.index("--render") + 1], a[a.index("--render") + 2]) if "--render" in a else None
    rgain, rhair = float(opt("--render-gain", 1.0)), float(opt("--render-hair", 0.0))
    rhp = int(opt("--render-hp", 0)) or None
    edge, soft, rel = float(opt("--edge", 0.0)), float(opt("--soft", 0.0)), float(opt("--hp-rel", 0.0))
    agc, hhp, hlim = float(opt("--agc", 0.0)), int(opt("--hair-hp", 0)), float(opt("--hair-limit", 0.0))
    split, agree, fdir = bool(int(opt("--split", 0))), float(opt("--agree", 0.0)), opt("--full")
    os.makedirs(a[3], exist_ok=True)
    n = 0
    for f in frames_in(a[0]):
        if span and not span[0] <= f <= span[1]:
            continue
        toned, teacher, tin = load(a[0], f), load(a[1], f), load(a[2], f)
        if teacher is None or tin is None:
            continue
        skin = hair = None
        if wdir:
            w = load(wdir, f, "w")
            if w is not None:
                skin, hair = np.clip((w[..., 0] + w[..., 1]) / 255.0, 0, 1), w[..., 2] / 255.0
        rn, rb = (load(rdirs[0], f), load(rdirs[1], f)) if rdirs else (None, None)
        mk = load(opt("--masks"), f, "m") if "--masks" in a else None
        R, hb, k, kh = key_residual(teacher, tin, load(fdir, f) if fdir else None, hp, rel, edge, soft, hair, hhp, skin,
                                    split, agree, float(opt("--hp-ref", 0.0)), int(opt("--full-hp", 0)),
                                    float(opt("--full-gain", 1.0)), mk, float(opt("--skin-grain", 0.0)),
                                    float(opt("--hair-strands", 0.0)), float(opt("--grain-band", 2.0)),
                                    "--feat-guard" in a)
        fr = hair * (mk[..., 0] > 127) if ("--fringe-skin" in a or "--fringe-zone" in a) and mk is not None and hair is not None else None
        kz = hair * (mk[..., 0] > 127) if "--keep-fringe" in a and mk is not None and hair is not None else None
        out = compose(toned, None, tin, s, hb, skin, lim * k, msk * k, hair, hg, glim, gmsk, even,
                      rn, rb, rgain, rhair, rhp, pre=R, agc=agc, hair_limit=(hlim or glim) * kh if kh < 1 else hlim,
                      own=float(opt("--own", 0.0)) * kh, fine_keep=float(opt("--fine-keep", 0.0)), fine_share=k,
                      skin_mid=float(opt("--skin-mid", 1.0)), fringe=fr, hair_over="--hair-over" in a, hair_flow=float(opt("--hair-flow", 0.0)),
                      flow_keep=float(opt("--flow-keep", 0.0)), flow_mix=float(opt("--flow-mix", 1.0)),
                      hair_mid=float(opt("--hair-mid", 1.0)),
                      sil=outline_fade(mk, float(opt("--outline"))) if "--outline" in a else None,
                      mid_plain=float(opt("--skin-mid-plain")) if "--skin-mid-plain" in a else None,
                      limit_plain=float(opt("--skin-limit-plain", 0.0)) * k,
                      skin_low=float(opt("--skin-low", 0.0)), skin_low_gain=float(opt("--skin-low-gain", 1.0)),
                      fringe_zone=int(opt("--fringe-zone", 0)), plain_zone=float(opt("--plain-zone", 0.0)),
                      hair_keep=float(opt("--hair-keep", 0.0)), plain_cmin=float(opt("--plain-cmin", 0.3)), keep_zone=kz,
                      plain_soft="--plain-soft" in a, hair_rim=float(opt("--hair-rim", 0.0)),
                      plain_local=float(opt("--plain-local", 0.0)))
        Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(a[3], f"f_{f:04d}.png"))
        n += 1
    print(f"structure: {n} frames -> {a[3]}; structure {s} hp {hp}")


if __name__ == "__main__":
    main()
