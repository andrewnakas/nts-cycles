"""Tone branch of NR-1 v2 (2026-11-19; the design log rows 20-22).

The reference's lead over our pass is mostly low-frequency tone: skin and hair clearly darker with their colour kept,
saturated prints and irises muted, blacks lifted and highlights rolled off. Our generative pass is built to hold the
render's low frequencies, so that change is made here instead, analytically:

  regions   face / hands / hair from the renderer's Region mask (m_####.png, R/G/B), "rest" = everything else. Skin
            the mask does not cover (neck, forearms) is found by growing the face and hands seeds through pixels of
            the shot's own skin colour that lie at the seed's depth.
  per region, ONE set of constants per shot (so no flicker and no change of shape):
            a monotone CIELAB lightness curve through the region's own 5/25/50/75/95 percentiles, a chroma scale and
            a hue rotation. In the rest of the frame chroma is soft-limited instead of scaled, so saturated prints
            compress and neutral walls stay; emitters are left alone. Eye whites, irises and teeth (face pixels
            with far less red than the skin) lose chroma, warm slightly, and the bright ones are pulled down
            toward the skin; the cool rim light at the face's outline moves toward skin colour.
  target    the frozen look brief (eval/look_brief.py, data/look/brief_v1.json): relative region statistics, no
            pixels (PROVENANCE.md "The look", user's decision 2026-11-06). The curves are "the brief applied to this
            shot's own native statistics". No reference image is opened here.
  strength  the brief's per-frame entries show the change depends on the shot: a face that stands out of a dark room
            is pulled down most (x0.60), a bright room less (x0.76), a close-up least (x0.86). fit-cond fits that from
            the brief and our own native renders (data/look/brief_v1_cond.json, used by apply unless --no-cond); each
            shot's change is the brief's median change times its predicted share.
  knobs     --tone t   0 = off (pixel-exact), 1 = the brief (for this kind of shot), in between or beyond = the same
                       change scaled (in log ratio / degrees)
            --flat f   generic camera-style black lift and highlight roll-off on the whole frame (nts/finish.grade)

Run: python -m nts.tone apply <frames_dir> <masks_dir> <out_dir> [--brief data/look/brief_v1.json] [--tone 1]
         [--flat 0] [--geom <exr_dir>] [--frames a-b] [--cuts f1,f2,...] [--eye 0.4] [--no-grow] [--params out.json]
         [--weights-out <dir>]   (w_####.png: the region weights used, R face+grown skin, G hands, B hair)
         [--workers n]           (frames in parallel: n processes, 0 = one per CPU core; default env NTS_WORKERS,
                                  else 1. The files written are byte-identical whatever n is)
         [--only f1,f2,...]      (write just these frames; the shot's constants still come from the whole shot)
         [--rest-tone 0]         (> 0: the rest of the frame takes this share of the brief, independent of --tone
                                  and of the shot's strength; 1 = the brief's full muting of clothes and props)
         [--hair-tone 1]         (the hair's tone strength relative to the rest: 1.6 with --tone 0.6 gives the hair
                                  about the brief's full change while skin takes 0.6 of it. Untested beyond
                                  stills: above ~1.3 the forehead under a thin fringe shows a patch, because the
                                  hair mask covers skin there)
         [--share-power 1]       (the shot's predicted share is raised to this power: 0.5 halves, in log terms, how
                                  far a close-up or a dark room departs from the median change; chosen by eye)
         [--rim-l 1] [--rim-c 1]  (rim light on the face's outline: share of the pull down to skin lightness, and
                                  share of the skin's saturation it takes on; 1 / 1 = plain skin)
         [--garment-tone 0]      (> 0, needs --geom: the hero's garment takes this share of the hair's lightness
                                  change, as one factor on the whole garment; see garment_weight)
         [--fringe-skin]         (the rule for the pale rim on the hair's outline keeps off pixels the render
                                  marks as skin too: a fringe over the forehead with skin showing through it.
                                  Without it a close-up's whole fringe zone can be painted hair-dark, ats08)
         [--skin-rim]            (the garment's weight keeps off a light surface right beside the skin: the skin's
                                  own rim light, e.g. the far side of a neck lit from behind. Without it the
                                  garment's growth reached it in some frames only: a patch that flickers, ti620)
         [--skin-close]          (thin gaps in the face's grown weight are closed where a wide colour gate
                                  passes: the shadow line under the jaw of a close-up, see weights)
         [--shot-hold]           (--garment-hold, and the same whole-shot vote for the skin's growth beyond the
                                  masks and for the torso zone; needs --geom)
         [--hands-tight]         (the hands' growth beyond their mask, meant for a bare forearm, runs only through
                                  pixels close to the skin's colour, and the 2 px reach past the mask's edge only
                                  onto skin-coloured pixels. Without it the growth took a wood desk under typing
                                  hands, a pink prop, a sleeve lit pink: darkened as skin, in speckles that change
                                  from frame to frame, tbs05. What the wide growth took still counts as "not
                                  garment": it is all that keeps the garment's growth off a desk the sleeves lie on)
         [--garment-hold]        (needs --geom: each surface point of the hero is garment or not for the whole
                                  shot, by the vote of all its frames along the renderer's motion; without it
                                  the growth takes lit sleeve edges, cuffs and collars in some frames only.
                                  The shot is read twice: see garment_votes)
         [--garment-curve 0]     (> 1, e.g. 1.8: that factor is reached only at this multiple of the garment's median
                                  lightness; the median and darker stay as rendered)
     python -m nts.tone fit-cond <brief.json> <refs_dir> <out.json>
     python -m nts.tone smoke
frames f_####.png|jpg, masks m_####.png. --geom (a render_lr folder with Depth) lets the skin growth use depth.
--cuts lists the first frame of each later shot (default: cuts.json in frames_dir, else in --geom, else one shot).
Needs numpy and Pillow (OpenImageIO only with --geom).
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PCT = (5, 25, 50, 75, 95)
REGIONS = ("face", "hands", "hair")
M = np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]], np.float32)  # rgb @ M
MI = np.linalg.inv(M).astype(np.float32)
WHITE = np.array([0.9505, 1.0, 1.089], np.float32)
GROW_DS = 4        # skin growth runs at 1/4 size
GROW_ITERS = 60    # reach in 1/4-size pixels (240 px at full size)
GROW_DEPTH = (0.2, 0.35)  # metres from the seed's depth range: full weight inside, none beyond
HANDS_GATE = (3, 6)  # --hands-tight: the colour gate (squared spreads) of the hands' growth; the face's is (9, 16)
HAIR_FEATHER = 6   # px
EYE_L = 0.82       # lightness factor at tone 1 for bright eye whites and teeth (chroma factor is --eye)
GARM_DEPTH = (1.0, 1.5)   # metres from the face's depth range within which a surface can be the hero's garment
GARM_EDGE = 14.0   # lightness step between neighbours (1/4 size) that the garment's growth does not cross
GARM_DSTEP = 0.05  # log-depth step between neighbours (1/4 size) that it does not cross
SKIN_CLOSE = 0.012  # --skin-close: gaps in the face's weight up to twice this share of the face's height are closed
RIM_NEAR = 0.06    # --skin-rim: reach from the skin, as a share of the head's size
RIM_L = (66.0, 74.0)  # --skin-rim: lightness from which a surface beside the skin counts as the skin's rim light
RIM = 0.8          # share of the way the rim light at the head's outline moves to the surface's own colour at tone 1
OUTLINE = 0.03     # width of the head's outline band (rim light), as a share of sqrt(head area)
FACE_EDGE = 0.07   # width of the face region's edge band that the eye and teeth rule leaves alone
SAT_L0 = 10.0      # saturation = chroma / (L + SAT_L0)
EVEN = (1.15, 1.6)  # skin saturation relative to the skin's median: untouched below, soft-limited toward the top
EMIT_L = (80.0, 95.0)  # lightness range over which the chroma limit fades out (screens, lamps)


# ---- colour (the brief's own CIELAB: eval/look_score.lab + eval/face.lab_ab) -----------------------------------------
def _srgb_lin_lut(dtype):
    """sRGB decode of the 256 8-bit levels, computed with to_lab's own expression in the precision the input would
    get: img / 255.0 stays float32 for float32 input and becomes float64 for uint8 input."""
    c = np.arange(256).astype(dtype) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


_LIN = {np.dtype(np.float32): _srgb_lin_lut(np.float32), np.dtype(np.uint8): _srgb_lin_lut(np.float64)}


def _levels(img):
    """img as uint8 if every value is a whole 8-bit level (frames read from PNG), else None."""
    if img.dtype == np.uint8:
        return img
    if img.dtype != np.float32 or img.size == 0:
        return None
    u = img.astype(np.uint8)
    return u if np.array_equal(u, img) else None


def to_lab(img):
    """8-bit sRGB (H, W, 3) -> (L, a, b). Frames of whole 8-bit levels (uint8, or float32 holding whole levels) take
    the sRGB decode from a 256-entry table made with the same expression at the same precision (bit-identical, a
    fraction of the cost of the power function)."""
    u = _levels(img)
    if u is not None:
        lin = _LIN[img.dtype][u]
    else:
        c = img / 255.0
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)
    xyz = lin @ M / WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return 116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])


def from_lab(L, a, b):
    fy = (L + 16) / 116
    f = np.stack([fy + a / 500, fy, fy - b / 200], -1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * WHITE
    lin = np.clip(xyz @ MI, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055) * 255


# ---- small image helpers -------------------------------------------------------------------------------------------
def box(a, r):
    """Box mean over a (2r+1) square, edges repeated, float64 running sums, float32 out. Slices of the cumsum replace
    the np.take copies of the first version (bit-identical)."""
    for ax in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax, dtype=np.float64)
        n = a.shape[ax]
        hi, lo = [slice(None)] * a.ndim, [slice(None)] * a.ndim
        hi[ax], lo[ax] = slice(2 * r + 1, 2 * r + 1 + n), slice(0, n)
        d = c[tuple(hi)] - c[tuple(lo)]
        d /= (2 * r + 1)
        a = d.astype(np.float32)
    return a


def max3(m):
    """3 x 3 grey dilation, edges repeated. Kept as the original 9-way reduce: a two-pass version can return the
    other sign of a zero on ties, as maxr below."""
    p = np.pad(m, 1, mode="edge")
    h, w = m.shape
    return np.maximum.reduce([p[i:i + h, j:j + w] for i in range(3) for j in range(3)])


def maxr(m, r):
    """Grey dilation over a (2r+1) square, edges repeated. Kept as the original reduce: a faster running maximum
    (van Herk / Gil-Werman, tried 2026-11-24) can return the other sign of a zero on ties, which changed one output
    pixel by one level (at140 f_0167, default options)."""
    for ax in (0, 1):
        pad = [(0, 0)] * m.ndim
        pad[ax] = (r, r)
        p = np.pad(m, pad, mode="edge")
        n = m.shape[ax]
        m = np.maximum.reduce([p[(slice(None),) * ax + (slice(i, i + n),)] for i in range(2 * r + 1)])
    return m


def down(a, k=GROW_DS):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k, *a.shape[2:]).mean((1, 3))


def up(a, hw):
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize((hw[1], hw[0]), Image.BILINEAR))


def frame_path(d, f, stem="f"):
    p = [q for q in glob.glob(os.path.join(d, f"{stem}_{f:04d}.*")) if q[-4:] in (".png", ".jpg")]
    return p[0] if p else None


def load(d, f, stem="f"):
    p = frame_path(d, f, stem)
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) if p else None


def frames_in(d):
    out = set()
    for p in glob.glob(os.path.join(d, "f_*.*")):
        b = os.path.basename(p)
        if b[-4:] in (".png", ".jpg") and b[2:6].isdigit():
            out.add(int(b[2:6]))
    return sorted(out)


# ---- regions -------------------------------------------------------------------------------------------------------
def skin_model(samples):
    """Shot-constant skin colour model from (L, a, b) samples of the seed regions: centre and robust spread."""
    L, a, b = samples
    am, bm = np.median(a), np.median(b)
    sa = max(1.4826 * np.median(np.abs(a - am)), 2.5)
    sb = max(1.4826 * np.median(np.abs(b - bm)), 2.5)
    return dict(a=float(am), b=float(bm), sa=float(sa), sb=float(sb), L=float(np.median(L)),
                lo=float(np.percentile(L, 2) * 0.35), hi=float(min(np.percentile(L, 98) * 1.15, 100.0)))


def skin_gate(L, a, b, model, full, zero):
    """1 where the colour is within sqrt(full) spreads of the shot's skin colour, fading to 0 at sqrt(zero)."""
    m2 = ((a - model["a"]) / model["sa"]) ** 2 + ((b - model["b"]) / model["sb"]) ** 2
    return (np.clip((zero - m2) / (zero - full), 0, 1) * (L > model["lo"]) * (L < model["hi"])).astype(np.float32)


def grow(seed, cand, iters=GROW_ITERS):
    """Growth of seed (0..1) through cand (0..1), both at 1/4 size (morphological reconstruction, limited reach):
    a pixel joins if a neighbour has joined and it is itself a candidate."""
    g = seed.copy()
    for _ in range(iters):
        g = np.maximum(seed, np.minimum(max3(g), cand))
    return g


def chest_box(mask):
    """(x0, y0, x1, y1), head size, chin row, face centre column: the box straight below the chin, from the face and
    hair masks; None without a face."""
    ys, xs = np.nonzero((mask[..., 0] + mask[..., 2]) > 0.5)
    fy, fx = np.nonzero((mask[..., 0] > 0.5) & (mask[..., 2] < 0.5))
    if len(ys) < 50 or len(fy) < 50:
        return None
    h, w = mask.shape[:2]
    side = int(max(ys.max() - ys.min(), xs.max() - xs.min()))
    chin, cx = int(fy.max()), int(np.median(fx))
    return ((max(0, int(cx - 0.5 * side)), min(h, int(chin + 0.15 * side)), min(w, int(cx + 0.5 * side)),
             min(h, int(chin + 0.9 * side))), side, chin, cx)


def _nb_max(X, M):
    """Largest absolute difference to a 4-neighbour that lies in M."""
    out = np.zeros_like(X)
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        out = np.maximum(out, np.abs(X - np.roll(X, sh, ax)) * np.roll(M, sh, ax))
    return out


def garment_weight(L, a, b, skinw, mask, depth, gm, hold=None, out=None):
    """0..1 weight of the hero's garment, or None. The renderer gives no garment mask, so: pixels of the garment's
    own colour (gm, measured on the chest over the shot) on the hero's unmasked surface are seeds; the garment is
    what a growth from them reaches without crossing a lightness edge or a depth step (so the lit shoulder and
    sleeve belong to it; a print, a held object and the wall do not). At full size a pixel counts by its own colour,
    which keeps a coloured print's edge sharp.
    out: a dict that receives the growth's result at 1/4 size ("g") and the cells it could have taken ("on").
    hold: a 1/4-size map that replaces the growth's result (--garment-hold, see garment_votes), or a dict with that
    map as "g" and the torso zone held over the shot as "zone" (--shot-hold)."""
    hz = None
    if isinstance(hold, dict):
        hz, hold = hold.get("zone"), hold.get("g")
    cb = chest_box(mask)
    if cb is None or (mask[..., 0] > 0.5).sum() < 50:
        return None
    _, side, chin, cx = cb
    lo, hi = np.percentile(depth[mask[..., 0] > 0.5], (5, 95))
    on = np.clip((GARM_DEPTH[1] - np.maximum(lo - depth, depth - hi)) / (GARM_DEPTH[1] - GARM_DEPTH[0]), 0, 1)
    yy = np.arange(L.shape[0], dtype=np.float32)[:, None]
    xx = np.arange(L.shape[1], dtype=np.float32)[None, :]
    zone = (np.clip((yy - (chin - 1.0 * side)) / (0.2 * side), 0, 1) * np.clip(((chin + 4.0 * side) - yy) / (0.4 * side), 0, 1)
            * np.clip((2.4 * side - np.abs(xx - cx)) / (0.4 * side), 0, 1))
    if out is not None:
        out["zone"] = down(zone)
    if hz is not None:
        # --shot-hold: the zone hangs on the head's size and the chin's row in THIS frame; a hand or a prop in front
        # of the head moves both (ats07: head size 178 -> 168 px from one frame to the next, the zone's lower edge
        # 50 px up, and a band across the thighs lost the garment's curve: 0.84 -> 1.03). Each surface point takes
        # the mean of its zone values over the shot instead.
        zone = up(hz, L.shape)
    body = on * zone
    Bq = (down(body * (1 - skinw)) > 0.5).astype(np.float32)
    Lq, aq, bq = down(L), down(a), down(b)
    lq = np.log(np.maximum(down(depth), 1e-4))
    like = (np.abs(Lq - gm["L"]) < max(4.0, 3 * (gm["L"] - gm["l25"]))) & (np.hypot(aq - gm["a"], bq - gm["b"]) < 4)
    cand = (Bq * (_nb_max(Lq, Bq) < GARM_EDGE) * (_nb_max(lq, Bq) < GARM_DSTEP)).astype(np.float32)
    rim = None
    if gm.get("skin_rim") and gm["L"] + max(4.0, 3 * (gm["L"] - gm["l25"])) < RIM_L[0]:
        # --skin-rim: a light surface right beside the skin is the skin's own rim light (the far side of a neck lit
        # from behind: bleached, so it fails the skin's colour test and counts as unmasked surface of the hero), not
        # lit cloth. The growth reached it through a link one cell wide that opened and closed from frame to frame,
        # and the garment's curve then took the patch from white to grey and back (ti620: 168 -> 149 or 101 in
        # runs of 7-18 frames; rs345: a strip up the side of the neck; user, A/B round afadaf: "the shadows flicker
        # on the left by the neck"). The growth does not enter such cells and the weight is 0 on them, whatever the
        # growth found: they keep the tone of the rest of the frame. Not for a garment that is light itself.
        n = max(4, int(round(RIM_NEAR * side / GROW_DS)))
        rim = box(maxr((down(skinw) > 0.5).astype(np.float32), n), max(1, n // 3))
        cand = cand * np.clip(1 - 2 * rim * np.clip((Lq - RIM_L[0]) / (RIM_L[1] - RIM_L[0]), 0, 1), 0, 1)
    seed = cand * like
    (x0, y0, x1, y1), k = cb[0], GROW_DS
    chest = np.zeros_like(seed)
    chest[y0 // k:y1 // k, x0 // k:x1 // k] = seed[y0 // k:y1 // k, x0 // k:x1 // k]
    if chest.sum() < 20:
        chest = seed  # the chest is covered: any garment-coloured surface of the hero
    if chest.sum() < 20 and hold is None:
        if out is not None:
            out["g"], out["on"] = np.zeros_like(seed), Bq
        return None

    def reach(g):  # growth to convergence, 12 steps at a time
        for _ in range(40):
            g2 = grow(g, cand, 12)
            if float((g2 - g).sum()) < 0.5:
                break
            g = g2
        return g2

    # From the chest only: other dark things at the hero's depth (a colleague's hair, the floor under the desk) have
    # the garment's colour too. A sleeve that an arm in front of it cuts off from the torso is taken back: garment-
    # coloured surface right beside what was found.
    if chest.sum() >= 20 and (hold is None or out is not None):  # with hold the growth's result is not used
        g = reach(chest)
        g = reach(np.maximum(g, seed * maxr(g, max(1, int(round(0.15 * side / k))))))
        g = maxr(g, 1) * Bq  # the pixels beside an edge failed the edge test themselves
    else:
        g = np.zeros_like(seed)
    if out is not None:
        out["g"], out["on"] = g, Bq
    if hold is not None:
        g = hold * Bq
    own = np.clip((12.0 - np.hypot(a - gm["a"], b - gm["b"])) / 6.0, 0, 1)
    wg = np.clip(up(box(g, 1), L.shape) * 1.3, 0, 1) * body * own
    if rim is not None:
        wg = wg * (1 - np.clip(up(rim, L.shape), 0, 1) * np.clip((L - RIM_L[0]) / (RIM_L[1] - RIM_L[0]), 0, 1))
    return wg


def garment_votes(frames_dir, masks_dir, shot, prm, seq, grow_skin=True, shot_hold=False, workers=1):
    """--garment-hold, pass 1 over one shot: {frame: 1/4-size map, 1 where the surface point there is garment}.
    The growth in garment_weight is a yes/no flood. Where a lit part hangs on the garment by a link one cell wide
    (a lit sleeve edge, a cuff behind the hand, a collar), the link opens and closes from frame to frame and the
    part takes the garment's curve in some frames only: jumps of 8-25 % in runs of 1-40 frames on all 16 clips
    looked at (runs/v2_tone_flips.py, 2026-11-22; user, A/B round afadaf: "the shadows flicker"). Ramps in place
    of the yes/no tests were tried and dropped: they halve the jump and spread a faint garment weight over desk
    and props. Here every frame's flood is kept as it is and each surface point takes the vote of the whole shot:
    it is followed through the shot along the renderer's motion (positions are carried, not pictures, so nothing
    blurs with the number of frames), a frame votes where the point is on the hero's unmasked surface and the
    depth agrees from one frame to the next, and the point is garment in every frame if more than half of the
    votes say so. Whether a point is cloth cannot change inside a shot.
    shot_hold (--shot-hold): along the same paths, the same vote for the skin's growth beyond the renderer's masks
    (face, hands; every frame the point is followed to votes) and the mean of garment_weight's torso zone. The
    result is then a dict per frame: {"g", "zone", "face", "hands"}."""
    gp = prm.get("garment")
    if seq is None or not (gp or shot_hold):
        return None
    g, on, mvp, mvn, dep, sk, zn = {}, {}, {}, {}, {}, {}, {}
    ctx = (frames_dir, masks_dir, prm, grow_skin, shot_hold, seq.src)
    from nts.par import pmap
    # pass 1, frame by frame (independent: in parallel with --workers)
    for f, r in zip(shot, pmap(_vote_inputs, [(ctx, f) for f in shot], workers)):
        if r is None:
            continue
        g[f], on[f], mvp[f], mvn[f], dep[f] = r["g"], r["on"], r["mvp"], r["mvn"], r["dep"]
        if shot_hold:
            sk[f], zn[f] = r["sk"], r["zn"]
    if not g:
        return None
    # the vote: every frame follows each point through the whole shot (also independent per frame)
    held = dict(zip(list(g), pmap(_vote_one, list(g), workers, _vote_init, ((g, on, mvp, mvn, dep, sk, zn, shot_hold),))))
    return held


_SEQ = {}


def _seq_of(src):
    """Sequence of a geometry folder, one per process."""
    if src not in _SEQ:
        from nts.io_exr import Sequence
        _SEQ[src] = Sequence(src)
    return _SEQ[src]


def _vote_inputs(task):
    """garment_votes' pass 1 for one frame: the garment's growth, the skin's growth (shot_hold) and the motion and
    depth at 1/4 size; None if the frame or its geometry is missing."""
    (frames_dir, masks_dir, prm, grow_skin, shot_hold, src), f = task
    from nts import warp as W
    seq = _seq_of(src)
    gp = prm.get("garment")
    k = GROW_DS
    img = load(frames_dir, f)
    if img is None or f not in seq.files:
        return None
    m = load(masks_dir, f, "m")
    m = np.zeros_like(img) if m is None else m / 255.0
    fr = seq.read(f)
    depth = W.upsample_nearest(fr.depth, seq.scale, img.shape[:2])
    L, a, b = to_lab(img)
    ow = {} if shot_hold else None
    w = weights(m, (L, a, b) if grow_skin else None, prm.get("skin_model") if grow_skin else None, depth, None, ow)
    wr = 1 - (w["face"] + w["hands"] + w["hair"])
    o = {}
    if gp:
        garment_weight(L, a, b, w.get("skin_g", np.clip(1 - wr, 0, 1)), m, depth, gp, None, o)
    hw = (img.shape[0] // k, img.shape[1] // k)
    r = {"g": o.get("g", np.zeros(hw, np.float32)), "on": o.get("on", np.zeros(hw, np.float32))}
    if shot_hold:
        r["sk"] = {q: ow[q].astype(np.float16) if q in ow else np.zeros(hw, np.float16) for q in ("face", "hands", "block")}
        r["zn"] = o["zone"].astype(np.float16) if "zone" in o else None
    sc = seq.scale / k
    nx = getattr(fr, "mv_next", None)
    r["mvp"] = W.hr_motion(fr.mv, fr.depth, sc, hw).astype(np.float32)
    r["mvn"] = None if nx is None else W.hr_motion(nx, fr.depth, sc, hw).astype(np.float32)
    r["dep"] = W.upsample_nearest(fr.depth, sc, hw).astype(np.float32)
    return r


_VOTE = None


def _vote_init(state):
    global _VOTE
    _VOTE = state


def _vote_one(t):
    """The whole shot's vote for the points of frame t (garment_votes)."""
    g, on, mvp, mvn, dep, sk, zn, shot_hold = _VOTE
    h, wd = next(iter(g.values())).shape
    ys, xs = np.mgrid[0:h, 0:wd].astype(np.float32)

    def cell(px, py):
        return np.clip(np.rint(py), 0, h - 1).astype(np.intp), np.clip(np.rint(px), 0, wd - 1).astype(np.intp)

    num, den = g[t] * on[t], on[t].copy()
    if shot_hold:
        sn = {q: sk[t][q].astype(np.float32) for q in ("face", "hands", "block")}
        cnt = np.ones((h, wd), np.float32)
        zs, zc = (None if zn[t] is None else zn[t].astype(np.float32)), np.ones((h, wd), np.float32)
    for mv, step in ((mvp, -1), (mvn, 1)):
        px, py, ok, d0, cur = xs, ys, np.ones((h, wd), bool), dep[t], t
        while cur + step in g and mv[cur] is not None and ok.any():
            mo = mv[cur][cell(px, py)]
            px, py = px + mo[..., 0], py + mo[..., 1]
            cur += step
            c = cell(px, py)
            d1 = dep[cur][c]
            ok = ok & (px > -0.5) & (px < wd - 0.5) & (py > -0.5) & (py < h - 0.5) & (np.abs(d1 - d0) < 0.05 * d0)
            d0 = d1
            v = ok * on[cur][c]
            num, den = num + v * g[cur][c], den + v
            if shot_hold:
                okf = ok.astype(np.float32)
                cnt += okf
                for q in sn:
                    sn[q] += okf * sk[cur][q][c]
                if zs is not None and zn[cur] is not None:
                    zs += okf * zn[cur][c]
                    zc += okf
    hg = np.clip((num / np.maximum(den, 1e-3) - 0.4) / 0.2, 0, 1).astype(np.float32)
    if shot_hold:
        held = {"g": hg, "zone": None if zs is None else zs / zc}
        # What the garment's growth took in a fair share of ALL the frames the point was followed to is cloth,
        # not skin: dark cloth in warm light has the colour of dark skin (ats04, a sleeve lit red from below:
        # the hand's growth ran down it in most frames, and the pixels' own colour test then switched the
        # skin's curve on and off, 0.83 or 1.03). From a fifth of the frames for the hands' growth (a bare
        # forearm is never the garment's: a lightness edge and the skin's own weight keep the growth off it),
        # from half for the face's (the neck must stay skin).
        share = num / cnt
        # "block": the hands' growth by the wide colour gate (--hands-tight, see weights)
        for q, lim in (("face", 0.5), ("hands", 0.2), ("block", 0.2)):  # a wider ramp than the garment's: this growth is soft
            held[q] = (np.clip((sn[q] / cnt - 0.25) / 0.5, 0, 1)
                       * (1 - np.clip((share - lim) / 0.15, 0, 1))).astype(np.float32)
        return held
    return hg


def edge_alpha(X, W, m0, far, reach=2, win=4):
    """The share of skin in the outline pixels behind the hero (far), within reach px of solid skin weight W; and
    the solid skin pixels themselves (floats 0/1). The renderer's depth has one sample per pixel, so the weights end
    on a staircase; the picture does not: its outline pixels are part skin, part background. A pixel's share is its
    colour's place between the skin just inside (F, local mean) and the background (B): the matting estimate for
    known F and B (Smith & Blinn 1996). B is the nearby background pixel closest in colour to it, not a local mean:
    where the background has an edge of its own behind the jaw (wall over a white bench, rs345 350) a mean of both
    gave pure wall pixels a share of 0.7-1 and others 0, a jagged dark line (x16, 2026-11-21). A pixel that equals
    some background pixel nearby gets 0 whatever the skin's colour. Never less than the mask's own coverage; where
    skin and background are within 6 dE the mask's coverage alone."""
    solid = ((W > 0.9) & ~far).astype(np.float32)
    zone = far & (maxr(solid, reach) > 0.5)
    z = np.zeros(W.shape, np.float32)
    if not zone.any():
        return z, solid
    ys, xs = np.nonzero(zone)
    y0, y1 = max(0, ys.min() - 2 * win), min(W.shape[0], ys.max() + 2 * win + 1)
    x0, x1 = max(0, xs.min() - 2 * win), min(W.shape[1], xs.max() + 2 * win + 1)
    Xc, sc, mc, fc, zc = X[y0:y1, x0:x1], solid[y0:y1, x0:x1], m0[y0:y1, x0:x1], far[y0:y1, x0:x1], zone[y0:y1, x0:x1]
    bg = fc & (maxr(sc, reach + 1) < 0.5) & (mc < 0.02)
    # F from the nearest solid pixels (5 x 5; 7 x 7 only where that holds none): a wider mean reaches past the lit
    # outline into the darker cheek, takes F for darker than it is at the edge and the pixel's share for larger (rs345
    # 350 row 524: wall 167, outline pixels 190 / 217 -> 133 / 140, a dark pixel OUTSIDE a lighter one)
    d2, d3 = box(sc, 2), box(sc, 3)
    dn = np.where(d2 > 0.02, d2, d3)
    F = np.where((d2 > 0.02)[..., None], box(Xc * sc[..., None], 2) / np.maximum(d2, 1e-4)[..., None],
                 box(Xc * sc[..., None], 3) / np.maximum(d3, 1e-4)[..., None])
    best = np.full(sc.shape, np.inf, np.float32)
    B = Xc.copy()
    for dy in range(-win, win + 1):
        for dx in range(-win, win + 1):
            if dy == 0 and dx == 0:
                continue
            Bq = np.roll(Xc, (dy, dx), (0, 1))
            d = np.where(np.roll(bg, (dy, dx), (0, 1)), ((Xc - Bq) ** 2).sum(-1), np.inf).astype(np.float32)
            u = zc & (d < best)
            best[u] = d[u]
            B[u] = Bq[u]
    FB = F - B
    den = (FB ** 2).sum(-1)
    al = np.clip(((Xc - B) * FB).sum(-1) / np.maximum(den, 1e-4), 0, 1)
    al = np.where(np.isfinite(best) & (dn > 0.02) & (den > 36), al, mc)
    z[y0:y1, x0:x1] = np.where(zc, np.maximum(mc, al), 0)
    return z, solid


def weights(mask, lab=None, model=None, depth=None, hold=None, out=None):
    """Region weights for one frame: dict face/hands/hair (H, W) in 0..1, summing to at most 1 (the rest is 'rest').
    mask: m_####.png as floats 0..1 (R face, G hands, B hair). With lab + model the face and hands weights are grown
    over uncovered skin (neck, forearms); depth (H, W, metres) keeps the growth on the hero's own surface.
    out: a dict that receives the growth's result at 1/4 size per region ("face", "hands"; "block" with
    --hands-tight).
    hold: a dict of 1/4-size maps that replace the growth's result (--shot-hold, see garment_votes)."""
    face, hands, hair = mask[..., 0], mask[..., 1], mask[..., 2]
    # the mask's anti-aliased edge leaves a 1-2 px rim of skin untouched, which reads as a glow once the skin is
    # darker: let skin reach 2 px past its edge
    face0, hands0 = face, hands
    face, hands = maxr(face, 2), maxr(hands, 2)
    hands_tight = bool(lab is not None and model is not None and model.get("hands_tight"))
    hands_w = hands  # the hands' weight by the wide rule (see the growth below): kept for the garment's sake
    if hands_tight:
        # --hands-tight: ... but a hand lies ON things (a desk 2 cm under typing fingers is inside the depth range
        # below), and 2 px of a light desk toned as skin are a grey line round the hand (tbs05 422 row 1100: desk
        # 199 -> 169, 178). The reach is for skin the mask's edge left out, so it needs the skin's colour.
        hands = np.maximum(hands0, hands * skin_gate(lab[0], lab[1], lab[2], model, 9, 16))
    far = {}
    if depth is not None and lab is not None:
        # ... but only on the hero's own surface. Behind it (beyond the region's depth range) the reach was 2 px of
        # WALL at full weight: toned like skin, a grey line outside ear and jaw against a light wall (rs345 350 row
        # 497: wall 195 -> 148 and 173, x13; batch Q: "halo round head and ear"). There the weight is 0, and an
        # outline pixel that is part skin gets its share of the skin's change in apply_frame (edge_alpha).
        for k, m0 in (("face", face0), ("hands", hands0)):
            if (m0 > 0.5).sum() >= 50:
                lo, hi = np.percentile(depth[m0 > 0.5], (5, 95))
                far[k] = np.maximum(lo - depth, depth - hi) > GROW_DEPTH[1]
                # ... and the outline pixel on the hero's side of that staircase which is itself part background
                # (the mask's coverage under 1, or skin the mask does not cover, next to a pixel behind the hero).
                # At full weight it was toned as solid skin: no in-between pixel, a stepped outline (rs345 350 row
                # 516, L: wall 70, outline pixel 81 -> 54, skin 55). It joins the outline
                # pixels and takes its share of the skin's change (edge_alpha).
                far[k] = far[k] | ((m0 < 0.98) & (maxr(far[k].astype(np.float32), 1) > 0.5))
        if "face" in far:
            face = np.where(far["face"], 0.0, face).astype(np.float32)
        if "hands" in far:
            hands = np.where(far["hands"], 0.0, hands).astype(np.float32)
            hands_w = np.where(far["hands"], 0.0, hands_w).astype(np.float32)
    if lab is not None and model is not None:
        L, a, b = lab
        tight = depth is None  # without depth only colour stops the growth, so ask for a closer match
        cand = skin_gate(down(L), down(a), down(b), model, *((4, 9) if tight else (9, 16)))
        gate = skin_gate(L, a, b, model, 9, 16)  # keeps the feather off the shirt and the wall
        # --hands-tight: the hands' growth is there for a bare forearm, which has the hand's own colour (inside the
        # hands mask the squared distance has a median near 1, 75 % under 4). In the 12 clips with hands so far (long
        # sleeves) it found real skin once, a strip of wrist (ats07, distance about 1), and otherwise took what the
        # wide gate lets pass, at 5-10: a wood desk, a pink prop, a sleeve lit pink (runs/v2_hands_scan.py). On
        # mid-grey wood that is a dark patch in speckles which
        # change with every frame (tbs05, a desk shot from above). The flood asks for the closer match; the wide
        # gate still shapes the result at full size, so real skin keeps an even weight.
        # The wide growth is still run ("block") and handed to the garment's growth as skin: on a desk the sleeves
        # lie on, that patch round the hands was the only thing between the garment's flood and the whole desk
        # (tbs05 without it: desk, keyboard, laptop and books took the garment's curve, light grey to dark grey).
        cand_h = skin_gate(down(L), down(a), down(b), model, *HANDS_GATE) if hands_tight else cand
        hw = face.shape
        for k, seed_full in (("face", face), ("hands", hands)) + ((("block", hands_w),) if hands_tight else ()):
            seed = down(seed_full).astype(np.float32)
            if seed.max() < 0.5:
                continue
            c, on = (cand_h if k == "hands" else cand), 1.0
            if depth is not None:
                d = down(depth)
                lo, hi = np.percentile(d[seed > 0.5], (5, 95))
                c = c * np.clip((GROW_DEPTH[1] - np.maximum(lo - d, d - hi)) / (GROW_DEPTH[1] - GROW_DEPTH[0]), 0, 1)
                # the same test at full size on the result: the growth runs at 1/4 size and is blurred, so it
                # reached 4-8 px over the outline, and a wall of about the skin's colour passes the colour gate: it
                # was darkened with the skin, a grey line and a dim aura round ear, jaw and neck against a light
                # wall (rs345 350, x13; three judges of batch Q: "halo round head and ear")
                on = np.clip((GROW_DEPTH[1] - np.maximum(lo - depth, depth - hi)) / (GROW_DEPTH[1] - GROW_DEPTH[0]), 0, 1)
            if hold is not None and hold.get(k) is not None:
                # --shot-hold: whether a surface point is skin is the vote of its whole shot. The growth is a
                # yes/no flood through skin-coloured cells: a dark sleeve end lit red passes the colour test in
                # some frames only, and the hand's weight then runs up the cuff (ats04: tone change 0.77 or 1.03
                # in runs of 2-10 frames).
                gr = np.maximum(hold[k], seed)
            else:
                gr = grow(seed, c.astype(np.float32))
            if out is not None:
                out[k] = gr
            g = up(box(gr, 1), hw) * gate * on
            if k == "face":
                face = np.maximum(face, g)
            elif k == "hands":
                hands = np.maximum(hands, g * (1 - face))
            else:
                hands_w = np.maximum(hands_w, g * (1 - face))
        if model.get("close"):
            # --skin-close: the shadow line under the jaw is darker and redder than the skin model allows, so the
            # colour gate cut a thin band out of the grown weight between the face mask's lower edge and the neck:
            # 8 % lighter than the skin on both sides, and it comes and goes with the jaw (pty332, a close-up:
            # tone change 0.83 or 0.91-0.94 in runs of 3-7 frames, runs/v2_tone_flips.py). A gap in the face's
            # weight that a closing of this size bridges is skin if a wide colour gate lets it pass ...
            ys, xs = np.nonzero(face0 > 0.5)
            if len(ys) >= 50:
                r = max(3, int(round(SKIN_CLOSE * (ys.max() - ys.min() + 1))))
                ya, xa = np.nonzero(face > 0.5)
                y0, y1, x0, x1 = max(0, ya.min() - r), ya.max() + r + 1, max(0, xa.min() - r), xa.max() + r + 1
                fc = face[y0:y1, x0:x1]
                closed = -maxr(-maxr(fc, r), r)
                # ... and if it is darker than the skin's median: a shadow. A light gap is rim light, which keeps
                # the tone of the rest of the frame (ti620: without this the closing took the edge of the rim-lit
                # patch on the neck, up to 52 levels).
                # Only beside the face mask itself (within 8 % of its height): further out the grown weight has
                # loose ends of its own (ti620: dots along a pink-lit sleeve edge, which the closing joined up).
                near = up(maxr(down((face0 > 0.5).astype(np.float32)), max(1, int(round(0.08 * (ys.max() - ys.min() + 1) / GROW_DS)))),
                          face.shape)[y0:y1, x0:x1]
                wide = (skin_gate(L[y0:y1, x0:x1], a[y0:y1, x0:x1], b[y0:y1, x0:x1], model, 25, 36)
                        * np.clip((model["L"] - L[y0:y1, x0:x1]) / 4.0, 0, 1) * np.clip(2 * near - 0.5, 0, 1))
                face = face.copy()
                face[y0:y1, x0:x1] = np.maximum(fc, closed * wide)
        if "face" in far:  # the growth must not hand the outline pixels their full weight back
            face = np.where(far["face"], 0.0, face).astype(np.float32)
        if "hands" in far:
            hands = np.where(far["hands"], 0.0, hands).astype(np.float32)
            hands_w = np.where(far["hands"], 0.0, hands_w).astype(np.float32)
    # the hair's outer few pixels are see-through in the render, so the full change there draws an outline: fade the
    # hair weight in over HAIR_FEATHER px from its edge
    hair = np.clip(2 * box(box(hair, HAIR_FEATHER // 2), HAIR_FEATHER // 2) - 1, 0, 1)
    edge = src = None
    if far:
        X = np.stack(lab, -1)
        edge, src = np.zeros(face.shape, np.float32), np.zeros(face.shape, np.float32)
        for k, W, m0 in (("face", face, face0), ("hands", hands, hands0)):
            if k in far:
                al, so = edge_alpha(X, W * (1 - hair), m0, far[k])
                edge, src = np.maximum(edge, al * (1 - hair)), np.maximum(src, so)
    face = face * (1 - hair)  # the fringe over the forehead is hair
    s = face + hands + hair
    n = np.where(s > 1, s, 1.0)
    out = {"face": face / n, "hands": hands / n, "hair": hair / n}
    if hands_tight:
        out["skin_g"] = np.clip(face + hands_w + hair, 0, 1)  # what the garment's growth must keep off
    if edge is not None and edge.max() > 0:
        out["edge"], out["edge_src"] = edge, src
    return out


# ---- per-shot constants --------------------------------------------------------------------------------------------
def shot_stats(frames_dir, masks_dir, frames, n=10):
    """Native statistics of one shot, from up to n evenly spaced frames: per region the L percentiles the brief uses,
    plus skin samples for the colour model and rest chroma samples for the soft limit."""
    pick = [frames[i] for i in sorted(set(np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)))]
    acc = {k: [] for k in REGIONS + ("rest",)}
    skin, restc, area, hairc, garm = [], [], [], [], []
    for f in pick:
        img, m = load(frames_dir, f), load(masks_dir, f, "m")
        if img is None:
            continue
        m = np.zeros_like(img) if m is None else m / 255.0
        L, a, b = to_lab(img[::2, ::2])
        m = m[::2, ::2]
        for i, k in enumerate(REGIONS):
            sel = m[..., i] > 0.5
            if k == "face":
                sel &= m[..., 2] < 0.5  # the fringe over the forehead is hair
                area.append(100.0 * sel.mean())
            if sel.sum() >= 50:
                acc[k].append(L[sel])
                if k in ("face", "hands"):
                    skin.append(np.stack([L[sel], a[sel], b[sel]]))
                if k == "hair":
                    pure = sel & (m[..., 0] < 0.5)  # not the fringe over the forehead: skin shows through it
                    if pure.sum() >= 50:
                        hairc.append(np.stack([L[pure], a[pure], b[pure]]))
        rest = m.max(-1) < 0.01
        cb = chest_box(m)
        if cb is not None:
            x0, y0, x1, y1 = cb[0]
            sel = rest[y0:y1, x0:x1]
            if sel.sum() >= 200:
                garm.append(np.stack([L[y0:y1, x0:x1][sel], a[y0:y1, x0:x1][sel], b[y0:y1, x0:x1][sel]])[:, ::3])
        acc["rest"].append(L[rest][::7])
        c = np.hypot(a, b)
        restc.append(np.stack([c[rest][::7], L[rest][::7]]))
    st = {}
    for k, v in acc.items():
        if v:
            st[k] = [float(max(p, 3.0)) for p in np.percentile(np.concatenate(v), PCT)]
    st["face_pct"] = float(np.mean(area)) if area else 0.0
    if hairc:
        h = np.concatenate(hairc, 1)
        # hi: the hair's normal upper lightness, mirrored from its dark side (median + (median - lower quartile)),
        # because a wide pale rim-light band can itself be the upper quartile
        l25, l50 = np.percentile(h[0], (25, 50))
        st["hair_model"] = dict(L=float(l50), hi=float(2 * l50 - l25), a=float(np.median(h[1])),
                                b=float(np.median(h[2])))
    if garm:
        # the hero's garment: what is straight below the chin (a print or a held object there is a minority)
        gsm = np.concatenate(garm, 1)
        l25, l50 = np.percentile(gsm[0], (25, 50))
        st["garment_model"] = dict(L=float(l50), l25=float(l25), a=float(np.median(gsm[1])), b=float(np.median(gsm[2])))
    if skin:
        st["skin_model"] = skin_model(np.concatenate(skin, 1))
    if restc:
        st["_restc"] = np.concatenate(restc, 1)
    return st


def solve_cmax(c, L, ratio):
    """Saturation soft limit sm (saturation = chroma / (L + SAT_L0)) so that mean(limited chroma) / mean(chroma) =
    ratio over the rest of the frame, emitters (bright pixels) excluded from the limit as in apply. Limiting
    saturation instead of chroma also reaches dark, strongly coloured areas (a deep teal backdrop, blue shadows),
    not only bright prints."""
    if ratio >= 0.999 or c.mean() < 1e-3:
        return 1e6
    e = np.clip((L - EMIT_L[0]) / (EMIT_L[1] - EMIT_L[0]), 0, 1)
    sat = c / (L + SAT_L0)
    lo, hi = 0.01, 50.0
    for _ in range(40):
        sm = np.sqrt(lo * hi)
        r = (c * e + c * (sm * np.tanh(sat / sm) / np.maximum(sat, 1e-6)) * (1 - e)).mean() / c.mean()
        lo, hi = (sm, hi) if r < ratio else (lo, sm)
    return float(np.sqrt(lo * hi))


def cond_features(face_l, rest_l, face_pct):
    """What the shot-dependent strength is predicted from: how far the face stands out of the room (CIELAB L
    difference / 100) and how much of a close-up it is (0 below 10% of the frame, 1 from 28%)."""
    return (face_l - rest_l) / 100.0, float(np.clip((face_pct - 10.0) / 18.0, 0, 1))


def shot_strength(stats, cond):
    """Share of the brief's median change this shot gets. The brief's own per-frame entries show the change is not
    one number: a face that stands out of a dark room is pulled down most, a close-up least (fit_cond)."""
    if not cond or "face" not in stats or "rest" not in stats:
        return 1.0
    c, s = cond_features(stats["face"][2], stats["rest"][2], stats.get("face_pct", 0.0))
    return float(np.clip((cond["a"] + cond["b"] * c + cond["c"] * s) / cond["ref"], 0.15, 1.3))


def fit_cond(brief_path, refs, out_path):
    """Least-squares fit of the brief's per-frame face lightness log ratio on cond_features of OUR native render of
    the same frame (refs/native_<episode>_f<frame>.png + refs/m_<episode>_<frame>.png). Reads the brief and our own
    renders only; no reference image is opened."""
    import re
    brief = json.load(open(brief_path))
    X, y, names = [], [], []
    for p in sorted(glob.glob(os.path.join(refs, "native_*_f*.png"))):
        ep, f = re.search(r"native_(.+)_f(\d+)\.png$", p).groups()
        d, mp = brief["per_frame"].get(f"{ep}_f{f}"), os.path.join(refs, f"m_{ep}_{f}.png")
        if d is None or "face.p50" not in d or not os.path.exists(mp):
            continue
        img = np.asarray(Image.open(p).convert("RGB"))[::2, ::2].astype(np.float32)
        m = np.asarray(Image.open(mp).convert("RGB").resize((img.shape[1], img.shape[0]), Image.NEAREST)) / 255.0
        L = to_lab(img)[0]
        face, rest = (m[..., 0] > 0.5) & (m[..., 2] < 0.5), m.max(-1) < 0.01
        if face.sum() < 50:
            continue
        X.append((1.0,) + cond_features(float(np.median(L[face])), float(np.median(L[rest])), 100.0 * face.mean()))
        y.append(d["face.p50"])
        names.append(f"{ep}_f{f}")
    X, y = np.array(X), np.array(y)
    co = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ co
    out = dict(a=float(co[0]), b=float(co[1]), c=float(co[2]), ref=float(brief["median"]["face.p50"]), n=len(y),
               resid_sd=float(e.std()), const_sd=float((y - np.median(y)).std()),
               form="log(face L ratio) = a + b*(native face L - native rest L)/100 + c*clip((face % of frame - 10)/18, 0, 1)",
               source=f"{os.path.basename(brief_path)} per_frame + our native renders in {refs}; no reference image read")
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"tone fit-cond: {len(y)} frames, a {co[0]:.3f} b {co[1]:.3f} c {co[2]:.3f}; residual sd {e.std():.3f} "
          f"(one number for every shot: {out['const_sd']:.3f}) -> {out_path}")
    return out


def shot_params(stats, brief, tone=1.0, eye=0.4, cond=None, share_power=1.0, hair_tone=1.0, rest_tone=0.0,
                garment_tone=0.0):
    """The shot's constants: the brief's log ratios and hue shifts scaled by `tone` (times the shot's own strength
    when a conditional fit is given), applied to the shot's own native percentiles."""
    med = brief["median"]
    user, strength = tone, shot_strength(stats, cond) ** share_power
    tone = tone * strength
    prm = {"tone": tone, "strength": strength}
    for k in REGIONS + ("rest",):
        if k not in stats or f"{k}.p50" not in med:
            continue
        P = np.asarray(stats[k], np.float64) + np.arange(len(PCT)) * 1e-3
        tk = tone * (hair_tone if k == "hair" else 1.0)  # hair may take more of the brief than skin does
        if k == "rest" and rest_tone > 0:
            # the rest of the frame (clothes, props, walls) takes this share of the brief whatever the shot's own
            # strength: that strength is fitted on faces, and the brief's rest saturation ratio is nearly the same
            # on every reference frame (median x0.86, spread 0.05)
            tk = rest_tone
        T = np.maximum.accumulate(P * np.exp(tk * np.array([med[f"{k}.p{p}"] for p in PCT])))
        x = [0.0] + P.tolist() + [100.0]
        y = [0.0] + T.tolist() + [float(min(100.0, max(T[-1], 100.0 * T[-1] / P[-1])))]
        ck = "face" if k == "hands" and "face.sat" in med else k  # one skin: hands take the face's colour change
        prm[k] = dict(x=x, y=y, sat=float(np.exp(tk * med.get(f"{ck}.sat", 0.0))),
                      hue=float(tk * med.get(f"{ck}.hue", 0.0)))
    if "rest" in prm:
        c, L = stats.get("_restc", (np.zeros(1), np.zeros(1)))
        prm["rest"]["cmax"] = solve_cmax(c, L, prm["rest"]["sat"])
    for k in ("skin_model", "hair_model"):
        if k in stats:
            prm[k] = stats[k]
    if garment_tone > 0 and "garment_model" in stats and "hair.p50" in med:
        # The brief has no garment statistic (clothes are part of "rest", whose lightness it leaves alone). But the
        # change it asks of skin and hair is less light on the hero, and a garment left at the render's lightness
        # under a face a third darker glows: its lit shoulders stood above the wall (at140 145: L 64 against 50)
        # and all three judges of batch Q named a light rim on the shoulders. So the garment takes a share of the
        # hair's change (the hero's other non-skin surface), as one factor: its own shading keeps its shape.
        lr = float(np.mean([med[f"hair.p{p}"] for p in (25, 50, 75)]))
        prm["garment"] = dict(stats["garment_model"], factor=float(np.exp(garment_tone * tone * hair_tone * lr)))
    # rim and evenness follow the knob (full from 0.5); the eye rule also follows the shot's strength, so a close-up
    # (small share) keeps more of its eye colour
    u = min(2.0 * user, 1.0)
    ue = min(2.0 * user * strength, 1.0)
    prm["eye"], prm["eye_l"], prm["rim"], prm["even"] = float(eye ** ue), float(EYE_L ** ue), float(RIM * u), float(u)
    return prm


# ---- apply ---------------------------------------------------------------------------------------------------------
def _curve(L, r):
    return np.interp(L, r["x"], r["y"]).astype(np.float32)


def _rot(a, b, deg, k):
    t = np.radians(deg)
    return k * (a * np.cos(t) - b * np.sin(t)), k * (a * np.sin(t) + b * np.cos(t))


def apply_frame(img, mask, prm, depth=None, grow_skin=True, flat=0.0, w_out=None, g_hold=None):
    """img 8-bit floats (H, W, 3), mask floats 0..1 (H, W, 3) -> toned 8-bit uint8. w_out: a dict that receives the
    region weights used."""
    L, a, b = to_lab(img)
    w = weights(mask, (L, a, b) if grow_skin else None, prm.get("skin_model") if grow_skin else None, depth,
                g_hold if isinstance(g_hold, dict) else None)
    if w_out is not None:
        w_out.update(w)
    if prm.get("tone", 1.0) == 0 and flat == 0:
        return np.clip(img + 0.5, 0, 255).astype(np.uint8)
    # rest of the frame: curve, chroma soft limit that spares emitters
    r = prm.get("rest")
    if r:
        Lo = _curve(L, r)
        sat = (np.hypot(a, b) + 1e-6) / (L + SAT_L0)
        e = np.clip((L - EMIT_L[0]) / (EMIT_L[1] - EMIT_L[0]), 0, 1)
        k = e + (1 - e) * r["cmax"] * np.tanh(sat / r["cmax"]) / sat
        ao, bo = a * k, b * k
    else:
        Lo, ao, bo = L, a, b
    wr = 1 - (w["face"] + w["hands"] + w["hair"])
    gp = prm.get("garment")
    if gp and depth is not None:
        wg = garment_weight(L, a, b, w.get("skin_g", np.clip(1 - wr, 0, 1)), mask, depth, gp, g_hold)
        if wg is not None:
            kg = 1 - wg * (1 - gp["factor"])  # one factor on lightness and chroma: the garment's saturation stays
            if gp.get("curve"):
                # --garment-curve: the factor is reached at the garment's lit parts only (curve x its median
                # lightness, about its p90); the median and everything darker stay. One factor on the whole
                # garment took the light rim off the shoulders but left a dark sweater a fifth darker with its
                # highlights as far above its median as before, which reads as gloss ("wetsuit-like clothes",
                # blind judges of the held-out clips, 2026-11-22). A camera's highlight roll-off, which the brief
                # asks of the whole frame, does the same: lit parts come down, the body of the tone stays.
                l0, c = max(gp["L"], 1e-3), float(gp["curve"])
                g = max((c * gp["factor"] - 1) / (c - 1), 0.3)        # slope above the median
                kg = 1 - wg * (1 - (l0 + np.maximum(L - l0, 0) * g) / np.maximum(L, l0))
            Lo, ao, bo = Lo * kg, ao * kg, bo * kg
            if w_out is not None:
                w_out["garment"] = wg
    Lo, ao, bo = Lo * wr, ao * wr, bo * wr
    # inner: 1 inside the head (face + hair), 0 in a thin band along its outline against the background (rim light)
    hq = (down(w["face"] + w["hair"]) > 0.5).astype(np.float32)
    rr = max(1, int(round(OUTLINE * np.sqrt(hq.sum() * GROW_DS ** 2) / GROW_DS)))
    inner = np.clip(up(box(1 - maxr(1 - hq, rr), 1), L.shape), 0, 1)
    # core: 1 in the face's interior, 0 within a wider band of the face region's own outline (hair line, temple,
    # jaw). Eyes and teeth lie in the interior; a pale strip of light along the hair line is lit skin, not an eye.
    fq = (down(w["face"]) > 0.5).astype(np.float32)
    rf = max(1, int(round(FACE_EDGE * np.sqrt(fq.sum() * GROW_DS ** 2) / GROW_DS)))
    core = np.clip(up(box(1 - maxr(1 - fq, rf), max(1, rf // 2)), L.shape), 0, 1)
    # (The rim-light rule below acts on face-weighted pixels that are not skin-coloured. Wall pixels behind a jaw or
    # an ear were exactly that while the weights reached 2 px over the outline, and were pulled down to skin
    # lightness: a grey line against a light wall, rs345 350, x13. With depth the weights now end at the outline.)
    for key in REGIONS:
        q, wk = prm.get(key), w[key]
        if q is None:  # no such region in this shot's statistics: leave those pixels as they are
            Lo, ao, bo = Lo + L * wk, ao + a * wk, bo + b * wk
            continue
        ak, bk = _rot(a, b, q["hue"], q["sat"])
        Lk = _curve(L, q)
        sm, hm = prm.get("skin_model"), prm.get("hair_model")
        if key in ("face", "hands") and sm and prm.get("even", 0) > 0:
            # even the skin's colour: saturation well above the skin's median (orange eye sockets, red blotches,
            # which stand out once the skin is darker) is soft-limited toward it
            q0, q1 = EVEN
            rel = (np.hypot(a, b) / (L + SAT_L0)) / (np.hypot(sm["a"], sm["b"]) / (sm["L"] + SAT_L0))
            lim = q0 + (q1 - q0) * np.tanh(np.maximum(rel - q0, 0) / (q1 - q0))
            # reds (lips, ears) are meant to be more saturated than the skin: the limit acts fully from the skin's
            # own hue toward orange and yellow, and only a third as much 30 degrees toward red
            dh = np.degrees(np.arctan2(b, a)) - np.degrees(np.arctan2(sm["b"], sm["a"]))
            hw_ = 0.35 + 0.65 * np.clip((dh + 30) / 20, 0, 1)
            kk = 1 - prm["even"] * hw_ * (1 - np.where(rel > q0, lim / np.maximum(rel, 1e-6), 1.0))
            ak, bk = ak * kk, bk * kk
        if key == "face" and sm:
            # Face pixels with far less of the skin's own colour for their lightness (e) are not skin:
            #  inside the face they are eye whites, irises and teeth. A render leaves them brighter and cleaner
            #    than the skin beside them ("glowing eyes", blue-white teeth); under a brow and lids a camera does
            #    not. They lose colour, turn slightly warm, and the bright ones are pulled down.
            #  at the head's outline they are the cool rim light on jaw and temple, which reads as a hard blue edge:
            #    there colour and lightness move toward the skin's own.
            # e: projection of (a, b) / L on the skin's (CIELAB blue has a positive a, so "less red" alone misses
            # irises)
            nrm = (sm["a"] ** 2 + sm["b"] ** 2) / sm["L"]
            t = 1 - (a * sm["a"] + b * sm["b"]) / np.maximum(L, 1.0) / nrm
            e = np.clip((t - 0.55) / 0.3, 0, 1)
            ee, er = e * core, e * (1 - inner) * prm.get("rim", 0.0)
            ak, bk = ak * (1 - ee * (1 - prm.get("eye", 1.0))), bk * (1 - ee * (1 - prm.get("eye", 1.0)))
            # teeth (lower half of the face mask's box) stay bright and turn warmer; eye whites are pulled down
            fy = np.nonzero(mask[..., 0].max(1) > 0.5)[0]
            low = (np.clip(((np.arange(L.shape[0]) - fy[0]) / max(fy[-1] - fy[0], 1) - 0.5) / 0.1, 0, 1)[:, None]
                   if len(fy) else 0.0)
            warm = ee * (1 - prm.get("eye", 1.0)) * (1 + low)
            ak, bk = ak + 1.0 * warm, bk + 3.5 * warm
            el = prm.get("eye_l", 1.0)
            Lk = Lk * (1 - ee * np.clip((L - 0.9 * sm["L"]) / 10, 0, 1) * (1 - (el + (0.97 - el) * low)))
            ta, tb = _rot(sm["a"], sm["b"], q["hue"], q["sat"])
            k = np.clip(L / sm["L"], 0.3, 1.3)
            # rim_c: how much of the skin's saturation the rim light takes on; rim_l: how far its lightness is
            # pulled down to the skin's. At 1 / 1 (the rule until 2026-11-21) the lit outline becomes plain skin:
            # rs345 350 row 543, L across the jaw: render 85 88 84 74 70 79 87 87 83, ours 53 55 55 55 52 55 55 55 55.
            # The reference keeps the light and only warms it; three judges of batch Q called ours "flatter".
            rc, rl = prm.get("rim_c", 1.0), prm.get("rim_l", 1.0)
            ak, bk = ak + er * (ta * k * rc - ak), bk + er * (tb * k * rc - bk)
            Lk = Lk - er * rl * np.maximum(Lk - float(_curve(np.float32(sm["L"]), q)), 0)
        Lo, ao, bo = Lo + Lk * wk, ao + ak * wk, bo + bk * wk
    hm, qh = prm.get("hair_model"), prm.get("hair")
    if hm and qh and prm.get("rim", 0) > 0:
        # The same rim light on the hair's outline: a band that is paler than the hair's normal upper lightness
        # and cooler than its colour. It reads as a halo round the head once the hair is darker. The band is often
        # wider than the outline zone (a close-up), so it is followed inward from the outline through connected
        # pale or cool hair; lightness and colour move toward the hair's own. Applied on the composed frame with
        # the mask's own coverage, because the band is brightest in the outermost pixels, where the hair's tone
        # weight is feathered out. Not where the hair's edge lies over the face: there a bright strip is lit skin.
        by_face = np.clip(up(box(maxr(down(w["face"]), 3), 1), L.shape), 0, 1)
        if depth is not None:
            # Beside the temple and the cheek the hair is a band 10-20 px wide between the face and the background:
            # the rule above switched itself off there and the edge-lit strands stayed 25-35 levels above the wall
            # and the darkened hair, a light outline round the head (at140 195, rs345 350; batch Q, all three judges:
            # "light halo outlining hair"). Hair that has the background (a depth step) within the outline band's
            # width is an outline, whatever is on its other side.
            # (Within a third of the band's width only: over the whole width it also took the sideburn, a tongue
            # of thin hair between cheek and ear, and painted it hair-dark: a grey bar on zd400's far head,
            # a smudge on at140's cheek, 2026-11-21.)
            n = max(2, rr * GROW_DS // 3)
            sil = (maxr(depth, n) > 1.12 * depth).astype(np.float32)
            by_face = by_face * (1 - np.clip(box(sil, 1) * 1.5, 0, 1))
        cover = mask[..., 2] * (1 - by_face)
        if prm.get("fringe_skin"):
            # where the render marks skin as well (the fringe over the forehead), a pale pixel is skin seen
            # through the strands, not a lit strand: followed inward from the temples the rule painted the
            # whole zone hair-dark, a grey patch on the forehead of a close-up (ats08, all frames)
            cover = cover * (1 - np.clip(mask[..., 0], 0, 1))
        bright = np.clip((L - hm["hi"] - 4) / 10, 0, 1)
        cool = np.clip((hm["b"] - b - 2) / 4, 0, 1)
        cand = np.maximum(bright, cool) * cover
        cq = down(cand).astype(np.float32)
        # (grown by one 1/4-size pixel before the blur: the blur halved the band on the hair's outermost pixels,
        # which are the brightest ones; the mask's coverage keeps it on the hair)
        band = np.clip(up(box(maxr(grow(cq * down(1 - inner).astype(np.float32), cq), 1), 1), L.shape), 0, 1)
        ec = np.minimum(cand, band) * prm["rim"]
        el = np.minimum(bright * cover, band) * prm["rim"]
        ta, tb = _rot(hm["a"], hm["b"], qh["hue"], qh["sat"])
        ao, bo = ao + ec * (ta - ao), bo + ec * (tb - bo)
        Lo = Lo - el * np.maximum(Lo - float(_curve(np.float32(hm["hi"]), qh)), 0)
    if "edge" in w:
        # Outline pixels behind the hero: part skin, part background. They take their share (edge alpha) of the
        # change the skin right beside them received: X' = rest(X) + alpha * (F' - F), F and F' the local mean of
        # the solid skin before and after. Without it they keep the render's lightness beside darkened skin: a
        # light line where the outline is lit from behind, on the staircase of the depth samples.
        al, src = w["edge"], w["edge_src"]
        d2, d3 = box(src, 2), box(src, 3)
        al = np.where(d3 > 0.02, al, 0.0)

        def near(v):  # the change of the nearest solid skin (5 x 5, else 7 x 7), as F in edge_alpha
            return np.where(d2 > 0.02, box(v * src, 2) / np.maximum(d2, 1e-4), box(v * src, 3) / np.maximum(d3, 1e-4))
        Lo, ao, bo = Lo + al * near(Lo - L), ao + al * near(ao - a), bo + al * near(bo - b)
    out = from_lab(Lo, ao, bo)
    if flat > 0:
        from nts.finish import grade
        out = grade(out / 255.0, dict(sat=1.0, chroma_max=10.0, toe=0.016 * flat, shoulder=1 - 0.15 * flat)) * 255
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _tone_frame(task):
    """run()'s pass 2 for one frame: toned frame and (weights_out) its region weights."""
    (frames_dir, masks_dir, out_dir, weights_out, prm, grow_skin, flat, src), f, g_hold = task
    img, m = load(frames_dir, f), load(masks_dir, f, "m")
    m = np.zeros_like(img) if m is None else m / 255.0
    depth = None
    if src is not None:
        from nts import warp as W
        seq = _seq_of(src)
        if f in seq.files:
            depth = W.upsample_nearest(seq.read(f).depth, seq.scale, img.shape[:2])
    w = {} if weights_out else None
    out = apply_frame(img, m, prm, depth, grow_skin, flat, w, g_hold)
    Image.fromarray(out).save(os.path.join(out_dir, f"f_{f:04d}.png"))
    if weights_out:
        wi = np.stack([w["face"], w["hands"], w["hair"]], -1)
        Image.fromarray((wi * 255 + 0.5).astype(np.uint8)).save(os.path.join(weights_out, f"w_{f:04d}.png"))


def shots_of(frames, cuts):
    cuts = sorted(c for c in cuts if frames[0] < c <= frames[-1])
    edges = [frames[0]] + cuts + [frames[-1] + 1]
    return [[f for f in frames if a <= f < b] for a, b in zip(edges[:-1], edges[1:])]


def run(frames_dir, masks_dir, out_dir, brief_path, tone=1.0, flat=0.0, geom=None, span=None, cuts=None, eye=0.4,
        grow_skin=True, params_out=None, weights_out=None, only=None, cond_path=None, share_power=1.0,
        hair_tone=1.0, rest_tone=0.0, garment_tone=0.0, rim_l=1.0, rim_c=1.0, garment_curve=0.0,
        fringe_skin=False, skin_rim=False, garment_hold=False, skin_close=False, shot_hold=False, hands_tight=False,
        workers=1):
    from nts.par import pmap
    brief = json.load(open(brief_path))
    cond = json.load(open(cond_path)) if cond_path and os.path.exists(cond_path) else None
    frames = frames_in(frames_dir)  # constants always come from the whole shot, whatever part is written
    want = {f for f in frames if (span is None or span[0] <= f <= span[1]) and (only is None or f in only)}
    if not want:
        sys.exit(f"tone: no frames to write in {frames_dir}")
    if cuts is None:
        # the frames' own cuts.json, else the geometry folder's (the render writes it there): processed frame
        # folders carry none, and a two-shot clip was toned as one shot until 2026-11-21 (zd400, cut at 439)
        cp = [os.path.join(d, "cuts.json") for d in (frames_dir, geom) if d]
        cp = [q for q in cp if os.path.exists(q)]
        cuts = json.load(open(cp[0])) if cp else []
        cuts = cuts.get("cuts", []) if isinstance(cuts, dict) else cuts
    seq = None
    if geom:
        from nts import warp as W
        from nts.io_exr import Sequence
        seq = Sequence(geom)
        _SEQ[seq.src] = seq
    os.makedirs(out_dir, exist_ok=True)
    log = {}
    for shot in shots_of(frames, cuts):
        if not want & set(shot):
            continue
        st = shot_stats(frames_dir, masks_dir, shot)
        prm = shot_params(st, brief, tone, eye, cond, share_power, hair_tone, rest_tone, garment_tone)
        prm["rim_l"], prm["rim_c"] = float(rim_l), float(rim_c)
        if garment_curve > 1 and "garment" in prm:
            prm["garment"]["curve"] = float(garment_curve)
        if skin_rim and "garment" in prm:
            prm["garment"]["skin_rim"] = 1
        if skin_close and prm.get("skin_model"):
            prm["skin_model"]["close"] = 1
        if hands_tight and prm.get("skin_model"):
            prm["skin_model"]["hands_tight"] = 1
        hold = None
        if seq is not None and (shot_hold or (garment_hold and "garment" in prm)):
            if "garment" in prm:
                prm["garment"]["hold"] = 2 if shot_hold else 1
            hold = garment_votes(frames_dir, masks_dir, shot, prm, seq, grow_skin, shot_hold, workers)
        if fringe_skin:
            prm["fringe_skin"] = 1
        log[str(shot[0])] = {k: v for k, v in prm.items()}
        ctx = (frames_dir, masks_dir, out_dir, weights_out, prm, grow_skin, flat, None if seq is None else seq.src)
        if weights_out:
            os.makedirs(weights_out, exist_ok=True)
        pmap(_tone_frame, [(ctx, f, None if hold is None else hold.get(f)) for f in shot if f in want], workers)
    if params_out:
        json.dump(log, open(params_out, "w"), indent=1)
    share = "/".join("%.2f" % v["strength"] for v in log.values()) if cond else "off"
    print(f"tone: {len(want)} frames -> {out_dir}; tone {tone} x shot strength {share} flat {flat} "
          f"shots {len(log)}{' depth' if seq else ''}")
    return log


def smoke():
    """Synthetic check, no data needed: tone 0 is pixel-exact; tone 1 lands on the brief's ratios; constants do not
    change inside a shot."""
    import tempfile
    rng = np.random.default_rng(0)
    d = tempfile.mkdtemp()
    fd, md = os.path.join(d, "f"), os.path.join(d, "m")
    os.makedirs(fd), os.makedirs(md)
    h, w = 240, 136
    mask = np.zeros((h, w, 3), np.uint8)
    mask[60:120, 40:96, 0] = 255   # face
    mask[40:60, 40:96, 2] = 255    # hair
    mask[180:210, 20:50, 1] = 255  # hands
    for f in range(1, 5):
        img = rng.uniform(30, 220, (h, w, 3)).astype(np.float32) * 0.3 + 60
        img[60:120, 40:96] = np.array([222, 170, 150]) + rng.normal(0, 8, (60, 56, 3))
        img[40:60, 40:96] = np.array([120, 100, 80]) + rng.normal(0, 8, (20, 56, 3))
        img[180:210, 20:50] = np.array([215, 165, 145]) + rng.normal(0, 8, (30, 30, 3))
        img[10:30, 10:60] = np.array([40, 120, 235])  # a saturated print
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(fd, f"f_{f:04d}.png"))
        Image.fromarray(mask).save(os.path.join(md, f"m_{f:04d}.png"))
    med = {}
    for k, (lr, sat, hue) in {"face": (-0.44, -0.06, -8.4), "hands": (-0.33, 0.2, -12.9), "hair": (-0.49, -0.12, -4.8),
                              "rest": (0.02, -0.15, 2.0)}.items():
        med.update({f"{k}.p{p}": lr for p in PCT})
        med[f"{k}.sat"], med[f"{k}.hue"] = sat, hue
    bp = os.path.join(d, "brief.json")
    json.dump({"median": med}, open(bp, "w"))
    o0, o1 = os.path.join(d, "o0"), os.path.join(d, "o1")
    run(fd, md, o0, bp, tone=0.0, grow_skin=False)
    log = run(fd, md, o1, bp, tone=1.0, grow_skin=False)
    src, a0, a1 = load(fd, 2), load(o0, 2), load(o1, 2)
    assert np.array_equal(src.astype(np.uint8), a0.astype(np.uint8)), "tone 0 must be pixel-exact"
    face = mask[..., 0] > 0
    r = np.median(to_lab(a1)[0][face]) / np.median(to_lab(src)[0][face])
    assert abs(r - np.exp(-0.44)) < 0.03, f"face lightness ratio {r:.3f}, expected {np.exp(-0.44):.3f}"
    cs, c1 = (np.hypot(*to_lab(x)[1:])[10:30, 10:60].mean() for x in (src, a1))
    assert c1 < 0.9 * cs, "the saturated print must lose chroma"
    assert len(log) == 1, "one shot, one set of constants"
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    print(f"tone smoke OK: face L ratio {r:.3f} (brief {np.exp(-0.44):.3f}), print chroma x{c1 / cs:.2f}")


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "smoke":
        return smoke()
    if a[0] == "fit-cond" and len(a) >= 4:
        return fit_cond(a[1], a[2], a[3])
    if a[0] != "apply" or len(a) < 4:
        sys.exit(__doc__)
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    span = tuple(map(int, opt("--frames").split("-"))) if "--frames" in a else None
    cuts = [int(x) for x in opt("--cuts").split(",") if x] if "--cuts" in a else None
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    run(a[1], a[2], a[3], opt("--brief", os.path.join(here, "data/look/brief_v1.json")), float(opt("--tone", 1.0)),
        float(opt("--flat", 0.0)), opt("--geom"), span, cuts, float(opt("--eye", 0.4)), "--no-grow" not in a,
        opt("--params"), opt("--weights-out"), {int(x) for x in opt("--only").split(",")} if "--only" in a else None,
        None if "--no-cond" in a else opt("--cond", os.path.join(here, "data/look/brief_v1_cond.json")),
        float(opt("--share-power", 1.0)), float(opt("--hair-tone", 1.0)), float(opt("--rest-tone", 0.0)),
        float(opt("--garment-tone", 0.0)), float(opt("--rim-l", 1.0)), float(opt("--rim-c", 1.0)),
        float(opt("--garment-curve", 0.0)), "--fringe-skin" in a, "--skin-rim" in a, "--garment-hold" in a, "--skin-close" in a, "--shot-hold" in a,
        "--hands-tight" in a, opt("--workers"))


if __name__ == "__main__":
    main()
