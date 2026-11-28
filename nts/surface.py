"""Keyframe correspondence by surface identity (2026-11-26): for every pixel of a frame, the place in the keyframe
that shows the same bit of surface, from the renderer's "surf" pass (datagen/blender/render_lr.py, setup_surf).

No motion is used, and frames do not depend on each other. Each pixel of the surf pass holds
  uv   the shaded point's coordinates in its object's active UV map: a mesh attribute, so they move with the surface
       under any deformation;
  g    its Blender Generated texture coordinates (the mesh's rest shape, normalised to its bounding box). They tell
       apart UV islands that share coordinates (mirrored halves), but they are not exact under deformation: on at140
       the hands' moved by several pixels' worth from one frame to the next, so they enter the match at a small
       weight (WG) where the object has UVs, and alone only where it has none;
  id   the object's random number,
averaged over samples that all go through the pixel's centre. A pixel whose samples saw more than one point (an
outline, a strand, a UV seam) shows a spread (sid = (|g|^2, id, id^2), uv.B = u^2 + v^2) and is left out; so is one
whose values were added up along the ray (a negative spread: Cycles adds every AOV a camera ray meets before an opaque
surface; render_lr.py writes them only on opaque ones, but older passes have such pixels under alpha cards).
The feature matched is f = (u, v, WG g) (or g alone without UVs).

A frame pixel's keyframe position u is where the keyframe's field f_k(u) equals the pixel's f, on the same object:
  1. a nearest-point search in a hash grid of the keyframe's points (over the object's three widest dimensions of f;
     cell: twice the object's typical pixel footprint), on every 4th pixel;
  2. every pixel starts from its block's result and is refined by Gauss-Newton on f_k: u += (J^T J)^-1 J^T (f - f_k(u)),
     J = the keyframe's kx2 slope of f at u (differences on the same object, not across a UV seam);
  3. pixels whose result fails the check (the keyframe pixel at u shows another object, or f_k(u) is further than half
     a pixel's footprint from f) get a full search of their own and the same refinement; whatever still fails gets no
     keyframe position.
f_k(u) is bilinear between the four keyframe pixels around u when they are all the object; next to an outline it is
extended to first order (f + J (u - p)) from the pixel p that u lies in.
A point the keyframe does not show (hidden there, or outside its frame) has no keyframe point within reach, so it
gets none: occlusion needs no separate test.

settle() then evens out the result. Each pixel's position is found on its own, to a few hundredths of a pixel, and
next to a pixel without a match the result is the least sure. A texture read at positions that are each a little off
shimmers from frame to frame, most where the surface is small in the frame. So a matched pixel's offset (keyframe
position minus its own position) is replaced by the mean offset of the matched pixels around it that lie at its own
depth: the offset of a surface changes slowly across it, so the mean keeps it and drops the per-pixel error. Only
depth decides which neighbours count; no object or motion data is read.

Re-texturing through a rendered surface-coordinate pass is long-standing compositing practice (UV and
reference-position passes); the search is a plain spatial hash and the refinement plain least squares.
"""
import numpy as np


VAR_G = 1e-5      # spread of a pixel's surface points (coordinate units^2) above which it counts as mixed
VAR_ID = 1e-7     # the same for the object number
ID_SCALE = 2.0 ** 20
CELL = 2.0        # grid cell, in footprints
MAXC = 12         # points read per grid cell
STRIDE = 4        # sparse search: one pixel in STRIDE x STRIDE
GN_STEPS = 5
STOP = 2e-3  # px: a Gauss-Newton step this small ends that point's refinement (float32 noise is ~1e-4)
TOL = 0.5         # accepted distance between f and f_k(u), in footprints
WG = 0.05         # weight of the Generated coordinates next to the UVs
SEAM = 4.0        # a one-sided slope over SEAM times the other side's is a jump (a UV seam), not a slope
SETTLE = 2        # settle(): offsets are averaged over (2 SETTLE + 1)^2 pixels
SETTLE_REL = 0.05  # settle(): a neighbour counts when its depth is within this share of the pixel's own


def _layer(ch, name):
    for sfx in (("R", "G", "B"), ("X", "Y", "Z")):
        k = [f"{name}.{c}" for c in sfx]
        if all(c in ch for c in k):
            return np.stack([ch[c] for c in k], -1).astype(np.float32)
    raise KeyError(f"no {name} layer (have {sorted(ch)[:16]})")


class Surf:
    """One frame of the surf pass. f (h, w, k) float32: the feature matched (module docstring); g (h, w, 3) the
    Generated coordinates; ids (h, w) int64, -1 where no single surface point was seen; world (h, w) bool, True where
    no object was hit; depth (h, w) or None. uv (h, w, 3) = (u, v, u^2 + v^2), or None for a pass without UVs."""

    def __init__(self, g, sid, depth=None, uv=None):
        g = np.asarray(g, np.float32)
        sid = np.asarray(sid, np.float32)
        spread_g = sid[..., 0] - (g * g).sum(-1)
        spread_id = sid[..., 2] - sid[..., 1] ** 2
        ok = (sid[..., 1] > 0) & (np.abs(spread_g) < VAR_G) & (np.abs(spread_id) < VAR_ID) & np.isfinite(g).all(-1)
        if uv is not None:
            uv = np.asarray(uv, np.float32)
            ok &= (np.abs(uv[..., 2] - (uv[..., :2] ** 2).sum(-1)) < VAR_G) & np.isfinite(uv).all(-1)
            self.f = np.concatenate([uv[..., :2], WG * g], -1)
        else:
            self.f = g
        self.g = g
        self.ids = np.where(ok, np.rint(sid[..., 1].astype(np.float64) * ID_SCALE), -1).astype(np.int64)
        self.world = sid[..., 1] <= 0  # no object: the world background
        self.depth = depth
        self.shape = g.shape[:2]


def read(path):
    from nts.io_exr import read_layers
    ch = read_layers(path)
    uv = _layer(ch, "Uv") if any(c.startswith("Uv.") for c in ch) else None
    return Surf(_layer(ch, "Surf"), _layer(ch, "Sid"), ch.get("Depth.V"), uv)


def _slope(g, ids, axis):
    """Central difference of g along axis (1 = x, 0 = y) between pixels of the same object; one-sided at an object's
    edge or where one side jumps (a UV seam: SEAM times the other side or more); NaN where neither neighbour is the
    same object."""
    h, w = ids.shape
    gp, gm = np.full_like(g, np.nan), np.full_like(g, np.nan)
    ip, im = np.full((h, w), -2, np.int64), np.full((h, w), -2, np.int64)
    if axis == 1:
        gp[:, :-1], gm[:, 1:], ip[:, :-1], im[:, 1:] = g[:, 1:], g[:, :-1], ids[:, 1:], ids[:, :-1]
    else:
        gp[:-1], gm[1:], ip[:-1], im[1:] = g[1:], g[:-1], ids[1:], ids[:-1]
    fwd, bwd = ((ip == ids) & (ids >= 0))[..., None], ((im == ids) & (ids >= 0))[..., None]
    df, db = gp - g, g - gm
    with np.errstate(invalid="ignore"):
        nf, nb = np.linalg.norm(df, axis=-1)[..., None], np.linalg.norm(db, axis=-1)[..., None]
        fwd_ok = fwd & ~(bwd & (nf > SEAM * nb))
        bwd_ok = bwd & ~(fwd & (nb > SEAM * nf))
    return np.where(fwd_ok & bwd_ok, 0.5 * (df + db), np.where(fwd_ok, df, np.where(bwd_ok, db, np.nan))).astype(np.float32)


def _smooth_pairs(f, ids, j, axis):
    """(h, w) bool: pixel and its next one along axis (1 = x, 0 = y) are the same object and the field runs smoothly
    between them (no UV seam): |f(next) - f| <= SEAM * the larger of their slopes along axis."""
    h, w = ids.shape
    out = np.zeros((h, w), bool)
    a, b = (np.s_[:, :-1], np.s_[:, 1:]) if axis == 1 else (np.s_[:-1], np.s_[1:])
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(f[b] - f[a], axis=-1)
        sl = np.fmax(np.linalg.norm(j[a], axis=-1), np.linalg.norm(j[b], axis=-1))
        out[a] = (ids[a] == ids[b]) & (ids[a] >= 0) & (d <= SEAM * sl)
    return out


class Key:
    """A keyframe's surf pass prepared for lookups: slopes, footprints and, per object, a hash grid of its points."""

    def __init__(self, s):
        self.s = s
        self.jx, self.jy = _slope(s.f, s.ids, 1), _slope(s.f, s.ids, 0)
        a, b, c = (self.jx ** 2).sum(-1), (self.jx * self.jy).sum(-1), (self.jy ** 2).sum(-1)
        with np.errstate(invalid="ignore"):
            fp = np.sqrt(np.sqrt(np.maximum(a * c - b * b, 0)))  # side of a pixel's patch of surface, in f units
        self.fp = np.where(np.isfinite(fp), fp, np.nan).astype(np.float32)
        # pixel pairs the field runs smoothly across (same object, no UV seam): cx to the right, cy below. Across a
        # seam the field jumps; a pair counts as a jump when the change between the two pixels is over SEAM times
        # the slope either of them has along that axis
        self.cx, self.cy = _smooth_pairs(s.f, s.ids, self.jx, 1), _smooth_pairs(s.f, s.ids, self.jy, 0)
        # field, slopes and footprint in one contiguous buffer: one lookup per Gauss-Newton step reads them all
        self.stack = np.ascontiguousarray(np.concatenate(
            [s.f, self.jx, self.jy, np.nan_to_num(self.fp, nan=0.0)[..., None]], -1), np.float32)
        self.foot = {}
        good = (s.ids >= 0) & np.isfinite(self.fp) & (self.fp > 0)
        for o in np.unique(s.ids[good]):
            v = self.fp[good & (s.ids == o)]
            if len(v) >= 16:
                f = float(np.median(v))
                if f > 1e-7:  # an object whose coordinates do not change across it is skipped
                    self.foot[int(o)] = f
        self._grids = {}

    def grid(self, o):
        if o not in self._grids:
            s = self.s
            k = s.f.shape[-1]
            flat = np.flatnonzero((s.ids == o).ravel() & np.isfinite(self.fp).ravel())
            full = s.f.reshape(-1, k)[flat].astype(np.float64)
            dm = np.sort(np.argsort(-full.std(0))[:3])  # the grid runs over the three widest dimensions
            pts = full[:, dm]
            cell = CELL * self.foot[o]
            lo = pts.min(0) - cell
            c = np.floor((pts - lo) / cell).astype(np.int64)
            dims = c.max(0) + 3
            key = (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]
            order = np.argsort(key, kind="stable")
            uk, start, cnt = np.unique(key[order], return_index=True, return_counts=True)
            self._grids[o] = (cell, lo, dims, uk, start, cnt, flat[order], full[order], dm)
        return self._grids[o]

    def nearest(self, o, q):
        """Flat keyframe pixel index of the nearest point of object o to each row of q (n, k), or -1 if none is
        within the grid neighbourhood (about two footprints)."""
        cell, lo, dims, uk, start, cnt, flat, pts, dm = self.grid(o)
        q = q.astype(np.float64)
        c = np.floor((q[:, dm] - lo) / cell).astype(np.int64)
        best = np.full(len(q), -1, np.int64)
        bd = np.full(len(q), np.inf)
        inside = ((c >= -1) & (c <= dims - 2)).all(1)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cc = c + (dx, dy, dz)
                    ok = inside & ((cc >= 0) & (cc < dims)).all(1)
                    k = (cc[:, 0] * dims[1] + cc[:, 1]) * dims[2] + cc[:, 2]
                    pos = np.clip(np.searchsorted(uk, k), 0, len(uk) - 1)
                    n = np.where(ok & (uk[pos] == k), cnt[pos], 0)
                    st = start[pos]
                    for j in range(MAXC):
                        m = np.flatnonzero(n > j)
                        if not len(m):
                            break
                        i = st[m] + j
                        d = ((pts[i] - q[m]) ** 2).sum(1)
                        b = d < bd[m]
                        bd[m[b]], best[m[b]] = d[b], flat[i[b]]
        return best


def _lookup(buf, x, y):
    """Bilinear lookup of buf (h, w, c) float32 at x, y (n,): (n, c) float32; edges clamped as in nts.warp. A tap
    that is NaN makes that channel NaN."""
    h, w = buf.shape[:2]
    x, y = np.clip(x, 0, w - 1.001).astype(np.float32), np.clip(y, 0, h - 1.001).astype(np.float32)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    flat, i = buf.reshape(h * w, -1), y0 * w + x0
    top, bot = flat.take(i, axis=0), flat.take(i + w, axis=0)
    top += (flat.take(i + 1, axis=0) - top) * fx
    bot += (flat.take(i + w + 1, axis=0) - bot) * fx
    top += (bot - top) * fy
    return top


def _lookup_obj(key, x, y, ids):
    """The keyframe's field, slopes and footprint at x, y (n,) for points of object ids (n,): (n, c) float32, NaN
    where the keyframe does not show that object there. Bilinear where the four pixels around the point are that
    object with slopes and no UV seam runs between them; next to an outline (another object, a mixed pixel, no
    slope) or a seam the value is extended to first order from the pixel the point lies in, which must show the
    object. A point under the outline's last pixel is then still found, where the bilinear lookup mixed in the other
    side: on hp622n 13 % of the face, every outline of brows, lids, lips and hairline and the head's UV seams, had no
    match and was re-seated, with jitter from frame to frame."""
    s, buf = key.s, key.stack
    h, w = s.shape
    k = s.f.shape[-1]
    x, y = np.clip(x, 0, w - 1.001).astype(np.float32), np.clip(y, 0, h - 1.001).astype(np.float32)
    v = _lookup(buf, x, y)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    inner = np.isfinite(v).all(1) & (v[:, 3 * k] > 0) & (s.ids[y0, x0] == ids)
    inner &= key.cx[y0, x0] & key.cx[y0 + 1, x0] & key.cy[y0, x0] & key.cy[y0, x0 + 1]
    edge = np.flatnonzero(~inner)
    if len(edge):
        px, py = np.rint(x[edge]).astype(np.int64), np.rint(y[edge]).astype(np.int64)
        t = buf[py, px]
        ok = (s.ids[py, px] == ids[edge]) & np.isfinite(t).all(1) & (t[:, 3 * k] > 0)
        dx, dy = (x[edge] - px)[:, None], (y[edge] - py)[:, None]
        t[:, :k] += t[:, k:2 * k] * dx + t[:, 2 * k:3 * k] * dy
        t[~ok] = np.nan
        v[edge] = t
    return v


def _refine(key, q, ids, u):
    """Gauss-Newton on the keyframe's field from start positions u (n, 2: x, y). Returns (u, ok). A point stops
    once its step is under STOP px (or the step cannot be solved: it would not move again either)."""
    s = key.s
    h, w = s.shape
    k = q.shape[1]
    u = np.array(u, np.float32)
    err, fp = np.full(len(u), np.inf, np.float32), np.zeros(len(u), np.float32)
    act = np.arange(len(u))
    for it in range(GN_STEPS + 1):
        if not len(act):
            break
        ua = u[act]
        v = _lookup_obj(key, ua[:, 0], ua[:, 1], ids[act])
        gk, jx, jy = v[:, :k], v[:, k:2 * k], v[:, 2 * k:3 * k]
        r = q[act] - gk
        # the match's error and footprint where it stands (a point that stops moves by under STOP px after this)
        err[act], fp[act] = np.sqrt((r * r).sum(1)), v[:, 3 * k]  # no footprint (0): nothing is close enough
        if it == GN_STEPS:  # the last pass only measures the points that were still moving
            break
        a, b, c = (jx * jx).sum(1), (jx * jy).sum(1), (jy * jy).sum(1)
        p, t = (jx * r).sum(1), (jy * r).sum(1)
        det = a * c - b * b
        good = np.isfinite(det) & (det > 1e-30)
        det = np.where(good, det, 1.0)
        dx = np.where(good, (c * p - b * t) / det, 0.0)
        dy = np.where(good, (a * t - b * p) / det, 0.0)
        step = np.clip(np.stack([dx, dy], 1), -3.0, 3.0).astype(np.float32)
        ua = ua + step
        ua[:, 0] = np.clip(ua[:, 0], 0, w - 1)
        ua[:, 1] = np.clip(ua[:, 1], 0, h - 1)
        u[act] = ua
        act = act[np.abs(step).max(1) > STOP]
    # the pixel the point ends in must show its object (a last step under STOP px can still cross into another)
    same = s.ids[np.clip(np.rint(u[:, 1]).astype(np.int64), 0, h - 1), np.clip(np.rint(u[:, 0]).astype(np.int64), 0, w - 1)] == ids
    return u, same & np.isfinite(err) & (err <= TOL * fp)


def match(key, cur, win=None, objects=None):
    """Keyframe positions for frame cur's pixels in win (x0, y0, x1, y1; None = whole frame).
    Returns (map (h, w, 2) float32 keyframe pixel coordinates x, y; valid (h, w) float32 0/1). Pixels with no match
    keep their own position in map, with valid 0. objects: only these object ids are looked up (None: all the
    keyframe knows)."""
    H, W_ = cur.shape
    x0, y0, x1, y1 = win or (0, 0, W_, H)
    ids = cur.ids[y0:y1, x0:x1]
    g = cur.f[y0:y1, x0:x1]
    k = g.shape[-1]
    hh, ww = ids.shape
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    out = np.stack([xs, ys], -1)
    valid = np.zeros((hh, ww), np.float32)
    known = set(key.foot) if objects is None else set(key.foot) & set(objects)
    want = np.isin(ids, np.fromiter(known, np.int64, len(known))) if known else np.zeros_like(ids, bool)
    if not want.any():
        return out, valid
    # 1. sparse search
    sp = np.zeros_like(want)
    sp[::STRIDE, ::STRIDE] = True
    sp &= want
    hit = np.full((hh, ww), -1, np.int64)
    for o in np.unique(ids[sp]):
        m = sp & (ids == o)
        hit[m] = key.nearest(int(o), g[m])
    # 2. the sparse results refined, then every pixel starts from its block's refined result moved by its own place in
    # the block (a start within a fraction of a pixel, where the block's corner alone was up to 3 px off)
    kw = key.s.shape[1]
    sk = np.flatnonzero((hit >= 0).ravel())
    spos = np.full((hh * ww, 2), np.nan, np.float32)
    if len(sk):
        h0 = hit.ravel()[sk]
        us, _ = _refine(key, g.reshape(-1, k)[sk], ids.ravel()[sk], np.stack([h0 % kw, h0 // kw], 1).astype(np.float32))
        spos[sk] = us
    spos = spos.reshape(hh, ww, 2)
    by, bx = (np.arange(hh) // STRIDE) * STRIDE, (np.arange(ww) // STRIDE) * STRIDE
    seed = hit[by[:, None], bx[None, :]]
    seed_ok = want & (seed >= 0) & (ids[by[:, None], bx[None, :]] == ids)
    sel = np.flatnonzero(seed_ok.ravel())
    if len(sel):
        off = np.stack([np.broadcast_to((np.arange(ww) - bx)[None, :], (hh, ww)),
                        np.broadcast_to((np.arange(hh) - by)[:, None], (hh, ww))], -1).reshape(-1, 2)[sel]
        u0 = spos[by[:, None], bx[None, :]].reshape(-1, 2)[sel] + off.astype(np.float32)
        u0[:, 0] = np.clip(u0[:, 0], 0, kw - 1)
        u0[:, 1] = np.clip(u0[:, 1], 0, key.s.shape[0] - 1)
        u, ok = _refine(key, g.reshape(-1, k)[sel], ids.ravel()[sel], u0)
        out.reshape(-1, 2)[sel[ok]] = u[ok]
        valid.ravel()[sel[ok]] = 1.0
    # 3. the rest: a search of their own
    rest = want & (valid == 0)
    for o in np.unique(ids[rest]):
        m = np.flatnonzero((rest & (ids == o)).ravel())
        h0 = key.nearest(int(o), g.reshape(-1, k)[m])
        f = h0 >= 0
        if not f.any():
            continue
        m, h0 = m[f], h0[f]
        u, ok = _refine(key, g.reshape(-1, k)[m], ids.ravel()[m], np.stack([h0 % kw, h0 // kw], 1).astype(np.float32))
        out.reshape(-1, 2)[m[ok]] = u[ok]
        valid.ravel()[m[ok]] = 1.0
    return out, valid


def settle(m, valid, depth, win=None, r=SETTLE, rel=SETTLE_REL):
    """match()'s map with every matched pixel's offset (map - the pixel's own position) replaced by the mean offset
    of the matched pixels in the (2r+1)^2 window around it whose depth is within rel of its own (module docstring).
    m, valid: match()'s result on win (None = the whole frame); depth (h, w): the frame's depth on the same
    rectangle. Pixels without a match are returned as they are and lend nothing; r = 0 returns m."""
    if r <= 0:
        return m
    x0, y0 = (win[0], win[1]) if win else (0, 0)
    hh, ww = valid.shape
    ys, xs = np.mgrid[y0:y0 + hh, x0:x0 + ww].astype(np.float32)
    g = np.stack([xs, ys], -1)
    v = valid > 0.999
    depth = depth.astype(np.float32)
    off = np.where(v[..., None], m - g, 0.0).astype(np.float32)
    lo, hi = depth * (1.0 - rel), depth * (1.0 + rel)
    num, den = np.zeros_like(off), np.zeros((hh, ww), np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ya, yb, xa, xb = max(0, -dy), min(hh, hh - dy), max(0, -dx), min(ww, ww - dx)
            here = (slice(ya, yb), slice(xa, xb))
            there = (slice(ya + dy, yb + dy), slice(xa + dx, xb + dx))
            w = v[there] & (depth[there] > lo[here]) & (depth[there] < hi[here])
            num[here] += off[there] * w[..., None]
            den[here] += w
    k = v & (den > 0)
    out = m.copy()
    out[k] = g[k] + num[k] / den[k][:, None]
    return out
