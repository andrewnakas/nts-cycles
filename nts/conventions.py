"""Single source of truth for how Blender Cycles buffers map into image space. Measured in cycles-harness (CONVENTIONS.md).

Image space everywhere in this repo: row 0 = top, +x right, +y down, units = pixels of the buffer they belong to.
- Motion: Blender Vector.xy points from a pixel in frame n to the same surface point in frame n-1, in render pixels with
  +y up. In image space that is mv = (Vector.X, -Vector.Y): prev_pos = cur_pos + mv.
- Jitter: jitter.json holds the lens-shift offset (jx, jy) in render pixels, +y up. A lens shift of +jx pans the view
  right, so pixel x sees the scene point that sits at x + jx in the unshifted image: the sample of pixel centre
  (x+0.5, y+0.5) lands at (x+0.5+jx, y+0.5-jy) in image space. The (-jx, +jy) measured in cycles-harness is the negation
  of that, because the upscaler there takes the offset that undoes the jitter. To re-check on renders, run
  baselines/accumulate.py with --sample-sign and --mv-sign: with a wrong sign its output falls apart.
- Depth: Depth.V is linear camera distance in scene units.
"""
import math

import numpy as np

MV_SIGN = (1.0, -1.0)
SAMPLE_SIGN = (1.0, -1.0)  # lens-shift jitter -> image-space sample offset
LR_FILTER, GT_FILTER = 0.5, 1.5  # Cycles pixel filter width: 0.5 for jittered low-res input, 1.5 for references
OUT = (1080, 1920)               # (w, h) of every final frame

# NTS-F quality modes, the usual ladder of per-axis scale factors of temporal upscalers (native AA 1.0, Quality 1.5,
# Balanced ~1.72, Performance 2.0, Ultra Performance 3.0). Render sizes for the 1080x1920 output keep the aspect
# exact (scale_of() asserts it), so Balanced is 630x1120 (x1.714). datagen/blender/render_lr.py MODES renders them.
QUALITY_MODES = {"native_aa": (1080, 1920), "quality": (720, 1280), "balanced": (630, 1120), "performance": (540, 960),
                 "ultra_performance": (360, 640)}


def quality_mode(name):
    """(render (w, h), per-axis scale, share of output pixels rendered) of a quality mode."""
    w, h = QUALITY_MODES[name]
    s = OUT[0] / w
    return (w, h), s, (w * h) / (OUT[0] * OUT[1])


def mv_image(vector_xy):
    """Blender Vector.XY (h, w, 2) -> image-space motion (current -> previous), render pixels."""
    return vector_xy * np.array(MV_SIGN, np.float32)


def mv_next_image(vector_zw):
    """Blender Vector.ZW (h, w, 2) -> image-space motion (current -> NEXT), render pixels. The opposite sign of XY:
    measured 2026-11-19 on at140 (frames 150/170/190, moving pixels): warping frame f+1 by this gives a mean error of
    6.9 / 4.7 / 3.6 against frame f, the other sign 29.0 / 23.1 / 16.9, no warp 23.6 / 16.3 / 13.1."""
    return -vector_zw * np.array(MV_SIGN, np.float32)


def sample_offset(jx, jy):
    """Where the low-res pixel's sample really sits, relative to its centre, in image-space render pixels."""
    return jx * SAMPLE_SIGN[0], jy * SAMPLE_SIGN[1]


def sample_positions_hr(h, w, jx, jy, scale):
    """Continuous high-res image coordinates (h, w, 2) of every low-res sample (pixel-centre convention: centre of
    pixel i is i + 0.5)."""
    ox, oy = sample_offset(jx, jy)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([(xs + 0.5 + ox) * scale, (ys + 0.5 + oy) * scale], axis=-1)


def halton(i, base):
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def jitter_phases(render_h, out_h=OUT[1]):
    return max(8, round(8 * (out_h / render_h) ** 2))


def jitter_seq(n, phases, start=0):
    """Halton(2,3) offsets in [-0.5, 0.5) render pixels for frames 1..n; start picks a random phase for augmentation."""
    return [(halton((i + start) % phases + 1, 2) - 0.5, halton((i + start) % phases + 1, 3) - 0.5) for i in range(n)]


def resets(frames, cuts):
    """Per-frame history-reset flags: first frame, every cut, any gap."""
    cuts, out, prev = set(cuts), [], None
    for f in frames:
        out.append(prev is None or f != prev + 1 or f in cuts)
        prev = f
    return out


def scale_of(meta):
    sx, sy = meta["out"][0] / meta["render"][0], meta["out"][1] / meta["render"][1]
    assert math.isclose(sx, sy, rel_tol=1e-3), f"non-uniform scale {sx} vs {sy}"
    return sx


# The temporal accumulator's constants (baselines/accumulate.py and its GPU port nts/rt), chosen by nts/rt/fit.py
# against native renders (2026-11-27): Gaussian splat footprint (output px), clip-box widening (>= 1), history weight
# cap, weight shrink per unit of clip movement, weight of the bilinear frame in the sum, disocclusion depth tolerance.
# sigma was fitted at x2; accum_sigma() scales it with the render's pixel spacing for the other modes.
ACCUM = dict(sigma=0.586, expand=1.32, wmax=16.8, shrink=11.0, prior=0.0117, rel=0.169)


def accum_sigma(scale):
    """The splat footprint (output px) at a given scale: ACCUM's x2 value in proportion to the render's pixel spacing
    (0.29 at x1, the hand-set value there; only x2 is fitted and measured)."""
    return ACCUM["sigma"] * scale / 2.0
