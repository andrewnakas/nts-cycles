"""The tone step's change, held steady in time (design log row 116).

The tone step decides frame by frame which outline pixels are skin and which are rim light or background. On the
far side of a neck (and under an ear) that decision flips between two states from one frame to the next, so the
toned frame flickers where the steadied base does not (user, A/B round afadaf, rs345 and ti620: "the left side of
neck and below ear is totally flickering"). Here the CHANGE the tone step made (toned / base, per channel) is
averaged along the renderer's motion, forward and backward within a shot, and put back on the base frame: every
detail still comes from the current frame, only the change is held. A surface point keeps one change through the
shot, which is what a per-shot tone curve means. Well inside the renderer's own face and hands masks the per-frame
change stays: eyes and teeth have their own rules there, and they open and close.

  python -m nts.tone_steady <geom_exr_dir> <base_dir> <toned_dir> <masks_dir> <out_dir>
         [--alpha 0.15]   weight of the current frame in each direction (1 = off: the output equals the toned frame)
         [--face-r 0]     px the masks are shrunk by before they are left alone (0 = 4 % of the face's height, >= 6)
         [--frames a-b]
         [--also <build_dir> <out_dir2>]   a build made on the toned frames (a carry result) gets the same
                                           correction: out2 = build + (steadied - toned)
         [--neck-fade 0 --weights <tone weights dir>]   with --also: what the build added (build - toned) is faded
                                           out on the skin the tone step found beyond the renderer's own face
                                           mask (the neck), from the mask's edge over this fraction of the face's
                                           height. Under a moving jaw the carried detail is covered and uncovered
                                           from frame to frame: a band under the chin that flickers (ti620).
Reads our own frames, masks and motion only. Nothing is deleted; it refuses to write into a folder with frames."""
import glob
import os
import sys

import numpy as np
from PIL import Image

from nts import warp as W
from nts.io_exr import Sequence
from nts.tone import box, maxr

C = 4.0  # offset of the ratio: keeps it tame in the blacks


def frame(d, f, stem="f"):
    p = [q for q in glob.glob(os.path.join(d, f"{stem}_{f:04d}.*")) if q[-4:] in (".png", ".jpg")]
    return np.asarray(Image.open(p[0]).convert("RGB")).astype(np.float32) if p else None


def zone(mask, r):
    """1 where the change is held in time, 0 well inside the renderer's face and hands masks (mask 0..1, R face,
    G hands, B hair; the fringe over the forehead is hair)."""
    solid = (((mask[..., 0] > 0.5) & (mask[..., 2] < 0.5)) | (mask[..., 1] > 0.5)).astype(np.float32)
    if r <= 0:
        ys = np.nonzero(solid.max(1) > 0.5)[0]
        r = max(6, int(round(0.04 * (ys[-1] - ys[0] + 1)))) if len(ys) else 6
    inner = 1 - maxr(1 - solid, r)  # the masks shrunk by r px
    return 1 - np.clip(box(inner, max(1, r // 2)), 0, 1)


def held(seq, frames, change, rev, alpha):
    """The change of every frame averaged along the motion in one direction: {frame: (H, W, 3) float16}."""
    acc = prev_depth = last = None
    out = {}
    for f in (frames[::-1] if rev else frames):
        R = change(f)
        fr = seq.read(f)
        hw = R.shape[:2]
        depth = W.upsample_nearest(fr.depth, seq.scale, hw)
        if rev:
            cut = last != f + 1 or bool(seq.reset.get(f + 1, True))
        else:
            cut = last != f - 1 or bool(seq.reset[f])
        if acc is None or cut:
            acc = R
        else:
            mv = W.hr_motion(fr.mv_next if rev else fr.mv, fr.depth, seq.scale, hw)
            occ = W.disocclusion(prev_depth, depth, mv)
            acc = np.where(occ[..., None], R, alpha * R + (1 - alpha) * W.warp(acc, mv)).astype(np.float32)
        prev_depth, last = depth, f
        out[f] = acc.astype(np.float16)
    return out


def neck(mask, wface, frac):
    """0..1: skin by the tone step's weight (wface) that lies beyond the renderer's face and hair masks, rising
    from their edge over frac of the face's height."""
    head = ((mask[..., 0] > 0.5) | (mask[..., 2] > 0.5)).astype(np.float32)
    ys = np.nonzero(head.max(1) > 0.5)[0]
    r = max(2, int(round(frac * (ys[-1] - ys[0] + 1) / 2))) if len(ys) else 2
    return np.clip(box(wface * (1 - maxr(head, r)), r), 0, 1)


def run(geom, base_dir, toned_dir, masks_dir, out_dir, alpha=0.15, face_r=0, span=None, also=None, neck_fade=0.0,
        weights_dir=None):
    seq = Sequence(geom)
    frames = sorted(int(os.path.basename(p)[2:6]) for p in glob.glob(os.path.join(toned_dir, "f_*.png")))
    frames = [f for f in frames if f in seq.files and (span is None or span[0] <= f <= span[1])]
    for d in [out_dir] + ([also[1]] if also else []):
        assert not glob.glob(os.path.join(d, "f_*.png")), d + " already has frames: pick another name"
        os.makedirs(d, exist_ok=True)

    def change(f):
        return (frame(toned_dir, f) + C) / (frame(base_dir, f) + C)

    fwd = held(seq, frames, change, False, alpha)
    acc = prev_depth = last = None
    moved = []
    for f in frames[::-1]:  # the backward direction, combined and written as it goes
        base, toned = frame(base_dir, f), frame(toned_dir, f)
        R = (toned + C) / (base + C)
        fr = seq.read(f)
        hw = R.shape[:2]
        depth = W.upsample_nearest(fr.depth, seq.scale, hw)
        cut = last != f + 1 or bool(seq.reset.get(f + 1, True))
        if acc is None or cut:
            acc = R
        else:
            mv = W.hr_motion(fr.mv_next, fr.depth, seq.scale, hw)
            occ = W.disocclusion(prev_depth, depth, mv)
            acc = np.where(occ[..., None], R, alpha * R + (1 - alpha) * W.warp(acc, mv)).astype(np.float32)
        prev_depth, last = depth, f
        m = frame(masks_dir, f, "m")
        S = zone(np.zeros_like(R) if m is None else m / 255.0, face_r)[..., None]
        A = 0.5 * (fwd.pop(f).astype(np.float32) + acc)
        out = (base + C) * (R + S * (A - R)) - C
        Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(out_dir, f"f_{f:04d}.png"))
        moved.append(float(np.abs(out - toned).mean()))
        if also:
            b = frame(also[0], f)
            if b is not None:
                o2 = b + (out - toned)
                if neck_fade > 0 and weights_dir and m is not None:
                    wf = frame(weights_dir, f, "w")
                    if wf is not None:
                        o2 = o2 - neck(m / 255.0, wf[..., 0] / 255.0, neck_fade)[..., None] * (b - toned)
                Image.fromarray(np.clip(o2 + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(also[1], f"f_{f:04d}.png"))
    print("tone_steady: %d frames -> %s; alpha %.2f; mean change against the toned frames %.3f levels (largest frame %.3f)%s" % (
        len(frames), out_dir, alpha, float(np.mean(moved)) if moved else 0, max(moved) if moved else 0,
        "; also -> " + also[1] if also else ""))


def main():
    a = sys.argv[1:]
    if len(a) < 5:
        print(__doc__)
        return 1

    def opt(k, d=None):
        return a[a.index(k) + 1] if k in a else d
    span = tuple(map(int, opt("--frames").split("-"))) if "--frames" in a else None
    also = (a[a.index("--also") + 1], a[a.index("--also") + 2]) if "--also" in a else None
    run(a[0], a[1], a[2], a[3], a[4], float(opt("--alpha", 0.15)), int(opt("--face-r", 0)), span, also,
        float(opt("--neck-fade", 0.0)), opt("--weights"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
