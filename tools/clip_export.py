"""Export an eval clip in the NR-1 clip layout, so the student runs on eval clips exactly as it
trains on datagen/jobs/gen_frames.sh CLIP=12 output (2026-11-19).

Run (4050, ~/scene-project/py): clip_export.py <tag> <out_dir> [--look v9d]
Reads, under data/eval/ab/<tag>/ (zd400 included):
  <look>/f_####.png          the render (our input)
  masks/m_####.png           R face, G hands, B hair
  gbuf_<look>/f_####.*       albedo / normal / depth images (nts/gbuf.py); optional, the student sees zeros without
and data/raw/<tag>_<look>/   the slim EXRs: Vector + Depth, cuts.json.
Writes <out>/<tag>_<f>.jpg, .reg.png, .alb.jpg / .nrm.png / .dep.png, and .mv.npy (float16, 1/8 size, full-size pixels,
current -> previous, as tools/frames_to_train.py) except on the clip's first frame and on cuts.
"""
import glob
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts.io_exr import Sequence  # noqa: E402


def main():
    a = sys.argv
    if len(a) < 3:
        sys.exit(__doc__)
    tag, out = a[1], a[2]
    look = a[a.index("--look") + 1] if "--look" in a else "v9d"
    ab = os.path.join("data/eval/ab", tag)
    seq = Sequence(os.path.join("data/raw", f"{tag}_{look}"))
    os.makedirs(out, exist_ok=True)
    n = nmv = ngb = 0
    prev = None
    for f in seq.frames:
        src = os.path.join(ab, look, f"f_{f:04d}.png")
        if not os.path.exists(src):
            prev = None
            continue
        st = os.path.join(out, f"{tag}_{f}")
        Image.open(src).convert("RGB").save(st + ".jpg", quality=95)
        m = os.path.join(ab, "masks", f"m_{f:04d}.png")
        if os.path.exists(m):
            r = (np.asarray(Image.open(m).convert("RGB")) > 127).astype(np.uint8) * 255
            Image.fromarray(r).save(st + ".reg.png")
        gb = glob.glob(os.path.join(ab, f"gbuf_{look}", f"f_{f:04d}.*"))
        for p in gb:
            shutil.copyfile(p, st + p[p.index("."):])
        ngb += bool(gb)
        fr = seq.read(f)
        if prev == f - 1 and not fr.reset:
            mv = fr.mv.astype(np.float32)
            if seq.scale != 1:
                raise SystemExit(f"{tag}: EXRs at 1/{seq.scale} size; the NR-1 layout expects full-size renders")
            h, w = mv.shape[:2]
            q = [np.asarray(Image.fromarray(mv[..., i]).resize((w // 8, h // 8), Image.BILINEAR)) for i in (0, 1)]
            np.save(st + ".mv.npy", np.stack(q, -1).astype(np.float16))
            nmv += 1
        prev = f
        n += 1
    print(f"clip_export {tag}: {n} frames, {nmv} with motion, {ngb} with G-buffers -> {out}")


if __name__ == "__main__":
    main()
