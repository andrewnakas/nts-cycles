"""Pack a head-box render (render_lr mode "roi") into one small file of head crops, for the generative realism model.

Run: pack_heads.py <roi_dir> <out.npz>
Per rendered frame f (crop = that frame's head box from roi.json), float16 unless noted:
  <f>_rgb    scene-linear colour      <f>_alb  albedo        <f>_nrm  normal
  <f>_depth  linear camera distance   <f>_mv   motion to the previous frame, image space px (current -> previous)
  <f>_reg    uint8 region mask (R face, G hands, B hair)
plus frames, boxes ([x0, y0, x1, y1] per frame), view (transform, look, exposure) and meta (json string).
Frames come in consecutive pairs (f, f+1), so <f+1>_mv links the pair for temporal training.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts import conventions as cv  # noqa: E402
from nts.io_exr import read_layers, stack  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    boxes = json.load(open(os.path.join(src, "roi.json")))["frames"]
    meta = json.load(open(os.path.join(src, "meta.json")))
    out, frames, bx = {}, [], []
    for p in sorted(glob.glob(os.path.join(src, "f_*.exr"))):
        f = int(os.path.basename(p)[2:6])
        if str(f) not in boxes:
            continue
        x0, y0, x1, y1 = boxes[str(f)]
        ch = read_layers(p)
        c = lambda a: np.ascontiguousarray(a[y0:y1, x0:x1])  # noqa: E731
        rgb = np.nan_to_num(c(stack(ch, "Image.R", "Image.G", "Image.B")), nan=0.0, posinf=0.0, neginf=0.0)
        if rgb.max() <= 0:  # the border did not cover this frame (should not happen)
            continue
        out[f"{f}_rgb"] = np.clip(rgb, 0, 6e4).astype(np.float16)
        out[f"{f}_alb"] = c(stack(ch, "Albedo.R", "Albedo.G", "Albedo.B")).astype(np.float16)
        out[f"{f}_nrm"] = c(stack(ch, "Normal.X", "Normal.Y", "Normal.Z")).astype(np.float16)
        out[f"{f}_depth"] = np.clip(c(ch["Depth.V"]), 0, 1e4).astype(np.float16)
        out[f"{f}_mv"] = cv.mv_image(c(stack(ch, "Vector.X", "Vector.Y"))).astype(np.float16)
        out[f"{f}_reg"] = (np.clip(c(stack(ch, "Region.R", "Region.G", "Region.B")), 0, 1) * 255 + 0.5).astype(np.uint8)
        frames.append(f)
        bx.append([x0, y0, x1, y1])
    if not frames:
        sys.exit(f"pack_heads: nothing usable in {src}")
    np.savez_compressed(dst, frames=np.array(frames), boxes=np.array(bx), view=np.array(meta["view"], dtype=object),
                        meta=json.dumps(meta), **out)
    print(f"pack_heads: {len(frames)} frames, boxes {bx[0][2] - bx[0][0]}..{max(b[2] - b[0] for b in bx)} px -> {dst}")


if __name__ == "__main__":
    main()
