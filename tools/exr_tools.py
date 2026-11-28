"""Small jobs on render_lr EXR folders (4050, ~/scene-project/py), one frame at a time so nothing large is held.

  exr_tools.py png <exr_dir> <out_dir> [--ev 0.3] [--masks <dir>] [--gbuf <dir>]
                                    AgX f_####.png (view from meta.json + ev); with --masks, the Region AOV as m_####.png;
                                    with --gbuf, f_####.alb.jpg / .nrm.png / .dep.png (nts/gbuf.py, for NR-1)
  exr_tools.py gbuf <exr_dir> <out_dir>   only the G-buffer images (cheap re-renders of clips whose EXRs were stripped)
  exr_tools.py strip <exr_dir>      rewrite every EXR keeping only Depth, Vector and Region (what the reference pass
                                    and the motion metrics need): about a tenth of the size. The colour layers are lost.
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts.io_exr import read_layers, stack  # noqa: E402

KEEP = ("Depth.", "Vector.", "Region.")


def gbuf(src, dst):
    from nts.gbuf import save_gbuf
    os.makedirs(dst, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(src, "f_*.exr"))):
        n += bool(save_gbuf(read_layers(p), os.path.join(dst, os.path.basename(p)[:6])))
    print(f"gbuf: {n} frames -> {dst}")


def png(src, dst, ev, masks, gb=None):
    from nts.tonemap import Tonemapper
    v = json.load(open(os.path.join(src, "meta.json")))["view"]
    v[2] += ev
    tm = Tonemapper(*v)
    os.makedirs(dst, exist_ok=True)
    if masks:
        os.makedirs(masks, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(src, "f_*.exr"))):
        ch = read_layers(p)
        f = os.path.basename(p)[2:6]
        rgb = np.nan_to_num(stack(ch, "Image.R", "Image.G", "Image.B"), nan=0.0, posinf=0.0, neginf=0.0)
        Image.fromarray(tm(rgb)).save(os.path.join(dst, f"f_{f}.png"))
        if masks and "Region.R" in ch:
            m = np.clip(stack(ch, "Region.R", "Region.G", "Region.B"), 0, 1)
            Image.fromarray((m * 255 + 0.5).astype(np.uint8)).save(os.path.join(masks, f"m_{f}.png"))
        if gb:
            from nts.gbuf import save_gbuf
            os.makedirs(gb, exist_ok=True)
            save_gbuf(ch, os.path.join(gb, f"f_{f}"))
        n += 1
    print(f"png: {n} frames -> {dst}")


def strip(src):
    import OpenImageIO as oiio
    before = after = 0
    for p in sorted(glob.glob(os.path.join(src, "f_*.exr"))):
        ch = read_layers(p)
        names = [n for n in ch if n.startswith(KEEP)]
        if len(names) == len(ch):
            continue
        before += os.path.getsize(p)
        h, w = ch[names[0]].shape
        spec = oiio.ImageSpec(w, h, len(names), "half")
        spec.channelnames = names
        spec.attribute("compression", "zip")
        tmp = p + ".tmp.exr"
        out = oiio.ImageOutput.create(tmp)
        out.open(tmp, spec)
        out.write_image(np.stack([ch[n] for n in names], -1).astype(np.float32))
        out.close()
        os.replace(tmp, p)
        after += os.path.getsize(p)
    print(f"strip: {src} {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB")


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 3:
        sys.exit(__doc__)
    if a[1] == "strip":
        strip(a[2])
    elif a[1] == "gbuf":
        gbuf(a[2], a[3])
    else:
        png(a[2], a[3], float(a[a.index("--ev") + 1]) if "--ev" in a else 0.3,
            a[a.index("--masks") + 1] if "--masks" in a else None, a[a.index("--gbuf") + 1] if "--gbuf" in a else None)
