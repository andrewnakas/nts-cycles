"""Pack a guide-mode render (render_lr --nts guide: few-sample full-res albedo + normal) next to a packed shot, in the
same frame order (the guide folder's sorted frames, which is pack.py's order for the same range):
<pack_dir>/<name>.npy (T, 6, H, W) f16. Run it BEFORE pack.py, so a pack is complete once pack.py logs it. This is the realistic, cheap guide NTS-1 sees at inference;
guide.npy (from the native render) is the converged version.

Run: pack_guide.py <guide_exr_dir> <pack_dir> [name=guide16]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts.io_exr import read_layers, stack  # noqa: E402


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, pack = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else "guide16"
    import glob
    import re
    frames = sorted(int(re.search(r"f_(\d+)\.exr$", q).group(1)) for q in glob.glob(os.path.join(src, "f_*.exr")))
    os.makedirs(pack, exist_ok=True)
    out = None
    for i, f in enumerate(frames):
        ch = read_layers(os.path.join(src, f"f_{f:04d}.exr"))
        a = np.nan_to_num(stack(ch, "Albedo.R", "Albedo.G", "Albedo.B", "Normal.X", "Normal.Y", "Normal.Z"))
        if out is None:
            out = np.lib.format.open_memmap(os.path.join(pack, f"{name}.npy"), "w+", np.float16,
                                            (len(frames), 6) + a.shape[:2])
        out[i] = a.transpose(2, 0, 1)
    out.flush()
    print(f"pack_guide: {len(frames)} frames -> {pack}/{name}.npy")


if __name__ == "__main__":
    main()
