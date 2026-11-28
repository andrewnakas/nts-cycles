"""Stage A: pack one render's EXR frames into a compressed fp16 npz and delete the EXRs (the 4050 has little disk).
  input  (full): rgb (T,3,H,W), albedo (T,3), normal (T,3), depth (T,1), mv (T,2: Vector.X/Y, render px), region (T,H,W) u8
  target (--rgb-only): rgb (T,3,H,W)
Run: stage_a_pack.py <exr_dir> <out.npz> [--rgb-only]"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts.io_exr import read_layers, stack  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
rgb_only = "--rgb-only" in sys.argv
files = sorted(glob.glob(os.path.join(src, "f_*.exr")))
if not files:
    sys.exit(f"stage_a_pack: no EXRs in {src}")
keys = {"rgb": ("Image.R", "Image.G", "Image.B")}
if not rgb_only:
    keys.update(albedo=("Albedo.R", "Albedo.G", "Albedo.B"), normal=("Normal.X", "Normal.Y", "Normal.Z"),
                depth=("Depth.V",), mv=("Vector.X", "Vector.Y"))
arr = {k: [] for k in keys}
region = []
for f in files:
    ch = read_layers(f)
    for k, names in keys.items():
        a = np.nan_to_num(stack(ch, *names), nan=0.0, posinf=1e4, neginf=0.0).astype(np.float16)
        arr[k].append(np.moveaxis(a.reshape(a.shape[0], a.shape[1], -1), -1, 0))  # (H, W, C) -> (C, H, W)
    if not rgb_only and "Region.R" in ch:
        r = stack(ch, "Region.R", "Region.G", "Region.B") > 0.5
        region.append((r[..., 0] * 1 + r[..., 1] * 2 + r[..., 2] * 4).astype(np.uint8))
data = {k: np.stack(v) for k, v in arr.items()}
if region:
    data["region"] = np.stack(region)
data["frames"] = np.array([int(os.path.basename(f)[2:6]) for f in files])
np.savez_compressed(out, **data)
for f in files:
    os.remove(f)
print(f"stage_a_pack: {len(files)} frames -> {out} {os.path.getsize(out) / 1e6:.0f} MB")
