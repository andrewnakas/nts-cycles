"""Motion-compensated flicker: how much a face changes from frame to frame beyond what the real motion explains.
eval.face.face_temporal compares each method's change with the reference render's change, so any change of look (a
darker face, more texture) counts as flicker once the head moves. Here the previous frame of each method is warped to
the current one with optical flow computed on the reference clip (same geometry for every method), and the residual
is measured inside the eroded face mask, in 8-bit display units. Lower = steadier. Eval only.
Run: python -m eval.flicker_flow <root> <a-b> ref=<dir> label=<dir> ...
"""
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from eval.face import load_mask

root, span = sys.argv[1], sys.argv[2]
methods = [a.split("=", 1) for a in sys.argv[3:]]
a, b = map(int, span.split("-"))
L = lambda d, f: np.asarray(Image.open(f"{root}/{d}/f_{f:04d}.png").convert("RGB")).astype(np.float32)
ref = methods[0][1]
flows = {}
for f in range(a + 1, b + 1):
    g0 = cv2.cvtColor(L(ref, f).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(L(ref, f - 1).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    flows[f] = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 4, 21, 5, 7, 1.5, 0)  # current -> previous
print("| method | face residual | face residual p90 |\n|---|---|---|")
for lab, d in methods:
    res = []
    for f in range(a + 1, b + 1):
        face = binary_erosion(load_mask(f"{root}/masks/m_{f:04d}.png")[0], iterations=4)
        fl = flows[f]
        h, w = fl.shape[:2]
        mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        prev = cv2.remap(L(d, f - 1), mx + fl[..., 0], my + fl[..., 1], cv2.INTER_LINEAR)
        res.append(np.abs(L(d, f) - prev).mean(-1)[face])
    r = np.concatenate(res)
    print(f"| {lab} | {r.mean():.2f} | {np.percentile(r, 90):.2f} |")
