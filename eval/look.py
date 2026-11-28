"""How a variant's look differs from native Cycles, per region (face / hands / rest): ratios of brightness, contrast,
fine detail, broad shading and saturation. Used to say what the reference realism pass actually does, and how close a
clean change (skin material, enhancer) gets to it.

Run: look.py <gt_dir> <masks_dir> <frames a-b[:step]> label=dir [label=dir ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval.face import lab_ab, load_mask  # noqa: E402
from eval.metrics import box, frames_in, load, luma  # noqa: E402

KEYS = ("bright", "contrast", "detail", "shading", "sat")


def ratios(g, t, m):
    lg, lt = luma(g), luma(t)
    hg, ht = lg - box(lg, 6), lt - box(lt, 6)
    mg, mt = box(lg, 24), box(lt, 24)
    cg, ct = np.hypot(*lab_ab(g)[m].T).mean(), np.hypot(*lab_ab(t)[m].T).mean()
    r = {"bright": lt[m].mean() / lg[m].mean(), "contrast": lt[m].std() / lg[m].std(),
            "detail": ht[m].std() / hg[m].std(), "shading": mt[m].std() / mg[m].std(), "sat": ct / cg}
    return {k: float(v) for k, v in r.items()}


def main():
    gt, masks, span = sys.argv[1], sys.argv[2], sys.argv[3]
    rng, _, step = span.partition(":")
    a, b = map(int, rng.split("-"))
    gfr = frames_in(gt)
    for arg in sys.argv[4:]:
        lab, d = arg.split("=", 1)
        te = frames_in(d)
        acc = {}
        for f in range(a, b + 1, int(step or 1)):
            if f not in te or f not in gfr:
                continue
            g, t = load(gfr[f]), load(te[f], None)
            face, hands = load_mask(os.path.join(masks, f"m_{f:04d}.png"))
            for name, m in (("face", face), ("hands", hands), ("rest", ~(face | hands))):
                if m.sum() > 100:
                    for k, v in ratios(g, t, m).items():
                        acc.setdefault(name, {}).setdefault(k, []).append(v)
        for name, r in acc.items():
            print(f"{lab:14s} {name:6s} " + "  ".join(f"{k} {np.mean(r[k]):.2f}" for k in KEYS))


if __name__ == "__main__":
    main()
