"""Rank look-search candidates by how close their broad appearance is to a visual reference (blurred CIELAB ΔE),
weighted toward the head (face + hair + a margin for neck/ears) and hands. Fine texture is blurred away on purpose:
we want the reference's lighting and skin, not its invented grain. The reference is only compared against, never
trained on.

Run: look_score.py <search_dir> <reference.png> <mask.png> [--top 10]
     look_score.py <search_dir,...> --refs <refs_dir> [--top 10]
       multi-frame: refs_dir holds ref_f####.png (the visual reference) and m_####.png per frame; each search dir
       holds <cand>_f####.png. A candidate's score is the mean over every frame it rendered (all dirs).
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval.face import lab_ab  # noqa: E402
from eval.metrics import box, load  # noqa: E402

W, H = 1080, 1920


def lab(img):
    c = img / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    L = np.where(y > 0.008856, 116 * np.cbrt(y) - 16, 903.3 * y)
    return np.concatenate([L[..., None], lab_ab(img)], -1)


def blurred_lab(img, r=8):
    return np.stack([box(lab(img)[..., k], r) for k in range(3)], -1)


def weights(mask_png):
    a = np.asarray(Image.open(mask_png).convert("RGB").resize((W, H), Image.NEAREST), np.float32) / 255
    head = np.clip(box(np.maximum(a[..., 0], a[..., 2]), 40) * 4, 0, 1)   # face + hair, grown to neck and ears
    hands = np.clip(box(a[..., 1], 20) * 4, 0, 1)
    return 1.0 + 6.0 * head + 3.0 * hands, head > 0.5, hands > 0.5


def score(cand, ref_lab, wt, head, hands):
    d = np.linalg.norm(blurred_lab(cand) - ref_lab, axis=-1)
    return {"total": float((d * wt).sum() / wt.sum()), "head": float(d[head].mean()), "hands": float(d[hands].mean()),
            "rest": float(d[~(head | hands)].mean())}


def main_multi():
    dirs = sys.argv[1].split(",")
    refs = sys.argv[sys.argv.index("--refs") + 1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 10
    per = {}
    meta = {}
    for d in dirs:
        meta.update({c["name"]: c for c in json.load(open(os.path.join(d, "search.json")))["candidates"]})
        for p in sorted(glob.glob(os.path.join(d, "*_f*.png"))):
            name, f = os.path.splitext(os.path.basename(p))[0].rsplit("_f", 1)
            rp = os.path.join(refs, f"ref_{os.path.basename(d)}_f{f}.png")
            mp = os.path.join(refs, f"m_{os.path.basename(d)}_{f}.png")
            if not (os.path.exists(rp) and os.path.exists(mp)):
                continue
            key = (d, f)
            if key not in per:
                per[key] = (blurred_lab(load(rp, (W, H))),) + weights(mp)
            per.setdefault(name, []).append(score(load(p, (W, H)), *per[key]))
    rows = []
    for name, c in meta.items():
        sc = per.get(name)
        if isinstance(sc, list) and sc:
            agg = {k: float(np.mean([s[k] for s in sc])) for k in sc[0]}
            agg["worst"] = float(max(s["total"] for s in sc))
            agg["frames"] = len(sc)
            rows.append((agg, c))
    rows.sort(key=lambda r: r[0]["total"])
    json.dump([{"score": s, "cand": c} for s, c in rows], open(os.path.join(dirs[0], "ranked_multi.json"), "w"), indent=1)
    for s, c in rows[:top]:
        print(f"{c['name']:8s} ΔE mean {s['total']:.2f} worst {s['worst']:.2f} head {s['head']:.2f} hands {s['hands']:.2f} "
              f"rest {s['rest']:.2f} ({s['frames']} frames)  " + json.dumps({k: v for k, v in c.items() if k != 'name'}))


def main():
    if "--refs" in sys.argv:
        return main_multi()
    src, ref, mask = sys.argv[1:4]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 10
    ref_lab = blurred_lab(load(ref, (W, H)))
    wt, head, hands = weights(mask)
    meta = {c["name"]: c for c in json.load(open(os.path.join(src, "search.json")))["candidates"]}
    rows = []
    for p in sorted(glob.glob(os.path.join(src, "*.png"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name in meta:
            rows.append((score(load(p, (W, H)), ref_lab, wt, head, hands), meta[name]))
    rows.sort(key=lambda r: r[0]["total"])
    json.dump([{"score": s, "cand": c} for s, c in rows], open(os.path.join(src, "ranked.json"), "w"), indent=1)
    for s, c in rows[:top]:
        print(f"{c['name']:8s} ΔE total {s['total']:.2f} head {s['head']:.2f} hands {s['hands']:.2f} rest {s['rest']:.2f}  "
              + json.dumps({k: v for k, v in c.items() if k != 'name'}))


if __name__ == "__main__":
    main()
