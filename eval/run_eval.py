"""Score every variant of every test shot against native Cycles: whole frame, face identity, skin tone, flicker, and
the first frames after a cut. Writes a markdown table (and JSON) so ours vs the reference reads at a glance.

Run: run_eval.py eval/testset.json [--out runs/eval] [--shots a,b]
testset.json: {"shots": [{"name", "gt": dir, "cuts": file, "frames": [a, b], "static": [a, b] (optional),
               "masks": dir (m_####.png) | "face_box": [x0,y0,x1,y1], "hands_box": [...] (optional),
               "variants": {"label": dir, ...}}]}
Reference frames come in through an import script that is not in this repo and are just another variant: evaluation only.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval import face as fm  # noqa: E402
from eval.metrics import frames_in, load, luma, psnr, sharpness, ssim  # noqa: E402

COLS = [("psnr", "PSNR", "{:.2f}"), ("ssim", "SSIM", "{:.4f}"), ("terr", "tErr", "{:.3f}"), ("sharp", "sharp", "{:.2f}"),
        ("face_psnr", "face PSNR", "{:.2f}"), ("face_hp_psnr", "face struct", "{:.2f}"),
        ("face_sym_delta", "face sym Δ", "{:+.2f}"), ("face_terr", "face swim", "{:.3f}"),
        ("skin_tone_delta", "skin Δ", "{:.2f}"), ("flicker", "static flicker", "{:.3f}"),
        ("postcut_psnr", "post-cut PSNR", "{:.2f}")]


def score_shot(sh):
    gt = frames_in(sh["gt"])
    cuts = set(json.load(open(sh["cuts"]))) if sh.get("cuts") and os.path.exists(sh["cuts"]) else set()
    a, b = sh.get("frames", [min(gt), max(gt)])
    static = sh.get("static")
    res = {}
    for label, d in sh["variants"].items():
        te = frames_in(d)
        fs = [f for f in sorted(set(gt) & set(te)) if a <= f <= b]
        if not fs:
            res[label] = {"frames": 0}
            continue
        rows, prev, stack = [], None, []
        last_cut = None
        for f in fs:
            g = load(gt[f])
            t = load(te[f], (g.shape[1], g.shape[0]))
            if f in cuts or last_cut is None:
                last_cut = f if f in cuts else fs[0]
            r = {"psnr": psnr(g, t), "ssim": ssim(g, t), "sharp": sharpness(t) / max(sharpness(g), 1e-6)}
            if sh.get("masks"):
                mp = os.path.join(sh["masks"], f"m_{f:04d}.png")
                face, hands = fm.load_mask(mp, (g.shape[1], g.shape[0])) if os.path.exists(mp) else (None, None)
            else:
                face = fm.box_mask(g.shape, sh["face_box"]) if sh.get("face_box") else None
                hands = fm.box_mask(g.shape, sh["hands_box"]) if sh.get("hands_box") else None
            if face is not None and face.sum() > 100:
                r.update(fm.frame_metrics(g, t, face, hands))
            if prev and prev[0] == f - 1 and f not in cuts:
                r["terr"] = float(np.mean(np.abs((t - prev[2]) - (g - prev[1]))))
                if face is not None and face.sum() > 100:
                    r["face_terr"] = fm.face_temporal(prev[1], g, prev[2], t, face)
            if f - last_cut < 4 and last_cut in cuts:
                r["postcut_psnr"] = r["psnr"]
            if static and static[0] <= f <= static[1]:
                if f in cuts:
                    stack.append(None)  # a cut ends a static stretch: shimmer is measured within shots only
                stack.append((luma(g), luma(t)))
            prev = (f, g, t)
            rows.append(r)
        agg = {k: float(np.nanmean([r[k] for r in rows if k in r])) for k, _, _ in COLS if any(k in r for r in rows)}
        segs, cur = [], []
        for s in stack + [None]:
            if s is None:
                if len(cur) > 2:
                    segs.append(cur)
                cur = []
            else:
                cur.append(s)
        if segs:
            vals, ns = [], []
            for seg in segs:
                gs, ts = np.stack([s[0] for s in seg]), np.stack([s[1] for s in seg])
                gstd = gs.std(0)
                mask = gstd < np.percentile(gstd, 60)
                vals.append(float(ts.std(0)[mask].mean()))
                ns.append(len(seg))
            agg["flicker"] = float(np.average(vals, weights=ns))
        agg["frames"] = len(fs)
        res[label] = agg
    return res


def table(name, res):
    have = [c for c in COLS if any(c[0] in r for r in res.values())]
    lines = [f"### {name}", "", "| variant | " + " | ".join(h for _, h, _ in have) + " | frames |",
             "|---|" + "---|" * (len(have) + 1)]
    for label, r in res.items():
        cells = [fmt.format(r[k]) if k in r and r[k] == r[k] else "–" for k, _, fmt in have]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {r.get('frames', 0)} |")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    spec = json.load(open(sys.argv[1]))
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "runs/eval"
    only = set(sys.argv[sys.argv.index("--shots") + 1].split(",")) if "--shots" in sys.argv else None
    os.makedirs(out, exist_ok=True)
    allres, md = {}, ["# NTS eval vs native Cycles", "",
                      "Higher is better for PSNR/SSIM/face struct; lower for tErr, face swim, skin Δ, flicker; "
                      "sharp ≈ 1 and face sym Δ ≈ 0 mean 'like native'.", ""]
    for sh in spec["shots"]:
        if only and sh["name"] not in only:
            continue
        allres[sh["name"]] = res = score_shot(sh)
        md.append(table(sh["name"], res))
        print(md[-1], flush=True)
    open(os.path.join(out, "eval.md"), "w").write("\n".join(md))
    json.dump(allres, open(os.path.join(out, "eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
