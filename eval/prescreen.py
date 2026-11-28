"""Blind pre-screen (NR-1 v2 step 1): picture sheets for a judge who knows nothing about the project.

The judge is a fresh subagent given only <out>/TASK.md and <out>/sheets/. It is used to choose between our own
candidates and to estimate where one stands against the reference before the user spends a round on it. It is trusted only
after calibration: on rounds the user already judged it must side with the user (runs/prescreen_cal.md).

build   python -m eval.prescreen build <out> <spec.json> [--prod] [--seed 0]
        spec: {"items": [{"id": "at140_0150", "variants": {"ours": "<image>", "ref5": "<image>", ...},
                          "faces": {"ours": "<zoom image>", ...}   (optional, ready-made face zooms)
                          "mask": "<m_####.png>"}]}                (optional, face zoom cut from the Region mask)
        --prod finishes every image the way an episode ships (eval.abtest.PROD_VF, needs ffmpeg).
        Writes <out>/sheets/NN_full.jpg and NN_face.jpg (variants left to right = A, B, C, ...; order shuffled per
        item), <out>/key.json (not in sheets/) and <out>/TASK.md.
score   python -m eval.prescreen score <out> <answers.json> [--user <results.json>]
        answers: the judge's JSON list [{"item": "01", "pick": "A", "worst": "B", "sure": 2, "why": "..."}].
        --user: picks by item id {"<id>": {"pick": "<label>"}} from a round the user judged, for agreement.
Needs numpy and Pillow.
"""
import json
import os
import random
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

TASK = """# Which version looks like real camera footage?

Each item NN has two picture files in the `sheets` folder next to this file:
- `NN_full.jpg`: the whole frame
- `NN_face.jpg`: a zoom on the face

Each file shows the SAME moment of the same shot, processed in {n} different ways and placed side by side. From left
to right the versions are {letters}. The order is shuffled for every item, so A in one item has nothing to do with A
in another item.

For every item, look at both files, then decide:
- `pick`: the version that looks MOST like live-action footage of a real person, shot on a real camera
- `worst`: the version that looks LEAST like it (most like a video game or a computer render)

Judge the whole picture: skin, hair, eyes, clothing, the room, colour and light. Do not reward sharpness for its own
sake, and do not mark a version down only for being darker or softer if that is what a camera would give. If two
versions are very close, still choose, and say so with `sure`.

Reply with ONLY a JSON list, one object per item, in item order:

[{{"item": "01", "pick": "A", "worst": "B", "rank": "{rank}", "sure": 2, "why": "what gives the losing version(s) away, 15 words at most"}}]

`rank` lists every version from most real to least real (its first letter is `pick`, its last is `worst`).
`sure` is 1 (coin flip), 2 (fairly sure) or 3 (obvious).
"""


def finish(src, dst):
    from eval.abtest import PROD_VF
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", "scale=1080:1920:flags=bicubic," + PROD_VF,
                    "-frames:v", "1", dst], check=True)


def face_box(mask_path, pad=0.9):
    m = np.asarray(Image.open(mask_path).convert("RGB"))[..., 0] > 127
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    H, W = m.shape
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    h = min(int(max(xs.max() - xs.min(), ys.max() - ys.min()) * pad), W // 2)
    cx, cy = min(max(cx, h), W - h), min(max(cy, h), H - h)
    return (cx - h, cy - h, cx + h, cy + h)


def row(ims, size, gap=10):
    sheet = Image.new("RGB", (len(ims) * size[0] + (len(ims) - 1) * gap, size[1]), (0, 0, 0))
    for i, im in enumerate(ims):
        sheet.paste(im.resize(size, Image.LANCZOS), (i * (size[0] + gap), 0))
    return sheet


def build(out, spec, prod=False, seed=0):
    items = json.load(open(spec))["items"]
    rng = random.Random(seed)
    sd, td = os.path.join(out, "sheets"), os.path.join(out, "tmp")
    os.makedirs(sd, exist_ok=True)
    os.makedirs(td, exist_ok=True)
    key, n = {}, 0
    for k, it in enumerate(items, 1):
        labels = list(it["variants"])
        rng.shuffle(labels)
        n = max(n, len(labels))
        full, face = [], []
        box = face_box(it["mask"]) if it.get("mask") else None
        for lb in labels:
            src = it["variants"][lb]
            if prod:
                dst = os.path.join(td, f"{k:02d}_{len(full)}.png")
                finish(src, dst)
                src = dst
            im = Image.open(src).convert("RGB")
            full.append(im)
            if it.get("faces"):
                face.append(Image.open(it["faces"][lb]).convert("RGB"))
            elif box:
                face.append(im.crop(box))
        pair = len(labels) == 2
        w = 540 if pair else 384
        row(full, (w, w * full[0].height // full[0].width)).save(os.path.join(sd, f"{k:02d}_full.jpg"), quality=93)
        if face:
            fw = 720 if pair else 384
            row(face, (fw, fw)).save(os.path.join(sd, f"{k:02d}_face.jpg"), quality=93)
        key[f"{k:02d}"] = {"id": it["id"], "order": labels}
    json.dump(key, open(os.path.join(out, "key.json"), "w"), indent=1)
    letters = ", ".join("ABCDEFGH"[:n])
    open(os.path.join(out, "TASK.md"), "w").write(TASK.format(n=n, letters=letters, rank="ABCDEFGH"[:n]))
    for f in os.listdir(td):
        os.remove(os.path.join(td, f))
    os.rmdir(td)
    print(f"prescreen: {len(items)} items x {n} versions -> {sd}")


def score(out, answers, user=None):
    key = json.load(open(os.path.join(out, "key.json")))
    ans = json.load(open(answers))
    usr = json.load(open(user)) if user else {}
    best, worst, agree, both, rows, pos, beat = {}, {}, 0, 0, [], {}, {}
    for a in ans:
        k = key.get(str(a["item"]).zfill(2))
        if not k:
            continue
        lab = lambda c: k["order"]["ABCDEFGH".index(c)] if c in "ABCDEFGH"[:len(k["order"])] else None  # noqa: E731
        p, w = lab(a.get("pick", "")), lab(a.get("worst", ""))
        if p:
            best[p] = best.get(p, 0) + 1
        if w:
            worst[w] = worst.get(w, 0) + 1
        order = [lab(c) for c in a.get("rank", "") if lab(c)]
        for i, x in enumerate(order):
            pos.setdefault(x, []).append(i + 1)
            for y in order[i + 1:]:
                beat[(x, y)] = beat.get((x, y), 0) + 1
        u = usr.get(k["id"], {}).get("pick")
        if u and p:
            both += 1
            agree += u == p
        rows.append(f"| {a['item']} | {k['id']} | {p} | {w} | {a.get('sure', '')} | {u or ''} | {a.get('why', '')} |")
    labels = sorted(set(best) | set(worst) | {x for k in key.values() for x in k["order"]})
    print(f"items judged: {len(rows)}")
    for lb in labels:
        mr = f"   mean rank {sum(pos[lb]) / len(pos[lb]):.2f}" if lb in pos else ""
        print(f"  {lb:14s} picked {best.get(lb, 0):3d}   worst {worst.get(lb, 0):3d}{mr}")
    if len(labels) > 2 and beat:
        print("  ranked above (row over column): " + " ".join(f"{y:>9s}" for y in labels))
        for x in labels:
            print(f"  {x:>30s}: " + " ".join(f"{beat.get((x, y), 0):9d}" if x != y else "        -" for y in labels))
    if both:
        print(f"  agreement with the user: {agree}/{both}")
    print("\n| item | id | pick | worst | sure | user | why |\n|---|---|---|---|---|---|---|")
    print("\n".join(rows))


def main():
    a = sys.argv[1:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    if len(a) >= 3 and a[0] == "build":
        build(a[1], a[2], "--prod" in a, int(opt("--seed", 0)))
    elif len(a) >= 3 and a[0] == "score":
        score(a[1], a[2], opt("--user"))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
