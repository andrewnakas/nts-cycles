"""Eye-only A/B ladder (stage 2 of the look match, PROVENANCE.md "The look"): the user picks, by eye, which Cycles
variant looks closest to the reference. No number from the reference enters this tool. It only places the reference
image beside the variants on the page.

Variants are relit from the light-group renders (and the skin basis, if rendered) with nts/lookfit.compose, so a
round costs no GPU time. Each round moves one setting at a time by ±step around the current centre. The user's
pick becomes the next centre, and the step halves whenever the centre wins.

Run: ab_ladder.py build <round_dir> <params.json> <lg_dir>[,...] <refs_dir> <ep_f####,...> [--axes a,b,c] [--step 0.3]
       params.json: a lookfit result ({"params": ...}) or a previous round's next.json
       axes: light groups (HeroEyes, set, ...), skin.value / skin.sat / skin.hue, hair, exposure
       -> round_dir/index.html (open locally, pick per frame, "Download results") + key.json (not used by the page)
     ab_ladder.py next <round_dir> results.json  -> round_dir/next.json (the new centre and step) and a tally
     ab_ladder.py build-real <round_dir> <refs_dir> <ep_f####,...> name=<verify_dir> [name=<verify_dir> ...]
       real Cycles renders instead of relit predictions: each verify_dir holds <episode>/f_####.exr made with a look
       fit (datagen/jobs/look_verify.sh); "native" may point at the light-group renders (data/look/lg, <ep>_lg)
"""
import copy
import json
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval import look_brief as lb  # noqa: E402
from nts import lookfit as lf  # noqa: E402

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Closest Look</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#101012;--fg:#eee;--mut:#999;--acc:#6cf}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f6f6f4;--fg:#111;--mut:#555;--acc:#06c}}
:root[data-theme="dark"]{--bg:#101012;--fg:#eee;--mut:#999;--acc:#6cf}
body{background:var(--bg);color:var(--fg);font:15px system-ui;margin:0;padding:16px}
h2{font-size:16px;margin:24px 0 8px}.row{display:flex;flex-wrap:wrap;gap:8px}
figure{margin:0;display:flex;flex-direction:column;align-items:center;gap:4px;cursor:pointer;border:2px solid transparent;
border-radius:8px;padding:4px}figure.ref{cursor:default;border-color:var(--mut)}figure.on{border-color:var(--acc)}
figure img{width:min(240px,44vw);border-radius:4px}figcaption{color:var(--mut);font-size:13px}
textarea{width:100%;max-width:720px;font:inherit;background:transparent;color:var(--fg);border:1px solid var(--mut);
border-radius:6px;padding:6px}button{font:inherit;padding:10px 18px;border-radius:8px;border:1px solid var(--acc);
background:transparent;color:var(--acc);cursor:pointer;margin-top:16px}
</style></head><body>
<p>Round __ROUND__. For each frame, click the variant that looks <b>closest to the reference</b> (outlined grey). The
face zoom is underneath each one. Add a note if none is close ("face still too pale", "too much shadow"…).</p>
__BODY__
<button id="dl">Download results</button>
<script>
const picks={};
document.querySelectorAll('figure[data-v]').forEach(f=>f.onclick=()=>{
  const fr=f.dataset.f;document.querySelectorAll(`figure[data-f="${fr}"]`).forEach(x=>x.classList.remove('on'));
  f.classList.add('on');picks[fr]=f.dataset.v;});
document.getElementById('dl').onclick=()=>{
  const notes={};document.querySelectorAll('textarea').forEach(t=>{if(t.value)notes[t.dataset.f]=t.value});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify({picks,notes},null,1)],{type:'application/json'}));
  a.download='results.json';a.click();};
</script></body></html>"""


def get(p, axis):
    if axis in p["w"]:
        return np.log(p["w"][axis])
    if axis.startswith("skin."):
        k = axis.split(".")[1]
        return p["skin"][k] if k == "hue" else np.log(p["skin"][k])
    if axis == "hair":
        return float(np.mean(np.log(p["hair"])))
    return p["exposure"]


def put(p, axis, val):
    p = copy.deepcopy(p)
    if axis in p["w"]:
        p["w"][axis] = float(np.exp(val))
    elif axis.startswith("skin."):
        k = axis.split(".")[1]
        p["skin"][k] = float(val if k == "hue" else np.exp(val))
    elif axis == "hair":
        d = val - get(p, "hair")
        p["hair"] = [float(h * np.exp(d)) for h in p["hair"]]
    else:
        p["exposure"] = float(val)
    return p


def variants(center, axes, step):
    out = {"centre": center}
    for a in axes:
        s = step * (0.1 if a == "skin.hue" else 1.0)  # hue is in HSV turns, not log units
        for sign, tag in ((1, "+"), (-1, "-")):
            out[f"{a}{tag}"] = put(center, a, get(center, a) + sign * s)
    return out


def face_crop(img, lay):
    y0, y1, x0, x1 = lay.get("face_box", (0, img.height, 0, img.width))
    m = int(max(y1 - y0, x1 - x0) * 0.9)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    return img.crop((cx - m, cy - m, cx + m, cy + m)).resize((480, 480), Image.LANCZOS)


def build(rd, params_path, lg_dirs, refs, keys, axes, step):
    src = json.load(open(params_path))
    center, rnd = src.get("params", src.get("centre")), src.get("round", 0) + 1
    step = src.get("step", step)
    use_basis = isinstance(center["skin"], dict)
    os.makedirs(os.path.join(rd, "img"), exist_ok=True)
    lut = lf.AgxLUT(*json.load(open(os.path.join(lg_dirs[0], "meta.json")))["view"])
    vs = variants(center, axes, step)
    letters = {}
    body = []
    for key in keys:
        ep, f = key.rsplit("_f", 1)
        d = next(x for x in lg_dirs if os.path.basename(os.path.normpath(x)) == f"{ep}_lg")
        path = os.path.join(d, f"f_{f}.exr")
        groups, mist, region, _ = lf.load_lg(path)
        bas = lf.load_basis(path, region) if use_basis else None
        if use_basis and bas is None:
            sys.exit(f"{key}: the centre uses the skin basis but {d}_skin* is missing")
        lay = lb.layout(region)
        ref = Image.open(os.path.join(refs, f"ref_{key}.png")).convert("RGB")
        ref.resize((540, 960), Image.LANCZOS).save(os.path.join(rd, "img", f"{key}_ref.jpg"), quality=92)
        face_crop(ref, lay).save(os.path.join(rd, "img", f"{key}_ref_face.jpg"), quality=92)
        names = list(vs)
        random.shuffle(names)
        letters[key] = {chr(65 + i): n for i, n in enumerate(names)}
        figs = [f'<figure class="ref"><img src="img/{key}_ref.jpg" alt="reference"><img src="img/{key}_ref_face.jpg" '
                f'alt="reference face"><figcaption>reference</figcaption></figure>']
        for letter, n in letters[key].items():
            img = Image.fromarray(lut(lf.compose(groups, mist, region, vs[n], bas)).round().clip(0, 255).astype(np.uint8))
            img.resize((540, 960), Image.LANCZOS).save(os.path.join(rd, "img", f"{key}_{letter}.jpg"), quality=92)
            face_crop(img, lay).save(os.path.join(rd, "img", f"{key}_{letter}_face.jpg"), quality=92)
            figs.append(f'<figure data-f="{key}" data-v="{letter}"><img src="img/{key}_{letter}.jpg" alt="{letter}">'
                        f'<img src="img/{key}_{letter}_face.jpg" alt="{letter} face"><figcaption>{letter}</figcaption>'
                        f'</figure>')
        body.append(f'<h2>{key}</h2><div class="row">{"".join(figs)}</div>'
                    f'<textarea data-f="{key}" rows="2" placeholder="note (optional)"></textarea>')
        print(f"{key}: {len(vs)} variants", flush=True)
    open(os.path.join(rd, "index.html"), "w").write(PAGE.replace("__ROUND__", str(rnd)).replace("__BODY__", "".join(body)))
    json.dump({"round": rnd, "step": step, "axes": axes, "centre": center, "variants": vs, "letters": letters},
              open(os.path.join(rd, "key.json"), "w"), indent=1)
    print(f"round {rnd}: open {os.path.join(rd, 'index.html')}")


def render_png(vdir, ep, f):
    """A verify render's Image through its own view transform (which carries the fit's exposure)."""
    from nts.io_exr import read_layers, stack
    from nts.tonemap import Tonemapper
    d = os.path.join(vdir, ep) if os.path.isdir(os.path.join(vdir, ep)) else os.path.join(vdir, f"{ep}_lg")
    meta = json.load(open(os.path.join(d, "meta.json")))
    ch = read_layers(os.path.join(d, f"f_{int(f):04d}.exr"))
    img = np.nan_to_num(stack(ch, "Image.R", "Image.G", "Image.B"))
    return Image.fromarray(Tonemapper(*meta["view"])(img)), stack(ch, "Region.R", "Region.G", "Region.B")


def build_real(rd, refs, keys, named):
    os.makedirs(os.path.join(rd, "img"), exist_ok=True)
    letters, body = {}, []
    for key in keys:
        ep, f = key.rsplit("_f", 1)
        ref = Image.open(os.path.join(refs, f"ref_{key}.png")).convert("RGB")
        imgs = {n: render_png(d, ep, f) for n, d in named.items()}
        lay = lb.layout(next(iter(imgs.values()))[1])
        ref.resize((540, 960), Image.LANCZOS).save(os.path.join(rd, "img", f"{key}_ref.jpg"), quality=92)
        face_crop(ref, lay).save(os.path.join(rd, "img", f"{key}_ref_face.jpg"), quality=92)
        names = list(named)
        random.shuffle(names)
        letters[key] = {chr(65 + i): n for i, n in enumerate(names)}
        figs = [f'<figure class="ref"><img src="img/{key}_ref.jpg" alt="reference"><img src="img/{key}_ref_face.jpg" '
                f'alt="reference face"><figcaption>reference</figcaption></figure>']
        for letter, n in letters[key].items():
            img = imgs[n][0]
            img.resize((540, 960), Image.LANCZOS).save(os.path.join(rd, "img", f"{key}_{letter}.jpg"), quality=92)
            face_crop(img, lay).save(os.path.join(rd, "img", f"{key}_{letter}_face.jpg"), quality=92)
            figs.append(f'<figure data-f="{key}" data-v="{letter}"><img src="img/{key}_{letter}.jpg" alt="{letter}">'
                        f'<img src="img/{key}_{letter}_face.jpg" alt="{letter} face"><figcaption>{letter}</figcaption>'
                        f'</figure>')
        body.append(f'<h2>{key}</h2><div class="row">{"".join(figs)}</div>'
                    f'<textarea data-f="{key}" rows="2" placeholder="note (optional)"></textarea>')
        print(f"{key}: {len(named)} real renders", flush=True)
    open(os.path.join(rd, "index.html"), "w").write(PAGE.replace("__ROUND__", "real renders").replace(
        "__BODY__", "".join(body)))
    json.dump({"round": 0, "step": 0, "letters": letters, "variants": dict(named)},
              open(os.path.join(rd, "key.json"), "w"), indent=1)
    print(f"open {os.path.join(rd, 'index.html')}")


def next_round(rd, results):
    key = json.load(open(os.path.join(rd, "key.json")))
    res = json.load(open(results))
    votes = {}
    for fr, letter in res["picks"].items():
        n = key["letters"][fr][letter]
        votes[n] = votes.get(n, 0) + 1
    tally = sorted(votes.items(), key=lambda kv: -kv[1])
    print("votes:", tally)
    for fr, note in res.get("notes", {}).items():
        print(f"  note {fr}: {note}")
    win = tally[0][0] if tally else "centre"
    step = key["step"] / 2 if win == "centre" else key["step"]
    json.dump({"round": key["round"], "step": step, "centre": key["variants"][win], "won": win, "votes": votes,
               "notes": res.get("notes", {})}, open(os.path.join(rd, "next.json"), "w"), indent=1)
    print(f"next centre: {win}; step {step:.3f} -> {os.path.join(rd, 'next.json')}")


def main():
    a = sys.argv
    if len(a) >= 4 and a[1] == "next":
        return next_round(a[2], a[3])
    if len(a) >= 6 and a[1] == "build-real":
        return build_real(a[2], a[3], a[4].split(","), dict(x.split("=", 1) for x in a[5:]))
    if len(a) < 7 or a[1] != "build":
        sys.exit(__doc__)
    axes = a[a.index("--axes") + 1].split(",") if "--axes" in a else ["HeroEyes", "HeroKey", "exposure"]
    step = float(a[a.index("--step") + 1]) if "--step" in a else 0.3
    build(a[2], a[3], a[4].split(","), a[5], a[6].split(","), axes, step)


if __name__ == "__main__":
    main()
