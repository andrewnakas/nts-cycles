"""Blind still A/B with notes: "which would you rather ship?" for two sets of 1080x1920 stills with the same
file names (e.g. ours vs the reference on the 24 look-reference stills). Random sides; each pair shows the full frame
and a 2x face crop; a free-text box per pair asks what gave the loser away. Eval only.

Build: python -m eval.abstill build <out_dir> <refs_dir> A=<dir|refs:native_|refs:ref_> B=<...>
       (dirs hold <ep>_f####.png; "refs:<prefix>" reads <refs_dir>/<prefix><ep>_f####.png)
  -> out_dir/index.html + img/*.jpg + key.json. Open index.html, judge, press "Download results".
Score: python -m eval.abstill score <out_dir>/key.json results.json
"""
import glob
import json
import os
import random
import sys

import numpy as np
from PIL import Image

from eval.face import load_mask

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Which Would You Ship</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#101012;--fg:#eee;--mut:#999;--acc:#6cf;--line:#333}
@media (prefers-color-scheme: light){:root{--bg:#f6f6f4;--fg:#111;--mut:#555;--acc:#06c;--line:#ccc}}
body{background:var(--bg);color:var(--fg);font:15px system-ui;margin:0;padding:12px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px;max-width:1400px;margin:auto}
.side{display:flex;flex-wrap:wrap;gap:6px;align-items:flex-start;justify-content:center}.tag{flex-basis:100%;text-align:center}
.side img{max-width:100%;border:1px solid var(--line)}.full{max-height:58vh}.face{max-height:58vh;max-width:60%}
.bar{display:flex;gap:8px;justify-content:center;margin:10px 0;flex-wrap:wrap}
button{font:inherit;padding:10px 18px;border-radius:8px;border:1px solid var(--mut);background:transparent;color:var(--fg);cursor:pointer}
button.on{border-color:var(--acc);color:var(--acc)}
textarea{width:min(900px,96%);display:block;margin:auto;background:transparent;color:var(--fg);border:1px solid var(--mut);border-radius:8px;font:inherit;padding:8px}
h1{font-size:20px;text-align:center;margin:4px}p{color:var(--mut);text-align:center;margin:4px}
.tag{font-weight:700;font-size:22px;letter-spacing:.08em}.side{padding:6px;border:3px solid transparent;border-radius:10px}
.side.on{border-color:var(--acc)}.side.on .tag::after{content:"  ✔ YOUR PICK";color:var(--acc)}
</style></head><body>
<h1>Click the one you would rather ship: the one that looks more like real footage.</h1>
<p>Your pick gets a blue frame. Both are the same shot; one side is ours, one is reference, sides are random.</p><p id="pos"></p>
<div class="bar"><button data-c="left">&#9664; Ship LEFT</button><button data-c="same">Can't tell</button><button data-c="right">Ship RIGHT &#9654;</button></div>
<textarea id="note" rows="2" placeholder="What looked fake in the one you REJECTED? (skin, hair, eyes, clothes, light, colour, ...)"></textarea>
<div class="bar"><button id="prev">&#9664; Prev</button><button id="next">Next &#9654;</button><button id="dl">Download results</button></div>
<div class="pair"><div class="side" id="sl" data-s="left"><div class="tag">LEFT</div><img id="lf" class="face"><img id="l" class="full"></div>
<div class="side" id="sr" data-s="right"><div class="tag">RIGHT</div><img id="rf" class="face"><img id="r" class="full"></div></div>
<script>
const P=__PAIRS__,K="abstill___ROUND__";let i=0,ch={};try{ch=JSON.parse(localStorage.getItem(K)||"{}")}catch(e){}
const $=id=>document.getElementById(id),note=$("note");
function save(){try{localStorage.setItem(K,JSON.stringify(ch))}catch(e){}}
function show(){const p=P[i];$("l").src=p.left;$("r").src=p.right;$("lf").src=p.lface;$("rf").src=p.rface;
 const c=ch[p.id]||{};note.value=c.note||"";
 $("pos").textContent=`pair ${i+1} of ${P.length} · ${Object.values(ch).filter(v=>v.pick).length} judged`;
 document.querySelectorAll("[data-c]").forEach(b=>b.classList.toggle("on",c.pick===b.dataset.c));
 document.querySelectorAll("[data-s]").forEach(d=>d.classList.toggle("on",c.pick===d.dataset.s))}
function pick(v){const id=P[i].id;ch[id]={...(ch[id]||{}),pick:v};save();show()}
note.oninput=()=>{const id=P[i].id;ch[id]={...(ch[id]||{}),note:note.value};save()};
document.querySelectorAll("[data-c]").forEach(b=>b.onclick=()=>pick(b.dataset.c));
document.querySelectorAll("[data-s]").forEach(d=>{d.style.cursor="pointer";d.onclick=()=>pick(d.dataset.s)});
$("prev").onclick=()=>{if(i>0)i--;show()};$("next").onclick=()=>{if(i<P.length-1)i++;show()};
$("dl").onclick=()=>{const a=document.createElement("a");
 a.href=URL.createObjectURL(new Blob([JSON.stringify(ch,null,1)],{type:"application/json"}));a.download="results___ROUND__.json";a.click()};
show();
</script></body></html>"""


def resolve(spec, refs, key):
    if spec.startswith("refs:"):
        return os.path.join(refs, spec[5:] + key + ".png")
    return os.path.join(spec, key + ".png")


def face_crop(im, mask_path):
    face = load_mask(mask_path)[0]
    ys, xs = np.nonzero(face)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    s = int(max(np.ptp(xs), np.ptp(ys)) * 0.75) + 20
    s = min(s, im.width // 2, im.height // 2)
    cx, cy = min(max(cx, s), im.width - s), min(max(cy, s), im.height - s)
    return im.crop((cx - s, cy - s, cx + s, cy + s)).resize((420, 420), Image.LANCZOS)


def build(out, refs, a, b):
    (na, sa), (nb, sb) = a.split("=", 1), b.split("=", 1)
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    rng = random.Random()
    pairs, key = [], {}
    for nat in sorted(glob.glob(os.path.join(refs, "native_*.png"))):
        k = os.path.basename(nat)[7:-4]
        ep, fr = k.rsplit("_f", 1)
        pa, pb, mk = resolve(sa, refs, k), resolve(sb, refs, k), os.path.join(refs, f"m_{ep}_{fr}.png")
        if not (os.path.exists(pa) and os.path.exists(pb) and os.path.exists(mk)):
            continue
        ids = {}
        for lab, p in ((na, pa), (nb, pb)):
            cid = f"{rng.getrandbits(32):08x}"
            im = Image.open(p).convert("RGB")
            im.save(os.path.join(out, "img", cid + ".jpg"), quality=93)
            face_crop(im, mk).save(os.path.join(out, "img", cid + "f.jpg"), quality=93)
            ids[lab] = cid
        flip = rng.random() < 0.5
        L, R = (nb, na) if flip else (na, nb)
        pid = f"{rng.getrandbits(32):08x}"
        pairs.append({"id": pid, "left": f"img/{ids[L]}.jpg", "right": f"img/{ids[R]}.jpg",
                      "lface": f"img/{ids[L]}f.jpg", "rface": f"img/{ids[R]}f.jpg"})
        key[pid] = {"still": k, "left": L, "right": R}
    rng.shuffle(pairs)
    rid = f"{rng.getrandbits(24):06x}"
    open(os.path.join(out, "index.html"), "w").write(PAGE.replace("__PAIRS__", json.dumps(pairs)).replace("__ROUND__", rid))
    json.dump({"a": na, "b": nb, "round": rid, "pairs": key}, open(os.path.join(out, "key.json"), "w"), indent=1)
    print(f"abstill: {len(pairs)} pairs -> {os.path.join(out, 'index.html')}")


def score(key_path, res_path):
    key, res = json.load(open(key_path)), json.load(open(res_path))
    wins = {key["a"]: 0, key["b"]: 0, "same": 0}
    for pid, c in res.items():
        k = key["pairs"].get(pid)
        if not k or not c.get("pick"):
            continue
        who = "same" if c["pick"] == "same" else k[c["pick"]]
        wins[who] += 1
        print(f"{k['still']:26s} picked: {who:12s} rejected-side note: {c.get('note', '')}")
    n = sum(wins.values())
    print(f"\n{n} judged: " + ", ".join(f"{k} {v} ({v / max(n, 1):.0%})" for k, v in wins.items()))


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 4:
        sys.exit(__doc__)
    if a[1] == "score":
        score(a[2], a[3])
    else:
        build(a[2], a[3], a[4], a[5])
