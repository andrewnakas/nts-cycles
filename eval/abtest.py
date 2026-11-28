"""Blind A/B: "which looks more real?" Ours vs the reference (or any two variants), unlabeled, random sides.

Build: abtest.py build <out_dir> --a ours=<testset variant label> --b ref5=<label> eval/testset.json [--crop x0,y0,x1,y1]
   or: abtest.py build <out_dir> --a ours=<dir> --b ref5=<dir> --dirs <shot name> [--face x0,y0,x1,y1]
       (two frame directories on this machine instead of a testset; --face adds a zoomed face clip under each)
       more: --shots "typing:400-438,wide:439-459" (one pair per shot), --masks <dir of m_####.png> (face zoom box
       per shot from the masks), --widths 1080,720 (one pair per delivery width), --prod (the episode finishing
       chain of scene-project cutscene/finish.py and CRF 23 on both sides), --vf <ffmpeg filters>, --crf N
  -> out_dir/index.html + clips/*.mp4 + key.json (which side is which; not referenced by the page)
  Open index.html locally and judge every pair (one-tap reasons for the rejected side are optional). The page
  downloads results_<round>.json by itself after the last pick and shows a one-line code to paste into a chat.
Score: abtest.py score <out_dir>/key.json <results.json | "AB-<round> ...code...">  -> per-shot and overall win rates
"""
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval.metrics import frames_in  # noqa: E402

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Which Would You Ship</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#101012;--fg:#eee;--mut:#999;--acc:#6cf}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f6f6f4;--fg:#111;--mut:#555;--acc:#06c}}
body{background:var(--bg);color:var(--fg);font:15px system-ui;margin:0;padding:16px}
.pair{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px;max-width:1400px;margin:auto}
.side{display:flex;flex-wrap:wrap;gap:6px;align-items:flex-start;justify-content:center;padding:6px;border:3px solid transparent;border-radius:10px;cursor:pointer}
.side{min-width:0}.side video{background:#000;max-width:100%}.full{max-height:58vh}.face{max-height:58vh;max-width:60%}
.tag{font-weight:700;font-size:22px;letter-spacing:.08em;flex-basis:100%;text-align:center}.side.on{border-color:var(--acc)}
.side.on .tag::after{content:"  ✔ YOUR PICK";color:var(--acc)}
textarea{width:min(900px,96%);display:block;margin:auto;background:transparent;color:var(--fg);border:1px solid var(--mut);border-radius:8px;font:inherit;padding:8px}
.bar{display:flex;gap:8px;justify-content:center;margin:12px 0}button{font:inherit;padding:10px 18px;border-radius:8px;
border:1px solid var(--mut);background:transparent;color:var(--fg);cursor:pointer}button.on{border-color:var(--acc);color:var(--acc)}
h1{font-size:18px;text-align:center;margin:4px}p{color:var(--mut);text-align:center;margin:4px}
.chips{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin:6px 0}.chips button{padding:6px 12px;border-radius:999px}
.chips span{flex-basis:100%;text-align:center;color:var(--mut)}
#done{display:none;max-width:900px;margin:10px auto;padding:10px;border:2px solid var(--acc);border-radius:10px;text-align:center}
#code{width:96%;font:13px ui-monospace,monospace;background:transparent;color:var(--fg);border:1px solid var(--mut);border-radius:6px;padding:6px}
</style></head><body>
<h1>Click the clip you would rather ship: the one that looks more like real footage (and steadier).</h1>
<p>Your pick gets a blue frame. Same shot on both sides; one is ours (in some pairs a test mix), one is reference, sides are random.
The results file saves itself after the last pick.</p><p id="pos"></p>
<div class="bar"><button data-c="left">&#9664; Ship LEFT</button><button data-c="same">Can't tell</button><button data-c="right">Ship RIGHT &#9654;</button></div>
<div class="chips" id="why"><span>Optional: what was wrong with the one you REJECTED? Tap any that apply.</span>__CHIPS__</div>
<textarea id="note" rows="1" placeholder="Anything else about the one you REJECTED (optional)"></textarea>
<div id="done"><b>All pairs judged. Your results file was saved to Downloads.</b><br>You can also copy this line into the chat:<br>
<input id="code" readonly onclick="this.select()"> <button id="copy">Copy</button></div>
<div class="pair"><div class="side" data-s="left"><div class="tag">LEFT</div><video id="lf" class="face" muted loop playsinline></video><video id="l" class="full" muted loop playsinline></video></div>
<div class="side" data-s="right"><div class="tag">RIGHT</div><video id="rf" class="face" muted loop playsinline></video><video id="r" class="full" muted loop playsinline></video></div></div>
<div class="bar"><button id="prev">◀ Prev</button><button id="next">Next ▶</button><button id="dl">Download results</button></div>
<script>
const P=__PAIRS__,K="ab___ROUND__";let i=0,ch={};try{ch=JSON.parse(localStorage.getItem(K)||"{}")}catch(e){}
const $=id=>document.getElementById(id),L=$("l"),V=["r","lf","rf"].map($),note=$("note");
function code(){return "AB-__ROUND__ "+P.map(p=>{const c=ch[p.id]||{};return c.pick?p.id.slice(0,4)+"="+c.pick[0].toUpperCase()+(c.why||[]).map(w=>"."+w).join(""):""}).filter(x=>x).join(" ")}
function judged(){return P.filter(p=>(ch[p.id]||{}).pick).length}
function dl(){const a=document.createElement("a");
 a.href=URL.createObjectURL(new Blob([JSON.stringify(ch,null,1)],{type:"application/json"}));a.download="results___ROUND__.json";a.click()}
function save(){try{localStorage.setItem(K,JSON.stringify(ch))}catch(e){};$("code").value=code();$("done").style.display=judged()==P.length?"block":"none"}
function show(){const p=P[i];L.src=p.left;$("r").src=p.right;
 $("lf").src=p.lface||"";$("rf").src=p.rface||"";$("lf").hidden=$("rf").hidden=!p.lface;
 [L,...V].forEach(v=>{if(v.src)v.play()});const c=ch[p.id]||{};note.value=c.note||"";
 $("pos").textContent=`pair ${i+1} of ${P.length} · ${Object.values(ch).filter(v=>v.pick).length} judged`;
 document.querySelectorAll("[data-c]").forEach(b=>b.classList.toggle("on",c.pick===b.dataset.c));
 document.querySelectorAll("[data-w]").forEach(b=>b.classList.toggle("on",(c.why||[]).includes(b.dataset.w)));
 document.querySelectorAll("[data-s]").forEach(d=>d.classList.toggle("on",c.pick===d.dataset.s))}
function sync(){V.forEach(v=>{if(v.src&&Math.abs(v.currentTime-L.currentTime)>0.05)v.currentTime=L.currentTime})}
L.addEventListener("seeked",sync);L.addEventListener("play",()=>{sync();V.forEach(v=>{if(v.src)v.play()})});
setInterval(sync,500);
function pick(v){const id=P[i].id,was=judged();ch[id]={...(ch[id]||{}),pick:v};save();show();
 if(was<P.length&&judged()==P.length)dl()}
document.querySelectorAll("[data-w]").forEach(b=>b.onclick=()=>{const id=P[i].id,c=ch[id]||{},w=new Set(c.why||[]);
 w.has(b.dataset.w)?w.delete(b.dataset.w):w.add(b.dataset.w);ch[id]={...c,why:[...w]};save();show()});
$("copy").onclick=()=>{$("code").select();try{navigator.clipboard.writeText($("code").value)}catch(e){document.execCommand("copy")}};
note.oninput=()=>{const id=P[i].id;ch[id]={...(ch[id]||{}),note:note.value};save()};
document.querySelectorAll("[data-c]").forEach(b=>b.onclick=()=>pick(b.dataset.c));
document.querySelectorAll("[data-s]").forEach(d=>d.onclick=()=>pick(d.dataset.s));
$("prev").onclick=()=>{if(i>0)i--;show()};$("next").onclick=()=>{if(i<P.length-1)i++;show()};
$("dl").onclick=dl;
save();show();
</script></body></html>"""


WHY = {"sk": "skin", "ha": "hair", "ey": "eyes", "co": "colours", "li": "lighting", "so": "too soft", "fl": "flicker",
       "st": "stubble", "ar": "odd detail", "cg": "looks CG"}  # one-tap reasons for the rejected side
PROD_VF = ("gblur=sigma=0.35,eq=contrast=1.07:saturation=1.08,colorbalance=rs=-0.04:gs=-0.01:bs=0.05:rh=0.05:gh=0.01:bh=-0.04,"
           "vignette=angle=0.45,noise=alls=7:allf=t")  # scene-project cutscene/finish.py, the look every episode ships with
OPT = {"vf": "", "crf": "12", "width": 1080}


def face_box(masks, frames, pad=0.9):
    """Square box around the face over these frames, from the Region masks (R = face), in 1080x1920 pixels."""
    import numpy as np
    from PIL import Image
    x0 = y0 = 10 ** 6
    x1 = y1 = 0
    for f in frames[::3]:
        m = np.asarray(Image.open(os.path.join(masks, f"m_{f:04d}.png")).convert("RGB"))[..., 0] > 127
        ys, xs = np.nonzero(m)
        if len(xs):
            x0, y0, x1, y1 = min(x0, xs.min()), min(y0, ys.min()), max(x1, xs.max()), max(y1, ys.max())
    cx, cy, h = (x0 + x1) // 2, (y0 + y1) // 2, int(max(x1 - x0, y1 - y0) * pad) // 2 * 2
    h = min(h, 540)  # close-ups (pty332): the face fills the width, so the zoom is the whole width, square
    cx, cy = min(max(cx, h), 1080 - h), min(max(cy, h), 1920 - h)
    return (cx - h, cy - h, cx + h, cy + h)


def clip(src_dir, frames, dst, crop):
    fr = frames_in(src_dir)
    lst = dst + ".txt"
    with open(lst, "w") as fp:
        for f in frames:
            fp.write(f"file '{os.path.abspath(fr[f])}'\nduration {1 / 30:.6f}\n")
    # finish the whole frame first (as the episode would be), then scale to the delivery width, then crop
    k = OPT["width"] / 1080
    vf = "scale=1080:1920:flags=bicubic" + ("," + OPT["vf"] if OPT["vf"] else "") + (f",scale={OPT['width']}:-2:flags=bicubic" if k != 1 else "")
    if crop:
        c = [int(v * k) // 2 * 2 for v in crop]
        vf += f",crop={c[2] - c[0]}:{c[3] - c[1]}:{c[0]}:{c[1]}"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-vf", vf + ",pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-r", "30", "-c:v", "libx264", "-crf", OPT["crf"], "-pix_fmt", "yuv420p", dst], check=True)
    os.remove(lst)


def build(out, a, b, spec, crop, face=None, dirs=None, shots=None, masks=None, widths=(1080,)):
    (na, la), (nb, lb) = a.split("=", 1), b.split("=", 1)
    os.makedirs(os.path.join(out, "clips"), exist_ok=True)
    pairs, key = [], {}
    rng = random.Random()
    if dirs:  # --shots "name:a-b,name2:c-d" splits the two folders at cuts; each shot x each delivery width is a pair
        parts = [x.split(":") for x in shots.split(",")] if shots else [(dirs, None)]
        todo = [{"name": f"{n} @{w}", "variants": {la: la, lb: lb}, "span": r, "width": w} for n, r in parts for w in widths]
    else:
        todo = json.load(open(spec))["shots"]  # [{"name", "variants": {label: dir}, "span": "a-b", "width", "masks"}]
    for sh in todo:
        va, vb = sh["variants"].get(la), sh["variants"].get(lb)
        if not (va and vb):
            continue
        common = sorted(set(frames_in(va)) & set(frames_in(vb)))
        if sh.get("span"):
            lo, hi = map(int, sh["span"].split("-"))
            common = [f for f in common if lo <= f <= hi]
        if not common:
            continue
        OPT["width"] = sh.get("width", 1080)
        if sh.get("masks") or masks:  # a testset shot may name its own mask folder (one page across several clips)
            face = face_box(sh.get("masks") or masks, common)
        ids = [f"{rng.getrandbits(32):08x}" for _ in range(2)]
        for d, cid in ((va, ids[0]), (vb, ids[1])):
            clip(d, common, os.path.join(out, "clips", f"{cid}.mp4"), crop)
            if face:
                clip(d, common, os.path.join(out, "clips", f"{cid}f.mp4"), face)
        pid = f"{rng.getrandbits(32):08x}"
        flip = rng.random() < 0.5
        left, right = (ids[1], ids[0]) if flip else (ids[0], ids[1])
        pairs.append({"id": pid, "left": f"clips/{left}.mp4", "right": f"clips/{right}.mp4"})
        if face:
            pairs[-1].update(lface=f"clips/{left}f.mp4", rface=f"clips/{right}f.mp4")
        key[pid] = {"shot": sh["name"], "left": nb if flip else na, "right": na if flip else nb}
    rng.shuffle(pairs)
    rid = f"{rng.getrandbits(24):06x}"
    chips = "".join(f'<button data-w="{k}">{v}</button>' for k, v in WHY.items())
    open(os.path.join(out, "index.html"), "w").write(
        PAGE.replace("__PAIRS__", json.dumps(pairs)).replace("__ROUND__", rid).replace("__CHIPS__", chips))
    json.dump({"a": na, "b": nb, "round": rid, "pairs": key}, open(os.path.join(out, "key.json"), "w"), indent=1)
    print(f"abtest: {len(pairs)} pairs -> {os.path.join(out, 'index.html')}")


def read_results(src, key):
    """A results JSON from the page, or the one-line code it shows ("AB-<round> <id4>=<L|R|S>[.reason]...")."""
    if os.path.exists(src):
        return json.load(open(src))
    res = {}
    for tok in src.split()[1:]:
        k, v = tok.split("=", 1)
        pid = [p for p in key["pairs"] if p.startswith(k)]
        parts = v.split(".")
        if pid:
            res[pid[0]] = {"pick": {"L": "left", "R": "right", "S": "same"}[parts[0]], "why": parts[1:]}
    return res


def score(key_path, res_src):
    key = json.load(open(key_path))
    res = read_results(res_src, key)
    wins = {key["a"]: 0, key["b"]: 0, "same": 0}
    why = {}
    for pid, c in res.items():
        k = key["pairs"].get(pid)
        if not k:
            continue
        c = c if isinstance(c, dict) else {"pick": c}
        if not c.get("pick"):
            continue
        who = "same" if c["pick"] == "same" else k[c["pick"]]
        wins[who] += 1
        lost = "" if who == "same" else k["right" if c["pick"] == "left" else "left"]
        rs = [WHY.get(w, w) for w in c.get("why", [])]
        for r in rs:
            why.setdefault(lost, {}).setdefault(r, 0)
            why[lost][r] += 1
        print(f"{k['shot']:24s} picked: {who:12s} rejected {lost}: {', '.join(rs)} {c.get('note', '')}")
    n = sum(wins.values())
    print(f"\n{n} judged: " + ", ".join(f"{k} {v} ({v / max(n, 1):.0%})" for k, v in wins.items()))
    for lost, d in why.items():
        print(f"reasons against {lost}: " + ", ".join(f"{r} x{c}" for r, c in sorted(d.items(), key=lambda x: -x[1])))


def main():
    a = sys.argv
    if len(a) < 3:
        sys.exit(__doc__)
    if a[1] == "score":
        return score(a[2], a[3])
    box = lambda k: tuple(map(int, a[a.index(k) + 1].split(","))) if k in a else None  # noqa: E731
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    dirs = opt("--dirs")
    spec = None if dirs else [x for x in a[3:] if x.endswith(".json")][0]
    if "--prod" in a:  # the episode's own finishing chain and encode quality on both sides
        OPT["vf"], OPT["crf"] = PROD_VF, "23"
    OPT["vf"], OPT["crf"] = opt("--vf", OPT["vf"]), opt("--crf", OPT["crf"])
    build(a[2], a[a.index("--a") + 1], a[a.index("--b") + 1], spec, box("--crop"), box("--face"), dirs,
          opt("--shots"), opt("--masks"), tuple(map(int, opt("--widths", "1080").split(","))))


if __name__ == "__main__":
    main()
