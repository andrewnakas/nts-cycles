"""CC0 / CC-BY photos of people in rooms at our framing, from Openverse, for the frame-level and cloth realism probes
(eval/variety.py). Eval only: these are never training targets.

Each kept photo has one Haar face whose height is 3-20 % of a 9:16 portrait crop around it (our stills: face about
90 px of 1920, close-ups excepted). Writes into <out_dir>:
  frames.npy  (N, 480, 270, 3) uint8 sRGB, the 9:16 crop resized
  cloth.npy   (M, 224, 224, 3) uint8, square torso crop under the face (where it fits)
  manifest.json  creator, licence, source URL per photo (CC-BY attribution)
Run (Mac): python tools/fetch_scenes.py <out_dir> [N=900]
"""
import io
import json
import os
import sys
import time

import cv2
import numpy as np
import requests
from PIL import Image

out, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 900
Q = ["man sitting at desk computer", "person working at computer office", "programmer at desk", "man standing indoors",
     "woman standing indoors", "people talking office", "man in t-shirt indoors", "person giving presentation",
     "man talking room", "woman working laptop", "student at desk", "man in kitchen", "person in living room",
     "coworkers office", "man gesturing", "developer office", "startup office people", "man in hoodie",
     "person in server room", "man in bedroom", "people in meeting room", "woman talking", "man explaining whiteboard",
     "person standing portrait full body", "candid man indoors", "candid woman indoors", "streamer at desk",
     "gamer at computer", "office worker standing", "man at kitchen counter"]
det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
os.makedirs(out, exist_ok=True)
frames, cloth, man, seen = [], [], [], set()
sess = requests.Session()
sess.headers["User-Agent"] = "nts-eval-fetch/0.1"
LIC_OK = ("CC0", "Public domain", "PD", "CC BY 1", "CC BY 2", "CC BY 3", "CC BY 4")  # no SA / NC / ND


def commons(q, page):
    """Wikimedia Commons search (fallback while Openverse returns 504), filtered to CC0 / PD / CC BY."""
    r = sess.get("https://commons.wikimedia.org/w/api.php", timeout=30, params={
        "action": "query", "generator": "search", "gsrsearch": q + " filetype:bitmap", "gsrnamespace": 6,
        "gsrlimit": 40, "gsroffset": 40 * (page - 1), "prop": "imageinfo", "iiprop": "url|extmetadata",
        "iiurlwidth": 1600, "iiextmetadatafilter": "LicenseShortName|Artist", "format": "json"}).json()
    out = []
    for p in r.get("query", {}).get("pages", {}).values():
        ii = p["imageinfo"][0]
        md = ii.get("extmetadata", {})
        lic = md.get("LicenseShortName", {}).get("value", "")
        if lic.startswith(LIC_OK):
            out.append({"url": ii.get("thumburl") or ii["url"], "creator": md.get("Artist", {}).get("value", ""),
                        "license": lic, "license_version": "", "foreign_landing_url": ii.get("descriptionurl")})
    return out


SRC = os.environ.get("SRC", "commons")
if SRC.endswith(".json"):  # re-fetch the originals behind an earlier manifest (Openverse is down; direct URLs work)
    _m = json.load(open(SRC))
    _u = list({x["url"]: x for x in _m}.values())
    Q = [None]
for q in Q:
    for page in range(1, 9):
        if len(frames) >= N:
            break
        try:
            if SRC.endswith(".json"):
                res = _u[(page - 1) * 160:page * 160]
            elif SRC == "openverse":
                res = sess.get("https://api.openverse.org/v1/images/", timeout=30, params={
                    "q": q, "license": "cc0,by", "page": page, "page_size": 20, "category": "photograph"}).json().get("results", [])
            else:
                res = commons(q, page)
        except Exception as e:
            print("api error", e, flush=True)
            time.sleep(5)
            continue
        for it in res:
            if it["url"] in seen or len(frames) >= N:
                continue
            seen.add(it["url"])
            try:
                im = Image.open(io.BytesIO(sess.get(it["url"], timeout=20).content)).convert("RGB")
            except Exception:
                continue
            im.thumbnail((1600, 1600))
            a = np.asarray(im)
            H, W = a.shape[:2]
            fs = det.detectMultiScale(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), 1.1, 6, minSize=(24, 24))
            if len(fs) == 0:
                continue
            x, y, w, h = max(fs, key=lambda f: f[2])
            ch = min(H, int(W * 16 / 9))  # tallest 9:16 crop that fits
            cw = int(ch * 9 / 16)
            if not 0.025 <= h / ch <= 0.30:
                continue
            cx = min(max(x + w // 2 - cw // 2, 0), W - cw)
            cy = min(max(y + h // 2 - int(0.35 * ch), 0), H - ch)  # face in the upper part, like our shots
            frames.append(np.asarray(Image.fromarray(a[cy:cy + ch, cx:cx + cw]).resize((270, 480), Image.LANCZOS)))
            s, tx, ty = int(2.4 * w), x + w // 2, y + int(1.4 * h)
            if tx - s // 2 >= 0 and tx + s // 2 <= W and ty + s <= H:
                cloth.append(np.asarray(Image.fromarray(a[ty:ty + s, tx - s // 2:tx + s // 2]).resize((224, 224), Image.LANCZOS)))
            man.append({k: it.get(k) for k in ("creator", "license", "license_version", "foreign_landing_url", "url")})
        print(q, page, len(frames), len(cloth), flush=True)
        if page % 2 == 0:
            np.save(os.path.join(out, "frames.npy"), np.stack(frames)) if frames else None
np.save(os.path.join(out, "frames.npy"), np.stack(frames))
np.save(os.path.join(out, "cloth.npy"), np.stack(cloth))
json.dump(man, open(os.path.join(out, "manifest.json"), "w"))
print("FETCH_DONE", len(frames), len(cloth))
