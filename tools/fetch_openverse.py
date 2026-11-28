"""Real-photo side of the generative realism model: portrait photos under CC0 / Public Domain Mark / CC BY only,
from the Openverse API. Each photo's licence, creator and source page are kept in photos.jsonl (CC BY needs the
attribution; the clean-room rule needs the record).

Run: fetch_openverse.py <out_dir> [--per-query 400] [--max 12000] [--queries "a;b;c"] [--source flickr,stocksnap]
  -> out_dir/img/<id>.jpg (longest side <= 1280) + out_dir/photos.jsonl (one line per kept photo)
Credentials: ~/.config/openverse/register.json (client_id, client_secret); without a verified app the API allows
only a few hundred requests a day. Resumable: photos already in photos.jsonl are skipped. Needs requests, Pillow.
"""
import io
import json
import os
import sys
import time

import requests
from PIL import Image

API = "https://api.openverse.org/v1"
UA = {"User-Agent": "render-realism-photo-set/1.0"}
QUERIES = ["portrait man", "portrait woman", "headshot", "man face", "woman face", "man smiling", "man talking",
           "person speaking", "office worker portrait", "programmer portrait", "young man portrait",
           "old man portrait", "elderly woman portrait", "man beard portrait", "student portrait", "street portrait",
           "candid portrait", "man glasses portrait", "woman glasses portrait", "interview person", "speaker conference",
           "man looking at camera", "people at desk", "man short hair", "man profile face", "teenager portrait",
           "businessman", "scientist portrait", "engineer portrait", "worker portrait"]


def token():
    c = json.load(open(os.path.expanduser("~/.config/openverse/register.json")))
    r = requests.post(f"{API}/auth_tokens/token/", data={"client_id": c["client_id"], "client_secret": c["client_secret"],
                                                         "grant_type": "client_credentials"}, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def get(url, hdr, **params):
    for wait in (0, 5, 30, 120):
        time.sleep(wait)
        r = requests.get(url, params=params, headers=hdr, timeout=40)
        if r.status_code == 429:
            continue
        return r
    return r


def main():
    a = sys.argv
    if len(a) < 2:
        sys.exit(__doc__)
    out = a[1]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    per_q, cap = int(opt("--per-query", 400)), int(opt("--max", 12000))
    queries = opt("--queries", None)
    queries = queries.split(";") if queries else QUERIES
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    log = os.path.join(out, "photos.jsonl")
    have = {json.loads(l)["id"] for l in open(log)} if os.path.exists(log) else set()
    try:
        hdr = {**UA, "Authorization": f"Bearer {token()}"}
    except Exception as e:  # unverified or missing app: anonymous limits
        print("no token, anonymous access:", e)
        hdr = dict(UA)
    page_size = 20 if "Authorization" not in hdr else 50
    n = len(have)
    with open(log, "a") as fp:
        for q in queries:
            got = 0
            for page in range(1, per_q // page_size + 2):
                # modern photo sites only: CC0 "portrait" searches are otherwise museum paintings and antique prints
                r = get(f"{API}/images/", hdr, q=q, license="cc0,by", source=opt("--source", "flickr,stocksnap"),
                        page_size=page_size, page=page, mature="false")
                if r.status_code != 200:
                    print(f"{q!r} page {page}: HTTP {r.status_code} {r.text[:120]}")
                    break
                res = r.json().get("results", [])
                if not res:
                    break
                for it in res:
                    if it["id"] in have or it.get("license") not in ("cc0", "by"):
                        continue
                    if min(it.get("width") or 0, it.get("height") or 0) < 600:
                        continue
                    try:
                        b = requests.get(it["url"], headers=UA, timeout=40).content
                        im = Image.open(io.BytesIO(b)).convert("RGB")
                    except Exception:
                        continue
                    sm = im.resize((64, 64)).convert("HSV")
                    if sum(sm.getdata(1)) / 4096 < 25:  # black-and-white or faded: not what a phone camera sees
                        continue
                    im.thumbnail((1280, 1280), Image.LANCZOS)
                    im.save(os.path.join(out, "img", it["id"] + ".jpg"), quality=92)
                    fp.write(json.dumps({k: it.get(k) for k in ("id", "title", "creator", "creator_url", "license",
                                                                "license_version", "license_url", "foreign_landing_url",
                                                                "source", "url")} | {"query": q}) + "\n")
                    fp.flush()
                    have.add(it["id"])
                    n += 1
                    got += 1
                if got >= per_q or n >= cap:
                    break
            print(f"{q!r}: +{got} (total {n})", flush=True)
            if n >= cap:
                break
    print(f"fetch_openverse: {n} photos in {out}")


if __name__ == "__main__":
    main()
