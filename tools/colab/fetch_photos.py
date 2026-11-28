# Colab VM: CC0 / CC-BY portrait photos from Openverse (api.openverse.org) for NTS-R stage B, face-cropped.
# Writes /content/photos/faces.npy (N, 3, 256, 256) uint8 sRGB: square crops 2.1x the detected face box (OpenCV Haar),
# so a face spans ~120 px like ours in a 256 crop; and /content/photos/manifest.json (creator, licence, source URL
# for every photo used: attribution for CC-BY). Only licence cc0 and by are requested. Run: python3 fetch_photos.py [N]
import io, json, os, sys, time
import cv2, numpy as np, requests
from PIL import Image

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
Q = ["portrait", "face portrait", "headshot", "man portrait", "woman portrait", "office worker", "programmer",
     "person smiling", "old man portrait", "old woman portrait", "young man", "young woman", "people indoors",
     "student portrait", "business portrait", "candid portrait", "african portrait", "asian portrait", "selfie"]
det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
os.makedirs("/content/photos", exist_ok=True)
faces, man, seen = [], [], set()
for q in Q:
    for page in range(1, 13):
        if len(faces) >= N:
            break
        try:
            r = requests.get("https://api.openverse.org/v1/images/", timeout=30, params={
                "q": q, "license": "cc0,by", "page": page, "page_size": 20, "category": "photograph"}).json()
        except Exception:
            time.sleep(5); continue
        for it in r.get("results", []):
            if it["url"] in seen:
                continue
            seen.add(it["url"])
            try:
                im = Image.open(io.BytesIO(requests.get(it["url"], timeout=20).content)).convert("RGB")
            except Exception:
                continue
            im.thumbnail((1600, 1600))
            a = np.asarray(im)
            for (x, y, w, h) in det.detectMultiScale(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), 1.1, 6, minSize=(80, 80))[:2]:
                s = int(w * 2.1) // 2; cx, cy = x + w // 2, y + h // 2
                if cx - s < 0 or cy - s < 0 or cx + s > a.shape[1] or cy + s > a.shape[0]:
                    continue
                faces.append(np.asarray(Image.fromarray(a[cy - s:cy + s, cx - s:cx + s]).resize((256, 256), Image.LANCZOS)))
                man.append({k: it.get(k) for k in ("creator", "license", "license_version", "foreign_landing_url", "url")})
        print(q, page, len(faces), flush=True)
np.save("/content/photos/faces.npy", np.stack(faces).transpose(0, 3, 1, 2))
json.dump(man, open("/content/photos/manifest.json", "w"))
print("FETCH_DONE", len(faces))
