"""Pick training shots across every episode except the test one: one 100-frame window from the longest shot of each
distinct (episode, set, camera) so the data covers every set, framing and character. Skips windows already in train_v0
and the zero-day test episode (its sets are used: every episode is shot in them, so the zero-day test measures unseen
shots in known sets). Output is round-robin across episodes, so a partial run still covers the most variety. Prints lines "<episode> <a-b> <name>" for datagen/jobs/train_v1.sh.

Run (on the 4050): py tools/pick_shots.py [--len 100] [--max-per-ep 3]
"""
import glob
import json
import os
import sys

FPS = 30
TEST = {"zero-day", "zero-day-ref", "zero-day-native_aa"}
USED = {("agentic-time", 135), ("agentic-time", 280), ("last-programmer", 398), ("ram-shortage", 582),
        ("ram-shortage", 894)}
a = sys.argv
L = int(a[a.index("--len") + 1]) if "--len" in a else 100
MAXN = int(a[a.index("--max-per-ep") + 1]) if "--max-per-ep" in a else 3
per_ep = []
for tl in sorted(glob.glob(os.path.expanduser("~/scene-project/*/timeline.json"))):
    ep = os.path.basename(os.path.dirname(tl))
    if ep in TEST:
        continue
    t = json.load(open(tl))
    fps = t.get("fps", FPS)
    best = {}
    for line in t["lines"]:
        for sh in line.get("shots", []):
            a0, a1 = round(sh["t0"] * fps) + 1, round(sh["t1"] * fps)
            key = (sh.get("set"), sh.get("cam") or sh.get("camera") or sh.get("shot"))
            if a1 - a0 + 1 >= L and (key not in best or a1 - a0 > best[key][1] - best[key][0]):
                best[key] = (a0, a1)
    picks = sorted(best.items(), key=lambda kv: -(kv[1][1] - kv[1][0]))[:MAXN]
    rows = []
    for (st, cam), (a0, a1) in picks:
        s = (a0 + a1 - L) // 2 + 1
        if any(e == ep and abs(u - s) < L for e, u in USED):
            continue
        name = f"{ep.replace('-', '')[:6]}_{st}_{cam}_{s}"
        rows.append(f"{ep} {s}-{s + L - 1} {name}")
    per_ep.append(rows)
# round-robin across episodes: the first night covers the most episodes and sets
for i in range(MAXN):
    for rows in per_ep:
        if i < len(rows):
            print(rows[i])
