"""Stage A sample picker: prints "<episode> <frame>" for a seed, a random frame (with frame+1 in the same shot) from a
random scene-project episode that has a timeline. Run: stage_a_pick.py <seed> [episodes_root]
NTS_PICK_LEN=n asks for n consecutive frames in one shot (default 2: the frame pair)."""
import glob
import json
import os
import random
import sys

seed = int(sys.argv[1])
root = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/scene-project")
rng = random.Random(seed * 7919 + 1)
eps = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "timeline.json")))
ep = rng.choice(eps)
tl = json.load(open(os.path.join(ep, "timeline.json")))
fps = tl.get("fps", 30)
shots = [(round(s["t0"] * fps) + 1, round(s["t1"] * fps) + 1) for ln in tl["lines"] for s in ln.get("shots", [])]
n = int(os.environ.get("NTS_PICK_LEN", 2))
shots = [(a, b) for a, b in shots if b - a >= max(4, n + 2)]
a, b = rng.choice(shots)
print(os.path.basename(ep), rng.randint(a + 1, b - n))
