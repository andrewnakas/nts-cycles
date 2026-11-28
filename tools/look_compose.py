"""Look-fit iteration bookkeeping.

  look_compose.py base <fit.json> <out_dir>
      From a fit, write the JSONs datagen/jobs/lg_iter.sh renders with: look_base.json (the fit's skin and hair as real
      materials, lights at 1, exposure 0) and look_v/s/h.json (one skin setting stepped), plus basis.json (the steps).
  look_compose.py absolute <base.json> <fit2.json>
      Add "absolute" settings to a fit made on that base: skin value/sat multiply and hue adds onto the base's, hair
      gains multiply, lights and exposure are already absolute (the base had them at 1 and 0).
"""
import copy
import json
import os
import sys

STEPS = {"v": ["value", 0.75], "s": ["sat", 1.3], "h": ["hue", -0.03]}


def absolute_of(j):
    return j.get("absolute", j["params"])


def base(fit_path, out):
    p = copy.deepcopy(absolute_of(json.load(open(fit_path))))
    p["w"] = dict.fromkeys(p["w"], 1.0)
    p["exposure"] = 0.0
    os.makedirs(out, exist_ok=True)
    json.dump({"params": p, "from": fit_path}, open(os.path.join(out, "look_base.json"), "w"), indent=1)
    for b, (k, step) in STEPS.items():
        q = copy.deepcopy(p)
        q["skin"][k] = q["skin"][k] + step if k == "hue" else q["skin"][k] * step
        json.dump({"params": q}, open(os.path.join(out, f"look_{b}.json"), "w"), indent=1)
    json.dump(STEPS, open(os.path.join(out, "basis.json"), "w"))
    print(f"base {p['skin']} hair {p['hair']} -> {out}")


def absolute(base_path, fit2_path):
    b = json.load(open(base_path))["params"]
    j = json.load(open(fit2_path))
    r = j["params"]
    a = {"w": r["w"], "exposure": r["exposure"],
         "skin": {"value": b["skin"]["value"] * r["skin"]["value"], "sat": b["skin"]["sat"] * r["skin"]["sat"],
                  "hue": b["skin"]["hue"] + r["skin"]["hue"]},
         "hair": [x * y for x, y in zip(b["hair"], r["hair"])]}
    j["absolute"] = a
    json.dump(j, open(fit2_path, "w"), indent=1)
    print(json.dumps(a))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    (base if sys.argv[1] == "base" else absolute)(sys.argv[2], sys.argv[3])
