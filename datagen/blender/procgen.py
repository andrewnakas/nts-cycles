"""Procedural variety for the realism-pass training data (the realism-pass plan, stage A).

randomize(human, seed) replaces scene-project' hero character (cutscene/human.py LOOK, PHENOTYPE, FACE_SHAPE) with
a random MPFB person before the scene builds it. The choices stay consistent: the skin decides gender, age and
ethnicity, the outfit matches the gender, and face shape is a mild random mix of the hero's face targets. Sets,
cameras, animation and lights come from the episode, so every render is a different person in a real scene.

Enabled from scene-project cutscene/scene.py with env NTS_PROCGEN=<seed> (off by default). All assets are MPFB's
own (CC0); nothing is downloaded.
"""
import os
import random

SUITS = {"male": ["male_casualsuit01", "male_casualsuit02", "male_casualsuit03", "male_casualsuit04", "male_casualsuit05",
                  "male_casualsuit06", "male_elegantsuit01", "male_worksuit01"],
         "female": ["female_casualsuit01", "female_casualsuit02", "female_elegantsuit01", "female_sportsuit01"]}
GLASSES = ["frankyaye_glasses_library_male", "kwnet_at_optical_glasses", "toigo_round_glasses_leopard",
           "spamrakuen_tbm_glasses_frames_01", "spamrakuen_sagerfrogs_glasses_01"]
HAIR = {"male": ["short01", "short02", "short03", "short04", "afro01"],
        "female": ["bob01", "bob02", "braid01", "long01", "ponytail01", "afro01", "short01", "short03"]}
AGE = {"young": (0.35, 0.5), "middleage": (0.55, 0.75), "old": (0.8, 0.95)}


def _assets(data, kind):
    d = os.path.join(data, kind)
    return sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))) if os.path.isdir(d) else []


def _mhclo(kind, name):
    return f"{kind}/{name}/{name}.mhclo"


def randomize(human, seed):
    rng = random.Random(seed)
    data = human.DATA
    # <age>_<ethnicity>_<gender>[2], e.g. young_african_female; not toon01 or the full-body "special_suit" skins
    skins = [s for s in _assets(data, "skins") if s.split("_")[0] in AGE and "special" not in s]
    skin = rng.choice(skins)
    age_word, ethnic, gender = skin.split("_")[0], skin.split("_")[1], skin.split("_")[2].rstrip("0123456789")
    clothes = [_mhclo("clothes", rng.choice(SUITS[gender])), _mhclo("clothes", rng.choice(
        [s for s in ("shoes01", "shoes02", "shoes03", "shoes04", "shoes05", "shoes06")]))]
    if rng.random() < 0.25:
        clothes.append(_mhclo("clothes", rng.choice(GLASSES)))
    human.LOOK = {**human.LOOK,
                  "skin": f"skins/{skin}/{skin}.mhmat",
                  "hair": _mhclo("hair", rng.choice(HAIR[gender])),
                  "eyebrows": _mhclo("eyebrows", rng.choice(_assets(data, "eyebrows"))),
                  "eyelashes": _mhclo("eyelashes", rng.choice(_assets(data, "eyelashes"))),
                  "neckbeard": gender == "male" and rng.random() < 0.15,
                  "clothes": clothes}
    race = {"african": 0.05, "asian": 0.05, "caucasian": 0.05}
    race[ethnic if ethnic in race else "caucasian"] = 0.9
    lo, hi = AGE[age_word]
    human.PHENOTYPE = {"gender": 1.0 if gender == "male" else 0.0, "age": rng.uniform(lo, hi),
                       "muscle": rng.uniform(0.3, 0.85), "weight": rng.uniform(0.3, 0.8),
                       "height": rng.uniform(0.35, 0.75), "proportions": rng.uniform(0.5, 1.0),
                       "cupsize": rng.uniform(0.3, 0.7), "firmness": rng.uniform(0.3, 0.7), "race": race}
    human.FACE_SHAPE = {k: rng.uniform(0.0, 0.6) for k in human.FACE_SHAPE if rng.random() < 0.5}
    summary = {"seed": seed, "skin": skin, "hair": human.LOOK["hair"].split("/")[1],
               "clothes": [c.split("/")[1] for c in clothes], "neckbeard": human.LOOK["neckbeard"],
               "phenotype": {k: round(v, 2) for k, v in human.PHENOTYPE.items() if k != "race"}}
    print("procgen", summary, flush=True)
    return summary
