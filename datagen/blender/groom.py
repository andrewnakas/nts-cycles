"""Groom v2 (2026-11-12): strand hair that reads as hair at our framing, built on strand_hair's root sampling.

What the first strand attempts got wrong, and what changes here:
  colour     sampled per root from MPFB's card texture -> orange clumps and grey blocks. Here melanin and redness are
             fixed numbers from the look file (one hair colour per character), with a small per-clump variation.
  shape      straight 3 cm sticks along the shell tangent -> spikes. Here each strand is bent to stay at its own
             distance from the head centre (it follows the skull like combed hair) and only lifts off by a few mm.
  structure  every strand independent -> a uniform fuzz. Here strands gather into clumps: each one is pulled toward
             a guide strand more and more toward its tip, which gives the locks and gaps real hair shows.
  silhouette nothing left the shell -> the same smooth cap. Here per-clump lift and a few flyaways break the outline.
The shell stays underneath in one flat dark colour as the dense under-layer, so no scalp shows between strands.

Look-file key "groom" (all optional):
  {"n": 60000, "len": 5.0, "mel": 0.55, "red": 0.35, "clump": 0.7, "per_clump": 30, "lift": 2.5, "fly": 0.03,
   "radius": 0.09, "shell": [0.03, 0.022, 0.016], "hero_only": true}
  len cm, lift mm, radius mm (root; the tip is 40% of it).
Called from render_lr.apply_full_look. Positions are rebuilt from the deformed shell on every frame change.
"""
import math

import bpy
import numpy as np

import strand_hair as sh

PTS = 8
_state = []


def _material(name, mel, red, color=None):
    m = sh._material(name)  # Principled Hair + Region AOV hair bit
    h = next(n for n in m.node_tree.nodes if n.type == "BSDF_HAIR_PRINCIPLED")
    for sock in ("Melanin", "Melanin Redness"):
        for ln in list(h.inputs[sock].links):
            m.node_tree.links.remove(ln)
    if color:  # direct colouring: melanin always comes out a warm, saturated brown under this lighting
        h.parametrization = "COLOR"
        h.inputs["Color"].default_value = (*color, 1.0)
    else:
        h.inputs["Melanin"].default_value = mel
        h.inputs["Melanin Redness"].default_value = red
    h.inputs["Roughness"].default_value = 0.35
    h.inputs["Radial Roughness"].default_value = 0.5
    if "Random Color" in h.inputs:
        h.inputs["Random Color"].default_value = 0.2
    return m


def _positions(s, P, UV):
    root, n, t = sh._frames(P, UV, s["tri"], s["bary"])
    t = t * s["sign"]
    b = np.cross(n, t)
    c = P.reshape(-1, 3).mean(0)  # the shell's centre stands in for the head centre
    r0 = np.linalg.norm(root - c, axis=1)
    k = np.linspace(0, 1, PTS, dtype=np.float32)[None, :, None]
    L = s["len"][:, None, None]
    # run along the combing direction with a little sideways wave, then put every point back at the root's distance
    # from the head centre plus a lift that grows toward the tip: the strand hugs the skull instead of poking out
    p = root[:, None] + t[:, None] * L * k + b[:, None] * (s["wave"][:, None, None] * np.sin(math.pi * k))
    d = p - c
    d /= np.linalg.norm(d, axis=2, keepdims=True) + 1e-9
    rad = r0[:, None, None] + 0.0004 + s["lift"][:, None, None] * k ** 1.3
    p = c + d * rad
    g = s["guide"]  # pull toward the clump's guide strand, more toward the tip
    pull = (s["clump"][:, None, None] * k ** 0.8)
    p = p + pull * (p[g] - p)
    return p.reshape(-1, 3).astype(np.float32)


def _update(scene, depsgraph=None):
    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    for s in _state:
        s["curves"].pass_index = s["shell"].pass_index
        P, UV = sh._tris(s["shell"].evaluated_get(dg))
        s["curves"].data.position_data.foreach_set("vector", _positions(s, P, UV).ravel())
        s["curves"].data.update_tag()


def _is_hero(o):
    rig = o.find_armature() or o.parent
    return bool(rig) and not rig.name.upper().startswith("NPC")


def build(scene, cfg):
    from mathutils import kdtree
    from render_lr import object_type
    count = int(cfg.get("n", 60000))
    dg = bpy.context.evaluated_depsgraph_get()
    made = 0
    for ob in [o for o in scene.objects if o.type == "MESH" and object_type(o).lower() == "hair"]:
        if cfg.get("hero_only", True) and not _is_hero(ob):
            continue
        P, UV = sh._tris(ob.evaluated_get(dg))
        area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
        rng = np.random.default_rng(0)
        tri = rng.choice(len(P), count, p=area / area.sum())
        r1, r2 = rng.random(count), rng.random(count)
        sq = np.sqrt(r1)
        bary = np.stack([1 - sq, sq * (1 - r2), sq * r2], 1).astype(np.float32)
        root, _, t = sh._frames(P, UV, tri, bary)
        # MPFB shells are several alpha cards: the haircut is where the texture is opaque, not where the mesh ends
        tex = sh._texture(ob)

        def alpha_at(tr, bc):
            if tex is None:
                return np.ones(len(tr), np.float32)
            uv = (UV[tr] * bc[..., None]).sum(1) % 1.0
            hh, ww = tex.shape[:2]
            return tex[(uv[:, 1] * (hh - 1)).astype(int), (uv[:, 0] * (ww - 1)).astype(int), 3]
        keep = alpha_at(tri, bary) > 0.6
        tri, bary, root = tri[keep], bary[keep], root[keep]
        n = len(tri)
        sign = np.float32(1.0 if (t[:, 2]).mean() < 0 else -1.0)  # grow the way that points down on average
        fly = rng.random(n) < cfg.get("fly", 0.03)
        length = cfg.get("len", 5.0) / 100 * rng.uniform(0.6, 1.2, n)
        # most strands stop at the shell's open edge (the haircut's outline); flyaways and one in five pass it a little
        length = length.astype(np.float32)
        # clumps: guides are a random subset; every strand follows the nearest guide root
        ng = max(1, n // int(cfg.get("per_clump", 30)))
        gi = rng.choice(n, ng, replace=False)
        kd = kdtree.KDTree(ng)
        for i, j in enumerate(gi):
            kd.insert(root[j], i)
        kd.balance()
        guide = np.array([gi[kd.find(r)[1]] for r in root], np.int64)
        guide[gi] = gi
        clump = (cfg.get("clump", 0.7) * rng.uniform(0.6, 1.0, n)).astype(np.float32)
        clump[fly] = 0.0
        clump[gi] = 0.0
        lift_mm = cfg.get("lift", 2.5)
        glift = rng.uniform(0.2, 1.0, n) ** 1.5 * lift_mm / 1000  # per clump, so whole locks stand off the head
        lift = (glift[guide] + rng.uniform(0, 0.0006, n)).astype(np.float32)
        lift[fly] = rng.uniform(1.5, 3.5, int(fly.sum())) * lift_mm / 1000
        s = {"shell": ob, "tri": tri, "bary": bary, "len": length, "sign": sign, "guide": guide, "clump": clump,
             "lift": lift, "wave": rng.normal(0, 0.0012, n).astype(np.float32)}
        # cut every strand where it leaves the haircut: walk its points and stop at the first one that is not over
        # an opaque part of the shell (a few strands may run one step past, for a slightly uneven edge)
        from mathutils.bvhtree import BVHTree
        bvh = BVHTree.FromPolygons([tuple(v) for v in P.reshape(-1, 3)], [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(P))])
        free = dict(s, clump=np.zeros(n, np.float32), lift=np.zeros(n, np.float32))
        pts = _positions(free, P, UV).reshape(n, PTS, 3)
        last = np.full(n, PTS - 1, np.int32)
        alive = np.ones(n, bool)
        for k in range(1, PTS):
            hit_t, hit_b, ok = np.zeros(n, np.int64), np.zeros((n, 3), np.float32), np.zeros(n, bool)
            for i in np.nonzero(alive)[0]:
                loc, _, idx, dist = bvh.find_nearest(pts[i, k].tolist(), 0.006)
                if loc is None:
                    continue
                a, b, c = P[idx]
                v0, v1, v2 = b - a, c - a, np.array(loc, np.float32) - a
                d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
                den = d00 * d11 - d01 * d01 + 1e-20
                bv, bw = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
                hit_t[i], hit_b[i], ok[i] = idx, (1 - bv - bw, bv, bw), True
            ok &= alpha_at(hit_t, hit_b) > 0.35
            ended = alive & ~ok
            last[ended] = k - 1
            alive &= ok
        extra = (fly | (rng.random(n) < 0.15)).astype(np.float32) * 0.6
        s["len"] = (length * np.clip((last + extra) / (PTS - 1), 0.08, 1.0)).astype(np.float32)
        cd = bpy.data.hair_curves.new(f"{ob.name}_groom")
        cd.add_curves([PTS] * n)
        r = cfg.get("radius", 0.09) / 1000
        ra = cd.attributes.get("radius") or cd.attributes.new("radius", "FLOAT", "POINT")
        ra.data.foreach_set("value", np.tile(np.linspace(r, 0.4 * r, PTS, dtype=np.float32), n))
        gmel = rng.normal(0, 0.05, n)
        mel = np.clip(cfg.get("mel", 0.55) + gmel[guide] + rng.normal(0, 0.02, n), 0.05, 0.98).astype(np.float32)
        cd.attributes.new("groom_mel", "FLOAT", "CURVE").data.foreach_set("value", mel)
        cd.materials.append(_material(f"{ob.name}_groom_mat", cfg.get("mel", 0.55), cfg.get("red", 0.35), cfg.get("color")))
        co = bpy.data.objects.new(f"{ob.name}_groom", cd)
        scene.collection.objects.link(co)
        s["curves"] = co
        _state.append(s)
        sh._flat_shell(ob, cfg.get("shell", (0.03, 0.022, 0.016)))
        for sl in ob.material_slots:  # matte under-layer: a glossy flat shell reads as a plastic patch where it shows
            b = next((x for x in sl.material.node_tree.nodes if x.type == "BSDF_PRINCIPLED"), None) if sl.material else None
            if b:
                b.inputs["Roughness"].default_value = 0.95
                b.inputs["Specular IOR Level"].default_value = 0.1
        made += n
    if made:
        scene.cycles_curves.shape = "THICK"
        bpy.app.handlers.frame_change_post.append(_update)
        _update(scene, dg)
    print(f"groom: {made} strands on {len(_state)} shells", flush=True)
    return made
