"""Realism pack (2026-11-12): surface detail a real camera sees at our framing (face about 90 px tall in 1080x1920)
that the MakeHuman assets lack. Every change is authored on the surface (UV or rest-pose coordinates), so it is the same
person by construction and cannot flicker.

What the native/reference still pairs show missing (not copied from them; they are eval-only):
  skin   colour is too even and too orange: real skin at this size shows mottle (redder and browner patches a few px
         across), pigment specks/freckles of 1-2 px and roughness that varies (oilier T-zone, matte cheeks).
         human.skin_realism has blotch/speck noise, but its speck (UV 700) is sub-pixel at our framing and averages out.
  cloth  flat dark plastic: real knit shows thread-level brightness variation, visible rows, fuzz at grazing angles
         and soft creases. human.cloth_realism has a bump-only weave whose 900 UV rows are also sub-pixel.
  eyes   iris too saturated and bright, sclera too white.

Look-file key "realism" (looks/v8.json), all optional:
  {"skin": {"mottle": 0.25, "mottle_scale": 120, "freck": 0.35, "freck_scale": 260, "freck_dens": 0.3,
            "rough_var": 0.12},
   "cloth": {"knit": 0.18, "knit_scale": 260, "fibre": 0.10, "bump": 0.5, "crease": 0.6},
   "eyes": {"iris_sat": 0.7, "sclera": 0.85}}
Called from render_lr.apply_full_look.
"""
import bpy

SKIN_GROUP = "human.body"


def _io(node, ident, out=False):
    return next(s for s in (node.outputs if out else node.inputs) if s.identifier == ident)


def _mix_rgb(nt, a, b, blend="MULTIPLY", fac=1.0):
    m = nt.nodes.new("ShaderNodeMix")
    m.data_type, m.blend_type = "RGBA", blend
    m.inputs["Factor"].default_value = fac
    nt.links.new(a, _io(m, "A_Color"))
    if isinstance(b, tuple):
        _io(m, "B_Color").default_value = b
    else:
        nt.links.new(b, _io(m, "B_Color"))
    return _io(m, "Result_Color", True)


def _ramp(nt, fac, stops):
    r = nt.nodes.new("ShaderNodeValToRGB")
    els = r.color_ramp.elements
    while len(els) < len(stops):
        els.new(0.5)
    for e, (pos, col) in zip(els, stops):
        e.position, e.color = pos, (*col, 1.0)
    nt.links.new(fac, r.inputs["Fac"])
    return r.outputs["Color"]


def _uv(nt):
    return nt.nodes.new("ShaderNodeTexCoord").outputs["UV"]


def _noise(nt, vec, scale, detail=2.0, rough=0.5):
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value, n.inputs["Detail"].default_value = scale, detail
    n.inputs["Roughness"].default_value = rough
    nt.links.new(vec, n.inputs["Vector"])
    return n.outputs["Fac"]


def _voronoi_f1(nt, vec, scale, rand=1.0):
    v = nt.nodes.new("ShaderNodeTexVoronoi")
    v.feature = "F1"
    v.inputs["Scale"].default_value, v.inputs["Randomness"].default_value = scale, rand
    nt.links.new(vec, v.inputs["Vector"])
    return v


REGION_ATTR = "nts_skin"


def _lerp(nt, a, b, fac):
    """a -> b by a factor socket (or float)."""
    m = nt.nodes.new("ShaderNodeMix")
    m.data_type, m.blend_type, m.clamp_factor = "RGBA", "MIX", True
    if isinstance(fac, float):
        m.inputs["Factor"].default_value = fac
    else:
        nt.links.new(fac, m.inputs["Factor"])
    nt.links.new(a, _io(m, "A_Color"))
    nt.links.new(b, _io(m, "B_Color"))
    return _io(m, "Result_Color", True)


def _math(nt, op, a, b=None, clamp=False):
    m = nt.nodes.new("ShaderNodeMath")
    m.operation, m.use_clamp = op, clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            m.inputs[i].default_value = v
        else:
            nt.links.new(v, m.inputs[i])
    return m.outputs[0]


def _is_female(o):
    for sl in o.material_slots:
        m = sl.material
        if not (m and m.use_nodes):
            continue
        trees = [m.node_tree] + [n.node_tree for n in m.node_tree.nodes if n.type == "GROUP" and n.node_tree]
        for n in (x for t in trees for x in t.nodes):
            if n.type == "TEX_IMAGE" and n.image and "female" in n.image.name.lower():
                return True
    return False


def paint_skin_regions(scene):
    """Per-vertex face regions on every MakeHuman body (found by its joint helper vertex groups), from rest-pose
    landmarks, as the point colour attribute nts_skin: R = redness (nose, cheeks, eyelids), G = under-eye,
    B = beard-shadow zone (jaw, chin, upper lip; not the lips), A = oilier zone (forehead, nose ridge).
    Coordinates are in interocular distances from the point between the eyes, so it fits any MPFB character."""
    import numpy as np
    done = []
    for o in scene.objects:
        if o.type != "MESH" or "joint-l-eye" not in o.vertex_groups or REGION_ATTR in o.data.attributes:
            continue
        me = o.data
        n = len(me.vertices)
        co = np.empty(n * 3, np.float32)
        (me.shape_keys.key_blocks[0].data if me.shape_keys else me.vertices).foreach_get("co", co)
        co = co.reshape(n, 3)
        if me.shape_keys:  # MPFB keeps the character's shape in shape keys: add the active ones to the basis
            kb = me.shape_keys.key_blocks
            base = co.copy()
            for k in kb[1:]:
                if abs(k.value) > 1e-4:
                    d = np.empty(n * 3, np.float32)
                    k.data.foreach_get("co", d)
                    co += k.value * (d.reshape(n, 3) - base)
        want = {o.vertex_groups[g].index: g for g in ("joint-l-eye", "joint-r-eye", "joint-mouth", "joint-head", "lips",
                                                      "body", "ears") if g in o.vertex_groups}
        member = {g: np.zeros(n, bool) for g in want.values()}
        for v in me.vertices:
            for ge in v.groups:
                g = want.get(ge.group)
                if g and ge.weight > 0.5:
                    member[g][v.index] = True
        if not all(member[g].any() for g in ("joint-l-eye", "joint-r-eye", "joint-mouth", "joint-head")):
            continue
        cen = {g: co[member[g]].mean(0) for g in ("joint-l-eye", "joint-r-eye", "joint-mouth", "joint-head")}
        el, er = cen["joint-l-eye"], cen["joint-r-eye"]
        mid, d = (el + er) / 2, float(np.linalg.norm(el - er))
        fy = 1.0 if mid[1] > cen["joint-head"][1] else -1.0  # which way the face points along Y
        x, y, z = (co[:, 0] - mid[0]) / d, fy * (co[:, 1] - mid[1]) / d, (co[:, 2] - mid[2]) / d
        skin_v = member.get("body", np.ones(n, bool))
        centre = skin_v & (np.abs(x) < 0.2) & (z > -1.1) & (z < -0.1)  # the nose tip is the most forward point there
        if not centre.any():
            continue
        tip = np.argmax(np.where(centre, y, -1e9))
        yn, zn = float(y[tip]), float(z[tip])
        zm, zchin = zn - 0.38, zn - 1.0  # mouth line and chin, from average face proportions below the nose tip

        def ss(v, a, b):  # smoothstep from a (0) to b (1); a > b gives a falling edge
            t = np.clip((v - a) / (b - a), 0, 1)
            return t * t * (3 - 2 * t)
        front = ss(y, yn - 1.6, yn - 1.0)  # fades out toward the ears and the back of the head
        g = lambda dx, sx, dz, sz: np.exp(-0.5 * ((dx / sx) ** 2 + (dz / sz) ** 2))  # noqa: E731
        nose = g(x, 0.26, z - zn, 0.26) * ss(y, yn - 0.75, yn - 0.3)
        cheek = g(np.abs(x) - 0.72, 0.30, z - (zn + 0.05), 0.30) * front
        eye_r = 0.26  # eyeball radius in interocular distances (MakeHuman eyes)
        de = np.minimum(np.linalg.norm(co - el, axis=1), np.linalg.norm(co - er, axis=1)) / d
        lid = ss(de, eye_r * 1.3, eye_r * 1.05)
        red = np.clip(0.9 * nose + 0.7 * cheek + 0.45 * lid, 0, 1)
        under = g(np.abs(x) - 0.5, 0.24, z + 0.36, 0.11) * front
        beard = ss(z, zn - 0.02, zn - 0.2) * ss(z, zchin - 1.3, zchin - 0.6) * ss(y, yn - 1.75, yn - 1.2)
        beard *= 1 - 0.55 * cheek  # the hollow of the cheek has little beard
        beard *= 1 - ss(np.abs(x), 0.25, 0.6) * ss(z, zm + 0.1, zm + 0.45)  # nothing beside the nose wings
        if "lips" in member:
            beard[member["lips"]] = 0
        if _is_female(o):  # MPFB skins are named <age>_<ethnicity>_<gender>: no beard shadow on women
            beard[:] = 0
        oil = np.clip(g(x, 0.6, z - 0.8, 0.38) * front + 0.8 * g(x, 0.12, z - (zn + 0.3), 0.35) * front, 0, 1)
        col = np.stack([red, under, beard, oil], 1).astype(np.float32)
        col[~skin_v] = 0
        a = me.attributes.new(REGION_ATTR, "FLOAT_COLOR", "POINT")
        a.data.foreach_set("color", col.ravel())
        done.append(f"{o.name}: d={d * 100:.1f}cm nose=({yn:.2f},{zn:.2f}) beard_verts={int((beard > 0.5).sum())}")
    return done


def ears(cfg):
    """MPFB gives the ears their own material slot around the shared skin group: more subsurface and a redder
    mix-in there, so the rim light shows through the way it does on a real ear."""
    n = 0
    for m in bpy.data.materials:
        if not (m.use_nodes and m.name.lower().split(".")[1:2] == [cfg.get("part", "ears")]):
            continue
        for nd in m.node_tree.nodes:
            if nd.type != "GROUP":
                continue
            for name, key, mode in (("SSS strength", "sss", "set"), ("SSS radius scale", "radius", "mul"),
                                    ("colorMixInStrength", "red", "set")):
                if name in nd.inputs and key in cfg:
                    nd.inputs[name].default_value = cfg[key] if mode == "set" else nd.inputs[name].default_value * cfg[key]
            n += 1
    return n


def skin(cfg):
    """Mottle, freckles and roughness variation inside MPFB's shared skin group(s), multiplied over the base colour
    after everything human.skin_realism did (so the mouth-interior mix and the look's HSV still apply on top)."""
    done = []
    for nt in bpy.data.node_groups:
        if not nt.name.lower().startswith(SKIN_GROUP):
            continue
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None or not bsdf.inputs["Base Color"].is_linked:
            continue
        uv = _uv(nt)
        col = bsdf.inputs["Base Color"].links[0].from_socket
        reg, local = cfg.get("regions"), None
        if reg:  # paint_skin_regions: R redness, G under-eye, B beard zone, A oil
            at = nt.nodes.new("ShaderNodeAttribute")
            at.attribute_type, at.attribute_name = "GEOMETRY", REGION_ATTR
            sep = nt.nodes.new("ShaderNodeSeparateColor")
            nt.links.new(at.outputs["Color"], sep.inputs["Color"])
            R, G, B, A = sep.outputs[0], sep.outputs[1], sep.outputs[2], at.outputs["Alpha"]
            for mask, key, tint in ((R, "red", (1.0, 0.78, 0.78)), (G, "under", (0.72, 0.66, 0.74)),
                                    (B, "beard", (0.60, 0.61, 0.66))):
                k = reg.get(key, 0)
                if k > 0:
                    t = tuple(reg.get(key + "_tint", tint))
                    f = _math(nt, "MULTIPLY", mask, float(k), clamp=True)
                    if key == "beard" and reg.get("stubble", 0) > 0:  # follicle speckle: the shadow is dots, not a wash
                        dots = _noise(nt, uv, reg.get("stubble_scale", 1400.0), detail=1.0, rough=0.4)
                        mr = nt.nodes.new("ShaderNodeMapRange")
                        nt.links.new(dots, mr.inputs["Value"])
                        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.35, 0.65
                        mr.inputs["To Min"].default_value = 1 - reg["stubble"]
                        mr.inputs["To Max"].default_value = 1 + reg["stubble"]
                        f = _math(nt, "MULTIPLY", f, mr.outputs["Result"], clamp=True)
                    col = _lerp(nt, col, _mix_rgb(nt, col, (*t, 1.0)), f)
            if reg.get("detail_local", 0) > 0:  # mottle and freckles mostly where real faces have them
                local = _math(nt, "MULTIPLY_ADD", R, float(reg["detail_local"]), clamp=True)
                local.node.inputs[2].default_value = reg.get("detail_base", 0.3)
        if cfg.get("mottle", 0) > 0:  # redder <-> browner patches, two octaves
            k = cfg["mottle"]
            f = _noise(nt, uv, cfg.get("mottle_scale", 120.0), detail=3.0, rough=0.6)
            m = _ramp(nt, f, [(0.3, (1 - 0.10 * k, 1 - 0.55 * k, 1 - 0.60 * k)), (0.5, (1, 1, 1)),
                              (0.7, (1 - 0.35 * k, 1 - 0.45 * k, 1 - 0.55 * k))])
            col = _lerp(nt, col, _mix_rgb(nt, col, m), local) if local is not None else _mix_rgb(nt, col, m)
        if cfg.get("freck", 0) > 0:  # small brown pigment spots where a random cell's centre is close
            k, dens = cfg["freck"], cfg.get("freck_dens", 0.3)
            v = _voronoi_f1(nt, uv, cfg.get("freck_scale", 260.0))
            gate = _noise(nt, uv, cfg.get("freck_scale", 260.0) / 6, detail=1.0)  # clustered, not uniform
            d = nt.nodes.new("ShaderNodeMath")
            d.operation = "SUBTRACT"
            nt.links.new(_io(v, "Distance", True), d.inputs[0])
            nt.links.new(gate, d.inputs[1])  # nearer where the gate noise is high
            spot = _ramp(nt, d.outputs[0], [(-0.6 + dens, (1 - 0.35 * k, 1 - 0.50 * k, 1 - 0.60 * k)),
                                            (-0.45 + dens, (1, 1, 1))])
            col = _lerp(nt, col, _mix_rgb(nt, col, spot), local) if local is not None else _mix_rgb(nt, col, spot)
        if cfg.get("cavity", 0) > 0:  # skin darkens in creases and sockets (occlusion of its own scattered light)
            ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
            ao.only_local, ao.samples = True, 8
            ao.inputs["Distance"].default_value = cfg.get("cavity_dist", 0.03)
            mr = nt.nodes.new("ShaderNodeMapRange")
            nt.links.new(ao.outputs["AO"], mr.inputs["Value"])
            mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 1 - cfg["cavity"], 1.0
            cav = nt.nodes.new("ShaderNodeCombineColor")
            for i in range(3):
                nt.links.new(mr.outputs["Result"], cav.inputs[i])
            col = _mix_rgb(nt, col, cav.outputs["Color"])
        nt.links.new(col, bsdf.inputs["Base Color"])
        if cfg.get("rough_var", 0) > 0 and bsdf.inputs["Roughness"].is_linked:
            r = bsdf.inputs["Roughness"].links[0].from_socket
            f = _noise(nt, uv, 40.0, detail=4.0)
            off = nt.nodes.new("ShaderNodeMath")
            off.operation = "MULTIPLY_ADD"  # (f - 0.5) * 2k + r
            sub = nt.nodes.new("ShaderNodeMath")
            sub.operation = "SUBTRACT"
            nt.links.new(f, sub.inputs[0])
            sub.inputs[1].default_value = 0.5
            nt.links.new(sub.outputs[0], off.inputs[0])
            off.inputs[1].default_value = 2 * cfg["rough_var"]
            nt.links.new(r, off.inputs[2])
            off.use_clamp = True
            nt.links.new(off.outputs[0], bsdf.inputs["Roughness"])
        if reg and (reg.get("oil", 0) > 0 or reg.get("beard_rough", 0) > 0) and bsdf.inputs["Roughness"].is_linked:
            r = bsdf.inputs["Roughness"].links[0].from_socket  # shinier forehead and nose, slightly matte beard zone
            r = _math(nt, "SUBTRACT", r, _math(nt, "MULTIPLY", A, float(reg.get("oil", 0))))
            r = _math(nt, "ADD", r, _math(nt, "MULTIPLY", B, float(reg.get("beard_rough", 0))), clamp=True)
            nt.links.new(r, bsdf.inputs["Roughness"])
        done.append(nt.name)
    return done


def _is_cloth(o):
    n = o.name.lower()
    return o.type == "MESH" and any(k in n for k in ("suit", "shirt", "pants", "jeans", "hoodie", "tshirt", "dress",
                                                    "sweater", "jacket", "skirt", "trouser"))


def cloth(scene, cfg):
    """Thread-level brightness variation (knit rows plus fibre noise) in the colour, a stronger weave bump and soft
    creases, on every garment in the shot (hero and background people). Printed graphics get the same texture, as
    a real print on knit does."""
    done = set()
    for o in scene.objects:
        if not _is_cloth(o):
            continue
        for sl in o.material_slots:
            m = sl.material
            if not (m and m.use_nodes) or m.name in done or m.get("nts_rp_cloth"):
                continue
            nt = m.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue
            m["nts_rp_cloth"] = True
            uv = _uv(nt)
            ks = cfg.get("knit_scale", 260.0)
            wave = nt.nodes.new("ShaderNodeTexWave")  # knit rows
            wave.inputs["Scale"].default_value, wave.inputs["Distortion"].default_value = ks, 2.0
            wave.inputs["Detail"].default_value = 1.0
            nt.links.new(uv, wave.inputs["Vector"])
            fib = _noise(nt, uv, ks * 1.7, detail=3.0, rough=0.7)
            g = nt.nodes.new("ShaderNodeMath")
            g.operation = "MULTIPLY_ADD"
            nt.links.new(wave.outputs["Fac"], g.inputs[0])
            g.inputs[1].default_value = cfg.get("knit", 0.18)
            nt.links.new(fib, g.inputs[2])  # fibre noise (0..1, mean .5) + rows
            lo = 1 - cfg.get("knit", 0.18) - cfg.get("fibre", 0.10)
            mr = nt.nodes.new("ShaderNodeMapRange")
            nt.links.new(g.outputs[0], mr.inputs["Value"])
            mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.2, 0.5 + cfg.get("knit", 0.18)
            mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = lo, 1.0
            base = bsdf.inputs["Base Color"]
            src = base.links[0].from_socket if base.is_linked else None
            if src is None:
                rgb = nt.nodes.new("ShaderNodeRGB")
                rgb.outputs[0].default_value = base.default_value
                src = rgb.outputs[0]
            nt.links.new(_mix_rgb(nt, src, mr.outputs["Result"]), base)
            # bump: rows + fibres at thread scale, creases a few cm across
            crease = _noise(nt, uv, 14.0, detail=5.0, rough=0.55)
            bump = nt.nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value, bump.inputs["Distance"].default_value = cfg.get("bump", 0.5), 0.001
            nt.links.new(g.outputs[0], bump.inputs["Height"])
            bump2 = nt.nodes.new("ShaderNodeBump")
            bump2.inputs["Strength"].default_value, bump2.inputs["Distance"].default_value = cfg.get("crease", 0.6), 0.01
            nt.links.new(crease, bump2.inputs["Height"])
            if bsdf.inputs["Normal"].is_linked:
                nt.links.new(bsdf.inputs["Normal"].links[0].from_socket, bump2.inputs["Normal"])
            nt.links.new(bump2.outputs["Normal"], bump.inputs["Normal"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
            mt = float(cfg.get("matte", 0.0))
            if mt > 0:
                # Cotton and wool have no gloss: on the default surface the lights leave a soft sheen on the shoulders
                # and sleeves that reads as neoprene ("wetsuit-like clothes", blind judges 2026-11-22). matte moves
                # the roughness that share of the way to 1 and takes that share of 0.8 off the specular level.
                r = bsdf.inputs["Roughness"]
                if r.is_linked:
                    ma = nt.nodes.new("ShaderNodeMath")
                    ma.operation = "MULTIPLY_ADD"
                    nt.links.new(r.links[0].from_socket, ma.inputs[0])
                    ma.inputs[1].default_value, ma.inputs[2].default_value = 1 - mt, mt
                    nt.links.new(ma.outputs[0], r)
                else:
                    r.default_value = r.default_value + mt * (1 - r.default_value)
                sp = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
                if sp is not None and not sp.is_linked:
                    sp.default_value = sp.default_value * (1 - 0.8 * mt)
                for k in ("Coat Weight", "Sheen Weight"):
                    if bsdf.inputs.get(k) is not None and not bsdf.inputs[k].is_linked:
                        bsdf.inputs[k].default_value = bsdf.inputs[k].default_value * (1 - mt)
            done.add(m.name)
    return sorted(done)


def eyes(scene, cfg):
    """Iris saturation and sclera brightness on MPFB procedural eyes: a multiply on every colour input of the eye
    materials' node groups is too blunt, so act on the group inputs MPFB exposes (IrisMinorColor etc.) when present."""
    n = 0
    mats = set()
    for o in scene.objects:  # MPFB eye materials are named after the asset ("low-poly"), so find them via the object
        nm = o.name.lower()
        if o.type == "MESH" and ("eye" in nm or "-poly" in nm) and not any(k in nm for k in ("brow", "lash")):
            mats |= {s.material for s in o.material_slots if s.material and s.material.use_nodes}
    if not mats:
        print("realism_pack.eyes: no eye objects among", sorted(o.name for o in scene.objects if o.name.startswith("Human")))
    for m in mats:
        for nd in m.node_tree.nodes:  # MPFB's EnhancedEye group: explicit colours beat scaling the stock blue
            if nd.type != "GROUP":
                continue
            for key, name in (("iris_major", "IrisMajorColor"), ("iris_minor", "IrisMinorColor"),
                              ("iris_edge", "IrisSection4Color"), ("sclera_rgb", "EyeWhiteColor")):
                if key in cfg and name in nd.inputs:
                    nd.inputs[name].default_value = (*cfg[key], 1.0)
                    n += 1
            for key, name in (("pupil", "PupilSize"), ("coat", "Clearcoat")):
                if key in cfg and name in nd.inputs:
                    nd.inputs[name].default_value = cfg[key]
        if any(k in cfg for k in ("iris_major", "sclera_rgb")):
            continue
        trees = [m.node_tree] + [nd.node_tree for nd in m.node_tree.nodes if nd.type == "GROUP" and nd.node_tree]
        for nd in (x for t in trees for x in t.nodes):
            if nd.type == "RGB":  # MPFB stores eye colours as named RGB nodes (IrisMajorColor, EyeWhiteColor, ...)
                nm, o = (nd.label or nd.name).lower(), nd.outputs[0]
                c = list(o.default_value)
                if "iris" in nm:
                    mean, s = sum(c[:3]) / 3, cfg.get("iris_sat", 0.7)
                    o.default_value = (*(mean + s * (v - mean) for v in c[:3]), c[3])
                    n += 1
                elif "white" in nm or "sclera" in nm:
                    o.default_value = (*(v * cfg.get("sclera", 0.85) for v in c[:3]), c[3])
                    n += 1
                continue
            for inp in getattr(nd, "inputs", ()):
                name = inp.name.lower()
                if inp.type != "RGBA" or inp.is_linked:
                    continue
                c = list(inp.default_value)
                if "iris" in name:
                    mean = sum(c[:3]) / 3
                    s = cfg.get("iris_sat", 0.7)
                    inp.default_value = (*(mean + s * (v - mean) for v in c[:3]), c[3])
                    n += 1
                elif "sclera" in name or "white" in name:
                    k = cfg.get("sclera", 0.85)
                    inp.default_value = (*(v * k for v in c[:3]), c[3])
                    n += 1
    return n


def hair(scene, cfg):
    """Strand texture on the MPFB hair shell, keeping its silhouette (curve strands, v6/v7, changed the silhouette into
    clumps and spikes). Along the shell's UV V axis (the strand direction): fine strand-to-strand brightness variation,
    darker gaps between clumps a few strands wide, a matching bump, and an ash tint multiplied over the texture."""
    import render_lr
    n = 0
    for o in scene.objects:
        if o.type != "MESH" or "hair" not in (render_lr.object_type(o).lower() or o.name.lower()):
            continue
        for sl in o.material_slots:
            m = sl.material
            if not (m and m.use_nodes) or m.get("nts_rp_hair"):
                continue
            nt = m.node_tree
            bsdf = next((x for x in nt.nodes if x.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue
            m["nts_rp_hair"] = True
            # the shell's UVs barely vary across the visible hair (noise on them came out flat); rest-pose Generated
            # coordinates (0-1 over the shell's bounding box) are stable under deformation and do vary
            tc = nt.nodes.new("ShaderNodeTexCoord")
            uv = tc.outputs["Generated"] if cfg.get("coords", "generated") == "generated" else tc.outputs["UV"]

            def stretched(su, sv):
                mp = nt.nodes.new("ShaderNodeMapping")
                mp.inputs["Scale"].default_value = (su, sv, sv)  # Generated: streaks run along Y (front-back) and Z
                nt.links.new(uv, mp.inputs["Vector"])
                return mp.outputs["Vector"]
            s = cfg.get("strand_scale", 700.0)
            fine = _noise(nt, stretched(s, s / 10), 1.0, detail=3.0, rough=0.6)
            cs = cfg.get("clump_scale", s / 8)
            clump = _noise(nt, stretched(cs, cs / 2), 1.0, detail=2.0, rough=0.5)
            c = cfg.get("contrast", 0.45)
            fr = nt.nodes.new("ShaderNodeMapRange")
            nt.links.new(fine, fr.inputs["Value"])
            fr.inputs["From Min"].default_value, fr.inputs["From Max"].default_value = 0.3, 0.7
            fr.inputs["To Min"].default_value, fr.inputs["To Max"].default_value = 1 - c, 1.0
            cr = nt.nodes.new("ShaderNodeMapRange")
            nt.links.new(clump, cr.inputs["Value"])
            cr.inputs["From Min"].default_value, cr.inputs["From Max"].default_value = 0.35, 0.6
            cr.inputs["To Min"].default_value, cr.inputs["To Max"].default_value = 1 - cfg.get("clump", 0.35), 1.0
            base = bsdf.inputs["Base Color"]
            src = base.links[0].from_socket if base.is_linked else None
            if src is None:
                rgb = nt.nodes.new("ShaderNodeRGB")
                rgb.outputs[0].default_value = base.default_value
                src = rgb.outputs[0]
            col = _mix_rgb(nt, _mix_rgb(nt, src, fr.outputs["Result"]), cr.outputs["Result"])
            if cfg.get("tint"):
                col = _mix_rgb(nt, col, (*cfg["tint"], 1.0))
            if cfg.get("tips", 0) > 0:  # sun-lightened ends: brighter toward the top/outside of the shell and per clump
                sp = nt.nodes.new("ShaderNodeSeparateXYZ")
                nt.links.new(tc.outputs["Generated"], sp.inputs[0])
                tipf = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", sp.outputs[2], clump), float(cfg["tips"]), clamp=True)
                col = _lerp(nt, col, _mix_rgb(nt, col, (1.5, 1.4, 1.3, 1.0)), tipf)
            nt.links.new(col, base)
            alpha = bsdf.inputs["Alpha"]
            if cfg.get("edge", 0) > 0 and alpha.is_linked:  # break up the soft edge of the alpha card into strands
                e = float(cfg["edge"])
                thin = _noise(nt, stretched(s * 1.5, s / 12), 1.0, detail=2.0, rough=0.5)
                cut = _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", 1.0, thin), 2 * e)   # 0..2e, more where no strand
                a2 = _math(nt, "DIVIDE", _math(nt, "SUBTRACT", alpha.links[0].from_socket, cut), 1 - e, clamp=True)
                nt.links.new(a2, alpha)
            h = nt.nodes.new("ShaderNodeMath")
            h.operation = "MULTIPLY"
            nt.links.new(fine, h.inputs[0])
            nt.links.new(clump, h.inputs[1])
            bump = nt.nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value, bump.inputs["Distance"].default_value = cfg.get("bump", 0.6), 0.0008
            nt.links.new(h.outputs[0], bump.inputs["Height"])
            if bsdf.inputs["Normal"].is_linked:
                nt.links.new(bsdf.inputs["Normal"].links[0].from_socket, bump.inputs["Normal"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
            # the native shell reads as a smooth ash cap mostly through sheen-like specular on a dark albedo: make it
            # matte and let the strand noise break up what specular is left
            if "Specular IOR Level" in bsdf.inputs and not bsdf.inputs["Specular IOR Level"].is_linked:
                bsdf.inputs["Specular IOR Level"].default_value = cfg.get("spec", 0.3)
            if not bsdf.inputs["Roughness"].is_linked:
                r0 = cfg.get("roughness", 0.7)
                rr = nt.nodes.new("ShaderNodeMapRange")
                nt.links.new(fine, rr.inputs["Value"])
                rr.inputs["To Min"].default_value, rr.inputs["To Max"].default_value = r0 + 0.15, r0 - 0.15
                rr.clamp = True
                nt.links.new(rr.outputs["Result"], bsdf.inputs["Roughness"])
            if "Anisotropic" in bsdf.inputs and not bsdf.inputs["Anisotropic"].is_linked:
                bsdf.inputs["Anisotropic"].default_value = 0.6
                tan = nt.nodes.new("ShaderNodeTangent")
                tan.direction_type = "UV_MAP"
                nt.links.new(tan.outputs["Tangent"], bsdf.inputs["Tangent"])
            n += 1
    return n


def apply(scene, cfg):
    import os
    if os.environ.get("NTS_RP_DUMP"):
        import inspect_head
        inspect_head.dump(scene, os.environ["NTS_RP_DUMP"])
        raise SystemExit(0)
    done = {}
    if cfg.get("skin"):
        if cfg["skin"].get("regions"):
            done["regions"] = paint_skin_regions(scene)
        done["skin"] = skin(cfg["skin"])
    for part in ("ears", "lips"):  # MPFB material slots around the shared skin group
        if cfg.get(part):
            done[part] = ears({**cfg[part], "part": part})
    if cfg.get("cloth"):
        done["cloth"] = cloth(scene, cfg["cloth"])
    if cfg.get("eyes"):
        done["eyes"] = eyes(scene, cfg["eyes"])
    if cfg.get("hair"):
        done["hair"] = hair(scene, cfg["hair"])
    print("realism_pack:", done)
    return done
