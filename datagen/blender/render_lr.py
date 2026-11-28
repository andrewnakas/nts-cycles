"""Blender-side render setup for NTS: the cycles-harness harness_setup layout plus the passes the network and the judging need.

Call setup_nts(scene, out_dir, mode, seed=0, halton_start=0) after the scene's normal render setup (scene-project
cutscene/scene.py does it for --nts <mode>). Writes, like harness_setup:
  out_dir/f_####.exr  multilayer half: Image (scene-linear, after glare + haze), Depth.V, Vector.XYZW,
                      plus Albedo.RGB, Normal.XYZ (Cycles denoising-data passes) and Region.RGB (see below)
  jitter.json, cuts.json, meta.json
Region.RGB is an AOV mask of the hero: R = face (head/jaw vertex groups + eyes, brows, lashes, teeth),
G = hands (hand and finger groups), B = hair. It drives the identity and skin-tone metrics and the enhancer's gains.
mode "native" renders 1080x1920 unjittered with the 1.5 px filter: ground truth with the same masks.
mode "hero" is native plus expensive settings (render_hero.py): the target the realism pass learns toward.
Quality modes for NTS-F, the usual ladder (nts/conventions.QUALITY_MODES, docs/INTEGRATION.md): "native_aa" (native size,
jittered), "quality", "balanced", "performance" (= "perf"), "ultra_performance" (= "uperf"); all jittered with the
0.5 px filter and 8 x (output / render)^2 Halton phases.
"""
import json
import os
import re

import bpy

OUT = (1080, 1920)
MODES = {"uperf": (360, 640), "perf": (540, 960), "ultra": (405, 720), "quality": (720, 1280), "native": (1080, 1920), "guide": (1080, 1920),
         "hero": (1080, 1920), "roi": (1080, 1920),
         # the usual quality-mode ladder of temporal upscalers (nts/conventions.QUALITY_MODES): same factors and names.
         # "native_aa" is native resolution WITH jitter and the 0.5 px filter: temporal anti-aliasing, no upscaling.
         "native_aa": (1080, 1920), "balanced": (630, 1120), "performance": (540, 960), "ultra_performance": (360, 640),
         # "surf": surface identity per pixel for the keyframe carry and frame generation (nts/surface.py): see setup_surf
         "surf": (1080, 1920)}
UNJITTERED = ("native", "hero", "guide", "roi", "surf")
FACE_GROUPS = re.compile(r"^(head|jaw)$")
HAND_GROUPS = re.compile(r"^(hand|thumb|index|middle|ring|pinky)_", re.I)
FACE_OBJECTS = re.compile(r"eye|brow|lash|teeth|tongue", re.I)
HAIR_OBJECTS = re.compile(r"hair|beard", re.I)
REGION = "nts_region"


def halton(i, base):
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def jitter_seq(n, phases, start=0):
    return [(halton((i + start) % phases + 1, 2) - 0.5, halton((i + start) % phases + 1, 3) - 0.5) for i in range(n)]


HERO_INDEX = 77  # object pass index that gates the region AOV: props and NPCs sharing a material read as 0


def object_type(o):
    """MPFB's own asset type (Basemesh, Eyes, Eyebrows, Eyelashes, Hair, Teeth, Tongue, Clothes, ...) when available."""
    try:
        from bl_ext.user_default.mpfb.services.objectservice import ObjectService
        return ObjectService.get_object_type(o) or ""
    except Exception:
        return ""


def paint_regions(scene):
    """Per-vertex region colours on the hero's skinned meshes, read back through an AOV in each of their materials."""
    arms = [o for o in scene.objects if o.type == "ARMATURE" and not o.name.startswith("NPC")]
    hero = [o for o in scene.objects if o.type == "MESH" and
            (any(m.type == "ARMATURE" and m.object in arms for m in o.modifiers) or  # skinned body, eyes, hair, clothes
             (o.parent in arms and any(m.type == "SURFACE_DEFORM" for m in o.modifiers)))]  # beard shells
    mats = set()
    for o in hero:
        me = o.data
        if REGION in me.color_attributes:
            me.color_attributes.remove(me.color_attributes[REGION])
        attr = me.color_attributes.new(REGION, "FLOAT_COLOR", "POINT")
        kind = object_type(o).lower() or o.name.lower()
        whole = ((1.0, 0.0, 0.0) if FACE_OBJECTS.search(kind) else
                 (0.0, 0.0, 1.0) if HAIR_OBJECTS.search(kind) else None)
        face = [g.index for g in o.vertex_groups if FACE_GROUPS.match(g.name)]
        hand = [g.index for g in o.vertex_groups if HAND_GROUPS.match(g.name)]
        clothes = kind.startswith("cloth")
        for v in me.vertices:
            if whole:
                c = whole
            elif clothes:
                c = (0.0, 0.0, 0.0)  # sleeves over the wrist aren't skin
            else:
                w = {g.group: g.weight for g in v.groups}
                c = (1.0 if sum(w.get(i, 0) for i in face) > 0.5 else 0.0,
                     1.0 if sum(w.get(i, 0) for i in hand) > 0.5 else 0.0, 0.0)
            attr.data[v.index].color = (*c, 1.0)
        o.pass_index = HERO_INDEX
        mats |= {s.material for s in o.material_slots if s.material}
    for m in mats:
        if not m.use_nodes or any(n.bl_idname == "ShaderNodeOutputAOV" and n.aov_name == REGION for n in m.node_tree.nodes):
            continue
        nt = m.node_tree
        a = nt.nodes.new("ShaderNodeAttribute")
        a.attribute_name = REGION
        info = nt.nodes.new("ShaderNodeObjectInfo")
        gate = nt.nodes.new("ShaderNodeMath")
        gate.operation = "COMPARE"
        gate.inputs[1].default_value, gate.inputs[2].default_value = HERO_INDEX, 0.5
        nt.links.new(info.outputs["Object Index"], gate.inputs[0])
        mix = nt.nodes.new("ShaderNodeVectorMath")
        mix.operation = "SCALE"
        nt.links.new(a.outputs["Color"], mix.inputs[0])
        nt.links.new(gate.outputs[0], mix.inputs["Scale"])
        out = nt.nodes.new("ShaderNodeOutputAOV")
        out.aov_name = REGION
        nt.links.new(mix.outputs[0], out.inputs["Color"])
    return len(hero), len(mats)


SURF, SID, UVS = "nts_surf", "nts_sid", "nts_uv"


def paint_surface():
    """Surface-identity AOVs for the "surf" mode (nts/surface.py), in every material:
      nts_surf = the shaded point's Generated texture coordinate: its place on the object's UNDEFORMED mesh,
                 normalised to the mesh's bounding box, so it stays on the same bit of skin or cloth under armature
                 and shape-key deformation (Blender: "Generated", from the original coordinates);
      nts_sid  = (|surf|^2, the object's random number, its square);
      nts_uv   = (u, v, u^2 + v^2) of the active UV map: a mesh attribute, so it moves with the surface under any
                 deformation (on at140 the hands' Generated coordinates moved by several pixels' worth from one
                 frame to the next; their UVs do not), (0, 0, 0) where there is none.
    Averaged over a pixel's samples, a pixel that saw one surface point has sid.R == |surf|^2, sid.B == sid.G^2 and
    uv.B == uv.R^2 + uv.G^2; anything mixed (an outline under the filter, a strand, an alpha card, a UV seam) shows
    as a positive difference.
    Cycles writes an AOV at every surface a camera ray meets until one with alpha >= the view layer's pass alpha
    threshold (0.5), and adds the values up. Through the clear part of an eyelash or eyebrow card the pixel got the
    card's values PLUS the skin's: a negative difference, so it passed as one surface, with an id and coordinates
    that are neither (11-26, pty332n: the whole eyelid under the lash card, the brows, the hairline under the hair
    cards). So the three AOVs are written only where the material's alpha reaches that threshold (alpha_gate()):
    the clear part of a card writes nothing and the pixel shows the skin behind it, the same surface the renderer's
    depth and motion passes take there."""
    thr = bpy.context.scene.view_layers[0].pass_alpha_threshold
    n = 0
    for m in bpy.data.materials:
        if m.is_grease_pencil:
            continue
        if not m.use_nodes:
            m.use_nodes = True
        nt = m.node_tree
        if any(nd.bl_idname == "ShaderNodeOutputAOV" and nd.aov_name == SURF for nd in nt.nodes):
            continue
        tc = nt.nodes.new("ShaderNodeTexCoord")
        info = nt.nodes.new("ShaderNodeObjectInfo")
        sq = nt.nodes.new("ShaderNodeVectorMath")
        sq.operation = "DOT_PRODUCT"
        nt.links.new(tc.outputs["Generated"], sq.inputs[0])
        nt.links.new(tc.outputs["Generated"], sq.inputs[1])
        r2 = nt.nodes.new("ShaderNodeMath")
        r2.operation = "MULTIPLY"
        nt.links.new(info.outputs["Random"], r2.inputs[0])
        nt.links.new(info.outputs["Random"], r2.inputs[1])
        sid = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(sq.outputs["Value"], sid.inputs[0])
        nt.links.new(info.outputs["Random"], sid.inputs[1])
        nt.links.new(r2.outputs[0], sid.inputs[2])
        uq = nt.nodes.new("ShaderNodeVectorMath")
        uq.operation = "DOT_PRODUCT"
        nt.links.new(tc.outputs["UV"], uq.inputs[0])
        nt.links.new(tc.outputs["UV"], uq.inputs[1])
        us = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["UV"], us.inputs[0])
        uv = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(us.outputs[0], uv.inputs[0])
        nt.links.new(us.outputs[1], uv.inputs[1])
        nt.links.new(uq.outputs["Value"], uv.inputs[2])
        gate = alpha_gate(nt, thr)
        for name, sock in ((SURF, tc.outputs["Generated"]), (SID, sid.outputs[0]), (UVS, uv.outputs[0])):
            out = nt.nodes.new("ShaderNodeOutputAOV")
            out.aov_name = name
            if gate is not None:
                sc = nt.nodes.new("ShaderNodeVectorMath")
                sc.operation = "SCALE"
                nt.links.new(sock, sc.inputs[0])
                if isinstance(gate, float):
                    sc.inputs["Scale"].default_value = gate
                else:
                    nt.links.new(gate, sc.inputs["Scale"])
                sock = sc.outputs[0]
            nt.links.new(sock, out.inputs["Color"])
        n += 1
    return n


def alpha_source(nt, node, depth=0):
    """Where the alpha of the Principled BSDF behind shader node `node` comes from, as seen in node tree nt: an output
    socket of nt, a constant, or None (some other build; that material's AOVs are left ungated). Inside a node group
    the alpha is passed out through a new group output, nts_alpha, unless it is one of the group's own inputs."""
    if node.bl_idname == "ShaderNodeBsdfPrincipled":
        a = node.inputs["Alpha"]
        return a.links[0].from_socket if a.is_linked else float(a.default_value)
    if node.bl_idname != "ShaderNodeGroup" or node.node_tree is None or depth > 2:
        return None
    gt = node.node_tree
    go = next((o for o in gt.nodes if o.bl_idname == "NodeGroupOutput" and o.is_active_output), None)
    sh = next((i for i in go.inputs if i.type == "SHADER" and i.is_linked), None) if go else None
    inner = alpha_source(gt, sh.links[0].from_node, depth + 1) if sh else None
    if inner is None or isinstance(inner, float):
        return inner
    if inner.node.bl_idname == "NodeGroupInput":
        a = next(i for i in node.inputs if i.identifier == inner.identifier)
        return a.links[0].from_socket if a.is_linked else float(a.default_value)
    if "nts_alpha" not in go.inputs:
        gt.interface.new_socket("nts_alpha", in_out="OUTPUT", socket_type="NodeSocketFloat")
        gt.links.new(inner, go.inputs["nts_alpha"])
    return node.outputs["nts_alpha"]


def alpha_gate(nt, thr):
    """1 where material nt's surface is opaque enough for Cycles' data passes (alpha >= thr), else 0: a socket, a
    constant, or None (always 1, or a material alpha_source() cannot read)."""
    o = next((nd for nd in nt.nodes if nd.bl_idname == "ShaderNodeOutputMaterial" and nd.is_active_output), None)
    if o is None or not o.inputs["Surface"].is_linked:
        return None
    a = alpha_source(nt, o.inputs["Surface"].links[0].from_node)
    if a is None or isinstance(a, float):
        return None if a is None or a >= thr else 0.0
    ge = nt.nodes.new("ShaderNodeMath")
    ge.operation = "GREATER_THAN"
    nt.links.new(a, ge.inputs[0])
    ge.inputs[1].default_value = thr - 1e-6
    return ge.outputs[0]


def setup_surf(scene):
    """The "surf" pass: every sample of a pixel through its centre (0.01 px filter), no depth of field, no motion
    blur, no light, so a pixel that sees one opaque surface gets that surface point's coordinate exactly."""
    sc = scene.cycles
    sc.use_denoising = False
    sc.use_adaptive_sampling = False
    sc.max_bounces = sc.diffuse_bounces = sc.glossy_bounces = sc.transmission_bounces = sc.volume_bounces = 0
    sc.caustics_reflective = sc.caustics_refractive = False
    for o in scene.objects:
        if o.type == "LIGHT":
            o.hide_render = True
        if o.type == "CAMERA":
            o.data.dof.use_dof = False
    vl = scene.view_layers[0]
    # NTS_SURF_CHECK=1 also writes Vector, Albedo and Region, for checking the surface correspondence against the
    # renderer's motion and against texture that sits on the surface (the albedo's detail)
    vl.cycles.denoising_store_passes = os.environ.get("NTS_SURF_CHECK") == "1"
    have = [a.name for a in vl.aovs]
    for name in (SURF, SID, UVS):
        if name not in have:
            aov = vl.aovs.add()
            aov.name, aov.type = name, "COLOR"
    return paint_surface()


def adjust_skin(spec):
    """Look test, from the env var NTS_SKIN="value=0.7,sat=1.1,hue=-0.02,detail=0.6": scale the hero skin's base colour
    brightness/saturation, shift its hue (HSV node units, 1 = full turn) and scale its pore normal-map strength, inside MPFB's shared skin group(s). Geometry is untouched,
    so identity can't change. Returns the list of node trees edited."""
    kv = dict(x.split("=") for x in spec.split(",") if x)
    value, sat, detail = float(kv.get("value", 1)), float(kv.get("sat", 1)), float(kv.get("detail", 1))
    hue = float(kv.get("hue", 0))
    edited = []
    for nt in bpy.data.node_groups:
        if not nt.name.lower().startswith("human.body"):
            continue
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if not bsdf:
            continue
        base = bsdf.inputs["Base Color"]
        if base.is_linked:
            src = base.links[0].from_socket
            hsv = nt.nodes.new("ShaderNodeHueSaturation")
            hsv.inputs["Saturation"].default_value, hsv.inputs["Value"].default_value = sat, value
            hsv.inputs["Hue"].default_value = 0.5 + hue
            nt.links.new(src, hsv.inputs["Color"])
            nt.links.new(hsv.outputs["Color"], base)
        for n in nt.nodes:
            if n.type == "NORMAL_MAP":
                n.inputs["Strength"].default_value *= detail
            elif n.type == "BUMP":
                n.inputs["Strength"].default_value *= detail
        edited.append(nt.name)
    print(f"adjust_skin {spec}: {edited}")
    return edited


def apply_stubble(scene, density):
    """Look test only (env NTS_STUBBLE=<density 0..1>): short stubble shells on the hero's jaw, chin, cheeks and upper
    lip, reusing the rig's own beard-shell fur (scene-project cutscene/human.py shell_fur). The character's defaults in
    human.py are untouched. Returns the number of shell objects added."""
    import human
    arms = [o for o in scene.objects if o.type == "ARMATURE" and not o.name.startswith("NPC")]
    base = next(o for o in scene.objects if o.type == "MESH" and "body" in o.vertex_groups and
                any(m.type == "ARMATURE" and m.object in arms for m in o.modifiers))
    rig_ = base.parent
    # human.py builds beards with the rig at the origin in rest pose; at render time it's placed and posed per shot,
    # so recreate build conditions while the shells are made and bound, then restore
    from mathutils import Matrix
    saved_mw, saved_pose = rig_.matrix_world.copy(), rig_.data.pose_position
    rig_.matrix_world = Matrix.Identity(4)
    rig_.data.pose_position = "REST"
    bpy.context.view_layer.update()
    jaw = (rig_.matrix_world @ rig_.data.bones["head"].head_local).z
    pts = human._shaped_surface(base)
    body = base.vertex_groups["body"].index
    in_body = {v.index for v in base.data.vertices if any(e.group == body for e in v.groups)}
    mid = [p for i, (p, n) in enumerate(pts) if i in in_body and abs(p.x) < 0.005]

    def front(z0, z1):
        band = [p.y for p in mid if z0 <= p.z < z1]
        return min(band) if band else 0.0
    neck_y = front(jaw - 0.18, jaw - 0.12)
    chin = min(p.z for p in mid if jaw - 0.15 < p.z < jaw and p.y < neck_y - 0.03)
    nose = min((p for p in mid if chin < p.z < chin + 0.15), key=lambda p: p.y).z
    mouth = (chin + nose) / 2 + 0.005
    g = base.vertex_groups.new(name="stubble")
    for i, (p, n) in enumerate(pts):
        if i not in in_body or n.y > 0.3:
            continue
        ax = abs(p.x)
        lips = abs(p.z - mouth) < 0.012 and ax < 0.026
        if lips:
            continue
        if (chin - 0.012 < p.z < mouth and ax < 0.075) or \
           (mouth <= p.z < nose - 0.008 and ax < 0.028) or \
           (mouth - 0.02 <= p.z < nose + 0.012 and 0.03 < ax < 0.072):
            g.add([i], 1.0, "REPLACE")
    before = {o.name for o in bpy.data.objects}
    human.shell_fur(base, g, "Stubble", pts, shells=2, depth=0.0012)
    rig_.matrix_world, rig_.data.pose_position = saved_mw, saved_pose
    bpy.context.view_layer.update()
    new = [o for o in bpy.data.objects if o.name not in before]
    for o in new:  # stubble: sparse fine dots a little darker than skin, not a solid beard
        for s in o.material_slots:
            nt = s.material.node_tree
            for nd in nt.nodes:
                if nd.type == "BSDF_PRINCIPLED":
                    nd.inputs["Base Color"].default_value = (0.10, 0.07, 0.05, 1)
                elif nd.type == "MAPPING":
                    nd.inputs["Scale"].default_value = (2600, 2600, 2600)
                elif nd.type == "MAP_RANGE":
                    lo = nd.inputs["From Min"].default_value
                    if abs(lo - 0.32) < 1e-3:  # bald patches: keep some, fewer when denser
                        nd.inputs["From Min"].default_value = 0.32 - 0.1 * density
                        nd.inputs["From Max"].default_value = 0.46 - 0.1 * density
                    else:  # strand mask: about 10-25 % coverage
                        nd.inputs["From Min"].default_value = 0.60 - 0.06 * density
                        nd.inputs["From Max"].default_value = 0.63 - 0.06 * density
    print(f"apply_stubble density {density}: {len(new)} shells, {len(g.id_data.vertex_groups)} groups")
    return len(new)


def apply_hair_strands(scene, strength=0.4, axis="v", scale=900.0):
    """Look work (env NTS_HAIR=strength[,axis]): the hero's hair is a low-poly shell with one colour texture, so it
    renders as a smooth cap. Give it strands: fine noise stretched along one UV axis drives a bump, with anisotropic
    highlights following the same direction and a little per-strand brightness variation. Geometry is untouched."""
    n_done = 0
    for o in scene.objects:
        if o.type != "MESH" or object_type(o).lower() != "hair":
            continue
        for sl in o.material_slots:
            m = sl.material
            if not m or not m.use_nodes:
                continue
            nt = m.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue
            uv = nt.nodes.new("ShaderNodeTexCoord")
            mp = nt.nodes.new("ShaderNodeMapping")
            mp.inputs["Scale"].default_value = (scale, 6.0, 1.0) if axis == "v" else (6.0, scale, 1.0)
            nt.links.new(uv.outputs["UV"], mp.inputs["Vector"])
            noise = nt.nodes.new("ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value = 1.0
            noise.inputs["Detail"].default_value = 4.0
            nt.links.new(mp.outputs["Vector"], noise.inputs["Vector"])
            bump = nt.nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = strength
            bump.inputs["Distance"].default_value = 0.0005
            nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
            # strand-to-strand brightness variation: darken the base colour by up to 25 %
            base = bsdf.inputs["Base Color"]
            if base.is_linked:
                rng = nt.nodes.new("ShaderNodeMapRange")
                rng.inputs["To Min"].default_value, rng.inputs["To Max"].default_value = 0.75, 1.0
                nt.links.new(noise.outputs["Fac"], rng.inputs["Value"])
                mul = nt.nodes.new("ShaderNodeMix")
                mul.data_type, mul.blend_type = "RGBA", "MULTIPLY"
                mul.inputs["Factor"].default_value = 1.0
                nt.links.new(base.links[0].from_socket, mul.inputs[6])
                nt.links.new(rng.outputs["Result"], mul.inputs[7])
                nt.links.new(mul.outputs[2], base)
            # anisotropic highlights along the strands
            if "Anisotropic" in bsdf.inputs:
                bsdf.inputs["Anisotropic"].default_value = 0.7
                tan = nt.nodes.new("ShaderNodeTangent")
                tan.direction_type = "UV_MAP"
                nt.links.new(tan.outputs["Tangent"], bsdf.inputs["Tangent"])
            bsdf.inputs["Roughness"].default_value = min(bsdf.inputs["Roughness"].default_value, 0.45)
            n_done += 1
    print(f"apply_hair_strands strength={strength} axis={axis}: {n_done} materials")
    return n_done


def apply_full_look(scene, path):
    """Everything a look file (looks/*.json, or a lookfit result) sets, as real scene changes: lights, exposure and
    hair colour (apply_look), skin colour and pore relief (adjust_skin), hair strands (apply_hair_strands). Used by
    setup_nts (env NTS_LOOK) and by scene-project cutscene/scene.py --look <name>."""
    j = json.load(open(path))
    if j.get("only_realism"):
        # The episode's own look (lights, exposure, skin, hair untouched) plus the listed parts of the surface pack
        # only, e.g. looks/v11c.json: cloth. For the NR-1 route, whose tone step and keyframe edit are built on the
        # episode's own head; the judges of the held-out clips named the flat cloth, not the face (2026-11-22).
        import realism_pack
        done = {"realism": realism_pack.apply(scene, j["realism"])}
        scene["nts_look"] = path
        return done
    done = apply_look(scene, path)
    sk = j.get("absolute", j["params"])["skin"]
    spec = f"value={sk['value']},sat={sk['sat']},hue={sk['hue']}" + (f",detail={sk['detail']}" if "detail" in sk else "")
    done["skin_groups"] = adjust_skin(spec)
    if j.get("hair_strands"):
        done["hair_strands"] = apply_hair_strands(scene, *j["hair_strands"])
    if "blotch" in sk:
        done["blotch"] = adjust_blotch(sk["blotch"])
    st = j.get("strands")  # production strand-curve hair, e.g. {"n": 150000, "len": 1.5, "mel": [0.3, 0.05], "shell": 0.6}
    if st and not os.environ.get("NTS_STRANDS"):  # NTS_STRANDS (super renders) builds them in setup_nts instead
        import strand_hair
        os.environ["NTS_STRAND_MEL"] = ",".join(str(v) for v in st.get("mel", (0.25, 0.1)))
        os.environ["NTS_SHELL_DARK"] = str(st.get("shell", 0.6))
        os.environ["NTS_STRAND_UNIFORM"] = "1" if st.get("uniform") else "0"
        done["strands"] = strand_hair.build(scene, int(st.get("n", 150000)), float(st.get("len", 1.5)))
    if j.get("groom"):  # strand hair v2: fixed colour, skull-hugging clumped strands (datagen/blender/groom.py)
        import groom
        done["groom"] = groom.build(scene, j["groom"])
    if j.get("realism"):  # surface detail pack, datagen/blender/realism_pack.py
        import realism_pack
        done["realism"] = realism_pack.apply(scene, j["realism"])
    scene["nts_look"] = path
    return done


def adjust_blotch(k):
    """Scale the depth of human.skin_realism's colour variation (blotch ramp (0.95,0.78,0.77)->white and speckle ramp
    (0.86,0.84,0.84)->white) by k: 1 = as built, 2 = twice as deep. Returns the number of ramps changed."""
    n = 0
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        for nd in m.node_tree.nodes:
            if nd.type != "VALTORGB":
                continue
            c = tuple(nd.color_ramp.elements[0].color)[:3]
            for ref in ((0.95, 0.78, 0.77), (0.86, 0.84, 0.84)):
                if all(abs(a - b) < 0.02 for a, b in zip(c, ref)):
                    nd.color_ramp.elements[0].color = (*(max(0.0, 1 - k * (1 - r)) for r in ref), 1)
                    n += 1
    return n


def apply_look(scene, path):
    """A look fit (nts/lookfit.py result JSON) as real scene changes: light-group energies, hero hair albedo, exposure.
    Skin colour goes through adjust_skin. Returns counts of what was changed, for meta.json."""
    from lightgroups import apply_light_weights
    j = json.load(open(path))
    p = j.get("absolute", j["params"])  # an iterated fit carries its absolute settings (tools/look_compose.py)
    done = {"lights": apply_light_weights(scene, p["w"])}
    gain = p["hair"]
    hair_mats = set()
    for o in scene.objects:
        if o.type != "MESH" or not HAIR_OBJECTS.search(object_type(o).lower() or o.name.lower()):
            continue
        for s in o.material_slots:
            m = s.material
            if not m or not m.use_nodes or m.name in hair_mats:
                continue
            for n in list(m.node_tree.nodes):
                base = n.inputs.get("Base Color") if n.type == "BSDF_PRINCIPLED" else (
                    n.inputs.get("Color") if n.type == "BSDF_HAIR_PRINCIPLED" else None)
                if base is None:
                    continue
                mul = m.node_tree.nodes.new("ShaderNodeMix")
                mul.data_type, mul.blend_type = "RGBA", "MULTIPLY"
                mul.inputs["Factor"].default_value = 1.0
                mul.inputs[7].default_value = (*gain, 1.0)  # B
                if base.is_linked:
                    m.node_tree.links.new(base.links[0].from_socket, mul.inputs[6])
                else:
                    mul.inputs[6].default_value = base.default_value
                m.node_tree.links.new(mul.outputs[2], base)
                hair_mats.add(m.name)
    done["hair_materials"] = len(hair_mats)
    scene.view_settings.exposure += p["exposure"]
    print(f"apply_look {path}: {done}")
    return done


def _frames_arg(scene):
    """Frames this Blender run will render: scene.py's --frames a-b,c-d when given, else the whole timeline."""
    import sys
    a = sys.argv
    if "--frames" in a:
        out = []
        for r in a[a.index("--frames") + 1].split(","):
            lo, hi = map(int, r.split("-"))
            out += range(lo, hi + 1)
        return out
    return list(range(scene.frame_start, scene.frame_end + 1))


def setup_roi(scene, out_dir, pad=None):
    """Foveated render (NTS-F): only a box around the hero's head is path-traced, at full resolution, into a
    full-size frame (black outside). The box is fixed per shot (cameras are locked off): the union of the head's
    projected bounds over the shot's frames, padded. A frame-change handler moves the render border at every cut.
    Writes roi.json {"frames": {f: [x0, y0, x1, y1]}} in output pixels, row 0 at the top."""
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    pad = float(os.environ.get("NTS_ROI_PAD", 1.45)) if pad is None else pad
    rig = next((o for o in scene.objects if o.type == "ARMATURE" and not o.name.upper().startswith("NPC")
                and "head" in o.pose.bones), None)
    if rig is None:
        raise RuntimeError("roi: no hero rig with a head bone")
    frames = _frames_arg(scene)
    cuts = sorted({m.frame for m in scene.timeline_markers if m.camera} | {scene.frame_start})
    shot_of = lambda f: max(c for c in cuts if c <= f) if any(c <= f for c in cuts) else cuts[0]  # noqa: E731
    W, H = MODES["roi"]
    per_shot = {}
    cur = scene.frame_current
    for f in sorted(set(frames[::3]) | {frames[0], frames[-1]}):  # every third frame is enough for a padded box
        scene.frame_set(f)
        b = rig.pose.bones["head"]
        h0, h1 = rig.matrix_world @ b.head, rig.matrix_world @ b.tail
        c, r = (h0 + h1) / 2 + (h1 - h0) * 0.25, (h1 - h0).length * pad  # head bone: neck top to crown, ~16 cm
        pts = [world_to_camera_view(scene, scene.camera, c + Vector(d) * r)
               for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
        if min(p.z for p in pts) <= 0:
            continue
        xs, ys = [p.x for p in pts], [p.y for p in pts]
        hw = max(max(xs) - min(xs), (max(ys) - min(ys)) * H / W) / 2  # square in pixels, as wide as the widest axis
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        box = [cx - hw, cy - hw * W / H, cx + hw, cy + hw * W / H]
        k = shot_of(f)
        o = per_shot.get(k)
        per_shot[k] = box if o is None else [min(o[0], box[0]), min(o[1], box[1]), max(o[2], box[2]), max(o[3], box[3])]
    scene.frame_set(cur)
    for k, b in per_shot.items():  # clamp to the frame and snap outward to whole pixels
        per_shot[k] = [max(0.0, int(b[0] * W) / W), max(0.0, int(b[1] * H) / H),
                       min(1.0, (int(b[2] * W) + 1) / W), min(1.0, (int(b[3] * H) + 1) / H)]
    r = scene.render
    r.use_border, r.use_crop_to_border = True, False

    def apply(sc, *_):
        b = per_shot.get(shot_of(sc.frame_current)) or next(iter(per_shot.values()))
        sc.render.border_min_x, sc.render.border_min_y, sc.render.border_max_x, sc.render.border_max_y = b
    apply(scene)
    bpy.app.handlers.frame_change_pre.append(apply)
    px = {str(f): [round(b[0] * W), round((1 - b[3]) * H), round(b[2] * W), round((1 - b[1]) * H)]
          for f in frames for b in [per_shot.get(shot_of(f))] if b}
    with open(os.path.join(out_dir, "roi.json"), "w") as fp:
        json.dump({"units": "output px [x0, y0, x1, y1], row 0 at the top", "pad": pad, "frames": px}, fp)
    area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in per_shot.values()) / max(len(per_shot), 1)
    print(f"roi: {len(per_shot)} shot boxes, mean {area:.1%} of the frame", flush=True)
    return per_shot


def setup_nts(scene, out_dir, mode="perf", seed=0, halton_start=0, filter_width=None):
    if os.environ.get("NTS_LOOK_SEARCH"):  # clean-room look search: renders candidates and exits Blender
        from look_search import run
        run(scene, os.environ["NTS_LOOK_SEARCH"], out_dir)
    w, h = MODES[mode]
    native = mode in UNJITTERED
    if mode == "guide":  # full-res texture guide for baselines/guided.py: only Albedo/Normal/Depth/Region matter
        scene.cycles.use_denoising = False
        scene.cycles.use_adaptive_sampling = False  # every pixel's albedo gets the same number of AA samples
        scene.cycles.diffuse_bounces = 0  # no indirect light; glossy/transmission stay so the albedo pass follows
        scene.cycles.volume_bounces = 0   # mirror-like paths exactly as in the low-res render
        scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False
        if os.environ.get("NTS_GUIDE_DARK") == "1":  # albedo needs no light: skip light sampling and shadow rays
            for o in scene.objects:
                if o.type == "LIGHT":
                    o.hide_render = True
            if scene.world and scene.world.use_nodes:
                for n in scene.world.node_tree.nodes:
                    if n.type == "BACKGROUND":
                        n.inputs["Strength"].default_value = 0.0
    if mode == "hero":
        from render_hero import setup_hero
        setup_hero(scene)
    if os.environ.get("NTS_STRANDS"):  # strand-curve hair on the MPFB shell (super renders)
        import strand_hair
        strand_hair.from_env(scene)
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False  # Cycles only writes the Vector pass without motion blur
    scene.render.use_persistent_data = True
    scene.cycles.seed = seed
    if filter_width is None:
        filter_width = 0.01 if mode == "surf" else 1.5 if native else 0.5
    scene.cycles.filter_width = filter_width
    vl = scene.view_layers[0]
    vl.use_pass_z = vl.use_pass_vector = True
    eevee = scene.render.engine.startswith("BLENDER_EEVEE")
    if eevee:  # rasterised guide (baselines/guided.py): albedo = diffuse + glossy colour, clean AA, ~free
        vl.use_pass_diffuse_color = vl.use_pass_glossy_color = vl.use_pass_normal = True
    else:
        vl.cycles.denoising_store_passes = True  # Denoising Albedo / Normal
    if REGION not in [a.name for a in vl.aovs]:
        aov = vl.aovs.add()
        aov.name, aov.type = REGION, "COLOR"
    n_hero, n_mats = paint_regions(scene)
    lg_names, lg_counts = [], {}
    if os.environ.get("NTS_LIGHTGROUPS") == "1":  # per-emitter passes for the exact look fit
        from lightgroups import setup_lightgroups
        lg_names, lg_counts = setup_lightgroups(scene)
    skin = os.environ.get("NTS_SKIN", "")
    if os.environ.get("NTS_DUMP_HAIR"):  # inspect the hero's meshes and their shading, then exit (look work)
        arms = [o for o in scene.objects if o.type == "ARMATURE" and not o.name.startswith("NPC")]
        for o in scene.objects:
            if o.type != "MESH" or not (o.parent in arms or any(m.type == "ARMATURE" and m.object in arms
                                                                  for m in o.modifiers)):
                continue
            print(f"MESH {o.name} type={object_type(o)} verts={len(o.data.vertices)} psys={len(o.particle_systems)}",
                  flush=True)
            for sl in o.material_slots:
                m = sl.material
                if m and m.use_nodes:
                    print(f"  MAT {m.name}", flush=True)
                    for nd in m.node_tree.nodes:
                        extra = f" img={nd.image.name} {tuple(nd.image.size)}" if nd.type == "TEX_IMAGE" and nd.image else ""
                        print(f"    NODE {nd.type} {nd.name}{extra}", flush=True)
        os._exit(0)  # before any render: no GPU use
    look = os.environ.get("NTS_LOOK", "")
    hair_spec = os.environ.get("NTS_HAIR", "")
    if hair_spec:
        hs = hair_spec.split(",")
        apply_hair_strands(scene, float(hs[0]), hs[1] if len(hs) > 1 else "v")
    stubble = apply_stubble(scene, float(os.environ["NTS_STUBBLE"])) if os.environ.get("NTS_STUBBLE") else 0
    look_done = apply_look(scene, look) if look else {}
    if look:
        j = json.load(open(look))
        sk = j.get("absolute", j["params"])["skin"]
        skin = f"value={sk['value']},sat={sk['sat']},hue={sk['hue']}" + (f",detail={sk['detail']}" if "detail" in sk else "")
    edited = adjust_skin(skin) if skin else []
    n_surf = setup_surf(scene) if mode == "surf" else 0  # after the look: it may add or change materials
    os.makedirs(out_dir, exist_ok=True)

    scene.use_nodes = True
    nt = scene.node_tree
    rl, comp = nt.nodes["Render Layers"], nt.nodes["Composite"]
    image = comp.inputs["Image"].links[0].from_socket if comp.inputs["Image"].links else rl.outputs["Image"]
    fo = nt.nodes.new("CompositorNodeOutputFile")
    fo.format.file_format = "OPEN_EXR_MULTILAYER"
    fo.format.color_depth = "32" if mode == "surf" else "16"  # surface coordinates need float precision
    fo.format.exr_codec = "ZIP"
    fo.base_path = os.path.join(out_dir, "f_")
    if mode == "surf":  # Depth (for the re-seating's layers) and the surface AOVs only; no colour
        fo.layer_slots[0].name = "Depth"
        nt.links.new(rl.outputs["Depth"], fo.inputs[0])

        def xyz(sock):  # an AOV's three values as a vector: saved as X, Y, Z without the AOV's constant alpha
            sep, comb = nt.nodes.new("CompositorNodeSeparateColor"), nt.nodes.new("CompositorNodeCombineXYZ")
            nt.links.new(sock, sep.inputs[0])
            for i in range(3):
                nt.links.new(sep.outputs[i], comb.inputs[i])
            return comb.outputs[0]
        slots = (("Surf", xyz(rl.outputs[SURF])), ("Sid", xyz(rl.outputs[SID])), ("Uv", xyz(rl.outputs[UVS])))
        if os.environ.get("NTS_SURF_CHECK") == "1":
            slots += (("Vector", rl.outputs["Vector"]), ("Albedo", rl.outputs["Denoising Albedo"]),
                      ("Region", rl.outputs[REGION]))
    else:
        nt.links.new(image, fo.inputs[0])
        if eevee:
            add = nt.nodes.new("CompositorNodeMixRGB")
            add.blend_type, add.inputs[0].default_value = "ADD", 1.0
            nt.links.new(rl.outputs["DiffCol"], add.inputs[1])
            nt.links.new(rl.outputs["GlossCol"], add.inputs[2])
            albedo, normal = add.outputs[0], rl.outputs["Normal"]
        else:
            albedo, normal = rl.outputs["Denoising Albedo"], rl.outputs["Denoising Normal"]
        slots = (("Depth", rl.outputs["Depth"]), ("Vector", rl.outputs["Vector"]), ("Albedo", albedo),
                 ("Normal", normal), ("Region", rl.outputs[REGION]))
    for slot, sock in slots:
        fo.layer_slots.new(slot)
        nt.links.new(sock, fo.inputs[slot])
    if lg_names:
        from lightgroups import add_lightgroup_outputs
        bpy.context.view_layer.update()
        add_lightgroup_outputs(scene, fo, lg_names)
    scene.render.filepath = os.path.join(out_dir, "lowres_")
    scene.render.image_settings.file_format = "PNG"

    n = scene.frame_end
    phases = max(8, round(8 * (OUT[1] / h) ** 2))
    jit = jitter_seq(n, phases, halton_start) if not native else [(0.0, 0.0)] * n
    side = max(w, h)  # Blender's lens shift is a fraction of the larger frame side
    cams = [o for o in scene.objects if o.type == "CAMERA"]
    for cam in cams:
        sx, sy = cam.data.shift_x, cam.data.shift_y
        for f, (jx, jy) in enumerate(jit, start=1):
            cam.data.shift_x, cam.data.shift_y = sx + jx / side, sy + jy / side
            cam.data.keyframe_insert("shift_x", frame=f)
            cam.data.keyframe_insert("shift_y", frame=f)
    with open(os.path.join(out_dir, "jitter.json"), "w") as fp:
        json.dump({"phases": phases, "start": halton_start, "units": "render px, +x right +y up, applied as lens shift",
                   "frames": {str(f): j for f, j in enumerate(jit, start=1)}}, fp)
    cuts = sorted({m.frame for m in scene.timeline_markers if m.camera} | {scene.frame_start})
    with open(os.path.join(out_dir, "cuts.json"), "w") as fp:
        json.dump(cuts, fp)
    if mode == "roi":
        setup_roi(scene, out_dir)
    cam = scene.camera or (cams[0] if cams else None)
    with open(os.path.join(out_dir, "meta.json"), "w") as fp:
        json.dump({"render": [w, h], "out": list(OUT), "mode": mode, "filter_width": filter_width, "seed": seed,
                   "samples": scene.cycles.samples, "skin": skin, "skin_groups": edited, "look": look, "look_applied": look_done, "stubble_shells": stubble, "lightgroups": lg_counts, "hero_meshes": n_hero, "region_materials": n_mats, "surf_materials": n_surf,
                   "clip": [cam.data.clip_start, cam.data.clip_end] if cam else [0.1, 100.0],
                   "view": [scene.view_settings.view_transform, scene.view_settings.look, scene.view_settings.exposure],
                   "frames": [scene.frame_start, scene.frame_end]}, fp)
    return w, h
