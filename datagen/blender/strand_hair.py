"""Strand hair for super renders (the realism-pass plan, stage A): real curve geometry on MPFB's hair shell.

MPFB hair is a low-poly shell with one colour texture, so even the hero render shows a smooth cap. This grows N curve
strands on the shell: each root is a fixed (triangle, barycentric) sample, area-weighted with a fixed seed, so strands
are the same every frame. Each strand runs along the shell's UV V direction (the strand direction in MPFB's textures;
see render_lr.apply_hair_strands) and takes its colour (as melanin and redness) from the shell texture at its root UV. Shading is the
Principled Hair BSDF (melanin fitted from the texture), with the shell kept underneath, darkened, as the dense under-layer.
Positions are rebuilt from the evaluated (armature-deformed) shell on every frame change, so strands follow the head.

Enabled with env NTS_STRANDS=<count>[,<length cm>] in render_lr.setup_nts (opt-in, e.g. NTS_STRANDS=150000,1.5).
"""
import math
import os

import bpy
import numpy as np

PTS = 5
_state = []


def _tris(ob_eval):
    me = ob_eval.to_mesh()
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    vi, li = np.zeros(nt * 3, np.int32), np.zeros(nt * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", vi)
    me.loop_triangles.foreach_get("loops", li)
    co = np.zeros(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    uv = np.zeros(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    ob_eval.to_mesh_clear()
    m = np.array(ob_eval.matrix_world, np.float32)
    co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    return co[vi.reshape(-1, 3)], uv.reshape(-1, 2)[li.reshape(-1, 3)]


def _boundary_dist(ob_eval, roots):
    """Distance from each root to the shell's open edge (edges with one face): the hairline, nape and fringe tips."""
    import bmesh
    from mathutils import kdtree
    bm = bmesh.new()
    bm.from_object(ob_eval, bpy.context.evaluated_depsgraph_get())
    m = ob_eval.matrix_world
    pts = {v.index: m @ v.co for e in bm.edges if len(e.link_faces) == 1 for v in e.verts}
    bm.free()
    if not pts:
        return np.full(len(roots), 1.0, np.float32)
    kd = kdtree.KDTree(len(pts))
    for i, (_, c) in enumerate(pts.items()):
        kd.insert(c, i)
    kd.balance()
    return np.array([kd.find(r)[2] for r in roots], np.float32)


def _frames(P, UV, tri, bary):
    """Root position, normal and unit dP/dv for each strand from its triangle."""
    p = P[tri]
    root = (p * bary[..., None]).sum(1)
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    u = UV[tri]
    du1, du2 = u[:, 1] - u[:, 0], u[:, 2] - u[:, 0]
    det = du1[:, 0] * du2[:, 1] - du2[:, 0] * du1[:, 1]
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    tv = (-du2[:, 0, None] * e1 + du1[:, 0, None] * e2) / det[:, None]  # dP/dv
    tv -= n * (tv * n).sum(1, keepdims=True)
    tv /= np.linalg.norm(tv, axis=1, keepdims=True) + 1e-12
    return root, n, tv


def _texture(ob):
    for sl in ob.material_slots:
        m = sl.material
        if m and m.use_nodes:
            for nd in m.node_tree.nodes:
                if nd.type == "TEX_IMAGE" and nd.image and nd.outputs["Color"].is_linked:
                    im = nd.image
                    px = np.zeros(im.size[0] * im.size[1] * 4, np.float32)
                    im.pixels.foreach_get(px)
                    px = px.reshape(im.size[1], im.size[0], 4)
                    if im.colorspace_settings.name.lower().startswith("srgb"):  # pixels come unconverted
                        px[..., :3] = np.where(px[..., :3] <= 0.04045, px[..., :3] / 12.92,
                                               ((px[..., :3] + 0.055) / 1.055) ** 2.4)
                    return px
    return None


def _material(name):
    from render_lr import HERO_INDEX, REGION
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    h = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    h.parametrization = "MELANIN"
    h.inputs["Roughness"].default_value = 0.3
    h.inputs["Radial Roughness"].default_value = 0.4
    if "Random Color" in h.inputs:
        h.inputs["Random Color"].default_value = 0.15
    if "Random Roughness" in h.inputs:
        h.inputs["Random Roughness"].default_value = 0.2
    for name, sock in (("strand_mel", "Melanin"), ("strand_red", "Melanin Redness")):
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_name = name
        nt.links.new(at.outputs["Fac"], h.inputs[sock])
    nt.links.new(h.outputs[0], out.inputs["Surface"])
    # Region AOV: hair bit (B) on the hero's strands, like the shell (render_lr tags hero materials the same way)
    info = nt.nodes.new("ShaderNodeObjectInfo")
    gate = nt.nodes.new("ShaderNodeMath")
    gate.operation = "COMPARE"
    gate.inputs[1].default_value, gate.inputs[2].default_value = HERO_INDEX, 0.5
    nt.links.new(info.outputs["Object Index"], gate.inputs[0])
    comb = nt.nodes.new("ShaderNodeCombineColor")
    nt.links.new(gate.outputs[0], comb.inputs[2])
    aov = nt.nodes.new("ShaderNodeOutputAOV")
    aov.aov_name = REGION
    nt.links.new(comb.outputs[0], aov.inputs["Color"])
    return m


def _flat_shell(ob, rgb):
    """Shell base colour -> one flat linear colour (alpha/cards untouched), the dense under-layer of uniform strands."""
    for sl in ob.material_slots:
        m = sl.material
        if not m or not m.use_nodes:
            continue
        b = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if b is None:
            continue
        base = b.inputs["Base Color"]
        for ln in list(base.links):
            m.node_tree.links.remove(ln)
        base.default_value = (*(float(v) for v in rgb), 1.0)
        b.inputs["Roughness"].default_value = 0.6


def _darken_shell(ob, f=0.6):
    for sl in ob.material_slots:
        m = sl.material
        if not m or not m.use_nodes:
            continue
        b = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if b is None:
            continue
        base = b.inputs["Base Color"]
        mul = m.node_tree.nodes.new("ShaderNodeMix")
        mul.data_type, mul.blend_type = "RGBA", "MULTIPLY"
        mul.inputs["Factor"].default_value = 1.0
        mul.inputs[7].default_value = (f, f, f, 1)
        if base.is_linked:
            m.node_tree.links.new(base.links[0].from_socket, mul.inputs[6])
        else:
            mul.inputs[6].default_value = base.default_value
        m.node_tree.links.new(mul.outputs[2], base)


def _positions(s, P, UV):
    root, n, t = _frames(P, UV, s["tri"], s["bary"])
    b = np.cross(n, t)
    k = np.linspace(0, 1, PTS, dtype=np.float32)[None, :, None]
    L = s["len"][:, None, None] * s["sign"]
    pos = (root[:, None] - n[:, None] * 0.001 + t[:, None] * L * k
           + n[:, None] * (s["lift"][:, None, None] * np.sin(math.pi * k * 0.8))
           + b[:, None] * (s["wave"][:, None, None] * k * k))
    return pos.reshape(-1, 3)


def _update(scene, depsgraph=None):
    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    for s in _state:
        P, UV = _tris(s["shell"].evaluated_get(dg))
        s["curves"].data.position_data.foreach_set("vector", _positions(s, P, UV).ravel())
        s["curves"].data.update_tag()


def build(scene, count=120000, length_cm=3.0, seed=0):
    from render_lr import object_type
    dg = bpy.context.evaluated_depsgraph_get()
    made = 0
    for ob in [o for o in scene.objects if o.type == "MESH" and object_type(o).lower() == "hair"]:
        P, UV = _tris(ob.evaluated_get(dg))
        area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
        rng = np.random.default_rng(seed)
        tri = rng.choice(len(P), count, p=area / area.sum())
        r1, r2 = rng.random(count), rng.random(count)
        sq = np.sqrt(r1)
        bary = np.stack([1 - sq, sq * (1 - r2), sq * r2], 1).astype(np.float32)
        s = {"shell": ob, "tri": tri, "bary": bary,
             "len": (length_cm / 100 * rng.uniform(0.6, 1.3, count)).astype(np.float32),
             "lift": rng.uniform(0.0002, 0.0012, count).astype(np.float32),
             "wave": rng.normal(0, 0.0015, count).astype(np.float32)}
        # V runs one way over the whole shell; strands grow the way that points down on average (crown to tips)
        root, _, t = _frames(P, UV, tri, bary)
        # no strand reaches past the shell's open edge, so the fringe and hairline (the face's frame) keep their shape
        bd = _boundary_dist(ob.evaluated_get(dg), root)
        s["len"] = np.minimum(s["len"], 0.8 * bd).astype(np.float32)
        # no strands rooted within 1.2 cm of the edge: there they shadowed the forehead into a dark band that the
        # realism pass learned to paint (runs/compare_ntsr4/face_0430.png); the shell edge stays as the hairline
        keep = bd >= 0.012
        count = int(keep.sum())
        for k in ("tri", "bary", "len", "lift", "wave"):
            s[k] = s[k][keep]
        tri, bary = s["tri"], s["bary"]
        s["sign"] = np.float32(1.0 if (t[:, 2]).mean() < 0 else -1.0)
        cd = bpy.data.hair_curves.new(f"{ob.name}_strands")
        cd.add_curves([PTS] * count)
        rad = np.linspace(0.00007, 0.00002, PTS, dtype=np.float32)
        ra = cd.attributes.get("radius") or cd.attributes.new("radius", "FLOAT", "POINT")
        ra.data.foreach_set("value", np.tile(rad, count))
        tex = _texture(ob)
        col = np.full((count, 3), 0.05, np.float32)
        if tex is not None:  # linear colour at the root; transparent card texels (often white) take the opaque mean
            uv = (UV[tri] * bary[..., None]).sum(1) % 1.0
            h, w = tex.shape[:2]
            px = tex[(uv[:, 1] * (h - 1)).astype(int), (uv[:, 0] * (w - 1)).astype(int)]
            op = tex[..., 3] > 0.5
            mean = tex[..., :3][op].mean(0) if op.any() else col[0]
            col = np.where(px[:, 3:4] > 0.5, px[:, :3], mean).astype(np.float32)
        # physically based colour: direct colouring turned brown textures ash-grey (runs/gen_test/strand_col.jpg), so
        # fit Principled Hair melanin from the texture's linear luminance (darker = more melanin) and redness from
        # its red-blue balance, per strand, with mild per-strand variation
        # env NTS_STRAND_MEL=<melanin at luminance 0.125>,<redness scale>; 0.35,0.3 matched a neutral brown by eye
        # (runs/gen_test/strand_mel3.jpg); higher redness went ginger
        m0, rs = (float(v) for v in os.environ.get("NTS_STRAND_MEL", "0.25,0.1").split(","))
        lum = np.maximum(col @ np.array([0.2126, 0.7152, 0.0722], np.float32), 1e-4)
        lum = 0.3 * lum + 0.7 * lum.mean()  # the texture has lighting baked in; full per-strand contrast looked clumpy
        uniform = os.environ.get("NTS_STRAND_UNIFORM") == "1"
        if uniform:  # one colour per character: per-root texture sampling copied MPFB's card patches (runs/variety_v6)
            col = np.broadcast_to(col.mean(0), col.shape).astype(np.float32)
            lum = np.full(count, lum.mean(), np.float32)
        mel = np.clip(m0 - 0.32 * np.log2(lum / 0.125), 0.05, 0.98) + rng.normal(0, 0.03, count)
        red = np.clip((col[:, 0] - col[:, 2]) / (col[:, 0] + 1e-4) * rs, 0, 1)
        for name, v in (("strand_mel", np.clip(mel, 0, 1)), ("strand_red", red)):
            cd.attributes.new(name, "FLOAT", "CURVE").data.foreach_set("value", v.astype(np.float32))
        cd.materials.append(_material(f"{ob.name}_strand_mat"))
        co = bpy.data.objects.new(f"{ob.name}_strands", cd)
        co.pass_index = ob.pass_index
        scene.collection.objects.link(co)
        s["curves"] = co
        _state.append(s)
        if uniform:  # flat under-layer in the strands' own colour, so no texture blocks show between strands
            _flat_shell(ob, col.mean(0) * float(os.environ.get("NTS_SHELL_DARK", 0.6)))
        else:
            _darken_shell(ob, float(os.environ.get("NTS_SHELL_DARK", 0.6)))  # under-layer albedo factor
        made += count
    if made:
        scene.cycles_curves.shape = "THICK"
        bpy.app.handlers.frame_change_post.append(_update)
        _update(scene, dg)
    print(f"strand_hair: {made} strands on {len(_state)} shells, length ~{length_cm} cm", flush=True)
    return made


def from_env(scene):
    v = os.environ.get("NTS_STRANDS")
    if not v:
        return 0
    parts = v.split(",")
    return build(scene, int(parts[0]), float(parts[1]) if len(parts) > 1 else 3.0)
