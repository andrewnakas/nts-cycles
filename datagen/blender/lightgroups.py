"""Light-group decomposition for the exact look fit (nts/lookfit.py).

Path tracing is linear in emitter strength, so a frame rendered with every emitter in a named light group can be
re-lit afterwards exactly: image(w) = sum_g w_g * pass_g. One render per frame then lets numpy solve the global light
settings across many frames in milliseconds per evaluation, instead of re-rendering candidates.

Groups: one per hero rig light (HeroKey, HeroFill, HeroRim, HeroKicker, HeroEyes), "set" (every other lamp), "emit"
(emissive meshes: monitors, LED strips), "world". Each group's Combined pass is denoised in the compositor with the
render's own albedo/normal, and written as a layer LG_<group> in the multilayer EXR alongside the usual passes.

Enabled from render_lr.setup_nts with env NTS_LIGHTGROUPS=1 (use with --nts native).
"""
import bpy

HERO = ("HeroKey", "HeroFill", "HeroRim", "HeroKicker", "HeroEyes")


def is_emissive(mat):
    if not mat or not mat.use_nodes:
        return False
    for n in mat.node_tree.nodes:
        if n.type == "EMISSION" and n.inputs["Strength"].default_value > 0:
            return True
        if n.type == "BSDF_PRINCIPLED":
            s = n.inputs.get("Emission Strength")
            if s is not None and (s.default_value > 0 or s.is_linked):
                col = n.inputs.get("Emission Color")
                if col is None or col.is_linked or any(c > 0 for c in col.default_value[:3]):
                    return True
    return False


def group_of(o):
    """The light group an object's emission belongs to, or None."""
    if o.type == "LIGHT":
        return o.name if o.name in HERO else "set"
    if o.type in ("MESH", "CURVE", "FONT") and any(is_emissive(s.material) for s in o.material_slots):
        return "emit"
    return None


def scale_emission(mat, k):
    """Multiply a material's emission by k (Emission nodes and Principled emission strength, linked or not)."""
    n_done = 0
    for n in mat.node_tree.nodes:
        sock = (n.inputs["Strength"] if n.type == "EMISSION" else
                n.inputs.get("Emission Strength") if n.type == "BSDF_PRINCIPLED" else None)
        if sock is None:
            continue
        if sock.is_linked:
            m = mat.node_tree.nodes.new("ShaderNodeMath")
            m.operation, m.inputs[1].default_value = "MULTIPLY", k
            mat.node_tree.links.new(sock.links[0].from_socket, m.inputs[0])
            mat.node_tree.links.new(m.outputs[0], sock)
        else:
            sock.default_value *= k
        n_done += 1
    return n_done


def apply_light_weights(scene, w):
    """Make the light-group weights of a look fit (nts/lookfit.py) real: the same groups setup_lightgroups uses."""
    done, mats = dict.fromkeys(w, 0), set()
    for o in scene.objects:
        g = group_of(o)
        if g is None or g not in w:
            continue
        if o.type == "LIGHT":
            o.data.energy *= w[g]
            done[g] += 1
        else:
            for s in o.material_slots:
                if s.material and is_emissive(s.material) and s.material.name not in mats:
                    mats.add(s.material.name)
                    done[g] += scale_emission(s.material, w[g])
    if scene.world and "world" in w and scene.world.use_nodes:
        for n in scene.world.node_tree.nodes:
            if n.type == "BACKGROUND":
                n.inputs["Strength"].default_value *= w["world"]
                done["world"] += 1
    return done


def setup_lightgroups(scene):
    vl = scene.view_layers[0]
    names = list(HERO) + ["set", "emit", "world"]
    have = {g.name for g in vl.lightgroups}
    for n in names:
        if n not in have:
            vl.lightgroups.add(name=n)
    counts = dict.fromkeys(names, 0)
    for o in scene.objects:
        g = group_of(o)
        if g is None:
            continue
        o.lightgroup = g
        counts[g] += 1
    if scene.world:
        scene.world.lightgroup = "world"
        counts["world"] = 1
    return names, counts


def add_lightgroup_outputs(scene, file_output, names):
    """Denoise each Combined_<group> pass with the render's albedo/normal and wire it into the multilayer File Output."""
    nt = scene.node_tree
    rl = nt.nodes["Render Layers"]
    # the production compositor adds depth haze, mix(image, haze, 0.55 * mist): affine, so the fit re-applies it after
    # summing the groups and needs the raw mist factor
    if "Mist" in rl.outputs:
        file_output.layer_slots.new("Mist")
        nt.links.new(rl.outputs["Mist"], file_output.inputs["Mist"])
    for n in names:
        sock = rl.outputs.get(f"Combined_{n}")
        if sock is None:
            continue
        dn = nt.nodes.new("CompositorNodeDenoise")
        dn.prefilter = "ACCURATE"
        nt.links.new(sock, dn.inputs["Image"])
        if "Denoising Albedo" in rl.outputs:
            nt.links.new(rl.outputs["Denoising Albedo"], dn.inputs["Albedo"])
            nt.links.new(rl.outputs["Denoising Normal"], dn.inputs["Normal"])
        file_output.layer_slots.new(f"LG_{n}")
        nt.links.new(dn.outputs["Image"], file_output.inputs[f"LG_{n}"])
