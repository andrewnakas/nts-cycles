"""Clean-room look search: get Cycles itself to render the look the user picked (the reference pass 2 was the visual
guide), by tuning scene settings only: skin material, hero lights, exposure. The training data stays 100% our own
renders. The reference image is looked at by eval/look_score.py, never trained on (see PROVENANCE.md).

Builds the scene once, then renders one frame per candidate. Triggered from render_lr.setup_nts when the env var
NTS_LOOK_SEARCH points at a candidates JSON:
  [{"name": "c00", "skin_value": 0.8, "skin_sat": 1.0, "sss": 0.3, "skin_detail": 1.5, "exposure": 0.35, "look": "AgX - Punchy",
    "lights": {"HeroKey": [energy_mult, size_mult], "HeroFill": [..], ...}}, ...]
Writes <out_dir>/<name>.png (AgX display, what the pipeline would show) and exits Blender.
With several frames (NTS_LOOK_FRAME="60,185,430"), writes <name>_f####.png per frame.
Env: NTS_LOOK_FRAME (default 430; comma list allowed), NTS_LOOK_RES (default 540x960), NTS_LOOK_SAMPLES (default 32).
"""
import json
import os
import sys

import bpy


def skin_groups():
    return [nt for nt in bpy.data.node_groups if nt.name.lower().startswith("human.body")]


def skin_hsv_nodes():
    """One Hue/Saturation node in front of each skin group's Base Color, created once and then set per candidate."""
    nodes = []
    for nt in skin_groups():
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if not bsdf or not bsdf.inputs["Base Color"].is_linked:
            continue
        hsv = nt.nodes.get("NTS_look_hsv")
        if hsv is None:
            src = bsdf.inputs["Base Color"].links[0].from_socket
            hsv = nt.nodes.new("ShaderNodeHueSaturation")
            hsv.name = "NTS_look_hsv"
            nt.links.new(src, hsv.inputs["Color"])
            nt.links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
        nodes.append(hsv)
    return nodes


def sss_inputs():
    out = []
    for m in bpy.data.materials:
        if m.use_nodes:
            out += [n.inputs["SSS strength"] for n in m.node_tree.nodes if n.type == "GROUP" and "SSS strength" in n.inputs]
    return out


def skin_detail_inputs():
    """Normal-map / bump strengths inside the skin group(s): the pore and wrinkle relief."""
    out = []
    for nt in skin_groups():
        out += [n.inputs["Strength"] for n in nt.nodes if n.type in ("NORMAL_MAP", "BUMP")]
    return out


def run(scene, cand_path, out_dir):
    cands = json.load(open(cand_path))
    frames = [int(x) for x in os.environ.get("NTS_LOOK_FRAME", "430").split(",")]
    w, h = map(int, os.environ.get("NTS_LOOK_RES", "540x960").split("x"))
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = w, h, 100
    scene.cycles.samples = int(os.environ.get("NTS_LOOK_SAMPLES", 32))
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = "PNG"
    scene.use_nodes = True  # keep the production compositor (glare off, haze) so the look includes it
    os.makedirs(out_dir, exist_ok=True)
    hsv, sss = skin_hsv_nodes(), sss_inputs()
    base_sss = [s.default_value for s in sss]
    detail = skin_detail_inputs()
    base_detail = [d.default_value for d in detail]
    lights = {o.name: o.data for o in scene.objects if o.type == "LIGHT" and o.name.startswith("Hero")}
    base_l = {k: (l.energy, l.size) for k, l in lights.items()}
    base_exp, base_look = scene.view_settings.exposure, scene.view_settings.look
    log = []
    for c in cands:
        for n in hsv:
            n.inputs["Value"].default_value = c.get("skin_value", 1.0)
            n.inputs["Saturation"].default_value = c.get("skin_sat", 1.0)
        for s, b in zip(sss, base_sss):
            s.default_value = c.get("sss", b)
        for d, b in zip(detail, base_detail):
            d.default_value = b * c.get("skin_detail", 1.0)
        for k, l in lights.items():
            em, sm = c.get("lights", {}).get(k, (1.0, 1.0))
            l.energy, l.size = base_l[k][0] * em, base_l[k][1] * sm
        scene.view_settings.exposure = c.get("exposure", base_exp)
        scene.view_settings.look = c.get("look", base_look)
        for f in frames:
            scene.frame_set(f)  # timeline markers pick the shot's camera
            tag = f"{c['name']}.png" if len(frames) == 1 else f"{c['name']}_f{f:04d}.png"
            scene.render.filepath = os.path.join(out_dir, tag)
            bpy.ops.render.render(write_still=True)
        log.append(c)
    json.dump({"frames": frames, "res": [w, h], "skin_groups": len(hsv), "sss_inputs": len(sss), "detail_inputs": len(detail), "lights": list(lights),
               "candidates": log}, open(os.path.join(out_dir, "search.json"), "w"), indent=1)
    print(f"look_search: {len(cands)} candidates -> {out_dir}", flush=True)
    sys.exit(0)
