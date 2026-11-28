"""Hero-Cycles: the expensive render the realism pass learns toward. Same geometry, camera and animation as production,
so the target can't change anyone's identity; only light transport and material fidelity go up.

Used through render_lr.setup_nts(scene, out, "hero") (scene.py: --nts hero --samples 2048). Settings, all measurable
by toggling them off again:
- samples from --samples (default 2048 here), adaptive threshold 0.004, OIDN with the accurate prefilter
- bounces 16 total / 12 diffuse / 12 glossy / 16 transmission, volume 4, caustics on, filter glossy 0.5 (not 1.0)
- skin: random-walk (skin) subsurface on every Principled BSDF that has subsurface weight
- displacement: materials with a displacement output render "displacement and bump" with adaptive subdivision off
  (dicing would multiply VRAM on the 6 GB 4050; cloud boxes can turn it on with NTS_HERO_DICING=1)
- light tree on, clamp indirect 10 (keeps fireflies out of the target without dulling highlights)
"""
import os

import bpy


def setup_hero(scene, samples=None):
    c = scene.cycles
    c.samples = samples or (c.samples if c.samples > 512 else 2048)
    c.adaptive_threshold = 0.004
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    c.denoising_prefilter = "ACCURATE"
    c.max_bounces, c.diffuse_bounces, c.glossy_bounces = 16, 12, 12
    c.transmission_bounces, c.volume_bounces, c.transparent_max_bounces = 16, 4, 32
    c.caustics_reflective = c.caustics_refractive = True
    c.blur_glossy = 0.5
    c.sample_clamp_indirect = 10.0
    c.use_light_tree = True
    sss = disp = 0
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        for n in m.node_tree.nodes:
            if n.type == "BSDF_PRINCIPLED" and n.inputs["Subsurface Weight"].default_value > 0:
                n.subsurface_method = "RANDOM_WALK_SKIN"
                sss += 1
        out = m.node_tree.get_output_node("CYCLES")
        if out and out.inputs["Displacement"].is_linked:
            m.displacement_method = "BOTH"
            disp += 1
    if os.environ.get("NTS_HERO_DICING") == "1":
        c.feature_set = "EXPERIMENTAL"
        for o in scene.objects:
            if o.type == "MESH" and any(md.type == "SUBSURF" for md in o.modifiers):
                o.cycles.use_adaptive_subdivision = True
    print(f"render_hero: {c.samples} spp, {sss} SSS materials -> random walk skin, {disp} with displacement")
