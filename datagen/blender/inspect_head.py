"""Dump what the hero head is made of (objects, vertex groups, bones, eye/skin/hair node trees) to a text file.
Debug aid for realism_pack: set NTS_RP_DUMP=<path> and render with any look that has a "realism" key; Blender quits
after the dump."""
import bpy
from mathutils import Vector


def _tree(nt, out, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if nt.name in seen:
        return
    seen.add(nt.name)
    pad = "  " * depth
    out.append(f"{pad}TREE {nt.name} ({len(nt.nodes)} nodes)")
    for n in nt.nodes:
        vals = []
        for i in n.inputs:
            if i.is_linked:
                l = i.links[0]
                vals.append(f"{i.name}<-{l.from_node.name}.{l.from_socket.name}")
            elif hasattr(i, "default_value"):
                v = i.default_value
                try:
                    v = tuple(round(x, 3) for x in v)
                except TypeError:
                    v = round(v, 3) if isinstance(v, float) else v
                vals.append(f"{i.name}={v}")
        extra = ""
        if n.type == "RGB":
            extra = f" rgb={tuple(round(x, 3) for x in n.outputs[0].default_value)}"
        if n.type == "TEX_IMAGE" and n.image:
            extra = f" image={n.image.name} {tuple(n.image.size)}"
        if n.type == "VALUE":
            extra = f" value={round(n.outputs[0].default_value, 4)}"
        out.append(f"{pad} {n.type} '{n.name}' label='{n.label}'{extra} :: " + "; ".join(vals)[:900])
        if n.type == "GROUP" and n.node_tree:
            _tree(n.node_tree, out, depth + 1, seen)


def dump(scene, path):
    out = []
    for o in scene.objects:
        if o.type == "ARMATURE":
            names = [b.name for b in o.data.bones]
            out.append(f"ARMATURE {o.name}: {len(names)} bones")
            out.append("  " + " ".join(names))
            for b in o.pose.bones:
                if any(k in b.name.lower() for k in ("head", "eye", "jaw", "neck", "ear", "nose", "lip", "cheek", "brow")):
                    h = o.matrix_world @ b.head
                    t = o.matrix_world @ b.tail
                    out.append(f"  bone {b.name} head={tuple(round(x, 3) for x in h)} tail={tuple(round(x, 3) for x in t)}")
    for o in scene.objects:
        if o.type not in ("MESH", "CURVES"):
            continue
        if not o.name.startswith("Human") and not any(k in o.name.lower() for k in ("eye", "hair", "poly", "brow", "lash")):
            continue
        bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
        lo = tuple(round(min(v[i] for v in bb), 3) for i in range(3))
        hi = tuple(round(max(v[i] for v in bb), 3) for i in range(3))
        nv = len(o.data.vertices) if o.type == "MESH" else len(o.data.points)
        out.append(f"OBJECT {o.name} type={o.type} verts={nv} parent={o.parent.name if o.parent else None} bbox={lo}..{hi}")
        out.append("  modifiers: " + ", ".join(f"{m.type}:{m.name}" for m in o.modifiers))
        if o.type == "MESH":
            out.append("  vgroups: " + " ".join(g.name for g in o.vertex_groups))
            out.append("  uv: " + " ".join(u.name for u in o.data.uv_layers) + " | attrs: " + " ".join(a.name for a in o.data.attributes))
            out.append("  shape keys: " + (" ".join(k.name for k in o.data.shape_keys.key_blocks)[:600] if o.data.shape_keys else "-"))
        for s in o.material_slots:
            if s.material:
                out.append(f"  MATERIAL {s.material.name} blend={getattr(s.material, 'blend_method', '')}")
                nm = o.name.lower()
                full = any(k in nm for k in ("eye", "poly", "hair", "brow", "lash")) or ".body" in nm or nm == "human"
                if s.material.use_nodes and full:
                    _tree(s.material.node_tree, out, 2)
    open(path, "w").write("\n".join(out))
    print("inspect_head: wrote", path, len(out), "lines")
