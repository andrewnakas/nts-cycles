"""nts/rt for an engine that is not Python: the final shaders of a build and everything needed to run them.

  prep.wgsl, resolve.wgsl   the two passes, filled in exactly as RTUpscaler compiles them. WGSL runs as it is on
                            WebGPU (Dawn, wgpu-native, browsers); a translator (naga, Tint) turns it into SPIR-V, MSL
                            or HLSL for an engine's own renderer.
  interface.json            resources (format, size, usage), each pass's bindings, its dispatch grid, the uniform's
                            fields with the formula of every value, the constants, and the per-frame steps.

Host is a minimal runner built from interface.json alone (no RTUpscaler code): it shows what an engine does each frame,
and tests/test_rt_upscaler.py checks it gives RTUpscaler's frames, so the file is known to be complete.

Run: python -m nts.rt.export <out_dir> [--build fast|compat] [--history catmull5|catmull|bilinear]
         [--depth-test match|nearest]
"""
import ast
import json
import math
import operator
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from nts.rt.upscaler import BUILDS, DEPTH_CLAMP, DEPTH_TEST, FITTED, PARAMS, sources  # noqa: E402

WG = ((16, 16), (32, 4))


def interface(build="fast", history="catmull5", depth_test=DEPTH_TEST, constants=None):
    """constants: values that replace the fitted ones in interface.json."""
    b = BUILDS[build]
    filt = "float" if b["hw"] else "unfilterable-float"
    res = {
        "params": dict(kind="uniform", bytes=PARAMS.size, note="written once per frame before both passes"),
        "color": dict(kind="input", size="render", formats=f"any float colour format ({b['color']} here)",
                      note="HDR scene colour, linear; filterable in the fast build"),
        "depth": dict(kind="input", size="render", formats="the depth buffer (depth32float) or any float format",
                      note="see params.flags.y / params.dep for its encoding"),
        "motion": dict(kind="input", size="render", formats="rg float formats (rg16float typical)",
                       note="scaled by params.acc.zw into render px from this frame to the previous one"),
        "aux0": dict(kind="texture", size="render", format=b["aux"], usage=["storage", "sampled"]),
        "aux1": dict(kind="texture", size="render", format=b["aux"], usage=["storage", "sampled"]),
        "lo": dict(kind="texture", size="render", format=b["stat"], usage=["storage", "sampled"]),
        "hi": dict(kind="texture", size="render", format=b["stat"], usage=["storage", "sampled"]),
        "hist0": dict(kind="texture", size="display", format=b["hist"], usage=["storage", "sampled", "copy_src"],
                      note="rgb = the frame, a = its accumulated weight"),
        "hist1": dict(kind="texture", size="display", format=b["hist"], usage=["storage", "sampled", "copy_src"]),
        "stub_sampled": dict(kind="texture", size="1x1", format=b["color"], usage=["sampled"],
                             note="guided-mode inputs, bound but never read in this mode"),
        "stub_storage": dict(kind="texture", size="1x1", format=b["color"], usage=["storage", "sampled"],
                             note="guided-mode output of pass A, never written in this mode"),
        "stub_out": dict(kind="texture", size="1x1", format=b["hist"], usage=["storage"],
                         note="guided-mode output of pass B, never written in this mode"),
        "near": dict(kind="sampler", filter="nearest", note="textureGather of the previous depth"),
    }
    if b["hw"]:
        res["linear"] = dict(kind="sampler", filter="linear", address="clamp-to-edge")
    tex = lambda i, r, st="unfilterable-float": dict(binding=i, resource=r, type="texture", sample_type=st)  # noqa
    sto = lambda i, r, f: dict(binding=i, resource=r, type="storage", access="write-only", format=f)  # noqa
    uni = dict(binding=0, resource="params", type="uniform")
    prep = [uni, tex(1, "color"), tex(2, "depth"), tex(3, "motion"), sto(4, "aux{cur}", b["aux"]),
            sto(5, "lo", b["stat"]), sto(6, "hi", b["stat"]), tex(7, "stub_sampled"),
            sto(8, "stub_storage", b["color"])]
    resolve = [uni, tex(1, "color", filt), tex(2, "aux{cur}"), tex(3, "aux{prev}"), tex(4, "lo"), tex(5, "hi"),
               tex(6, "hist{prev}", filt), sto(7, "hist{cur}", b["hist"]), tex(10, "stub_sampled"),
               tex(11, "stub_sampled", filt), sto(12, "stub_out", b["hist"]),
               dict(binding=9, resource="near", type="sampler", sampler="non-filtering")]
    if b["hw"]:
        resolve.append(dict(binding=8, resource="linear", type="sampler", sampler="filtering"))
    params = [
        ("sizes", "vec4<u32>", ["W", "H", "w", "h"], "display and render size"),
        ("jit", "vec4<f32>", ["ox", "oy", "W / w", "1 / (2 * foot * foot)"],
         "sample offset (render px: render pixel (i, j) samples (i + 0.5 + ox, j + 0.5 + oy)), x scale, footprint "
         "(foot, render px)"),
        ("acc", "vec4<f32>", ["expand", "wmax", "mv_x", "mv_y"], "box widening (>= 1), weight cap, motion factors"),
        ("lim", "vec4<f32>", ["w - 1.001", "h - 1.001", "W - 1.001", "H - 1.001"], "bilinear edge clip"),
        ("tun", "vec4<f32>", ["rel", "prior", "shrink", str(DEPTH_CLAMP)],
         "depth tolerance, bilinear-frame weight, weight shrink, depth clamp"),
        ("flt", "vec4<f32>", ["expand", "0", "0", "0"], "box widening where the pixel did not move (>= expand)"),
        ("flags", "vec4<u32>", ["reset", "depth_mode", "w_prev", "h_prev"],
         "reset 1 on the first frame and on cuts; depth mode 0 linear, 1 z = 1 / (A d + B), 2 z = A d + B; the "
         "previous frame's render size (= w, h unless the render size changes between frames)"),
        ("dep", "vec4<f32>", ["depth_a", "depth_b", "H / h", "0"], "depth encoding, y scale"),
        ("bits", "vec4<u32>", ["2139095040", "0", "0", "0"],
         "0x7f800000, the float exponent mask: the shaders' NaN / inf tests read it from here, so that a compiler "
         "that assumes floats are finite cannot fold them away"),
    ]
    off = 0
    fields = []
    for name, typ, vals, note in params:
        fields.append(dict(field=name, type=typ, offset=off, values=vals, note=note))
        off += 16
    assert off == PARAMS.size
    return dict(
        build=build, history=history, depth_test=depth_test, wgsl_entry="main",
        resources=res,
        passes=[dict(name="prep", shader="prep.wgsl", grid="render", workgroup=list(WG[0]), bindings=prep),
                dict(name="resolve", shader="resolve.wgsl", grid="display", workgroup=list(WG[1]), bindings=resolve)],
        params=dict(bytes=PARAMS.size, fields=fields),
        constants=dict(FITTED, **(constants or {})),
        depth_modes=dict(
            linear="depth_mode 0: linear view depth",
            reversed_perspective="depth_mode 1: A = 1/near - 1/far, B = 1/far (far = inf: A = 1/near, B = 0)",
            standard_perspective="depth_mode 1: A = 1/far - 1/near, B = 1/near",
            orthographic="depth_mode 2: A = far - near, B = near (reversed: A = near - far, B = far)"),
        motion=dict(px="mv_x, mv_y = 1, 1", uv="w, h", ndc="w / 2, -h / 2",
                    note="negate both when the vectors point from the previous frame to this one"),
        jitter=dict(projection="ox = -jx * w / 2, oy = jy * h / 2 when clip.xy += (jx, jy) * clip.w",
                    sequence="Halton(2, 3), max(8, round(8 * (display / render)^2)) phases; texture mip bias "
                             "log2(render / display)"),
        dynamic_resolution="allocate the render-size resources at the largest render size; each frame, render into "
                           "their top-left w x h and set sizes.zw = (w, h), flags.zw = the previous frame's (w, h), "
                           "jit.z = W / w, dep.z = H / h, lim.xy, and the motion factors for that size",
        frame=["cur = t % 2, prev = 1 - cur (t counts frames from 0)",
               "write params (reset = 1 when t == 0 or on a cut)",
               "dispatch prep over ceil(w / 16) x ceil(h / 16)",
               "dispatch resolve over ceil(W / 32) x ceil(H / 4)",
               "the frame is hist{cur}.rgb"])


def write(out, build="fast", history="catmull5", depth_test=DEPTH_TEST, constants=None):
    os.makedirs(out, exist_ok=True)
    sa, sb = sources(BUILDS[build], WG, history, depth_test=depth_test)
    for name, src in (("prep.wgsl", sa), ("resolve.wgsl", sb)):
        with open(os.path.join(out, name), "w") as fh:
            fh.write(src)
    with open(os.path.join(out, "interface.json"), "w") as fh:
        json.dump(interface(build, history, depth_test, constants), fh, indent=1)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def evaluate(expr, names):
    """The arithmetic of interface.json's params (names, numbers, + - * /, unary minus, brackets); nothing else."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            return names[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError(f"not allowed in a params expression: {expr!r}")
    return ev(ast.parse(expr, mode="eval"))


class Host:
    """Runs the exported passes from interface.json alone (wgpu-py standing in for an engine's renderer)."""

    def __init__(self, folder, dev, out_hw, in_hw, depth=(0, 0.0, 0.0), mv_scale=(1.0, 1.0), **constants):
        import struct

        import wgpu
        self.dev, self.wgpu = dev, wgpu
        self.spec = json.load(open(os.path.join(folder, "interface.json")))
        (self.H, self.W), (self.h, self.w) = out_hw, in_hw
        c = {k: v for k, v in self.spec["constants"].items() if not isinstance(v, str)}
        c.update(constants)
        self.names = dict(c, W=self.W, H=self.H, w=self.w, h=self.h, w_prev=self.w, h_prev=self.h,
                          mv_x=mv_scale[0], mv_y=mv_scale[1],
                          depth_mode=depth[0], depth_a=depth[1], depth_b=depth[2])
        self.struct = struct
        U = wgpu.TextureUsage
        usage = dict(storage=U.STORAGE_BINDING, sampled=U.TEXTURE_BINDING, copy_src=U.COPY_SRC)
        sizes = {"render": (self.w, self.h), "display": (self.W, self.H), "1x1": (1, 1)}
        self.res = {}
        for name, r in self.spec["resources"].items():
            if r["kind"] == "texture":
                u = 0
                for x in r["usage"]:
                    u |= usage[x]
                w, h = sizes[r["size"]]
                self.res[name] = dev.create_texture(size=(w, h, 1), format=r["format"], usage=u)
            elif r["kind"] == "sampler":
                f = r["filter"]
                self.res[name] = dev.create_sampler(mag_filter=f, min_filter=f)
            elif r["kind"] == "uniform":
                self.res[name] = dev.create_buffer(size=r["bytes"],
                                                   usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.passes = []
        for p in self.spec["passes"]:
            entries = []
            for bd in p["bindings"]:
                e = {"binding": bd["binding"], "visibility": wgpu.ShaderStage.COMPUTE}
                if bd["type"] == "uniform":
                    e["buffer"] = {"type": "uniform"}
                elif bd["type"] == "texture":
                    e["texture"] = {"sample_type": bd["sample_type"], "view_dimension": "2d"}
                elif bd["type"] == "storage":
                    e["storage_texture"] = {"access": bd["access"], "format": bd["format"], "view_dimension": "2d"}
                else:
                    e["sampler"] = {"type": bd["sampler"]}
                entries.append(e)
            layout = dev.create_bind_group_layout(entries=entries)
            src = open(os.path.join(folder, p["shader"])).read()
            pipe = dev.create_compute_pipeline(
                layout=dev.create_pipeline_layout(bind_group_layouts=[layout]),
                compute={"module": dev.create_shader_module(code=src), "entry_point": self.spec["wgsl_entry"]})
            self.passes.append((p, layout, pipe))
        self.t = 0

    def params(self, offset, reset):
        n = dict(self.names, ox=offset[0], oy=offset[1], reset=int(bool(reset)))
        blob = b""
        for f in self.spec["params"]["fields"]:
            vals = [evaluate(v, n) for v in f["values"]]
            blob += self.struct.pack("<4I" if f["type"] == "vec4<u32>" else "<4f",
                                     *([int(v) for v in vals] if f["type"] == "vec4<u32>" else vals))
        return blob

    def frame(self, enc, color, depth, motion, offset, reset=False):
        """One frame on the engine's textures; returns the texture holding it (rgb)."""
        cur = self.t % 2
        names = {"cur": cur, "prev": 1 - cur}
        inputs = {"color": color, "depth": depth, "motion": motion}
        self.dev.queue.write_buffer(self.res["params"], 0, self.params(offset, reset or self.t == 0))
        for p, layout, pipe in self.passes:
            entries = []
            for bd in p["bindings"]:
                r = bd["resource"].format(**names)
                obj = inputs.get(r, self.res.get(r))
                if bd["type"] == "uniform":
                    entries.append({"binding": bd["binding"], "resource": {"buffer": obj, "offset": 0,
                                                                           "size": self.spec["params"]["bytes"]}})
                elif bd["type"] == "sampler":
                    entries.append({"binding": bd["binding"], "resource": obj})
                else:
                    entries.append({"binding": bd["binding"], "resource": obj.create_view()})
            group = self.dev.create_bind_group(layout=layout, entries=entries)
            gx, gy = p["workgroup"]
            w, h = (self.w, self.h) if p["grid"] == "render" else (self.W, self.H)
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, group)
            cp.dispatch_workgroups(math.ceil(w / gx), math.ceil(h / gy))
            cp.end()
        self.t += 1
        return self.res[f"hist{cur}"]


def main():
    from baselines.accumulate import arg
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        sys.exit(__doc__)
    write(sys.argv[1], arg("--build", "fast"), arg("--history", "catmull5"), arg("--depth-test", DEPTH_TEST))
    print(f"export: prep.wgsl, resolve.wgsl, interface.json -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
