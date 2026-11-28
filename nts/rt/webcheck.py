"""The exported shaders in a browser's WebGPU, checked against wgpu: another compiler (Chrome: Tint, Safari: WebKit's,
Firefox: naga) and another host (nts/rt/web/host.js, written from interface.json alone) on the same frames.

Run: python -m nts.rt.webcheck <dir> [--build fast|compat] [--serve 8123]
Writes <dir>: the export (prep.wgsl, resolve.wgsl, interface.json), host.js and check.html, and the test vectors
(vectors.json, vectors.bin): a short synthetic sequence (a panning HDR pattern, a nearer block moving faster, a cut)
and the frames export.Host makes from it on this machine's wgpu. --serve serves <dir> on 127.0.0.1 only; open
http://127.0.0.1:<port>/check.html. The page prints, per frame, the share of identical values, the largest difference
and the PSNR against the wgpu frames (peak = the frame's largest value; 999 = identical).
"""
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from nts import conventions as cv  # noqa: E402
from nts.rt import export, gpu  # noqa: E402

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
OUT_HW, IN_HW, N, CUT = (72, 128), (36, 64), 12, 8  # display width a multiple of 32 (row pitch of the readback)


def scene(t, hw, out_hw, offset):
    """Frame t at render size hw: rgb (HDR), linear depth, motion in render px toward the previous frame."""
    (h, w), (H, W) = hw, out_hw
    sx, sy = W / w, H / h
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    X, Y = (xs + 0.5 + offset[0]) * sx, (ys + 0.5 + offset[1]) * sy
    shift = 40.0 if t >= CUT else 0.0  # the cut: another view
    bx, by = X - 0.6 * t + shift, Y - 0.2 * t
    fx, fy = X - (30 + 1.5 * t), Y - (30 + 0.5 * t)
    fg = (np.abs(fx) < 14) & (np.abs(fy) < 10)
    bg = np.stack([0.5 + 0.4 * np.sin(0.35 * bx + 0.1 * by), 0.4 + 0.3 * np.cos(0.3 * by),
                   0.3 + 0.25 * np.sin(0.25 * bx) * np.cos(0.4 * by)], -1)
    fgc = np.stack([2.0 + 0.8 * np.sin(0.6 * fx), 1.0 + 0.5 * np.cos(0.5 * fy), 3.0 + np.sin(0.3 * (fx + fy))], -1)
    rgb = np.where(fg[..., None], fgc, bg)
    depth = np.where(fg, 2.0, 5.0)
    v = np.where(fg[..., None], np.array([1.5, 0.5]), np.array([0.6, 0.2]))  # display px per frame
    mv = -v / np.array([sx, sy])
    return rgb.astype(np.float32), depth.astype(np.float32), mv.astype(np.float32)


def vectors(folder, dev, build):
    import wgpu
    (H, W), (h, w) = OUT_HW, IN_HW
    host = export.Host(folder, dev, OUT_HW, IN_HW)
    fmt = host.spec["resources"]["hist0"]["format"]
    px = {"rgba16float": 8, "rgba32float": 16}[fmt]
    U = wgpu.TextureUsage
    tex = {k: dev.create_texture(size=(w, h, 1), format=f, usage=U.TEXTURE_BINDING | U.COPY_DST)
           for k, f in (("color", "rgba16float"), ("depth", "r32float"), ("motion", "rg32float"))}
    jit = cv.jitter_seq(N, cv.jitter_phases(h))
    blob, frames = bytearray(), []
    for t in range(N):
        off = cv.sample_offset(*jit[t])
        rgb, depth, mv = scene(t, IN_HW, OUT_HW, off)
        color = np.concatenate([rgb, np.ones((h, w, 1), np.float32)], -1).astype(np.float16)
        rec = dict(index=t, offset=[float(off[0]), float(off[1])], reset=t in (0, CUT))
        for k, a in (("color", color), ("depth", depth), ("motion", mv)):
            a = np.ascontiguousarray(a)
            dev.queue.write_texture({"texture": tex[k]}, a, {"bytes_per_row": a.strides[0]}, (w, h, 1))
            rec[k] = len(blob)
            blob += a.tobytes()
        enc = dev.create_command_encoder()
        hist = host.frame(enc, tex["color"], tex["depth"], tex["motion"], off, rec["reset"])
        dev.queue.submit([enc.finish()])
        ref = dev.queue.read_texture({"texture": hist}, {"bytes_per_row": W * px}, (W, H, 1))
        rec["ref"] = len(blob)
        blob += bytes(ref)
        frames.append(rec)
    with open(os.path.join(folder, "vectors.bin"), "wb") as f:
        f.write(blob)
    json.dump(dict(build=build, hist_format=fmt, out_hw=OUT_HW, in_hw=IN_HW, depth=[0, 0.0, 0.0], mv_scale=[1.0, 1.0],
                   wgpu_adapter=dev.adapter.info.get("device", ""), frames=frames),
              open(os.path.join(folder, "vectors.json"), "w"), indent=1)


def main():
    from baselines.accumulate import arg
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        sys.exit(__doc__)
    folder, build = sys.argv[1], arg("--build", "fast")
    os.makedirs(folder, exist_ok=True)
    export.write(folder, build, arg("--history", "catmull5"))
    for f in ("host.js", "check.html"):
        shutil.copy(os.path.join(WEB, f), folder)
    dev, name = gpu.device(arg("--adapter"))
    vectors(folder, dev, build)
    print(f"webcheck: {build} build, {N} frames {IN_HW[1]}x{IN_HW[0]} -> {OUT_HW[1]}x{OUT_HW[0]} made on {name} -> {folder}")
    if arg("--serve"):
        import functools
        import http.server
        port = int(arg("--serve"))
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=folder)
        print(f"open http://127.0.0.1:{port}/check.html", flush=True)
        http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    main()
