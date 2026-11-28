# Putting nts/rt in an engine

nts/rt is two GPU passes per frame. It needs what a game already has at render size: colour, the depth buffer, motion
vectors, the jitter offset. It produces the frame at display size. This page is the contract: every input, its
conventions, the resources, and the steps of a frame. [`REALTIME.md`](REALTIME.md) has the method, the costs and the
measured quality; [`INTEGRATION.md`](INTEGRATION.md) is the offline side (which Blender passes give these inputs).

There are two ways in:

- **wgpu (Python, or any wgpu / WebGPU host):** `RTUpscaler.dispatch()` records both passes into the engine's own
  command encoder, reading its own textures. No copies, no CPU round trip.
- **Any other renderer:** `python -m nts.rt.export <dir>` writes the two shaders of a build (`prep.wgsl`,
  `resolve.wgsl`) and `interface.json`: every resource, every binding, the uniform's layout with the formula of each
  value, and the steps of a frame. WGSL runs as it is on WebGPU (Dawn, wgpu-native, browsers); a translator (naga,
  Tint) turns it into SPIR-V, MSL or HLSL. `export.Host` is a minimal runner built from `interface.json` alone, and
  `tests/test_rt_upscaler.py` checks that it gives `RTUpscaler`'s frames bit for bit.
- **A browser:** `nts/rt/web/host.js` is the same runner in JavaScript for WebGPU, again from `interface.json`
  alone. `python -m nts.rt.webcheck <dir> --serve 8123` writes the export, the host, a check page and a test
  sequence with the frames wgpu makes from it; the page runs the sequence in the browser and compares.

## Inputs, every frame

| Input | Size | Format | Convention |
|---|---|---|---|
| Colour | render | any float format (`rgba16float` typical; filterable in the `fast` build) | Linear HDR scene colour, before tonemapping, post-processing and UI. Pre-exposed is fine. |
| Depth | render | the depth buffer itself (`depth32float`) or any float format | See *Depth*. Bound as it is, no copy. |
| Motion | render | `rg16float` or `rg32float` | See *Motion*. Everything that moves needs it (skinned meshes, animated objects); the camera's own motion is part of it. |
| Sample offset | 2 floats | | See *Jitter*. |
| Reset | 1 flag | | Camera cuts and teleports. The first frame is a reset by itself. |

The render size may be any size, and the display / render ratio may differ slightly between the axes: a rounded size
such as 1707 x 960 for 2560 x 1440 at Quality is handled exactly. The display size is fixed per upscaler: make a new
one when it changes.

**Dynamic resolution** (the render size changing from frame to frame): create the upscaler with
`max_in_hw=(max_h, max_w)`, render each frame into the top-left of max-size targets, and pass that frame's size to
`dispatch(..., render_hw=(h, w))`, with `mv_scale=motion_scale(...)` for that size if your vectors are in UV or NDC.
History carries across size changes: in `test_dynamic_resolution`, switching between x2, x1.48 and x1.67 every six
frames scores between the fixed-size runs at the smallest and the largest size, and a constant size inside larger
targets gives the fixed-size frames bit for bit.

### Depth

The upscaler works in linear view depth. Tell it how your depth buffer encodes it (`nts/rt/upscaler.py`):

| Your depth buffer | Pass |
|---|---|
| Reversed-Z perspective, 1 at near, 0 at far (far may be infinite) | `depth=depth_device(near, far=math.inf, reversed=True)` |
| Standard perspective, 0 at near, 1 at far (D3D, Vulkan, Metal, and OpenGL's window depth in [0, 1]) | `depth=depth_device(near, far, reversed=False)` |
| Orthographic | `depth=depth_ortho(near, far, reversed=...)` |
| Linear view depth (distance along the view axis) | `depth=depth_linear()` (the default) |

Depth decides which surface's motion a pixel follows and when the surface seen at a pixel changed (a relative
tolerance, 16.9 % by default). NaN or infinite depth reads as background.

### Motion

`mv_scale=motion_scale(units, toward, render_hw)` turns your vectors into the upscaler's (render pixels, from this
frame's position to the previous one, x right, y down):

| Your vectors | `units` |
|---|---|
| Render pixels, y down | `"px"` |
| Texture coordinates (UV), y down, 1 = the whole image | `"uv"` |
| NDC, y up, 2 = the whole image | `"ndc"` |

`toward="previous"` when a vector points from where the surface is now to where it was (previous minus current);
`"current"` when it points the other way. Motion vectors must be computed from unjittered positions.

### Jitter

Each frame the projection is offset by a sub-pixel amount, and the upscaler is told where each render pixel's sample
lies: render pixel (i, j) samples the image point (i + 0.5 + ox, j + 0.5 + oy), in render pixels, x right, y down. If
you jitter by adding (jx, jy) to the projected x, y in NDC (`clip.xy += (jx, jy) * clip.w`), pass
`offset_from_ndc(jx, jy, render_hw)`.

The sequence used here is Halton(2, 3) offsets in [-0.5, 0.5) render pixels, with max(8, round(8 x (display /
render)^2)) phases before it repeats (`nts.conventions.jitter_seq`, `jitter_phases`). As is usual for temporal
upscaling, bias texture mip selection by log2(render / display) so that textures carry display-size detail.

The raster test scene (`nts/rt/scene.py`) feeds the upscaler the way a game does (reversed-Z depth with an infinite
far plane, UV motion, projection jitter) and is the worked example. Its signs were checked by scoring: the right
jitter beats a flipped axis by 2 to 3.5 dB, the right motion beats flipped motion by 4.6 dB.

## A frame, with wgpu

```python
from nts.rt.upscaler import RTUpscaler, depth_device, motion_scale, offset_from_ndc

up = RTUpscaler(display_h / render_h, (display_h, display_w), in_hw=(render_h, render_w), build="fast", dev=device,
                depth=depth_device(near), mv_scale=motion_scale("uv", "previous", (render_h, render_w)))

# every frame, after the main pass and before post-processing:
up.dispatch(encoder, color_tex, depth_tex, motion_tex, offset_from_ndc(jx, jy, (render_h, render_w)), reset=cut)
device.queue.submit([encoder.finish()])   # one submission per frame: the uniform is written by queue.write_buffer
frame = up.output                          # display size; rgb = the frame, a = its accumulated weight
```

Post-processing, tonemapping and UI then run at display size on `up.output`, which stays valid until the next
`dispatch()`.

## Resources

`fast` build (half floats), 1080p output from 540p (Performance): two history textures at display size (rgba16float,
2 x 16.6 MB) and four at render size (motion and depth twice, the clip box's minimum and maximum: 4 x 4.1 MB), about
50 MB in all. The `compat` build (float32, the numpy reference) takes twice that.

## Things to know

- **Transparent effects without depth or motion** (particles, glass) are best drawn after upscaling, at display size,
  as UI is. The upscaler has no per-pixel mask for them, by design.
- **Broken input values** (NaN, inf from a shader bug) are dropped: they never reach the history, and the frames
  match a clean run again two frames later (`test_broken_inputs_recover`). The NaN / inf tests read the exponent mask
  from the uniform (`params.bits.x = 0x7f800000`): a host that writes the uniform itself must set it, or a compiler
  that assumes floats are finite (an NVIDIA Vulkan driver did) can fold the tests away.
- **History** is reset on the first frame, on `reset=True`, and per pixel wherever depth says a different surface is
  now visible.
- **Guided mode** (`guided=True`, `dispatch(..., albedo=, albedo_hr=)`: albedo at render and at display size) is
  sharper and steadier on our path-traced clips (+3.9 dB), but needs a display-size albedo pass from the engine.
  `export.py` covers the plain mode only.
- **Negative colour** is never made up: the widened clip box stops at zero unless the samples go below it, so
  non-negative input gives non-negative output (`test_no_negative_colour`).
- **Not covered:** sharpening (none is applied). D3D12 / HLSL and AMD / Intel GPUs have not been run; see REALTIME.md
  for what has.
