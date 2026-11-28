# Integration: the buffers a temporal upscaler takes, and where they come from here

A real-time engine hands its temporal upscaler a set of buffers every frame. An offline renderer can produce the same set, exactly
and without the estimates a game has to make. This page maps one onto the other and lists what each part of this repo
reads. Sign conventions are fixed in one place, [`nts/conventions.py`](../nts/conventions.py), and re-checked on
renders with `baselines/accumulate.py` (`--sample-sign`, `--mv-sign`: with a wrong sign its output falls apart).

## Inputs, one by one

| Input | Here | Notes |
|---|---|---|
| Colour, render resolution, jittered | `f_####.exr` layer **Image**, scene-linear (after glare and haze) | Multilayer half-float EXR written by `datagen/blender/render_lr.py`. NR-1 itself works on display-referred frames (`f_####.png`); NTS-F and NTS-G read the EXR or the PNG |
| Depth | layer **Depth.V** | Linear camera distance in scene units, not a hardware z-buffer. Two surfaces count as the same within 5 % (`nts.warp.disocclusion`, `nts.framegen.DEPTH_REL`) |
| Motion vectors | layer **Vector.XY** (to the previous frame), **Vector.ZW** (to the next) | Render pixels, +y up. Image space: `mv = (X, -Y)`, `prev_pos = cur_pos + mv` (`conventions.mv_image`); next-frame motion is `conventions.mv_next_image`. The next-frame half is what NTS-G uses; a real-time engine has no such buffer |
| Jitter offset | `jitter.json` | Halton(2, 3), applied as a camera lens shift per frame, in render pixels, +y up. Phase count is 8 × (output / render)², the usual guideline for temporal upscalers: 8 (native), 18 (Quality), 24 (Balanced), 32 (Performance), 72 (Ultra Performance). Real-time upscalers usually take the negated offset (the one that undoes the jitter); see the docstring of `nts/conventions.py` |
| Reset (history invalid) | `cuts.json` | Frames that start a camera cut, from Blender's camera markers. `conventions.resets` adds the first frame and any gap in the frame numbers |
| Exposure / pre-exposure | `meta.json` `"view"` | View transform, look and exposure of the render. NTS-F accumulates scene-linear colour, so exposure needs no special handling; `nts/tonemap.py` applies the view transform through OpenColorIO |
| Output size, quality mode | `meta.json` `"render"`, `"out"`; render mode | See below |
| Region mask (offline only) | layer **Region.RGB** | AOV mask of the main character: R face, G hands, B hair. NR-1's tone and carry steps and the evaluation need it |
| Albedo, normal (offline only) | layers **Albedo.RGB**, **Normal.XYZ** | Cycles' denoising data passes. Used by the guided baselines |

## Quality modes

The usual ladder of scale factors of temporal upscalers. Each is a render mode of `render_lr.py` (and a key of
`nts.conventions.QUALITY_MODES`) for the 1080×1920 output:

| mode | render | scale per axis | pixels rendered |
|---|---|---|---|
| `native_aa` | 1080×1920, jittered | 1.0 | 100 % |
| `quality` | 720×1280 | 1.5 | 44 % |
| `balanced` | 630×1120 | 1.71 | 34 % |
| `performance` (also `perf`) | 540×960 | 2.0 | 25 % |
| `ultra_performance` (also `uperf`) | 360×640 | 3.0 | 11 % |

Balanced is 630×1120 rather than the customary 1/1.72, because the render size has to divide the output with the aspect
kept exactly (`conventions.scale_of` asserts it). Jittered modes render with the 0.5 px pixel filter; native
references (`native`, `hero`) are unjittered with the 1.5 px filter.

## Rendering the inputs

```python
# inside Blender, after the scene's own render setup
import sys; sys.path.insert(0, "<repo>/datagen/blender")
from render_lr import setup_nts
setup_nts(bpy.context.scene, "/renders/shot_a", mode="performance")   # then render the animation
```

Then, outside Blender:

| Step | Command | Reads |
|---|---|---|
| NTS-F upscale | `python -m nts.pipeline /renders/shot_a none out/up` | EXR Image, Depth, Vector.XY, jitter, cuts |
| NTS-G 2× frames | `python -m nts.framegen /renders/shot_a frames/ out/inbetween` | EXR Depth, Vector.XYZW, cuts; frames as `f_####.png` |
| NR-1 base and tone | `gen.steady2`, `nts.tone` (see the README) | EXR Depth, Vector, Region (or `m_####.png` masks), cuts; frames |
| Flicker check | `python -m eval.flicker_mv /renders/shot_a 1-120 region out=frames/` | EXR Depth, Vector, Region |

## Frame generation from a half-rate render

Cycles' Vector pass always points at the neighbouring scene frames (n−1 and n+1), whatever the render's frame step.
So a render with frame step 2 (frames 1, 3, 5, …) gives, for every missing frame n, the motion from n−1 forward and
from n+1 backward, which is exactly what `nts.framegen` needs. Check this once on your Blender version: render a short
clip at every frame and run `python -m nts.framegen <geom> <frames> --check`. It generates each frame from its two
neighbours and prints its PSNR against the real frame.
