# Real-time upscaling on the GPU: nts/rt (2026-11-28)

`nts/rt` runs NTS-F's temporal upscaler as a GPU pass a game could call once per frame: the same inputs a game gives
its upscaler (the jittered render's colour, depth and motion vectors, the jitter offset, a reset flag on cuts), out
comes the frame at display size. There is no neural network anywhere: not in the shader, not in the fitting of its
constants. The shaders are WGSL, run through wgpu: measured here on Metal and Vulkan, and tested on every change
on Mesa's software Vulkan driver (lavapipe) in CI (wgpu also targets D3D12, not tried). [`ENGINE.md`](ENGINE.md) is
the contract for putting it in an engine: inputs and their conventions, resources, the steps of a frame.

## How it works

Two compute passes per frame. State between frames: two history textures at display size (ping-ponged) and two small
textures at render size.

1. **prep** (one thread per render pixel, `shaders/prep.wgsl`). The motion of the nearest surface in the 3x3
   neighbourhood (motion dilation by depth, as `nts/warp.py`), the pixel's depth, and the per-channel minimum and
   maximum of the 3x3 colour neighbourhood. The 3x3 tile of each workgroup is read once into workgroup memory.
2. **resolve** (one thread per display pixel, `shaders/resolve.wgsl`).
   - This frame's samples: the display pixel's position in the jittered render, and the 2x2 render pixels around
     it, weighted by a Gaussian of their distance from it in render pixels (`nts/gather.py`; one filtered
     lookup in the `fast` build).
   - The previous output and its accumulated weight, fetched along the motion with Catmull-Rom filtering (five
     filtered lookups: the four corner taps of the 4x4 filter are left out and the rest renormalised).
   - A depth test against the previous frame's depths under the reprojected spot: if none of them matches the
     pixel's depth (within a relative tolerance), or the spot is off screen, another surface is visible there and the
     history is dropped.
   - The history clipped to the min/max box of the render pixel the display pixel falls in, widened about its middle
     by a factor of at least 1 (never narrowed). History that had to move a lot to fit loses weight.
   - One weighted sum: history x its weight + samples x their weights + the bilinear upscale of this frame x a small
     fixed weight (where history and samples are thin, as after a cut or in newly visible areas, the bilinear frame
     fills in). The weight stored for next frame is capped.

## Guided modes (opt-in)

Both carry lighting, not texture, through time: shading = colour / (albedo + eps) is accumulated and the frame is
shading x (albedo + eps), so texture detail comes from an albedo image rather than from the history. They differ in
where that albedo comes from, which is what the engine has to supply.

- **Guided** (`RTUpscaler(guided=True)`): the engine also gives the albedo at display size (a G-buffer pass at full
  resolution). History is also dropped where the display albedo, warped from the last frame, differs from this
  frame's (another surface: an eyelid closed, a hand passed). GPU port of `baselines/guided.py`; matches its
  `GuidedAccumulator` to 134-152 dB.
- **Self-guided** (`SelfGuided`): only the render's own albedo, which a deferred renderer already has. A second
  upscaler accumulates the albedo (longer history, wider box: albedo carries no lighting noise) and its output, copied
  on the GPU, is the display albedo. No albedo-change test here, since that albedo is itself a history. GPU port of
  `nts/pipeline.py --self-guide` with the min/max box; matches the numpy composition to 128-147 dB. Costs about two
  plain upscales.

## In an engine

`RTUpscaler.dispatch()` records both passes into the engine's command encoder, reading its own colour target, depth
buffer and motion target: no copies. It takes the conventions engines use as they are: a reversed-Z or standard
depth buffer (infinite far plane included) or linear depth, motion in render pixels, UV or NDC in either direction,
and the jitter as a projection offset. The render size may change from frame to frame (dynamic resolution), and need
not divide the display size. NaN or infinite inputs are dropped before they can reach the history.
`python -m nts.rt.export` writes the shaders and an `interface.json` for renderers that are not wgpu.

**Compilers and drivers it has run on**, with the same frames out (`python -m nts.rt.webcheck`: 12 frames of an HDR
test sequence with motion, depth edges and a cut, run by a second host written from `interface.json` alone and
compared with wgpu's frames on the same machine):

| WGSL compiler, driver | `fast` build | `compat` build (float32) |
|---|---|---|
| wgpu (naga), Metal, Apple M1 | the reference | the reference |
| Chrome 158 WebGPU (Tint), Metal, Apple M1 | bit for bit | within 2.4e-7 (171 dB) |
| Safari 27.0.1 WebGPU (WebKit), Metal, Apple M1 | bit for bit | within 4.8e-7 (170 dB) |
| wgpu (naga), Vulkan: RTX 4050 laptop, GTX 1080, Mesa lavapipe (CPU) | `tests/test_rt_upscaler.py` passes | passes |

The `compat` differences are float32 rounding (each compiler fuses multiply-adds its own way). D3D12 / HLSL has not
been run.

`nts/rt/scene.py` is a small raster scene drawn with wgpu the way a game draws (reversed-Z infinite depth, UV motion,
projection jitter, mip bias), with a 32-sample native reference: the test content for game-like (noise-free) frames,
and the worked example of the contract. `python -m nts.rt.view --scene strafe` draws it live and upscales it.

## Builds

| build | buffers | filtering | use |
|---|---|---|---|
| `compat` | float32 | numpy's own bilinear formula | proof: matches the numpy accumulator (`baselines/accumulate.py`, with this frame's samples gathered as the GPU does) to 130-161 dB on synthetic and real frames (`tests/test_rt_upscaler.py`, `nts.rt.run --compare`) |
| `fast` (default) | half float | the texture unit's | games: within 72-77 dB of `compat` |
| `fast_hist32` | half float, history float32 | the texture unit's | 5 % less static flicker than `fast` on our shots, for +0.19 to 0.35 ms at 1080p on an RTX 4050 laptop GPU (twice the history's memory traffic). Needs the device feature `float32-filterable` |

## Cost

GPU time of both passes per frame (`fast` build, the defaults), from GPU timestamps around each pass: 300 frames of
a real clip (our renders, resampled to each render size) replayed back to back with all inputs already on the GPU,
after 2 s of warm-up (an idle laptop GPU otherwise runs at a low clock and reads slow); checked against wall-clock
time per frame. Like the published FSR 2 table, the figures leave out the engine's own work around the call
(producing depth and motion vectors, resource transitions). `python -m nts.rt.bench` prints these tables.

Milliseconds per frame, minimum / median:

| output | GPU | Quality | Balanced | Performance | Ultra Performance |
|---|---|---|---|---|---|
| 1920x1080 | RTX 4050 laptop (Vulkan), another job rendering | 0.53 / 0.66 | 0.51 / 0.65 | 0.39 / 0.44 | 0.31 / 0.34 |
| 1920x1080 | GTX 1080 (Vulkan) | 0.46 / 0.46 | 0.41 / 0.41 | 0.37 / 0.37 | 0.31 / 0.32 |
| 1920x1080 | Apple M1, 8-core GPU (Metal) | 2.19 / 2.25 | 2.00 / 2.06 | 1.82 / 1.87 | 1.54 / 1.58 |
| 2560x1440 | RTX 4050 laptop, another job rendering | 0.92 / 1.82 | 0.88 / 2.26 | 0.72 / 0.91 | 0.55 / 0.73 |
| 2560x1440 | Apple M1 | 3.91 / 4.03 | 3.56 / 3.63 | 3.25 / 3.29 | 2.74 / 2.81 |
| 3840x2160 | RTX 4050 laptop, another job rendering | 2.64 / 3.32 | 1.92 / 4.11 | 1.76 / 3.78 | 1.56 / 3.73 |
| 3840x2160 | Apple M1 | 8.77 / 9.02 | 7.96 / 8.07 | 7.27 / 7.34 | 6.10 / 6.27 |

On the 4050 another job (a Blender render) had the GPU at 99 % throughout, so the medians there include waiting for
it; the minimum is the closer figure for a GPU that is ours. Not measured yet: the 4050 with no other job running,
and the GTX 1080 at 1440p and 4K (other jobs held all but 0.4 GB of its memory).

Mode costs (minimum per frame, both GPUs sharing with other work): guided mode adds 0.29 ms at 1080p Performance on
the 4050 and 0.61 ms on the M1 (two more display-size reads and one more write); self-guided costs 2.2 to 2.5 times
the plain upscaler (0.97 against 0.39 ms on the 4050, 4.12 against 1.84 ms on the M1); `fast_hist32` adds 0.19-0.35
ms at 1080p on the 4050 and 0.26-0.30 ms on the M1.

For scale, AMD's published FSR 2.2 costs at 1920x1080 output, Performance mode (DX12, rounded to 0.1 ms, FSR 2 README):
0.2 ms on an RX 7900 XTX, 0.5 ms on an RX 6650 XT and RX 6700 XT, 1.1 ms on an RX 590. Different GPUs and APIs, so a
scale, not a race.

## Quality

Scored against native Cycles renders of the same frames, through Blender's view transform (AgX, AgX - Punchy,
+0.35 EV): 8-bit PSNR (higher is better), temporal error tErr and static flicker (lower is better), all as
`eval/run_eval.py` defines them.

**Nine packed shots** (2x, frames 8-31 of each; six shots the constants were fitted on, three held out and never
searched on; `nts/rt/ab.py`, fast build):

| variant | fit PSNR | tErr | flicker | held-out PSNR | tErr | flicker |
|---|---|---|---|---|---|---|
| bilinear history, hand-set constants (prior weight 0.1) | 37.63 | 0.550 | 0.198 | 36.85 | 0.479 | 0.199 |
| Catmull-Rom history (5 lookups) | 38.54 | 0.543 | 0.197 | 37.63 | 0.470 | 0.199 |
| and prior weight 0.03 | 38.75 | 0.541 | 0.196 | 38.24 | 0.462 | 0.194 |
| **fitted constants (the defaults)** | 38.42 | 0.526 | 0.162 | 37.96 | 0.431 | 0.170 |
| fitted, 0.2.0's depth test (`depth_test="nearest"`) | 38.46 | 0.544 | 0.165 | 38.03 | 0.455 | 0.172 |
| self-guided, fitted | 38.43 | 0.540 | 0.187 | 37.74 | 0.435 | 0.178 |
| self-guided, albedo box not widened | 38.93 | 0.529 | 0.195 | 37.86 | 0.445 | 0.182 |

The fit trades PSNR for steadiness: on the held-out shots -0.28 dB for 7 % less temporal error and 12 % less static
flicker. The matching depth test does the same on a smaller scale here (-0.07 dB, 5 % less temporal error) and gains
outright on the raster scene (below). How much PSNR a step may give up is set by the weight 3 in J (below); other
weights pick other points of the same trade-off, all scored here (fit shots, PSNR / tErr / flicker):

| point | constants (foot, expand, wmax, shrink, prior, rel) | PSNR | tErr | flicker |
|---|---|---|---|---|
| hand-set start | 0.234, 1, 12, 4, 0.03, 0.05 | 38.75 | 0.541 | 0.196 |
| narrower box | 0.293, 1.15, 16.8, 11, 0.0117, 0.169 | 38.55 | 0.526 | 0.171 |
| **fitted (J, weight 3)** | 0.293, 1.32, 16.8, 11, 0.0117, 0.169 | 38.42 | 0.526 | 0.162 |
| wider footprint | 0.367, 1.32, 16.8, 11, 0.0117, 0.169 | 37.97 | 0.515 | 0.152 |

Self-guided mode is not better on these shots (lower held-out PSNR, more flicker) and stays an opt-in.

**Zero-day clip** (2x, frames 400-459, `eval/run_eval.py`):

| variant | PSNR | tErr | static flicker | sharp | face PSNR | post-cut PSNR |
|---|---|---|---|---|---|---|
| bicubic | 33.56 | 2.357 | 0.909 | 0.93 | 28.67 | 36.85 |
| nts/rt, hand-set constants | 36.80 | 0.540 | 0.268 | 0.81 | 31.55 | 40.27 |
| **nts/rt, fitted constants (defaults), `fast`** | 36.62 | 0.490 | 0.230 | 0.80 | 31.64 | 40.39 |
| nts/rt, fitted constants, `compat` (= the numpy accumulator, `baselines/accumulate.py`) | 36.61 | 0.489 | 0.218 | 0.80 | 31.62 | 40.38 |
| nts/rt self-guided, fitted | 36.85 | 0.502 | 0.259 | 0.84 | 31.36 | 40.55 |
| nts/rt guided (display-size albedo), fitted | 40.67 | 0.370 | 0.202 | 1.00 | 33.22 | 42.79 |

The clip was never part of the fit. Here the fitted constants trade 0.18 dB of PSNR for 9 % less temporal error and
14 % less static flicker, with faces and the frames after the cut a little better (sharp: 1 = as sharp as native).
Against 0.2.0's defaults (36.62 / 0.531 / 0.237) the PSNR is the same, with 8 % less temporal error. The `compat`
run matches the numpy accumulator with the same constants to 156-161 dB per frame (`nts.rt.run --compare`); its
lower flicker than `fast` is the half-float history (`fast_hist32` closes it). Guided mode, where the engine can
afford a display-size albedo pass, is 4.0 dB over the plain upscaler.

**Raster scene** (game-like, noise-free: `nts/rt/scene.py`, 1920x1080 output, seven camera shots of 40 frames,
frames 8-39 scored against a 32-sample native render, through the same view transform; `python -m nts.rt.scene
score`). Bilinear is the render stretched to display size. Mean of the seven shots per mode, PSNR / tErr:

| mode (render size) | nts/rt | bilinear | nts/rt, still camera (PSNR / tErr / flicker) | bilinear, still camera |
|---|---|---|---|---|
| Native AA (1920x1080, x1) | 33.34 / 2.51 | 28.75 / 4.11 | 39.06 / 0.35 / 0.50 | 28.97 / 3.56 / 3.91 |
| Quality (1280x720, x1.5) | 31.25 / 3.18 | 28.03 / 5.27 | 34.38 / 0.63 / 0.92 | 28.01 / 4.18 / 4.53 |
| Balanced (1120x630, x1.71) | 30.34 / 3.51 | 27.04 / 5.97 | 32.77 / 0.78 / 1.14 | 27.06 / 4.71 / 5.14 |
| Performance (960x540, x2) | 29.22 / 4.07 | 26.17 / 6.80 | 31.22 / 0.87 / 1.23 | 26.20 / 5.12 / 5.57 |
| Ultra Performance (640x360, x3) | 26.86 / 5.48 | 23.83 / 9.30 | 27.75 / 1.40 / 1.83 | 23.78 / 7.12 / 7.79 |

nts/rt is ahead in every one of the 35 shot x mode cells, by 1.9 to 10.1 dB, with less temporal error in every
cell; the gain is largest with a still camera (4.0 to 10.1 dB), where the history holds longest. These rows use the
constants fitted on the Cycles shots; the static flicker metric only means something with a still camera. Against
0.2.0's depth test (the nearest of the previous depths only, which drops the history along every silhouette), the
matching test gains 0.12 to 0.38 dB per mode here, and most on the still shot. A known weak spot, seen in crops of the
still shot at x2: a zone plate (a pattern finer than the render's pixels) keeps faint alias rings, since the clip box
is drawn from the aliased render.

## Known limits

- Detail finer than the render's pixels (a zone plate in the still shot) keeps faint alias rings: the clip box is
  drawn from the aliased render.
- On the path-traced shots the fitted constants give up 0.18 to 0.28 dB of PSNR against the hand-set ones for 7 to
  14 % less temporal error and flicker (the weight 3 in J, below, sets that trade).
- Ultra Performance (x3) is the softest mode: 26.9 dB mean on the raster scene against 29.2 dB at Performance.
- No per-pixel mask for effects without depth or motion (particles, glass): draw them after upscaling.
- No sharpening is applied.
- Not run yet: D3D12 / HLSL, AMD and Intel GPUs, the 4050 with no other job on it, the GTX 1080 above 1080p.

## Constants

`nts/rt/fit.py` searches the constants (footprint, clip widening, weight cap, weight shrink, prior weight, depth
tolerance) one at a time against our native renders, scored through Blender's view transform: 8-bit PSNR,
temporal error (tErr) and static flicker. One number decides a step: J = PSNR - 3 (ln tErr + ln flicker), so 1 % more
temporal error or flicker costs as much as 0.03 dB. Three shots are held out and only scored at the end.
`nts/rt/ab.py` scores design switches the same way (an A/B row each). Kept: Catmull-Rom history (+0.8 dB on the fit
shots, +0.7 dB held out). Not kept: the clip box in YCoCg (+0.05 dB, more flicker), clipping toward the box's middle
instead of per channel (+0.07 dB, 8 % more flicker), noise before half-float storage (no change), a wider box on
pixels that did not move (factor 1.25: 7 % less flicker for -0.07 dB, before the fit widened the box everywhere).
The 5-lookup Catmull-Rom filter scores the same as the 9-lookup one (+0.03 dB) for a quarter of its extra cost.

The constants are `nts.rt.upscaler.FITTED`: those of `nts/conventions.py` (`ACCUM`, shared with the numpy
accumulator), with the footprint in render pixels (0.293), so that it keeps its size relative to the render at every
scale. They were fitted at 2x only, and refitted on the same shots after the two changes of 0.3.0 (this frame's
samples gathered from the render side, the matching depth test): no better point was found. The raster scene scores
every scale (above): quality falls off evenly from x1 to x3, with no scale where the constants break down. A fit on
the raster scene at Performance (foot 0.1875, a box widened 2.27x, shrink 20) gained 0.3 dB at Performance and 0.5 dB
at Ultra Performance there but lost 0.9 dB at Native AA, so one set of constants serves every mode and both kinds of
content.

## Run it

```bash
pip install -e ".[rt]"                       # wgpu; ".[rt-view]" adds the viewer's window
pytest tests/test_rt_upscaler.py             # skips without a GPU adapter
python -m nts.rt.run <lr_dir> <out_dir> --build fast --frames 400-459     # a render folder -> frames for eval/
python -m nts.rt.bench --src synthetic --warm-ms 2000   # cost per pass, every mode, 1080p / 1440p / 4K
python -m nts.rt.ab <pack_dir> ... --hold <pack_dir>,... --variants "base;history=bilinear"   # A/B rows
python -m nts.rt.fit <pack_dir> ... --hold <pack_dir>,... --log fit.jsonl --out fit.json     # refit the constants
python -m nts.rt.view <pack_dir>             # live, split screen against bilinear; --record out.mp4 for a video
python -m nts.rt.view --scene orbit --mode quality      # the raster scene drawn and upscaled live
python -m nts.rt.scene score --out 1080x1920 --json scene.jsonl   # every shot x mode against the 32-sample native
python -m nts.rt.export shaders/                         # WGSL + interface.json for another renderer
NTS_RT_ADAPTER=llvmpipe NTS_RT_REQUIRE=1 pytest tests/test_rt_upscaler.py   # on Mesa lavapipe, as CI does
```
