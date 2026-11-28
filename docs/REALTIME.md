# Real-time upscaling on the GPU: nts/rt (2026-11-27)

`nts/rt` runs NTS-F's temporal upscaler as a GPU pass a game could call once per frame: the same inputs a game gives
its upscaler (the jittered render's colour, depth and motion vectors, the jitter offset, a reset flag on cuts), out
comes the frame at display size. There is no neural network anywhere: not in the shader, not in the fitting of its
constants. The shaders are WGSL, run through wgpu: measured here on Metal and Vulkan (wgpu also targets D3D12,
not tried).

## How it works

Two compute passes per frame. State between frames: two history textures at display size (ping-ponged) and two small
textures at render size.

1. **prep** (one thread per render pixel, `shaders/prep.wgsl`). The motion of the nearest surface in the 3x3
   neighbourhood (motion dilation by depth, as `nts/warp.py`), the pixel's depth, and the per-channel minimum and
   maximum of the 3x3 colour neighbourhood. The 3x3 tile of each workgroup is read once into workgroup memory.
2. **resolve** (one thread per display pixel, `shaders/resolve.wgsl`).
   - This frame's jittered samples within a pixel of the display pixel, Gaussian-weighted (the splat of
     `nts/splat.py`, written as a gather: up to 2x2 samples, merged into one filtered lookup).
   - The previous output and its accumulated weight, fetched along the motion with Catmull-Rom filtering (five
     filtered lookups: the four corner taps of the 4x4 filter are left out and the rest renormalised).
   - A depth test against the previous frame's depth at the reprojected spot: another surface (or off screen) means
     no history.
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
  `GuidedAccumulator` to 145-149 dB.
- **Self-guided** (`SelfGuided`): only the render's own albedo, which a deferred renderer already has. A second
  upscaler accumulates the albedo (longer history, wider box: albedo carries no lighting noise) and its output, copied
  on the GPU, is the display albedo. No albedo-change test here, since that albedo is itself a history. GPU port of
  `nts/pipeline.py --self-guide` with the min/max box; matches the numpy composition to 138-141 dB. Costs about two
  plain upscales.

## Builds

| build | buffers | filtering | use |
|---|---|---|---|
| `compat` | float32 | numpy's own bilinear formula | proof: matches the numpy accumulator (`baselines/accumulate.py`) to 139-162 dB on synthetic and real frames (`tests/test_rt_upscaler.py`) |
| `fast` (default) | half float | the texture unit's | games: within 72-82 dB of `compat` |
| `fast_hist32` | half float, history float32 | the texture unit's | 5 % less static flicker than `fast` on our shots, for +0.10 to 0.26 ms at 1080p on an RTX 4050 laptop GPU (twice the history's memory traffic). Needs the device feature `float32-filterable` |

## Cost

GPU time of both passes per frame (`fast` build, the defaults), from GPU timestamps around each pass: 300 frames of
a real clip (our renders, resampled to each render size) replayed back to back with all inputs already on the GPU,
after 2 s of warm-up (an idle laptop GPU otherwise runs at a low clock and reads slow); checked against wall-clock
time per frame. Like the published FSR 2 table, the figures leave out the engine's own work around the call
(producing depth and motion vectors, resource transitions). `python -m nts.rt.bench` prints these tables.

Milliseconds per frame, minimum / median:

| output | GPU | Quality | Balanced | Performance | Ultra Performance |
|---|---|---|---|---|---|
| 1920x1080 | RTX 4050 laptop (Vulkan), another job rendering | 0.54 / 0.77 | 0.54 / 0.77 | 0.44 / 0.49 | 0.39 / 0.44 |
| 1920x1080 | GTX 1080 (Vulkan) | 0.51 / 0.51 | 0.48 / 0.48 | 0.44 / 0.45 | 0.40 / 0.40 |
| 1920x1080 | Apple M1, 8-core GPU (Metal) | 2.57 / 2.59 | 2.39 / 2.40 | 2.24 / 2.25 | 2.02 / 2.03 |
| 2560x1440 | RTX 4050 laptop, another job rendering | 1.04 / 1.51 | 1.00 / 3.33 | 0.78 / 1.52 | 0.83 / 1.08 |
| 2560x1440 | Apple M1 | 4.57 / 4.59 | 4.24 / 4.27 | 3.97 / 3.99 | 3.58 / 3.60 |
| 3840x2160 | RTX 4050 laptop, another job rendering | 2.42 / 5.53 | 1.98 / 2.34 | 2.39 / 5.36 | 1.78 / 3.97 |
| 3840x2160 | Apple M1 | 10.2 / 10.6 | 9.54 / 9.65 | 8.94 / 9.06 | 8.03 / 8.06 |

On the 4050 another job (a Blender render) had the GPU at 98 % throughout, so the medians there include waiting for
it; the minimum is the closer figure for a GPU that is ours. Not measured yet: the 4050 with no other job running,
and the GTX 1080 at 1440p and 4K (other jobs held all but 0.4 GB of its memory).

Mode costs: guided mode adds 0.19 ms at 1080p Performance on the 4050 (0.50 ms on the M1: two more display-size
reads and one more write), self-guided about doubles the cost (4.87 against 2.25 ms on the M1), `fast_hist32` adds
0.10-0.26 ms at 1080p on the 4050 and 0.10-0.11 ms on the M1.

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
| bilinear history, hand-set constants (prior weight 0.1) | 37.77 | 0.572 | 0.204 | 37.06 | 0.516 | 0.205 |
| Catmull-Rom history (5 lookups) | 38.62 | 0.566 | 0.203 | 37.74 | 0.509 | 0.205 |
| and prior weight 0.03 | 38.78 | 0.567 | 0.203 | 38.18 | 0.506 | 0.200 |
| **fitted constants (the defaults)** | 38.49 | 0.548 | 0.167 | 38.10 | 0.461 | 0.173 |
| self-guided, fitted | 38.43 | 0.561 | 0.192 | 37.77 | 0.462 | 0.180 |
| self-guided, albedo box not widened | 38.95 | 0.551 | 0.200 | 37.86 | 0.473 | 0.184 |

The fit trades PSNR for steadiness: on the held-out shots -0.07 dB for 9 % less temporal error and 14 % less static
flicker. How much PSNR a step may give up is set by the weight 3 in J (below); other weights pick other points of the
same trade-off, all scored by the fit (fit shots, PSNR / tErr / flicker):

| point | constants (sigma, expand, wmax, shrink, prior, rel) | PSNR | tErr | flicker |
|---|---|---|---|---|
| hand-set start | 0.469, 1, 12, 4, 0.03, 0.05 | 38.78 | 0.567 | 0.203 |
| narrower box | 0.586, 1.15, 16.8, 11, 0.0117, 0.169 | 38.62 | 0.548 | 0.177 |
| **fitted (J, weight 3)** | 0.586, 1.32, 16.8, 11, 0.0117, 0.169 | 38.49 | 0.548 | 0.167 |
| wider footprint | 0.733, 1.32, 16.8, 11, 0.0117, 0.169 | 38.23 | 0.540 | 0.159 |

Self-guided mode is not better on these shots (lower held-out PSNR, more flicker) and stays an opt-in.

**Zero-day clip** (2x, frames 400-459, `eval/run_eval.py`):

| variant | PSNR | tErr | static flicker | sharp | face PSNR | post-cut PSNR |
|---|---|---|---|---|---|---|
| bicubic | 33.56 | 2.357 | 0.909 | 0.93 | 28.67 | 36.85 |
| nts/rt, hand-set constants | 36.25 | 0.632 | 0.321 | 0.86 | 31.35 | 39.49 |
| **nts/rt, fitted constants (defaults), `fast`** | 36.62 | 0.531 | 0.237 | 0.81 | 31.60 | 40.35 |
| nts/rt, fitted constants, `compat` (= the numpy accumulator, `baselines/accumulate.py`) | 36.61 | 0.530 | 0.226 | 0.81 | 31.59 | 40.34 |
| nts/rt self-guided, hand-set | 36.34 | 0.610 | 0.307 | 0.87 | 31.18 | 40.26 |
| nts/rt guided (display-size albedo), fitted | 40.50 | 0.393 | 0.206 | 1.00 | 33.21 | 42.73 |

The clip was never part of the fit. Here the fitted constants gain on every count except sharpness (sharp: 1 =
as sharp as native; the wider box and footprint make the frame steadier and a little softer). The `compat` run matches
the numpy accumulator with the same constants to 155-160 dB per frame (`nts.rt.run --compare`); its lower flicker
than `fast` is the half-float history (`fast_hist32` closes it). Guided mode, where the engine can afford a
display-size albedo pass, is 3.9 dB over the plain upscaler.

## Constants

`nts/rt/fit.py` searches the constants (footprint sigma, clip widening, weight cap, weight shrink, prior weight,
depth tolerance) one at a time against our native renders, scored through Blender's view transform: 8-bit PSNR,
temporal error (tErr) and static flicker. One number decides a step: J = PSNR - 3 (ln tErr + ln flicker), so 1 % more
temporal error or flicker costs as much as 0.03 dB. Three shots are held out and only scored at the end.
`nts/rt/ab.py` scores design switches the same way (an A/B row each). Kept: Catmull-Rom history (+0.8 dB on the fit
shots, +0.7 dB held out). Not kept: the clip box in YCoCg (+0.05 dB, more flicker), clipping toward the box's middle
instead of per channel (+0.07 dB, 8 % more flicker), noise before half-float storage (no change), a wider box on
pixels that did not move (factor 1.25: 7 % less flicker for -0.07 dB, before the fit widened the box everywhere).
The 5-lookup Catmull-Rom filter scores the same as the 9-lookup one (+0.03 dB) for a quarter of its extra cost.

The constants live in `nts/conventions.py` (`ACCUM`), shared by the GPU port and the numpy accumulator. They were
fitted at 2x only. The footprint sigma is scaled with the render's pixel spacing at other scales (sigma x scale / 2:
0.586 at 2x, 0.29 at x1, the hand-set value there); the others are used as they are. Quality at scales other than 2x
has not been measured.

## Run it

```bash
pip install -e ".[rt]"                       # wgpu; ".[rt-view]" adds the viewer's window
pytest tests/test_rt_upscaler.py             # skips without a GPU adapter
python -m nts.rt.run <lr_dir> <out_dir> --build fast --frames 400-459     # a render folder -> frames for eval/
python -m nts.rt.bench --src synthetic --warm-ms 2000   # cost per pass, every mode, 1080p / 1440p / 4K
python -m nts.rt.ab <pack_dir> ... --hold <pack_dir>,... --variants "base;history=bilinear"   # A/B rows
python -m nts.rt.fit <pack_dir> ... --hold <pack_dir>,... --log fit.jsonl --out fit.json     # refit the constants
python -m nts.rt.view <pack_dir>             # live, split screen against bilinear; --record out.mp4 for a video
```
