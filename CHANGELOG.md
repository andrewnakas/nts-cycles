# Changelog

## 0.3.0 (2026-11-28)

Since the first public release. All of it is in `nts/rt`, the real-time GPU upscaler, unless stated.

**Engine use**
- `RTUpscaler.dispatch()`: both passes recorded into the caller's command encoder, reading the engine's own colour,
  depth and motion textures (no copies); `RTUpscaler.output` is the frame. `docs/ENGINE.md` is the contract.
- Depth as the engine has it: `depth_device()` (reversed or standard Z, finite or infinite far plane, bound as the
  `depth32float` depth buffer itself), `depth_ortho()`, `depth_linear()`.
- Motion in render pixels, UV or NDC, either direction (`motion_scale()`); jitter as a projection offset
  (`offset_from_ndc()`).
- Render sizes that do not divide the display size (each axis scaled exactly), and dynamic resolution: the render
  size may change every frame (`max_in_hw=`, `dispatch(..., render_hw=)`).
- NaN and infinite inputs are dropped before they reach the history, in every build.
- `python -m nts.rt.export`: the WGSL of a build plus `interface.json` (resources, bindings, the uniform's layout and
  formulas, the steps of a frame) for renderers that are not wgpu; `export.Host` runs from it and gives the same
  frames bit for bit.

**Testing and quality**
- `nts/rt/scene.py`: a raster test scene drawn with wgpu as a game draws it (reversed-Z infinite depth, UV motion,
  projection jitter, mip bias) with a 32-sample native reference; seven camera shots; `score` (every shot and quality
  mode), `fit`, `still`.
- `python -m nts.rt.view --scene <shot>`: the scene drawn and upscaled live.
- GPU tests run in CI on Mesa's software Vulkan driver (lavapipe); `NTS_RT_ADAPTER` picks an adapter,
  `NTS_RT_REQUIRE=1` makes a missing adapter a failure instead of a skip.
- `nts/rt/web/host.js`: the exported passes run in a browser's WebGPU from `interface.json` alone;
  `python -m nts.rt.webcheck` checks a browser against wgpu (Chrome 158 and Safari 27: the `fast` build bit for bit).
- New tests: engine conventions, the depth buffer bound directly, broken inputs, uneven render sizes, the exported
  interface, dynamic resolution, no negative colour.

**Fixes**
- NaN / inf from the engine reached the frame on an RTX 4050 laptop GPU (NVIDIA Vulkan driver), in both builds: the
  driver folded the shaders' NaN / inf tests away (Vulkan lets a compiler assume floats are finite). The tests now
  take the exponent mask from the uniform (`params.bits`), which no compiler can fold; `test_broken_inputs_recover`
  passes on the RTX 4050, the GTX 1080, Apple M1 and lavapipe. The uniform is 144 bytes (was 128).
- Negative colour from non-negative input: the clip box, widened about its middle, could reach below zero where the
  render is dark next to bright, and Catmull-Rom history overshooting at that edge came out negative (-0.1 on a test
  sequence; NaN after a log tonemapper). The widened box now stops at zero unless the samples go below it; the numpy
  accumulator does the same.
- `nts.tonemap.Tonemapper` changed its input in place (OpenColorIO's `applyRGB` works in place): a reference frame
  tonemapped twice came out wrong. It now copies; it also tonemaps in 16 bands on threads (3.5x faster, same output).
- The footprint follows the render's pixel spacing at every scale (it is set in render pixels): at x1 the fitted
  2x value in display pixels was too wide.

**Changed**
- This frame's samples are gathered from the render side: each display pixel finds its own position in the jittered
  render and takes the 2x2 render pixels around it, weighted by a Gaussian of their distance in render pixels. The
  footprint is a render-pixel constant, `RTUpscaler(foot=)` (0.293, the fitted footprint at 2x): `sigma=` (display
  pixels) is gone, as are `SelfGuided(alb_sigma=)` (now `alb_foot=`, 0.25) and `nts.rt.run --sigma` (`--foot`). On
  the raster scene it scores the same as 0.2.0 from x1 to x2 and +0.28 dB at x3; on the Cycles shots -0.07 dB. It
  is also faster (four fixed taps, one filtered lookup in the `fast` build): a frame (both passes) takes 18 to 26 %
  less GPU time on Apple M1 at every mode and output size, and 1080p Performance costs 0.37 ms on a GTX 1080 (was 0.44).
- Depth test: history is kept where any of the previous frame's depths under the reprojected footprint matches the
  pixel's (`depth_test="match"`, the default). 0.2.0 compared only the nearest of them, which dropped the history
  along every silhouette, even with the camera still. Raster scene: +0.12 to +0.38 dB per quality mode, 4 to 6 %
  less temporal error; Cycles shots (held out): -0.07 dB, 5 % less temporal error. `depth_test="nearest"` is the
  0.2.0 test.
- A refit of the constants with both changes found no better point; they are unchanged
  (`nts.rt.upscaler.FITTED`, with the footprint in render pixels).
- The numpy accumulator (`baselines/accumulate.py`, NTS-F offline) takes this frame's samples the same way, so the
  GPU's `compat` build and the numpy code agree by default. `nts/splat.py` is now `nts/gather.py` (`gather`; the splat
  is removed). `Accumulator(foot=)` (render px) replaces `sigma=` (`sigma=`, and `--sigma` in `accumulate.py` and
  `guided.py`, still work as `foot = sigma / scale`, with a deprecation warning). `conventions.ACCUM` holds `foot`
  (0.293) in place of `sigma`; `conventions.accum_sigma` and `conventions.sample_positions_hr` are removed. The
  numpy accumulator keeps 0.2.0's depth test by default (`depth_test="nearest"`).
- `RTUpscaler(mv_sign=)` is now `mv_scale=` (a factor per axis; `mv_sign=` still works, with a deprecation warning);
  `params()` takes the sample offset, not the Blender jitter.
- New extra `rt-eval` (OpenColorIO, for scoring).
