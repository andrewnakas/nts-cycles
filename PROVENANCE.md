# Provenance

Every design choice here is traced to a public paper or to our own measurements. No NVIDIA or AMD code, weights or
binaries are used, and neither is the FSR 4 source AMD leaked in August 2025.

**DLSS.** During development, frames the author made with DLSS (4.5 and 5) on their own machine were used in two ways
only: as an evaluation yardstick beside our own results, and as the visual reference for the look described under
"The look" below. They were never a training input, a target or a loss term, and no model was given them. None of
those frames and no number measured on them is in this repo, and the repo does not link or vendor NVIDIA's NGX.
DLSS is a trademark of NVIDIA; this project is not affiliated with NVIDIA.

| Choice | Where it lives | Source |
|---|---|---|
| This frame's samples (since 0.3.0; the numpy accumulator and nts/rt): each output pixel finds its own position in the render's jittered sample grid and takes the 2x2 render pixels around it, weighted by a separable Gaussian of their distance in render pixels; in nts/rt's `fast` build the four taps are fetched as one bilinear lookup at their balance point | `nts/gather.py`, `nts/rt/shaders/resolve.wgsl` | Inverse-mapped (gather) resampling of a jittered render: Herzog et al., *Spatio-Temporal Upsampling on the GPU*, I3D 2010. Separable weights merged into one bilinear lookup: Yang et al., *Amortized Supersampling*, SIGGRAPH Asia 2009. Written from the papers |
| Halton(2,3) jitter, 8·ratio² phases | `nts/conventions.py`, `datagen/blender/render_lr.py` | Public FSR 2 docs (MIT) and the FSR 4 manual on GPUOpen; phases measured in our earlier render harness (our code, not in this repo) |
| MV dilation by nearest depth | `nts/warp.py` | Karis, *High Quality Temporal Supersampling*, SIGGRAPH 2014 course; FSR 2 (MIT) docs |
| Depth-test disocclusion | `nts/warp.py` | FSR 2 depth clip (MIT docs); NSRD "reliable warping" (CVPR 2024, MIT code, not copied) |
| R2P one-step realism generator (SD 1.5 + LCM-LoRA, residual latent, LoRA, projected DINOv2 discriminator) | `gen/r2p_train.py` | SD 1.5 (CreativeML OpenRAIL-M), LCM-LoRA (OpenRAIL++), DINOv2 (Apache-2.0); projected GAN, Sauer et al. 2021 (paper); one-step image translation, Parmar et al. 2024 (paper) |
| NR-1 teacher (2026-11-18 on) | `gen/`, `docs/DESIGN.md` | Each choice is listed with its source in `docs/DESIGN.md` |
| Keyframe structure teacher, pilot (2026-11-20 on): FLUX.2 [klein] 4B edits of our own keyframes (head crops and, from 2026-11-21, whole frames), used as released (inference only, no fine-tune, no LoRA, nothing of ours trained on its output); only the detail band of the edit is added to our frame, carried through the shot by surface identity (2026-11-26: each frame finds the keyframe's view of every surface point it shows in the renderer's surf pass) | `tools/kaggle/run_pilot.py`, `gen/structure.py`, `gen/carry.py`, `nts/surface.py` | FLUX.2 [klein] 4B, Black Forest Labs: Apache-2.0, open weights (`black-forest-labs/FLUX.2-klein-4B`; the 9B model is non-commercial and is not used). Surface-coordinate passes for re-texturing: compositing practice (UV and reference-position passes); the matching (`nts/surface.py`) is our own. Its inputs are our Cycles renders only; no third-party frame is ever given to it. Notes: `docs/DESIGN.md`, "Carry" |
| Frame generation (NTS-G, 2026-11-26). Surface mode: each pixel of the in-between frame looks its own surface point up in the neighbouring frames' surf passes. Vector mode: the neighbours' colours forward-warped along their own motion with bilinear splats, a depth order and a push-pull fill | `nts/framegen.py`, `nts/surface.py` | Forward image warping with a depth order: Mark, McMillan & Bishop, *Post-Rendering 3D Warping*, I3D 1997; McMillan, *An Image-Based Approach to Three-Dimensional Computer Graphics*, 1997. Push-pull fill: Gortler et al., *The Lumigraph*, SIGGRAPH 1996. Surface-coordinate passes for re-texturing: compositing practice (UV and reference-position passes). Written from these, with no optical flow and no network |
| Cumulative-sum box filters read by slices; a 256-entry sRGB decode table; one set of bilinear taps per motion field | `nts/tone.py`, `nts/warp.py`, `gen/steady2.py` (2026-11-24 speed pass) | Standard numerical practice, no outside source. Exactness against the replaced code: `tests/test_speed_equivalence.py` |
| Real-time GPU upscaler (nts/rt, 2026-11-27): NTS-F's accumulator as two WGSL passes per frame (render-size prep, output-size resolve) | `nts/rt/`, `nts/rt/shaders/` | Our own port of `baselines/accumulate.py`, checked against it in `tests/test_rt_upscaler.py`. Runs through wgpu (wgpu-py, BSD-2; wgpu-native, MIT/Apache-2.0). No upscaler's shader source was read or copied |
| Depth test in nts/rt (since 0.3.0): history kept where any of the previous frame's depths under the reprojected 2x2 footprint matches the pixel's depth within a relative tolerance | `nts/rt/shaders/resolve.wgsl`, `nts/warp.py` (`disocclusion(match=True)`) | Per-tap depth tests on reprojection: Nehab et al., *Accelerating Real-Time Shading with Reverse Reprojection Caching*, Graphics Hardware 2007 |
| History clipped to the per-channel min and max of the 3x3 render neighbourhood, taken at the render pixel the output pixel falls in, only ever widened | `nts/rt/shaders/prep.wgsl`, `baselines/accumulate.py` `minmax_box` | Karis, *High Quality Temporal Supersampling*, SIGGRAPH 2014 course |
| Catmull-Rom history in nts/rt: the 4x4 filter with the middle taps merged into bilinear lookups, corner taps left out and the rest renormalised (five lookups) | `nts/rt/upscaler.py` (`hist_cr`), `baselines/accumulate.py` `warp_catmull` | Catmull & Rom, 1974. Merging two positive taps into one bilinear lookup and leaving out the corners: Jimenez, *Filmic SMAA*, SIGGRAPH 2016 course. Written from the formulas |
| Bilinear frame as a fixed-weight term of the one weighted sum (history, samples, prior) | `nts/rt/shaders/resolve.wgsl`, `baselines/accumulate.py` (`prior`) | Our own |
| nts/rt guided modes: shading = colour / (albedo + eps) accumulated and multiplied back by an albedo image (display-size albedo from the engine, or in self-guided mode the render's own albedo accumulated by a second upscaler); history dropped where the display albedo changed | `nts/rt/upscaler.py` (`guided`, `SelfGuided`), `baselines/guided.py` (`GuidedAccumulator`), `nts/pipeline.py --self-guide` | Albedo demodulation: Schied et al., *Spatiotemporal Variance-Guided Filtering*, HPG 2017. The albedo-change test and the self-guided albedo history: our own, measured on our renders |
| nts/rt constants (footprint, weight cap, clip widening, weight shrink, prior weight, depth tolerance) | `nts/rt/fit.py`, `nts/rt/ab.py` | Plain coordinate search against our own native renders; no network, no gradients |
| Metrics | `eval/` | Ported from our earlier render harness (our code). FLIP is BSD-3 (NVlabs). LPIPS is BSD-2 (optional). |

## Helper weights and their training data (checked 2026-11-18)
Open-licensed code is not enough: a checkpoint trained on non-commercial data is kept out of the training loop.
- Out: ADE20K-trained segmenters (Mask2Former, OneFormer, MSeg), RAFT weights (FlyingThings3D, "any commercial use is
  prohibited"), face parsers trained on CelebAMask-HQ, Deep Video Prior (no licence). We use Blender's exact motion
  and our own Region AOV instead.
- Eval only: facenet (VGGFace2) for the identity metric.
- Open item: LPIPS (BSD-2 code; ImageNet-trained AlexNet/VGG backbone) is a training loss in R2P. Replacement ready to
  swap in if wanted: a DINOv2 (Apache-2.0, LVD-142M) feature distance.

## Data
Training data is our own Blender renders of our own episode sets and character, plus CC0 Poly Haven assets. Each
non-CC0 asset is listed with its licence and attribution before use (for fetched photos: the `manifest.json` that
`tools/fetch_scenes.py` writes next to them).

## Patents
This file traces each idea to its source. It is not a patent clearance: anyone shipping this code in a product should get their own review.

## The look (how the scene's appearance was chosen)
The user preferred a darker, relit skin look, first seen in the reference output named above (internal evaluation
only). We reproduce
that look **in Cycles itself**, as one global set of scene settings (lights, skin and hair materials, exposure).

How the reference is used (user's decision, 2026-11-06, replacing the earlier blurred-colour comparison):
1. **Brief.** `eval/look_brief.py measure` reads each reference once, next to our native render of the same frame,
   and writes `data/look/brief_v1.json`: a few hundred *relative* region statistics, aggregated over all frames by
   median. Examples: face-cell brightness ratios in face-normalised coordinates, tonal percentiles, chroma and hue
   shifts, and band-pass texture energy ratios. The brief holds no images or pixels, and the user reviews it and
   freezes it, like a colourist's written notes.
2. **Fit.** `nts/lookfit.py fit` solves the scene settings so that our own renders match *the brief applied to our
   own native renders*. The fit loop never opens a reference image.
3. **Final choice by eye.** `tools/ab_ladder.py` shows the user blind pairs of Cycles variants beside the reference,
   and the user picks. No numbers from the reference enter this stage.

From 2026-11-19 the same frozen brief also drives a second, non-learned step: `nts/tone.py` applies per-region
lightness curves, chroma scales and hue shifts to our finished frames so that their region statistics match *the
brief applied to the frame's own statistics* (`docs/DESIGN.md`, "Tone"). It reads `brief_v1.json` only,
never a reference image, and has no trained parameters. Its strength (`--tone`) and the eye/teeth constants are set
by eye on our own frames, as in step 3. How much of the brief's change a shot gets is a three-number fit
(`nts.tone fit-cond`, `data/look/brief_v1_cond.json`) of the brief's per-frame face ratios on statistics of our own
native renders of those frames; it too reads no reference image.

From 2026-11-22 four switches steady that step's skin and garment weights (`--skin-rim`, `--garment-hold`,
`--skin-close`, `--shot-hold`). They read our own frames, the renderer's masks, depth and motion vectors only; their constants (the
lightness from which a surface beside skin counts as rim light, its reach, and the vote's half) were set on our
own frames with a detector that compares our toned frames with our own steadied render. No reference image is read.

The hand-written rules added on 2026-11-19 (eye and teeth, evened skin colour, rim
light, the saturation limit, the strength chosen for A/B #1) were chosen the same way as step 3: by looking at our
own stills beside the reference, and from the written notes of a blind judge (`eval/prescreen.py`) who ranked
unlabeled stills of our candidates and the reference. Only those notes and our own judgement reached the rules; no
pixel or measured statistic of a reference frame is read by the code, apart from the frozen brief.

The reference is never a training input, target or loss term for any learned component. Every frame the model trains
on, input and ground truth alike, is our own Cycles render with the chosen settings.
