# Design notes

The choices behind each public part of this repo: what was chosen, where it lives in the code, where the idea comes
from (a paper, textbook practice, or our own work) and the measurement or observation that decided it. Numbers are
our own, taken on our own renders and outputs. Licences, model weights and data are covered in
[`PROVENANCE.md`](../PROVENANCE.md); how to run each part is in the [README](../README.md).

## Steadied base

`gen/steady2.py` makes the frame every later step starts from: the Cycles render with its sampling noise averaged
along the renderer's own motion and its texture left as sharp as rendered. The frame is split in two bands. The
coarse band (three box passes, sigma about 1.4 px: shading, colour, noise) is accumulated along the motion. The fine
band (the rest: texture) is held on the surface and not averaged, because averaging removes it. Cuts and newly
visible pixels take the current frame.

Two measures recur on this page. Flicker is the mean change from one frame to the next along the renderer's exact
motion, in 8-bit levels, with newly visible pixels left out (`eval/flicker_mv.py`). The fine-detail ratio is the
energy of the output's finest band over the input render's (`eval/nr_board.py`), given as background / face.

| Choice | Where | Source | What decided it |
|---|---|---|---|
| Two bands: coarse accumulated with a 3×3 min/max clamp; fine held with a Catmull-Rom warp, refreshed slowly and rescaled to the current frame's local fine energy | `gen/steady2.py` `main`, `warp_cr` | Catmull-Rom history sampling: Karis 2014, *High Quality Temporal Supersampling* (talk); variance/histogram-preserving blending: Heitz & Neyret 2018 (paper) | Averaging the whole frame (`gen/steady.py`) cancels texture that is uncorrelated between frames: steadying a per-frame generative pass on at140 took the ratio from 1.21 / 1.09 to 0.96 / 0.67; two bands gave 1.10 / 0.95. |
| The held band's energy is matched per sub-band, the top 1 px band and the rest (`--bands 2`) | `gen/steady2.py` `split` | Heitz & Neyret 2018 (paper) | The bicubic warp loses the finest band fastest and one gain over the whole band left it short (four-clip mean 1.11 / 0.96); per sub-band the ratio is 1.21 / 1.10, against 1.21 / 1.12 before steadying. |
| The texture-energy target is accumulated along the motion (`--energy-alpha 0.2`); the fine band has its own, looser render-colour tolerance (`--fine-tol 40`) and a slow refresh (`--fine-alpha 0.08`) | `gen/steady2.py` `main` | ours | The render's own noise on faces failed the coarse tolerance (6 levels), so the held band was refreshed from uncorrelated texture far more often than set: face flicker on rs345 / zd400 fell from 1.26 / 1.13 to 1.03 / 0.86 with the ratio unchanged (1.16 / 1.07). |
| The energy match uses a smooth window of about 15 px (three box passes, sigma about 4.5 px) | `gen/steady2.py` `win` | ours | With one hard-edged 15 px box every strong pixel cast a square plateau in the energy map and the feedback grew a lattice of embossed dots on hair and skin: at140 frame 199, pixels with a fine band over 12 levels 1,516 (box), 991 (smooth), 898 in the frame itself; gain roughness 0.051 against 0.005. |
| Nothing is held within 12 px of a fast-moving outline (a depth step whose sides move 1 to 3 px a frame apart), and history taken from beside one is not held either (`--edge-reset 12`) | `gen/steady2.py` `moving_edges` | History rejection at disocclusions is standard temporal anti-aliasing practice (Karis 2014, talk; Yang, Liu & Salvi 2020, survey); the test is ours | Defocus spreads a moving hand's colour past its geometric outline, and clamped history left a fringe on the hand and contour lines on the wall: at140 frame 195, wall crop, mean / p99 difference to the render 0.79 / 9.0 (8 px, no history rule) against 0.46 / 4.3. |
| Tone and carry start from this base, not from the raw render | `tools/nr_clip.sh` `base` | ours | A per-shot tone curve does not steady render noise: face flicker of the render 1.68 / 1.61 (at140 / rs345), of the steadied base 0.77 / 0.74. |

## Tone

`nts/tone.py` changes low-frequency lightness and colour per region, with no network. The regions are face, hands
and hair from the renderer's Region mask, and the rest of the frame. Each region gets one set of constants per shot,
so the step cannot add flicker or change a shape, and `--tone 0` is pixel-exact. The target is the frozen look brief
(relative region statistics, no pixels; see [`PROVENANCE.md`](../PROVENANCE.md), "The look"), applied to the shot's
own statistics. `--tone 0.7`, `--share-power 0.5` and the eye and teeth constants were set by eye on our own frames.

Most of the code decides where a region's weight ends (uncovered skin, the anti-aliased outline, the garment) and
keeps those decisions from changing inside a shot. These rules read the frame, the masks, depth and motion only.

| Choice | Where | Source | What decided it |
|---|---|---|---|
| Per region a monotone CIELAB lightness curve through the region's own 5/25/50/75/95 percentiles, a chroma scale and a hue rotation; one set of constants per shot | `nts/tone.py` `shot_stats`, `shot_params`, `apply_frame` | Curve through percentiles: standard histogram specification (Gonzalez & Woods, textbook); target: the frozen look brief | Constant per shot, so it adds no flicker and cannot move geometry; hands take the face's chroma and hue change, because a separate hands value turned the hands red on rs345. |
| Shot-dependent strength: a share predicted from how far the face stands out of the room and how much of a close-up it is | `nts/tone.py` `fit_cond`, `shot_strength`, `--share-power` | ours: a three-number least-squares fit of the brief's per-frame entries on statistics of our own native renders | With one strength for every shot the close-up pty332 went muddy. |
| Skin the mask does not cover (neck, forearms) is grown from the face and hands at 1/4 size through pixels of the shot's own skin colour at the seed's depth, then gated by colour and depth at full size; the hands' growth only through pixels close to the skin's colour (`--hands-tight`) | `nts/tone.py` `weights`, `grow`, `skin_model` | Morphological reconstruction (Vincent 1993, paper) | Stills showed a bright neck under a darker face without it; ungated at full size the growth spilled 4 to 8 px onto a wall of about skin colour, and under typing hands it took 30,800 px of a wood desk against a hands mask of 25,700 px (tbs05; 600 px with `--hands-tight`). |
| An outline pixel takes the share of the skin's change that its own colour shows, with the nearest background colour within 4 px as the background | `nts/tone.py` `edge_alpha` | Smith & Blinn 1996, *Blue screen matting* | Depth has one sample per pixel, so a depth-gated weight ends on a staircase; with a local mean as background, pure wall pixels behind a jaw went from 163 to 122, with the nearest background colour to 167 (rs345 frame 350). |
| Eye whites, irises and teeth lose chroma and the bright ones lightness, in the interior of the face region only (`--eye`); skin saturation above 1.15 × the skin's median is soft-limited, weighted by hue | `nts/tone.py` `apply_frame` | ours | Lit skin and an eye white have the same colour, so position decides: applied to the whole face, the eye rule turned the lit strip of skin along pty332's hairline grey. Darker skin showed orange patches round the eyes; without the hue weight the lips went grey on the close-up. |
| Rim light on the outline of face and hair moves toward the region's own colour and lightness (`--rim-l`, `--rim-c`, `--fringe-skin`) | `nts/tone.py` `apply_frame` | ours | The render's rim stays bright when the head darkens: at140 frame 195, hair rim 180 against 120 inside and 95 on the wall before, 109 / 100 / 101 after. |
| Rest of the frame: a soft limit on saturation, solved per shot, emitters exempt (`--rest-tone 1.0`) | `nts/tone.py` `solve_cmax` | Chroma soft limit as in `nts/finish.py` (ours); target: the frozen look brief | A limit on chroma only reached bright prints; the dark teal backdrop of the close-up kept its full colour. |
| The garment is grown from chest-coloured pixels without crossing a lightness edge or a depth step, and takes a share of the hair's lightness change (`--garment-tone 0.8`) that is reached only at its lit parts (`--garment-curve 1.8`) | `nts/tone.py` `garment_weight`, `chest_box` | Seeded region growing (Adams & Bischof 1994) on our own depth and colour | With skin and hair darker, the lit shoulders stood above the wall (at140 frame 145: L 64 on the shoulder, 50 on the wall); one factor on the whole garment kept its highlights as far above its median as before (p90 / p50 1.79), which read as gloss. |
| Whole-shot votes: each surface point, followed through the shot along the renderer's motion, is garment or grown skin in every frame or in none (`--garment-hold`, `--shot-hold`); the growths keep off rim-lit skin and close thin gaps (`--skin-rim`, `--skin-close`) | `nts/tone.py` `garment_votes`, `weights` | ours | A per-frame yes/no flood switched: a neck patch on ti620 went between about 100 and 150 levels five times in 60 frames (largest step 51), and stays within 149 to 161 with `--skin-rim` (largest step 1.2). |

## Keyframe edit

One generative edit per shot supplies what the render lacks: the surface of skin and the strands of hair. The model
is FLUX.2 klein 4B (Apache-2.0, open weights), used as released: inference only, no fine-tune, no LoRA. It sees one
frame per shot (by default the middle one), so it has nothing to flicker against. `tools/nr_clip.sh keys` writes the
toned keyframe and its head box; `tools/kaggle/run_pilot.py` is the kernel script that edits the head crop. The
networks trained in the research phase are not part of this repo.

The edit is never used as a picture. It is photographic, but it redraws the person: pasted whole, plain edits scored
a face match (`eval/nr_identity.py`, an evaluation-only face-embedding cosine to the render) of 0.77 to 0.89 on four
key stills. Only its difference to the frame it was made from, limited in band and level, reaches the output.

| Choice | Where | Source | What decided it |
|---|---|---|---|
| FLUX.2 klein 4B as released, on a head crop edited at 768 px | `tools/kaggle/run_pilot.py` (`HEAD`), `tools/nr_clip.sh` `keys` | FLUX.2 klein model card (Apache-2.0, open weights) | A 16 GB T4 runs a 768 px edit; 1024 px edits and whole frames 608 px wide ran out of memory. |
| The edit starts from the toned keyframe itself with noise mixed in, `x = (1 - s) x0 + s noise` at s = 0.85, three steps, a text prompt and no reference image | `tools/kaggle/run_pilot.py` `edit`, `start_sigmas` | Meng et al. 2022, "SDEdit: guided image synthesis and editing with stochastic differential equations" (ICLR); diffusers' Flux2KleinPipeline (Apache-2.0) takes `latents` and `sigmas` | With the sharp frame as reference image the model returns its input (mean change on the face 1.9 to 5.4 levels at s = 0.30 to 0.75, at140); without one, what survives the noise (face shape, hair mass, colours) cannot be redrawn and the surface comes from the model. s was chosen by eye on our own frames; face match of the raw edit falls with it (at140: 0.93 at 0.45, 0.86 at 0.75, 0.74 at 0.85). |
| One short, fixed prompt; the no-reference edit gets a one-line description of the subject (prompt `n`) | `tools/kaggle/run_pilot.py` `PROMPTS` | ours | With a reference image, prompts that described the person replaced the face on all four clips (a closed mouth where the render's is open, a narrower jaw), and prompts without a description kept pose, mouth and face shape on all 24 keys; without a reference the noised keyframe fixes where everything is. |
| Prompts are encoded once, the text encoder is dropped and the transformer stays on the GPU | `tools/kaggle/run_pilot.py` `load` | diffusers `Flux2KleinPipeline.__call__(prompt_embeds=...)`, public source | With model offload an edit took about 105 s, most of it moving two models of about 8 GB on and off the card. |
| One key per shot | `gen/carry.py` `--keys` | ours | On rs345 a key carried 54 frames showed no loss, and a second, separately edited key made the face step 5.4 levels at the switch against 2.2 to 2.3 in the single-key builds; on a 158-frame shot the face's fine energy over the toned frame stayed between ×1.19 and ×1.30 at every distance from the key (both measured in 2026-11, before the surface carry). |

## Structure

`gen/structure.py` decides what of the edit reaches the frame. `key_residual` takes the edit minus the frame it was
made from and removes the low band (three box passes; the radius follows the head's size), so the edit's tone never
crosses over. `compose` adds the result to the toned frame under limits per region (features, plain skin, hair, the
rest). `gen.carry --compose` calls the same two functions for every frame, in float.

The file has more options than the driver uses: each was a test on stills, and those that lost stay in the code, off
by default (see "Tried and dropped"). The recipe in use is the option list in `tools/nr_clip.sh`.

| Choice | Where | Source | What decided it |
|---|---|---|---|
| Only the edit's detail band is used: the residual to the edited frame with its low band removed | `gen/structure.py` `residual`, `key_residual` (`--hp`) | ours; base/detail split as in Durand & Dorsey 2002 | The low and mid bands of the edit belong to another face (face match above); a residual to the edited frame is pixel-aligned with the render and carries no tone. |
| The band radius follows the head: head side / `--hp-rel`, narrowed above a 480 px head box, and the budget shrunk by the head's area below it (`--hp-ref 480`) | `gen/structure.py` `band_for`, `key_residual` | ours | A fixed 16 px band is 3.5 % of a 450 px head box and 8 % of a 206 px one, the size of eyes and mouth: on zd400's far shot face match was 0.87 at the full budget and 0.965 at 0.43 of it. |
| Limit by a smooth gain, not by clipping (`--agc 0.8`); each region's own low band is removed (`--split 1`) | `gen/structure.py` `compose`, `residual`, `low_in` | Automatic gain control is textbook signal processing; normalised convolution: Knutsson & Westin 1993 (paper) | The edit's hair detail is ±40 levels and a tanh limit at 12 squared it into two flat levels; one blur across hair, skin and background let the edit's darker hair ring into its neighbours as a halo round the head. |
| Features keep a thin budget (skin limit 6, mid band × 0.3); plain skin and hair take the wide band (`--skin-mid-plain`, `--skin-limit-plain`) | `gen/structure.py` `compose`, `plain` | ours | Face match moves with the skin's band, not with hair: with only hair opened it was 0.95 to 0.98 on five key stills, with only plain skin opened 0.75 to 0.78. How far plain skin is opened was chosen by eye on our own frames. |
| Plain skin is tested against the skin colour around the pixel (`--plain-local`), and its borders scale with the face (`--plain-soft`) | `gen/structure.py` `plain` | ours | Against one median over all skin, none of ti620's face counted as plain (0.54 to 0.75 on the other clips) and it kept 2.1 of the edit's 8.1 levels of fine detail; with the local colour 0.57 to 0.64 and 7.9, the other seven clips bit-identical. |
| No small dark specks on the lower face (`--masks`): a grey closing of the detail's lightness, used as a smoothed envelope | `gen/structure.py` `beard_zone`, `speck_free` | ours | The edits drew stubble dots on upper lip and chin that the character does not have; the raw closing left a mosaic of 5 px blocks, the envelope does not. Face match at140 / rs345 0.858 / 0.925 before, 0.875 / 0.926 with the rule. |
| Nothing is added on the hero's skin outline (`--outline 1.5`) | `gen/structure.py` `outline_fade` | ours | The outline's pixels are mixtures the renderer has already anti-aliased; three layers that each ended on a hard rule at the mask's 50 % contour summed to a stair-stepped jaw with a dark 1 px line outside it. |
| Outside the head the head edit's detail may only darken, within head side / 64 px; the hair's outer outline may not brighten (`--hair-rim`) | `gen/structure.py` `key_residual`, `compose` | ours | The edit's outline is crisper than the render's and lifted the wall in the 3 px beside the head by 0.4 to 4.3 levels (0.3 to 1.8 darker with the rule); tb600's hair border stood 45 levels over the toned frame at the key, 15 with the rule. |
| Where hair and face masks overlap (a fringe over the forehead) the hair's rules win (`--hair-over`) | `gen/structure.py` `compose` | ours | Fringe pixels kept the render's dark strand next to hair-only pixels that took the edit's lighter forehead: a dotted line along the hairline on four stills, gone with the option. |
| Optional, off in the driver: whole-frame detail from a second edit (`--full`), lightness only, off print and screen text, only where the edit kept the render's shapes (`--agree`); border rules at the hairline (`--plain-fill`, `--seam`, `--skin-agree`) | `gen/structure.py` `key_residual`, `agreement`, `text_zone`, `compose`, `scalp` | Guided-filter local affine model: He, Sun & Tang 2010 (paper); white top-hat: Serra 1982 (textbook morphology) | Whole-frame detail with colour and a 16 px band cut text PSNR after a colour fit to 21.9 to 23.5 dB on three clips (tone alone 26.7 to 29.0), lightness only with the text guard 26.5 to 29.4 dB. Border rules: on tbs05 the edit drew its ear about 10 px nearer the face, a light band of 21 levels on the render's temple (3 darker with the rules). |

## Carry

`gen/carry.py` puts the keyframe's detail on every other frame of the shot. Nothing is warped from frame to frame:
for every pixel of a frame, `nts/surface.py` finds the keyframe position that shows the same point of the same
surface, and the keyframe's residual is read there with one Catmull-Rom lookup, `R_f(x) = R_k(map_f(x))`. Every
frame is matched against the keyframe on its own, so the detail is resampled exactly once however far the frame is
from its key, nothing builds up along a shot, and frames run in parallel (`--workers`). An earlier version advected
coordinates along the renderer's motion vectors; the public carry uses no motion.

**The surf pass** is a second, cheap render of every frame (`datagen/blender/render_lr.py` mode `surf`). Every
sample goes through the pixel's centre, with no depth of field, motion blur, lights or bounces, and three AOVs record
the object's random number, the shaded point's UV coordinates and its Generated coordinates (its place on the
undeformed mesh, normalised to the mesh's bounding box). UV is a mesh attribute, so it moves with the surface under
any deformation and is the main coordinate. Generated coordinates tell apart UV islands that share coordinates
(mirrored halves) but are not exact under deformation, so they enter the match at a weight of 0.05 beside the UVs,
and alone only where an object has no UVs. Each AOV also stores a second moment (|g|², id², u² + v²): a pixel
whose samples saw more than one point (an outline, a strand, a UV seam) shows a spread and is left out.

**The alpha gate.** Cycles adds up an AOV over every surface a camera ray meets before one whose alpha reaches the
view layer's threshold (0.5), so through the clear part of an eyelash or eyebrow card a pixel got the card's values
plus the skin's and passed as one surface with coordinates that are neither. `alpha_gate` writes the AOVs only where
the material's alpha reaches that threshold: the pixel then shows the skin behind the card, as the depth pass does.

**The match** (`nts.surface.match`) works per object on the feature f = (u, v, 0.05 g). (1) A nearest-point search
in a hash grid of the keyframe's points (cells of two pixel footprints), on every 4th pixel. (2) Every pixel starts
from its block's result and is refined by Gauss-Newton on the keyframe's field, `u += (J^T J)^-1 J^T (f - f_k(u))`,
with slopes taken on the same object and never across a UV seam. (3) The check: the keyframe pixel at u must show
the same object, and f_k(u) must lie within half a pixel's footprint of f. A pixel that fails gets a full search of
its own and the same refinement; one that still fails gets no keyframe position. A point the keyframe does not show
has no keyframe point within reach, so occlusion needs no separate test.

**Seen again, and re-seated.** A matched point can look different on another frame (a shadow passes over it), and
mixed pixels have no match. `seen_again` compares the frame's lightly blurred render with the keyframe's at the
mapped position (`--photo 20`: full detail up to 20 levels apart, none from 30). A pixel that fails or has no match
is re-seated: the offset `map - x` is nearly constant over a small rigid patch, so it takes the normalised average of
the agreeing offsets around it (`reseat`, `--fill 2`: three box passes of radius 2), from neighbours at its own
depth only. That position is kept if it passes the same test and showed the same surface at the keyframe; if not,
the pixel gets no detail there.

**Evened offsets, short reach.** Every pixel is matched on its own, so the keyframe positions of neighbouring pixels
carry small independent errors, and texture read at them shimmers from frame to frame. Before the test, `settle`
replaces a matched pixel's offset `map - x` by the mean offset of the matched pixels in the 5 × 5 window around it
whose depth is within 5 % of its own (`--settle 2`). A surface's offset changes slowly across it, so the mean keeps
the offset and drops the per-pixel error; only depth decides which neighbours count. The pixels without a match are
thin: lash lines, the lip line, the outlines of jaw and ears, a band of skin coming into view. Re-seated from a
36-pixel neighbourhood (`--fill 12`, the earlier default) whose make-up changed with every frame, their positions
wobbled and stepped against their matched neighbours; most of the face flicker this carry had over the earlier
motion-vector version sat at those steps. The re-seating now reaches about 6 pixels.

**Head rules** (`follow_head`). `--follow`: the key's dark detail reaches a few pixels past the head's outline
(strands over the wall); a pixel within that reach takes the mean offset of the head pixels around it, and the key's
outline rule is applied again on each frame's own outline. `--stretch`: no detail where the map changes area by more
than about the given factor, as on a closing eyelid, whose pixels all point into the thin fold it was at the key.
The map steps where two surfaces that moved apart meet, so its slopes are taken from the smooth side of a step.

| Choice | Where | Source | What decided it |
|---|---|---|---|
| A surface-coordinate pass per frame (object number, UV, Generated) | `datagen/blender/render_lr.py` `setup_surf`, `paint_surface` | Re-texturing through a rendered surface-coordinate pass is long-standing compositing practice (UV and reference-position passes) | It names the same surface point on any two frames, with no motion and no dependence between frames. |
| UV first, Generated at weight 0.05 | `nts/surface.py` `Surf` (`WG`) | ours | On at140 the hands' Generated coordinates moved by several pixels' worth from one frame to the next; their UVs do not. |
| AOVs gated by the material's alpha | `datagen/blender/render_lr.py` `alpha_gate`, `alpha_source` | ours | On pty332n the whole eyelid under the lash card, the brows and the hairline under the hair cards carried summed values that passed as one surface. |
| Hash-grid search, Gauss-Newton refinement, check at half a footprint | `nts/surface.py` `match`, `Key.nearest`, `_refine` | ours: a plain spatial hash and plain least squares | Synthetic scenes in `tests/test_surface.py`; about 3 s per 1080×1920 frame on one core ([`SPEED.md`](SPEED.md)). |
| Next to an outline the keyframe's field is extended to first order from the pixel the point lies in, not interpolated across the outline | `nts/surface.py` `_lookup_obj` | ours | On hp622n 13 % of the face (the outlines of brows, lids, lips and hairline, the head's UV seams) had no match and was re-seated, with jitter from frame to frame. |
| Seen-again test and re-seating, gated by depth | `gen/carry.py` `agree`, `reseat`, `seen_again` (`--photo`, `--fill`) | History rejection by colour, as in temporal anti-aliasing: Yang, Liu & Salvi 2020 (survey); the re-seating is ours | On at140n (key 170) re-seating gives detail to 12 % of the wanted part that the match alone leaves without (share without detail 0.34 against 0.45) and lowers flicker (0.80 against 0.84); without the depth gate a wall uncovered by a moving hand took a mix of the hand's offsets and its own. |
| Matched offsets evened over 5 × 5 px at the pixel's own depth; re-seating reach about 6 px | `nts/surface.py` `settle`, `gen/carry.py` `carried_surf` (`--settle 2`, `--fill 2`) | ours | Face flicker on the 8 test clips 0.72 before, 0.70 after (the earlier motion-vector version: 0.70); on hp622n and pty332n 0.82 and 0.52 before, 0.79 and 0.50 after, with carried detail on slightly more of the face. Evening over 7 × 7 or 9 × 9 px gave about the same; a reach of 3 or 9 px was worse. |
| `--follow`, `--stretch`, slopes from the smooth side of a map step | `gen/carry.py` `follow_head`, `slopes` | ours | On hp622 frame 676 the head's strands stayed behind as a dotted line beside the ear and the lash line smeared over a closing lid. Central differences read a map step as a large stretch and dropped detail along it; slopes from the smooth side leave face flicker as it was on hp622n and pty332n and keep detail on a little more of the face (share without detail 0.280 and 0.130 against 0.286 and 0.133). |


## Frame generation (NTS-G)

`nts/framegen.py` makes the frame n between two finished frames A (n − 1) and B (n + 1), with no optical flow and no
network. In surface mode (`--surf`) the in-between frame's own surf pass is rendered, and each of its pixels looks
its surface point up in A and in B and reads the colour there: mixed by time if both show it, the one colour if one
does, and from the vector-mode frame (`--geom`) or a push-pull fill if neither does or the pixel is mixed. The
outlines are the in-between frame's own. Vector mode needs no extra render: Cycles' Vector pass gives each pixel of
A its motion to frame n and each pixel of B its motion from it; each colour is carried along its motion and shared
among the four pixels where it lands, nearest surface first. Lighting that changes between A and B is averaged.

| Choice | Where | Source | What decided it |
|---|---|---|---|
| Surface mode: every pixel of the new frame finds its own surface point in both neighbours | `nts/framegen.py` `between_surf`, `nts/surface.py` `match` | Surface-coordinate passes for re-texturing: compositing practice (UV and reference-position passes); the matching is ours | at140, 58 frames, mean PSNR against the real rendered frame: 44.21 dB, vector mode 43.61 dB, a plain blend of the neighbours 36.12 dB; ahead of vector mode on 51 of the 58 frames ([gallery](GALLERY.md#frame-generation-nts-g)). |
| Vector mode: forward warping with bilinear splats and a depth order | `nts/framegen.py` `splat`, `between` | Mark, McMillan & Bishop, *Post-Rendering 3D Warping*, I3D 1997; McMillan, *An Image-Based Approach to Three-Dimensional Computer Graphics*, 1997 | Nearest surface first keeps outlines anti-aliased; weighting the shares by each pixel's stretch lost 2.4 dB on at140, because the slopes of real motion fields are too noisy. |
| Push-pull fill where nothing landed | `nts/framegen.py` `fill` | Gortler et al., *The Lumigraph*, SIGGRAPH 1996 | In surface mode the pixels filled this way (3.5 % on at140) held 87 % of the error, so they are taken from the vector-mode frame when `--geom` is given. |
| A point that only one neighbour shows takes the mean of that colour and the vector-mode colour | `nts/framegen.py` `between_surf` | ours | Such a point (about 1 % of a frame, next to a moving outline) is often lit differently in its one witness: 43.55 dB with the one colour alone, below vector mode, 44.19 dB with the mean in the same test. |
| Next-frame motion is the negated Vector.ZW in image space | `nts/conventions.py` `mv_next_image` | our measurement | at140, frames 150 / 170 / 190: warp error 6.9 / 4.7 / 3.6 with this sign, 29.0 / 23.1 / 16.9 with the other. |

## Upscaling (NTS-F)

`nts/pipeline.py` upscales a jittered low-resolution render of the whole frame by temporal accumulation
(`baselines/accumulate.py`), steadies a natively rendered box around the head and lays it over the upscaled frame
with a feathered edge. No network, no weights; the buffers and render modes are in [`INTEGRATION.md`](INTEGRATION.md).

| Choice | Where | Source | What decided it |
|---|---|---|---|
| Each output pixel finds its own position in the jittered render and takes the 2x2 render pixels around it, weighted by a Gaussian of their distance in render pixels | `nts/gather.py` `gather` | Inverse-mapped (gather) resampling of a jittered render: Herzog et al., *Spatio-Temporal Upsampling on the GPU*, I3D 2010 | With a static camera and N jitter phases the sum converges to a multi-sample reconstruction, not an interpolation. Since 0.3.0: 0.2.0 splatted each sample at its output position; the gather scores the same from x1 to x2, better at x3, and is faster on a GPU. |
| Halton(2,3) jitter with 8 × (output / render)² phases | `nts/conventions.py` `jitter_seq`, `jitter_phases` | Public FSR 2 docs (MIT) and the FSR 4 manual on GPUOpen | Phase counts measured in our earlier render harness (not in this repo). |
| Motion vectors dilated by nearest depth | `nts/warp.py` `dilate_mv`, `hr_motion` | Karis, *High Quality Temporal Supersampling*, SIGGRAPH 2014 course; FSR 2 (MIT) docs | Edges of moving foreground objects carry the foreground's motion. |
| History clipped to the per-channel min and max of the current 3×3 neighbourhood (widened about its middle, never narrowed), its weight reduced by how far it was clipped | `baselines/accumulate.py` `minmax_box`, `Accumulator.step` | Karis, *High Quality Temporal Supersampling*, SIGGRAPH 2014 course | History that had to move a lot to fit the neighbourhood is less trustworthy. |
| Disocclusion by a depth test against the nearest previous depth in a 2×2 footprint (relative tolerance, fitted: 16.9 %) | `nts/warp.py` `disocclusion` | FSR 2 depth clip (MIT docs); NSRD "reliable warping" (CVPR 2024, MIT code, not copied) | A bilinear depth lookup would invent surfaces at silhouettes. |
| The head box is rendered at native resolution; its lighting (colour / albedo) is accumulated and multiplied back by the current albedo | `nts/pipeline.py`, `datagen/blender/render_lr.py` `setup_roi` | ours | The face is never upscaled, so it cannot swim or change identity; texture stays exact and only render noise is averaged. |
| Quality modes ×1, ×1.5, ×1.71, ×2, ×3 | `nts/conventions.py` `QUALITY_MODES` | The usual ladder of scale factors of temporal upscalers | Balanced is 630×1120, so the render size divides the output with the aspect kept exactly. |
| Sign conventions in one file, checked by the accumulator (`--sample-sign`, `--mv-sign`) | `nts/conventions.py`, `baselines/accumulate.py` | our measurement on renders | With a wrong jitter or motion sign the accumulation falls apart. |

## Speed

The CPU stages were made faster on 2026-11-24 without changing their output: the frames written are the same bytes
at any worker or thread count. One rule decided every change: the new function must equal the one it replaced bit
for bit (`tests/test_speed_equivalence.py` holds each replaced function verbatim). Tables: [`SPEED.md`](SPEED.md).

| Choice | Where | Source | What decided it |
|---|---|---|---|
| One set of bilinear taps per motion field, flat gathers; quarter-size motion dilates only the pixels that sampling keeps | `nts/warp.py` `Warper`, `hr_motion` | Standard numerical practice, no outside source | Bilinear warp 208 to 148 ms a frame, quarter-size motion 141 to 25 ms; the steadier warps up to six images along the same motion. |
| Cumulative-sum box filters read by slices; a 256-entry sRGB decode table per input precision | `nts/tone.py` `box`, `to_lab`, `gen/steady2.py` `box` | Standard numerical practice, no outside source | Box filter r = 8: 31 to 21 ms and 75 to 52 ms; sRGB to CIELAB 83 to 62 ms. |
| Independent frames in worker processes, results in frame order (`--workers`) | `nts/par.py` `pmap` | Standard practice, no outside source | at140, 60 frames: tone 548 s with one process, 256 s with four; carry 904 s and 264 s. |
| The steadier reads the next frame and runs its warps in threads (`--threads 3`) | `gen/steady2.py` `main` | Standard practice, no outside source | 10 frames: 130 s with one thread, 104 s with three, for 0.7 GB more memory. |

## Tried and dropped

- Averaging the whole finished frame over time (`gen/steady.py`): it cancels texture (at140 face ratio 1.09 to 0.67).
- Blending a forward and a backward steadied run (`gen/steady_bi.py`): cross-fading two differently held textures
  pulses; face flicker 1.44 to 1.69 against 1.02 forward only.
- The tone change averaged along the motion (`nts/tone_steady.py`): it turned a 50-level switch into a slow pulse
  and left its cause; ramps in place of the flood's yes/no tests spread a faint garment weight over desk and props.
- The sharp keyframe as the edit's reference image: the model copies it. Prompts that describe the person, with a
  reference: they replace the face.
- A one-step video restoration model on every frame (SeedVR2-3B, `tools/kaggle/run_seedvr.py`): 84 to 193 s per
  frame on a T4, about 42 GPU-hours for an 1,800-frame episode.
- An edge-preserving (guided-filter) base for the residual (`gen.structure --edge`): it also removed the mid-size
  shading that makes the composite read as a face with depth.
- A second texture source from a more detailed look of the same render (`--render`): green irises with colour
  residuals, a dark line above the upper lip, dashed mottling in light hair.
- Band-limited grain on smooth skin (`--skin-grain`): three blind judges picked the still without it, six of six.
- Hair filtered along the render's strand direction, synthesised strands, a lower mid band (`--hair-flow`,
  `--hair-strands`, `--hair-mid`): blind pairs gave no consistent order; plain hair beat the half-gain band 16 : 2.
- Softening the head detail by 0.5 px (`--soft`): made redundant by `speck_free`, and it cost the finest band (face
  fine energy × 1.00 / 1.12 of the render's softened, × 1.21 / 1.37 without).
- An O(1) running maximum (van Herk / Gil-Werman) for the tone step's dilations: 7× faster, one pixel one level off.
- For carry pixels without a match, a position from the point's own surface coordinates (the keyframe's field
  extended to first order from the nearest keyframe pixel): slower than the short re-seating and steadier only than
  the long one.
- Holding a re-seated position from the neighbouring frame, where the same point is found by surface identity: no
  steadier, and frames would no longer be matched independently.
- Two keyframe positions mixed across a one-pixel border between surfaces: no steadier. A one-pixel blend of two
  unrelated textures flickers almost as much as a hard step.
