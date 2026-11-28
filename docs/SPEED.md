# Speed (2026-11-24)

The CPU stages of NR-1 were made faster without changing their output: the frames written are the same files, byte
for byte, at any `--workers` or `--threads` count. Nothing about the method changed; the speed comes from doing the
same arithmetic in the same order with less memory traffic, and from running independent frames at the same time.

## How it was checked

- **Per function**: `tests/test_speed_equivalence.py` holds each replaced function verbatim and checks the new one
  against it bit for bit (`np.array_equal`, same dtype and shape) on random data with the awkward cases: NaN, inf
  depth, ties in depth, 1-pixel-wide arrays, bool masks, windows wider than the array. It runs in CI.
- **Per stage, on real frames** (at140, the 4050 render box): the old code and the new code run on the same inputs,
  and every output is compared pixel by pixel. Steadied base (10 frames, `--threads` 1 and 3), tone (60 frames:
  frames, weight maps and `params.json`, `--workers` 1 and 4) and the carry (60 frames, `--workers` 1, 3 and 4
  against each other): largest difference 0 in every case. The tone step was also compared on 12 frames on a Mac,
  with the production options and with the defaults (no geometry): identical both ways.

## Per step

One 1080×1920 frame of at140 (real motion, depth and colour), one core, best of 3, CPU time. MacBook (Apple silicon,
numpy 1.26).

| step | before, ms | after, ms | speedup |
|---|---|---|---|
| bilinear warp, RGB (`nts.warp.warp`) | 208 | 148 | 1.4× |
| Catmull-Rom warp (`gen.steady2.warp_cr`) | 846 | 602 | 1.4× |
| motion to quarter size, for the tone step's votes (`nts.warp.hr_motion`) | 141 | 25 | 5.6× |
| motion ×2 with depth dilation (`nts.warp.hr_motion`) | 44 | 43 | 1.0× |
| disocclusion test (`nts.warp.disocclusion`) | 49 | 51 | 1.0× |
| box filter r = 8 (`gen.steady2.box`) | 75 | 52 | 1.4× |
| box filter r = 8 (`nts.tone.box`) | 31 | 21 | 1.5× |
| sRGB to CIELAB (`nts.tone.to_lab`) | 83 | 62 | 1.3× |
| Gaussian σ = 3 (`gen.structure.gauss`) | 289 | 106 | 2.7× |

What changed, function by function:

- **Warps** (`nts/warp.py`): the four bilinear taps are read from a flat view with `take` instead of 2-D fancy
  indexing, and the weights are multiplied in place. A `Warper` works out the taps and weights of one motion field
  once and applies them to every image warped along it. The steadier warps up to six images per frame along the same
  motion. The pixel grid is cached.
- **Motion at a smaller size**: when the tone step needs motion at a quarter of the render size, only the pixels that
  nearest-neighbour sampling keeps are dilated, not the whole frame.
- **Box filters**: the cumulative-sum differences are slices, not `np.take` copies, and the division is in place.
- **sRGB decode**: frames read from 8-bit PNGs only hold the 256 levels, so the decode is a 256-entry table lookup.
  The old formula ran in float32 for float32 input and in float64 for uint8 input, so there are two tables, each
  built at the precision its input got. Any other input takes the old formula.
- **Gaussian**: taps are slices of the padded array, not `np.take` copies.

Tried and dropped: a running maximum in O(1) per pixel (van Herk / Gil-Werman) for the tone step's grey dilations,
7× faster on that step. Where a window holds both −0.0 and +0.0, a maximum taken in a different order can return the
other zero, and later arithmetic turned that into one output pixel one level off (at140 frame 167, default options).
`maxr` and `max3` stay as they were.

## Per stage

at140 on the 4050 render box, old and new code run back to back (2026-11-25). The box is shared with other jobs;
everything ran at `nice 19` with 2 BLAS threads, so wall times are noisy and the CPU column is the steadier figure.
Peak memory is per process.

| stage | | wall | CPU (user) | peak memory |
|---|---|---|---|---|
| steadied base, frames 140-149 | before | 164 s | 55 s | 1.8 GB |
| | after, `--threads 1` | 130 s | 45 s | 2.5 GB |
| | after, `--threads 3` (default) | **104 s** | 56 s | 3.2 GB |
| tone, 60 frames | before | 682 s | 422 s | 1.2 GB |
| | after, `--workers 1` | 548 s | 298 s | 1.4 GB |
| | after, `--workers 4` | **256 s** | 430 s | 1.4 GB per worker |
| carry + compose, 60 frames (`--surf`, 2026-11-26) | `--workers 1` | 904 s | 450 s (+ 354 s system) | 1.7 GB |
| | `--workers 4` | **264 s** | 522 s (+ 396 s system) | 1.4 GB per worker |

With threads, the steadier reads the next frame's inputs while the current one is processed, and runs one frame's
warps at the same time (numpy releases the GIL). That costs memory: the inputs of two frames are held at once.
The carry finds every frame's keyframe positions in the surf passes (`nts/surface.py`): per frame about 1 s to read
the frame's surf pass (13 float channels) and 3 s to match it on one core, then the compose. It was timed while the
box ran other jobs (load average 13 on 12 cores); its high system time is page faults on the large float arrays.
Side by side under the same load with `--workers 3`, at140 (rendered again from today's scene) took 213 s with it and 209 s with the carry along the
renderer's motion vectors it replaced (2026-11-26). Surf passes rendered from now on have 10 channels (the AOVs'
constant alpha is no longer saved; a third faster to read on a test scene).

## Using it

| stage | option | default |
|---|---|---|
| `gen.steady2` | `--threads N` (env `NTS_THREADS`) | 3 |
| `nts.tone apply` | `--workers N` (env `NTS_WORKERS`; 0 = one per core) | 1 |
| `gen.carry` | `--workers N` (same) | 1 |
| `nts.framegen` | `--workers N` (same) | 1 |

Worker processes are forked on Linux, so they share the shot's data, and spawned on macOS and Windows. Results come
back in frame order and every frame is written by one worker, so the output does not depend on the count. Memory
grows with the worker count, and a machine that starts to swap gets slower, not faster.
