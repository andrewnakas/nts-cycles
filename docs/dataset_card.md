---
license: cc-by-4.0
task_categories:
- image-to-image
tags:
- super-resolution
- temporal-upscaling
- rendering
- blender
- cycles
pretty_name: NTS-Cycles temporal upscaling training pairs
---

# NTS-Cycles training pairs

Training data for **NTS-1**, an open, clean-room neural temporal upscaler for path-traced renders.
- Every frame is our own Blender Cycles render of our own scenes: the "LAST HUMAN DEV" short-video sets and
  characters.
- No frame comes from a proprietary or third-party upscaler, and none is used as a target.
- See `PROVENANCE.md` in the nts-cycles repository.

## Layout
There is one folder per shot and seed (`<shot>_s<seed>/`), each holding 100 consecutive frames at 30 fps.

| file | shape | contents |
|---|---|---|
| `lr.npy` | (T, 12, 540, 960) f16 | jittered low-res render: linear RGB, albedo, normal, depth, motion vectors (x, y, render px) |
| `gt.npy` | (T, 3, 1920, 1080) f16 | native 1080×1920 Cycles render, linear |
| `region.npy` | (T, 1920, 1080) u8 | bit 1 face, 2 hands, 4 hair |
| `guide.npy` | (T, 6, 1920, 1080) f16 | full-res albedo + normal from the native render (seed 0 only) |
| `guide16.npy` | (T, 6, 1920, 1080) f16 | the same from a cheap 16-spp guide render (seed 0, newer shots) |
| `meta.json` | | frame numbers, per-frame sub-pixel jitter, history-reset flags, scale, source |

Notes:
- The arrays are `numpy.save` files; load them memory-mapped with `np.load(path, mmap_mode="r")`.
- The view transform is Blender's AgX "Punchy" at exposure 0.35.
- `test_zd_s0` is the held-out test shot. Don't train on it.
