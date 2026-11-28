# Render-to-photo heads (R2P)

Unpaired training data for a model that turns a rendered (CG) human head into a photoreal one: rendered head crops on
one side, head crops of real photographs on the other (512×512), plus whole frames for the same task at frame level:
full 1080×1920 renders on one side and whole-scene photographs on the other.

## Contents

| folder | what | count |
|---|---|---|
| `data/cg/` | Blender Cycles renders of MakeHuman/MPFB characters in studio and office sets, cropped to the head. `*.jpg` is the image, `*.reg.png` a region mask (red = face, green = hands, blue = hair), `*.mv.npy` the motion from the previous frame (float16, 64×64×2, offsets in 512-px units, current → previous) for consecutive-frame pairs. | see `COUNTS.txt` |
| `data/real/` | Head crops of photographs, framed like the renders. | see `COUNTS.txt` |
| `data/eval/` | Head crops of four rendered clips (`eval_zd400`, `eval_rs345`, `eval_at140`, `eval_pty332`), consecutive frames, with region masks. Not used for training. | see `COUNTS.txt` |
| `data/cgf/` | Whole 1080×1920 renders of the same characters and sets, in consecutive-frame pairs. `*.jpg` image, `*.reg.png` region mask (same colours), `*.mv.npy` motion from the previous frame (float16, 240×135×2, offsets in full-size pixels, current → previous). | see `COUNTS.txt` |
| `data/realf/` | Whole-scene photographs of people indoors, scaled down so faces are about as large as in the renders (55–210 px wide). `scenes.jsonl` lists the scale and face boxes per photo. | see `COUNTS.txt` |
| `data/evalf/` | The whole frames of the four eval clips. Not used for training. | see `COUNTS.txt` |
| `ATTRIBUTION.csv` | One row per file in `data/real/` and `data/realf/`: title, creator, licence, licence link, source page, and what was changed. | |

## Licences

This dataset is a collection of items under different open licences. All of them allow commercial use.

- **`data/cg/`, `data/cgf/`, `data/eval/` and `data/evalf/` (renders, masks, motion):** made by the dataset author and released under
  **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). The characters are built from MakeHuman / MPFB assets,
  which are CC0.
- **`data/real/` and `data/realf/` (photographs):** third-party photographs published by their creators under **CC0 1.0** or a
  **CC BY** licence (the version is listed per file). They were found through the Openverse API, restricted to those
  licences; no NonCommercial, NoDerivatives or ShareAlike material is included. Each item is an adaptation: in
  `data/real/` the photo was cropped to the head, resized to 512×512 and re-encoded; in `data/realf/` it was resized
  and re-encoded (the change is listed per file). Each crop stays under its original licence.
  **If you redistribute or publish these crops, keep `ATTRIBUTION.csv` with them**: for CC BY items it carries the
  required credit, licence link, source link and note of changes. The creators do not endorse this dataset or
  anything made with it.

## People in the photographs

Creative Commons licences cover copyright only. They do not grant publicity, privacy or personality rights of the
people shown. The photos are here so a model can learn what photographed skin, hair, clothes and rooms look like in general, not
to identify or reproduce any individual. If you are pictured, or you are a creator, and want an item removed, open a
discussion on this dataset's page and it will be taken out.

## What is not in here

No output of any proprietary upscaler or enhancer, and no images from datasets with non-commercial terms.

## Files that describe the data

- `ATTRIBUTION.csv`: columns `file, title, creator, creator_url, license, license_url, source_page, provider, changes`.
- `COUNTS.txt`: number of files per folder and the licence breakdown of `data/real/` and `data/realf/`.
