"""Identity column for the NR board (2026-11-18): facenet cosine between each method's face and its own input render's
face, from the crops eval/nr_board.py --faces writes (<dir>/<method>_<clip>.npz: out, ref, 160 px). Eval only.
facenet-pytorch InceptionResnetV1 (VGGFace2 weights; the dataset is research-only, which is why this stays in eval
and never touches training). Gate: >= 0.95.

Run (Mac CPU, crops copied from the 4050's runs/board_faces):
  uv run --python 3.11 --with 'torch==2.2.2' --with 'torchvision==0.17.2' --with 'numpy<2' --with facenet-pytorch \
    --with pillow python -m eval.nr_identity <faces_dir> [>> runs/board.md]
"""
import glob
import os
import sys

import numpy as np
import torch


def main():
    from facenet_pytorch import InceptionResnetV1
    m = InceptionResnetV1(pretrained="vggface2").eval()

    def emb(a):
        t = torch.from_numpy((a.astype(np.float32) - 127.5) / 128.0).permute(0, 3, 1, 2)
        with torch.no_grad():
            e = m(t).numpy()
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    res = {}
    for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))):
        lab, clip = os.path.basename(p)[:-4].rsplit("_", 1)
        z = np.load(p)
        res.setdefault(lab, {})[clip] = float((emb(z["out"]) * emb(z["ref"])).sum(1).mean())
    clips = sorted({c for v in res.values() for c in v})
    print("\n| identity | " + " | ".join(clips) + " | mean |\n|" + "---|" * (len(clips) + 2))
    for lab, v in res.items():
        print(f"| {lab} | " + " | ".join(f"{v[c]:.3f}" if c in v else "-" for c in clips) + f" | {np.mean(list(v.values())):.3f} |")


if __name__ == "__main__":
    main()
