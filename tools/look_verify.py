"""Check a look fit against real Cycles renders made with it (datagen/jobs/look_verify.sh): for each frame, the fit's
predicted image (light groups + skin basis, nts/lookfit.compose) against the real render, as blurred CIELAB dE on the
head and the whole frame, plus each one's descriptor error to the brief. No reference image is read.

Run: look_verify.py <fit.json> <verify_dir> [lg_root]   (verify_dir/<episode>/f_####.exr; light groups in lg_root,
     default data/look/lg)
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval import look_brief as lb  # noqa: E402
from eval.look_score import blurred_lab  # noqa: E402
from nts import lookfit as lf  # noqa: E402


def main():
    fit_path, vdir = sys.argv[1], sys.argv[2]
    fit = json.load(open(fit_path))
    p = fit["params"]
    brief = json.load(open(fit["brief"]))
    lut = None
    rows = []
    for ep_dir in sorted(glob.glob(os.path.join(vdir, "*/"))):
        ep = os.path.basename(os.path.normpath(ep_dir))
        for real_path in sorted(glob.glob(os.path.join(ep_dir, "f_*.exr"))):
            f = os.path.basename(real_path)
            lg_path = os.path.join(sys.argv[3] if len(sys.argv) > 3 else "data/look/lg", f"{ep}_lg", f)
            if lut is None:
                lut = lf.AgxLUT(*json.load(open(os.path.join(os.path.dirname(lg_path), "meta.json")))["view"])
            groups, mist, region, _ = lf.load_lg(lg_path)
            pred = lut(lf.compose(groups, mist, region, p, lf.load_basis(lg_path, region), lf.basis_steps(lg_path)))
            _, _, _, image = lf.load_lg(real_path)
            # the EXR's Image is scene-linear; the fit's exposure lives in the view transform, as in compose()
            real = lut(image * 2.0 ** p["exposure"])
            d = np.linalg.norm(blurred_lab(pred, r=2) - blurred_lab(real, r=2), axis=-1)
            head = np.maximum(region[..., 0], region[..., 2]) > 0.5
            lay = lb.layout(region)
            native_desc = lb.describe_full(lut(lf.compose(groups, mist, region, lf.p0(sorted(groups), True),
                                                          lf.load_basis(lg_path, region), lf.basis_steps(lg_path))), region, lay)[0]
            target = lb.apply(brief, native_desc)
            fr = type("F", (), {})()
            fr.describe = lambda q, _l, im=real: lb.describe_full(im, region, lay)[0]
            fr.target = target
            real_err = lf.frame_error(fr, None, None)[0]
            fr.describe = lambda q, _l, im=pred: lb.describe_full(im, region, lay)[0]
            pred_err = lf.frame_error(fr, None, None)[0]
            rows.append((f"{ep}_{f[:-4]}", float(d[head].mean()) if head.any() else float("nan"), float(d.mean()),
                         pred_err, real_err))
            print(f"{rows[-1][0]:28s} pred-vs-real dE head {rows[-1][1]:.2f} frame {rows[-1][2]:.2f} | brief error "
                  f"pred {pred_err:.2f} real {real_err:.2f}", flush=True)
    a = np.array([r[1:] for r in rows])
    print(f"MEAN pred-vs-real dE head {np.nanmean(a[:, 0]):.2f} frame {a[:, 1].mean():.2f} | brief error pred "
          f"{a[:, 2].mean():.2f} real {a[:, 3].mean():.2f}")


if __name__ == "__main__":
    main()
