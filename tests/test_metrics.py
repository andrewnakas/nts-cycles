"""Pins eval/metrics.py to the numbers cycles-harness measured on its calibration scene (CONVENTIONS.md), so every later
comparison (ours vs the reference) is on a harness that's known to be right. Skips when the 4050's test data isn't there.
Run: python tests/test_metrics.py  (on the 4050, with ~/scene-project/py)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval.metrics import compare  # noqa: E402

T = os.path.expanduser(os.environ.get("HARNESS_TEST", "~/cycles-harness/test"))
# folder -> (PSNR, SSIM, tErr) as measured by cycles-harness tools/metrics.py --skip 4
EXPECTED = {"lowres": (32.7, 0.964, 2.60), "cal_fw0.5": (44.0, 0.9965, 0.51)}


def test_calibration_parity():
    if not os.path.isdir(os.path.join(T, "gt")):
        print("skip: no calibration data at", T)
        return
    cuts = json.load(open(os.path.join(T, "calib_perf", "cuts.json")))
    for d, (p, s, e) in EXPECTED.items():
        agg, _ = compare(os.path.join(T, "gt"), os.path.join(T, d), skip=4, cuts=cuts)
        print(f"{d}: {agg['psnr']:.3f} dB  {agg['ssim']:.4f}  tErr {agg['terr']:.3f}")
        assert abs(agg["psnr"] - p) <= 0.05 + 0.005, (d, agg)   # reported values are rounded to 0.1 dB
        assert abs(agg["ssim"] - s) <= 0.0006, (d, agg)
        assert abs(agg["terr"] - e) <= 0.006, (d, agg)


if __name__ == "__main__":
    test_calibration_parity()
    print("ok")
