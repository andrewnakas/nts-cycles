"""Profile one descriptor evaluation of the look fit (where the 0.2 s goes)."""
import cProfile
import json
import os
import pstats
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nts import lookfit as lf  # noqa: E402

lg, brief = sys.argv[1], sys.argv[2]
lut = lf.AgxLUT(*json.load(open(os.path.join(lg, "meta.json")))["view"])
fr = lf.Frame("x", os.path.join(lg, "f_0430.exr"), json.load(open(brief)), lut)
print({k: v[2].shape for k, v in fr.parts.items()})
p = lf.p0(sorted(fr.parts["small"][0]))
cProfile.run("for _ in range(5): lf.frame_error(fr, p, lut)", "/tmp/lf.prof")
pstats.Stats("/tmp/lf.prof").sort_stats("cumulative").print_stats(14)
