"""Eval only (never read by the pipeline): face match of candidate face crops against the render's face, on a CPU
Kaggle kernel. Same code as eval/nr_identity.py (facenet-pytorch, vggface2 weights; cosine to the render's face).
Used when the Mac cannot run it (its `heavy` slots are taken). Input: faces/<set>/<method>_<clip>.npz in nts-pilot2."""
import base64, glob, os, subprocess, sys

CODE_B64 = ""  # eval/nr_identity.py (filled in by push_ident.sh)
SETS = ""  # comma list of sets under faces/ (filled in by push_ident.sh)

subprocess.run(sys.executable + " -m pip install -q --no-deps facenet-pytorch 2>&1 | tail -n 2", shell=True)
open("/kaggle/working/nr_identity.py", "wb").write(base64.b64decode(CODE_B64))
roots = glob.glob("/kaggle/input/**/faces", recursive=True)
assert roots, "no faces/ folder in the dataset"
for s in [x for x in SETS.split(",") if x]:
    d = os.path.join(roots[0], s)
    print("== set", s, len(glob.glob(d + "/*.npz")), "crops", flush=True)
    r = subprocess.run([sys.executable, "/kaggle/working/nr_identity.py", d], capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-1500:], flush=True)
    open("/kaggle/working/identity_%s.md" % s, "w").write(r.stdout)
