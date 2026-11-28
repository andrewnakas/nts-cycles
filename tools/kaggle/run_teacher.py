# Kaggle kernel: phase 1 teacher pilot (tools/teacher_wan.py) on the nts-pilot dataset (clips/<name>/in/f_####.jpg plus
# teacher_wan.py). Output: /kaggle/working/out/<name>_s<strength>/f_####.png and teacher.log. ARGS patched by the driver.
import glob, os, subprocess, sys
ARGS = "--model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --strengths 0.3,0.5 --steps 30"
subprocess.run(f"{sys.executable} -m pip install -q -U diffusers transformers accelerate ftfy", shell=True, check=True)
src = [p for p in glob.glob("/kaggle/input/**/teacher_wan.py", recursive=True)][0]
clips = os.path.join(os.path.dirname(src), "clips")
print("clips:", sorted(os.listdir(clips)), flush=True)
os.chdir("/kaggle/working")
r = subprocess.run(f"{sys.executable} {src} --clips {clips} --out out {ARGS} 2>&1 | tee teacher.log", shell=True)
os.remove("out/prompt_embeds.pt") if os.path.exists("out/prompt_embeds.pt") else None
