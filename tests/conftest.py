"""pytest setup: the repo root on sys.path, and the torch tests left out where torch is not installed (CI, the 4050's
Blender Python). Every test file also runs as a plain script."""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

collect_ignore = [] if importlib.util.find_spec("torch") else ["test_torch_ops.py"]
