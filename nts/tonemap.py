"""Blender's exact view chain (scene-linear Rec.709 -> exposure -> AgX look -> AgX sRGB), via OCIO with Blender's own
config. Ported from cycles-harness run_harness.py, where it matched Blender's PNGs to mean 0.28 / max 1 (8-bit)."""
import os

import numpy as np

# Blender's own config; NTS_OCIO_CONFIG points at a copy on a machine without Blender
OCIO_CONFIG = os.environ.get("NTS_OCIO_CONFIG",
                             os.path.expanduser("~/blender-4.5/4.5/datafiles/colormanagement/config.ocio"))


_POOL = None


def _pool():
    """Threads for large frames: OCIO's applyRGB releases the GIL, so row bands run in parallel (same output)."""
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(int(os.environ.get("NTS_TONEMAP_THREADS", min(8, os.cpu_count() or 1))))
    return _POOL


class Tonemapper:
    def __init__(self, view="AgX", look="AgX - Punchy", exposure=0.35):
        import PyOpenColorIO as ocio
        cfg = ocio.Config.CreateFromFile(OCIO_CONFIG)
        pipe = ocio.LegacyViewingPipeline()
        pipe.setDisplayViewTransform(ocio.DisplayViewTransform(src="Linear Rec.709", display="sRGB", view=view))
        if look and look != "None":
            pipe.setLooksOverrideEnabled(True)
            pipe.setLooksOverride(look)
        pipe.setLinearCC(ocio.MatrixTransform.Scale([2.0 ** exposure] * 3 + [1.0]))
        self.cpu = pipe.getProcessor(cfg).getDefaultCPUProcessor()

    def __call__(self, rgb):
        img = np.array(rgb, dtype=np.float32, order="C", copy=True)  # applyRGB works in place: never on the caller's
        if img.ndim == 3 and img.shape[0] >= 64 and img.size >= 3 * 2 ** 18:
            list(_pool().map(self.cpu.applyRGB, np.array_split(img, 16, 0)))  # bands of whole rows: contiguous views
        else:
            self.cpu.applyRGB(img)
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
