"""Blender's exact view chain (scene-linear Rec.709 -> exposure -> AgX look -> AgX sRGB), via OCIO with Blender's own
config. Ported from cycles-harness run_harness.py, where it matched Blender's PNGs to mean 0.28 / max 1 (8-bit)."""
import os

import numpy as np

# Blender's own config; NTS_OCIO_CONFIG points at a copy on a machine without Blender
OCIO_CONFIG = os.environ.get("NTS_OCIO_CONFIG",
                             os.path.expanduser("~/blender-4.5/4.5/datafiles/colormanagement/config.ocio"))


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
        img = np.ascontiguousarray(rgb, dtype=np.float32)
        self.cpu.applyRGB(img)
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
