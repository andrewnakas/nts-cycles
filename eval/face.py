"""Identity and skin metrics: the failures neural realism passes are known for (a face that becomes "a different
person", faces that swim in motion, hands and face in different skin tones). All in 8-bit display space against the
native Cycles render.

Masks come from the Region AOV of render_lr.py (native-mode GT render): masks/m_####.png, R = face, G = hands.
For older footage without masks, a box (x0,y0,x1,y1 in 1080x1920 pixels) stands in for the face.
"""
import numpy as np
from PIL import Image

from .metrics import box, hp_psnr, load, psnr, sharpness


def load_mask(path, size=None):
    im = Image.open(path).convert("RGB")
    if size and im.size != size:
        im = im.resize(size, Image.NEAREST)
    a = np.asarray(im)
    return a[..., 0] > 127, a[..., 1] > 127


def box_mask(shape, b):
    m = np.zeros(shape[:2], bool)
    x0, y0, x1, y1 = b
    m[y0:y1, x0:x1] = True
    return m


def lab_ab(rgb):
    """sRGB 8-bit -> CIE a*, b* (D65). Only chroma matters for skin-tone consistency."""
    c = rgb / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]], np.float32)
    xyz /= np.array([0.9505, 1.0, 1.089], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def symmetry(img, mask):
    """Mean |face - mirrored face| inside the mask's bounding box, mirrored about its centre column (lower = more
    symmetric). Compare to GT's own value: the native face is the target, not perfect symmetry."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        return float("nan")
    crop, m = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1], mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    both = m & m[:, ::-1]
    return float(np.abs(crop - crop[:, ::-1])[both].mean())


def frame_metrics(gt, test, face, hands=None):
    """Per-frame face metrics. hp_psnr separates structure change (identity) from tone change (allowed for a
    realism pass); sharp_ratio guards against winning by blurring."""
    r = {"face_psnr": psnr(gt, test, face), "face_hp_psnr": hp_psnr(gt, test, face),
         "face_sharp_ratio": sharpness(test, face) / max(sharpness(gt, face), 1e-6),
         "face_sym_delta": symmetry(test, face) - symmetry(gt, face)}
    if hands is not None and hands.sum() > 100:
        ab_t, ab_g = lab_ab(test), lab_ab(gt)
        dt = ab_t[face].mean(0) - ab_t[hands].mean(0)
        dg = ab_g[face].mean(0) - ab_g[hands].mean(0)
        r["skin_tone_delta"] = float(np.hypot(*(dt - dg)))  # how much the face-vs-hands chroma gap moved vs GT
    return r


def face_temporal(gt_prev, gt, t_prev, t, face):
    """Frame-to-frame change inside the face that the real motion doesn't explain (swimming features)."""
    return float(np.abs((t - t_prev) - (gt - gt_prev))[face].mean())


def face_lpips(gt, test, face, net=None):
    """Optional LPIPS on the face crop; returns None where torch/lpips aren't installed (the 4050 has no torch)."""
    try:
        import lpips
        import torch
    except ImportError:
        return None
    ys, xs = np.nonzero(face)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    net = net or lpips.LPIPS(net="alex", verbose=False)

    def t(a):
        return torch.from_numpy(a[sl] / 127.5 - 1).permute(2, 0, 1)[None].float()
    with torch.no_grad():
        return float(net(t(gt), t(test)))


__all__ = ["load", "load_mask", "box_mask", "frame_metrics", "face_temporal", "face_lpips", "symmetry", "box"]
