"""GPU access for nts.rt: adapter choice, device and GPU timestamps (wgpu: Vulkan, Metal or D3D12, no window needed).

NTS_RT_ADAPTER=<text> picks the first adapter whose name or backend contains <text> (e.g. "4050", "1080", "llvmpipe").
Otherwise: a discrete GPU before an integrated one before a software one, and Vulkan / Metal / D3D12 before OpenGL.
"""
import os

import numpy as np
import wgpu

_KIND = {"DiscreteGPU": 0, "IntegratedGPU": 1, "VirtualGPU": 2, "Unknown": 3, "CPU": 4}
_BACKEND = {"Vulkan": 0, "Metal": 0, "D3D12": 0, "OpenGL": 1}


def label(a):
    i = a.info
    return f"{i['device']} ({i['adapter_type']}, {i['backend_type']})"


def adapters():
    return list(wgpu.gpu.enumerate_adapters_sync())


def adapter(want=None):
    want = os.environ.get("NTS_RT_ADAPTER", "") if want is None else want
    ads = adapters()
    if not ads:
        raise RuntimeError("no GPU adapter found")
    if want:
        hit = [a for a in ads if want.lower() in label(a).lower()]
        if not hit:
            raise RuntimeError(f"no adapter matches {want!r}: " + "; ".join(label(a) for a in ads))
        return hit[0]
    return min(ads, key=lambda a: (_KIND.get(a.info["adapter_type"], 3), _BACKEND.get(a.info["backend_type"], 2)))


def device(want=None):
    """(device, adapter label). Timestamp queries and filtering of float32 textures are switched on when the adapter
    has them."""
    a = adapter(want)
    feats = [f for f in ("timestamp-query", "float32-filterable") if f in a.features]
    return a.request_device_sync(required_features=feats), label(a)


class Timer:
    """GPU timestamps for `n` compute passes: pass k writes queries 2k (start) and 2k + 1 (end)."""

    def __init__(self, dev, n):
        self.ok = "timestamp-query" in dev.features
        self.dev, self.n = dev, n
        if self.ok:
            self.qs = dev.create_query_set(type=wgpu.QueryType.timestamp, count=2 * n)
            self.buf = dev.create_buffer(size=16 * n, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)

    def writes(self, k):
        if not self.ok:
            return None
        return {"query_set": self.qs, "beginning_of_pass_write_index": 2 * k, "end_of_pass_write_index": 2 * k + 1}

    def resolve(self, enc, n=None):
        if self.ok:
            enc.resolve_query_set(self.qs, 0, 2 * (n or self.n), self.buf, 0)

    def read(self, n=None):
        """Duration of passes 0..n-1 in ms, as the GPU reports them (raw ticks taken as ns: check against wall clock)."""
        if not self.ok:
            return None
        n = n or self.n
        t = np.frombuffer(self.dev.queue.read_buffer(self.buf, 0, 16 * n), np.uint64).astype(np.int64)
        return (t[1::2] - t[0::2]) / 1e6
