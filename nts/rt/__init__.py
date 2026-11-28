"""NTS-F in real time: the jittered temporal upscaler of baselines/accumulate.py as GPU compute shaders (WGSL, wgpu).

No neural network. Same buffers a game hands its temporal upscaler (docs/INTEGRATION.md): colour at render
resolution, depth, motion vectors, the jitter offset and a reset flag.
  upscaler.py  RTUpscaler, the two passes (shaders/prep.wgsl at render size, shaders/resolve.wgsl at output size)
  gpu.py       adapter choice (NTS_RT_ADAPTER picks one by name), device, timestamp timing
  run.py       upscale a render folder into frames the eval harness reads
"""
