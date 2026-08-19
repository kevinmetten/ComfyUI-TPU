# Google Colab v5e TPU

This branch adds an optional PyTorch/XLA accelerator path. Ordinary CUDA, ROCm, XPU, MPS, and CPU installations do not import or require `torch-xla`.

## Start ComfyUI

Use a Colab v5e-1 TPU runtime and install matching `torch`, `torchvision`, and `torch-xla[tpu]` wheels from the official PyTorch/XLA TPU package index. Then run:

```bash
python tools/tpu_probe.py
python main.py --tpu --listen 0.0.0.0
```

TPU execution defaults to BF16, PyTorch scaled-dot-product attention, host offload, normal smart-memory loading, and a writable `.cache/tpu_xla` persistent compilation cache. `--tpu-cache-dir PATH` selects another local cache. Fine-grained asynchronous CUDA-style offload is not advertised on TPU; transfers remain synchronous until profiling on v5e demonstrates a supported overlap mechanism.

## Portable cache

The cache is local and does not require Google Drive:

```bash
python tools/tpu_cache.py export comfyui-v5e-tpu-cache.zip
python tools/tpu_cache.py import comfyui-v5e-tpu-cache.zip
```

Import before starting ComfyUI or the probe. The archive manifest is diagnostic; runtime cache keys decide whether an artifact can be reused, so incomplete or stale archives may remain and new graphs continue compiling into the same directory.

## Validation

`tools/tpu_probe.py` executes transfers, FP32/BF16/FP16 matrix multiplication, convolution, normalization, SDPA, RoPE-like layout operations, INT8 matrix multiplication, non-contiguous layout handling, and an installed Comfy Kitchen eager INT8 quantizer. Unsupported operations are reported individually rather than hiding a CPU fallback. Use `--large` for an additional transformer-sized matrix multiplication.

The first execution of each new shape compiles. Keep batch, latent dimensions, frame count, and token length stable across sampler steps to maximize reuse. TPU profiling should use the PyTorch/XLA profiler in the Colab runtime; no hardware performance claim is made until those traces are collected.
