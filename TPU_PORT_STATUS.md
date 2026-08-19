# TPU Port Status

## Selected TPU architecture

PyTorch/XLA with PJRT is selected because it is the publicly packaged PyTorch integration for Google Cloud/Colab TPU. TorchTPU was not available in this environment as a verified Colab v5e runtime. The optional runtime is isolated in `comfy/tpu.py`, preserving every existing backend and allowing a future runtime replacement at one boundary.

## Historical XLA findings

The historical `radna0/ComfyUI-XLA` repository and ComfyUI PR #5657 could not be fetched from this execution environment because outbound GitHub and web access returned authorization/network errors. The requested archaeology remains hardware-validation work; no historical patch was copied. The current implementation independently applies the durable integration ideas described by that work: explicit XLA device selection, memory reporting, synchronization, normal smart memory, and persistent cache initialization before device creation.

## ComfyUI modifications

`--tpu` explicitly selects TPU and fails clearly when `torch-xla` is absent. `--tpu-cache-dir` configures persistent compilation. Model management treats `xla:0` as the accelerator, uses XLA HBM data, selects BF16 and PyTorch SDPA, synchronizes XLA, retains CPU offload and generic model movement, and avoids CUDA cache/stream behavior.

## Comfy-Kitchen findings

The installed Comfy Kitchen 0.2.31 source was inspected. Its registry includes portable eager implementations for quantization, INT8 linear, ConvRot, RoPE, AdaLN, and related tensor layouts. ComfyUI already disables CUDA and Triton backends on a non-CUDA TPU installation, leaving eager dispatch. The eager `int8_linear` dequantizes/multiplies through portable PyTorch operations; whether XLA fuses or retains native INT8 GEMM requires HLO/profiler evidence. A dedicated Kitchen TPU backend must not be advertised before that evidence exists.

Direct access to the authorized `Comfy-Kitchen-TPU` repository was blocked by the environment's outbound GitHub proxy (HTTP 403). No speculative secondary-repository patch was created because v5e HLO/profile results are required to choose a correct kernel implementation.

## INT8 and ConvRot

Existing checkpoint representations and eager dispatch are preserved unchanged. The capability probe exercises INT8 and Comfy Kitchen quantization, but native/efficient INT8 GEMM and complete ConvRot checkpoint inference are **TPU HARDWARE VALIDATION PENDING**. No checkpoint conversion is introduced.

## Attention

TPU selects PyTorch scaled-dot-product attention. CUDA FlashAttention, xFormers, SageAttention, and CUDA Comfy Kitchen attention are not selected. Throughput and peak-HBM validation remain pending on v5e.

## Smart memory

Generic ComfyUI model loading, partial loading, unloading, `Tensor.to(xla)` staging, CPU offload, and memory reserve calculations remain active. HBM total/free values come from XLA rather than a hard-coded v5e capacity. Transfers are synchronous; asynchronous prefetch is not claimed without runtime profiling.

## Compilation and portable cache

The writable cache is initialized before the first device is created. Empty first run, compilation, ZIP export, later upload/import, reuse, and continued accumulation are supported by `tools/tpu_cache.py`. New shapes, dtypes, layouts, batch sizes, frame counts, and sequence lengths may produce new graphs.

## Colab

Open `colab/ComfyUI_TPU_v5e.ipynb`, select a v5e-1 TPU runtime, execute setup, optionally import a cache ZIP, run the probe, and launch the server/tunnel cells. Models may live in Colab storage or paths configured through `extra_model_paths.yaml`; Drive and Hugging Face downloads remain explicitly user initiated.

## Hardware validation and performance

Status: **IMPLEMENTED**, **STATICALLY TESTED**, **SOFTWARE TESTED ON CPU WITH MOCKED XLA**, **TPU HARDWARE VALIDATION PENDING**. No v5e was exposed to this environment, so minimal ComfyUI inference, real INT8, ConvRot, video, oversized-model staging, HLO inspection, compilation timing, sampling speed, HBM peak, host RAM, and transfer/offload measurements are not claimed.

## Failed experiments and remaining limitations

GitHub clone and web research were blocked by the environment. PyTorch/XLA was not installed and no TPU device was available. Efficient native INT8/ConvRot, fallback traces, stable graph counts across real workflows, transfer overlap, video inference, and a possible Comfy Kitchen TPU kernel must be decided from the notebook's real-v5e probe and profiler output.
