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

## Hardware validated on real Colab v5e-1

The first hardware run validated XLA device creation, 16,909,336,576-byte HBM reporting, FP32/BF16/FP16 matmul, BF16 linear and conv2d, layer norm, SDPA, RoPE-style operations, generic INT8 matmul, non-contiguous operations, host/TPU transfers, eager Kitchen quantization, functional Kitchen INT8 and ConvRot linear, ComfyUI startup, PyTorch attention selection, and a persistent Cloudflare heartbeat launch. Current v5e detection additionally covers `TPU_ACCELERATOR_TYPE` and numeric VFIO groups.

## Hardware issue discovered

Comfy-Kitchen 0.2.31 eager `int8_linear` calls `torch._int_mm`. PyTorch/XLA 2.9 reported four `aten::_int_mm` CPU fallbacks for the two normal and two ConvRot executions. Generic INT8 matmul is XLA-native but returns INT8 and cannot replace the required INT32 accumulation. The correctness fallback remains; fully TPU-native Kitchen INT8 is not claimed.

## Failed experiments and remaining limitations

The validated generic INT8 matmul cannot replace `_int_mm`: it returns INT8 and overflowed a simple wide-accumulation example, whereas `_int_mm` returned INT32. Local research access to upstream repositories remains blocked, and this environment has no TPU, so an unvalidated production kernel was not substituted. The next implementation boundary is a Comfy-Kitchen TPU backend with an s8×s8→s32 lowering, validated by the new experiment harness before dispatch is enabled.

## Fixes in this review pass

- Memory reporting now uses the public `torch_xla.core.xla_model.get_memory_info()` API. Modern `bytes_limit`/`bytes_used` results are normalized to total/free bytes; the older public kilobyte result is accepted by a validated compatibility adapter. Private `_XLAC` memory calls are no longer used.
- Synchronization now calls `torch_xla.sync(wait=True)`. Probe timings bracket two independently completed executions and distinguish first execution from the repeated execution.
- TPU initialization is idempotent for the same compilation cache and rejects cache changes after the XLA device exists.
- The Colab notebook uses `https://github.com/kevinmetten/ComfyUI-TPU.git` and the existing `codex/build-comfyui-backend-for-google-colab-tpu` PR branch from one configuration cell. Change only `BRANCH` to `master` after merge.
- Colab installs Torch, TorchVision, TorchAudio, PyTorch/XLA, and the TPU runtime through one resolver transaction when the existing Torch/XLA major-minor versions are not compatible. ComfyUI requirements are installed without subsequently replacing that resolved stack.
- The probe directly executes Comfy Kitchen rowwise/tensorwise INT8 quantization, activation rotation/quantization, ConvRot weight quantization, simple/ConvRot dequantization, `int8_linear`, and end-to-end ConvRot `int8_linear`, with device/dtype/finite/error metadata and XLA metrics/compiler artifacts.
- Cache manifests now record detected environment, Python, Torch, XLA, libtpu, ComfyUI, and Comfy Kitchen source/version/commit information. Imports warn about version differences without rejecting or deleting cache entries.
- CUDA SDPA global configuration, CUDA pinned memory, non-blocking transfer claims, and async offload are disabled for TPU. Generic smart-memory movement and synchronous CPU staging remain enabled.

## Still pending hardware validation

- Fully TPU-native Comfy-Kitchen wide-accumulation INT8 linear.
- Real checkpoint inference and MiniMax H3 INT8 ConvRot.
- Current video-model inference and performance comparison with T4.
- Dynamic offload behavior, oversized-model staging, and real-model HBM peaks.
- Compilation reuse across sampler steps and portable cache reuse in a fresh Colab session.

## Final static hardening before v5e validation

- The notebook startup cell now uses a separately named startup-log path and syntactically valid process assertion. Unit tests compile every notebook Python cell in addition to checking the repository and PR branch.
- `tools.tpu_colab_setup` dynamically selects the latest stable release advertised by the official PyTorch/XLA TPU index. A pip dry-run resolves an exact Torch, TorchVision, TorchAudio, PyTorch/XLA, and libtpu tuple; the tuple is installed together only when the existing environment is incomplete or incompatible.
- ComfyUI requirements continue to exclude Torch packages, and the notebook verifies afterward that none of the exact selected TPU versions changed.
- Probe inputs are created outside target timings. Each execution cycle is isolated by blocking pre- and post-operation synchronization, so lazy work from setup or a previous test cannot enter another operation's duration.
- Comfy Kitchen quantization results are passed explicitly to their dependent dequantization and linear probes. Failures mark dependents `blocked` with `blocked_by` rather than producing misleading secondary errors.
- Core, pre-Kitchen, post-Kitchen, and final XLA metrics snapshots are saved separately. Diagnostic environment variables are set before the first `torch_xla` import and TPU runtime initialization.
- The ConvRot comparison remains `F.linear(x, weight, bias)`: the eager implementation applies the same normalized orthogonal Hadamard transform to activation and weight inner dimensions, so their product is mathematically the original linear transform before quantization error.
