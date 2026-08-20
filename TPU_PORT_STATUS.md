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

- Actual Colab v5e-1 device boot and HBM reporting.
- Real ComfyUI model inference and frontend workflow completion.
- Native INT8 lowering/HLO and CPU-fallback analysis.
- INT8 ConvRot correctness and performance on realistic transformer layers.
- SDPA throughput and peak HBM.
- Current video-model inference.
- Compilation reuse across sampler steps and restored sessions.
- Dynamic offload behavior and oversized-model staging.
- Portable cache reuse across independent Colab sessions.
- Compilation, steady execution, host transfer, HBM, and host-RAM profiling.

## Final static hardening before v5e validation

- The notebook startup cell now uses a separately named startup-log path and syntactically valid process assertion. Unit tests compile every notebook Python cell in addition to checking the repository and PR branch.
- `tools.tpu_colab_setup` dynamically selects the latest stable release advertised by the official PyTorch/XLA TPU index. A pip dry-run resolves an exact Torch, TorchVision, TorchAudio, PyTorch/XLA, and libtpu tuple; the tuple is installed together only when the existing environment is incomplete or incompatible.
- ComfyUI requirements continue to exclude Torch packages, and the notebook verifies afterward that none of the exact selected TPU versions changed.
- Probe inputs are created outside target timings. Each execution cycle is isolated by blocking pre- and post-operation synchronization, so lazy work from setup or a previous test cannot enter another operation's duration.
- Comfy Kitchen quantization results are passed explicitly to their dependent dequantization and linear probes. Failures mark dependents `blocked` with `blocked_by` rather than producing misleading secondary errors.
- Core, pre-Kitchen, post-Kitchen, and final XLA metrics snapshots are saved separately. Diagnostic environment variables are set before the first `torch_xla` import and TPU runtime initialization.
- The ConvRot comparison remains `F.linear(x, weight, bias)`: the eager implementation applies the same normalized orthogonal Hadamard transform to activation and weight inner dimensions, so their product is mathematically the original linear transform before quantization error.
