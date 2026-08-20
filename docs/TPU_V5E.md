# Google Colab v5e TPU

This branch adds an optional PyTorch/XLA accelerator path. Ordinary CUDA, ROCm, XPU, MPS, and CPU installations do not import or require `torch-xla`.

The Colab notebook clones `https://github.com/kevinmetten/ComfyUI-TPU.git`. While PR #1 is under review its single `BRANCH` setting targets `codex/build-comfyui-backend-for-google-colab-tpu`; change that setting to `master` after the PR is merged. The default dependency remains upstream PyPI `comfy-kitchen` from `requirements.txt`. An optional `COMFY_KITCHEN_SPEC` setting is available for a future TPU fork, but should remain unset until v5e diagnostics demonstrate that a dedicated backend is needed.

## Start ComfyUI

Use the notebook on a Colab v5e-1 TPU runtime, or run `python -m tools.tpu_colab_setup --ensure` to resolve the current compatible TPU stack before starting manually:

```bash
python -m tools.tpu_probe --output tpu_probe.json --diagnostics diagnostics
python main.py --tpu --listen 0.0.0.0
```

TPU execution defaults to BF16, PyTorch scaled-dot-product attention, host offload, normal smart-memory loading, and a writable `.cache/tpu_xla` persistent compilation cache. `--tpu-cache-dir PATH` selects another local cache. Fine-grained asynchronous CUDA-style offload is not advertised on TPU; transfers remain synchronous until profiling on v5e demonstrates a supported overlap mechanism.

## Portable cache

The cache is local and does not require Google Drive:

```bash
python -m tools.tpu_cache export comfyui-v5e-tpu-cache.zip
python -m tools.tpu_cache import comfyui-v5e-tpu-cache.zip
```

Import before starting ComfyUI or the probe. The archive manifest is diagnostic; runtime cache keys decide whether an artifact can be reused, so incomplete or stale archives may remain and new graphs continue compiling into the same directory.

## Validation

`tools/tpu_probe.py` executes every operation twice with a blocking XLA synchronization after each run, separately recording the compile/first-execution and repeated-execution durations. It covers transfers, FP32/BF16/FP16 matrix multiplication, convolution, normalization, SDPA, RoPE-like layout operations, generic INT8 matrix multiplication, and the actual Comfy Kitchen rowwise/tensorwise quantizers, ConvRot primitives, dequantizers, and `int8_linear` paths. Unsupported operations are recorded without aborting the report. `--large` changes the Kitchen linear workload from the small 256-feature shape to a representative 4096-feature shape.

`--diagnostics diagnostics` enables PyTorch/XLA debug metrics and compiler dumps. Preserve `tpu_probe.json`, the diagnostics directory, and the ComfyUI startup log after the first hardware run.

The Colab probe also enables `--int8-experiments --large`, comparing `_int_mm`, generic INT8 matmul, and an INT32-matmul candidate at 32×4096 and 256×4096 shapes against a CPU INT32-accumulation reference. Operation records distinguish output placement from CPU fallback and full XLA-native execution. The current Kitchen `_int_mm` path is expected to remain a correctness fallback until a dedicated s8×s8→s32 TPU lowering is implemented and validated.

The notebook runs `tools.tpu_colab_setup`, which queries the official TPU wheel index for the latest stable PyTorch/XLA release, asks pip's resolver for the exact matching Torch, TorchVision, TorchAudio, and libtpu tuple, and installs those exact versions together. A complete compatible installed tuple is retained. The filtered ComfyUI requirements install is followed by an exact version check so it cannot silently replace the selected TPU stack.

Every probe measurement starts with a blocking barrier, times only the named operation, and ends with another blocking barrier. Inputs are created once outside kernel timing. Comfy Kitchen prerequisites are recorded and synchronized as their own operations; a failed prerequisite marks dependent operations as blocked. Metrics snapshots are written after core operations and before/after Kitchen operations. The first execution of each new shape may compile. Keep batch, latent dimensions, frame count, and token length stable across sampler steps to maximize reuse. No hardware performance claim is made until v5e artifacts are collected.

TPU detection accepts legacy Colab address/name variables, `TPU_ACCELERATOR_TYPE`, `/dev/accel*`, and numeric `/dev/vfio/<group>` devices. The launch cell waits for HTTP 200 before starting cloudflared and remains active with a 60-second heartbeat; stopping the cell terminates only the child processes it created.
